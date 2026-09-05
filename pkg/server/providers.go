// Package server: runtime provider registration store.
//
// Ultiproxy must be usable as if freshly installed: zero configured providers,
// lanes added through the MCP surface (add_provider / remove_provider /
// list_providers), never by writing a config file. Runtime-registered lanes are
// OpenAI-compatible lanes (pkg/provider/openaicompat) and persist to
// data_dir/providers.json using the exact same atomic-write pattern as
// aliases.json (ModelCatalog) and timeouts.json (TimeoutManager), so a restart
// from the same DataDir restores them.
//
// JSON shape: openaicompat.Config cannot be marshalled directly (TokenSource is
// an interface and HTTPClient carries func fields), and Quirks.FreebuffActor is
// an injected *FreebuffAccountActor that is not JSON-serializable at all. The
// store therefore persists a DTO (storedProvider) holding only JSON-able
// fields:
//
//   - freebuff lanes are stored as a boolean flag (quirks.freebuff_actor). On
//     load the actor is reconstructed through ActorBuilder when one is
//     installed (cmd/ultiproxy installs the same *freebuffActorAdapter used by
//     the compile-time lane). When no builder is installed the lane still
//     loads, just without the serialized-request lock actor -- and the flag is
//     kept in memory (RuntimeProviderStore.freebuff) so Restore can rebuild the
//     actor once a builder exists, without ever turning an ordinary lane into a
//     freebuff lane.
//   - TokenSource is never persisted. openaicompat.New rebuilds token sources
//     from compiled-in vendor defaults plus the server's DataDir (credential
//     state lives under <DefaultDataDir>/credentials/<lane>), exactly the way
//     cmd/ultiproxy/providers.go builds them today.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// providerNamePattern constrains runtime lane names: lowercase letters, digits,
// underscores and dashes only. No slashes, no dots, no uppercase - this keeps
// the registry, alias catalog and timeouts map keys unambiguous.
var providerNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// ErrProviderNotStored is returned by Remove for a name that is not present in
// the runtime store.
var ErrProviderNotStored = errors.New("provider not present in runtime store")

// storedQuirks is the JSON-able projection of openaicompat.Quirks. FreebuffActor
// is deliberately a bool (see the package comment).
type storedQuirks struct {
	CodingPlanPath         bool           `json:"coding_plan_path,omitempty"`
	MaxTokensByModel       map[string]int `json:"max_tokens_by_model,omitempty"`
	EchoReasoning          bool           `json:"echo_reasoning,omitempty"`
	ModelListPassthrough   bool           `json:"model_list_passthrough,omitempty"`
	AuthViaOAuthManager    bool           `json:"auth_via_oauth_manager,omitempty"`
	CreditsQuotaObserver   string         `json:"credits_quota_observer,omitempty"`
	AuthViaSupabaseRefresh bool           `json:"auth_via_supabase_refresh,omitempty"`
	FreebuffActor          bool           `json:"freebuff_actor,omitempty"`
	FreebuffDefaultTool    bool           `json:"freebuff_default_tool,omitempty"`
	DefaultModel           string         `json:"default_model,omitempty"`
}

// storedProvider is the JSON-able projection of openaicompat.Config. Lanes
// never carry their own data dir or auth endpoint overrides: credential state
// is derived from the server's DefaultDataDir, and token endpoints use the
// compiled-in vendor defaults.
type storedProvider struct {
	Kind    string       `json:"kind,omitempty"` // "" or "openaicompat" | "antigravity"
	Name    string       `json:"name"`
	BaseURL string       `json:"base_url"`
	APIKey  string       `json:"api_key,omitempty"`
	Quirks  storedQuirks `json:"quirks"`
}

func toStoredProvider(cfg openaicompat.Config) storedProvider {
	return storedProvider{
		Name:    cfg.Name,
		BaseURL: cfg.BaseURL,
		APIKey:  cfg.APIKey,
		Quirks: storedQuirks{
			CodingPlanPath:         cfg.Quirks.CodingPlanPath,
			MaxTokensByModel:       cfg.Quirks.MaxTokensByModel,
			EchoReasoning:          cfg.Quirks.EchoReasoning,
			ModelListPassthrough:   cfg.Quirks.ModelListPassthrough,
			AuthViaOAuthManager:    cfg.Quirks.AuthViaOAuthManager,
			CreditsQuotaObserver:   cfg.Quirks.CreditsQuotaObserver,
			AuthViaSupabaseRefresh: cfg.Quirks.AuthViaSupabaseRefresh,
			FreebuffActor:          cfg.Quirks.FreebuffActor != nil,
			FreebuffDefaultTool:    cfg.Quirks.FreebuffDefaultTool,
			DefaultModel:           cfg.Quirks.DefaultModel,
		},
	}
}

func (s storedProvider) toConfig(actorBuilder func(openaicompat.Config) any) openaicompat.Config {
	cfg := openaicompat.Config{
		Name:    s.Name,
		BaseURL: s.BaseURL,
		APIKey:  s.APIKey,
		Quirks: openaicompat.Quirks{
			CodingPlanPath:         s.Quirks.CodingPlanPath,
			MaxTokensByModel:       s.Quirks.MaxTokensByModel,
			EchoReasoning:          s.Quirks.EchoReasoning,
			ModelListPassthrough:   s.Quirks.ModelListPassthrough,
			AuthViaOAuthManager:    s.Quirks.AuthViaOAuthManager,
			CreditsQuotaObserver:   s.Quirks.CreditsQuotaObserver,
			AuthViaSupabaseRefresh: s.Quirks.AuthViaSupabaseRefresh,
			FreebuffDefaultTool:    s.Quirks.FreebuffDefaultTool,
			DefaultModel:           s.Quirks.DefaultModel,
		},
	}
	if s.Quirks.FreebuffActor && actorBuilder != nil {
		cfg.Quirks.FreebuffActor = actorBuilder(cfg)
	}
	return cfg
}

// RuntimeProviderStore is a thread-safe registry of runtime-registered
// OpenAI-compatible lanes with JSON persistence, mirroring ModelCatalog.
// A nil/empty path means in-memory only (no persistence).
type RuntimeProviderStore struct {
	mu        sync.RWMutex
	path      string
	providers map[string]openaicompat.Config
	custom    map[string]storedProvider // non-openaicompat kinds (kind != "")
	restored  bool
	// freebuff holds the persisted quirks.freebuff_actor flag per
	// openai-compatible lane. The flag lives only in the JSON DTO: the
	// in-memory config carries an actor, not a bool, and that actor cannot be
	// rebuilt while ActorBuilder is still nil -- exactly the state
	// NewRuntimeProviderStore loads the file in. Restore reads this flag (never
	// "actor == nil") to decide which lanes are Freebuff.
	freebuff map[string]bool
	// DefaultDataDir is the server's general data directory. Runtime lanes do
	// NOT carry their own data dir; credential state lives under
	// <DefaultDataDir>/credentials/<lane> exactly like compile-time lanes.
	DefaultDataDir string
	ActorBuilder   func(openaicompat.Config) any // optional freebuff actor reconstruction hook (set by cmd)
	// LaneBuilder constructs compile-time-wired lane kinds that are not
	// openai-compatible (e.g. antigravity, anthropic, codex) from their stored
	// identity. It receives the lane name, kind, the server's general DataDir
	// and the lane's persisted api_key ("" for kinds that use no static key)
	// and returns the provider bundle. When nil, custom kinds can be stored but
	// not restored.
	LaneBuilder func(name, kind, dataDir, apiKey string) (provider.Provider, error)
}

// RuntimeProviderKind is the persisted provider kind discriminator.
const RuntimeProviderKindOpenAICompat = "openaicompat"

// NewRuntimeProviderStore builds a runtime provider store. When path is
// non-empty and the file exists, its entries are preloaded into memory (same
// best-effort behaviour as NewModelCatalog: a malformed file is ignored).
func NewRuntimeProviderStore(path string) *RuntimeProviderStore {
	s := &RuntimeProviderStore{
		path:      path,
		providers: make(map[string]openaicompat.Config),
		custom:    make(map[string]storedProvider),
		freebuff:  make(map[string]bool),
	}
	if path == "" {
		return s
	}
	if data, err := os.ReadFile(path); err == nil {
		var stored map[string]storedProvider
		if json.Unmarshal(data, &stored) == nil {
			for name, sp := range stored {
				if sp.Kind != "" && sp.Kind != RuntimeProviderKindOpenAICompat {
					s.custom[name] = sp
				} else {
					s.providers[name] = s.credentialDir(sp.toConfig(s.ActorBuilder))
					s.freebuff[name] = sp.Quirks.FreebuffActor
				}
			}
		}
	}
	return s
}

// Path returns the persistence path ("" for an in-memory store).
func (s *RuntimeProviderStore) Path() string {
	if s == nil {
		return ""
	}
	return s.path
}

// validateProviderConfig enforces the constraints that keep the registry
// stable: a lowercase lane name and a non-empty base URL.
func validateProviderConfig(cfg openaicompat.Config) error {
	cfg.Name = strings.TrimSpace(cfg.Name)
	if cfg.Name == "" {
		return errors.New("provider name is required")
	}
	if !providerNamePattern.MatchString(cfg.Name) {
		return fmt.Errorf("provider name %q must match %s (lowercase letters, digits, _ and -)", cfg.Name, providerNamePattern.String())
	}
	cfg.BaseURL = strings.TrimSpace(cfg.BaseURL)
	if cfg.BaseURL == "" {
		return errors.New("provider base_url is required")
	}
	if strings.ContainsAny(cfg.BaseURL, " \t\n") {
		return errors.New("provider base_url must not contain whitespace")
	}
	return nil
}

// List returns a copy of the runtime-registered provider configs keyed by lane
// name.
func (s *RuntimeProviderStore) List() map[string]openaicompat.Config {
	out := make(map[string]openaicompat.Config)
	if s == nil {
		return out
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for k, v := range s.providers {
		out[k] = v
	}
	return out
}

// Sorted returns lane names in stable order (for MCP listing).
func (s *RuntimeProviderStore) Sorted() []string {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	names := make([]string, 0, len(s.providers)+len(s.custom))
	for k := range s.providers {
		names = append(names, k)
	}
	for k := range s.custom {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Has reports whether a lane is present in the runtime store.
func (s *RuntimeProviderStore) Has(name string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.providers[name]
	if ok {
		return true
	}
	_, ok = s.custom[name]
	return ok
}

// Get returns a single stored config.
func (s *RuntimeProviderStore) Get(name string) (openaicompat.Config, bool) {
	if s == nil {
		return openaicompat.Config{}, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	cfg, ok := s.providers[name]
	return cfg, ok
}

// credentialDir assigns a lane's credential directory from the server's
// general DataDir. Runtime lanes never carry their own data dir, so restored
// lanes get <DefaultDataDir>/credentials/<lane> exactly like a freshly added
// one; the persisted DTO holds no data_dir of its own.
func (s *RuntimeProviderStore) credentialDir(cfg openaicompat.Config) openaicompat.Config {
	if cfg.DataDir == "" && s != nil && s.DefaultDataDir != "" {
		cfg.DataDir = filepath.Join(s.DefaultDataDir, "credentials", cfg.Name)
	}
	return cfg
}

// Add validates and stores a lane config, replacing any existing entry with
// the same name (replacement is intentional: re-running add_provider is how a
// lane's base URL or key gets rotated), then persists.
func (s *RuntimeProviderStore) Add(cfg openaicompat.Config) error {
	if s == nil {
		return errors.New("runtime provider store not configured")
	}
	if err := validateProviderConfig(cfg); err != nil {
		return err
	}
	cfg = s.credentialDir(cfg)
	// A freebuff lane added over MCP carries only a non-nil marker (the real
	// actor is not JSON-serializable and cannot cross that boundary). Build the
	// serialized-request actor here from the lane's own key / data dir so the
	// stored config is immediately usable, exactly like a restored lane.
	if cfg.Quirks.FreebuffActor != nil && s.ActorBuilder != nil {
		if _, ok := cfg.Quirks.FreebuffActor.(openaicompat.FreebuffActor); !ok {
			if actor := s.ActorBuilder(cfg); actor != nil {
				cfg.Quirks.FreebuffActor = actor
			}
		}
	}
	s.mu.Lock()
	if s.providers == nil {
		s.providers = make(map[string]openaicompat.Config)
	}
	if s.freebuff == nil {
		s.freebuff = make(map[string]bool)
	}
	delete(s.custom, cfg.Name)
	s.providers[cfg.Name] = cfg
	s.freebuff[cfg.Name] = cfg.Quirks.FreebuffActor != nil
	s.mu.Unlock()
	return s.persist()
}

// AddCustom stores a non-OpenAI-compatible lane (kind plus, for kinds that
// need one, the api_key), replacing any existing entry with the same name,
// then persists. Name validation stays the same; BaseURL is not required for
// compile-time-wired kinds. Credential storage is the server's general
// DataDir, not a per-lane dir. apiKey is empty for kinds that authenticate
// through a credential store instead of a static key (antigravity, codex).
func (s *RuntimeProviderStore) AddCustom(name, kind, apiKey string) error {
	if s == nil {
		return errors.New("runtime provider store not configured")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("provider name is required")
	}
	if !providerNamePattern.MatchString(name) {
		return fmt.Errorf("provider name %q must match %s (lowercase letters, digits, _ and -)", name, providerNamePattern.String())
	}
	if kind == "" || kind == RuntimeProviderKindOpenAICompat {
		return fmt.Errorf("custom provider kind %q invalid (use %q only for openai-compatible lanes)", kind, RuntimeProviderKindOpenAICompat)
	}
	s.mu.Lock()
	if s.custom == nil {
		s.custom = make(map[string]storedProvider)
	}
	delete(s.providers, name)
	delete(s.freebuff, name)
	s.custom[name] = storedProvider{Kind: kind, Name: name, APIKey: apiKey}
	s.mu.Unlock()
	return s.persist()
}

// Remove deletes a lane from the store and persists the change. It returns
// ErrProviderNotStored when the name was never runtime-registered.
func (s *RuntimeProviderStore) Remove(name string) error {
	if s == nil {
		return errors.New("runtime provider store not configured")
	}
	s.mu.Lock()
	_, ok := s.providers[name]
	if ok {
		delete(s.providers, name)
	} else {
		_, ok = s.custom[name]
		if ok {
			delete(s.custom, name)
		}
	}
	delete(s.freebuff, name)
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w: %q", ErrProviderNotStored, name)
	}
	return s.persist()
}

// Load re-reads the persisted file into memory and returns the resulting
// configs. For an in-memory store it returns the current contents. A missing
// file is not an error (empty map).
func (s *RuntimeProviderStore) Load() (map[string]openaicompat.Config, error) {
	out := make(map[string]openaicompat.Config)
	if s == nil {
		return out, nil
	}
	if s.path == "" {
		return s.List(), nil
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			s.mu.Lock()
			s.providers = make(map[string]openaicompat.Config)
			s.freebuff = make(map[string]bool)
			s.mu.Unlock()
			return out, nil
		}
		return nil, err
	}
	var stored map[string]storedProvider
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, fmt.Errorf("parse %s: %w", s.path, err)
	}
	s.mu.Lock()
	s.providers = make(map[string]openaicompat.Config, len(stored))
	s.custom = make(map[string]storedProvider, len(stored))
	s.freebuff = make(map[string]bool, len(stored))
	for name, sp := range stored {
		if sp.Kind != "" && sp.Kind != RuntimeProviderKindOpenAICompat {
			s.custom[name] = sp
		} else {
			s.providers[name] = s.credentialDir(sp.toConfig(s.ActorBuilder))
			s.freebuff[name] = sp.Quirks.FreebuffActor
		}
	}
	s.mu.Unlock()
	return s.List(), nil
}

// Restore registers every stored lane into the provider registry. It is
// idempotent (a second call is a no-op) so main and server.NewServer can both
// call it safely. It returns the lane names that were registered successfully.
func (s *RuntimeProviderStore) Restore(registry *provider.Registry) []string {
	if s == nil || registry == nil {
		return nil
	}
	s.mu.Lock()
	if s.restored {
		s.mu.Unlock()
		return nil
	}
	s.restored = true
	snapshot := make(map[string]openaicompat.Config, len(s.providers))
	customSnap := make(map[string]storedProvider, len(s.custom))
	freebuffSnap := make(map[string]bool, len(s.freebuff))
	builder := s.ActorBuilder
	laneBuilder := s.LaneBuilder
	for k, v := range s.providers {
		snapshot[k] = v
	}
	for k, v := range s.freebuff {
		freebuffSnap[k] = v
	}
	for k, v := range s.custom {
		customSnap[k] = v
	}
	s.mu.Unlock()

	registered := make([]string, 0, len(snapshot)+len(customSnap))
	for _, name := range sortedConfigNames(snapshot) {
		cfg := s.credentialDir(snapshot[name])
		if builder != nil && freebuffSnap[name] && cfg.Quirks.FreebuffActor == nil {
			// storedProvider only keeps a bool flag; rebuild the actor here, and
			// only for lanes persisted with quirks.freebuff_actor=true. A nil
			// actor is never itself evidence that a lane is Freebuff.
			cfg.Quirks.FreebuffActor = builder(cfg)
		}
		p, err := openaicompat.New(cfg)
		if err != nil {
			log.Printf("[providers] runtime %s: %v", name, err)
			continue
		}
		registry.Register(p.Provider())
		registered = append(registered, name)
		log.Printf("[providers] registered runtime %s", name)
	}
	for _, name := range sortedCustomNames(customSnap) {
		sp := customSnap[name]
		if laneBuilder == nil {
			log.Printf("[providers] runtime %s: no LaneBuilder for kind %q — not restored", name, sp.Kind)
			continue
		}
		bundle, err := laneBuilder(name, sp.Kind, s.DefaultDataDir, sp.APIKey)
		if err != nil {
			log.Printf("[providers] runtime %s: %v", name, err)
			continue
		}
		registry.Register(bundle)
		registered = append(registered, name)
		log.Printf("[providers] registered runtime %s (kind %s)", name, sp.Kind)
	}
	return registered
}

func sortedCustomNames(m map[string]storedProvider) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

func sortedConfigNames(m map[string]openaicompat.Config) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// persist writes the store atomically (temp file + rename), like catalog.go.
func (s *RuntimeProviderStore) persist() error {
	if s == nil || s.path == "" {
		return nil
	}
	s.mu.RLock()
	snapshot := make(map[string]storedProvider, len(s.providers)+len(s.custom))
	for k, v := range s.providers {
		sp := toStoredProvider(v)
		sp.Kind = RuntimeProviderKindOpenAICompat
		// A freebuff lane loaded while ActorBuilder was still nil carries no
		// in-memory actor, so toStoredProvider would write the flag as false.
		// The persisted boolean is the durable record: keep it.
		if s.freebuff[k] {
			sp.Quirks.FreebuffActor = true
		}
		snapshot[k] = sp
	}
	for k, v := range s.custom {
		snapshot[k] = v
	}
	s.mu.RUnlock()

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(s.path); dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}
