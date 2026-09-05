package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

// DefaultRequestTimeout is used when a provider has no explicit timeout.
const DefaultRequestTimeout = 120 * time.Second

// TimeoutManager holds per-provider request timeouts, mutable at runtime via
// MCP (get_provider_timeouts / set_provider_timeout) and persisted to
// data_dir/timeouts.json across restarts.
type TimeoutManager struct {
	mu          sync.RWMutex
	timeouts    map[string]time.Duration
	defaultDur  time.Duration
	persistPath string
}

// NewTimeoutManager builds a timeout manager from config defaults (a map of
// provider -> duration strings, e.g. {"vllm": "10m"}). Runtime overrides are
// loaded from persistPath and win over config.
func NewTimeoutManager(config map[string]string, defaultDur time.Duration, persistPath string) (*TimeoutManager, error) {
	if defaultDur <= 0 {
		defaultDur = DefaultRequestTimeout
	}
	tm := &TimeoutManager{
		timeouts:    make(map[string]time.Duration),
		defaultDur:  defaultDur,
		persistPath: persistPath,
	}
	for provider, durStr := range config {
		if d, err := time.ParseDuration(durStr); err == nil && d > 0 {
			tm.timeouts[provider] = d
		}
	}
	if persistPath == "" {
		return tm, nil
	}
	data, err := os.ReadFile(persistPath)
	switch {
	case err == nil:
		var runtime map[string]string
		if err := json.Unmarshal(data, &runtime); err != nil {
			// A truncated/garbage timeouts.json is control-plane corruption, not
			// "no runtime state". The error is returned AND logged, because some
			// callers (server.NewServer) discard it; the returned manager stays
			// usable with the config defaults so a discarded error degrades to
			// config timeouts instead of a nil-pointer panic.
			err := fmt.Errorf("timeout manager: corrupt persistence file %s: %w (fix or remove the file and restart)", persistPath, err)
			log.Printf("[timeouts] %v -- runtime timeout overrides were NOT loaded", err)
			return tm, err
		}
		for provider, durStr := range runtime {
			// A stored value that no longer parses is corruption of the same
			// file, not a value to drop on the floor: the operator would end
			// up with the default timeout and no hint why.
			d, err := time.ParseDuration(durStr)
			if err != nil || d <= 0 {
				err := fmt.Errorf("timeout manager: corrupt persistence file %s: provider %q has invalid timeout %q (fix or remove the file and restart)", persistPath, provider, durStr)
				log.Printf("[timeouts] %v -- runtime timeout overrides were NOT loaded", err)
				return tm, err
			}
			tm.timeouts[provider] = d
		}
	case errors.Is(err, os.ErrNotExist):
		// No persisted runtime state yet.
	default:
		err := fmt.Errorf("timeout manager: read persistence file %s: %w", persistPath, err)
		log.Printf("[timeouts] %v -- runtime timeout overrides were NOT loaded", err)
		return tm, err
	}
	return tm, nil
}

// Timeout returns the effective timeout for a provider (explicit or default).
func (tm *TimeoutManager) Timeout(provider string) time.Duration {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	if d, ok := tm.timeouts[provider]; ok {
		return d
	}
	return tm.defaultDur
}

// Set updates a provider timeout and persists the change.
//
// Mutation, snapshot and write are one critical section, so concurrent MCP
// set_provider_timeout calls serialize and cannot clobber each other's
// temporary file or rename an older snapshot over a newer mutation. The new
// timeout goes live only after the write succeeded: a failed persist leaves
// the effective timeouts unchanged and the error is returned.
func (tm *TimeoutManager) Set(provider string, d time.Duration) error {
	if provider == "" || d <= 0 {
		return errTimeoutInvalid
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	next := cloneTimeouts(tm.timeouts)
	next[provider] = d
	if err := tm.persistTimeouts(next); err != nil {
		return err
	}
	tm.timeouts = next
	return nil
}

// Remove clears a provider's explicit timeout (falls back to default), with
// the same persist-before-publish ordering as Set.
func (tm *TimeoutManager) Remove(provider string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	next := cloneTimeouts(tm.timeouts)
	delete(next, provider)
	if err := tm.persistTimeouts(next); err != nil {
		return err
	}
	tm.timeouts = next
	return nil
}

// List returns all explicit timeouts as duration strings.
func (tm *TimeoutManager) List() map[string]string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	out := make(map[string]string, len(tm.timeouts))
	for p, d := range tm.timeouts {
		out[p] = d.String()
	}
	return out
}

// persistTimeouts writes one complete timeout state atomically through a
// uniquely named temp file (see writeFileAtomic in catalog.go). Callers must
// hold tm.mu so the state written is the state that gets published.
func (tm *TimeoutManager) persistTimeouts(state map[string]time.Duration) error {
	if tm.persistPath == "" {
		return nil
	}
	snapshot := make(map[string]string, len(state))
	for p, d := range state {
		snapshot[p] = d.String()
	}
	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(tm.persistPath, data)
}

func cloneTimeouts(in map[string]time.Duration) map[string]time.Duration {
	out := make(map[string]time.Duration, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	return out
}

var errTimeoutInvalid = &timeoutError{"timeout must be provider name and positive duration"}

type timeoutError struct{ msg string }

func (e *timeoutError) Error() string { return e.msg }
