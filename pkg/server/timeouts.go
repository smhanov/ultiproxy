package server

import (
	"encoding/json"
	"os"
	"path/filepath"
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
	if persistPath != "" {
		if data, err := os.ReadFile(persistPath); err == nil {
			var runtime map[string]string
			if json.Unmarshal(data, &runtime) == nil {
				for provider, durStr := range runtime {
					if d, err := time.ParseDuration(durStr); err == nil && d > 0 {
						tm.timeouts[provider] = d
					}
				}
			}
		}
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
func (tm *TimeoutManager) Set(provider string, d time.Duration) error {
	if provider == "" || d <= 0 {
		return errTimeoutInvalid
	}
	tm.mu.Lock()
	tm.timeouts[provider] = d
	tm.mu.Unlock()
	return tm.persist()
}

// Remove clears a provider's explicit timeout (falls back to default).
func (tm *TimeoutManager) Remove(provider string) error {
	tm.mu.Lock()
	delete(tm.timeouts, provider)
	tm.mu.Unlock()
	return tm.persist()
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

func (tm *TimeoutManager) persist() error {
	if tm.persistPath == "" {
		return nil
	}
	tm.mu.RLock()
	snapshot := make(map[string]string, len(tm.timeouts))
	for p, d := range tm.timeouts {
		snapshot[p] = d.String()
	}
	tm.mu.RUnlock()
	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(tm.persistPath); dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	tmp := tm.persistPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, tm.persistPath)
}

var errTimeoutInvalid = &timeoutError{"timeout must be provider name and positive duration"}

type timeoutError struct{ msg string }

func (e *timeoutError) Error() string { return e.msg }
