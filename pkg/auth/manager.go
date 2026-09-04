package auth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

var (
	ErrNotFound        = errors.New("credential not found")
	ErrExpired         = errors.New("credential is expired")
	ErrGenerationStale = errors.New("generation CAS mismatch: stored generation is newer")
)

// Refresher defines the signature for token refresh logic.
type Refresher func(ctx context.Context, cred Credential) (Credential, error)

// Option configures Manager.
type Option func(*Manager)

// WithNow sets the clock function.
func WithNow(fn func() time.Time) Option {
	return func(m *Manager) {
		m.nowFn = fn
	}
}

// Manager coordinates credential storage, atomic file persistence, and singleflight refreshes.
type Manager struct {
	storageDir string
	refresher  Refresher
	nowFn      func() time.Time

	sf singleflight.Group

	mu    sync.RWMutex
	cache map[string]Credential
}

// NewManager creates a new credential Manager.
func NewManager(storageDir string, refresher Refresher, opts ...Option) (*Manager, error) {
	if err := os.MkdirAll(storageDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create storage dir: %w", err)
	}

	m := &Manager{
		storageDir: storageDir,
		refresher:  refresher,
		nowFn:      time.Now,
		cache:      make(map[string]Credential),
	}

	for _, opt := range opts {
		opt(m)
	}

	return m, nil
}

// SetRefresher installs or replaces the token refresher.
func (m *Manager) SetRefresher(r Refresher) {
	if m == nil {
		return
	}
	m.refresher = r
}

func (m *Manager) now() time.Time {
	if m.nowFn != nil {
		return m.nowFn()
	}
	return time.Now()
}

func (m *Manager) keyPath(key string) string {
	h := sha256.Sum256([]byte(key))
	filename := hex.EncodeToString(h[:]) + ".json"
	return filepath.Join(m.storageDir, filename)
}

// Store saves a credential in-memory and atomically persists it to disk.
func (m *Manager) Store(ctx context.Context, key string, cred Credential) error {
	m.mu.Lock()
	m.cache[key] = cred
	m.mu.Unlock()

	return m.persist(key, cred)
}

// persist writes temp file in same dir, fsyncs, renames, and enforces generation CAS.
func (m *Manager) persist(key string, cred Credential) error {
	targetPath := m.keyPath(key)

	// Check existing file generation if present (CAS)
	if existingData, err := os.ReadFile(targetPath); err == nil {
		var existing Credential
		if err := json.Unmarshal(existingData, &existing); err == nil {
			if cred.Generation < existing.Generation {
				return ErrGenerationStale
			}
		}
	}

	data, err := json.MarshalIndent(cred, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal credential: %w", err)
	}

	tempFile, err := os.CreateTemp(m.storageDir, "cred_*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tempName := tempFile.Name()
	defer os.Remove(tempName) // Clean up temp file on failure

	if _, err := tempFile.Write(data); err != nil {
		tempFile.Close()
		return fmt.Errorf("failed to write to temp file: %w", err)
	}

	// fsync
	if err := tempFile.Sync(); err != nil {
		tempFile.Close()
		return fmt.Errorf("failed to sync temp file: %w", err)
	}

	// chmod 0600
	if err := tempFile.Chmod(0600); err != nil {
		tempFile.Close()
		return fmt.Errorf("failed to chmod temp file: %w", err)
	}

	if err := tempFile.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tempName, targetPath); err != nil {
		return fmt.Errorf("failed to atomically rename credential file: %w", err)
	}

	return nil
}

// LoadFromDisk reads the persisted credential for key.
func (m *Manager) LoadFromDisk(key string) (Credential, error) {
	targetPath := m.keyPath(key)
	data, err := os.ReadFile(targetPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Credential{}, ErrNotFound
		}
		return Credential{}, err
	}

	var cred Credential
	if err := json.Unmarshal(data, &cred); err != nil {
		return Credential{}, fmt.Errorf("corrupt credential file: %w", err)
	}

	return cred, nil
}

// Get returns a valid token, refreshing if within 5 minutes of expiry.
// If expired and refresh fails, returns error.
func (m *Manager) Get(ctx context.Context, key string) (Credential, error) {
	m.mu.RLock()
	cred, exists := m.cache[key]
	m.mu.RUnlock()

	if !exists {
		var err error
		cred, err = m.LoadFromDisk(key)
		if err != nil {
			return Credential{}, err
		}
		m.mu.Lock()
		m.cache[key] = cred
		m.mu.Unlock()
	}

	now := m.now()
	timeUntilExpiry := cred.ExpiresAt.Sub(now)

	// If more than 5 minutes remaining, token is fully valid without refresh
	if timeUntilExpiry > 5*time.Minute {
		return cred, nil
	}

	// Token is within 5 minutes of expiry (or already expired) -> coordinate refresh via singleflight
	res, err, _ := m.sf.Do(key, func() (any, error) {
		// Double check latest cached cred in case another routine just refreshed it
		m.mu.RLock()
		latest := m.cache[key]
		m.mu.RUnlock()

		if latest.ExpiresAt.Sub(m.now()) > 5*time.Minute {
			return latest, nil
		}

		if m.refresher == nil {
			if m.now().After(latest.ExpiresAt) {
				return nil, ErrExpired
			}
			return latest, nil
		}

		refreshed, refErr := m.refresher(ctx, latest)
		if refErr != nil {
			// If already expired, return error
			if m.now().After(latest.ExpiresAt) {
				return nil, fmt.Errorf("%w: refresh failed: %v", ErrExpired, refErr)
			}
			// Not yet expired: return current valid token
			return latest, nil
		}

		// Increment generation if refresher didn't already
		if refreshed.Generation <= latest.Generation {
			refreshed.Generation = latest.Generation + 1
		}

		// Update cache and persist to disk
		m.mu.Lock()
		m.cache[key] = refreshed
		m.mu.Unlock()

		if err := m.persist(key, refreshed); err != nil {
			return nil, fmt.Errorf("failed to persist refreshed credential: %w", err)
		}

		return refreshed, nil
	})

	if err != nil {
		return Credential{}, err
	}

	return res.(Credential), nil
}
