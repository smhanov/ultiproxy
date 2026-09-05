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
	// revoked remembers the keys whose cached credential was invalidated but
	// not yet refreshed, so Get refreshes them even while they are still well
	// inside their expiry window.
	revoked map[string]struct{}
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
		revoked:    make(map[string]struct{}),
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

// Store durably persists a credential and only then publishes it to the
// in-memory cache. Ordering matters: when the disk write fails the caller gets
// the error and the new token is NOT left live in memory, so a working process
// can never hold a credential that a restart would silently lose.
func (m *Manager) Store(ctx context.Context, key string, cred Credential) error {
	if err := m.persist(key, cred); err != nil {
		return err
	}
	m.mu.Lock()
	m.cache[key] = cred
	if m.revoked != nil {
		delete(m.revoked, key)
	}
	m.mu.Unlock()
	return nil
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

// Invalidate revokes the credential cached for key so that the next Get
// refreshes it instead of returning a token the caller has reported as bad.
// Revocation survives until a refresh succeeds: without it, a caller asking for
// a fresh token (an explicit refresh, or a 401 from the upstream) would keep
// reading back the very credential it rejected.
//
// A non-empty accessToken only revokes when it is still the credential in
// hand, so invalidating a stale token that has already been replaced leaves the
// current one alone.
func (m *Manager) Invalidate(key, accessToken string) bool {
	if m == nil {
		return false
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if accessToken != "" {
		if cred, ok := m.cache[key]; ok && cred.AccessToken != accessToken {
			return false
		}
	}
	if m.revoked == nil {
		m.revoked = make(map[string]struct{})
	}
	m.revoked[key] = struct{}{}
	return true
}

// revokedKey reports whether the credential for key is waiting to be refreshed.
func (m *Manager) revokedKey(key string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.revoked[key]
	return ok
}

// Get returns a valid token, refreshing if within 5 minutes of expiry.
// A credential invalidated through Invalidate is refreshed on the next Get even
// when it is still well within its expiry window.
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
	// (unless it was explicitly invalidated).
	if timeUntilExpiry > 5*time.Minute && !m.revokedKey(key) {
		return cred, nil
	}

	// Token is within 5 minutes of expiry (or already expired) -> coordinate refresh via singleflight
	res, err, _ := m.sf.Do(key, func() (any, error) {
		// Double check latest cached cred in case another routine just refreshed it
		m.mu.RLock()
		latest := m.cache[key]
		m.mu.RUnlock()

		if latest.ExpiresAt.Sub(m.now()) > 5*time.Minute && !m.revokedKey(key) {
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
			// An explicitly invalidated credential must never be handed back
			// as if it were still good: the caller is the one who rejected it.
			if m.revokedKey(key) {
				return nil, fmt.Errorf("credential invalidated: refresh failed: %v", refErr)
			}
			// Not yet expired: return current valid token
			return latest, nil
		}

		// Increment generation if refresher didn't already
		if refreshed.Generation <= latest.Generation {
			refreshed.Generation = latest.Generation + 1
		}

		// Persist first, publish afterwards: a failed persist must not leave
		// the refreshed token live in memory (Store has the same ordering).
		if err := m.persist(key, refreshed); err != nil {
			return nil, fmt.Errorf("failed to persist refreshed credential: %w", err)
		}
		m.mu.Lock()
		m.cache[key] = refreshed
		if m.revoked != nil {
			delete(m.revoked, key)
		}
		m.mu.Unlock()

		return refreshed, nil
	})

	if err != nil {
		return Credential{}, err
	}

	return res.(Credential), nil
}
