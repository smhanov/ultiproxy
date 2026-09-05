package auth

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSingleflightConcurrentRefresh(t *testing.T) {
	// Requirements from spec:
	// "Test: 20 goroutines simultaneously Get() a token expiring in 4 minutes with a
	// fake refresher that sleeps 50ms and rotates the refresh token; assert exactly
	// ONE refresh call, all 20 get the same valid access token, and the persisted
	// file has the rotated refresh token with generation N+1."

	tempDir := t.TempDir()
	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeClock := func() time.Time { return currTime }

	var refreshCalls atomic.Int32

	fakeRefresher := func(ctx context.Context, cred Credential) (Credential, error) {
		refreshCalls.Add(1)
		time.Sleep(50 * time.Millisecond) // simulate network refresh
		return Credential{
			Tenant:       cred.Tenant,
			Provider:     cred.Provider,
			AccessToken:  "new-rotated-access-token",
			RefreshToken: "new-rotated-refresh-token",
			ExpiresAt:    currTime.Add(1 * time.Hour), // new expiry 1h ahead
			Generation:   cred.Generation + 1,
			ClientID:     cred.ClientID,
		}, nil
	}

	mgr, err := NewManager(tempDir, fakeRefresher, WithNow(fakeClock))
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	key := "tenant-alpha/anthropic"
	initialCred := Credential{
		Tenant:       "tenant-alpha",
		Provider:     "anthropic",
		AccessToken:  "old-access-token",
		RefreshToken: "old-refresh-token",
		ExpiresAt:    currTime.Add(4 * time.Minute), // expiring in 4 minutes (< 5 min)
		Generation:   1,
		ClientID:     "client-123",
	}

	if err := mgr.Store(context.Background(), key, initialCred); err != nil {
		t.Fatalf("failed to store initial cred: %v", err)
	}

	const numGoroutines = 20
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	results := make([]Credential, numGoroutines)
	errorsList := make([]error, numGoroutines)

	startBarrier := make(chan struct{})

	for i := 0; i < numGoroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			<-startBarrier // synchronize start
			c, err := mgr.Get(context.Background(), key)
			results[idx] = c
			errorsList[idx] = err
		}(i)
	}

	// Release all 20 goroutines at once
	close(startBarrier)
	wg.Wait()

	// 1. Assert exactly ONE refresh call
	if calls := refreshCalls.Load(); calls != 1 {
		t.Fatalf("expected exactly 1 refresh call, got %d", calls)
	}

	// 2. Assert all 20 get the same valid access token
	for i := 0; i < numGoroutines; i++ {
		if errorsList[i] != nil {
			t.Errorf("goroutine %d returned error: %v", i, errorsList[i])
		}
		if results[i].AccessToken != "new-rotated-access-token" {
			t.Errorf("goroutine %d got unexpected access token: %s", i, results[i].AccessToken)
		}
		if results[i].Generation != 2 {
			t.Errorf("goroutine %d got generation %d, expected 2", i, results[i].Generation)
		}
	}

	// 3. Assert persisted file has the rotated refresh token with generation N+1 (2)
	persisted, err := mgr.LoadFromDisk(key)
	if err != nil {
		t.Fatalf("failed to load persisted credential: %v", err)
	}
	if persisted.RefreshToken != "new-rotated-refresh-token" {
		t.Errorf("persisted refresh token mismatch: got %s", persisted.RefreshToken)
	}
	if persisted.Generation != 2 {
		t.Errorf("persisted generation mismatch: expected 2, got %d", persisted.Generation)
	}

	// Verify file permissions 0600
	targetPath := mgr.keyPath(key)
	info, err := os.Stat(targetPath)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}
	perm := info.Mode().Perm()
	if perm != 0600 {
		t.Errorf("expected file mode 0600, got %o", perm)
	}
}

func TestGenerationCASProtection(t *testing.T) {
	tempDir := t.TempDir()
	mgr, err := NewManager(tempDir, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	key := "t1/p1"
	credGen5 := Credential{
		Tenant:     "t1",
		Provider:   "p1",
		Generation: 5,
	}
	if err := mgr.Store(context.Background(), key, credGen5); err != nil {
		t.Fatalf("failed store gen 5: %v", err)
	}

	// Attempting to persist older generation (gen 4) must fail CAS check
	credGen4 := Credential{
		Tenant:     "t1",
		Provider:   "p1",
		Generation: 4,
	}
	err = mgr.persist(key, credGen4)
	if !errors.Is(err, ErrGenerationStale) {
		t.Errorf("expected ErrGenerationStale, got %v", err)
	}

	// Stored file must still have gen 5
	loaded, err := mgr.LoadFromDisk(key)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if loaded.Generation != 5 {
		t.Errorf("expected gen 5 preserved, got %d", loaded.Generation)
	}
}

func TestExpiredTokenRefreshFailure(t *testing.T) {
	tempDir := t.TempDir()
	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeClock := func() time.Time { return currTime }

	failRefresher := func(ctx context.Context, cred Credential) (Credential, error) {
		return Credential{}, errors.New("upstream auth service down")
	}

	mgr, err := NewManager(tempDir, failRefresher, WithNow(fakeClock))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	key := "t/p"
	expiredCred := Credential{
		Tenant:    "t",
		Provider:  "p",
		ExpiresAt: currTime.Add(-10 * time.Minute), // already expired
	}
	_ = mgr.Store(context.Background(), key, expiredCred)

	_, err = mgr.Get(context.Background(), key)
	if err == nil {
		t.Fatalf("expected error for expired token with failing refresher")
	}
	if !errors.Is(err, ErrExpired) {
		t.Errorf("expected ErrExpired, got %v", err)
	}
}

// breakStorage points the manager's storage dir at a regular file so
// os.CreateTemp fails: a deterministic persist failure that works regardless of
// the uid running the tests.
func breakStorage(t *testing.T, m *Manager) {
	t.Helper()
	blocker := filepath.Join(t.TempDir(), "blocker")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}
	m.storageDir = blocker
}

func cachedCred(t *testing.T, m *Manager, key string) (Credential, bool) {
	t.Helper()
	m.mu.RLock()
	defer m.mu.RUnlock()
	c, ok := m.cache[key]
	return c, ok
}

// TestManager_StorePersistFailureDoesNotCacheToken (T022 AC4): when the durable
// write fails, Store must return the error and must NOT leave the new token
// live in the in-memory cache.
func TestManager_StorePersistFailureDoesNotCacheToken(t *testing.T) {
	tempDir := t.TempDir()
	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	mgr, err := NewManager(tempDir, nil, WithNow(func() time.Time { return currTime }))
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	good := Credential{Tenant: "t", Provider: "p", AccessToken: "good", ExpiresAt: currTime.Add(time.Hour), Generation: 1}
	if err := mgr.Store(context.Background(), "k", good); err != nil {
		t.Fatalf("store healthy credential: %v", err)
	}
	if c, ok := cachedCred(t, mgr, "k"); !ok || c.AccessToken != "good" {
		t.Fatalf("healthy store did not populate the cache: %v %v", c, ok)
	}

	origDir := mgr.storageDir
	breakStorage(t, mgr)

	failing := Credential{Tenant: "t", Provider: "p", AccessToken: "never-persisted", ExpiresAt: currTime.Add(time.Hour), Generation: 2}
	if err := mgr.Store(context.Background(), "k2", failing); err == nil {
		t.Fatal("Store must fail when persistence fails")
	}
	if _, ok := cachedCred(t, mgr, "k2"); ok {
		t.Error("new token left in the in-memory cache although persistence failed")
	}
	// The pre-existing credential is untouched and still the cached one.
	if c, ok := cachedCred(t, mgr, "k"); !ok || c.AccessToken != "good" {
		t.Errorf("existing cache entry disturbed by the failed store: %v %v", c, ok)
	}

	// Observable through the public surface too: nothing was persisted for k2,
	// so reading the real storage dir must report not-found rather than hand
	// out the token.
	mgr.storageDir = origDir
	mgr.mu.Lock()
	delete(mgr.cache, "k2")
	mgr.mu.Unlock()
	if _, err := mgr.LoadFromDisk("k2"); !errors.Is(err, ErrNotFound) {
		t.Errorf("LoadFromDisk(k2) = %v; want ErrNotFound (token never reached disk)", err)
	}
}

// TestManager_RefreshPersistFailureDoesNotCacheToken (T022 AC4): a successful
// upstream refresh whose persistence fails must not publish the refreshed
// token into the cache.
func TestManager_RefreshPersistFailureDoesNotCacheToken(t *testing.T) {
	tempDir := t.TempDir()
	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	mgr, err := NewManager(tempDir, func(ctx context.Context, cred Credential) (Credential, error) {
		return Credential{
			Tenant:       cred.Tenant,
			Provider:     cred.Provider,
			AccessToken:  "refreshed-but-unpersisted",
			RefreshToken: "rotated",
			ExpiresAt:    currTime.Add(time.Hour),
			Generation:   cred.Generation + 1,
		}, nil
	}, WithNow(func() time.Time { return currTime }))
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	key := "t/p"
	nearlyExpired := Credential{Tenant: "t", Provider: "p", AccessToken: "old", ExpiresAt: currTime.Add(2 * time.Minute), Generation: 1}
	if err := mgr.Store(context.Background(), key, nearlyExpired); err != nil {
		t.Fatalf("store: %v", err)
	}

	origDir := mgr.storageDir
	breakStorage(t, mgr)
	if _, err := mgr.Get(context.Background(), key); err == nil {
		t.Fatal("Get must fail when the refreshed credential cannot be persisted")
	}
	mgr.storageDir = origDir // look at the real storage dir again
	if c, ok := cachedCred(t, mgr, key); !ok {
		t.Fatal("cache entry vanished entirely")
	} else if c.AccessToken != "old" || c.Generation != 1 {
		t.Errorf("refreshed token published to the cache despite persist failure: %+v", c)
	}
	if persisted, err := mgr.LoadFromDisk(key); err != nil || persisted.AccessToken != "old" {
		t.Errorf("disk state = %+v (%v); want the original token preserved", persisted, err)
	}
}
