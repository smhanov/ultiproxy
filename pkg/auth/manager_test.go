package auth

import (
	"context"
	"errors"
	"os"
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
