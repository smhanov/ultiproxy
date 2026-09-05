package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestTimeoutManagerSetAndReload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "timeouts.json")

	tm, err := NewTimeoutManager(map[string]string{"vllm": "10m"}, DefaultRequestTimeout, path)
	if err != nil {
		t.Fatalf("NewTimeoutManager: %v", err)
	}
	if got := tm.Timeout("vllm"); got != 10*time.Minute {
		t.Fatalf("config timeout = %v; want 10m", got)
	}
	if got := tm.Timeout("unknown"); got != DefaultRequestTimeout {
		t.Fatalf("default timeout = %v; want %v", got, DefaultRequestTimeout)
	}
	if err := tm.Set("vllm", 90*time.Second); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if got := tm.Timeout("vllm"); got != 90*time.Second {
		t.Fatalf("runtime timeout = %v; want 90s", got)
	}

	// Reload: the runtime override survives, config defaults still apply.
	tm2, err := NewTimeoutManager(map[string]string{"vllm": "10m"}, DefaultRequestTimeout, path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if got := tm2.Timeout("vllm"); got != 90*time.Second {
		t.Fatalf("reloaded timeout = %v; want 90s", got)
	}
	// Removing an explicit timeout drops back to the default (removing a
	// config-provided override is pre-existing semantics: the default applies).
	if err := tm2.Set("other", 5*time.Second); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if got := tm2.Timeout("other"); got != 5*time.Second {
		t.Fatalf("second runtime timeout = %v; want 5s", got)
	}
	if err := tm2.Remove("other"); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if got := tm2.Timeout("other"); got != DefaultRequestTimeout {
		t.Fatalf("timeout after remove = %v; want default", got)
	}
	if got := tm2.Timeout("vllm"); got != 90*time.Second {
		t.Fatalf("vllm timeout disturbed = %v; want 90s", got)
	}
}

// TestTimeoutManager_ConcurrentSetsBothSurviveOnDisk (T022 AC1): concurrent
// Set calls must both reach disk and leave valid JSON of the last completed
// mutation behind -- no shared .tmp, no lost update.
func TestTimeoutManager_ConcurrentSetsBothSurviveOnDisk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "timeouts.json")
	tm, err := NewTimeoutManager(nil, DefaultRequestTimeout, path)
	if err != nil {
		t.Fatalf("NewTimeoutManager: %v", err)
	}

	const workers = 24
	errs := make([]error, workers)
	names := make([]string, workers)
	for i := range names {
		names[i] = fmt.Sprintf("lane-%02d", i)
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			errs[i] = tm.Set(names[i], time.Duration(i+1)*time.Second)
		}(i)
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent Set(%q) reported a persistence error: %v", names[i], err)
		}
	}
	if err := tm.Set("final", 42*time.Second); err != nil {
		t.Fatalf("Set(final): %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read timeouts.json: %v", err)
	}
	var onDisk map[string]string
	if err := json.Unmarshal(data, &onDisk); err != nil {
		t.Fatalf("timeouts.json is not valid JSON (torn write): %v\n%s", err, data)
	}
	want := append([]string{"final"}, names...)
	for _, name := range want {
		if _, ok := onDisk[name]; !ok {
			t.Errorf("timeout for %q lost from timeouts.json: %s", name, data)
		}
	}

	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("leftover temp file %q next to timeouts.json", e.Name())
		}
	}
}

// TestTimeoutManager_PersistFailureLeavesLiveStateUnchanged (T022 AC2): a
// failed disk write must not change the effective timeouts, and the caller
// must still receive the error.
func TestTimeoutManager_PersistFailureLeavesLiveStateUnchanged(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}

	tm, err := NewTimeoutManager(map[string]string{"vllm": "10m"}, DefaultRequestTimeout, filepath.Join(dir, "timeouts.json"))
	if err != nil {
		t.Fatalf("NewTimeoutManager: %v", err)
	}
	tm.persistPath = filepath.Join(blocker, "timeouts.json")

	if err := tm.Set("vllm", 5*time.Second); err == nil {
		t.Fatal("Set must fail when persistence fails")
	}
	if got := tm.Timeout("vllm"); got != 10*time.Minute {
		t.Errorf("timeout changed despite failed persist: %v; want 10m", got)
	}
	if got := len(tm.List()); got != 1 {
		t.Errorf("explicit timeouts = %d after failed Set; want 1", got)
	}
	if err := tm.Set("brand-new", 5*time.Second); err == nil {
		t.Fatal("Set must fail when persistence fails")
	}
	if got := tm.Timeout("brand-new"); got != DefaultRequestTimeout {
		t.Errorf("new timeout became live despite failed persist: %v", got)
	}
	if err := tm.Remove("vllm"); err == nil {
		t.Fatal("Remove must fail when persistence fails")
	}
	if got := tm.Timeout("vllm"); got != 10*time.Minute {
		t.Errorf("config default lost after failed Remove: %v", got)
	}
}

// TestTimeoutManager_CorruptPersistenceFileFailsConstruction (T022 AC3): a
// truncated, garbage or semantically broken timeouts.json must be reported at
// construction instead of silently ignored.
func TestTimeoutManager_CorruptPersistenceFileFailsConstruction(t *testing.T) {
	for name, content := range map[string]string{
		"truncated":         `{"vllm": "10m"`,
		"garbage":           "not json",
		"empty":             "",
		"invalid-duration":  `{"vllm": "ten minutes"}`,
		"nonpositive-value": `{"vllm": "0s"}`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "timeouts.json")
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}

			var buf bytes.Buffer
			log.SetOutput(&buf)
			tm, err := NewTimeoutManager(map[string]string{"vllm": "10m"}, DefaultRequestTimeout, path)
			log.SetOutput(os.Stderr)

			if err == nil {
				t.Fatalf("NewTimeoutManager silently accepted corrupt %s persistence file", name)
			}
			if msg := buf.String(); !strings.Contains(msg, "corrupt") && !strings.Contains(msg, path) {
				t.Errorf("corruption must be logged naming the file, got %q", msg)
			}
			// Callers that discard the error (server.NewServer does) must still
			// get a usable manager instead of a nil one that panics on the
			// first request, with config timeouts still in force.
			if tm == nil {
				t.Fatal("corrupt persistence file yielded a nil manager: a caller that discards the error would panic")
			}
			if got := tm.Timeout("vllm"); got != 10*time.Minute {
				t.Errorf("config timeout lost = %v; want 10m", got)
			}
			if got := tm.Timeout("unknown"); got != DefaultRequestTimeout {
				t.Errorf("default timeout lost = %v", got)
			}
		})
	}
}
