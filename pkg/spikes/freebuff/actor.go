package freebuff

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gofrs/flock"
)

var (
	ErrLeaseConflict = errors.New("freebuff lease conflict: lock already held")
	ErrQueueFull     = errors.New("freebuff queue is full")
	ErrNotAcquired   = errors.New("freebuff actor lock not acquired")
	ErrClosed        = errors.New("freebuff actor is closed")
)

// Session represents the Freebuff upstream session state.
type Session struct {
	Status     string `json:"status,omitempty"`
	InstanceID string `json:"instanceId"`
	Model      string `json:"model"`
}

// AgentRun represents an upstream run instance.
type AgentRun struct {
	RunID      string `json:"run_id"`
	RunIDAlt   string `json:"runId"`
	AgentID    string `json:"agent_id"`
	InstanceID string `json:"instance_id"`
	Status     string `json:"status"`
}

func (r *AgentRun) GetRunID() string {
	if r.RunID != "" {
		return r.RunID
	}
	return r.RunIDAlt
}

type streamJob struct {
	ctx    context.Context
	req    *http.Request
	body   io.Reader
	respCh chan streamResult
}

type streamResult struct {
	body io.ReadCloser
	err  error
}

// Option configures the FreebuffAccountActor.
type Option func(*FreebuffAccountActor)

// WithBaseURL sets the upstream base URL.
func WithBaseURL(url string) Option {
	return func(a *FreebuffAccountActor) {
		a.baseURL = url
	}
}

// WithQueueCapacity sets the bounded queue capacity.
func WithQueueCapacity(cap int) Option {
	return func(a *FreebuffAccountActor) {
		a.queueCap = cap
	}
}

// WithInstanceID sets the initial instance ID.
func WithInstanceID(id string) Option {
	return func(a *FreebuffAccountActor) {
		a.instanceID = id
	}
}

// FreebuffAccountActor is a single-owner serialized actor for the Freebuff session.
type FreebuffAccountActor struct {
	lockPath   string
	httpClient *http.Client
	token      string
	baseURL    string

	fl       *flock.Flock
	isLocked bool
	mu       sync.Mutex

	instanceID string
	boundModel string
	currentRun *AgentRun

	queueCap   int
	queue      chan streamJob
	stopCh     chan struct{}
	workerDone chan struct{}
	closed     bool
}

// NewFreebuffAccountActor creates a new FreebuffAccountActor.
func NewFreebuffAccountActor(lockPath string, httpClient *http.Client, token string, opts ...Option) (*FreebuffAccountActor, error) {
	if lockPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get user home dir: %w", err)
		}
		lockPath = filepath.Join(home, ".local", "state", "ultiproxy", "locks", "freebuff.lock")
	}

	if err := os.MkdirAll(filepath.Dir(lockPath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create lock directory: %w", err)
	}

	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	actor := &FreebuffAccountActor{
		lockPath:   lockPath,
		httpClient: httpClient,
		token:      token,
		baseURL:    "https://api.codebuff.com",
		queueCap:   64,
		stopCh:     make(chan struct{}),
		workerDone: make(chan struct{}),
		fl:         flock.New(lockPath),
	}

	for _, opt := range opts {
		opt(actor)
	}

	actor.queue = make(chan streamJob, actor.queueCap)
	go actor.workerLoop()

	return actor, nil
}

// BaseURL returns current baseURL.
func (a *FreebuffAccountActor) BaseURL() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.baseURL
}

// Token returns current token.
func (a *FreebuffAccountActor) Token() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.token
}

// HTTPClient returns current http.Client.
func (a *FreebuffAccountActor) HTTPClient() *http.Client {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.httpClient
}

// SetInstanceID updates instance ID.
func (a *FreebuffAccountActor) SetInstanceID(id string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.instanceID = id
}

// SetToken updates the bearer token.
func (a *FreebuffAccountActor) SetToken(tok string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.token = tok
}

// TryAcquire attempts to acquire the advisory lock non-blockingly.
// Returns ErrLeaseConflict if already locked by another actor.
func (a *FreebuffAccountActor) TryAcquire() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.closed {
		return ErrClosed
	}
	if a.isLocked {
		return nil
	}

	locked, err := a.fl.TryLock()
	if err != nil {
		return fmt.Errorf("error attempting lock: %w", err)
	}
	if !locked {
		return ErrLeaseConflict
	}

	a.isLocked = true
	return nil
}

// Acquire acquires the advisory lock, waiting until context is done if held.
// If context expires, returns context error.
func (a *FreebuffAccountActor) Acquire(ctxs ...context.Context) error {
	var ctx context.Context
	if len(ctxs) > 0 && ctxs[0] != nil {
		ctx = ctxs[0]
	} else {
		ctx = context.Background()
	}

	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return ErrClosed
	}
	if a.isLocked {
		a.mu.Unlock()
		return nil
	}
	a.mu.Unlock()

	locked, err := a.fl.TryLockContext(ctx, 15*time.Millisecond)
	if err != nil {
		return err
	}
	if !locked {
		return ErrLeaseConflict
	}

	a.mu.Lock()
	a.isLocked = true
	a.mu.Unlock()
	return nil
}

// Release unlocks the advisory lock.
func (a *FreebuffAccountActor) Release() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.isLocked {
		return nil
	}
	err := a.fl.Unlock()
	a.isLocked = false
	return err
}

// InstanceID returns current instance ID.
func (a *FreebuffAccountActor) InstanceID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.instanceID
}

// BoundModel returns current bound model.
func (a *FreebuffAccountActor) BoundModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.boundModel
}

// Reconcile synchronizes actor state with GET /freebuff/session.
func (a *FreebuffAccountActor) Reconcile(ctxs ...context.Context) error {
	var ctx context.Context
	if len(ctxs) > 0 && ctxs[0] != nil {
		ctx = ctxs[0]
	} else {
		ctx = context.Background()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/freebuff/session", nil)
	if err != nil {
		return err
	}
	a.mu.Lock()
	tok := a.token
	instID := a.instanceID
	client := a.httpClient
	a.mu.Unlock()

	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if instID != "" {
		req.Header.Set("x-freebuff-instance-id", instID)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("unexpected status code %d from reconcile", resp.StatusCode)
	}

	var sess Session
	if err := json.NewDecoder(resp.Body).Decode(&sess); err != nil {
		return fmt.Errorf("failed to decode session: %w", err)
	}

	a.mu.Lock()
	a.instanceID = sess.InstanceID
	a.boundModel = sess.Model
	a.mu.Unlock()

	return nil
}

// Bind updates the bound model via POST /freebuff/session.
// Accepts Bind(model) or Bind(ctx, model).
func (a *FreebuffAccountActor) Bind(ctxOrModel any, optionalModel ...string) error {
	var ctx context.Context
	var model string

	switch v := ctxOrModel.(type) {
	case context.Context:
		ctx = v
		if len(optionalModel) > 0 {
			model = optionalModel[0]
		}
	case string:
		ctx = context.Background()
		model = v
	default:
		return fmt.Errorf("invalid arguments to Bind")
	}

	a.mu.Lock()
	instID := a.instanceID
	tok := a.token
	client := a.httpClient
	a.mu.Unlock()

	doBind := func() (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/freebuff/session", bytes.NewReader([]byte("{}")))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-freebuff-model", model)
		if tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		if instID != "" {
			req.Header.Set("x-freebuff-instance-id", instID)
		}
		return client.Do(req)
	}

	resp, err := doBind()
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusConflict {
		resp.Body.Close()
		_ = a.DeleteSession(ctx)
		a.mu.Lock()
		instID = a.instanceID
		a.mu.Unlock()
		resp, err = doBind()
		if err != nil {
			return err
		}
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("failed to bind model, status: %d: %s", resp.StatusCode, string(body))
	}

	var sess Session
	if err := json.Unmarshal(body, &sess); err == nil {
		a.mu.Lock()
		if sess.InstanceID != "" {
			a.instanceID = sess.InstanceID
		}
		if sess.Model != "" {
			a.boundModel = sess.Model
		} else {
			a.boundModel = model
		}
		a.mu.Unlock()
	} else {
		a.mu.Lock()
		a.boundModel = model
		a.mu.Unlock()
	}

	return nil
}

// DeleteSession deletes the Freebuff session via DELETE /freebuff/session.
func (a *FreebuffAccountActor) DeleteSession(ctxs ...context.Context) error {
	var ctx context.Context
	if len(ctxs) > 0 && ctxs[0] != nil {
		ctx = ctxs[0]
	} else {
		ctx = context.Background()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, a.baseURL+"/freebuff/session", nil)
	if err != nil {
		return err
	}
	a.mu.Lock()
	tok := a.token
	instID := a.instanceID
	client := a.httpClient
	a.mu.Unlock()

	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if instID != "" {
		req.Header.Set("x-freebuff-instance-id", instID)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("failed to delete session, status: %d", resp.StatusCode)
	}

	a.mu.Lock()
	a.instanceID = ""
	a.boundModel = ""
	a.mu.Unlock()

	return nil
}

// StartRun initiates an agent run via POST /agent-runs.
// Accepts StartRun(agentID) or StartRun(ctx, agentID).
func (a *FreebuffAccountActor) StartRun(ctxOrAgentID any, optionalAgentID ...string) (*AgentRun, error) {
	var ctx context.Context
	var agentID string

	switch v := ctxOrAgentID.(type) {
	case context.Context:
		ctx = v
		if len(optionalAgentID) > 0 {
			agentID = optionalAgentID[0]
		}
	case string:
		ctx = context.Background()
		agentID = v
	default:
		return nil, fmt.Errorf("invalid arguments to StartRun")
	}

	a.mu.Lock()
	instID := a.instanceID
	tok := a.token
	client := a.httpClient
	a.mu.Unlock()

	payload, err := json.Marshal(map[string]any{
		"action":         "START",
		"agentId":        agentID,
		"ancestorRunIds": []string{},
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/agent-runs", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if instID != "" {
		req.Header.Set("x-freebuff-instance-id", instID)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("failed to start agent run, status: %d", resp.StatusCode)
	}

	var run AgentRun
	if err := json.NewDecoder(resp.Body).Decode(&run); err != nil {
		return nil, fmt.Errorf("failed to decode agent run: %w", err)
	}

	a.mu.Lock()
	a.currentRun = &run
	a.mu.Unlock()

	return &run, nil
}

// FetchUsage calls POST /usage with fingerprintId via the actor.
func (a *FreebuffAccountActor) FetchUsage(ctx context.Context, fingerprintID string) ([]byte, error) {
	if fingerprintID == "" {
		fingerprintID = "cli-usage"
	}
	payload, err := json.Marshal(map[string]string{
		"fingerprintId": fingerprintID,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/usage", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	a.mu.Lock()
	tok := a.token
	instID := a.instanceID
	client := a.httpClient
	a.mu.Unlock()

	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if instID != "" {
		req.Header.Set("x-freebuff-instance-id", instID)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("failed to fetch usage, status: %d", resp.StatusCode)
	}

	return io.ReadAll(resp.Body)
}

// GetSession retrieves current session via GET /freebuff/session through actor.
func (a *FreebuffAccountActor) GetSession(ctx context.Context) (*Session, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/freebuff/session", nil)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	tok := a.token
	instID := a.instanceID
	client := a.httpClient
	a.mu.Unlock()

	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if instID != "" {
		req.Header.Set("x-freebuff-instance-id", instID)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("failed to get session, status: %d", resp.StatusCode)
	}

	var sess Session
	if err := json.NewDecoder(resp.Body).Decode(&sess); err != nil {
		return nil, err
	}

	return &sess, nil
}

// DoStream enqueues an arbitrary streaming HTTP request through the actor's FIFO queue.
func (a *FreebuffAccountActor) DoStream(ctx context.Context, req *http.Request) (io.ReadCloser, error) {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return nil, ErrClosed
	}
	a.mu.Unlock()

	respCh := make(chan streamResult, 1)
	job := streamJob{
		ctx:    ctx,
		req:    req,
		respCh: respCh,
	}

	select {
	case a.queue <- job:
	default:
		return nil, ErrQueueFull
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-respCh:
		return res.body, res.err
	case <-a.stopCh:
		return nil, ErrClosed
	}
}

// Stream serializes request bodies through the FIFO bounded queue.
// Accepts Stream(body) or Stream(ctx, body).
func (a *FreebuffAccountActor) Stream(ctxOrBody any, optionalBody ...io.Reader) (io.ReadCloser, error) {
	var ctx context.Context
	var body io.Reader

	switch v := ctxOrBody.(type) {
	case context.Context:
		ctx = v
		if len(optionalBody) > 0 {
			body = optionalBody[0]
		}
	case io.Reader:
		ctx = context.Background()
		body = v
	default:
		return nil, fmt.Errorf("invalid arguments to Stream")
	}

	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return nil, ErrClosed
	}
	a.mu.Unlock()

	respCh := make(chan streamResult, 1)
	job := streamJob{
		ctx:    ctx,
		body:   body,
		respCh: respCh,
	}

	// Non-blocking bounded enqueue
	select {
	case a.queue <- job:
	default:
		return nil, ErrQueueFull
	}

	// Wait for response or context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-respCh:
		return res.body, res.err
	case <-a.stopCh:
		return nil, ErrClosed
	}
}

// Pending returns the number of stream requests currently pending in queue.
func (a *FreebuffAccountActor) Pending() int {
	return len(a.queue)
}

// QueueSize returns the maximum capacity of the stream queue.
func (a *FreebuffAccountActor) QueueSize() int {
	return a.queueCap
}

// workerLoop processes stream requests sequentially (FIFO).
func (a *FreebuffAccountActor) workerLoop() {
	defer close(a.workerDone)

	for {
		select {
		case <-a.stopCh:
			return
		case job, ok := <-a.queue:
			if !ok {
				return
			}
			a.processStreamJob(job)
		}
	}
}

func (a *FreebuffAccountActor) processStreamJob(job streamJob) {
	// Check if already canceled before execution
	if err := job.ctx.Err(); err != nil {
		job.respCh <- streamResult{err: err}
		return
	}

	var req *http.Request
	if job.req != nil {
		req = job.req.WithContext(job.ctx)
	} else {
		var err error
		req, err = http.NewRequestWithContext(job.ctx, http.MethodPost, a.baseURL+"/agent-runs/stream", job.body)
		if err != nil {
			job.respCh <- streamResult{err: err}
			return
		}
		if a.token != "" {
			req.Header.Set("Authorization", "Bearer "+a.token)
		}
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		job.respCh <- streamResult{err: err}
		return
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		job.respCh <- streamResult{err: fmt.Errorf("stream failed with status %d", resp.StatusCode)}
		return
	}

	job.respCh <- streamResult{body: resp.Body}
}

// Close releases the lock and stops the actor.
func (a *FreebuffAccountActor) Close() error {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return nil
	}
	a.closed = true
	close(a.stopCh)
	a.mu.Unlock()

	// Drain queue with errors
	for {
		select {
		case job := <-a.queue:
			job.respCh <- streamResult{err: ErrClosed}
		default:
			goto drained
		}
	}
drained:

	<-a.workerDone
	_ = a.Release()
	return nil
}
