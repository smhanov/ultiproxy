package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/codec"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

type streamEventEncoder interface {
	EncodeEvent(evt ir.Event) error
	Close() error
}

// Telemetry error classes for outcomes the provider did not cause: the provider
// may have succeeded, but the proxy could not serialize the response or the
// client did not receive it. Such requests are recorded with
// stream_complete=0 so per-client accounting never claims a delivery that did
// not happen.
const (
	errClassResponseEncode = "response_encode_failed"
	errClassStreamEncode   = "stream_encode_failed"
	errClassStreamClose    = "stream_encode_close_failed"
	errClassClientWrite    = "client_write_failed"
	errClassClientGone     = "client_disconnected"
)

// logTelemetryError reports a telemetry enqueue failure instead of swallowing
// it: queue pressure (ErrQueueFull) and a closed writer (ErrClosed) otherwise
// disappear, and the accounting quietly becomes incomplete.
func logTelemetryError(what string, err error) {
	if err == nil {
		return
	}
	log.Printf("ultiproxy: telemetry %s: %v", what, err)
}

// stripProviderPrefix removes a "<provider>/" prefix from a model identifier.
// It only strips when the prefix exactly matches the resolved provider name
// (or the prefix names the provider's own namespace), preserving models that
// legitimately contain slashes (e.g. OpenRouter's "meta-llama/llama-3.3-70b").
func stripProviderPrefix(model, providerName string) string {
	if providerName == "" || model == "" {
		return model
	}
	idx := strings.Index(model, "/")
	if idx <= 0 {
		return model
	}
	prefix := model[:idx]
	if strings.EqualFold(prefix, providerName) {
		return model[idx+1:]
	}
	return model
}

// modelsCacheProvider is implemented by providers that discovered their
// upstream model list and can hand it back from cache. openaicompat lanes
// with the ModelListPassthrough quirk populate that cache at construction
// (and on every explicit FetchModels). The aggregated /v1/models handler
// reads the cache only, so listing models never fans out to the upstreams.
type modelsCacheProvider interface {
	CachedModels() []string
}

// defaultModelProvider is implemented by lanes that have a real default
// upstream model to send when a request names no model: openaicompat lanes
// with quirks.default_model, and antigravity's compiled-in default. The
// aggregated /v1/models handler uses it as the escape hatch that keeps a
// non-discoverable lane advertised: "<lane>/<default>" is a routable id, so it
// is listed even when the discovery cache is empty.
type defaultModelProvider interface {
	DefaultModel() string
}

// EnvHideTestLanes removes clearly-test lanes (probe/fake/...) from the model
// listing. Set it to 1 on a daemon whose registry carries wiring-exercise
// lanes, so production clients never see them as models.
const EnvHideTestLanes = "ULTIPROXY_HIDE_TEST_LANES"

// hideTestLanes reports whether test lanes must be left out of the model
// listing. Anything but an explicit truthy value keeps them listed, so the
// default changes nothing for real lanes.
func hideTestLanes() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvHideTestLanes))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// isTestLane reports whether a lane name marks a lane that exists to exercise
// ultiproxy's own wiring rather than serve a real upstream ("probe", "fake",
// "mock", "test", and the same words as a prefix).
func isTestLane(name string) bool {
	switch name {
	case "probe", "fake", "mock", "test":
		return true
	}
	for _, prefix := range []string{"probe-", "fake-", "mock-", "test-"} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// handleModels serves the aggregated model list.
//
// Sources, in order (first occurrence of an id wins):
//
//  1. the state snapshot's model map (aliases synced from the catalog, plus
//     any runtime toggle_model entries), honouring the Enabled flag;
//  2. the alias catalog itself, so a server built without a state manager
//     still lists its configured aliases;
//  3. every registered lane: one entry per cached discovered upstream model
//     as "<lane>/<model>", plus "<lane>/<default>" when the lane has a real
//     default model. Those ids are exactly what routing accepts. A lane name
//     on its own is NOT listed: "model": "<lane>" still routes (legacy
//     prefix form), but a lane name is not a model, and advertising it sent
//     clients to ids that reach no model. A lane with no discovery cache and
//     no default model - anthropic, codex, custom kinds - contributes no
//     entries at all: no fake ids are invented for it, and it is never probed
//     on the fly. Lanes named like test wiring ("probe", "fake", ...) are
//     skipped entirely when ULTIPROXY_HIDE_TEST_LANES is set.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	type ModelEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
		// ContextLength / MaxOutputTokens surface the alias catalog's
		// context_limit / max_output (OpenAI-style flat limits fields).
		// ContextLength is advisory metadata: it is reported to clients but
		// not enforced on the request path.
		ContextLength   int `json:"context_length,omitempty"`
		MaxModelLen     int `json:"max_model_len,omitempty"`
		MaxOutputTokens int `json:"max_output_tokens,omitempty"`
	}

	type ModelsResponse struct {
		Object string       `json:"object"`
		Data   []ModelEntry `json:"data"`
	}

	data := []ModelEntry{}
	seen := make(map[string]bool)
	add := func(id, ownedBy string, contextLength, maxOutput int) {
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		data = append(data, ModelEntry{
			ID:              id,
			Object:          "model",
			Created:         1700000000,
			OwnedBy:         ownedBy,
			ContextLength:   contextLength,
			MaxModelLen:     contextLength,
			MaxOutputTokens: maxOutput,
		})
	}

	var snap *state.RuntimeSnapshot
	if s.sm != nil {
		snap = s.sm.Snapshot()
	}

	// 1. State snapshot models (aliases as synced, plus runtime toggles).
	if snap != nil && snap.Models != nil {
		keys := make([]string, 0, len(snap.Models))
		for k := range snap.Models {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			m := snap.Models[k]
			if !m.Enabled {
				continue
			}
			add(m.ID, m.Provider, resolvedContext(m.ContextLimit, s.discoveredContext(m.Provider, aliasUpstream(s.catalog, m.ID))), m.MaxOutput)
		}
	}

	// 2. Alias catalog (covers servers built without a state manager).
	if s.catalog != nil {
		for _, alias := range s.catalog.Sorted() {
			entry, ok := s.catalog.Get(alias)
			if !ok {
				continue
			}
			if snap != nil && snap.Models != nil {
				if mr, ok := snap.Models[alias]; ok && !mr.Enabled {
					continue
				}
			}
			add(alias, entry.Provider, resolvedContext(entry.ContextLimit, s.discoveredContext(entry.Provider, entry.Upstream)), entry.MaxOutput)
		}
	}

	// 3. Registered lanes.
	if s.registry != nil {
		hideTest := hideTestLanes()
		for _, name := range s.registry.Names() {
			if hideTest && isTestLane(name) {
				continue
			}
			bundle, ok := s.registry.Get(name)
			if !ok || bundle.Inference == nil {
				continue
			}
			if cacher, ok := bundle.Inference.(modelsCacheProvider); ok {
				if meta, ok := bundle.Inference.(provider.ModelInfoCache); ok {
					info := meta.CachedModelInfo()
					sort.Slice(info, func(i, j int) bool { return info[i].ID < info[j].ID })
					for _, m := range info {
						add(name+"/"+m.ID, name, m.ContextLength, 0)
					}
				} else {
					discovered := cacher.CachedModels()
					sort.Strings(discovered)
					for _, m := range discovered {
						add(name+"/"+m, name, 0, 0)
					}
				}
			}
			// Escape hatch for lanes with no model discovery: a real default
			// model still yields exactly one routable, advertised id.
			if def, ok := bundle.Inference.(defaultModelProvider); ok {
				if m := def.DefaultModel(); m != "" {
					add(name+"/"+m, name, s.discoveredContext(name, m), 0)
				}
			}
		}
	}

	resp := ModelsResponse{
		Object: "list",
		Data:   data,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// resolvedContext prefers an operator-set alias context_limit over a
// discovered upstream window. Zero means unknown and stays omitted.
func resolvedContext(aliasLimit, discovered int) int {
	if aliasLimit > 0 {
		return aliasLimit
	}
	if discovered > 0 {
		return discovered
	}
	return 0
}

func aliasUpstream(catalog *ModelCatalog, alias string) string {
	if catalog == nil {
		return ""
	}
	entry, ok := catalog.Get(alias)
	if !ok {
		return ""
	}
	return entry.Upstream
}

func (s *Server) discoveredContext(lane, upstream string) int {
	if s == nil || s.registry == nil || lane == "" || upstream == "" {
		return 0
	}
	bundle, ok := s.registry.Get(lane)
	if !ok || bundle.Inference == nil {
		return 0
	}
	meta, ok := bundle.Inference.(provider.ModelInfoCache)
	if !ok {
		return 0
	}
	for _, m := range meta.CachedModelInfo() {
		if m.ID == upstream {
			return m.ContextLength
		}
	}
	return 0
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":{"message":"failed to read request body"}}`, http.StatusBadRequest)
		return
	}

	decoded, err := codec.DecodeChatCompletionRequest(body)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"invalid chat completion request: %v"}}`, err), http.StatusBadRequest)
		return
	}

	s.dispatchRequest(w, r, decoded.Messages, decoded.Options, decoded.Model, decoded.Stream,
		decoded.ToolsRequested,
		func(writer io.Writer) streamEventEncoder {
			return codec.NewOpenAIStreamEncoder(writer, decoded.Model, decoded.IncludeUsage)
		},
		func(resp *ir.Response, model string) ([]byte, error) {
			return codec.EncodeChatCompletionResponse(resp, model)
		},
	)
}

func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":{"message":"failed to read request body"}}`, http.StatusBadRequest)
		return
	}

	decoded, err := codec.DecodeMessagesRequest(body)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"invalid messages request: %v"}}`, err), http.StatusBadRequest)
		return
	}

	s.dispatchRequest(w, r, decoded.Messages, decoded.Options, decoded.Model, decoded.Stream,
		decoded.ToolsRequested,
		func(writer io.Writer) streamEventEncoder {
			return codec.NewAnthropicStreamEncoder(writer, decoded.Model)
		},
		func(resp *ir.Response, model string) ([]byte, error) {
			return codec.EncodeMessagesResponse(resp, model)
		},
	)
}

func (s *Server) dispatchRequest(
	w http.ResponseWriter,
	r *http.Request,
	messages []*ir.Message,
	options []provider.Option,
	model string,
	stream bool,
	toolsRequested bool,
	createEncoder func(io.Writer) streamEventEncoder,
	encodeResponse func(*ir.Response, string) ([]byte, error),
) {
	startTime := time.Now()
	clientKeyHash := ClientKeyHashFromContext(r.Context())

	// reqID ties this dispatch's request, attempt and usage telemetry rows
	// together (storage.AttemptRecord.RequestID / UsageRecord.RequestID point
	// at storage.RequestRecord.ID), so usage queries can join back to the
	// request that produced them.
	reqID := s.nextRequestID()

	// The request row is opened before its children and re-written with the
	// terminal outcome: storage upserts a record carrying a positive id, so
	// attempts and usage (which reference requests.id through a foreign key)
	// always land on an existing parent row, even when the writer flushes one
	// item per transaction.
	lastProvider := ""
	requestOpened := false
	openRequestRow := func() {
		if s.writer == nil || requestOpened {
			return
		}
		requestOpened = true
		logTelemetryError("open request row", s.writer.TrackRequest(storage.RequestRecord{
			ID:             reqID,
			ClientKeyHash:  clientKeyHash,
			RequestedModel: model,
			ResolvedModel:  model,
			Provider:       lastProvider,
			CreatedAt:      startTime.UTC().Format(time.RFC3339),
			StreamComplete: 0,
			ErrorClass:     "in_flight",
		}))
	}

	// trackAttempt records one provider attempt, opening the request row first so
	// the attempt request_id foreign key is satisfied.
	trackAttempt := func(rec storage.AttemptRecord) {
		if s.writer == nil {
			return
		}
		openRequestRow()
		logTelemetryError("track attempt", s.writer.TrackAttempt(rec))
	}

	// recordRequest writes the terminal request row exactly once per dispatch:
	// on success, on client disconnect, and when every candidate provider fails
	// before commit. Attempt and usage rows already reference reqID, so a
	// missing request row would orphan them (and break the usage joins).
	requestTracked := false
	recordRequest := func(finishReason, errorClass string, streamComplete int, ttftMs int64) {
		if s.writer == nil || requestTracked {
			return
		}
		requestTracked = true
		openRequestRow()
		logTelemetryError("record request", s.writer.TrackRequest(storage.RequestRecord{
			ID:             reqID,
			ClientKeyHash:  clientKeyHash,
			RequestedModel: model,
			ResolvedModel:  model,
			Provider:       lastProvider,
			CreatedAt:      startTime.UTC().Format(time.RFC3339),
			CompletedAt:    time.Now().UTC().Format(time.RFC3339),
			FinishReason:   finishReason,
			StreamComplete: streamComplete,
			ErrorClass:     errorClass,
			TTFTMs:         ttftMs,
			TotalMs:        time.Since(startTime).Milliseconds(),
		}))
	}

	failedProviders := make(map[string]bool)
	var lastErr error
	maxAttempts := 3
	if s.registry != nil && s.registry.Len() > maxAttempts {
		maxAttempts = s.registry.Len()
	}

	var attemptCount int

	for attemptCount < maxAttempts {
		attemptCount++

		routeCtx := ContextWithExcludedProviders(r.Context(), failedProviders)
		provName, err := s.router.Route(routeCtx, model)
		if err != nil {
			// Unknown model: fail immediately with 404 — failover across
			// lanes cannot help a model that maps to no lane at all.
			var uerr *UnknownModelError
			if errors.As(err, &uerr) {
				renderFailedBeforeCommit(w, err, "")
				return
			}
			// No provider available: every candidate lane has failed, so close
			// the request row the recorded attempts point at.
			if requestOpened {
				recordRequest("", errorClassFromError(err), 0, 0)
			}
			renderFailedBeforeCommit(w, lastErr, fmt.Sprintf("no available provider for model %q: %v", model, err))
			return
		}

		if s.registry == nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"message": "provider registry not configured",
					"type":    "bad_gateway",
				},
			})
			return
		}

		prov, ok := s.registry.Get(provName)
		if !ok || prov.Inference == nil {
			failedProviders[provName] = true
			continue
		}
		lastProvider = provName

		if toolsRequested && !prov.Capabilities.Tools {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"message": fmt.Sprintf("model %s does not support tools", model),
					"type":    "model_does_not_support_tools",
				},
			})
			return
		}

		// Resolve the upstream model id: model aliases map to explicit
		// upstream names (e.g. "qwenpoint-3.8" -> "Qwen/Qwen3.8-Instruct-AWQ");
		// otherwise strip a "<provider>/" prefix for prefixed names (e.g.
		// "zai/glm-5.3-flash" -> "glm-5.3-flash"). Mostly the provider
		// forwards the requested identifier verbatim. Last-apply wins, so this
		// overrides the codec's WithModel().
		upstream := ""
		var alias ModelAlias
		var hasAlias bool
		if s.catalog != nil {
			if entry, ok := s.catalog.Get(model); ok {
				alias, hasAlias = entry, true
				upstream = entry.Upstream
			}
		}
		if upstream == "" {
			upstream = stripProviderPrefix(model, provName)
		}
		opts := upstreamOptions(options, upstream, alias, hasAlias, clientKeyHash)

		// Apply the per-provider request timeout (default or MCP-configured).
		// Providers honor context cancellation while calling upstream.
		reqCtx := r.Context()
		if s.timeouts != nil {
			timeout := s.timeouts.Timeout(provName)
			if timeout > 0 {
				var cancel context.CancelFunc
				reqCtx, cancel = context.WithTimeout(reqCtx, timeout)
				defer cancel()
			}
		}

		if stream {
			streamChan, syncErr := prov.Inference.Stream(reqCtx, messages, opts...)
			if syncErr != nil {
				lastErr = syncErr
				// Synchronous error BEFORE headers committed -> failover!
				trackAttempt(storage.AttemptRecord{
					RequestID:  reqID,
					Attempt:    attemptCount,
					Provider:   provName,
					Model:      model,
					StatusCode: http.StatusBadGateway,
					ErrorClass: "stream_sync_error",
				})
				failedProviders[provName] = true
				continue
			}

			// Read first event synchronously before committing headers!
			// ttft is measured when headers commit (first event) and reused when the
			// client disconnects mid-stream.
			var ttft int64
			select {
			case firstEvt, ok := <-streamChan:
				if !ok {
					// Channel closed immediately without events
					trackAttempt(storage.AttemptRecord{
						RequestID:  reqID,
						Attempt:    attemptCount,
						Provider:   provName,
						Model:      model,
						StatusCode: http.StatusBadGateway,
						ErrorClass: "stream_empty",
					})
					failedProviders[provName] = true
					continue
				}

				if upErr, isUpErr := firstEvt.(ir.EventUpstreamError); isUpErr {
					lastErr = fmt.Errorf("upstream error: %s", upErr.Kind)
					// Upstream error BEFORE headers committed -> failover!
					trackAttempt(storage.AttemptRecord{
						RequestID:         reqID,
						Attempt:           attemptCount,
						Provider:          provName,
						Model:             model,
						StatusCode:        http.StatusBadGateway,
						ErrorClass:        upErr.Kind,
						RetryAfterSeconds: upErr.RetryAfterSeconds,
					})
					failedProviders[provName] = true
					continue
				}

				// First event is valid: COMMIT HEADERS!
				ttft = time.Since(startTime).Milliseconds()
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.Header().Set("Connection", "keep-alive")
				w.WriteHeader(http.StatusOK)

				flusher, canFlush := w.(http.Flusher)
				if canFlush {
					flusher.Flush()
				}

				trackAttempt(storage.AttemptRecord{
					RequestID:  reqID,
					Attempt:    attemptCount,
					Provider:   provName,
					Model:      model,
					StatusCode: http.StatusOK,
				})

				// NO FAILOVER AFTER FIRST BYTE: stream rest of events
				var finishReason string
				var errorClass string
				// deliveryFailed records that the client did not receive a complete
				// stream (encode/flush failure, or a client that hung up). The provider
				// may have succeeded, but the request row must not claim a successful
				// delivery, so stream_complete stays 0.
				deliveryFailed := false
				// Usage events are cumulative across a stream (Anthropic reports
				// one per delta): keep the last one and record a single usage row
				// for the request, so per-event rows cannot double-count tokens or
				// cost.
				var lastUsage *ir.EventUsageUpdate

				// classifyDelivery fills in the downstream failure class without ever
				// masking an upstream one.
				classifyDelivery := func(class string) {
					if errorClass == "" {
						errorClass = class
					}
				}

				encoder := createEncoder(w)
				if err := encoder.EncodeEvent(firstEvt); err != nil {
					deliveryFailed = true
					classifyDelivery(errClassStreamEncode)
				}
				if stopEvt, isStop := firstEvt.(ir.EventMessageStop); isStop {
					finishReason = stopEvt.FinishReason
				}
				if canFlush && !deliveryFailed {
					flusher.Flush()
				}

				for ev := range streamChan {
					if midStreamErr, isErr := ev.(ir.EventUpstreamError); isErr {
						// Upstream error mid-stream: emit error frame and TERMINATE. NEVER switch provider!
						errorClass = midStreamErr.Kind
						if err := encoder.EncodeEvent(midStreamErr); err != nil {
							deliveryFailed = true
						}
						if canFlush && !deliveryFailed {
							flusher.Flush()
						}
						break
					}

					if usageEvt, isUsage := ev.(ir.EventUsageUpdate); isUsage {
						usage := usageEvt
						lastUsage = &usage
					}

					if stopEvt, isStop := ev.(ir.EventMessageStop); isStop {
						finishReason = stopEvt.FinishReason
					}

					// The client is gone or the encoder already failed: stop writing
					// (the writes cannot succeed anyway) but keep draining so the
					// provider goroutine is never left blocked on the channel.
					if deliveryFailed {
						continue
					}

					select {
					case <-r.Context().Done():
						deliveryFailed = true
						classifyDelivery(errClassClientGone)
						continue
					default:
					}

					if err := encoder.EncodeEvent(ev); err != nil {
						deliveryFailed = true
						classifyDelivery(errClassStreamEncode)
						continue
					}
					if canFlush {
						flusher.Flush()
					}
				}

				if err := encoder.Close(); err != nil {
					deliveryFailed = true
					classifyDelivery(errClassStreamClose)
				}
				if canFlush && !deliveryFailed {
					flusher.Flush()
				}

				if lastUsage != nil {
					s.trackUsage(reqID, &ir.Usage{
						PromptTokens:             lastUsage.PromptTokens,
						CompletionTokens:         lastUsage.CompletionTokens,
						TotalTokens:              lastUsage.TotalTokens,
						CacheCreationInputTokens: lastUsage.CacheCreationInputTokens,
						CacheReadInputTokens:     lastUsage.CacheReadInputTokens,
						Cost:                     lastUsage.Cost,
					}, alias)
				}
				// A stream is only "complete" when nothing failed it: an
				// upstream error, an encode failure or a client that hung up all
				// leave stream_complete=0.
				streamComplete := 1
				if errorClass != "" || deliveryFailed {
					streamComplete = 0
				}
				recordRequest(finishReason, errorClass, streamComplete, ttft)
				return

			case <-r.Context().Done():
				// Client went away mid-stream: the request still happened, and its
				// attempts/usage rows are already queued against reqID, so close the
				// request row instead of orphaning them.
				recordRequest("", errClassClientGone, 0, ttft)
				return
			}

		} else {
			// Non-streaming request
			resp, genErr := prov.Inference.Generate(reqCtx, messages, opts...)
			if genErr != nil {
				lastErr = genErr
				trackAttempt(storage.AttemptRecord{
					RequestID:  reqID,
					Attempt:    attemptCount,
					Provider:   provName,
					Model:      model,
					StatusCode: http.StatusBadGateway,
					ErrorClass: "generate_error",
				})
				failedProviders[provName] = true
				continue // Failover before commit!
			}

			// Generate success: commit response
			trackAttempt(storage.AttemptRecord{
				RequestID:  reqID,
				Attempt:    attemptCount,
				Provider:   provName,
				Model:      model,
				StatusCode: http.StatusOK,
			})
			s.trackUsage(reqID, resp.Usage, alias)

			// The request row only claims a successful delivery once the response
			// was serialized AND handed to the client: a failed encode (the client
			// gets a 500) or a failed write (the client got nothing) is recorded
			// with stream_complete=0 and the matching error class instead.
			outBytes, err := encodeResponse(resp, model)
			if err != nil {
				recordRequest("", errClassResponseEncode, 0, 0)
				http.Error(w, fmt.Sprintf(`{"error":{"message":"failed to encode response: %v"}}`, err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			if _, err := w.Write(outBytes); err != nil {
				recordRequest(resp.FinishReason, errClassClientWrite, 0, 0)
				return
			}

			recordRequest(resp.FinishReason, "", 1, 0)
			return
		}
	}

	// All candidate providers failed before commit: close the request row so the
	// recorded attempts are not orphaned and the failure is counted.
	recordRequest("", errorClassFromError(lastErr), 0, 0)
	renderFailedBeforeCommit(w, lastErr, "all candidate providers failed before commit")
}

// upstreamOptions builds the provider option slice for one dispatch attempt:
// the client key hash for accounting, the resolved upstream model id (applied
// after the codec's own options so it wins), the catalog alias output cap and
// the catalog alias pricing.
//
// A catalog alias MaxOutput clamps max_tokens: a request that asks for more
// output tokens than the alias allows is reduced to the alias limit, and a
// request that asks for none inherits the alias limit instead of the lane
// default. ContextLimit is deliberately NOT enforced on the request path — it
// is advisory metadata surfaced through /v1/models (context_length), because
// estimating prompt token counts per provider would require a tokenizer the
// proxy does not own.
func upstreamOptions(base []provider.Option, upstream string, alias ModelAlias, hasAlias bool, clientKeyHash string) []provider.Option {
	opts := make([]provider.Option, 0, len(base)+4)
	opts = append(opts, base...)
	opts = append(opts,
		provider.WithClientKeyHash(clientKeyHash),
		provider.WithModel(upstream),
	)
	if !hasAlias {
		return opts
	}
	if alias.InputCost > 0 || alias.OutputCost > 0 {
		// Alias pricing rides along so llmhub lanes can price the request
		// themselves; the recorded usage cost is back-filled from the same rates
		// when the upstream reports none (see trackUsage).
		opts = append(opts, provider.WithCost(alias.InputCost, alias.OutputCost))
	}
	if alias.MaxOutput <= 0 {
		return opts
	}
	cfg := provider.NewRequestConfig(opts...)
	if cfg.MaxTokens > 0 && cfg.MaxTokens <= alias.MaxOutput {
		return opts
	}
	return append(opts, provider.WithMaxTokens(alias.MaxOutput))
}

// trackUsage records the token/cost accounting row of one dispatch.
//
// Cost is whatever the upstream reported (llmhub lanes: provider-reported cost,
// or the rates supplied through provider.WithCost); when the upstream prices
// nothing, the resolved alias input_cost/output_cost rates fill the estimate
// in, so lanes that never report a cost are still accounted for. Cached tokens
// are recorded as the sum of cache-creation and cache-read input tokens (the
// two flavours of prompt tokens served from a cache).
func (s *Server) trackUsage(reqID int64, u *ir.Usage, alias ModelAlias) {
	if s.writer == nil || u == nil {
		return
	}
	cost := u.Cost
	if cost == 0 {
		cost = estimatedCost(int64(u.PromptTokens), int64(u.CompletionTokens), alias.InputCost, alias.OutputCost)
	}
	logTelemetryError("track usage", s.writer.TrackUsage(storage.UsageRecord{
		RequestID:        reqID,
		PromptTokens:     int64(u.PromptTokens),
		CompletionTokens: int64(u.CompletionTokens),
		ReasoningTokens:  int64(u.ReasoningTokens),
		CachedTokens:     int64(u.CacheCreationInputTokens + u.CacheReadInputTokens),
		Cost:             cost,
	}))
}

// estimatedCost prices a prompt/completion token pair at the given
// US-dollar-per-million-token rates. Missing rates price nothing, so a
// provider-reported cost is never overwritten with 0.
func estimatedCost(promptTokens, completionTokens int64, inputCostPerMillion, outputCostPerMillion float64) float64 {
	if inputCostPerMillion <= 0 && outputCostPerMillion <= 0 {
		return 0
	}
	return float64(promptTokens)*inputCostPerMillion/1_000_000 + float64(completionTokens)*outputCostPerMillion/1_000_000
}

// classifyUpstreamError maps a dispatch error onto the HTTP status and the
// machine-readable error class used both for the client error response and for
// the telemetry request row.
func classifyUpstreamError(err error, fallbackMsg string) (statusCode int, errType, errMsg string) {
	statusCode = http.StatusBadGateway
	errMsg = fallbackMsg
	errType = "bad_gateway"
	var uerr *UnknownModelError
	if errors.As(err, &uerr) {
		return http.StatusNotFound, "unknown_model", uerr.Error()
	}
	if err != nil {
		errMsg = err.Error()
		msgLower := strings.ToLower(errMsg)
		if strings.Contains(msgLower, "status 429") || strings.Contains(msgLower, "http 429") || strings.Contains(msgLower, "429 too many") || strings.Contains(msgLower, "resource_exhausted") || strings.Contains(msgLower, "quota_exceeded") || strings.Contains(msgLower, "daily token limit") || strings.Contains(msgLower, "rate limit") {
			return http.StatusTooManyRequests, "rate_limit_exceeded", errMsg
		} else if strings.Contains(msgLower, "status 401") || strings.Contains(msgLower, "http 401") || strings.Contains(msgLower, "unauthorized") {
			return http.StatusUnauthorized, "authentication_error", errMsg
		} else if strings.Contains(msgLower, "status 403") || strings.Contains(msgLower, "http 403") || strings.Contains(msgLower, "forbidden") {
			return http.StatusForbidden, "permission_denied", errMsg
		} else if strings.Contains(msgLower, "status 404") || strings.Contains(msgLower, "http 404") || strings.Contains(msgLower, "not found") {
			return http.StatusNotFound, "not_found", errMsg
		}
	}
	return statusCode, errType, errMsg
}

// errorClassFromError returns the machine-readable error class of a failed
// dispatch (see classifyUpstreamError).
func errorClassFromError(err error) string {
	_, errType, _ := classifyUpstreamError(err, "")
	return errType
}

func renderFailedBeforeCommit(w http.ResponseWriter, lastErr error, fallbackMsg string) {
	statusCode, errType, errMsg := classifyUpstreamError(lastErr, fallbackMsg)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": errMsg,
			"type":    errType,
		},
	})
}
