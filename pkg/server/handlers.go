package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/codec"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

type streamEventEncoder interface {
	EncodeEvent(evt ir.Event) error
	Close() error
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

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	type ModelEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	type ModelsResponse struct {
		Object string       `json:"object"`
		Data   []ModelEntry `json:"data"`
	}

	var data []ModelEntry
	seen := make(map[string]bool)

	// Pull from StateManager if available
	if s.sm != nil {
		snap := s.sm.Snapshot()
		if snap != nil && snap.Models != nil {
			var keys []string
			for k := range snap.Models {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				m := snap.Models[k]
				if m.Enabled {
					data = append(data, ModelEntry{
						ID:      m.ID,
						Object:  "model",
						Created: 1700000000,
						OwnedBy: m.Provider,
					})
					seen[m.ID] = true
				}
			}
		}
	}

	if data == nil {
		data = []ModelEntry{}
	}

	resp := ModelsResponse{
		Object: "list",
		Data:   data,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
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
			// No provider available
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

		opts := append(options, provider.WithClientKeyHash(clientKeyHash))
		// Resolve the upstream model id: model aliases map to explicit
		// upstream names (e.g. "qwenpoint-3.8" -> "Qwen/Qwen3.8-Instruct-AWQ");
		// otherwise strip a "<provider>/" prefix for prefixed names (e.g.
		// "zai/glm-5.3-flash" -> "glm-5.3-flash"). Mostly the provider
		// forwards the requested identifier verbatim. Last-apply wins, so this
		// overrides the codec's WithModel().
		upstream := ""
		if s.catalog != nil {
			if u, ok := s.catalog.UpstreamName(model); ok {
				upstream = u
			}
		}
		if upstream == "" {
			upstream = stripProviderPrefix(model, provName)
		}
		opts = append(opts, provider.WithModel(upstream))

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
				if s.writer != nil {
					_ = s.writer.TrackAttempt(storage.AttemptRecord{
						Attempt:    attemptCount,
						Provider:   provName,
						Model:      model,
						StatusCode: http.StatusBadGateway,
						ErrorClass: "stream_sync_error",
					})
				}
				failedProviders[provName] = true
				continue
			}

			// Read first event synchronously before committing headers!
			select {
			case firstEvt, ok := <-streamChan:
				if !ok {
					// Channel closed immediately without events
					if s.writer != nil {
						_ = s.writer.TrackAttempt(storage.AttemptRecord{
							Attempt:    attemptCount,
							Provider:   provName,
							Model:      model,
							StatusCode: http.StatusBadGateway,
							ErrorClass: "stream_empty",
						})
					}
					failedProviders[provName] = true
					continue
				}

				if upErr, isUpErr := firstEvt.(ir.EventUpstreamError); isUpErr {
					lastErr = fmt.Errorf("upstream error: %s", upErr.Kind)
					// Upstream error BEFORE headers committed -> failover!
					if s.writer != nil {
						_ = s.writer.TrackAttempt(storage.AttemptRecord{
							Attempt:           attemptCount,
							Provider:          provName,
							Model:             model,
							StatusCode:        http.StatusBadGateway,
							ErrorClass:        upErr.Kind,
							RetryAfterSeconds: upErr.RetryAfterSeconds,
						})
					}
					failedProviders[provName] = true
					continue
				}

				// First event is valid: COMMIT HEADERS!
				ttft := time.Since(startTime).Milliseconds()
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.Header().Set("Connection", "keep-alive")
				w.WriteHeader(http.StatusOK)

				flusher, canFlush := w.(http.Flusher)
				if canFlush {
					flusher.Flush()
				}

				if s.writer != nil {
					_ = s.writer.TrackAttempt(storage.AttemptRecord{
						Attempt:    attemptCount,
						Provider:   provName,
						Model:      model,
						StatusCode: http.StatusOK,
					})
				}

				// NO FAILOVER AFTER FIRST BYTE: stream rest of events
				var finishReason string
				var errorClass string

				encoder := createEncoder(w)
				_ = encoder.EncodeEvent(firstEvt)
				if stopEvt, isStop := firstEvt.(ir.EventMessageStop); isStop {
					finishReason = stopEvt.FinishReason
				}
				if canFlush {
					flusher.Flush()
				}

				for ev := range streamChan {
					if midStreamErr, isErr := ev.(ir.EventUpstreamError); isErr {
						// Upstream error mid-stream: emit error frame and TERMINATE. NEVER switch provider!
						errorClass = midStreamErr.Kind
						_ = encoder.EncodeEvent(midStreamErr)
						if canFlush {
							flusher.Flush()
						}
						break
					}

					if usageEvt, isUsage := ev.(ir.EventUsageUpdate); isUsage {
						if s.writer != nil {
							_ = s.writer.TrackUsage(storage.UsageRecord{
								PromptTokens:     int64(usageEvt.PromptTokens),
								CompletionTokens: int64(usageEvt.CompletionTokens),
								Cost:             usageEvt.Cost,
							})
						}
					}

					if stopEvt, isStop := ev.(ir.EventMessageStop); isStop {
						finishReason = stopEvt.FinishReason
					}

					_ = encoder.EncodeEvent(ev)
					if canFlush {
						flusher.Flush()
					}
				}

				_ = encoder.Close()
				if canFlush {
					flusher.Flush()
				}

				totalMs := time.Since(startTime).Milliseconds()
				if s.writer != nil {
					_ = s.writer.TrackRequest(storage.RequestRecord{
						ClientKeyHash:  clientKeyHash,
						RequestedModel: model,
						ResolvedModel:  model,
						Provider:       provName,
						CreatedAt:      startTime.UTC().Format(time.RFC3339),
						CompletedAt:    time.Now().UTC().Format(time.RFC3339),
						FinishReason:   finishReason,
						StreamComplete: 1,
						ErrorClass:     errorClass,
						TTFTMs:         ttft,
						TotalMs:        totalMs,
					})
				}
				return

			case <-r.Context().Done():
				return
			}

		} else {
			// Non-streaming request
			resp, genErr := prov.Inference.Generate(reqCtx, messages, opts...)
			if genErr != nil {
				lastErr = genErr
				if s.writer != nil {
					_ = s.writer.TrackAttempt(storage.AttemptRecord{
						Attempt:    attemptCount,
						Provider:   provName,
						Model:      model,
						StatusCode: http.StatusBadGateway,
						ErrorClass: "generate_error",
					})
				}
				failedProviders[provName] = true
				continue // Failover before commit!
			}

			// Generate success: commit response
			if s.writer != nil {
				_ = s.writer.TrackAttempt(storage.AttemptRecord{
					Attempt:    attemptCount,
					Provider:   provName,
					Model:      model,
					StatusCode: http.StatusOK,
				})
				if resp.Usage != nil {
					_ = s.writer.TrackUsage(storage.UsageRecord{
						PromptTokens:     int64(resp.Usage.PromptTokens),
						CompletionTokens: int64(resp.Usage.CompletionTokens),
						ReasoningTokens:  int64(resp.Usage.ReasoningTokens),
						Cost:             resp.Usage.Cost,
					})
				}
				totalMs := time.Since(startTime).Milliseconds()
				_ = s.writer.TrackRequest(storage.RequestRecord{
					ClientKeyHash:  clientKeyHash,
					RequestedModel: model,
					ResolvedModel:  model,
					Provider:       provName,
					CreatedAt:      startTime.UTC().Format(time.RFC3339),
					CompletedAt:    time.Now().UTC().Format(time.RFC3339),
					FinishReason:   resp.FinishReason,
					StreamComplete: 1,
					TotalMs:        totalMs,
				})
			}

			outBytes, err := encodeResponse(resp, model)
			if err != nil {
				http.Error(w, fmt.Sprintf(`{"error":{"message":"failed to encode response: %v"}}`, err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(outBytes)
			return
		}
	}

	// All candidate providers failed before commit
	renderFailedBeforeCommit(w, lastErr, "all candidate providers failed before commit")
}

func renderFailedBeforeCommit(w http.ResponseWriter, lastErr error, fallbackMsg string) {
	statusCode := http.StatusBadGateway
	errMsg := fallbackMsg
	errType := "bad_gateway"
	var uerr *UnknownModelError
	if errors.As(lastErr, &uerr) {
		statusCode = http.StatusNotFound
		errType = "unknown_model"
		errMsg = uerr.Error()
	} else if lastErr != nil {
		errMsg = lastErr.Error()
		msgLower := strings.ToLower(errMsg)
		if strings.Contains(msgLower, "status 429") || strings.Contains(msgLower, "http 429") || strings.Contains(msgLower, "429 too many") || strings.Contains(msgLower, "resource_exhausted") || strings.Contains(msgLower, "quota_exceeded") || strings.Contains(msgLower, "daily token limit") || strings.Contains(msgLower, "rate limit") {
			statusCode = http.StatusTooManyRequests
			errType = "rate_limit_exceeded"
		} else if strings.Contains(msgLower, "status 401") || strings.Contains(msgLower, "http 401") || strings.Contains(msgLower, "unauthorized") {
			statusCode = http.StatusUnauthorized
			errType = "authentication_error"
		} else if strings.Contains(msgLower, "status 403") || strings.Contains(msgLower, "http 403") || strings.Contains(msgLower, "forbidden") {
			statusCode = http.StatusForbidden
			errType = "permission_denied"
		} else if strings.Contains(msgLower, "status 404") || strings.Contains(msgLower, "http 404") || strings.Contains(msgLower, "not found") {
			statusCode = http.StatusNotFound
			errType = "not_found"
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": errMsg,
			"type":    errType,
		},
	})
}
