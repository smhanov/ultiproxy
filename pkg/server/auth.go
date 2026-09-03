package server

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
)

type contextKey string

const (
	// ClientKeyHashKey is the context key for the sha256 hex digest of the authenticated key.
	ClientKeyHashKey contextKey = "client_key_hash"
	// ClientNameKey is the context key for the identified client name or 'admin'.
	ClientNameKey contextKey = "client_name"
)

// ClientKeyHashFromContext retrieves the client key hash from context.
func ClientKeyHashFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(ClientKeyHashKey).(string); ok {
		return v
	}
	return ""
}

// ClientNameFromContext retrieves the client name from context.
func ClientNameFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(ClientNameKey).(string); ok {
		return v
	}
	return ""
}

// AuthMiddleware enforces virtual client key authentication.
type AuthMiddleware struct {
	hasKeys         bool
	apiKeyHash      [32]byte
	hasAPIKey       bool
	clientKeyHashes map[string][32]byte // clientName -> sha256 hash
}

// NewAuthMiddleware constructs the auth middleware with precomputed SHA-256 hashes.
func NewAuthMiddleware(apiKey string, clientKeys map[string]string) *AuthMiddleware {
	m := &AuthMiddleware{
		clientKeyHashes: make(map[string][32]byte),
	}

	if apiKey != "" {
		m.apiKeyHash = sha256.Sum256([]byte(apiKey))
		m.hasAPIKey = true
		m.hasKeys = true
	}

	for name, key := range clientKeys {
		if key != "" {
			m.clientKeyHashes[name] = sha256.Sum256([]byte(key))
			m.hasKeys = true
		}
	}

	return m
}

// Wrap wraps an http.Handler with authentication.
func (m *AuthMiddleware) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Public paths that do not require auth
		if r.URL.Path == "/healthz" || r.URL.Path == "/llms.txt" {
			next.ServeHTTP(w, r)
			return
		}

		// If no keys configured, allow through
		if !m.hasKeys {
			next.ServeHTTP(w, r)
			return
		}

		authHeader := r.Header.Get("Authorization")
		if !strings.HasPrefix(authHeader, "Bearer ") {
			writeAuthError(w)
			return
		}

		presentedKey := strings.TrimPrefix(authHeader, "Bearer ")
		presentedHash := sha256.Sum256([]byte(presentedKey))

		matched := false
		clientName := ""

		// Check against static admin key
		if m.hasAPIKey && subtle.ConstantTimeCompare(presentedHash[:], m.apiKeyHash[:]) == 1 {
			matched = true
			clientName = "admin"
		}

		// Check against client keys
		if !matched {
			for name, hash := range m.clientKeyHashes {
				if subtle.ConstantTimeCompare(presentedHash[:], hash[:]) == 1 {
					matched = true
					clientName = name
					break
				}
			}
		}

		if !matched {
			writeAuthError(w)
			return
		}

		hexHash := hex.EncodeToString(presentedHash[:])
		ctx := context.WithValue(r.Context(), ClientKeyHashKey, hexHash)
		ctx = context.WithValue(ctx, ClientNameKey, clientName)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func writeAuthError(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": "Incorrect API key provided",
			"type":    "invalid_request_error",
			"param":   nil,
			"code":    "invalid_api_key",
		},
	})
}
