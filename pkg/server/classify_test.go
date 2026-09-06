package server

import (
	"errors"
	"net/http"
	"testing"
)

func TestClassifyUpstreamError_Structured400(t *testing.T) {
	status, typ, msg := classifyUpstreamError(statusError{code: 400, msg: `{"code":"1210","message":"Invalid API parameter"}`}, "fallback")
	if status != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", status)
	}
	if typ != "bad_request" {
		t.Fatalf("type = %q, want bad_request", typ)
	}
	if msg != `{"code":"1210","message":"Invalid API parameter"}` {
		t.Fatalf("message lost: %q", msg)
	}
}

func TestClassifyUpstreamError_Structured422(t *testing.T) {
	status, typ, _ := classifyUpstreamError(statusError{code: 422, msg: "unprocessable"}, "fallback")
	if status != http.StatusBadRequest || typ != "bad_request" {
		t.Fatalf("422 -> %d %s, want 400 bad_request", status, typ)
	}
}

func TestClassifyUpstreamError_StringSniff400(t *testing.T) {
	status, typ, _ := classifyUpstreamError(errors.New("openai: http 400: invalid"), "fallback")
	if status != http.StatusBadRequest || typ != "bad_request" {
		t.Fatalf("sniffed 400 -> %d %s", status, typ)
	}
}

func TestClassifyUpstreamError_UnchangedClasses(t *testing.T) {
	cases := []struct {
		err    error
		status int
		typ    string
	}{
		{&UnknownModelError{Model: "nope"}, http.StatusNotFound, "unknown_model"},
		{errors.New("http 401 unauthorized"), http.StatusUnauthorized, "authentication_error"},
		{errors.New("http 403 forbidden"), http.StatusForbidden, "permission_denied"},
		{errors.New("http 404 not found"), http.StatusNotFound, "not_found"},
		{errors.New("http 429 too many"), http.StatusTooManyRequests, "rate_limit_exceeded"},
		{errors.New("connection refused"), http.StatusBadGateway, "bad_gateway"},
	}
	for _, tc := range cases {
		status, typ, _ := classifyUpstreamError(tc.err, "fallback")
		if status != tc.status || typ != tc.typ {
			t.Errorf("%v -> %d %s, want %d %s", tc.err, status, typ, tc.status, tc.typ)
		}
	}
}
