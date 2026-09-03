package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

// ExecuteGenerate executes a non-streaming chat completion request and converts the response to IR.
func ExecuteGenerate(ctx context.Context, httpClient *http.Client, req *http.Request) (*ir.Response, error) {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpClient.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("upstream error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return ConvertResponse(&chatResp), nil
}

// BuildRequestBody serializes the ChatCompletionRequest to an io.Reader.
func BuildRequestBody(reqBody ChatCompletionRequest) (io.Reader, error) {
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}
