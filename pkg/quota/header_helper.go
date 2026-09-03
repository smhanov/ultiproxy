package quota

import (
	"net/http"
	"strings"
)

func getHeader(h http.Header, key string) string {
	if h == nil {
		return ""
	}
	if val := h.Get(key); val != "" {
		return val
	}
	keyLower := strings.ToLower(key)
	for k, v := range h {
		if strings.ToLower(k) == keyLower && len(v) > 0 {
			return v[0]
		}
	}
	return ""
}
