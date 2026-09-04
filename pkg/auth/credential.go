package auth

import (
	"time"
)

// Credential represents stored authentication and token information.
type Credential struct {
	Tenant       string    `json:"tenant"`
	Provider     string    `json:"provider"`
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresAt    time.Time `json:"expires_at"`
	Generation   uint64    `json:"generation"`
	ClientID     string    `json:"client_id"`
	ProjectID    string    `json:"project_id,omitempty"`
}
