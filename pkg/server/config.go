package server

import (
	"bytes"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config holds Ultiproxy server and storage configuration.
type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Storage StorageConfig `yaml:"storage"`
	DataDir string        `yaml:"data_dir"`
}

// ServerConfig contains HTTP listening and authentication settings.
type ServerConfig struct {
	Addr        string            `yaml:"addr"`
	APIKey      string            `yaml:"api_key"`
	ClientKeys  map[string]string `yaml:"client_keys"`
	LLMsTxtPath string            `yaml:"llms_txt_path"`
	// Models maps client-visible aliases to provider lanes + upstream ids.
	Models map[string]ModelAlias `yaml:"models"`
	// Timeouts maps provider lanes to request-timeout durations, e.g. {"vllm":"10m"}.
	Timeouts map[string]string `yaml:"timeouts"`
}

// StorageConfig contains SQLite database configuration.
type StorageConfig struct {
	DBPath string `yaml:"db_path"`
}

// DefaultConfig returns reasonable defaults. ULTIPROXY_ADDR overrides the
// listen address so a fresh zero-config install can bind 0.0.0.0 for remote
// agents without writing a config file.
func DefaultConfig() *Config {
	addr := os.Getenv("ULTIPROXY_ADDR")
	if addr == "" {
		addr = "127.0.0.1:9050"
	}
	cfg := &Config{
		Server: ServerConfig{
			Addr:        addr,
			ClientKeys:  make(map[string]string),
			LLMsTxtPath: "llms.txt",
			Models:      make(map[string]ModelAlias),
		},
		Storage: StorageConfig{
			DBPath: "ultiproxy.db",
		},
		DataDir: ".",
	}
	return cfg
}

// LoadConfig loads configuration from a YAML file. If path is empty, DefaultConfig is returned.
// Unknown YAML keys are rejected, so a config written for an older schema
// fails loudly instead of partially applying.
func LoadConfig(path string) (*Config, error) {
	cfg := DefaultConfig()
	if path == "" {
		return cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	// KnownFields(true) makes configuration fail closed: a stale key from an
	// old template (server.listen, server.api_keys, routing:, ...) aborts
	// startup with a parse error instead of being silently dropped, which
	// used to leave authentication off while the operator believed it was on.
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(cfg); err != nil {
		return nil, fmt.Errorf("parse config file: %w", err)
	}

	// Default listen address, overridable via ULTIPROXY_ADDR so a fresh
	// zero-config install can still bind 0.0.0.0 for remote agents without
	// writing a config file.
	if cfg.Server.Addr == "" {
		cfg.Server.Addr = os.Getenv("ULTIPROXY_ADDR")
	}
	if cfg.Server.Addr == "" {
		cfg.Server.Addr = "127.0.0.1:9050"
	}
	if cfg.Storage.DBPath == "" {
		cfg.Storage.DBPath = "ultiproxy.db"
	}
	if cfg.Server.ClientKeys == nil {
		cfg.Server.ClientKeys = make(map[string]string)
	}

	return cfg, nil
}
