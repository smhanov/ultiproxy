package server

import (
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
}

// StorageConfig contains SQLite database configuration.
type StorageConfig struct {
	DBPath string `yaml:"db_path"`
}

// DefaultConfig returns reasonable defaults.
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Addr:        "127.0.0.1:8317",
			ClientKeys:  make(map[string]string),
			LLMsTxtPath: "llms.txt",
		},
		Storage: StorageConfig{
			DBPath: "ultiproxy.db",
		},
		DataDir: ".",
	}
}

// LoadConfig loads configuration from a YAML file. If path is empty, DefaultConfig is returned.
func LoadConfig(path string) (*Config, error) {
	cfg := DefaultConfig()
	if path == "" {
		return cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config file: %w", err)
	}

	if cfg.Server.Addr == "" {
		cfg.Server.Addr = "127.0.0.1:8317"
	}
	if cfg.Storage.DBPath == "" {
		cfg.Storage.DBPath = "ultiproxy.db"
	}
	if cfg.Server.ClientKeys == nil {
		cfg.Server.ClientKeys = make(map[string]string)
	}

	return cfg, nil
}
