package sse

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// Event represents a parsed Server-Sent Event.
type Event struct {
	Type string
	Data []byte
	ID   string
}

// Scanner scans an SSE stream from an io.Reader.
type Scanner struct {
	scanner *bufio.Scanner
	event   Event
	err     error
}

// NewScanner returns a new SSE Scanner.
func NewScanner(r io.Reader) *Scanner {
	s := bufio.NewScanner(r)
	// Allow larger SSE lines/chunks (up to 10MB)
	buf := make([]byte, 64*1024)
	s.Buffer(buf, 10*1024*1024)
	return &Scanner{scanner: s}
}

// Scan reads until the next complete SSE event is found.
func (s *Scanner) Scan() bool {
	var currentType string
	var currentID string
	var dataLines [][]byte
	hasData := false

	for s.scanner.Scan() {
		line := s.scanner.Bytes()

		// Empty line dispatches the event if we have collected any data or type
		if len(line) == 0 {
			if hasData || currentType != "" || currentID != "" {
				s.event = Event{
					Type: currentType,
					ID:   currentID,
					Data: bytes.Join(dataLines, []byte("\n")),
				}
				return true
			}
			continue
		}

		// Comment line
		if line[0] == ':' {
			continue
		}

		// Parse field
		colonIdx := bytes.IndexByte(line, ':')
		var field string
		var val []byte
		if colonIdx == -1 {
			field = string(line)
			val = nil
		} else {
			field = string(line[:colonIdx])
			val = line[colonIdx+1:]
			if len(val) > 0 && val[0] == ' ' {
				val = val[1:]
			}
		}

		switch field {
		case "event":
			currentType = strings.TrimSpace(string(val))
		case "id":
			currentID = strings.TrimSpace(string(val))
		case "data":
			hasData = true
			lineCopy := make([]byte, len(val))
			copy(lineCopy, val)
			dataLines = append(dataLines, lineCopy)
		}
	}

	if err := s.scanner.Err(); err != nil {
		s.err = err
		return false
	}

	// Final event if stream ended without trailing newline
	if hasData || currentType != "" || currentID != "" {
		s.event = Event{
			Type: currentType,
			ID:   currentID,
			Data: bytes.Join(dataLines, []byte("\n")),
		}
		return true
	}

	return false
}

// Event returns the most recent event read by Scan.
func (s *Scanner) Event() Event {
	return s.event
}

// Err returns the first non-EOF error that was encountered.
func (s *Scanner) Err() error {
	return s.err
}
