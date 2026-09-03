package sse

import (
	"strings"
	"testing"
)

func TestScanner(t *testing.T) {
	input := ":comment\n\nevent: message_start\ndata: {\"id\":\"123\"}\n\nevent: text_delta\ndata: Hello \ndata: World\n\n"
	scanner := NewScanner(strings.NewReader(input))

	if !scanner.Scan() {
		t.Fatalf("expected first event, err: %v", scanner.Err())
	}
	ev1 := scanner.Event()
	if ev1.Type != "message_start" || string(ev1.Data) != `{"id":"123"}` {
		t.Fatalf("unexpected ev1: %+v", ev1)
	}

	if !scanner.Scan() {
		t.Fatalf("expected second event, err: %v", scanner.Err())
	}
	ev2 := scanner.Event()
	if ev2.Type != "text_delta" || string(ev2.Data) != "Hello \nWorld" {
		t.Fatalf("unexpected ev2: %+v", ev2)
	}

	if scanner.Scan() {
		t.Fatalf("did not expect third event")
	}
}
