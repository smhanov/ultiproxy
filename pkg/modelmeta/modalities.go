package modelmeta

import (
	"fmt"
	"strings"
)

// normalizeToken maps one upstream modality spelling onto the canonical token
// set. ok is false for anything unknown: an unknown modality is dropped rather
// than invented as something else.
func normalizeToken(in string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(in)) {
	case "text", "txt":
		return ModalityText, true
	case "image", "img", "images":
		return ModalityImage, true
	case "pdf":
		// pdf is a file modality: normalize, do not keep a separate token.
		return ModalityFile, true
	case "file", "files":
		return ModalityFile, true
	case "audio":
		return ModalityAudio, true
	case "video":
		return ModalityVideo, true
	default:
		return "", false
	}
}

// NormalizeModalities canonicalizes a modality array: tokens are lowercased
// and trimmed, pdf becomes file, unknown tokens are dropped, duplicates
// disappear, order is preserved. nil comes back as nil so "unknown" stays
// distinguishable from "known empty". Use ValidateModalities to reject an
// operator-supplied array instead of silently trimming it.
func NormalizeModalities(in []string) []string {
	if in == nil {
		return nil
	}
	var out []string
	seen := make(map[string]bool, len(in))
	for _, raw := range in {
		tok, ok := normalizeToken(raw)
		if !ok || seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	return out
}

// ValidateModalities reports the first token that does not normalize, so an
// operator-supplied array is rejected with the offending value instead of
// being silently trimmed.
func ValidateModalities(in []string) error {
	for _, raw := range in {
		if _, ok := normalizeToken(raw); !ok {
			return fmt.Errorf("unknown modality %q (use text, image, file, audio or video)", raw)
		}
	}
	return nil
}

// HasImage reports whether image is one of the advertised input modalities.
// supports_vision is derived from exactly this, and is never inferred from a
// model name or a lane capability.
func HasImage(in []string) bool {
	for _, mod := range in {
		if strings.EqualFold(strings.TrimSpace(mod), ModalityImage) {
			return true
		}
	}
	return false
}

// ParseModalityString splits an HF-style "modality" string such as
// "text+image->text" or "text+image+file->text" into input and output
// modality arrays. A string with no "->" is treated as input-only.
func ParseModalityString(s string) (in, out []string) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	input, output := s, ""
	if idx := strings.Index(s, "->"); idx >= 0 {
		input, output = s[:idx], s[idx+2:]
	}
	in = NormalizeModalities(strings.Split(input, "+"))
	out = NormalizeModalities(strings.Split(output, "+"))
	return in, out
}
