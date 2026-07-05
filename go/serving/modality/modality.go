// SPDX-Licence-Identifier: EUPL-1.2

// Package modality models the output modalities (RFC.md §6.12) as pure-Go
// types: the set of output kinds a caller requests (text, image, audio), the
// assistant content parts a backend produces, and the per-kind accounting that
// feeds usage reporting (§6.6).
//
// It is a serving-surface type package, not a model-maths library — it carries
// the shapes the router and host surfaces pass around, with no I/O and no
// dependency beyond the core framework, so it stays trivially unit-testable.
//
//	out, err := modality.Requested(req.Modalities)   // normalise the request
//	msg := modality.Assemble(parts)                  // collect backend parts
//	usage := modality.Counts(msg.Parts)              // tally for §6.6
package modality

import core "dappco.re/go"

// Modality is one selectable output type from the request's modalities field
// (§6.12). The wire form is the lower-case string; use ParseModality to read
// caller input and String to emit it.
type Modality string

const (
	// Text is the always-available output modality; implied when a request
	// names none.
	Text Modality = "text"
	// Image is image output (the image-generation server tool, §6.4), carried
	// back as image content parts.
	Image Modality = "image"
	// Audio is audio output where a backend supports it (§6.12), carried back
	// as audio content parts.
	Audio Modality = "audio"
)

// String returns the canonical lower-case wire form.
//
//	modality.Image.String() == "image"
func (m Modality) String() string { return string(m) }

// Valid reports whether m is one of the known modalities. The zero value is not
// valid.
//
//	modality.Audio.Valid() == true
//	modality.Modality("video").Valid() == false
func (m Modality) Valid() bool {
	switch m {
	case Text, Image, Audio:
		return true
	default:
		return false
	}
}

// ParseModality reads a wire string into a Modality, tolerant of surrounding
// whitespace and case (callers pass raw request values). Unknown values error.
//
//	m, err := modality.ParseModality("AUDIO") // -> Audio, nil
//	_, err := modality.ParseModality("video") // -> error
func ParseModality(s string) (Modality, error) {
	m := Modality(core.Lower(core.Trim(s)))
	if !m.Valid() {
		return "", core.E("modality", "unknown modality: "+s, nil)
	}
	return m, nil
}

// Requested validates and normalises a requested output set (§6.12): each entry
// is parsed (whitespace/case tolerant), duplicates collapse keeping first-seen
// order, and an empty or nil request implies a single Text modality — the
// output set is never empty. An unknown modality is rejected rather than
// silently dropped, so a typo surfaces instead of quietly changing the output.
//
//	out, err := modality.Requested([]modality.Modality{modality.Image, modality.Image})
//	// out == [image], err == nil
//	out, err := modality.Requested(nil)
//	// out == [text], err == nil
func Requested(in []Modality) ([]Modality, error) {
	out := make([]Modality, 0, len(in))
	seen := make(map[Modality]bool, len(in))
	for _, raw := range in {
		m, err := ParseModality(raw.String())
		if err != nil {
			return nil, err
		}
		if seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	if len(out) == 0 {
		return []Modality{Text}, nil
	}
	return out, nil
}
