// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/jang"
)

// InferJANG classifies a model's JANG/JANGTQ quantisation profile from its
// Hugging Face metadata (id, tags, filenames) when the metadata itself
// carries no explicit JANG block. Returns nil for ordinary (non-JANG)
// weights.
//
//	info := hf.InferJANG(meta)
func InferJANG(meta ModelMetadata) *jang.Info {
	// Fast-path classify before any heap work. inferJANGNeedlePresent
	// scans the id / tags / filenames in-place for "jang" and "jangtq"
	// tokens. The miss path (the dominant case across HF metadata)
	// returns jangNone in zero allocs. The JANGTQ branch needs only the
	// QuantizationConfig group size — no haystack scan — so we skip the
	// lowercase-buffer build entirely for those packs.
	id := firstNonEmpty(meta.ID, meta.ModelID)
	presence := inferJANGNeedlePresent(id, meta.Tags, meta.Files)
	switch presence {
	case jangNone:
		return nil
	case jangTQ:
		info := &jang.Info{
			Profile:          "JANGTQ",
			WeightFormat:     "mxtq",
			Method:           "affine+mxtq",
			GroupSize:        jangGroupSize(meta),
			BitsDefault:      2,
			RoutedExpertBits: 2,
		}
		info.Packed = jang.BuildPackedProfile(info)
		return info
	}
	// jangBasic — need to scan the haystack for a specific profile name
	// (jang_1l, jang_2s, etc.). Build the lowercase "id tag1 tag2
	// file1 file2" haystack in one pass; the buffer is the only
	// allocation specific to this branch.
	size := len(id)
	for _, tag := range meta.Tags {
		size += 1 + len(tag)
	}
	for _, file := range meta.Files {
		// Upper bound — max(Name, RFilename). Avoids the firstNonEmpty
		// scan here while still preventing growslice in the append loop.
		nameLen := max(len(file.RFilename), len(file.Name))
		size += 1 + nameLen
	}
	buf := make([]byte, 0, size)
	buf = appendLowerASCII(buf, id)
	for _, tag := range meta.Tags {
		buf = append(buf, ' ')
		buf = appendLowerASCII(buf, tag)
	}
	for _, file := range meta.Files {
		buf = append(buf, ' ')
		buf = appendLowerASCII(buf, file.filename())
	}
	needle := core.AsString(buf)
	profile := inferJANGProfileName(needle)
	info := &jang.Info{
		Profile:     profile,
		GroupSize:   jangGroupSize(meta),
		BitsDefault: core.FirstPositive(jang.ProfileBits(profile), 0),
	}
	info.Packed = jang.BuildPackedProfile(info)
	return info
}

// JANG token-presence states. Returned by inferJANGNeedlePresent so
// InferJANG can skip the lowercase-haystack build for the JANGTQ branch
// (which doesn't need a haystack scan past detection).
type jangPresence uint8

const (
	jangNone  jangPresence = 0
	jangBasic jangPresence = 1 // "jang" present, "jangtq" not
	jangTQ    jangPresence = 2 // "jangtq" present (implies "jang")
)

// inferJANGNeedlePresent classifies the strongest JANG token present in
// the id / tags / filenames in a single pass per component. Pure scan,
// no allocations — used to gate the lowercase-buffer build inside
// InferJANG. jangNone (the dominant case across HF metadata) returns in
// zero allocs after a tight byte scan. jangTQ short-circuits the
// haystack build downstream because the JANGTQ branch only needs the
// QuantizationConfig group size, not a needle scan.
func inferJANGNeedlePresent(id string, tags []string, files []ModelFile) jangPresence {
	state := scanJANGFold(id)
	if state == jangTQ {
		return jangTQ
	}
	for _, tag := range tags {
		s := scanJANGFold(tag)
		if s == jangTQ {
			return jangTQ
		}
		if s > state {
			state = s
		}
	}
	for _, file := range files {
		s := scanJANGFold(file.Name)
		if s == jangTQ {
			return jangTQ
		}
		if s > state {
			state = s
		}
		s = scanJANGFold(file.RFilename)
		if s == jangTQ {
			return jangTQ
		}
		if s > state {
			state = s
		}
	}
	return state
}

// scanJANGFold reports the strongest JANG token present in s — jangTQ
// when "jangtq" is found, jangBasic when only "jang" is found, jangNone
// otherwise. Single ASCII byte scan with case folding inline. Per
// starting position 'j', try the longer 6-byte "jangtq" match first;
// fall back to 4-byte "jang". Returns early on jangTQ.
func scanJANGFold(s string) jangPresence {
	if len(s) < 4 {
		return jangNone
	}
	state := jangNone
	last4 := len(s) - 4
	for i := 0; i <= last4; i++ {
		c0 := s[i]
		if c0 >= 'A' && c0 <= 'Z' {
			c0 += 'a' - 'A'
		}
		if c0 != 'j' {
			continue
		}
		c1 := s[i+1]
		if c1 >= 'A' && c1 <= 'Z' {
			c1 += 'a' - 'A'
		}
		if c1 != 'a' {
			continue
		}
		c2 := s[i+2]
		if c2 >= 'A' && c2 <= 'Z' {
			c2 += 'a' - 'A'
		}
		if c2 != 'n' {
			continue
		}
		c3 := s[i+3]
		if c3 >= 'A' && c3 <= 'Z' {
			c3 += 'a' - 'A'
		}
		if c3 != 'g' {
			continue
		}
		// "jang" matched at i. Probe for the "tq" extension if there's
		// room — jangtq is the strongest match.
		if i+6 <= len(s) {
			c4 := s[i+4]
			if c4 >= 'A' && c4 <= 'Z' {
				c4 += 'a' - 'A'
			}
			if c4 == 't' {
				c5 := s[i+5]
				if c5 >= 'A' && c5 <= 'Z' {
					c5 += 'a' - 'A'
				}
				if c5 == 'q' {
					return jangTQ
				}
			}
		}
		state = jangBasic
	}
	return state
}

// appendLowerASCII appends s to dst with ASCII A-Z mapped to a-z. Non-ASCII
// bytes pass through unchanged (consistent with the domain: model IDs, tags,
// filenames are all ASCII).
func appendLowerASCII(dst []byte, s string) []byte {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst = append(dst, c)
	}
	return dst
}

func jangGroupSize(meta ModelMetadata) int {
	if quant := meta.Config.QuantizationConfig; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	if quant := meta.Config.Quantization; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	return 64
}

// jangProfileLookup parallels needle/value forms with their UPPER variants.
// Hoisted out of inferJANGProfileName so the literal slice and the
// per-match core.Upper allocation are paid once at init, not per call.
var jangProfileLookup = [...]struct{ Lower, Upper string }{
	{"jang_1l", "JANG_1L"},
	{"jang_2s", "JANG_2S"},
	{"jang_2l", "JANG_2L"},
	{"jang_3l", "JANG_3L"},
	{"jang_4k", "JANG_4K"},
	{"jang_4m", "JANG_4M"},
}

func inferJANGProfileName(value string) string {
	for i := range jangProfileLookup {
		if core.Contains(value, jangProfileLookup[i].Lower) {
			return jangProfileLookup[i].Upper
		}
	}
	return "JANG"
}
