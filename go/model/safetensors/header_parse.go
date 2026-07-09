// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	core "dappco.re/go"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. The hand-rolled JSON parser fires these from a tight
// byte-walk; sharing instances also makes errors.Is comparable for
// callers wanting to distinguish "header truncated" from "missing
// colon" without parsing message text.
var (
	errUnterminatedString      = core.NewError("mlx: safetensors unterminated string")
	errUnknownLiteral          = core.NewError("mlx: safetensors unknown literal")
	errSkipValueToken          = core.NewError("mlx: safetensors unexpected token in skipValue")
	errTruncatedEscape         = core.NewError("mlx: safetensors truncated escape")
	errTensorExpectCommaBrace  = core.NewError("mlx: safetensors tensor expected ',' or '}'")
	errHeaderTruncated         = core.NewError("mlx: safetensors header truncated")
	errHeaderMissingColon      = core.NewError("mlx: safetensors header missing ':' after key")
	errHeaderKeyNotString      = core.NewError("mlx: safetensors header key is not a string")
	errHeaderNotJSONObject     = core.NewError("mlx: safetensors header is not a JSON object")
	errHeaderExpectCommaBrace  = core.NewError("mlx: safetensors header expected ',' or '}'")
	errExpectString            = core.NewError("mlx: safetensors expected string")
	errExpectBrace             = core.NewError("mlx: safetensors expected '{'")
	errExpectBracket           = core.NewError("mlx: safetensors expected '['")
	errExpectColon             = core.NewError("mlx: safetensors expected ':' inside object")
	errExpectCommaBraceObject  = core.NewError("mlx: safetensors expected ',' or '}' inside object")
	errExpectCommaBracketArray = core.NewError("mlx: safetensors expected ',' or ']' inside array")
)

// parseHeaderInto walks a safetensors JSON header bytes blob and emits
// one TensorRef per non-metadata tensor into idx. Every Shape slice is
// carved out of shapeSlab (pre-sized by the caller via a first-pass
// scan).
//
// The implementation hand-rolls a JSON walker for the well-known
// safetensors header shape:
//
//	{"tensor_name":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},
//	 ...,
//	 "__metadata__":{"format":"pt", ...}  // optional, body skipped
//	}
//
// Bypassing encoding/json removes the ~6 allocs per tensor that
// reflection-driven Unmarshal incurred (HeaderEntry struct, Shape slice,
// DataOffsets slice, key string, decodeState/literalStore overhead) —
// see Wave 8 W8-I profile. Tensor names are still allocated (they're
// load-bearing for the Index.Tensors map and Names slice); everything
// else is parsed into scalars or carved from the shared slab.
func parseHeaderInto(path string, data []byte, dataStart int64, idx *Index, shapeSlab *[]uint64) error {
	// Wrap the freshly-read headerBytes as an immutable string view
	// (no copy). Tensor names are returned as substring views into
	// this arena — one alloc for the entire header turns into N name
	// strings that share underlying memory. Per the AsString contract
	// the caller (ReadIndex) must not retain or mutate the source
	// []byte after this call, which it does not.
	arena := core.AsString(data)
	p := jsonParser{data: data}
	p.skipWS()
	if !p.expect('{') {
		return errHeaderNotJSONObject
	}
	p.skipWS()
	if p.peek() == '}' {
		p.pos++
		return nil
	}
	for {
		p.skipWS()
		// Peek at the raw byte span of the tensor name. For tensor
		// names (common case — no escapes) this is alloc-free; the
		// string conversion happens once at the end, downstream of
		// the __metadata__ check so the metadata key path costs zero
		// allocs.
		start, end, hasEsc, ok := p.peekStringSpan()
		if !ok {
			return errHeaderKeyNotString
		}
		isMetadata := !hasEsc && end-start == 12 && core.SliceEqual(data[start:end], _metadataKey)
		p.skipWS()
		if !p.expect(':') {
			return errHeaderMissingColon
		}
		p.skipWS()
		if isMetadata {
			if err := p.skipValue(); err != nil {
				return err
			}
		} else {
			name := nameFromSpan(arena, data, start, end, hasEsc)
			if _, dup := idx.Tensors[name]; dup {
				return core.NewError("mlx: duplicate tensor in safetensors header: " + name)
			}
			ref, err := p.parseTensorEntry(path, name, dataStart, shapeSlab)
			if err != nil {
				return err
			}
			idx.Tensors[name] = ref
			idx.Names = append(idx.Names, name)
		}
		p.skipWS()
		switch p.peek() {
		case ',':
			p.pos++
		case '}':
			p.pos++
			return nil
		default:
			return errHeaderExpectCommaBrace
		}
	}
}

// nameFromSpan returns a string view of a tensor name. For the common
// case (no escape sequences in the name — true for every real-world
// safetensors file) it is a zero-alloc substring slice of the arena.
// Escaped names fall through to the slow path which allocates a fresh
// string. Real safetensors writers never emit JSON escapes in tensor
// names, so this path is effectively never hit on production headers.
func nameFromSpan(arena string, data []byte, start, end int, hasEsc bool) string {
	if !hasEsc {
		return arena[start:end]
	}
	return materialiseString(data, start, end, hasEsc)
}

// _metadataKey is the literal bytes "__metadata__" — pre-stored to
// avoid an allocation on the bytes comparison in the hot loop.
var _metadataKey = []byte("__metadata__")

// materialiseString converts a previously-peeked string span into a
// string. The common case (no backslash escapes) is a single
// `string()` conversion. Escaped strings re-parse via the slow path.
func materialiseString(data []byte, start, end int, hasEsc bool) string {
	if !hasEsc {
		return string(data[start:end])
	}
	p := jsonParser{data: data, pos: start}
	s, _ := p.parseStringEscaped(start)
	return s
}

// jsonParser is a focused walker for the safetensors header. It is not
// a general-purpose JSON parser — it only supports the constructs that
// appear in real safetensors headers (objects, arrays, strings with
// standard escapes, integers, booleans, null).
type jsonParser struct {
	data []byte
	pos  int
}

func (p *jsonParser) peek() byte {
	if p.pos >= len(p.data) {
		return 0
	}
	return p.data[p.pos]
}

func (p *jsonParser) expect(c byte) bool {
	if p.pos >= len(p.data) || p.data[p.pos] != c {
		return false
	}
	p.pos++
	return true
}

func (p *jsonParser) skipWS() {
	for p.pos < len(p.data) {
		c := p.data[p.pos]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return
		}
		p.pos++
	}
}

// peekStringSpan reads the bounds of a JSON string without allocating.
// It returns (start, end, hasEsc, ok) where start..end is the byte
// range between the opening and closing quotes. hasEsc is true if any
// backslash escapes were encountered — the caller must use
// materialiseString to convert to a string in that case. p.pos is
// advanced past the closing quote.
func (p *jsonParser) peekStringSpan() (int, int, bool, bool) {
	if p.pos >= len(p.data) || p.data[p.pos] != '"' {
		return 0, 0, false, false
	}
	start := p.pos + 1
	i := start
	hasEsc := false
	for i < len(p.data) {
		c := p.data[i]
		if c == '"' {
			p.pos = i + 1
			return start, i, hasEsc, true
		}
		if c == '\\' {
			hasEsc = true
			// Skip the escape — \uXXXX is 6 bytes, others 2.
			if i+1 >= len(p.data) {
				return 0, 0, false, false
			}
			if p.data[i+1] == 'u' {
				i += 6
			} else {
				i += 2
			}
			continue
		}
		i++
	}
	return 0, 0, false, false
}

// parseStringEscaped is the slow path for strings with escape
// sequences. Allocates a fresh byte buffer; only used when a backslash
// is seen (rare in tensor names, possible in __metadata__ values
// although those are skipped wholesale).
func (p *jsonParser) parseStringEscaped(start int) (string, bool) {
	// Pre-size to the remaining-up-to-closing-quote span; safetensors
	// headers are small so over-alloc is bounded.
	buf := make([]byte, 0, len(p.data)-start)
	// Re-copy the verified-clean prefix.
	for i := start; i < p.pos; i++ {
		// shouldn't happen — parseString switches to this path before
		// advancing past the first backslash — but be safe.
		buf = append(buf, p.data[i])
	}
	i := p.pos
	for i < len(p.data) {
		c := p.data[i]
		if c == '"' {
			p.pos = i + 1
			return string(buf), true
		}
		if c == '\\' {
			if i+1 >= len(p.data) {
				return "", false
			}
			esc := p.data[i+1]
			switch esc {
			case '"', '\\', '/':
				buf = append(buf, esc)
				i += 2
			case 'b':
				buf = append(buf, '\b')
				i += 2
			case 'f':
				buf = append(buf, '\f')
				i += 2
			case 'n':
				buf = append(buf, '\n')
				i += 2
			case 'r':
				buf = append(buf, '\r')
				i += 2
			case 't':
				buf = append(buf, '\t')
				i += 2
			case 'u':
				// \uXXXX — decode 4 hex digits to a rune.
				if i+6 > len(p.data) {
					return "", false
				}
				r := uint32(0)
				for j := range 4 {
					h := p.data[i+2+j]
					var v uint32
					switch {
					case h >= '0' && h <= '9':
						v = uint32(h - '0')
					case h >= 'a' && h <= 'f':
						v = uint32(h-'a') + 10
					case h >= 'A' && h <= 'F':
						v = uint32(h-'A') + 10
					default:
						return "", false
					}
					r = r<<4 | v
				}
				// Encode as UTF-8.
				switch {
				case r < 0x80:
					buf = append(buf, byte(r))
				case r < 0x800:
					buf = append(buf, byte(0xc0|(r>>6)), byte(0x80|(r&0x3f)))
				default:
					buf = append(buf, byte(0xe0|(r>>12)), byte(0x80|((r>>6)&0x3f)), byte(0x80|(r&0x3f)))
				}
				i += 6
			default:
				return "", false
			}
		} else {
			buf = append(buf, c)
			i++
		}
	}
	return "", false
}

// parseInt64 reads a signed integer literal. Safetensors offsets and
// shapes are always plain integers — no scientific notation, no
// decimals. The parser accepts an optional minus sign for robustness.
func (p *jsonParser) parseInt64() (int64, bool) {
	if p.pos >= len(p.data) {
		return 0, false
	}
	neg := false
	if p.data[p.pos] == '-' {
		neg = true
		p.pos++
	}
	if p.pos >= len(p.data) || p.data[p.pos] < '0' || p.data[p.pos] > '9' {
		return 0, false
	}
	var v int64
	for p.pos < len(p.data) {
		c := p.data[p.pos]
		if c < '0' || c > '9' {
			break
		}
		v = v*10 + int64(c-'0')
		p.pos++
	}
	if neg {
		v = -v
	}
	return v, true
}

// parseTensorEntry reads one safetensors tensor entry body — the inner
// object with keys dtype/shape/data_offsets — and emits a TensorRef.
// Inner-key order is not fixed; entries from real models hit shape
// permutations from python's json.dumps default + the rust safetensors
// crate. We tolerate any of the six orderings without re-allocating.
//
// Inner keys are matched against canonical bytes without ever being
// converted to strings — this is the 3-allocs-per-tensor win that
// dropped IndexFiles_TwoShards below 200 allocs.
func (p *jsonParser) parseTensorEntry(path, name string, dataStart int64, shapeSlab *[]uint64) (TensorRef, error) {
	if !p.expect('{') {
		return TensorRef{}, core.NewError("mlx: safetensors tensor entry is not an object: " + name)
	}
	var (
		dtype       string
		shapeStart  int
		shapeLen    int
		offsetBegin int64
		offsetEnd   int64
		haveDtype   bool
		haveShape   bool
		haveOffsets bool
	)
	for {
		p.skipWS()
		keyStart, keyEnd, hasEsc, ok := p.peekStringSpan()
		if !ok {
			return TensorRef{}, core.NewError("mlx: safetensors tensor key parse failed: " + name)
		}
		p.skipWS()
		if !p.expect(':') {
			return TensorRef{}, core.NewError("mlx: safetensors tensor entry missing ':': " + name)
		}
		p.skipWS()
		// Dispatch on the raw byte span — no string materialisation.
		keyKind := unknownKey
		if !hasEsc {
			keyKind = innerKeyKind(p.data[keyStart:keyEnd])
		}
		switch keyKind {
		case dtypeKey:
			d, ok := p.parseInternedDType()
			if !ok {
				return TensorRef{}, core.NewError("mlx: safetensors dtype is not a string: " + name)
			}
			dtype = d
			haveDtype = true
		case shapeKey:
			s, l, err := p.parseShape(shapeSlab, name)
			if err != nil {
				return TensorRef{}, err
			}
			shapeStart = s
			shapeLen = l
			haveShape = true
		case dataOffsetsKey:
			begin, end, err := p.parseDataOffsets(name)
			if err != nil {
				return TensorRef{}, err
			}
			offsetBegin = begin
			offsetEnd = end
			haveOffsets = true
		default:
			// Forward-compat — unknown keys in tensor entries are
			// skipped silently (matches encoding/json with a struct
			// that has only known fields).
			if err := p.skipValue(); err != nil {
				return TensorRef{}, err
			}
		}
		p.skipWS()
		switch p.peek() {
		case ',':
			p.pos++
		case '}':
			p.pos++
			if !haveDtype || !haveShape || !haveOffsets {
				return TensorRef{}, core.NewError("mlx: safetensors tensor is missing required field: " + name)
			}
			if offsetBegin < 0 || offsetEnd < offsetBegin {
				return TensorRef{}, core.NewError("mlx: safetensors tensor offsets are invalid: " + name)
			}
			shape := (*shapeSlab)[shapeStart : shapeStart+shapeLen : shapeStart+shapeLen]
			elements := 1
			for _, dim := range shape {
				elements *= int(dim)
			}
			return TensorRef{
				Name:      name,
				Path:      path,
				DType:     dtype,
				Shape:     shape,
				Elements:  elements,
				DataStart: dataStart + offsetBegin,
				ByteLen:   offsetEnd - offsetBegin,
			}, nil
		default:
			return TensorRef{}, errTensorExpectCommaBrace
		}
	}
}

// innerKey is the discriminator for the three known keys inside a
// safetensors tensor entry. Anything else triggers the skip-value
// path.
type innerKey int

const (
	unknownKey innerKey = iota
	dtypeKey
	shapeKey
	dataOffsetsKey
)

// innerKeyKind matches a raw key byte span against the three known
// safetensors keys without ever allocating a string. The implementation
// is a length-first switch with direct byte compares — the same shape
// as DTypeByteSize's hand-rolled match.
func innerKeyKind(key []byte) innerKey {
	switch len(key) {
	case 5:
		// "shape" or "dtype" — both 5 bytes.
		if key[0] == 's' && key[1] == 'h' && key[2] == 'a' && key[3] == 'p' && key[4] == 'e' {
			return shapeKey
		}
		if key[0] == 'd' && key[1] == 't' && key[2] == 'y' && key[3] == 'p' && key[4] == 'e' {
			return dtypeKey
		}
	case 12:
		// "data_offsets"
		if key[0] == 'd' && key[1] == 'a' && key[2] == 't' && key[3] == 'a' &&
			key[4] == '_' && key[5] == 'o' && key[6] == 'f' && key[7] == 'f' &&
			key[8] == 's' && key[9] == 'e' && key[10] == 't' && key[11] == 's' {
			return dataOffsetsKey
		}
	}
	return unknownKey
}

// parseInternedDType reads a dtype JSON string and returns one of the
// pre-allocated canonical dtype constants. This avoids:
//   - the string conversion alloc on the raw dtype span
//   - the core.Upper alloc when the source is lowercase
//
// All safetensors writers in practice use uppercase canonical names
// (F32, F16, BF16, F64, U8, U16, U32, U64, I8, I16, I32, I64, BOOL,
// F8_E5M2, F8_E4M3FN). The interner returns the canonical pointer for
// any case variant; unknown dtypes fall through to a heap string so
// downstream DTypeByteSize errors carry the original spelling.
func (p *jsonParser) parseInternedDType() (string, bool) {
	if p.pos >= len(p.data) || p.data[p.pos] != '"' {
		return "", false
	}
	start := p.pos + 1
	i := start
	for i < len(p.data) {
		c := p.data[i]
		if c == '"' {
			p.pos = i + 1
			return internDType(p.data[start:i]), true
		}
		if c == '\\' {
			// dtype values are short ASCII tokens — escapes are not
			// expected, but if we see one fall through to the slow
			// path which yields the heap string.
			return p.parseStringEscaped(start)
		}
		i++
	}
	return "", false
}

// internDType returns the canonical uppercase string for the supplied
// dtype byte span without allocating in the common case. The match is
// case-insensitive — uppercase canonicals exact-match in the most
// common path, and the (rare) lowercase variants from older writers
// pick up the same canonical pointer.
func internDType(b []byte) string {
	switch len(b) {
	case 2:
		// I8, U8 — i / u + 8.
		c0 := b[0]
		if (c0 == 'I' || c0 == 'i') && b[1] == '8' {
			return "I8"
		}
		if (c0 == 'U' || c0 == 'u') && b[1] == '8' {
			return "U8"
		}
	case 3:
		// F16, F32, F64, I16, I32, I64, U16, U32, U64.
		c0 := b[0]
		c1 := b[1]
		c2 := b[2]
		// uppercase canonicals first — the fast path.
		switch {
		case c0 == 'F' && c1 == '3' && c2 == '2':
			return "F32"
		case c0 == 'F' && c1 == '1' && c2 == '6':
			return "F16"
		case c0 == 'F' && c1 == '6' && c2 == '4':
			return "F64"
		case c0 == 'I' && c1 == '3' && c2 == '2':
			return "I32"
		case c0 == 'I' && c1 == '6' && c2 == '4':
			return "I64"
		case c0 == 'I' && c1 == '1' && c2 == '6':
			return "I16"
		case c0 == 'U' && c1 == '3' && c2 == '2':
			return "U32"
		case c0 == 'U' && c1 == '6' && c2 == '4':
			return "U64"
		case c0 == 'U' && c1 == '1' && c2 == '6':
			return "U16"
		}
		// lowercase / mixed — single-character normalise.
		if c0 == 'f' || c0 == 'F' {
			if c1 == '3' && c2 == '2' {
				return "F32"
			}
			if c1 == '1' && c2 == '6' {
				return "F16"
			}
			if c1 == '6' && c2 == '4' {
				return "F64"
			}
		}
		if c0 == 'i' || c0 == 'I' {
			if c1 == '3' && c2 == '2' {
				return "I32"
			}
			if c1 == '6' && c2 == '4' {
				return "I64"
			}
			if c1 == '1' && c2 == '6' {
				return "I16"
			}
		}
		if c0 == 'u' || c0 == 'U' {
			if c1 == '3' && c2 == '2' {
				return "U32"
			}
			if c1 == '6' && c2 == '4' {
				return "U64"
			}
			if c1 == '1' && c2 == '6' {
				return "U16"
			}
		}
	case 4:
		// BF16, BOOL.
		c0 := b[0]
		if (c0 == 'B' || c0 == 'b') && (b[1] == 'F' || b[1] == 'f') && b[2] == '1' && b[3] == '6' {
			return "BF16"
		}
		if (c0 == 'B' || c0 == 'b') && (b[1] == 'O' || b[1] == 'o') && (b[2] == 'O' || b[2] == 'o') && (b[3] == 'L' || b[3] == 'l') {
			return "BOOL"
		}
	case 7:
		// F8_E5M2
		if (b[0] == 'F' || b[0] == 'f') && b[1] == '8' && b[2] == '_' &&
			(b[3] == 'E' || b[3] == 'e') && b[4] == '5' &&
			(b[5] == 'M' || b[5] == 'm') && b[6] == '2' {
			return "F8_E5M2"
		}
	case 9:
		// F8_E4M3FN
		if (b[0] == 'F' || b[0] == 'f') && b[1] == '8' && b[2] == '_' &&
			(b[3] == 'E' || b[3] == 'e') && b[4] == '4' &&
			(b[5] == 'M' || b[5] == 'm') && b[6] == '3' &&
			(b[7] == 'F' || b[7] == 'f') && (b[8] == 'N' || b[8] == 'n') {
			return "F8_E4M3FN"
		}
	}
	// Non-canonical dtype — uppercase the heap string so downstream
	// DTypeByteSize errors carry the user-visible form. core.Upper
	// is a no-op when already uppercase ASCII.
	return core.Upper(string(b))
}

// parseShape walks a JSON array of positive integers and appends each
// dim into shapeSlab as uint64. Returns the start index and length of
// the carved span. Callers slice shapeSlab directly with cap clamped
// so consumers cannot scribble past their dim range.
func (p *jsonParser) parseShape(shapeSlab *[]uint64, tensorName string) (int, int, error) {
	if !p.expect('[') {
		return 0, 0, core.NewError("mlx: safetensors shape is not an array: " + tensorName)
	}
	start := len(*shapeSlab)
	p.skipWS()
	if p.peek() == ']' {
		// Zero-dim shape — accept but produce empty slice.
		p.pos++
		return start, 0, nil
	}
	for {
		p.skipWS()
		dim, ok := p.parseInt64()
		if !ok {
			return 0, 0, core.NewError("mlx: safetensors shape dim is not an integer: " + tensorName)
		}
		if dim <= 0 {
			return 0, 0, core.NewError("mlx: safetensors tensor has invalid shape: " + tensorName)
		}
		*shapeSlab = append(*shapeSlab, uint64(dim))
		p.skipWS()
		switch p.peek() {
		case ',':
			p.pos++
		case ']':
			p.pos++
			return start, len(*shapeSlab) - start, nil
		default:
			return 0, 0, core.NewError("mlx: safetensors shape expected ',' or ']': " + tensorName)
		}
	}
}

// parseDataOffsets reads the [begin, end] array. It produces two raw
// int64s with no intermediate slice.
func (p *jsonParser) parseDataOffsets(tensorName string) (int64, int64, error) {
	if !p.expect('[') {
		return 0, 0, core.NewError("mlx: safetensors data_offsets is not an array: " + tensorName)
	}
	p.skipWS()
	begin, ok := p.parseInt64()
	if !ok {
		return 0, 0, core.NewError("mlx: safetensors data_offsets[0] is not an integer: " + tensorName)
	}
	p.skipWS()
	if !p.expect(',') {
		return 0, 0, core.NewError("mlx: safetensors data_offsets missing ',': " + tensorName)
	}
	p.skipWS()
	end, ok := p.parseInt64()
	if !ok {
		return 0, 0, core.NewError("mlx: safetensors data_offsets[1] is not an integer: " + tensorName)
	}
	p.skipWS()
	if !p.expect(']') {
		return 0, 0, core.NewError("mlx: safetensors data_offsets missing ']': " + tensorName)
	}
	return begin, end, nil
}

// skipValue walks a JSON value (any type) and discards it. Used for
// the __metadata__ entry's body (which can be an object with arbitrary
// structure) and for any unknown keys in a tensor entry.
func (p *jsonParser) skipValue() error {
	p.skipWS()
	if p.pos >= len(p.data) {
		return errHeaderTruncated
	}
	c := p.data[p.pos]
	switch {
	case c == '{':
		return p.skipObject()
	case c == '[':
		return p.skipArray()
	case c == '"':
		return p.skipString()
	case c == 't' || c == 'f' || c == 'n':
		return p.skipLiteral()
	case c == '-' || (c >= '0' && c <= '9'):
		// Skip number — accept any JSON number form (digits, sign,
		// decimal, exponent). We don't need the value.
		p.pos++
		for p.pos < len(p.data) {
			d := p.data[p.pos]
			if (d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-' {
				p.pos++
				continue
			}
			break
		}
		return nil
	}
	return errSkipValueToken
}

// skipObject consumes a balanced object {...} including all nested
// objects/arrays/strings.
func (p *jsonParser) skipObject() error {
	if !p.expect('{') {
		return errExpectBrace
	}
	p.skipWS()
	if p.peek() == '}' {
		p.pos++
		return nil
	}
	for {
		p.skipWS()
		if err := p.skipString(); err != nil {
			return err
		}
		p.skipWS()
		if !p.expect(':') {
			return errExpectColon
		}
		if err := p.skipValue(); err != nil {
			return err
		}
		p.skipWS()
		switch p.peek() {
		case ',':
			p.pos++
		case '}':
			p.pos++
			return nil
		default:
			return errExpectCommaBraceObject
		}
	}
}

// skipArray consumes a balanced array [...] including all nested
// elements.
func (p *jsonParser) skipArray() error {
	if !p.expect('[') {
		return errExpectBracket
	}
	p.skipWS()
	if p.peek() == ']' {
		p.pos++
		return nil
	}
	for {
		if err := p.skipValue(); err != nil {
			return err
		}
		p.skipWS()
		switch p.peek() {
		case ',':
			p.pos++
		case ']':
			p.pos++
			return nil
		default:
			return errExpectCommaBracketArray
		}
	}
}

// skipString consumes a string literal without materialising the
// contents — used inside skipObject (keys) and skipValue (string
// values).
func (p *jsonParser) skipString() error {
	if !p.expect('"') {
		return errExpectString
	}
	for p.pos < len(p.data) {
		c := p.data[p.pos]
		if c == '"' {
			p.pos++
			return nil
		}
		if c == '\\' {
			// Skip the escape sequence. \uXXXX is 6 bytes (the \u plus
			// 4 hex digits); the others are 2 bytes.
			if p.pos+1 >= len(p.data) {
				return errTruncatedEscape
			}
			if p.data[p.pos+1] == 'u' {
				p.pos += 6
			} else {
				p.pos += 2
			}
			continue
		}
		p.pos++
	}
	return errUnterminatedString
}

// skipLiteral consumes a true/false/null literal.
func (p *jsonParser) skipLiteral() error {
	switch p.peek() {
	case 't':
		if p.pos+4 <= len(p.data) && string(p.data[p.pos:p.pos+4]) == "true" {
			p.pos += 4
			return nil
		}
	case 'f':
		if p.pos+5 <= len(p.data) && string(p.data[p.pos:p.pos+5]) == "false" {
			p.pos += 5
			return nil
		}
	case 'n':
		if p.pos+4 <= len(p.data) && string(p.data[p.pos:p.pos+4]) == "null" {
			p.pos += 4
			return nil
		}
	}
	return errUnknownLiteral
}

// countTensorsAndDims is the cheap first pass over the header bytes.
// It scans for the structure of each tensor entry and accumulates two
// numbers: the count of non-metadata tensors and the total number of
// shape dims across all of them. These size the index map, Names
// slice, and shape slab in a single up-front allocation each.
//
// The scan is structural — it tracks JSON brace depth so it never
// confuses an inner __metadata__ block's shape-like values with real
// tensor shapes, and it skips strings cleanly so braces inside string
// literals don't perturb the depth count.
//
// Returns (-1, -1) when the header isn't a recognisable object — the
// caller falls back to a conservative size and the full parser still
// catches the malformed input.
func countTensorsAndDims(data []byte) (int, int) {
	pos := 0
	n := len(data)
	// skip leading whitespace
	for pos < n {
		c := data[pos]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		pos++
	}
	if pos >= n || data[pos] != '{' {
		return -1, -1
	}
	pos++

	tensors := 0
	totalDims := 0
	// We're now inside the top-level object. Each iteration consumes
	// one "key":value entry, where the value is itself an object.
	for {
		// skip ws
		for pos < n {
			c := data[pos]
			if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
				break
			}
			pos++
		}
		if pos >= n {
			return -1, -1
		}
		if data[pos] == '}' {
			return tensors, totalDims
		}
		if data[pos] != '"' {
			return -1, -1
		}
		// Read key — note start, scan to closing quote.
		pos++
		keyStart := pos
		for pos < n && data[pos] != '"' {
			if data[pos] == '\\' {
				if pos+1 < n && data[pos+1] == 'u' {
					pos += 6
				} else {
					pos += 2
				}
				continue
			}
			pos++
		}
		if pos >= n {
			return -1, -1
		}
		keyEnd := pos
		pos++ // closing quote
		isMetadata := keyEnd-keyStart == 12 && string(data[keyStart:keyEnd]) == "__metadata__"

		// skip ws, expect ':'
		for pos < n {
			c := data[pos]
			if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
				break
			}
			pos++
		}
		if pos >= n || data[pos] != ':' {
			return -1, -1
		}
		pos++

		// Inside the value. For tensor entries, count dims in "shape".
		// For __metadata__, skip the entire balanced object.
		if isMetadata {
			// Skip a balanced JSON value with string-aware bracket
			// counting.
			depth := 0
			for pos < n {
				c := data[pos]
				switch c {
				case '"':
					// skip string literal
					pos++
					for pos < n && data[pos] != '"' {
						if data[pos] == '\\' {
							if pos+1 < n && data[pos+1] == 'u' {
								pos += 6
							} else {
								pos += 2
							}
							continue
						}
						pos++
					}
					if pos >= n {
						return -1, -1
					}
					pos++
				case '{', '[':
					depth++
					pos++
				case '}', ']':
					depth--
					pos++
					if depth == 0 {
						goto afterMetadataValue
					}
				default:
					pos++
				}
			}
			return -1, -1
		afterMetadataValue:
		} else {
			// Walk into the tensor entry to count "shape" dims. We
			// know the structure but inner-key order isn't fixed.
			if pos >= n || data[pos] != '{' {
				return -1, -1
			}
			pos++
			depth := 1
			tensorDims := 0
			haveDims := false
			for pos < n && depth > 0 {
				c := data[pos]
				switch {
				case c == '"':
					// Read key/string.
					pos++
					keyS := pos
					for pos < n && data[pos] != '"' {
						if data[pos] == '\\' {
							if pos+1 < n && data[pos+1] == 'u' {
								pos += 6
							} else {
								pos += 2
							}
							continue
						}
						pos++
					}
					if pos >= n {
						return -1, -1
					}
					keyE := pos
					pos++ // closing quote
					if depth == 1 && !haveDims && keyE-keyS == 5 && string(data[keyS:keyE]) == "shape" {
						// Locate the ':' and the '[', then count
						// commas+1 to get dim count.
						for pos < n {
							c2 := data[pos]
							if c2 != ' ' && c2 != '\t' && c2 != '\n' && c2 != '\r' && c2 != ':' {
								break
							}
							pos++
						}
						if pos >= n || data[pos] != '[' {
							return -1, -1
						}
						pos++
						// Empty shape?
						for pos < n {
							c2 := data[pos]
							if c2 != ' ' && c2 != '\t' && c2 != '\n' && c2 != '\r' {
								break
							}
							pos++
						}
						if pos < n && data[pos] == ']' {
							pos++
							tensorDims = 0
							haveDims = true
							continue
						}
						// Count integers in the shape array.
						commas := 0
						for pos < n {
							c2 := data[pos]
							if c2 == ',' {
								commas++
								pos++
								continue
							}
							if c2 == ']' {
								pos++
								break
							}
							pos++
						}
						tensorDims = commas + 1
						haveDims = true
					}
				case c == '{' || c == '[':
					depth++
					pos++
				case c == '}' || c == ']':
					depth--
					pos++
				default:
					pos++
				}
			}
			tensors++
			totalDims += tensorDims
		}

		// skip ws
		for pos < n {
			c := data[pos]
			if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
				break
			}
			pos++
		}
		if pos >= n {
			return -1, -1
		}
		switch data[pos] {
		case ',':
			pos++
		case '}':
			return tensors, totalDims
		default:
			return -1, -1
		}
	}
}
