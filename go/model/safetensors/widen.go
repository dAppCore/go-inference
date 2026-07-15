// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"math"
	"unsafe"

	core "dappco.re/go"
)

// widen.go converts a checkpoint's F16 float tensors to BF16 at load. Many MLX community packs ship
// the un-quantised float tensors (norms, affine scales/biases, additive q/k/v biases, and any dense
// weights) as IEEE-754 half (F16, {"format":"mlx"}) rather than bfloat16 — but the byte-native engine
// binds shards zero-copy and its matvec/norm kernels read bfloat16_t, so an F16 tensor bound as-is is
// misread bit-for-bit into garbage. F16 and BF16 are DIFFERENT layouts (5-bit vs 8-bit exponent), so
// widening is a genuine numeric reformat, not a reinterpret — the tensor's Data is replaced with a
// fresh heap BF16 buffer. Requiring bf16 would exclude every F16 pack, so we widen rather than reject.

// Float32ToBFloat16 rounds a float32 to bfloat16 bits with round-to-nearest-even — the inverse of
// BFloat16ToFloat32, matching mlx's AsType(..., bfloat16) (and the engine's own f32→bf16 constant
// conversion) so a widened tensor equals the bf16 tensor mlx would have written for the same input.
// NaN is kept quiet (non-zero mantissa); Inf and signed zero survive exactly.
//
//	b := safetensors.Float32ToBFloat16(1.0) // 0x3F80
func Float32ToBFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it quiet, preserve a non-zero mantissa
		return uint16(bits>>16) | 0x0040
	}
	rounding := (bits>>16)&1 + 0x7fff // round-to-nearest-even on the truncated low 16 bits
	return uint16((bits + rounding) >> 16)
}

// WidenF16ToBF16 converts a raw little-endian F16 tensor payload to the equivalent BF16 payload,
// element by element (F16 → exact float32 → round-to-nearest-even bfloat16). The output is a fresh
// heap buffer of the SAME byte length (both half formats are 2 bytes/element), so a widened tensor
// keeps its shape and every downstream byte offset. A trailing odd byte (never a valid F16 payload)
// is dropped; an empty input yields an empty slice.
//
//	bf16 := safetensors.WidenF16ToBF16(tensor.Data)
func WidenF16ToBF16(f16 []byte) []byte {
	n := len(f16) / 2
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		h := uint16(f16[2*i]) | uint16(f16[2*i+1])<<8
		b := Float32ToBFloat16(Float16ToFloat32(h))
		out[2*i] = byte(b)
		out[2*i+1] = byte(b >> 8)
	}
	return out
}

// WidenF16TensorsToBF16 widens every F16 tensor in the mapping to BF16 in place: it replaces each
// F16 tensor's Data with a fresh heap BF16 buffer (WidenF16ToBF16) and rewrites its Dtype to "BF16",
// and records the new heap Data ranges so the zero-copy binder resident-binds them (they are no
// longer shard mmap views — see IsWidened). BF16 and non-float tensors (the packed 4-bit U32 codes,
// U8/I32/…) are left untouched, so a bf16 checkpoint stays byte-for-byte identical and its weights
// stay zero-copy. Returns the number of tensors widened. Call once, after any Normalize pass and
// before the assembler reads the tensors.
func (d *DirMapping) WidenF16TensorsToBF16() int {
	if d == nil {
		return 0
	}
	widened := 0
	for name, t := range d.Tensors {
		if core.Upper(t.Dtype) != "F16" || len(t.Data) == 0 {
			continue
		}
		wide := WidenF16ToBF16(t.Data)
		t.Data = wide
		t.Dtype = "BF16"
		d.Tensors[name] = t
		start := uintptr(unsafe.Pointer(&wide[0]))
		d.widened = append(d.widened, widenedRange{start: start, end: start + uintptr(len(wide))})
		widened++
	}
	return widened
}

// IsWidened reports whether b is (a view into) a tensor this mapping widened from F16 to BF16 — i.e.
// a fresh heap buffer, not a page-aligned shard mmap view. The zero-copy binder consults this so a
// widened companion tensor (a norm / scale / bias / q-k-v bias) binds resident instead of failing the
// "weight is not a view into any mapped shard" wrong-mapping guard the strict projection resolver keeps.
func (d *DirMapping) IsWidened(b []byte) bool {
	if d == nil || len(b) == 0 {
		return false
	}
	p := uintptr(unsafe.Pointer(&b[0]))
	for _, r := range d.widened {
		if p >= r.start && p < r.end {
			return true
		}
	}
	return false
}
