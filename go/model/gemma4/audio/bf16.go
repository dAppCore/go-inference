// SPDX-Licence-Identifier: EUPL-1.2

// Package audio is the engine-neutral, pure-host float32 port of the Gemma 4 Conformer audio tower
// (Mantis #1839). It carries the mel front-end (waveform → log-mel input_features) and the tower
// forward (subsample → 12 Conformer layers → output projection) as plain host arithmetic, consuming
// the neutral model.LoadedAudio payload the gemma4 assembler produces. Any backend — engine/hip on
// AMD, or engine/metal once its lane is free to migrate — can drive the same tower without a GPU
// dependency; the device GEMM path is a later optimisation layered over this reference forward.
//
// The math mirrors HF transformers Gemma4AudioModel operation-for-operation (the same reference the
// engine/metal native port was built from), so the shared HF goldens gate it directly.
package audio

import "math"

// bf16Size is the on-disk width of one bfloat16 weight element.
const bf16Size = 2

// bf16ToF32 widens one little-endian bfloat16 element (lo, hi bytes) to float32 — an exact upper-16-bit
// placement, matching the engine widen so identical weights produce identical activations.
func bf16ToF32(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// f32ToBF16 rounds one float32 to bfloat16 (round-to-nearest-even), keeping NaNs quiet — the same
// rounding the engine uses, so a round-trip through the feature buffer is byte-identical across engines.
func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it quiet, non-zero mantissa
		return uint16(bits>>16) | 0x0040
	}
	rounding := (bits>>16)&1 + 0x7fff
	return uint16((bits + rounding) >> 16)
}

// bf16ToF32Slice widens a whole bf16 byte buffer to a fresh float32 slice.
func bf16ToF32Slice(b []byte) []float32 {
	out := make([]float32, len(b)/bf16Size)
	for i := range out {
		out[i] = bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
	}
	return out
}

// f32ToBf16Slice rounds a whole float32 slice to a fresh bf16 byte buffer (little-endian).
func f32ToBf16Slice(f []float32) []byte {
	b := make([]byte, len(f)*bf16Size)
	for i, v := range f {
		h := f32ToBF16(v)
		b[i*bf16Size], b[i*bf16Size+1] = byte(h), byte(h>>8)
	}
	return b
}
