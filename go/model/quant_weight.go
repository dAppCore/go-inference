// SPDX-Licence-Identifier: EUPL-1.2

package model

// QuantWeight is one 2-D projection's MLX affine-quantised weight kept PACKED: the packed
// uint32 codes plus the bf16 group scales/biases, with the weight's own (Bits, GroupSize)
// and its logical (OutDim × InDim) shape. The composed hybrid lane carries these through to
// the engine's quant matvec seam instead of widening — a 27B checkpoint dequantised to f32
// is ~110 GB, dead on arrival on every machine, so packed-native is the feature, not an
// optimisation.
//
// Bits is always a width the shipped metallib dispatches (2/3/4/5/6/8): Bonsai's 1-bit packs
// are widened to 2-bit at load (mlxaffine.RepackB1ToB2 — an exact, zero-quality-change code
// widening) so there is no b_1 kernel to miss.
//
// Packed memory: usually OWNED (copied out of the checkpoint at load), so a QuantWeight outlives
// the mapping the engine binds it from. The composed zero-copy loader (LoadComposedDir) is the
// exception — Packed there is a VIEW into the checkpoint mmap (no heap copy, cutting load RSS), and
// the composed model owns that mapping and unmaps it on Close/finalize. Under that path a caller
// must NOT retain Packed beyond the model's lifetime. (A b1→b2 repack always allocates owned
// buffers, so a repacked weight is owned regardless of load path.)
type QuantWeight struct {
	Packed    []byte // MLX packed codes, little-endian uint32 words: [OutDim, InDim·Bits/32]
	Scales    []byte // bf16, one per group per row: [OutDim, InDim/GroupSize]
	Biases    []byte // bf16, one per group per row: [OutDim, InDim/GroupSize]
	Bits      int
	GroupSize int
	OutDim    int // logical rows — the N of the y = x·Wᵀ projection
	InDim     int // logical cols — the K; a whole number of groups
}

// ConcatQuantRows concatenates two MLX-affine quant weights along their output rows (dim 0) into a
// single [a.OutDim+b.OutDim, InDim] weight — a's rows first, then b's. Every per-row byte range
// (Packed, Scales, Biases) is laid out row-major with a stride fixed by (InDim, Bits, GroupSize), so
// with those three shared the concatenation is a plain byte append. Because a quant matvec
// dequantises and dots each output row INDEPENDENTLY (the row n slices in matNTQuantHost), a matvec
// over the result yields a's outputs in [0:a.OutDim] and b's in [a.OutDim:] BYTE-IDENTICALLY to two
// separate matvecs — which makes a fused [gate‖up] expert projection a numerically-free packing of
// the two halves (the composed MoE gate+up fusion; the single-expert twin of engine/metal's
// fuseExpertGateUpQuant). The concat MATERIALISES owned buffers, so the result outlives any mmap the
// inputs viewed. a and b must share InDim, Bits and GroupSize (gate and up of one expert always do);
// a nil input or a geometry mismatch returns nil.
//
//	gateUp := ConcatQuantRows(gateQ, upQ) // one matvec outputs [gate‖up]; split the halves for silu-mul
func ConcatQuantRows(a, b *QuantWeight) *QuantWeight {
	if a == nil || b == nil || a.InDim != b.InDim || a.Bits != b.Bits || a.GroupSize != b.GroupSize {
		return nil
	}
	cat := func(x, y []byte) []byte {
		out := make([]byte, 0, len(x)+len(y))
		out = append(out, x...)
		return append(out, y...)
	}
	return &QuantWeight{
		Packed:    cat(a.Packed, b.Packed),
		Scales:    cat(a.Scales, b.Scales),
		Biases:    cat(a.Biases, b.Biases),
		Bits:      a.Bits,
		GroupSize: a.GroupSize,
		OutDim:    a.OutDim + b.OutDim,
		InDim:     a.InDim,
	}
}
