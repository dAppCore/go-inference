// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// assistant_draft_qhead.go — the quantised drafter head (#53). The non-ordered
// gemma4 assistants' tied-embedding head is a full-vocab bf16 gemv per draft
// step: at 26B geometry that is a 537MB scan (262144×1024 bf16) — ~1.8ms of
// every ~2.1ms draft step, five steps per round. The TARGET's own tied head in
// a qat-4bit checkpoint runs the same projection 4-bit quantised, and its
// argmax IS the emitted stream's correctness bar — so the drafter's head
// joins that class: the bf16 embedding quantises ONCE at fused-draft build
// (MLX affine, group 64, 4-bit — the exact layout extractAffineCode /
// dequantizeAffineRowsF32 pin), and the per-step logits ride the fused CB as
// one affine-qmv instead of the bf16 gemv, cutting the scan to ~134MB+scales.
// Draft picks may differ from the bf16 head's in near-tie cases; the MTP
// verify bar makes that an acceptance-rate concern only, never a correctness
// one (accepted tokens are always the target's own greedy).

// mtpDraftQHeadDisabled restores the bf16 drafter head (LTHN_MTP_DRAFT_QHEAD=0)
// — the repro anchor for the quantised-head lane.
var mtpDraftQHeadDisabled = os.Getenv("LTHN_MTP_DRAFT_QHEAD") == "0"

// assistantQHeadGroupSize is the drafter head's affine group — matching the
// qat checkpoints' own head grouping keeps the error class identical.
const assistantQHeadGroupSize = 64

// quantiseAffine4RowsBF16 quantises a row-major bf16 matrix to MLX 4-bit
// affine: per group of assistantQHeadGroupSize columns, scale=(max−min)/15 and
// bias=min (bf16-stored), codes packed LSB-first (low nibble = even column).
// The layout is byte-compatible with the checkpoint loaders' QuantWeight
// triples — dequantizeAffineRowsF32 is the round-trip oracle in the tests.
func quantiseAffine4RowsBF16(w []byte, rows, cols int) (packed, scales, biases []byte, err error) {
	const bits, groupSize = 4, assistantQHeadGroupSize
	if rows <= 0 || cols <= 0 || cols%groupSize != 0 {
		return nil, nil, nil, core.NewError("native.quantiseAffine4RowsBF16: rows/cols must be positive and cols must divide by the group size")
	}
	if len(w) != rows*cols*bf16Size {
		return nil, nil, nil, core.NewError("native.quantiseAffine4RowsBF16: matrix must be rows*cols bf16 bytes")
	}
	rowPacked := cols * bits / 8
	groupsPerRow := cols / groupSize
	rowSB := groupsPerRow * bf16Size
	packed = make([]byte, rows*rowPacked)
	scales = make([]byte, rows*rowSB)
	biases = make([]byte, rows*rowSB)
	parallelRows(rows, func(r int) {
		wRow := w[r*cols*bf16Size : (r+1)*cols*bf16Size]
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		sRow := scales[r*rowSB : (r+1)*rowSB]
		bRow := biases[r*rowSB : (r+1)*rowSB]
		var vals [groupSize]float32
		for g := range groupsPerRow {
			base := g * groupSize
			minV, maxV := float32(0), float32(0)
			for j := range groupSize {
				f := bf16ToF32(wRow[(base+j)*bf16Size], wRow[(base+j)*bf16Size+1])
				vals[j] = f
				if j == 0 || f < minV {
					minV = f
				}
				if j == 0 || f > maxV {
					maxV = f
				}
			}
			scale := (maxV - minV) / 15
			// bf16-store FIRST, then code against the stored values — the qmv
			// kernel dequantises with the bf16-rounded scale/bias, so coding
			// against the f32 originals would bake in a reconstruction skew.
			sh := f32ToBF16(scale)
			bh := f32ToBF16(minV)
			sRow[g*bf16Size], sRow[g*bf16Size+1] = byte(sh), byte(sh>>8)
			bRow[g*bf16Size], bRow[g*bf16Size+1] = byte(bh), byte(bh>>8)
			qScale := bf16ToF32(byte(sh), byte(sh>>8))
			qBias := bf16ToF32(byte(bh), byte(bh>>8))
			inv := float32(0)
			if qScale != 0 {
				inv = 1 / qScale
			}
			for j := 0; j < groupSize; j += 2 {
				lo := affine4Code(vals[j], qBias, inv)
				hi := affine4Code(vals[j+1], qBias, inv)
				pRow[(base+j)>>1] = byte(lo | hi<<4)
			}
		}
	})
	return packed, scales, biases, nil
}

// affine4Code is one element's 4-bit affine code: round((w−bias)/scale)
// clamped to [0,15]; a zero scale (flat group) codes 0 and the bias carries
// the value exactly.
func affine4Code(w, bias, invScale float32) uint32 {
	if invScale == 0 {
		return 0
	}
	q := roundF32((w - bias) * invScale)
	if q < 0 {
		return 0
	}
	if q > 15 {
		return 15
	}
	return uint32(q)
}
