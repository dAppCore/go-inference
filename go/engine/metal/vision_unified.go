// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// vision_unified.go — the ENCODER-FREE vision embedder (gemma4_unified, 12B):
// raw 48px model patches project straight into the backbone with no SigLIP
// tower. Per the upstream reference: LayerNorm → patch dense (+bias) →
// LayerNorm → factorised per-axis position add → LayerNorm → scale-free
// RMSNorm → projection. The LayerNorms are nn.LayerNorm (weight+bias,
// PyTorch default eps); the pre-projection RMSNorm carries NO parameters
// (with_scale=False) — exactly x/rms(x) at the vision rms epsilon, NOT the
// gemma (1+w) rms op. The two matmuls ride the batched quant qmm (or the
// bf16 steel GEMM on unquantised packs); the norms and adds are host-side —
// the whole embedder is ≤280 rows, load-bound, and runs once per image.

// unifiedLinearRows projects rows [n × inDim] through a loaded linear
// (affine-quant or plain bf16), returning [n × OutDim] bf16. Bias, when the
// linear carries one, is added on the host.
func unifiedLinearRows(lin model.LoadedVisionLinear, x []byte, n int) ([]byte, error) {
	if lin.Weight == nil || n <= 0 {
		return nil, core.NewError("native.unifiedLinearRows: missing weight or empty rows")
	}
	outDim, inDim := lin.OutDim, lin.InDim
	if outDim <= 0 || inDim <= 0 {
		return nil, core.NewError("native.unifiedLinearRows: linear dims are undeclared")
	}
	if len(x) != n*inDim*bf16Size {
		return nil, core.NewError(core.Sprintf("native.unifiedLinearRows: x bytes = %d, want %d", len(x), n*inDim*bf16Size))
	}
	var out []byte
	if len(lin.Scales) == 0 { // plain bf16 weight
		got, err := MatMulBF16NT(x, lin.Weight, n, inDim, outDim)
		if err != nil {
			return nil, err
		}
		out = got
	} else {
		gs, bits := lin.GroupSize, lin.Bits
		if gs <= 0 || bits <= 0 || inDim%gs != 0 {
			return nil, core.NewError("native.unifiedLinearRows: malformed quant geometry")
		}
		out = make([]byte, n*outDim*bf16Size)
		var encErr error
		withAutoreleasePool(func() {
			wq := residentBytes(lin.Weight)
			scales := residentBytes(lin.Scales)
			biases := residentBytes(lin.Biases)
			xBuf := sharedBytes(x)
			outBuf := scratchBF16(n * outDim)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			encErr = encQMMTBF16At(enc, wq, scales, biases, xBuf, outBuf, 0, 0, 0, 0, 0, n, outDim, inDim, gs, bits)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			if encErr == nil {
				copy(out, unsafe.Slice((*byte)(outBuf.Contents()), n*outDim*bf16Size))
			}
		})
		if encErr != nil {
			return nil, encErr
		}
	}
	if len(lin.Bias) == outDim*bf16Size {
		for r := range n {
			row := out[r*outDim*bf16Size:]
			for c := range outDim {
				v := bf16ToF32(row[2*c], row[2*c+1]) + bf16ToF32(lin.Bias[2*c], lin.Bias[2*c+1])
				h := f32ToBF16(v)
				row[2*c] = byte(h)
				row[2*c+1] = byte(h >> 8)
			}
		}
	}
	return out, nil
}

// unifiedAddPositions adds the factorised per-axis position embeddings in
// place: row r gains pos[rowIdx, 0, :] + pos[colIdx, 1, :]. Indices are
// clamped to the table; a padded row (index < 0) gains nothing — matching the
// reference's valid-mask.
func unifiedAddPositions(h []byte, positions []int32, n, dim, posemb int, pos []byte) error {
	if len(pos) != posemb*2*dim*bf16Size {
		return core.NewError(core.Sprintf("native.unifiedAddPositions: pos table bytes = %d, want %d", len(pos), posemb*2*dim*bf16Size))
	}
	if len(positions) < n*2 {
		return core.NewError("native.unifiedAddPositions: positions are short")
	}
	for r := range n {
		row := h[r*dim*bf16Size:]
		for axis := range 2 {
			idx := int(positions[r*2+axis])
			if idx < 0 {
				continue
			}
			if idx >= posemb {
				idx = posemb - 1
			}
			table := pos[(idx*2+axis)*dim*bf16Size:]
			for c := range dim {
				v := bf16ToF32(row[2*c], row[2*c+1]) + bf16ToF32(table[2*c], table[2*c+1])
				b := f32ToBF16(v)
				row[2*c] = byte(b)
				row[2*c+1] = byte(b >> 8)
			}
		}
	}
	return nil
}

// unifiedRMSNoScale normalises each row to x/rms(x) in place — the reference's
// with_scale=False RMSNorm, deliberately NOT the gemma (1+w) rms op.
func unifiedRMSNoScale(h []byte, n, dim int, eps float32) {
	for r := range n {
		row := h[r*dim*bf16Size:]
		var ss float64
		for c := range dim {
			v := float64(bf16ToF32(row[2*c], row[2*c+1]))
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(dim)+float64(eps))
		for c := range dim {
			v := float32(float64(bf16ToF32(row[2*c], row[2*c+1])) * inv)
			b := f32ToBF16(v)
			row[2*c] = byte(b)
			row[2*c+1] = byte(b >> 8)
		}
	}
}

// UnifiedVisionProject runs the encoder-free embedder over n model patches
// ([n × ModelPatchSize²·3] bf16, kernel-grouped) with their per-patch (row,
// col) position indices, returning the [n × TextHidden] soft-token features
// ready for placeholder injection.
func UnifiedVisionProject(uv *model.LoadedUnifiedVision, patches []byte, positions []int32, n int) ([]byte, error) {
	if uv == nil {
		return nil, core.NewError("native.UnifiedVisionProject: nil payload")
	}
	if n <= 0 {
		return nil, core.NewError("native.UnifiedVisionProject: no patches")
	}
	cfg := uv.Cfg
	patchDim := cfg.ModelPatchSize * cfg.ModelPatchSize * 3
	if len(patches) != n*patchDim*bf16Size {
		return nil, core.NewError(core.Sprintf("native.UnifiedVisionProject: patches bytes = %d, want %d", len(patches), n*patchDim*bf16Size))
	}
	h, err := LayerNormBF16(patches, uv.PatchLN1W, uv.PatchLN1B, n, patchDim, cfg.LayerNormEps)
	if err != nil {
		return nil, core.E("native.UnifiedVisionProject", "patch_ln1", err)
	}
	if h, err = unifiedLinearRows(uv.PatchDense, h, n); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "patch_dense", err)
	}
	if h, err = LayerNormBF16(h, uv.PatchLN2W, uv.PatchLN2B, n, cfg.MMEmbedDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "patch_ln2", err)
	}
	if err = unifiedAddPositions(h, positions, n, cfg.MMEmbedDim, cfg.PosembSize, uv.PosEmbedding); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "pos_embedding", err)
	}
	if h, err = LayerNormBF16(h, uv.PosNormW, uv.PosNormB, n, cfg.MMEmbedDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "pos_norm", err)
	}
	unifiedRMSNoScale(h, n, cfg.MMEmbedDim, cfg.RMSNormEps)
	out, err := unifiedLinearRows(uv.Projection, h, n)
	if err != nil {
		return nil, core.E("native.UnifiedVisionProject", "embedding_projection", err)
	}
	return out, nil
}
