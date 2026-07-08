// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// unifiedVisionDiag prints per-stage magnitude stats (#351 instrument; env LTHN_VISION_DIAG).
var unifiedVisionDiag = os.Getenv("LTHN_VISION_DIAG") != ""

func unifiedVisionStats(label string, b []byte, dim int) {
	if !unifiedVisionDiag {
		return
	}
	n := len(b) / 2
	var sum, maxAbs float64
	for i := range n {
		v := math.Abs(float64(bf16ToF32(b[2*i], b[2*i+1])))
		sum += v
		if v > maxAbs {
			maxAbs = v
		}
	}
	row0 := ""
	if dim > 0 && n >= dim {
		var s0 float64
		for i := range dim {
			s0 += math.Abs(float64(bf16ToF32(b[2*i], b[2*i+1])))
		}
		row0 = core.Sprintf(" row0sum=%.2f", s0)
	}
	nativeTraceLog(core.Sprintf("vision-diag %s: sum|x|=%.1f max|x|=%.3f n=%d%s\n", label, sum, maxAbs, n, row0))
}

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

// UnifiedVisionImagePatches decodes PNG/JPEG bytes through the shared Gemma 4
// sizing rule (aspect-preserving resize onto the patch budget, dims rounded to
// PoolingKernel·PatchSize multiples, rescale to [0,1] — VisionImagePixels) and
// returns the unified embedder's inputs: KERNEL-GROUPED model patches
// [n × (PoolKernel·PatchSize)²·3] bf16 — each model patch the raster concat of
// its PoolKernel² teacher patches, each teacher patch the proven
// HWC-innermost 16px flatten — plus the per-patch (row, col) position indices.
func UnifiedVisionImagePatches(data []byte, cfg *VisionImageFeatureConfig) ([]byte, []int32, int, error) {
	if v := os.Getenv("LTHN_VISION_MAXTOK"); v != "" { // #351 instrument: shrink the span (1 token = mask-free)
		if r := core.ParseInt(v, 10, 32); r.OK {
			if iv := r.Value.(int64); iv > 0 {
				c := *normalizeVisionImageFeatureConfig(cfg)
				c.MaxSoftTokens = int32(iv)
				cfg = &c
			}
		}
	}
	pixels, th, tw, softTokens, err := VisionImagePixels(data, cfg)
	if err != nil {
		return nil, nil, 0, err
	}
	ncfg := normalizeVisionImageFeatureConfig(cfg)
	if ncfg == nil {
		ncfg = normalizeVisionImageFeatureConfig(&VisionImageFeatureConfig{})
	}
	ps, pool := int(ncfg.PatchSize), int(ncfg.PoolingKernelSize)
	mp := ps * pool
	gh, gw := int(th)/mp, int(tw)/mp
	n := gh * gw
	if n <= 0 {
		return nil, nil, 0, core.NewError("native.UnifiedVisionImagePatches: image produced no model patches")
	}
	if n != softTokens {
		return nil, nil, 0, core.NewError(core.Sprintf("native.UnifiedVisionImagePatches: model patches %d != soft tokens %d", n, softTokens))
	}
	patchDim := mp * mp * 3
	out := make([]byte, n*patchDim*bf16Size)
	positions := make([]int32, n*2)
	row := 0
	for R := range gh {
		for C := range gw {
			// Position ids are (x, y) — the reference meshgrids width-first
			// with indexing="xy", so axis 0 of the pos table is the COLUMN.
			positions[row*2] = int32(C)
			positions[row*2+1] = int32(R)
			// Teacher patches flatten [py][px][c] (the reference permutes the
			// CHW image to (gh, gw, ps, ps, C) before the row reshape); the
			// model patch concatenates its kernel's teachers in row-major
			// kernel order.
			col := 0
			for ki := range pool {
				for kj := range pool {
					for py := range ps {
						y := (R*pool+ki)*ps + py
						for px := range ps {
							x := (C*pool+kj)*ps + px
							src := (y*int(tw) + x) * 3
							for c := range 3 {
								hh := f32ToBF16(pixels[src+c])
								dst := (row*patchDim + col) * bf16Size
								out[dst], out[dst+1] = byte(hh), byte(hh>>8)
								col++
							}
						}
					}
				}
			}
			row++
		}
	}
	return out, positions, n, nil
}

// UnifiedAudioProject runs the encoder-free audio head: raw 16 kHz samples in
// [-1,1], zero-padded to a whole number of AudioSamplesPerToken tokens, each
// token's samples projected straight into the backbone (scale-free RMSNorm →
// embed_audio) — no mel front-end, no Conformer. Returns the [n × TextHidden]
// soft-token features and n.
func UnifiedAudioProject(uv *model.LoadedUnifiedVision, samples []float32) ([]byte, int, error) {
	if uv == nil || uv.AudioProjection.Weight == nil || uv.Cfg.AudioSamplesPerToken <= 0 {
		return nil, 0, core.NewError("native.UnifiedAudioProject: model carries no unified audio head")
	}
	if len(samples) == 0 {
		return nil, 0, core.NewError("native.UnifiedAudioProject: empty audio")
	}
	spt := uv.Cfg.AudioSamplesPerToken
	n := (len(samples) + spt - 1) / spt
	rows := make([]byte, n*spt*bf16Size)
	for i, v := range samples {
		b := f32ToBF16(v)
		rows[2*i] = byte(b)
		rows[2*i+1] = byte(b >> 8)
	}
	// the pad tail stays zero — the reference zero-pads to divisibility
	unifiedRMSNoScale(rows, n, spt, uv.Cfg.RMSNormEps)
	out, err := unifiedLinearRows(uv.AudioProjection, rows, n)
	if err != nil {
		return nil, 0, core.E("native.UnifiedAudioProject", "embed_audio", err)
	}
	return out, n, nil
}

// DecodeWAVMono16k parses a RIFF/WAVE file into [-1,1] float32 samples for the
// unified audio head. v1 accepts 16-bit PCM, mono, 16 kHz — anything else
// errors loud (no silent resample).
func DecodeWAVMono16k(data []byte) ([]float32, error) {
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, core.NewError("native.DecodeWAVMono16k: not a RIFF/WAVE file")
	}
	le16 := func(off int) int { return int(data[off]) | int(data[off+1])<<8 }
	le32 := func(off int) int {
		return int(data[off]) | int(data[off+1])<<8 | int(data[off+2])<<16 | int(data[off+3])<<24
	}
	var fmtOK bool
	var dataOff, dataLen int
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := le32(off + 4)
		body := off + 8
		if body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, core.NewError("native.DecodeWAVMono16k: short fmt chunk")
			}
			format, channels, rate, bits := le16(body), le16(body+2), le32(body+4), le16(body+14)
			if format != 1 || bits != 16 {
				return nil, core.NewError("native.DecodeWAVMono16k: want 16-bit PCM (format 1)")
			}
			if channels != 1 {
				return nil, core.NewError("native.DecodeWAVMono16k: want mono audio")
			}
			if rate != 16000 {
				return nil, core.NewError(core.Sprintf("native.DecodeWAVMono16k: want 16000 Hz, got %d (resample before sending)", rate))
			}
			fmtOK = true
		case "data":
			dataOff, dataLen = body, size
		}
		off = body + size + (size & 1) // chunks are word-aligned
	}
	if !fmtOK || dataLen < 2 {
		return nil, core.NewError("native.DecodeWAVMono16k: missing fmt or data chunk")
	}
	n := dataLen / 2
	out := make([]float32, n)
	for i := range n {
		s := int16(uint16(data[dataOff+2*i]) | uint16(data[dataOff+2*i+1])<<8)
		out[i] = float32(s) / 32768
	}
	return out, nil
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
	unifiedVisionStats("patches", patches, patchDim)
	h, err := LayerNormBF16(patches, uv.PatchLN1W, uv.PatchLN1B, n, patchDim, cfg.LayerNormEps)
	if err != nil {
		return nil, core.E("native.UnifiedVisionProject", "patch_ln1", err)
	}
	unifiedVisionStats("ln1", h, patchDim)
	if h, err = unifiedLinearRows(uv.PatchDense, h, n); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "patch_dense", err)
	}
	unifiedVisionStats("dense", h, cfg.MMEmbedDim)
	if h, err = LayerNormBF16(h, uv.PatchLN2W, uv.PatchLN2B, n, cfg.MMEmbedDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "patch_ln2", err)
	}
	if err = unifiedAddPositions(h, positions, n, cfg.MMEmbedDim, cfg.PosembSize, uv.PosEmbedding); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "pos_embedding", err)
	}
	unifiedVisionStats("ln2+pos", h, cfg.MMEmbedDim)
	if h, err = LayerNormBF16(h, uv.PosNormW, uv.PosNormB, n, cfg.MMEmbedDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("native.UnifiedVisionProject", "pos_norm", err)
	}
	unifiedRMSNoScale(h, n, cfg.MMEmbedDim, cfg.RMSNormEps)
	out, err := unifiedLinearRows(uv.Projection, h, n)
	if err != nil {
		return nil, core.E("native.UnifiedVisionProject", "embedding_projection", err)
	}
	unifiedVisionStats("features", out, cfg.TextHidden)
	return out, nil
}
