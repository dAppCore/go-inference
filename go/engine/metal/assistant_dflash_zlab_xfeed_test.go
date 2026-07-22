// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	coreio "dappco.re/go/io"
)

// TestSpeculativeModel_DFlashZLab_TorchExactContextFeed is the capture-vs-
// forward bisect for the accept-rate gap: it runs OUR metal drafter forward
// (DFlashZLabForward + the borrowed target head — the exact ProposeBlock
// pipeline) on TORCH-EXACT context features dumped from the reference
// implementation (scratchpad dflash_dump_ctx.py: extract_context_feature
// over the same prompt, f32), alongside the same forward on our OWN live
// ExtractAuxHiddensAllRaw capture, and reports each lane's match depth
// against the reference drafter's own proposals for the identical frame.
//
//   - torch-exact lane matches the reference deeply, live lane does not
//     -> the gap is CAPTURE fidelity (±0.25% is too loose for this consumer);
//   - both lanes stop short at the same positions
//     -> the gap is the drafter forward's own numerics on metal.
//
// Diagnostic instrument, env-gated on the dump directory; skips without it.
func TestSpeculativeModel_DFlashZLab_TorchExactContextFeed(t *testing.T) {
	dumpDir := core.Getenv("LTHN_DFLASH_XFEED_DIR")
	draftDir := core.Getenv("LTHN_DFLASH_ZLAB_CKPT")
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(dumpDir) == "" || core.Trim(draftDir) == "" || core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_XFEED_DIR (dflash_dump_ctx.py output), LTHN_DFLASH_ZLAB_CKPT and LTHN_DFLASH_ZLAB_TARGET")
	}

	target, err := LoadDir(targetDir, 0)
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer func() { _ = target.Close() }()
	zlab, err := loadZLabDFlashDrafter(draftDir)
	if err != nil {
		t.Fatalf("load z-lab drafter: %v", err)
	}

	prompts := map[string]string{
		"colours": "The three primary colours are red, blue, and",
		"capital": "The capital of France is",
		"fox":     "The quick brown fox jumps over the lazy",
		"fib":     "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
	}

	hidden := zlab.model.Cfg.Hidden
	numAux := zlab.model.Cfg.NumAux()
	blockLen := zlab.BlockSize()

	propose := func(ctxRaw []float32, ctxLen int, anchor int32) []int {
		t.Helper()
		anchorEmbRaw, eerr := target.embedID(anchor)
		if eerr != nil {
			t.Fatalf("embed anchor: %v", eerr)
		}
		anchorEmb := append([]byte(nil), anchorEmbRaw...)
		maskEmbRaw, merr := target.embedID(zlab.MaskTokenID())
		if merr != nil {
			t.Fatalf("embed mask: %v", merr)
		}
		noise := make([]float32, blockLen*hidden)
		copy(noise[:hidden], bf16ToF32Slice(anchorEmb))
		maskF32 := bf16ToF32Slice(maskEmbRaw)
		for j := 1; j < blockLen; j++ {
			copy(noise[j*hidden:(j+1)*hidden], maskF32)
		}
		out, ferr := DFlashZLabForward(zlab.model, noise, ctxRaw, ctxLen, blockLen)
		if ferr != nil {
			t.Fatalf("DFlashZLabForward: %v", ferr)
		}
		block := make([]int, 0, blockLen-1)
		for j := 1; j < blockLen; j++ {
			row := out[j*hidden : (j+1)*hidden]
			logits, herr := target.headLogitsScratch(f32ToBf16Slice(row), false)
			if herr != nil {
				t.Fatalf("head: %v", herr)
			}
			id, serr := greedyBF16Suppressed(logits, target.arch.Vocab, nil)
			if serr != nil {
				t.Fatalf("argmax: %v", serr)
			}
			block = append(block, int(id))
		}
		return block
	}

	matchDepth := func(got, want []int) int {
		d := 0
		for i := range min(len(got), len(want)) {
			if got[i] != want[i] {
				break
			}
			d++
		}
		return d
	}

	// Embedding byte-parity probe: our embedID row vs torch's dumped
	// embed_tokens row for the capital anchor — a nonzero diff here means the
	// engine's weight bytes are NOT the checkpoint's (e.g. a quantised load),
	// which would explain proposal drift on modest-margin slots everywhere.
	if rowStr, rerr := coreio.Local.Read(core.PathJoin(dumpDir, "anchor12095.f32")); rerr == nil {
		raw := []byte(rowStr)
		ours, eerr := target.embedID(12095)
		if eerr != nil {
			t.Fatalf("embedID probe: %v", eerr)
		}
		oursF := bf16ToF32Slice(ours)
		var sumSq, refSq float64
		exact := 0
		for i := range min(len(oursF), len(raw)/4) {
			tv := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
			ov := float64(oursF[i])
			d := ov - tv
			sumSq += d * d
			refSq += tv * tv
			if ov == tv {
				exact++
			}
		}
		t.Logf("embed row 12095: relL2=%.6f exact-elements=%d/%d ours[0:4]=%v torch[0:4]=[%.6f %.6f %.6f %.6f]",
			math.Sqrt(sumSq/refSq), exact, len(oursF), oursF[:4],
			math.Float32frombits(binary.LittleEndian.Uint32(raw[0:])), math.Float32frombits(binary.LittleEndian.Uint32(raw[4:])),
			math.Float32frombits(binary.LittleEndian.Uint32(raw[8:])), math.Float32frombits(binary.LittleEndian.Uint32(raw[12:])))
	}

	for name, prompt := range prompts {
		metaStr, rerr := coreio.Local.Read(core.PathJoin(dumpDir, name+".meta"))
		if rerr != nil {
			t.Fatalf("%s: read meta: %v", name, rerr)
		}
		fields := core.Fields(core.Trim(metaStr))
		if len(fields) < 2+blockLen-1 {
			t.Fatalf("%s: meta too short: %q", name, metaStr)
		}
		ctxLen := int(core.ParseInt(fields[0], 10, 32).Value.(int64))
		anchor := int32(core.ParseInt(fields[1], 10, 32).Value.(int64))
		refProps := make([]int, 0, blockLen-1)
		for _, f := range fields[2 : 2+blockLen-1] {
			refProps = append(refProps, int(core.ParseInt(f, 10, 32).Value.(int64)))
		}

		rawStr, berr := coreio.Local.Read(core.PathJoin(dumpDir, name+".ctx.f32"))
		if berr != nil {
			t.Fatalf("%s: read ctx: %v", name, berr)
		}
		raw := []byte(rawStr)
		want := ctxLen * numAux * hidden * 4
		if len(raw) != want {
			t.Fatalf("%s: ctx dump is %d bytes, want %d (ctxLen=%d numAux=%d hidden=%d)", name, len(raw), want, ctxLen, numAux, hidden)
		}
		torchCtx := make([]float32, ctxLen*numAux*hidden)
		for i := range torchCtx {
			torchCtx[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}

		// Live lane: our own capture over the same frame (prompt + anchor,
		// trimmed to the prompt rows — ProposeBlock's convention).
		tokIDs := speculativeTokenizerEncode(t, targetDir, prompt)
		liveSeq := append(append([]int32(nil), tokIDs...), anchor)
		liveRaw, aerr := ExtractAuxHiddensAllRaw(target, liveSeq, zlab.AuxLayers())
		if aerr != nil {
			t.Fatalf("%s: live capture: %v", name, aerr)
		}
		liveCtxLen := len(liveSeq) - 1
		if liveCtxLen != ctxLen {
			t.Fatalf("%s: live ctxLen %d vs dump %d — tokenisation drift", name, liveCtxLen, ctxLen)
		}
		liveCtx := liveRaw[:liveCtxLen*numAux*hidden]

		// Fused-context stage check: our fc+hidden_norm on the torch-exact raw
		// context vs torch's own fused rows — localises a weight-mapping fault
		// in the fc before any attention runs.
		if fusedStr, ferr := coreio.Local.Read(core.PathJoin(dumpDir, name+".fused.f32")); ferr == nil {
			fusedRaw := []byte(fusedStr)
			if len(fusedRaw) == ctxLen*hidden*4 {
				oursFused := qwenVisionMatNT(torchCtx, zlab.model.FC, ctxLen, numAux*hidden, hidden)
				oursFused = dflashRMSNormRows(oursFused, ctxLen, hidden, zlab.model.HiddenNorm, zlab.model.Cfg.Eps)
				for r := range ctxLen {
					var sumSq, refSq float64
					for i := range hidden {
						tv := float64(math.Float32frombits(binary.LittleEndian.Uint32(fusedRaw[(r*hidden+i)*4:])))
						ov := float64(oursFused[r*hidden+i])
						d := ov - tv
						sumSq += d * d
						refSq += tv * tv
					}
					rel := 0.0
					if refSq > 0 {
						rel = math.Sqrt(sumSq / refSq)
					}
					t.Logf("%s: fused row %2d relL2=%.4f", name, r, rel)
				}
			}
		}

		// Per-layer bisect: run our decoder layers one at a time on the SAME
		// fused context and compare each layer's block hidden against torch's
		// hooked capture — the first layer that blows up owns the fault.
		if layersStr, lyerr := coreio.Local.Read(core.PathJoin(dumpDir, name+".layers.f32")); lyerr == nil {
			lraw := []byte(layersStr)
			nLayers := len(zlab.model.Layers)
			if len(lraw) == nLayers*blockLen*hidden*4 {
				fused := qwenVisionMatNT(torchCtx, zlab.model.FC, ctxLen, numAux*hidden, hidden)
				fused = dflashRMSNormRows(fused, ctxLen, hidden, zlab.model.HiddenNorm, zlab.model.Cfg.Eps)
				cosT, sinT := dflashRopeCosSin(ctxLen+blockLen, zlab.model.Cfg.HeadDim, zlab.model.Cfg.RopeTheta)
				anchorEmbRaw2, e2 := target.embedID(anchor)
				if e2 != nil {
					t.Fatalf("%s: embed anchor: %v", name, e2)
				}
				anchorEmbRaw2 = append([]byte(nil), anchorEmbRaw2...) // pin: shared scratch
				maskEmbRaw2, m2 := target.embedID(zlab.MaskTokenID())
				if m2 != nil {
					t.Fatalf("%s: embed mask: %v", name, m2)
				}
				h := make([]float32, blockLen*hidden)
				copy(h[:hidden], bf16ToF32Slice(anchorEmbRaw2))
				mf := bf16ToF32Slice(maskEmbRaw2)
				for j := 1; j < blockLen; j++ {
					copy(h[j*hidden:(j+1)*hidden], mf)
				}
				for li := range nLayers {
					var lerr error
					h, lerr = dflashDecoderLayer(&zlab.model.Layers[li], h, fused, zlab.model.Cfg, blockLen, ctxLen, cosT, sinT)
					if lerr != nil {
						t.Fatalf("%s: layer %d: %v", name, li, lerr)
					}
					var worst float64
					worstPos := -1
					for j := range blockLen {
						var sumSq, refSq float64
						for i := range hidden {
							tv := float64(math.Float32frombits(binary.LittleEndian.Uint32(lraw[((li*blockLen+j)*hidden+i)*4:])))
							ov := float64(h[j*hidden+i])
							d := ov - tv
							sumSq += d * d
							refSq += tv * tv
						}
						if refSq > 0 {
							if rel := math.Sqrt(sumSq / refSq); rel > worst {
								worst, worstPos = rel, j
							}
						}
					}
					t.Logf("%s: after layer %d worst relL2=%.4f at pos %d", name, li, worst, worstPos)
				}
			}
		}

		// Layer-0 input-norm isolation: OUR dflashRMSNormRows on the same
		// block embeddings vs torch's hooked input_layernorm output — a
		// mismatch here is a norm-weight mapping fault and explains the
		// full-layer divergence with a clean attention.
		if nStr, nerr := coreio.Local.Read(core.PathJoin(dumpDir, name+".l0norm.f32")); nerr == nil {
			nraw := []byte(nStr)
			if len(nraw) == blockLen*hidden*4 {
				anchorEmbRaw3, e3 := target.embedID(anchor)
				if e3 != nil {
					t.Fatalf("%s: embed anchor: %v", name, e3)
				}
				anchorEmbRaw3 = append([]byte(nil), anchorEmbRaw3...) // embedID hands back shared scratch — pin before the next embed
				maskEmbRaw3, m3 := target.embedID(zlab.MaskTokenID())
				if m3 != nil {
					t.Fatalf("%s: embed mask: %v", name, m3)
				}
				noise := make([]float32, blockLen*hidden)
				copy(noise[:hidden], bf16ToF32Slice(anchorEmbRaw3))
				mf := bf16ToF32Slice(maskEmbRaw3)
				for j := 1; j < blockLen; j++ {
					copy(noise[j*hidden:(j+1)*hidden], mf)
				}
				oursNorm := dflashRMSNormRows(noise, blockLen, hidden, zlab.model.Layers[0].InputNorm, zlab.model.Cfg.Eps)
				for j := range 3 {
					var sumSq, refSq float64
					for i := range hidden {
						tv := float64(math.Float32frombits(binary.LittleEndian.Uint32(nraw[(j*hidden+i)*4:])))
						ov := float64(oursNorm[j*hidden+i])
						d := ov - tv
						sumSq += d * d
						refSq += tv * tv
					}
					rel := 0.0
					if refSq > 0 {
						rel = math.Sqrt(sumSq / refSq)
					}
					t.Logf("%s: l0-NORM pos %2d relL2=%.4f", name, j, rel)
				}
			}
		}
		if nStr, nerr := coreio.Local.Read(core.PathJoin(dumpDir, name+".l0norm.f32")); nerr == nil {
			aStr, aerr := coreio.Local.Read(core.PathJoin(dumpDir, name+".l0attn.f32"))
			if aerr == nil {
				nraw, araw := []byte(nStr), []byte(aStr)
				if len(nraw) == blockLen*hidden*4 && len(araw) == blockLen*hidden*4 {
					normed := make([]float32, blockLen*hidden)
					for i := range normed {
						normed[i] = math.Float32frombits(binary.LittleEndian.Uint32(nraw[i*4:]))
					}
					fused := qwenVisionMatNT(torchCtx, zlab.model.FC, ctxLen, numAux*hidden, hidden)
					fused = dflashRMSNormRows(fused, ctxLen, hidden, zlab.model.HiddenNorm, zlab.model.Cfg.Eps)
					cosT, sinT := dflashRopeCosSin(ctxLen+blockLen, zlab.model.Cfg.HeadDim, zlab.model.Cfg.RopeTheta)
					attnOurs, aterr := dflashAttention(&zlab.model.Layers[0], normed, fused, zlab.model.Cfg, blockLen, ctxLen, cosT, sinT)
					if aterr != nil {
						t.Fatalf("%s: l0 attention: %v", name, aterr)
					}
					for j := range blockLen {
						var sumSq, refSq float64
						for i := range hidden {
							tv := float64(math.Float32frombits(binary.LittleEndian.Uint32(araw[(j*hidden+i)*4:])))
							ov := float64(attnOurs[j*hidden+i])
							d := ov - tv
							sumSq += d * d
							refSq += tv * tv
						}
						rel := 0.0
						if refSq > 0 {
							rel = math.Sqrt(sumSq / refSq)
						}
						if j < 4 || rel > 0.05 {
							t.Logf("%s: l0-attn pos %2d relL2=%.4f", name, j, rel)
						}
					}
				}
			}
		}

		torchProps := propose(torchCtx, ctxLen, anchor)
		liveProps := propose(liveCtx, liveCtxLen, anchor)
		t.Logf("%s: ref=%v", name, refProps)
		t.Logf("%s: torch-exact-fed depth=%d props=%v", name, matchDepth(torchProps, refProps), torchProps)
		t.Logf("%s: live-capture-fed depth=%d props=%v", name, matchDepth(liveProps, refProps), liveProps)

		// Raw hidden-state deviation, our forward vs torch's, same torch-exact
		// inputs: uniform small error = accumulation noise; a localised
		// blow-up = structural.
		houtStr, herr := coreio.Local.Read(core.PathJoin(dumpDir, name+".hout.f32"))
		if herr != nil {
			t.Logf("%s: no hout dump (%v) — skipping row diff", name, herr)
			continue
		}
		hraw := []byte(houtStr)
		if len(hraw) != blockLen*hidden*4 {
			t.Fatalf("%s: hout is %d bytes, want %d", name, len(hraw), blockLen*hidden*4)
		}
		anchorEmbRaw, eerr := target.embedID(anchor)
		if eerr != nil {
			t.Fatalf("%s: embed anchor: %v", name, eerr)
		}
		anchorEmbRaw = append([]byte(nil), anchorEmbRaw...) // pin: shared scratch
		maskEmbRaw, merr := target.embedID(zlab.MaskTokenID())
		if merr != nil {
			t.Fatalf("%s: embed mask: %v", name, merr)
		}
		noise := make([]float32, blockLen*hidden)
		copy(noise[:hidden], bf16ToF32Slice(anchorEmbRaw))
		maskF32 := bf16ToF32Slice(maskEmbRaw)
		for j := 1; j < blockLen; j++ {
			copy(noise[j*hidden:(j+1)*hidden], maskF32)
		}
		ours, ferr := DFlashZLabForward(zlab.model, noise, torchCtx, ctxLen, blockLen)
		if ferr != nil {
			t.Fatalf("%s: forward for row diff: %v", name, ferr)
		}
		for j := range blockLen {
			var sumSq, refSq, maxAbs float64
			for i := range hidden {
				tv := float64(math.Float32frombits(binary.LittleEndian.Uint32(hraw[(j*hidden+i)*4:])))
				ov := float64(ours[j*hidden+i])
				d := ov - tv
				sumSq += d * d
				refSq += tv * tv
				if a := math.Abs(d); a > maxAbs {
					maxAbs = a
				}
			}
			rel := 0.0
			if refSq > 0 {
				rel = math.Sqrt(sumSq / refSq)
			}
			t.Logf("%s: pos %2d relL2=%.4f maxAbs=%.4f", name, j, rel, maxAbs)
		}
	}
}

// speculativeTokenizerEncode loads the target's tokenizer and encodes prompt —
// a tiny helper so the xfeed test tokenises exactly as the survey does.
func speculativeTokenizerEncode(t *testing.T, targetDir, prompt string) []int32 {
	t.Helper()
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetDir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	return tok.Encode(prompt)
}
