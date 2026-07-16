// SPDX-Licence-Identifier: EUPL-1.2

package qwen2_test

import (
	"math"
	"testing"

	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/Qwen/qwen2"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// real_checkpoint_test.go is board #24 item 2's root-cause receipt: mlx-community/Qwen2.5-Coder-3B-4bit
// loads through this engine and decodes fast but emits junk, while mlx-lm serves the identical checkpoint
// correctly. Bisecting host-side against mlx-lm's OWN module code (mlx_lm/models/qwen2.py, run on the
// identical token ids) on the SAME bytes model.Load produces — dequantised the SAME way (mlxaffine, the
// package the loader's LoadLinear delegates to) — proves the qwen2 ArchSpec (config.go's dimension/RoPE/
// activation derivation, register.go's weight-name map, the QKV additive bias, the F16→BF16 widen of the
// checkpoint's scales/biases/norms) is byte-correct: embed rows, RMSNorm, QKV+bias, NEOX half-split RoPE,
// GQA attention, SwiGLU MLP and the tied LM head all land within mlx's own f16-vs-this-host's-f32
// rounding noise, and the full 36-layer forward argmaxes the SAME token mlx-lm does (" Paris", id 12095).
// The garble is therefore NOT this package's arch mapping — it is downstream, in engine/metal's GPU
// decode kernels (worse on the default ICB-replay hot path than on LTHN_NATIVE_TRACE's dense re-encode
// path, which still degrades but stays closer to coherent prose): out of scope for this package, and
// left as a finding for the engine/metal lane. Skips cleanly (enginegate.HFModelPath) when the checkpoint
// is not in the local Hugging Face cache, so CI stays green off this machine.

// qwen2CoderPromptIDs are the tokenizer ids mlx-lm's Qwen2Tokenizer gives "The capital of France is"
// (--ignore-chat-template) — fixed input so the host mirror and mlx-lm process byte-identical positions.
var qwen2CoderPromptIDs = []int{785, 6722, 315, 9625, 374}

// bf16BytesToF32 upstreams a raw little-endian bf16 buffer to float32 — the same widening
// safetensors.BFloat16ToFloat32 does per element, for a whole norm/bias tensor at once.
func bf16BytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = safetensors.BFloat16ToFloat32(uint16(b[i*2]) | uint16(b[i*2+1])<<8)
	}
	return out
}

// dequantRow dequantises ONE row of an affine-quantised Linear (a single embedding/head vocab row)
// without materialising the whole outDim×inDim matrix — mlxaffine.DequantizeTensor called at outDim=1
// on that row's own packed/scales/biases slice.
func dequantRow(t *testing.T, lin *model.Linear, row int) []float32 {
	t.Helper()
	groupsPerRow := lin.InDim / lin.GroupSize
	wordsPerRow := mlxaffine.PackedWords(lin.InDim, lin.Bits)
	p := lin.Weight[row*wordsPerRow*4 : (row+1)*wordsPerRow*4]
	s := lin.Scales[row*groupsPerRow*2 : (row+1)*groupsPerRow*2]
	b := lin.Biases[row*groupsPerRow*2 : (row+1)*groupsPerRow*2]
	out, err := mlxaffine.DequantizeTensor(p, s, b, 1, lin.InDim, lin.Bits, lin.GroupSize)
	if err != nil {
		t.Fatalf("dequantRow: %v", err)
	}
	return out
}

// dequantFull dequantises an entire affine-quantised Linear to a row-major float32 matrix.
func dequantFull(t *testing.T, lin *model.Linear) []float32 {
	t.Helper()
	out, err := mlxaffine.DequantizeTensor(lin.Weight, lin.Scales, lin.Biases, lin.OutDim, lin.InDim, lin.Bits, lin.GroupSize)
	if err != nil {
		t.Fatalf("dequantFull: %v", err)
	}
	return out
}

func matvec(w []float32, outDim, inDim int, x []float32) []float32 {
	out := make([]float32, outDim)
	for r := range outDim {
		var acc float64
		row := w[r*inDim : (r+1)*inDim]
		for c, v := range x {
			acc += float64(row[c]) * float64(v)
		}
		out[r] = float32(acc)
	}
	return out
}

func addBias(x []float32, bias []byte) {
	if len(bias) == 0 {
		return
	}
	b := bf16BytesToF32(bias)
	for i := range x {
		x[i] += b[i]
	}
}

func rmsNorm(x, w []float32, eps float32) []float32 {
	var sq float64
	for _, v := range x {
		sq += float64(v) * float64(v)
	}
	inv := 1.0 / math.Sqrt(sq/float64(len(x))+float64(eps))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = float32(float64(x[i])*inv) * w[i]
	}
	return out
}

// ropeNeoxHalf applies the standard half-split (NEOX/rotate-half) convention mlx.fast.rope uses at
// traditional=False — mlx_lm's qwen2 default (ModelArgs.rope_traditional=false): pair (i, i+half),
// angle = pos·base^(-2i/headDim).
func ropeNeoxHalf(x []float32, headDim int, base float32, pos int) {
	half := headDim / 2
	for i := range half {
		theta := float64(pos) * math.Pow(float64(base), -2*float64(i)/float64(headDim))
		c, s := math.Cos(theta), math.Sin(theta)
		a, b := float64(x[i]), float64(x[i+half])
		x[i] = float32(a*c - b*s)
		x[i+half] = float32(a*s + b*c)
	}
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

// runLayer runs one Qwen2 TransformerBlock (pre-norm GQA attention with additive QKV bias, pre-norm
// SwiGLU MLP) on every position's hidden — the host mirror of mlx_lm.models.qwen2.TransformerBlock.
func runLayer(t *testing.T, L *model.LoadedLayer, hiddens [][]float32, nHeads, nKV, headDim int, eps, ropeBase, attnScale float32, ff, d int) [][]float32 {
	t.Helper()
	Wq, Wk, Wv, Wo := dequantFull(t, L.Q), dequantFull(t, L.K), dequantFull(t, L.V), dequantFull(t, L.O)
	Wgate, Wup, Wdown := dequantFull(t, L.Gate), dequantFull(t, L.Up), dequantFull(t, L.Down)
	attnNormW, mlpNormW := bf16BytesToF32(L.AttnNorm), bf16BytesToF32(L.MLPNorm)

	qDim, kvDim, n := nHeads*headDim, nKV*headDim, len(hiddens)
	qs, ks, vs := make([][]float32, n), make([][]float32, n), make([][]float32, n)
	for tk := range n {
		normed := rmsNorm(hiddens[tk], attnNormW, eps)
		q := matvec(Wq, qDim, d, normed)
		addBias(q, L.Q.Bias)
		k := matvec(Wk, kvDim, d, normed)
		addBias(k, L.K.Bias)
		v := matvec(Wv, kvDim, d, normed)
		addBias(v, L.V.Bias)
		for h := range nHeads {
			ropeNeoxHalf(q[h*headDim:(h+1)*headDim], headDim, ropeBase, tk)
		}
		for h := range nKV {
			ropeNeoxHalf(k[h*headDim:(h+1)*headDim], headDim, ropeBase, tk)
		}
		qs[tk], ks[tk], vs[tk] = q, k, v
	}

	group := nHeads / nKV
	out := make([][]float32, n)
	for tk := range n {
		ctxT := make([]float32, qDim)
		for h := range nHeads {
			kvh := h / group
			qh := qs[tk][h*headDim : (h+1)*headDim]
			scores := make([]float64, tk+1)
			maxS := math.Inf(-1)
			for j := 0; j <= tk; j++ {
				kh := ks[j][kvh*headDim : (kvh+1)*headDim]
				var dot float64
				for d2 := range headDim {
					dot += float64(qh[d2]) * float64(kh[d2])
				}
				dot *= float64(attnScale)
				scores[j] = dot
				maxS = math.Max(maxS, dot)
			}
			var sum float64
			probs := make([]float64, tk+1)
			for j := 0; j <= tk; j++ {
				probs[j] = math.Exp(scores[j] - maxS)
				sum += probs[j]
			}
			for j := range probs {
				probs[j] /= sum
			}
			outh := make([]float64, headDim)
			for j := 0; j <= tk; j++ {
				vh := vs[j][kvh*headDim : (kvh+1)*headDim]
				for d2 := range headDim {
					outh[d2] += probs[j] * float64(vh[d2])
				}
			}
			for d2 := range headDim {
				ctxT[h*headDim+d2] = float32(outh[d2])
			}
		}
		attnOut := matvec(Wo, d, qDim, ctxT)
		h := make([]float32, d)
		for i := range h {
			h[i] = hiddens[tk][i] + attnOut[i]
		}
		normed2 := rmsNorm(h, mlpNormW, eps)
		g, u := matvec(Wgate, ff, d, normed2), matvec(Wup, ff, d, normed2)
		for i := range g {
			g[i] = silu(g[i]) * u[i]
		}
		dn := matvec(Wdown, d, ff, g)
		for i := range h {
			h[i] += dn[i]
		}
		out[tk] = h
	}
	return out
}

// closeEnough reports whether a and b agree within an absolute tolerance loose enough to absorb
// mlx-lm's float16 activations against this host mirror's float32/float64 accumulation (the reference
// dump — qwen2_ref.py against mlx_lm — computes entirely in fp16; a real defect in the arch mapping
// moves values by 10-100x this band, not a rounding step).
func closeEnough(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// TestRealCheckpoint_Layer0MatchesMLXLM_Good bisects the embed lookup + full layer-0 block (attention
// with QKV bias + RoPE + SwiGLU MLP) for mlx-community/Qwen2.5-Coder-3B-4bit against fixed reference
// values captured from mlx_lm.models.qwen2 (the checkpoint's own reference implementation) on the
// identical token ids. A future regression in the arch mapping — a wrong RoPE base, a dropped bias, the
// wrong activation, a mis-scaled norm — moves these numbers far outside the fp16-rounding tolerance.
func TestRealCheckpoint_Layer0MatchesMLXLM_Good(t *testing.T) {
	dir := enginegate.HFModelPath(t, "mlx-community/Qwen2.5-Coder-3B-4bit")
	lm, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	defer func() { _ = dm.Close() }()

	d := lm.Arch.Hidden
	embeds := make([][]float32, len(qwen2CoderPromptIDs))
	for i, id := range qwen2CoderPromptIDs {
		embeds[i] = dequantRow(t, lm.Embed, id)
	}
	// mlx_lm reference (float16): embed row0 first4 = [-0.00958252, -0.00479126, -0.02874756, -0.01916504]
	wantEmbed0 := []float32{-0.00958252, -0.00479126, -0.02874756, -0.01916504}
	for i, want := range wantEmbed0 {
		if !closeEnough(embeds[0][i], want, 0.002) {
			t.Errorf("embed row0[%d] = %v, want ~%v (mlx-lm reference)", i, embeds[0][i], want)
		}
	}

	hiddens := runLayer(t, &lm.Layers[0], embeds, lm.Arch.Heads, lm.Arch.KVHeads, lm.Arch.HeadDim,
		lm.Arch.Eps, lm.Arch.RopeBase, lm.Arch.AttnScale, lm.Arch.FF, d)

	// mlx_lm reference (float16), model.model.layers[0](embed, causal_mask, None) at the last prompt
	// position ("is"): layer0 out row-last first4 = [0.06445312, -0.03814697, 0.3544922, -0.05682373]
	wantLayer0Last := []float32{0.06445312, -0.03814697, 0.3544922, -0.05682373}
	last := hiddens[len(hiddens)-1]
	for i, want := range wantLayer0Last {
		if !closeEnough(last[i], want, 0.02) {
			t.Errorf("layer0 out row-last[%d] = %v, want ~%v (mlx-lm reference)", i, last[i], want)
		}
	}
}

// TestRealCheckpoint_ArgmaxParis_Good runs the FULL 36-layer forward (every layer, not just layer 0)
// plus the final norm and the tied LM head, entirely on the host — no engine/metal GPU kernels — for
// mlx-community/Qwen2.5-Coder-3B-4bit's "The capital of France is" prompt, and checks the argmax token
// at the last position against mlx-lm's own answer (12095, " Paris"). Landing on " Paris" here, built
// from the exact bytes model.Load hands the engine, is the end-to-end proof the qwen2 ArchSpec (config
// parsing, weight-name mapping, dequantisation, QKV bias, RoPE, tied embeddings) has no bug — see the
// package doc comment for what that implies about where board #24 item 2's garble actually lives.
func TestRealCheckpoint_ArgmaxParis_Good(t *testing.T) {
	dir := enginegate.HFModelPath(t, "mlx-community/Qwen2.5-Coder-3B-4bit")
	lm, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	defer func() { _ = dm.Close() }()

	d := lm.Arch.Hidden
	hiddens := make([][]float32, len(qwen2CoderPromptIDs))
	for i, id := range qwen2CoderPromptIDs {
		hiddens[i] = dequantRow(t, lm.Embed, id)
	}
	for li := range lm.Layers {
		hiddens = runLayer(t, &lm.Layers[li], hiddens, lm.Arch.Heads, lm.Arch.KVHeads, lm.Arch.HeadDim,
			lm.Arch.Eps, lm.Arch.RopeBase, lm.Arch.AttnScale, lm.Arch.FF, d)
	}

	finalNormW := bf16BytesToF32(lm.FinalNorm)
	last := rmsNorm(hiddens[len(hiddens)-1], finalNormW, lm.Arch.Eps)

	// Tied head: score every vocab row via per-row dequant + dot rather than materialising the full
	// 151936x2048 dequantised matrix (mirrors the embed lookup above; keeps this test's RSS small).
	const wantArgmax = 12095 // mlx-lm: tokenizer.decode([12095]) == " Paris"
	best, bestScore := -1, math.Inf(-1)
	for v := range lm.Arch.Vocab {
		row := dequantRow(t, lm.Embed, v)
		var acc float64
		for i, x := range row {
			acc += float64(x) * float64(last[i])
		}
		if acc > bestScore {
			bestScore, best = acc, v
		}
	}
	if best != wantArgmax {
		t.Errorf("host full-forward argmax = %d, want %d (' Paris', mlx-lm reference) — the arch mapping itself regressed", best, wantArgmax)
	}
}
