// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// assistant_dflash_test.go proves the block-parallel DFlash draft forward two ways,
// with a tiny synthetic drafter built from varied (never constant — the vacuous-fixture
// trap) seeded weights, the r2-r5 metal-fixture pattern:
//
//   - PARITY: the metal forward's block proposals match a host reference of the SAME
//     maths — a pure-Go forward that mirrors every GPU op with the engine's own bf16
//     rounding (f32 accumulate, round result to bf16, exactly what MatVecBF16 / RMSNorm
//     / SDPA do). Since no public gemma-4 DFlash checkpoint exists, THIS is the honest
//     receipt: the dispatch is correct for the forward the memo designed, argmax tokens
//     agree and logits track to bf16 tolerance. No real-model accept-rate is claimed.
//   - LOSSLESS ENGAGEMENT: the metal forward, wired behind decode/dflash.BlockProposer
//     and driven through dflash.Generate (the fuzz-proven verify driver) against an
//     independent target oracle, fires real proposals AND yields output byte-identical
//     to plain decode (== dflash.Autoregress) — the counter-guarded invariant.
//   - LOADER: a DFlash pack on disk loads through the reactive assistant loader
//     (LoadDFlashDrafter → LoadAssistantDir → the registered gemma4 dflash spec) and
//     proposes a block, so the pack/loader path is exercised end to end.

const (
	dflBackbone = 8  // target hidden the verifier hiddens + anchor embedding live in
	dflHidden   = 16 // drafter hidden
	dflHeads    = 2  // drafter attention heads
	dflKV       = 1  // drafter kv heads
	dflHeadDim  = 64 // drafter head dim (full rotary; a metallib-backed SDPA size)
	dflFF       = 16 // drafter feed-forward width
	dflVocab    = 8  // drafter model vocab (embed_tokens; unused by the forward)
	dflDraft    = 6  // reduced draft vocab (dflash.lm_head rows)
	dflTarget   = 8  // target vocab the d2t map lands in
	dflNumAux   = 2  // fused verifier layers (aux_hidden_state_layer_ids)
	dflBlock    = 3  // γ — tokens proposed per block
	dflAnchor   = 5  // anchor token position (drives the block's rope offsets)
	dflEps      = float32(1e-6)
	dflScale    = float32(0.5)   // 1/sqrt(headDim)
	dflRope     = float32(10000) // rope theta
)

// dflashSyntheticDrafter builds the reusable synthetic DFlash drafter (arch inline, so
// scale/rope/eps are pinned for the reference) plus the fused-context inputs. The
// weights are varied (nativeAssistantProjectionFixture palette + syntheticFloat32), so
// no op collapses to a vacuous constant.
func dflashSyntheticDrafter(t testing.TB) (*DFlashDrafter, [][]byte, []byte) {
	t.Helper()
	arch := model.Arch{
		Hidden: dflHidden, Heads: dflHeads, KVHeads: dflKV, HeadDim: dflHeadDim,
		FF: dflFF, Vocab: dflVocab, Eps: dflEps, AttnScale: dflScale,
		RopeBase: dflRope, RopeScale: 1, RotaryDim: dflHeadDim,
		Layer: model.DeriveLayers([]string{"full_attention"}, 0),
	}
	m := &AssistantModel{
		Config: model.AssistantConfig{
			ModelType: "gemma4_dflash_assistant", Method: model.MTPDFlash,
			BackboneHidden: dflBackbone, Arch: arch, LayerTypes: []string{"full_attention"},
		},
		Arch:               arch,
		Tensors:            dflashSyntheticTensors(),
		BackboneHiddenSize: dflBackbone,
	}
	cfg := dflash.Config{BlockSize: dflBlock, AuxHiddenLayerIDs: []int{0, 1}}
	drafter, err := newDFlashDrafter(m, cfg)
	if err != nil {
		t.Fatalf("newDFlashDrafter: %v", err)
	}
	aux := [][]byte{
		toBF16Bytes(syntheticFloat32(dflBackbone, 201)),
		toBF16Bytes(syntheticFloat32(dflBackbone, 202)),
	}
	anchor := toBF16Bytes(syntheticFloat32(dflBackbone, 203))
	return drafter, aux, anchor
}

// dflashSyntheticTensors is the drafter's full bf16 tensor set: the MTP -assistant
// layout the ordinary loader validates, plus the DFlash extras (per-layer k_proj/v_proj
// injection weights, aux_projection, lm_head, d2t).
func dflashSyntheticTensors() map[string]safetensors.Tensor {
	mat := func(out, in int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: toBF16Bytes(nativeAssistantProjectionFixture(out, in))}
	}
	vec := func(n, salt int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: toBF16Bytes(syntheticFloat32(n, salt))}
	}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": mat(dflVocab, dflHidden),
		"model.norm.weight":         vec(dflHidden, 11),
		"pre_projection.weight":     mat(dflHidden, 2*dflBackbone),
		"post_projection.weight":    mat(dflBackbone, dflHidden),
		// DFlash extras
		dflashAuxProjectionWeight: mat(dflBackbone, dflNumAux*dflBackbone),
		dflashLMHeadWeight:        mat(dflDraft, dflHidden),
		dflashD2TTensor:           {Dtype: "BF16", Shape: []int{dflDraft}, Data: toBF16Bytes([]float32{0, 2, 4, 6, 1, 3})},
	}
	p := "model.layers.0"
	tensors[p+".input_layernorm.weight"] = vec(dflHidden, 21)
	tensors[p+".post_attention_layernorm.weight"] = vec(dflHidden, 23)
	tensors[p+".pre_feedforward_layernorm.weight"] = vec(dflHidden, 27)
	tensors[p+".post_feedforward_layernorm.weight"] = vec(dflHidden, 29)
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{1})}
	tensors[p+".self_attn.q_proj.weight"] = mat(dflHeads*dflHeadDim, dflHidden)
	tensors[p+".self_attn.q_norm.weight"] = vec(dflHeadDim, 31)
	tensors[p+".self_attn.o_proj.weight"] = mat(dflHidden, dflHeads*dflHeadDim)
	tensors[p+".self_attn.k_proj.weight"] = mat(dflKV*dflHeadDim, dflBackbone)
	tensors[p+".self_attn.v_proj.weight"] = mat(dflKV*dflHeadDim, dflBackbone)
	tensors[p+".mlp.gate_proj.weight"] = mat(dflFF, dflHidden)
	tensors[p+".mlp.up_proj.weight"] = mat(dflFF, dflHidden)
	tensors[p+".mlp.down_proj.weight"] = mat(dflHidden, dflFF)
	return tensors
}

// TestDFlashProposeBlockParity gates the metal forward against the bf16 host reference:
// the proposed target tokens must agree exactly, and the pre-argmax draft logits must
// track to bf16 tolerance — the r2-r5 "metal == host maths" receipt.
func TestDFlashProposeBlockParity(t *testing.T) {
	requireNativeRuntime(t)
	drafter, aux, anchor := dflashSyntheticDrafter(t)

	got, err := drafter.ProposeBlock(aux, anchor, dflAnchor)
	if err != nil {
		t.Fatalf("ProposeBlock: %v", err)
	}
	wantTokens, wantLogits := dflashHostReference(aux, anchor, dflAnchor)
	if len(got) != len(wantTokens) {
		t.Fatalf("proposed %d tokens, reference %d", len(got), len(wantTokens))
	}
	for j, tok := range got {
		if tok != wantTokens[j] {
			t.Fatalf("block[%d] = %d, host reference %d (proposals must match)", j, tok, wantTokens[j])
		}
		if tok < 0 || int(tok) >= dflTarget {
			t.Fatalf("block[%d] = %d outside target vocab %d", j, tok, dflTarget)
		}
	}
	// Secondary: the metal draft logits track the reference within bf16 tolerance,
	// so a lucky argmax tie is not masking a diverged forward.
	metalLogits := dflashMetalDraftLogits(t, drafter, aux, anchor, dflAnchor)
	for j := range wantLogits {
		for k := range wantLogits[j] {
			diff := float32(math.Abs(float64(metalLogits[j][k] - wantLogits[j][k])))
			tol := 0.02 * (1 + float32(math.Abs(float64(wantLogits[j][k]))))
			if diff > tol {
				t.Fatalf("block[%d] logit[%d] metal %.5f vs host %.5f (diff %.5f > tol %.5f)", j, k, metalLogits[j][k], wantLogits[j][k], diff, tol)
			}
		}
	}
}

// TestDFlashProposeBlockDeterministic pins that the forward is a pure function of its
// inputs — the same fused context proposes the same block every call.
func TestDFlashProposeBlockDeterministic(t *testing.T) {
	requireNativeRuntime(t)
	drafter, aux, anchor := dflashSyntheticDrafter(t)
	a, err := drafter.ProposeBlock(aux, anchor, dflAnchor)
	if err != nil {
		t.Fatalf("ProposeBlock a: %v", err)
	}
	b, err := drafter.ProposeBlock(aux, anchor, dflAnchor)
	if err != nil {
		t.Fatalf("ProposeBlock b: %v", err)
	}
	if len(a) != dflBlock {
		t.Fatalf("block length %d, want %d", len(a), dflBlock)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic block[%d]: %d then %d", i, a[i], b[i])
		}
	}
}

// TestDFlashProposeBlockLosslessEngagement wires the metal forward behind
// decode/dflash.BlockProposer and drives it through the verify driver (dflash.Generate)
// against an INDEPENDENT target oracle: the proposer fires real blocks and the emitted
// sequence is byte-identical to plain decode — DFlash's losslessness, end to end with
// a real engine proposer rather than the model-free stand-in.
func TestDFlashProposeBlockLosslessEngagement(t *testing.T) {
	requireNativeRuntime(t)
	drafter, aux, anchor := dflashSyntheticDrafter(t)

	// The target's greedy oracle — deterministic, independent of the drafter.
	next := func(prefix []int) int {
		if len(prefix) == 0 {
			return 0
		}
		return ((prefix[len(prefix)-1]*7 + 3) ^ len(prefix)) % dflTarget
	}
	proposer := NewDFlashProposer(drafter, func(ctx []int) ([][]byte, []byte, int, bool) {
		return aux, anchor, len(ctx) - 1, true
	})

	prompt := []int{1, 2, 3}
	const maxTokens = 24
	out, stats := dflash.Generate(prompt, maxTokens, proposer, next)
	base := dflash.Autoregress(prompt, maxTokens, next)

	if len(out) != len(base) {
		t.Fatalf("speculative output %d tokens, plain %d", len(out), len(base))
	}
	for i := range base {
		if out[i] != base[i] {
			t.Fatalf("LOSSLESS VIOLATED at %d: speculative %d, plain %d", i, out[i], base[i])
		}
	}
	if stats.ProposedTokens == 0 {
		t.Fatal("DFlash proposer never fired (no draft tokens offered)")
	}
	if stats.Rounds == 0 {
		t.Fatal("verify driver ran no speculative rounds")
	}
}

// TestLoadDFlashDrafter exercises the reactive pack loader: a DFlash checkpoint written
// to disk loads through LoadDFlashDrafter (config marker + registered spec + safetensors)
// and proposes a block — the loader is not hand-rolled.
func TestLoadDFlashDrafter(t *testing.T) {
	requireNativeRuntime(t)
	dir := writeDFlashDrafterDir(t)
	drafter, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer drafter.m.Close()
	if drafter.BlockSize() != dflBlock {
		t.Fatalf("loaded block size %d, want %d", drafter.BlockSize(), dflBlock)
	}
	if drafter.NumAux() != dflNumAux {
		t.Fatalf("loaded numAux %d, want %d", drafter.NumAux(), dflNumAux)
	}
	aux := [][]byte{
		toBF16Bytes(syntheticFloat32(dflBackbone, 201)),
		toBF16Bytes(syntheticFloat32(dflBackbone, 202)),
	}
	block, err := drafter.ProposeBlock(aux, toBF16Bytes(syntheticFloat32(dflBackbone, 203)), dflAnchor)
	if err != nil {
		t.Fatalf("loaded ProposeBlock: %v", err)
	}
	if len(block) != dflBlock {
		t.Fatalf("loaded drafter proposed %d tokens, want %d", len(block), dflBlock)
	}
}

// --- bf16 host reference: mirrors ProposeBlock op-for-op, rounding every result to
// --- bf16 exactly as the GPU primitives do (f32 accumulate → bf16 output). ---

func dflashHostReference(auxB [][]byte, anchorB []byte, anchorPos int) (tokens []int32, logits [][]float32) {
	d2t := []float32{0, 2, 4, 6, 1, 3}
	aux := make([][]float32, len(auxB))
	for i, a := range auxB {
		aux[i] = bf16Floats(a)
	}
	anchor := bf16Floats(anchorB)

	// fuse verifier hiddens → backbone context, then seed through pre_projection.
	var auxCat []float32
	for _, a := range aux {
		auxCat = append(auxCat, a...)
	}
	auxContext := dflashMatVecRef("dflash.aux_projection.weight", auxCat, dflBackbone, dflNumAux*dflBackbone)
	combined := append(append([]float32(nil), anchor...), auxContext...)
	seed := dflashMatVecRef("pre_projection.weight", combined, dflHidden, 2*dflBackbone)

	// injected K/V (numAux rows) per (single) layer.
	kInj, vInj := dflashInjectedRef(aux)

	for j := 0; j < dflBlock; j++ {
		h := dflashLayerRef(seed, kInj, vInj, anchorPos+j)
		normed := dflashRMSNormRef(h, dflashVecRef("model.norm.weight"), 1, dflHidden)
		row := dflashMatVecRef(dflashLMHeadWeight, normed, dflDraft, dflHidden)
		logits = append(logits, append([]float32(nil), row...))
		draftID := dflashArgmax(row)
		tokens = append(tokens, int32(d2t[draftID]))
	}
	return tokens, logits
}

// dflashInjectedRef projects each verifier hidden through the layer's k_proj/v_proj into
// the injected K/V rows (numAux rows of kvHeads*headDim), the reference of injectedKV.
func dflashInjectedRef(aux [][]float32) (k, v []float32) {
	row := dflKV * dflHeadDim
	for _, a := range aux {
		k = append(k, dflashMatVecRef("model.layers.0.self_attn.k_proj.weight", a, row, dflBackbone)...)
		v = append(v, dflashMatVecRef("model.layers.0.self_attn.v_proj.weight", a, row, dflBackbone)...)
	}
	return k, v
}

// dflashLayerRef mirrors draftLayerIntoScratch: the gemma4 sandwich-norm layer with the
// injected K/V cross-attention and rope at qPos.
func dflashLayerRef(h, kInj, vInj []float32, qPos int) []float32 {
	normed := dflashRMSNormRef(h, dflashVecRef("model.layers.0.input_layernorm.weight"), 1, dflHidden)
	// attention
	q := dflashMatVecRef("model.layers.0.self_attn.q_proj.weight", normed, dflHeads*dflHeadDim, dflHidden)
	q = dflashRMSNormRef(q, dflashVecRef("model.layers.0.self_attn.q_norm.weight"), dflHeads, dflHeadDim)
	q = dflashRopeRef(q, qPos)
	attn := dflashSDPARef(q, kInj, vInj)
	attnOut := dflashMatVecRef("model.layers.0.self_attn.o_proj.weight", attn, dflHidden, dflHeads*dflHeadDim)
	attnResid := dflashRMSNormRef(attnOut, dflashVecRef("model.layers.0.post_attention_layernorm.weight"), 1, dflHidden)
	h1 := dflashAddRef(h, attnResid)
	// mlp
	ffIn := dflashRMSNormRef(h1, dflashVecRef("model.layers.0.pre_feedforward_layernorm.weight"), 1, dflHidden)
	gate := dflashMatVecRef("model.layers.0.mlp.gate_proj.weight", ffIn, dflFF, dflHidden)
	up := dflashMatVecRef("model.layers.0.mlp.up_proj.weight", ffIn, dflFF, dflHidden)
	gated := dflashGeluGateMulRef(gate, up)
	ff := dflashMatVecRef("model.layers.0.mlp.down_proj.weight", gated, dflHidden, dflFF)
	ffResid := dflashRMSNormRef(ff, dflashVecRef("model.layers.0.post_feedforward_layernorm.weight"), 1, dflHidden)
	return dflashAddRef(h1, ffResid) // layer_scalar == 1.0 → no-op
}

// dflashRefTensors mirrors the fixture weights as bf16-exact f32, shared by the reference
// ops so the reference reads the SAME rounded weights the metal path does.
var dflashRefTensors = dflashSyntheticTensors()

func dflashVecRef(name string) []float32 { return bf16Floats(dflashRefTensors[name].Data) }

func dflashMatVecRef(name string, vec []float32, out, in int) []float32 {
	mat := bf16Floats(dflashRefTensors[name].Data)
	res := make([]float32, out)
	for o := 0; o < out; o++ {
		var sum float32
		for k := 0; k < in; k++ {
			sum += mat[o*in+k] * vec[k]
		}
		res[o] = dflashRoundBF16(sum)
	}
	return res
}

func dflashRMSNormRef(x, w []float32, rows, axis int) []float32 {
	out := make([]float32, rows*axis)
	for r := 0; r < rows; r++ {
		var ss float32
		for i := 0; i < axis; i++ {
			v := x[r*axis+i]
			ss += v * v
		}
		scale := float32(1.0 / math.Sqrt(float64(ss/float32(axis))+float64(dflEps)))
		for i := 0; i < axis; i++ {
			out[r*axis+i] = dflashRoundBF16(x[r*axis+i] * scale * w[i])
		}
	}
	return out
}

// dflashRopeRef mirrors RoPEDimsBF16Into (MLX non-traditional, contiguous pairs
// (d, d+rotaryDim/2)) with rotaryDim == headDim and scale 1.
func dflashRopeRef(q []float32, pos int) []float32 {
	out := append([]float32(nil), q...)
	half := dflHeadDim / 2
	for h := 0; h < dflHeads; h++ {
		base := h * dflHeadDim
		for d := 0; d < half; d++ {
			invFreq := float32(math.Pow(float64(dflRope), -2*float64(d)/float64(dflHeadDim)))
			theta := float64(pos) * float64(invFreq)
			cos := float32(math.Cos(theta))
			sin := float32(math.Sin(theta))
			x1 := q[base+d]
			x2 := q[base+d+half]
			out[base+d] = dflashRoundBF16(x1*cos - x2*sin)
			out[base+d+half] = dflashRoundBF16(x1*sin + x2*cos)
		}
	}
	return out
}

// dflashSDPARef mirrors SDPAInto: per head, softmax(scale·q·kᵀ) over the numAux injected
// rows, then the value average. kvHeads==1 so every query head reads the single kv head.
func dflashSDPARef(q, k, v []float32) []float32 {
	out := make([]float32, dflHeads*dflHeadDim)
	kvRow := dflKV * dflHeadDim
	for h := 0; h < dflHeads; h++ {
		kvHead := h / (dflHeads / dflKV)
		scores := make([]float32, dflNumAux)
		maxS := float32(math.Inf(-1))
		for t := 0; t < dflNumAux; t++ {
			var dot float32
			for d := 0; d < dflHeadDim; d++ {
				dot += q[h*dflHeadDim+d] * k[t*kvRow+kvHead*dflHeadDim+d]
			}
			scores[t] = dflScale * dot
			if scores[t] > maxS {
				maxS = scores[t]
			}
		}
		var sum float32
		for t := 0; t < dflNumAux; t++ {
			scores[t] = float32(math.Exp(float64(scores[t] - maxS)))
			sum += scores[t]
		}
		for d := 0; d < dflHeadDim; d++ {
			var acc float32
			for t := 0; t < dflNumAux; t++ {
				acc += (scores[t] / sum) * v[t*kvRow+kvHead*dflHeadDim+d]
			}
			out[h*dflHeadDim+d] = dflashRoundBF16(acc)
		}
	}
	return out
}

// dflashGeluGateMulRef mirrors GeluGateMulBF16Into: tanh-approximation GELU of the gate,
// times up.
func dflashGeluGateMulRef(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	for i := range gate {
		x := float64(gate[i])
		inner := x + 0.044715*x*x*x
		g := 0.5 * x * (1 + math.Tanh(0.7978845608028654*inner))
		out[i] = dflashRoundBF16(float32(g) * up[i])
	}
	return out
}

func dflashAddRef(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = dflashRoundBF16(a[i] + b[i])
	}
	return out
}

func dflashArgmax(logits []float32) int {
	best, bestVal := 0, float32(math.Inf(-1))
	for i, v := range logits {
		if v > bestVal {
			bestVal, best = v, i
		}
	}
	return best
}

func dflashRoundBF16(f float32) float32 {
	h := f32ToBF16(f)
	return bf16ToF32(byte(h), byte(h>>8))
}

// writeDFlashDrafterDir writes a loadable DFlash pack to a temp dir: a config.json
// carrying the DFlash marker + decoder arch a registered spec parses, the drafter
// tokenizer, and the tensors as safetensors — the reactive pack the loader consumes.
func writeDFlashDrafterDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	cfg := core.Sprintf(`{
		"model_type": "gemma4_dflash_assistant",
		"speculators_model_type": "dflash",
		"block_size": %d,
		"aux_hidden_state_layer_ids": [0, 1],
		"backbone_hidden_size": %d,
		"speculators_config": { "verifier": { "name": "synthetic-target" } },
		"text_config": {
			"model_type": "gemma4_dflash_assistant",
			"hidden_size": %d,
			"num_hidden_layers": 1,
			"intermediate_size": %d,
			"num_attention_heads": %d,
			"num_key_value_heads": %d,
			"head_dim": %d,
			"vocab_size": %d,
			"rms_norm_eps": 0.000001,
			"max_position_embeddings": 16,
			"layer_types": ["full_attention"],
			"rope_parameters": {
				"full_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0}
			}
		}
	}`, dflBlock, dflBackbone, dflHidden, dflFF, dflHeads, dflKV, dflHeadDim, dflVocab)
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(dflashSyntheticTensors())
	if err != nil {
		t.Fatalf("encode DFlash tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// dflashMetalDraftLogits re-runs the metal forward per block position and returns the
// pre-argmax draft logits (dflash.lm_head output), for the parity logit-tolerance check.
func dflashMetalDraftLogits(t testing.TB, d *DFlashDrafter, auxHiddens [][]byte, anchor []byte, anchorPos int) [][]float32 {
	t.Helper()
	auxContext, err := d.fuseAuxContext(auxHiddens)
	if err != nil {
		t.Fatalf("fuseAuxContext: %v", err)
	}
	seed, err := d.m.DraftInputProjectionInto(nil, anchor, auxContext)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	injected, err := d.injectedKV(auxHiddens)
	if err != nil {
		t.Fatalf("injectedKV: %v", err)
	}
	head, err := nativeAssistantBF16Matrix(d.m, dflashLMHeadWeight, d.draftVocab, d.m.Arch.Hidden)
	if err != nil {
		t.Fatalf("lm_head: %v", err)
	}
	hiddenBytes := d.m.Arch.Hidden * bf16Size
	out := make([][]float32, 0, d.blockSize)
	for j := 0; j < d.blockSize; j++ {
		h := append([]byte(nil), seed...)
		for li := range d.m.Arch.Layer {
			kv := injected[li]
			kv.Offset = anchorPos + j - (d.numAux - 1)
			next, lerr := d.m.draftLayerIntoScratch(d.scratch.bytes(assistantDraftScratchLayerOut, hiddenBytes), li, h, kv, &d.scratch)
			if lerr != nil {
				t.Fatalf("draft layer: %v", lerr)
			}
			h = append([]byte(nil), next...)
		}
		normed, ferr := d.m.DraftFinalNormInto(nil, h)
		if ferr != nil {
			t.Fatalf("final norm: %v", ferr)
		}
		logits, herr := MatVecBF16Into(nil, head.Data, normed, d.draftVocab, d.m.Arch.Hidden)
		if herr != nil {
			t.Fatalf("head matvec: %v", herr)
		}
		out = append(out, bf16Floats(logits))
	}
	return out
}
