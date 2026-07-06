// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model"
)

func TestRealE2BAssistantLoadMetadata(t *testing.T) {
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs(%s, %s): %v", targetDir, assistantDir, err)
	}
	defer pair.Close()

	assistant := pair.Assistant
	if assistant.ModelType() != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want gemma4_assistant", assistant.ModelType())
	}
	if assistant.NumLayers() != 4 {
		t.Fatalf("NumLayers = %d, want 4", assistant.NumLayers())
	}
	if assistant.BackboneHiddenSize <= 0 || assistant.Arch.Hidden <= 0 || assistant.Arch.Vocab <= 0 {
		t.Fatalf("assistant metadata = backbone %d arch %+v", assistant.BackboneHiddenSize, assistant.Arch)
	}
	if _, ok := assistant.Tensor("pre_projection.weight"); !ok {
		t.Fatal("pre_projection.weight was not retained")
	}
	if _, ok := assistant.Tensor("post_projection.weight"); !ok {
		t.Fatal("post_projection.weight was not retained")
	}
}

// TestRealE2BAssistantBoundaryHiddenGainExport pins the drafter-seed contract:
// BoundaryNormedHidden exports the REFERENCE boundary vector (HF
// hidden_states[-1] = x̂ ⊙ (1+final_norm_w)), not the step's gain-folded
// head-input copy. The gain vector's mean magnitude is ~14 on gemma4, so the
// exported sum must sit well above the raw retained sum — the gainless export
// (ratio 1.0) is the bug that made every MTP draft target-blind (~5% live
// acceptance).
func TestRealE2BAssistantBoundaryHiddenGainExport(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	target, err := LoadDir(targetDir, 4096)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()
	prompt := realE2BAssistantPrompt(t, targetDir)
	if err := target.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepareAssistantPrompt: %v", err)
	}
	exported, err := target.BoundaryNormedHidden()
	if err != nil {
		t.Fatalf("BoundaryNormedHidden: %v", err)
	}
	sumAbs := func(b []byte) float64 {
		s := 0.0
		for i := 0; i+1 < len(b); i += 2 {
			v := float64(bf16ToF32(b[i], b[i+1]))
			if v < 0 {
				v = -v
			}
			s += v
		}
		return s
	}
	raw, out := sumAbs(target.retainedHidden), sumAbs(exported)
	if out < raw*5 {
		t.Fatalf("exported boundary hidden sum|.| = %.1f vs raw retained %.1f — final-norm gain not applied (drafter seed is target-blind)", out, raw)
	}
}

// TestRealE2BAssistantAcceptanceFloor pins live draft acceptance on the real
// cached pair over an open-prose prompt. Before the proportional-rope pairing
// + boundary-hidden gain fixes this sat at ~5-12%; healthy runs land 20-40%.
// A drop below the floor means one of the draft-input contracts regressed
// (rope pairing, hidden gain, projection, or KV mapping).
func TestRealE2BAssistantAcceptanceFloor(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	target, err := LoadDir(targetDir, 4096)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()
	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	prompt := realE2BAssistantPrompt(t, targetDir)
	res, err := pair.GenerateFromSession(target, prompt, 64, -1, 4, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}
	if res.DraftTokens == 0 {
		t.Fatal("no tokens drafted — speculative lane did not engage")
	}
	// 15% floor: broken draft inputs sat at 5-12% on this prompt; healthy runs
	// land 20-40%. The floor catches the collapse class without flaking on
	// prompt-level noise.
	if res.AcceptedTokens*20 < res.DraftTokens*3 {
		t.Fatalf("draft acceptance %d/%d (%.0f%%) below the 15%% floor — a draft-input contract regressed",
			res.AcceptedTokens, res.DraftTokens, 100*float64(res.AcceptedTokens)/float64(res.DraftTokens))
	}
}

// TestNativeAssistantRoPEProportionalPairsFullHead pins the draft-rope pairing
// semantics deterministically: a proportional partial-rotary layer (rotaryDim
// 128 of headDim 512) rotates pair (d, d+headDim/2) over the FULL head, so a
// query with energy ONLY in dim 300 (the 256..319 rotated band) must come out
// changed at position > 0. The old contiguous-block path left every dim ≥ 128
// untouched — the misrotation that collapsed live MTP acceptance.
func TestNativeAssistantRoPEProportionalPairsFullHead(t *testing.T) {
	requireNativeRuntime(t)
	const nHeads, headDim, rotaryDim = 1, 512, 128
	m := &AssistantModel{Arch: model.Arch{
		Hidden: 256, RotaryDim: rotaryDim, RotaryDimLocal: 256,
		RopeBase: float32(math.Pow(1e6, float64(rotaryDim)/float64(headDim))), RopeLocalBase: 10000,
	}}
	layer := model.LayerSpec{Attention: model.GlobalAttention}

	q := make([]byte, headDim*bf16Size)
	one := f32ToBF16(1.0)
	q[300*bf16Size] = byte(one)
	q[300*bf16Size+1] = byte(one >> 8)

	out, err := nativeAssistantRoPEInto(nil, q, m, layer, nHeads, headDim, 7)
	if err != nil {
		t.Fatalf("nativeAssistantRoPEInto: %v", err)
	}
	v300 := bf16ToF32(out[300*bf16Size], out[300*bf16Size+1])
	v44 := bf16ToF32(out[44*bf16Size], out[44*bf16Size+1])
	if v300 == 1.0 && v44 == 0.0 {
		t.Fatalf("dim 300 passed through unrotated (%.4f, pair dim 44 = %.4f) — proportional rope is using the contiguous-block pairing", v300, v44)
	}
	// The tail beyond the rotated band (e.g. dim 400 pairs with 144: angle 72 ≥ 64
	// rotated angles) must stay identity.
	q400 := make([]byte, headDim*bf16Size)
	q400[400*bf16Size] = byte(one)
	q400[400*bf16Size+1] = byte(one >> 8)
	out400, err := nativeAssistantRoPEInto(nil, q400, m, layer, nHeads, headDim, 7)
	if err != nil {
		t.Fatalf("nativeAssistantRoPEInto(dim400): %v", err)
	}
	if got := bf16ToF32(out400[400*bf16Size], out400[400*bf16Size+1]); got != 1.0 {
		t.Fatalf("dim 400 (beyond the rotated angles) changed to %.4f, want identity", got)
	}
}

// TestRealE2BAssistantFusedDraftParity pins the fused single-command-buffer
// drafter step against the legacy per-op path: same prepared session, same
// prompt, both paths must draft the SAME token sequence (same kernels, same
// order, same operands — the fusion only removes per-op synchronisation).
func TestRealE2BAssistantFusedDraftParity(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	target, err := LoadDir(targetDir, 4096)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()
	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	prompt := realE2BAssistantPrompt(t, targetDir)
	if err := target.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepare: %v", err)
	}
	last := prompt[len(prompt)-1]

	SetAssistantFusedDraft(false)
	slow, err := pair.draftBlockFromSession(target, last, 8, true)
	SetAssistantFusedDraft(true)
	if err != nil {
		t.Fatalf("legacy draft block: %v", err)
	}
	fast, err := pair.draftBlockFromSession(target, last, 8, true)
	if err != nil {
		t.Fatalf("fused draft block: %v", err)
	}
	if pair.fused == nil {
		t.Fatal("fused drafter did not build for the real E2B assistant")
	}
	if len(slow.Tokens) != len(fast.Tokens) {
		t.Fatalf("token counts differ: legacy %d fused %d", len(slow.Tokens), len(fast.Tokens))
	}
	for i := range slow.Tokens {
		if slow.Tokens[i] != fast.Tokens[i] {
			t.Fatalf("draft %d differs: legacy %d fused %d (legacy %v fused %v)", i, slow.Tokens[i], fast.Tokens[i], slow.Tokens, fast.Tokens)
		}
	}
}

// TestRealBF16VerifyGreedyRowsParity pins the batched K-row verify head (all
// rows' lm_head+argmax chains in one command buffer) against the per-row
// greedy loop: identical tokens for identical hiddens. The batched forward
// must also engage on the bf16 arch (the quant lanes decline to sequential —
// tracked in #278's verify-lane reclaim map).
func TestRealBF16VerifyGreedyRowsParity(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-bf16")
	target, err := LoadDir(targetDir, 4096)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()
	prompt := realE2BAssistantPrompt(t, targetDir)
	if err := target.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepare: %v", err)
	}
	draft := []int32{506, 8134, 529, 506, 8134, 529}
	posBefore := target.pos
	hiddens, batched, err := target.verifyBatchedHiddens(draft)
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !batched {
		t.Fatal("bf16 arch did not take the batched verify forward")
	}
	out := make([]int32, len(hiddens))
	ok, err := target.greedyRowsFromHiddensInPool(hiddens, nil, out)
	if err != nil || !ok {
		t.Fatalf("greedyRowsFromHiddensInPool: ok=%v err=%v", ok, err)
	}
	for i, h := range hiddens {
		want, gerr := target.greedyFromHiddenInPool(h, nil)
		if gerr != nil {
			t.Fatalf("per-row greedy: %v", gerr)
		}
		if out[i] != want {
			t.Fatalf("row %d: batched greedy %d != per-row %d", i, out[i], want)
		}
	}
	target.pos = posBefore
	if err := target.truncateSpeculativeKV(target.pos); err != nil {
		t.Fatalf("truncate: %v", err)
	}
}
