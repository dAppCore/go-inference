// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/Qwen/qwen2"
)

// real_checkpoint_gpu_test.go is the GPU half of board #24 item 2's root-cause receipt — the
// sibling of model/arch/Qwen/qwen2/real_checkpoint_test.go's host-side ArgmaxParis proof. That
// file proved the arch mapping byte-correct on the host and (at the time) inferred the garble
// "lives downstream in engine/metal's GPU decode kernels". This test DISPROVES that inference:
// the production session (LoadDir → PrefillTokens → GenerateFromCache — the same object serve
// drives, ICB path included) fed mlx-lm's OWN token ids for "The capital of France is" argmaxes
// the SAME token mlx-lm does (12095, " Paris") and continues coherently. The engine forward is
// correct end-to-end on the real checkpoint.
//
// The garble users saw came from the INPUT side: every lem chat surface (generate -prompt,
// serve chat) frames the prompt through the checkpoint's ChatML template, and this snapshot is
// the BASE Coder model — a base model completing a chat-framed turn degenerates on ANY engine
// (mlx-lm with its own applied template loops "The capital放法 of France is France. The capital
// of France is…" on the identical snapshot). Serve chat wants the -Instruct variant; the base
// form is for raw completion. One real gap stays open from the diff: our ChatML render omits
// the default system block Qwen's chat_template.jinja injects (13 ids vs mlx-lm's 24 on this
// prompt) — tracked on the board, not part of this receipt.

// qwen2CoderGPUPromptIDs are mlx-lm's Qwen2Tokenizer ids for "The capital of France is"
// (--ignore-chat-template) — the same fixed input real_checkpoint_test.go bisected with.
// decode/tokenizer.Encode produces these exact ids from this snapshot's tokenizer.json, so
// the tokenizer is not a variable in this receipt.
var qwen2CoderGPUPromptIDs = []int32{785, 6722, 315, 9625, 374}

// TestRealCheckpointGPU_ArgmaxParis_Good runs mlx-community/Qwen2.5-Coder-3B-4bit through the
// production ArchSession — the full GPU decode path serve uses — on the fixed prompt ids and
// requires the first generated token to be mlx-lm's answer (12095, " Paris"). A regression in
// the GPU quant decode for the qwen2 shape class (QKV bias, NEOX rope, GQA, SwiGLU at
// 2048/16/2/128/11008) flips this argmax. Skips when the checkpoint is not in the local HF
// cache, so CI stays green off this machine.
func TestRealCheckpointGPU_ArgmaxParis_Good(t *testing.T) {
	dir := enginegate.HFModelPath(t, "mlx-community/Qwen2.5-Coder-3B-4bit")
	sess, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	if err := sess.PrefillTokens(qwen2CoderGPUPromptIDs); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	gen, err := sess.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	if len(gen) == 0 || gen[0] != 12095 {
		t.Fatalf("GPU forward argmax = %v, want first id 12095 (' Paris', mlx-lm reference) — the qwen2 GPU decode regressed", gen)
	}
	t.Logf("GPU generated ids: %v", gen)
}

// TestRealCheckpointGPU_Bonsai1BitRepack_Good locks #24's other half end-to-end: the 1-bit
// affine checkpoint (prism-ml/Bonsai-27B-mlx-1bit, qwen3_5 hybrid) loads through the b1→b2
// exact repack (mlxaffine.RepackB1ToB2: no b_1 kernel ships, and mlx-lm itself REJECTS bits=1
// outright) hooked at model.LoadLinear, the repacked OWNED weights bind resident through the
// registered-owned-range fallback (#60 — the wall that had this test in its _Bad decline form),
// and the model answers the capital question sanely on the GPU. The repack's exactness has its
// own unit gate (TestRepackB1ToB2_ExactDequant); the synthetic in-CI twin of this lane is
// TestLoadDirB1RepackServe_Good. Skips when the checkpoint is not in the local HF cache.
func TestRealCheckpointGPU_Bonsai1BitRepack_Good(t *testing.T) {
	dir := enginegate.HFModelPath(t, "prism-ml/Bonsai-27B-mlx-1bit")
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	ids := tok.Encode("The capital of France is")
	tm, err := LoadTokenModelDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	gen, err := model.Generate(tm, ids, 4, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	text := tok.Decode(gen)
	if !strings.Contains(text, "Paris") {
		t.Fatalf("1-bit Bonsai continuation %q (ids %v) does not name Paris — the b1→b2 repack lane regressed", text, gen)
	}
	t.Logf("Bonsai continuation: %q", text)
}
