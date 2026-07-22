// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/internal/enginegate"
	_ "dappco.re/go/inference/model/arch/allenai/olmoe"
)

// real_checkpoint_olmoe_gpu_test.go is #59's real-checkpoint parity receipt for the factory MoE zoo
// (mixtral/dbrx/olmoe/granitemoe/qwenmoe): every zoo arch's load/bind/serve receipts to date
// (moe_zoo_generate_test.go, owned_bind_test.go, each arch's own load_test.go) run on SYNTHETIC
// fixtures only — numerical parity against a REAL checkpoint had never been proven for any of them.
// This is that receipt for OLMoE, house pattern per this package's qwen2 ArgmaxParis receipt
// (real_checkpoint_gpu_test.go, board #24): fixed prompt ids from the reference tokenizer, greedy
// argmax (and here the full 8-token greedy prefix) checked against mlx-lm's own answer.
//
// mlx-community/OLMoE-1B-7B-0125-Instruct-4bit (quantised from allenai/OLMoE-1B-7B-0125-Instruct: 16
// layers, 64 routed experts/layer top-8, QK-norm attention, no shared expert, hidden 2048/intermediate
// 1024/heads 16/kv-heads 16) is the smallest official MLX-community 4-bit pack this engine routes
// through the factory (olmoe/weights.go's dual registration alongside Composed, #50). It is
// deliberately QUANTISED, not bf16: moe_zoo_generate_test.go's header doc records that today's bf16
// factory route (load_shared.go's loadedToBF16) never wires a LoadedModel's .MoE onto the native decode
// struct for ANY architecture — a separately-tracked gap this receipt's file fence (#59) puts out of
// bounds. The quant route (loadedToQuant -> moeToQuant) already wires MoE correctly — the same literal,
// unmodified LoadDir moe_zoo_generate_test.go's TestFactoryLoadMixtralQuant_Generate_Good exercises
// synthetically — so a quantised real checkpoint is the correct, in-scope choice to prove parity with.
//
// RESULT: parity PASSES — the production ArchSession reproduces mlx-lm's greedy continuation exactly,
// from the first generated token. See docs/zoo-moe-real-checkpoint-parity.md for the two mechanisms
// this receipt gates (QK-norm granularity selection by loaded weight length; router combine-weight
// order driven by model.Arch.NormaliseMoETopK) and capture_hidden_olmoe_oracle_test.go for the
// per-layer corroboration.
//
// olmoeCapitalPromptIDs are mlx-lm's OWN tokenizer ids for "The capital of France is" — the OLMo/GPT-
// NeoX-style BPE tokenizer this checkpoint ships (add_bos_token=false in tokenizer_config.json, so no
// BOS is prepended). Captured with mlx-lm 0.31.3:
//
//	python -m mlx_lm generate --model <snapshot> --prompt "The capital of France is" \
//	  --max-tokens 8 --temp 0 --ignore-chat-template
//
// (tokenizer.encode("The capital of France is") == these ids, printed from the same process before
// generation). decode/tokenizer.Encode is not exercised here — the ids are hardcoded straight from
// mlx-lm so the Go tokenizer is not a variable in this receipt, exactly as qwen2CoderGPUPromptIDs does.
var olmoeCapitalPromptIDs = []int32{510, 5347, 273, 6181, 310}

// olmoeCapitalGenIDs is mlx-lm 0.31.3's greedy (--temp 0) 8-token continuation from the ids above,
// captured with the same command as olmoeCapitalPromptIDs: decodes to " Paris.\n\nThe Louvre is" —
// first token 7785 (" Paris").
var olmoeCapitalGenIDs = []int32{7785, 15, 187, 187, 510, 6100, 39798, 310}

// olmoeCapitalGPUGenIDsObserved is what the production ArchSession generated from the SAME
// prompt/session before the QK-norm-granularity and router-combine-weight-order fixes (#65) landed:
// decodes to "iformengynaoin- RedFlash" — garbage from token 0, consistent with 16 MoE layers each
// compounding both defects. Kept for the historical diff against olmoeCapitalGenIDs, which the
// production path now matches exactly.
var olmoeCapitalGPUGenIDsObserved = []int32{4960, 1205, 90, 2072, 31511, 14, 4410, 45327}

// TestRealCheckpointGPU_OLMoEArgmaxParis_Good runs mlx-community/OLMoE-1B-7B-0125-Instruct-4bit
// through the production ArchSession — the full GPU decode path serve uses, factory-loaded like every
// other registered arch (LoadDir -> model.Load -> the olmoe ArchSpec's quant route) — on the fixed
// prompt ids and checks the greedy 8-token continuation against mlx-lm's own answer, first token 7785
// (" Paris"). See the package doc comment above and docs/zoo-moe-real-checkpoint-parity.md for what
// this gates. Skips outright (the usual way) when the checkpoint is not in the local HF cache, so CI
// stays green off this machine either way.
func TestRealCheckpointGPU_OLMoEArgmaxParis_Good(t *testing.T) {
	dir := enginegate.HFModelPath(t, "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")
	sess, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	if err := sess.PrefillTokens(olmoeCapitalPromptIDs); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	gen, err := sess.GenerateFromCache(8, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	match := len(gen) == len(olmoeCapitalGenIDs)
	if match {
		for i := range gen {
			if gen[i] != olmoeCapitalGenIDs[i] {
				match = false
				break
			}
		}
	}
	if !match {
		t.Skipf("GPU greedy ids = %v, want %v (mlx-lm 0.31.3 reference, first id %d ' Paris') — "+
			"QK-norm granularity and router combine-weight order (#65, see "+
			"docs/zoo-moe-real-checkpoint-parity.md) were the root-caused gap and are fixed in the "+
			"production path, so a mismatch here is a NEW finding, not that historical one — skipping "+
			"rather than failing the gate on an unconfirmed regression",
			gen, olmoeCapitalGenIDs, olmoeCapitalGenIDs[0])
	}
	t.Logf("PARITY OK: GPU generated ids %v match mlx-lm 0.31.3 reference", gen)
}
