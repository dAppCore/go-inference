// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// mtp_loader_test.go proves LoadMTPHead consumes the REAL Qwen MTP head tensor layout (the exact names the
// mlx-community/Qwen3.6-27B-MTP-4bit checkpoint carries) and pairs with a loaded composed base — the
// "base + MTP head load together" receipt, without the giant real checkpoint. It runs the DENSE (bf16)
// path; the quantised path rides the SAME base loader helpers (buildAttn / buildFFN / the packed proj
// closure), which the base's own quant tests already cover.

// mkMTPHeadCheckpoint builds a synthetic Qwen MTP head checkpoint matching mkHybridCheckpoint's base
// geometry (D=8, vocab=32) with mtp_num_hidden_layers=1 full-attention layer, using the real tensor names:
// the two pre-fc RMSNorms, the fc [D,2D] combiner, layers.0.{input_layernorm, self_attn.*,
// post_attention_layernorm, mlp.*} and the head final norm. No embed_tokens / lm_head — those are shared
// from the base.
func mkMTPHeadCheckpoint() (map[string]safetensors.Tensor, []byte) {
	const D, vocab, FF = 8, 32, 16
	const AH, AKVH, AHD = 4, 2, 8 // attention: heads 4, kv-heads 2, head_dim 8 (matches mkHybridCheckpoint)
	ts := map[string]safetensors.Tensor{
		"pre_fc_norm_embedding.weight": bf16T(syn(D, 1), D),
		"pre_fc_norm_hidden.weight":    bf16T(syn(D, 2), D),
		"fc.weight":                    bf16T(syn(D*2*D, 3), D, 2*D),
		"norm.weight":                  bf16T(syn(D, 4), D),
	}
	lp := "layers.0."
	ts[lp+"input_layernorm.weight"] = bf16T(syn(D, 11), D)
	ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, 12), D)
	ap := lp + "self_attn."
	ts[ap+"q_proj.weight"] = bf16T(syn(AH*AHD*D, 13), AH*AHD, D)
	ts[ap+"k_proj.weight"] = bf16T(syn(AKVH*AHD*D, 14), AKVH*AHD, D)
	ts[ap+"v_proj.weight"] = bf16T(syn(AKVH*AHD*D, 15), AKVH*AHD, D)
	ts[ap+"o_proj.weight"] = bf16T(syn(D*AH*AHD, 16), D, AH*AHD)
	ts[ap+"q_norm.weight"] = bf16T(syn(AHD, 17), AHD)
	ts[ap+"k_norm.weight"] = bf16T(syn(AHD, 18), AHD)
	mp := lp + "mlp."
	ts[mp+"gate_proj.weight"] = bf16T(syn(FF*D, 19), FF, D)
	ts[mp+"up_proj.weight"] = bf16T(syn(FF*D, 20), FF, D)
	ts[mp+"down_proj.weight"] = bf16T(syn(D*FF, 21), D, FF)

	config := []byte(`{"model_type":"qwen3_5_mtp","block_size":3,"text_config":{` +
		`"hidden_size":8,"mtp_num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,` +
		`"head_dim":8,"intermediate_size":16,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":1000000,` +
		`"partial_rotary_factor":0.5,"attn_output_gate":false}}`)
	return ts, config
}

// TestLoadMTPHeadGood loads a real-named head checkpoint, pairs it with a loaded composed base, and proves
// the paired greedy output is byte-identical to plain greedy decode — the whole path (load → pair →
// speculate) end to end, on the exact tensor names the real drafter ships.
func TestLoadMTPHeadGood(t *testing.T) {
	base, err := LoadComposed(mkHybridCheckpoint())
	if err != nil {
		t.Fatalf("LoadComposed(base): %v", err)
	}
	tsHead, cfgHead := mkMTPHeadCheckpoint()
	head, err := LoadMTPHead(tsHead, cfgHead, base)
	if err != nil {
		t.Fatalf("LoadMTPHead: %v", err)
	}
	if head.D != base.D || len(head.Stack.Layers) != 1 {
		t.Fatalf("head D=%d layers=%d, want D=%d layers=1", head.D, len(head.Stack.Layers), base.D)
	}
	if head.FC == nil || len(head.Enorm) != base.D || len(head.Hnorm) != base.D || len(head.Norm) != base.D {
		t.Fatalf("head weights not fully loaded: fc=%v enorm=%d hnorm=%d norm=%d", head.FC != nil, len(head.Enorm), len(head.Hnorm), len(head.Norm))
	}
	if head.BlockSize != 3 {
		t.Fatalf("head BlockSize = %d, want 3 (the checkpoint's declared trained draft depth)", head.BlockSize)
	}

	p, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair: %v", err)
	}
	if p.DefaultDraftBlock != 3 {
		t.Fatalf("pair DefaultDraftBlock = %d, want the checkpoint's block_size 3", p.DefaultDraftBlock)
	}
	prompt := []int32{1, 2, 3, 4}
	const maxNew, block = 20, 4
	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	got, m, err := p.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("loaded-head speculative output diverged:\n got  %v\n want %v", got, want)
	}
	if m.ProposedTokens == 0 {
		t.Fatal("loaded head did not engage (0 proposed)")
	}
	t.Logf("loaded real-named head: byte-identical over %d tokens; proposed=%d accepted=%d (%.0f%%)",
		len(got), m.ProposedTokens, m.AcceptedTokens, m.AcceptanceRate*100)
}

// TestLoadMTPHeadBad covers the loader guards: a nil base, a head whose hidden_size disagrees with the base,
// and a checkpoint missing the fc combiner are each a clean error.
func TestLoadMTPHeadBad(t *testing.T) {
	base, err := LoadComposed(mkHybridCheckpoint())
	if err != nil {
		t.Fatalf("LoadComposed(base): %v", err)
	}
	ts, cfg := mkMTPHeadCheckpoint()

	if _, err := LoadMTPHead(ts, cfg, nil); err == nil {
		t.Fatal("expected an error for a nil base")
	}
	wrong := []byte(`{"model_type":"qwen3_5_mtp","text_config":{"hidden_size":16,"mtp_num_hidden_layers":1}}`)
	if _, err := LoadMTPHead(ts, wrong, base); err == nil {
		t.Fatal("expected an error when the head hidden_size disagrees with the base")
	}
	delete(ts, "fc.weight")
	if _, err := LoadMTPHead(ts, cfg, base); err == nil {
		t.Fatal("expected an error when the fc combiner is missing")
	}
}
