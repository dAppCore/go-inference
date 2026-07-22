// SPDX-Licence-Identifier: EUPL-1.2

package dflash_test

import (
	"math"
	"math/rand"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/model/safetensors"
)

// zlab_test.go proves ZLabForward two ways, the same r2-r5 posture the rest of
// this codebase's GPU forwards use, adapted to a host-only reference:
//
//   - SYNTHETIC (always runs, no download): varied seeded weights exercise every
//     op (multi-layer residual, GQA broadcast, RoPE, cross+self attention,
//     SiLU MLP), gating shape, determinism and honest refusals.
//   - REAL CHECKPOINT (env-gated, skips cleanly without it — the
//     LEM_AWQ_REFERENCE_DIR pattern model/quant/awq/reference_test.go
//     established): testdata/zlab_qwen3_4b_oracle.json pins the exact inputs
//     and expected outputs an independent numpy re-implementation computed from
//     z-lab/Qwen3-4B-DFlash-b16's REAL downloaded weights (cross-checked against
//     that checkpoint's own modeling_dflash.py executed through transformers —
//     see docs/design-dflash-survey.md for the full receipt and the exact
//     reproduction steps); LTHN_DFLASH_ZLAB_CKPT points at a local directory
//     holding that checkpoint's model.safetensors, and this test loads the REAL
//     weights and checks ZLabForward's output against the pinned expectation at
//     3 depths (after layer 0, after layer 2, final norm) — proof this package's
//     reading of the real architecture, not just of its own doc comment, is
//     correct.

const (
	zlHidden  = 12
	zlHeads   = 4
	zlKV      = 2
	zlHeadDim = 8
	zlInter   = 20
	zlLayers  = 2
	zlNumAux  = 2
)

// zlabSyntheticFloats fills n values from a seeded PRNG (never a constant — the
// vacuous-fixture trap) so no op collapses to a no-op average.
func zlabSyntheticFloats(n int, salt int64) []float32 {
	r := rand.New(rand.NewSource(salt))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64()) * 0.05
	}
	return out
}

// zlabSyntheticWeights builds a complete, varied ZLabWeights set for the tiny
// zl* dims above — every tensor name a real z-lab checkpoint carries (fc,
// hidden_norm, norm, and per layer input/post layernorms + q/k/v/o proj +
// q/k norm + gate/up/down proj), nothing invented.
func zlabSyntheticWeights() dflash.ZLabWeights {
	w := dflash.ZLabWeights{}
	salt := int64(1)
	next := func(n int) []float32 {
		salt++
		return zlabSyntheticFloats(n, salt)
	}
	w["fc.weight"] = next(zlHidden * zlNumAux * zlHidden)
	w["hidden_norm.weight"] = next(zlHidden)
	w["norm.weight"] = next(zlHidden)
	qDim, kvDim := zlHeads*zlHeadDim, zlKV*zlHeadDim
	for li := 0; li < zlLayers; li++ {
		p := core.Sprintf("layers.%d.", li)
		w[p+"input_layernorm.weight"] = next(zlHidden)
		w[p+"post_attention_layernorm.weight"] = next(zlHidden)
		w[p+"self_attn.q_proj.weight"] = next(qDim * zlHidden)
		w[p+"self_attn.k_proj.weight"] = next(kvDim * zlHidden)
		w[p+"self_attn.v_proj.weight"] = next(kvDim * zlHidden)
		w[p+"self_attn.o_proj.weight"] = next(zlHidden * qDim)
		w[p+"self_attn.q_norm.weight"] = next(zlHeadDim)
		w[p+"self_attn.k_norm.weight"] = next(zlHeadDim)
		w[p+"mlp.gate_proj.weight"] = next(zlInter * zlHidden)
		w[p+"mlp.up_proj.weight"] = next(zlInter * zlHidden)
		w[p+"mlp.down_proj.weight"] = next(zlHidden * zlInter)
	}
	return w
}

func zlabTestArch() dflash.ZLabArch {
	return dflash.ZLabArch{
		Hidden: zlHidden, Heads: zlHeads, KVHeads: zlKV, HeadDim: zlHeadDim,
		Intermediate: zlInter, NumLayers: zlLayers, NumAux: zlNumAux,
		Eps: 1e-6, RopeTheta: 10000,
	}
}

// variance reports whether s has any spread at all — the vacuous-constant guard:
// a broken forward that zeroes or averages everything away still "succeeds" by
// shape alone, so every Good test also checks the output actually varies.
func variance(s []float32) float64 {
	if len(s) == 0 {
		return 0
	}
	var mean float64
	for _, v := range s {
		mean += float64(v)
	}
	mean /= float64(len(s))
	var ss float64
	for _, v := range s {
		d := float64(v) - mean
		ss += d * d
	}
	return ss / float64(len(s))
}

// TestZLabForward_Good runs the block-diffusion drafter forward over varied
// synthetic weights and checks shape, non-vacuousness and determinism — the
// same input proposes the same output every call, matching the metal engine's
// own TestDFlashProposeBlockDeterministic expectation for this forward's
// z-lab-shaped twin.
func TestZLabForward_Good(t *testing.T) {
	w := zlabSyntheticWeights()
	arch := zlabTestArch()
	const ctxLen, blockLen = 3, 2
	noise := zlabSyntheticFloats(blockLen*zlHidden, 900)
	target := zlabSyntheticFloats(ctxLen*zlNumAux*zlHidden, 901)

	final, layers, err := dflash.ZLabForward(w, arch, noise, target, ctxLen, blockLen)
	if err != nil {
		t.Fatalf("ZLabForward: %v", err)
	}
	if len(final) != blockLen*zlHidden {
		t.Fatalf("final len = %d, want %d", len(final), blockLen*zlHidden)
	}
	if len(layers) != zlLayers {
		t.Fatalf("layerOutputs count = %d, want %d", len(layers), zlLayers)
	}
	for i, lo := range layers {
		if len(lo) != blockLen*zlHidden {
			t.Fatalf("layerOutputs[%d] len = %d, want %d", i, len(lo), blockLen*zlHidden)
		}
		if variance(lo) <= 0 {
			t.Fatalf("layerOutputs[%d] is vacuously constant", i)
		}
	}
	if variance(final) <= 0 {
		t.Fatal("final output is vacuously constant")
	}

	final2, _, err := dflash.ZLabForward(w, arch, noise, target, ctxLen, blockLen)
	if err != nil {
		t.Fatalf("second ZLabForward: %v", err)
	}
	for i := range final {
		if final[i] != final2[i] {
			t.Fatalf("non-deterministic final[%d]: %g then %g", i, final[i], final2[i])
		}
	}
}

// TestZLabForward_Bad covers the honest refusals: a missing tensor is named, a
// GQA mismatch is rejected, and mis-shaped inputs are rejected rather than
// silently truncated or panicking.
func TestZLabForward_Bad(t *testing.T) {
	arch := zlabTestArch()
	const ctxLen, blockLen = 3, 2
	noise := zlabSyntheticFloats(blockLen*zlHidden, 910)
	target := zlabSyntheticFloats(ctxLen*zlNumAux*zlHidden, 911)

	t.Run("missing tensor", func(t *testing.T) {
		w := zlabSyntheticWeights()
		delete(w, "layers.1.mlp.down_proj.weight")
		if _, _, err := dflash.ZLabForward(w, arch, noise, target, ctxLen, blockLen); err == nil {
			t.Fatal("expected an error for a missing tensor, got nil")
		}
	})
	t.Run("heads not multiple of kv_heads", func(t *testing.T) {
		w := zlabSyntheticWeights()
		bad := arch
		bad.KVHeads = 3 // zlHeads=4 is not a multiple of 3
		if _, _, err := dflash.ZLabForward(w, bad, noise, target, ctxLen, blockLen); err == nil {
			t.Fatal("expected a GQA-mismatch error, got nil")
		}
	})
	t.Run("wrong noise embedding length", func(t *testing.T) {
		w := zlabSyntheticWeights()
		if _, _, err := dflash.ZLabForward(w, arch, noise[:len(noise)-1], target, ctxLen, blockLen); err == nil {
			t.Fatal("expected a shape error for a truncated noise embedding, got nil")
		}
	})
	t.Run("wrong target hidden length", func(t *testing.T) {
		w := zlabSyntheticWeights()
		if _, _, err := dflash.ZLabForward(w, arch, noise, target[:len(target)-1], ctxLen, blockLen); err == nil {
			t.Fatal("expected a shape error for a truncated target hidden, got nil")
		}
	})
	t.Run("incomplete arch", func(t *testing.T) {
		w := zlabSyntheticWeights()
		if _, _, err := dflash.ZLabForward(w, dflash.ZLabArch{}, noise, target, ctxLen, blockLen); err == nil {
			t.Fatal("expected an error for a zero-value arch, got nil")
		}
	})
}

// TestZLabForward_Ugly covers the degenerate-but-valid edges: no fused context
// at all (ctxLen 0 — pure intra-block self-attention, no verifier rows to
// cross-attend) and a single-position block.
func TestZLabForward_Ugly(t *testing.T) {
	arch := zlabTestArch()
	w := zlabSyntheticWeights()

	t.Run("zero context length", func(t *testing.T) {
		const blockLen = 2
		noise := zlabSyntheticFloats(blockLen*zlHidden, 920)
		final, layers, err := dflash.ZLabForward(w, arch, noise, nil, 0, blockLen)
		if err != nil {
			t.Fatalf("ZLabForward with ctxLen=0: %v", err)
		}
		if len(final) != blockLen*zlHidden || len(layers) != zlLayers {
			t.Fatalf("shape wrong for ctxLen=0: final=%d layers=%d", len(final), len(layers))
		}
		if variance(final) <= 0 {
			t.Fatal("ctxLen=0 output is vacuously constant")
		}
	})
	t.Run("single position block", func(t *testing.T) {
		const ctxLen = 3
		noise := zlabSyntheticFloats(1*zlHidden, 921)
		target := zlabSyntheticFloats(ctxLen*zlNumAux*zlHidden, 922)
		final, _, err := dflash.ZLabForward(w, arch, noise, target, ctxLen, 1)
		if err != nil {
			t.Fatalf("ZLabForward with blockLen=1: %v", err)
		}
		if len(final) != zlHidden {
			t.Fatalf("final len = %d, want %d", len(final), zlHidden)
		}
	})
}

// TestZLabWidenBF16_Good pins the bf16->f32 widening against known bit
// patterns: 1.0 (0x3F80) and -2.0 (0xC000), the top 16 bits of each value's
// IEEE754 float32 encoding, little-endian byte order (the safetensors layout).
func TestZLabWidenBF16_Good(t *testing.T) {
	raw := []byte{0x80, 0x3F, 0x00, 0xC0} // 1.0, -2.0
	got := dflash.ZLabWidenBF16(raw)
	want := []float32{1.0, -2.0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("widened[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// --- real-checkpoint oracle gate (env-gated: skips cleanly without it) ---

// zlabOracleFixture is testdata/zlab_qwen3_4b_oracle.json's shape: the exact
// inputs and expected per-depth outputs an independent numpy re-implementation
// computed from z-lab/Qwen3-4B-DFlash-b16's real downloaded weights (see
// docs/design-dflash-survey.md).
type zlabOracleFixture struct {
	Hidden          int         `json:"hidden"`
	Heads           int         `json:"heads"`
	KVHeads         int         `json:"kv_heads"`
	HeadDim         int         `json:"head_dim"`
	NLayers         int         `json:"n_layers"`
	NumAux          int         `json:"num_aux"`
	CtxLen          int         `json:"ctx_len"`
	BlockSize       int         `json:"block_size"`
	Eps             float64     `json:"eps"`
	RopeTheta       float64     `json:"rope_theta"`
	NoiseEmbedding  [][]float32 `json:"noise_embedding"`
	TargetHiddenRaw [][]float32 `json:"target_hidden_raw"`
	Depths          struct {
		AfterLayer0 [][]float32 `json:"after_layer_0"`
		AfterLayer2 [][]float32 `json:"after_layer_2"`
		FinalNorm   [][]float32 `json:"final_norm"`
	} `json:"depths"`
}

func flatten(rows [][]float32) []float32 {
	if len(rows) == 0 {
		return nil
	}
	out := make([]float32, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// zlabAssertClose gates got against want with a tolerance wide enough to absorb
// f32-accumulation-order noise between two independent implementations (Go here
// summing in float64, the numpy oracle in float32) but tight enough to catch a
// wrong op — the same relative-tolerance shape
// assistant_dflash_test.go's TestDFlashProposeBlockParity uses for its own
// metal-vs-host logit check.
func zlabAssertClose(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	var maxDiff float32
	for i := range want {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		tol := 0.02 * (1 + float32(math.Abs(float64(want[i]))))
		if diff > tol {
			t.Fatalf("%s[%d] = %v, want %v (diff %v > tol %v)", label, i, got[i], want[i], diff, tol)
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("%s: max abs diff %v across %d values", label, maxDiff, len(want))
}

// TestZLabForward_RealCheckpoint gates ZLabForward against the REAL downloaded
// z-lab/Qwen3-4B-DFlash-b16 checkpoint at 3 depths (after layer 0, after layer
// 2, final norm) — the receipt that this package's reading of the real
// architecture, not just its own doc comment, is correct. Skips cleanly without
// LTHN_DFLASH_ZLAB_CKPT (the model/quant/awq LEM_AWQ_REFERENCE_DIR pattern):
//
//	python -c "from huggingface_hub import snapshot_download; \
//	  print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16', \
//	  allow_patterns=['config.json','model.safetensors']))"
//	LTHN_DFLASH_ZLAB_CKPT=<printed dir> go test ./decode/dflash/... -run RealCheckpoint -v
func TestZLabForward_RealCheckpoint(t *testing.T) {
	dir := core.Getenv("LTHN_DFLASH_ZLAB_CKPT")
	if core.Trim(dir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_CKPT to a local z-lab/Qwen3-4B-DFlash-b16 snapshot (see test doc comment)")
	}

	fixtureBytes := core.ReadFile("testdata/zlab_qwen3_4b_oracle.json")
	if !fixtureBytes.OK {
		t.Fatalf("read oracle fixture: %v", fixtureBytes.Err())
	}
	var fx zlabOracleFixture
	if r := core.JSONUnmarshal(fixtureBytes.Value.([]byte), &fx); !r.OK {
		t.Fatalf("parse oracle fixture: %v", r.Err())
	}

	raw, err := safetensors.Load(core.PathJoin(dir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load real checkpoint: %v", err)
	}
	w := dflash.ZLabWeights{}
	for name, tensor := range raw {
		w[name] = dflash.ZLabWidenBF16(tensor.Data)
	}

	arch := dflash.ZLabArch{
		Hidden: fx.Hidden, Heads: fx.Heads, KVHeads: fx.KVHeads, HeadDim: fx.HeadDim,
		Intermediate: 9728, NumLayers: fx.NLayers, NumAux: fx.NumAux,
		Eps: float32(fx.Eps), RopeTheta: float32(fx.RopeTheta),
	}
	noise := flatten(fx.NoiseEmbedding)
	target := flatten(fx.TargetHiddenRaw)

	final, layers, err := dflash.ZLabForward(w, arch, noise, target, fx.CtxLen, fx.BlockSize)
	if err != nil {
		t.Fatalf("ZLabForward on real checkpoint: %v", err)
	}
	if len(layers) < 3 {
		t.Fatalf("expected at least 3 layer outputs, got %d", len(layers))
	}

	zlabAssertClose(t, "after_layer_0", layers[0], flatten(fx.Depths.AfterLayer0))
	zlabAssertClose(t, "after_layer_2", layers[2], flatten(fx.Depths.AfterLayer2))
	zlabAssertClose(t, "final_norm", final, flatten(fx.Depths.FinalNorm))
}
