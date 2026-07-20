// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"testing"

	core "dappco.re/go"
	decodedflash "dappco.re/go/inference/decode/dflash"
	zlabdflash "dappco.re/go/inference/model/arch/z-lab/dflash"
)

// dflash_zlab_test.go proves the engine's real-z-lab DFlash block forward
// against the in-tree ORACLE, decode/dflash.ZLabForward — the host-f32
// reference that is itself pinned to the real z-lab/Qwen3-4B-DFlash-b16
// checkpoint (cross-validated against its modeling_dflash.py executed through
// transformers; docs/design-dflash-survey.md §5). Parity against it
// transitively pins this forward to the real architecture:
//
//   - BELOW the device-GEMM work floor every projection runs the host f64
//     tier, so engine-vs-oracle divergence is only accumulation-order noise
//     (f64 here vs the reference's f32 linears) — gated tight;
//   - a second geometry pushes the fc projection ABOVE the 2^20 floor,
//     exercising the steel-GEMM tier when the device is up (qwenVisionMatNT
//     falls back to host on device error — deterministic either way, the
//     standing two-tier contract) — gated at a relative tolerance;
//   - the REAL-checkpoint gate (env LTHN_DFLASH_ZLAB_CKPT, skips cleanly —
//     the decode/dflash zlab_test.go pattern) loads real weights through the
//     arch package and checks the final-norm output against the same pinned
//     fixture the reference is gated on.

const (
	dzHidden  = 12
	dzHeads   = 4
	dzKV      = 2
	dzHeadDim = 8
	dzInter   = 20
	dzLayers  = 2
	dzAux     = 2
)

// dzSeeded fills n values from a seeded PRNG (never constant — the
// vacuous-fixture trap; small magnitude keeps softmax/SiLU well-conditioned).
func dzSeeded(n int, salt int64) []float32 {
	r := rand.New(rand.NewSource(salt))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64()) * 0.05
	}
	return out
}

// dzGeometry is one test geometry's dims — the payload builder and the oracle
// arch are both derived from it so the two views can never drift.
type dzGeometry struct {
	hidden, heads, kv, headDim, inter, layers, aux int
}

var dzTiny = dzGeometry{hidden: dzHidden, heads: dzHeads, kv: dzKV, headDim: dzHeadDim, inter: dzInter, layers: dzLayers, aux: dzAux}

// dzAboveFloor sizes the fc projection past the 2^20 M·K·N device-GEMM floor
// at ctxLen 48 (48·(3·96)·96 = 1,327,104), exercising the steel tier.
var dzAboveFloor = dzGeometry{hidden: 96, heads: 6, kv: 2, headDim: 16, inter: 128, layers: 2, aux: 3}

// dzBuild builds ONE seeded weight set as BOTH views: the engine payload
// (model/arch/z-lab/dflash.DraftModel) and the oracle's name-keyed map
// (decode/dflash.ZLabWeights) — the same backing slices, so parity tests
// compare implementations, never fixtures.
func dzBuild(g dzGeometry, salt int64) (*zlabdflash.DraftModel, decodedflash.ZLabWeights, decodedflash.ZLabArch) {
	next := func(n int) []float32 {
		salt++
		return dzSeeded(n, salt)
	}
	auxIDs := make([]int, g.aux)
	for i := range auxIDs {
		auxIDs[i] = 2*i + 1
	}
	cfg := zlabdflash.Config{
		Block:  decodedflash.Config{BlockSize: 4, AuxHiddenLayerIDs: auxIDs, MaskTokenID: 7},
		Hidden: g.hidden, Heads: g.heads, KVHeads: g.kv, HeadDim: g.headDim,
		Intermediate: g.inter, NumLayers: g.layers, Eps: 1e-6, RopeTheta: 10000,
	}
	m := &zlabdflash.DraftModel{
		Cfg:        cfg,
		FC:         next(g.hidden * g.aux * g.hidden),
		HiddenNorm: next(g.hidden),
		FinalNorm:  next(g.hidden),
		Layers:     make([]zlabdflash.DraftLayer, g.layers),
	}
	qDim, kvDim := g.heads*g.headDim, g.kv*g.headDim
	for li := range m.Layers {
		m.Layers[li] = zlabdflash.DraftLayer{
			InputNorm:    next(g.hidden),
			PostAttnNorm: next(g.hidden),
			Q:            next(qDim * g.hidden),
			K:            next(kvDim * g.hidden),
			V:            next(kvDim * g.hidden),
			O:            next(g.hidden * qDim),
			QNorm:        next(g.headDim),
			KNorm:        next(g.headDim),
			Gate:         next(g.inter * g.hidden),
			Up:           next(g.inter * g.hidden),
			Down:         next(g.hidden * g.inter),
		}
	}
	w := decodedflash.ZLabWeights{
		"fc.weight":          m.FC,
		"hidden_norm.weight": m.HiddenNorm,
		"norm.weight":        m.FinalNorm,
	}
	for li := range m.Layers {
		p := core.Sprintf("layers.%d.", li)
		l := &m.Layers[li]
		w[p+"input_layernorm.weight"] = l.InputNorm
		w[p+"post_attention_layernorm.weight"] = l.PostAttnNorm
		w[p+"self_attn.q_proj.weight"] = l.Q
		w[p+"self_attn.k_proj.weight"] = l.K
		w[p+"self_attn.v_proj.weight"] = l.V
		w[p+"self_attn.o_proj.weight"] = l.O
		w[p+"self_attn.q_norm.weight"] = l.QNorm
		w[p+"self_attn.k_norm.weight"] = l.KNorm
		w[p+"mlp.gate_proj.weight"] = l.Gate
		w[p+"mlp.up_proj.weight"] = l.Up
		w[p+"mlp.down_proj.weight"] = l.Down
	}
	arch := decodedflash.ZLabArch{
		Hidden: g.hidden, Heads: g.heads, KVHeads: g.kv, HeadDim: g.headDim,
		Intermediate: g.inter, NumLayers: g.layers, NumAux: g.aux,
		Eps: 1e-6, RopeTheta: 10000,
	}
	return m, w, arch
}

// dzAssertClose gates got against want at abs+relative tolerance and reports
// the max divergence — the zlabAssertClose shape.
func dzAssertClose(t *testing.T, label string, got, want []float32, absTol, relTol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	var maxDiff float64
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		tol := absTol + relTol*math.Abs(float64(want[i]))
		if diff > tol {
			t.Fatalf("%s[%d] = %v, want %v (diff %v > tol %v)", label, i, got[i], want[i], diff, tol)
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("%s: max abs diff %v across %d values", label, maxDiff, len(want))
}

// TestDFlashZLabForward_OracleParity_Good: engine forward vs
// decode/dflash.ZLabForward on identical synthetic weights, below the device
// floor (pure host tiers both sides), across context shapes including the
// no-context degenerate. Any structural slip — a transposed weight, a missed
// norm, GELU-for-SiLU, the numAux/ctxLen conflation — diverges by orders of
// magnitude more than the accumulation-order noise gated here.
func TestDFlashZLabForward_OracleParity_Good(t *testing.T) {
	m, w, arch := dzBuild(dzTiny, 100)
	for _, tc := range []struct {
		name             string
		ctxLen, blockLen int
	}{
		{"ctx3_block2", 3, 2},
		{"ctx1_block4", 1, 4},
		{"ctx0_block2", 0, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			noise := dzSeeded(tc.blockLen*dzHidden, 900+int64(tc.ctxLen))
			var target []float32
			if tc.ctxLen > 0 {
				target = dzSeeded(tc.ctxLen*dzAux*dzHidden, 901+int64(tc.ctxLen))
			}
			got, err := DFlashZLabForward(m, noise, target, tc.ctxLen, tc.blockLen)
			if err != nil {
				t.Fatalf("DFlashZLabForward: %v", err)
			}
			want, _, err := decodedflash.ZLabForward(w, arch, noise, target, tc.ctxLen, tc.blockLen)
			if err != nil {
				t.Fatalf("oracle ZLabForward: %v", err)
			}
			dzAssertClose(t, "final", got, want, 1e-5, 1e-4)
		})
	}
}

// TestDFlashZLabForward_AboveFloorParity_Good pushes the fc projection above
// the 2^20 work floor so the steel f32 GEMM carries it when the device is up
// (host fallback otherwise — the standing two-tier contract either way), and
// gates the whole forward against the oracle at a device-accumulation
// tolerance.
func TestDFlashZLabForward_AboveFloorParity_Good(t *testing.T) {
	g := dzAboveFloor
	m, w, arch := dzBuild(g, 4000)
	const ctxLen, blockLen = 48, 8
	noise := dzSeeded(blockLen*g.hidden, 4900)
	target := dzSeeded(ctxLen*g.aux*g.hidden, 4901)
	got, err := DFlashZLabForward(m, noise, target, ctxLen, blockLen)
	if err != nil {
		t.Fatalf("DFlashZLabForward: %v", err)
	}
	want, _, err := decodedflash.ZLabForward(w, arch, noise, target, ctxLen, blockLen)
	if err != nil {
		t.Fatalf("oracle ZLabForward: %v", err)
	}
	dzAssertClose(t, "final", got, want, 1e-4, 2e-3)
}

// TestDFlashZLabForward_Determinism_Good: an identical second call produces
// identical bytes (below the floor — the host tiers are strictly ordered; the
// device tier's determinism is the steel GEMM's own standing contract).
func TestDFlashZLabForward_Determinism_Good(t *testing.T) {
	m, _, _ := dzBuild(dzTiny, 200)
	const ctxLen, blockLen = 3, 2
	noise := dzSeeded(blockLen*dzHidden, 910)
	target := dzSeeded(ctxLen*dzAux*dzHidden, 911)
	first, err := DFlashZLabForward(m, noise, target, ctxLen, blockLen)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	again, err := DFlashZLabForward(m, noise, target, ctxLen, blockLen)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	for i := range first {
		if math.Float32bits(first[i]) != math.Float32bits(again[i]) {
			t.Fatalf("forward is not bit-deterministic at %d: %v vs %v", i, first[i], again[i])
		}
	}
}

// TestDFlashZLabForward_Bad covers the honest refusals: nil/incomplete
// payloads and mis-shaped inputs fail loudly, never truncate or panic.
func TestDFlashZLabForward_Bad(t *testing.T) {
	m, _, _ := dzBuild(dzTiny, 300)
	const ctxLen, blockLen = 3, 2
	noise := dzSeeded(blockLen*dzHidden, 920)
	target := dzSeeded(ctxLen*dzAux*dzHidden, 921)

	t.Run("nil model", func(t *testing.T) {
		if _, err := DFlashZLabForward(nil, noise, target, ctxLen, blockLen); err == nil {
			t.Fatal("nil payload must fail")
		}
	})
	t.Run("wrong noise length", func(t *testing.T) {
		if _, err := DFlashZLabForward(m, noise[:len(noise)-1], target, ctxLen, blockLen); err == nil {
			t.Fatal("a truncated noise embedding must fail")
		}
	})
	t.Run("wrong target length", func(t *testing.T) {
		if _, err := DFlashZLabForward(m, noise, target[:len(target)-1], ctxLen, blockLen); err == nil {
			t.Fatal("a truncated target hidden must fail")
		}
	})
	t.Run("zero block", func(t *testing.T) {
		if _, err := DFlashZLabForward(m, nil, target, ctxLen, 0); err == nil {
			t.Fatal("blockLen 0 must fail")
		}
	})
	t.Run("layer count mismatch", func(t *testing.T) {
		bad := *m
		bad.Layers = bad.Layers[:1] // config still declares dzLayers
		if _, err := DFlashZLabForward(&bad, noise, target, ctxLen, blockLen); err == nil {
			t.Fatal("a payload/config layer mismatch must fail")
		}
	})
}

// TestDFlashZLabForward_Ugly covers degenerate-but-valid edges: a single-token
// block with context, checked against the oracle rather than by shape alone.
func TestDFlashZLabForward_Ugly(t *testing.T) {
	m, w, arch := dzBuild(dzTiny, 400)
	const ctxLen = 3
	noise := dzSeeded(1*dzHidden, 930)
	target := dzSeeded(ctxLen*dzAux*dzHidden, 931)
	got, err := DFlashZLabForward(m, noise, target, ctxLen, 1)
	if err != nil {
		t.Fatalf("blockLen=1: %v", err)
	}
	want, _, err := decodedflash.ZLabForward(w, arch, noise, target, ctxLen, 1)
	if err != nil {
		t.Fatalf("oracle: %v", err)
	}
	dzAssertClose(t, "final", got, want, 1e-5, 1e-4)
}

// --- real-checkpoint gate (env-gated; skips cleanly without it) ---

// dzOracleFixture mirrors decode/dflash's zlabOracleFixture — the pinned
// inputs + expected outputs computed from the real checkpoint (see that file
// and docs/design-dflash-survey.md §5 for provenance + reproduction).
type dzOracleFixture struct {
	CtxLen          int         `json:"ctx_len"`
	BlockSize       int         `json:"block_size"`
	NoiseEmbedding  [][]float32 `json:"noise_embedding"`
	TargetHiddenRaw [][]float32 `json:"target_hidden_raw"`
	Depths          struct {
		FinalNorm [][]float32 `json:"final_norm"`
	} `json:"depths"`
}

func dzFlatten(rows [][]float32) []float32 {
	if len(rows) == 0 {
		return nil
	}
	out := make([]float32, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// TestDFlashZLabForward_RealCheckpoint loads the REAL z-lab/Qwen3-4B-DFlash-b16
// weights through the arch package (config + safetensors → payload) and gates
// the engine forward's final-norm output against the pinned oracle fixture —
// the same receipt decode/dflash's TestZLabForward_RealCheckpoint lands for
// the reference. Skips cleanly without LTHN_DFLASH_ZLAB_CKPT:
//
//	python -c "from huggingface_hub import snapshot_download; \
//	  print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16', \
//	  allow_patterns=['config.json','model.safetensors']))"
//	MLX_METALLIB_PATH=... LTHN_DFLASH_ZLAB_CKPT=<dir> \
//	  go test -tags metal_runtime -count=1 -run DFlashZLab ./engine/metal/ -v
func TestDFlashZLabForward_RealCheckpoint(t *testing.T) {
	dir := core.Getenv("LTHN_DFLASH_ZLAB_CKPT")
	if core.Trim(dir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_CKPT to a local z-lab/Qwen3-4B-DFlash-b16 snapshot (see test doc comment)")
	}
	fixtureBytes := core.ReadFile("../../decode/dflash/testdata/zlab_qwen3_4b_oracle.json")
	if !fixtureBytes.OK {
		t.Fatalf("read oracle fixture: %v", fixtureBytes.Err())
	}
	var fx dzOracleFixture
	if r := core.JSONUnmarshal(fixtureBytes.Value.([]byte), &fx); !r.OK {
		t.Fatalf("parse oracle fixture: %v", r.Err())
	}
	m, err := zlabdflash.Load(dir)
	if err != nil {
		t.Fatalf("load real checkpoint through the arch package: %v", err)
	}
	final, err := DFlashZLabForward(m, dzFlatten(fx.NoiseEmbedding), dzFlatten(fx.TargetHiddenRaw), fx.CtxLen, fx.BlockSize)
	if err != nil {
		t.Fatalf("DFlashZLabForward on real weights: %v", err)
	}
	// The 0.02 relative band decode/dflash's own real gate uses — wide enough
	// for two independent implementations' accumulation order, tight enough
	// that any structural slip fails by orders of magnitude.
	dzAssertClose(t, "final_norm", final, dzFlatten(fx.Depths.FinalNorm), 0.02, 0.02)
}
