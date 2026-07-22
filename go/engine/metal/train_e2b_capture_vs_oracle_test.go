// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
)

// train_e2b_capture_vs_oracle_test.go is the #44/#49 ground-truth discriminator: it
// compares ForwardCaptureHiddens' per-layer capture DIRECTLY against two references,
// with no host mirror in between:
//
//   - layer_out.f32 — the f32/f64-stationed numpy oracle (testdata/e2b_mirror_oracle.py),
//     and
//   - mlx_layer_out.f32 — mlx-lm's OWN bf16 forward over the same checkpoint and ids
//     (testdata/e2b_mlx_reference.py), the ecosystem reference implementation.
//
// #49 RESOLVED the #44 divergence (layers 8-14, worst 11 at cosine 0.83): it is NOT an
// engine defect. Proven by sublayer elimination (TestE2BSublayerProbeVsOracle — every
// station recomputed in full precision from the ENGINE'S OWN inputs reproduces the
// engine's outputs to 1-cos ≤ 3e-6; per-layer PLE weights, per-layer-input rows and
// layer scalars are bit-exact vs the checkpoint): the shared bf16 forward is
// arithmetically FAITHFUL at every sublayer. The mechanism is chaotic amplification of
// correctly-rounded bf16 residual-stream storage. Gemma giant channels (|h|≈100+, one
// bf16 ULP = 0.5) accumulate ±2 ULP of storage dust over layers 0-7; E2B's layer-8 MLP
// (38 near-cliff gelu gates for parity token 6) amplifies that ~30x, and its layer-8
// per-layer-input gate (branch rms ≥ residual rms, pli values to 18) another ~50x —
// 1-cos 1e-4 in, 0.30 out, for that ONE token; layers 9-11 churn it, the pli-dominated
// giant branches at 13/14 wash it back down. mlx-lm's own bf16 forward diverges from
// the f64 oracle with the SAME signature (first bad 8, worst 11: mlx 0.8316 vs engine
// 0.8342), while engine-vs-mlx stays ≥ 0.9985 at EVERY layer — bf16 forwards form a
// tight equivalence class (their station rounding lands on the same bf16 grid) that
// the f64 trajectory exits at the chaos layers. A 0.9999-vs-f64-oracle bar is
// therefore unreachable for ANY bf16-stationed forward, the reference implementation
// included; the correct hard bar is the one gated below.
//
// HARD GATES (all three, every layer):
//
//  1. engine-vs-mlx cosine ≥ 0.9975 — bf16-class parity with the reference
//     implementation (observed floor 0.99851 at layer 11; a single wrong per-layer
//     table/boundary/kernel craters this by orders of magnitude).
//  2. engine-vs-oracle cosine ≥ mlx-vs-oracle cosine − 0.0075 — the envelope: the
//     engine may not diverge from ground truth materially more than the reference
//     does at any layer (observed worst gap −0.0045 at layer 8).
//  3. where the reference itself holds ≥ 0.9999 vs the oracle (the pre-chaos layers),
//     the engine must hold ≥ 0.9998 (observed ≥ 0.99993).
//
//	E2B_BF16_DIR=<snapshot> E2B_MIRROR_ORACLE_DIR=<dump> MLX_METALLIB_PATH=... \
//	  go test -run TestForwardCaptureHiddensE2BVsOracle -v ./engine/metal/
//
// The dump dir must hold BOTH references (run e2b_mirror_oracle.py then
// e2b_mlx_reference.py into the same directory).
func TestForwardCaptureHiddensE2BVsOracle(t *testing.T) {
	dir := os.Getenv("E2B_BF16_DIR")
	oracleDir := os.Getenv("E2B_MIRROR_ORACLE_DIR")
	if dir == "" || oracleDir == "" {
		t.Skip("set E2B_BF16_DIR (bf16 snapshot) and E2B_MIRROR_ORACLE_DIR (testdata/e2b_mirror_oracle.py + e2b_mlx_reference.py dump) to run the capture-vs-oracle discriminator")
	}
	requireNativeRuntime(t)

	lm, err := LoadTokenModelDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(%s): %v", dir, err)
	}
	tm, ok := lm.(*NativeTokenModel)
	if !ok {
		t.Fatalf("E2B load did not produce a NativeTokenModel (got %T)", lm)
	}
	defer func() { _ = tm.Close() }()

	// A frozen, un-adapted session is enough — this discriminator is about the
	// capture, not training; NewLoRATrainer just gives the same ForwardCaptureHiddens
	// entry point TestLoRATrainerE2BSharedKVSFT_Good's B=0 anchor uses.
	tr, err := NewLoRATrainer(tm, inference.TrainingConfig{
		LoRA:         inference.LoRAConfig{Rank: 2, Alpha: 4, TargetKeys: []string{ProjQ, ProjV}},
		LearningRate: 0.02,
	})
	if err != nil {
		t.Fatalf("NewLoRATrainer: %v", err)
	}
	defer func() { _ = tr.Close() }()

	parityIDs := []int32{1204, 2381, 977, 4102, 355, 2048, 613, 1777} // the #42 harness's parity ids
	_, perLayer, err := tr.sess.ForwardCaptureHiddens(parityIDs)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}

	T, H, NL := len(parityIDs), tm.arch.Hidden, len(tm.arch.Layer)
	wantOracle := readF32File(t, filepath.Join(oracleDir, "layer_out.f32"), NL*T*H)
	mlxPath := filepath.Join(oracleDir, "mlx_layer_out.f32")
	if _, serr := os.Stat(mlxPath); serr != nil {
		t.Fatalf("dump dir %s lacks mlx_layer_out.f32 — run testdata/e2b_mlx_reference.py into it (the bf16 reference the hard gate compares against): %v", oracleDir, serr)
	}
	wantMLX := readF32File(t, mlxPath, NL*T*H)

	const (
		mlxParityBar  = 0.9975 // engine-vs-mlx floor (observed 0.99851 at layer 11)
		envelopeSlack = 0.0075 // engine may trail mlx's oracle-cosine by at most this (observed 0.0045)
		preChaosMLX   = 0.9999 // layers where the bf16 reference still matches the f64 oracle...
		preChaosBar   = 0.9998 // ...the engine must too (observed ≥ 0.99993)
	)
	worst, worstL := 2.0, -1
	for li := range perLayer {
		got := bf16ToF32Slice(perLayer[li])
		oracleCos := cosineF32(got, wantOracle[li*T*H:(li+1)*T*H])
		mlxCos := cosineF32(got, wantMLX[li*T*H:(li+1)*T*H])
		mlxOracleCos := cosineF32(wantMLX[li*T*H:(li+1)*T*H], wantOracle[li*T*H:(li+1)*T*H])
		t.Logf("capture layer %2d vs-oracle=%.6f vs-mlx=%.6f (mlx-vs-oracle=%.6f)", li, oracleCos, mlxCos, mlxOracleCos)
		if mlxCos < mlxParityBar {
			t.Fatalf("layer %d: engine capture diverges from the mlx-lm bf16 reference (cosine=%.6f < %.4f) — a real engine defect, not the #49 bf16-chaos envelope", li, mlxCos, mlxParityBar)
		}
		if oracleCos < mlxOracleCos-envelopeSlack {
			t.Fatalf("layer %d: engine trails the reference's ground-truth envelope (engine-vs-oracle=%.6f, mlx-vs-oracle=%.6f, slack %.4f)", li, oracleCos, mlxOracleCos, envelopeSlack)
		}
		if mlxOracleCos >= preChaosMLX && oracleCos < preChaosBar {
			t.Fatalf("layer %d: pre-chaos layer below bar (engine-vs-oracle=%.6f < %.4f while the bf16 reference holds %.6f)", li, oracleCos, preChaosBar, mlxOracleCos)
		}
		if oracleCos < worst {
			worst, worstL = oracleCos, li
		}
	}
	t.Logf("capture-vs-oracle: worst layer %d cosine=%.6f over %d layers (the #49 bf16-chaos envelope; engine-vs-mlx ≥ %.4f everywhere)", worstL, worst, NL, mlxParityBar)
}
