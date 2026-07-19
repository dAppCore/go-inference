// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
)

// train_e2b_capture_vs_oracle_test.go is the #44 discriminator: it compares
// ForwardCaptureHiddens' per-layer capture DIRECTLY against the numpy oracle
// (testdata/e2b_mirror_oracle.py — the same reference train_real_globals_probe_test.go
// proves the HOST MIRROR matches at cosine 1.000000 on every layer), with NO host
// mirror in between. TestLoRATrainerE2BSharedKVSFT_Good's B=0 anchor compares
// capture against the host mirror and originally assumed a below-bar cosine there
// named a capture-path bug (the #391 class — a carve/fence/route defect in
// ForwardCaptureHiddens or its ICB replay). This test disproves that: capture
// disagrees with the INDEPENDENT oracle by the same amount, on the same layers
// (8-14, worst at 11), on BOTH the recorded-ICB route (default) and the fully
// serial re-encode route (LTHN_DECODE_ICB=0) — so the defect is not in the
// capture/carve mechanism at all. It is a real divergence in the shared bf16
// forward core (decode_forward_arch.go) both routes call into, that the host
// mirror does not share and that "engine decode is proven correct daily" has not
// caught (see the #44 task notes for the elimination trail: Q8 KV on/off, ICB vs
// serial, LoRA-B-zero vs raw weights, MatFormer dFF/CacheIndex assignment — all
// ruled out).
//
// Stays a LOGGED diagnostic (not a hard gate) until the real fix lands in the
// fenced engine files (decode_forward_arch.go / decode_forward_arch_icb.go) that
// own the shared forward — flip the `worst < 0.9999` branch to t.Fatalf once it
// does; this test is the exact instrument to gate it with.
//
//	E2B_BF16_DIR=<snapshot> E2B_MIRROR_ORACLE_DIR=<dump> MLX_METALLIB_PATH=... \
//	  go test -run TestForwardCaptureHiddensE2BVsOracle -v ./engine/metal/
func TestForwardCaptureHiddensE2BVsOracle(t *testing.T) {
	dir := os.Getenv("E2B_BF16_DIR")
	oracleDir := os.Getenv("E2B_MIRROR_ORACLE_DIR")
	if dir == "" || oracleDir == "" {
		t.Skip("set E2B_BF16_DIR (bf16 snapshot) and E2B_MIRROR_ORACLE_DIR (testdata/e2b_mirror_oracle.py dump) to run the capture-vs-oracle discriminator")
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
	want := readF32File(t, filepath.Join(oracleDir, "layer_out.f32"), NL*T*H)

	worst, worstL := 2.0, -1
	for li := range perLayer {
		got := bf16ToF32Slice(perLayer[li])
		cos := cosineF32(got, want[li*T*H:(li+1)*T*H])
		t.Logf("capture-vs-oracle layer %2d cosine=%.6f", li, cos)
		if cos < worst {
			worst, worstL = cos, li
		}
	}
	t.Logf("capture-vs-oracle: worst layer %d cosine=%.6f over %d layers", worstL, worst, NL)
	if worst < 0.9999 {
		t.Logf("KNOWN #44: engine capture vs the INDEPENDENT numpy oracle diverges (worst layer %d cosine=%.6f) — verified NOT a capture-path defect (identical on ICB and serial routes); the shared bf16 forward core itself disagrees with ground truth for this layer range", worstL, worst)
	}
}
