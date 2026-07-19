// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
)

// train_e2b_sublayer_probe_test.go is the #49 drill-down under
// TestForwardCaptureHiddensE2BVsOracle: the discriminator proves the shared bf16
// forward diverges from the numpy oracle (testdata/e2b_mirror_oracle.py) on E2B
// layers 8-14, but a whole-layer cosine cannot name the sublayer. This probe runs
// the same capture on the FORCED-SERIAL route (icbDisabledForTest — the only route
// that fills capturedAttnHiddens) and splits every layer at its attention seam:
// the post-attention residual (x + Wo·attn, the engine's hBuf) against the
// oracle's attn_res.f32, and the layer output against layer_out.f32. A layer
// whose attn-half is clean but whose output is dirty indicts the MLP/PLE/scalar
// half; a dirty attn-half over a clean input indicts attention itself.
//
//	E2B_BF16_DIR=<snapshot> E2B_MIRROR_ORACLE_DIR=<dump> MLX_METALLIB_PATH=... \
//	  go test -run TestE2BSublayerProbeVsOracle -v ./engine/metal/
func TestE2BSublayerProbeVsOracle(t *testing.T) {
	dir := os.Getenv("E2B_BF16_DIR")
	oracleDir := os.Getenv("E2B_MIRROR_ORACLE_DIR")
	if dir == "" || oracleDir == "" {
		t.Skip("set E2B_BF16_DIR (bf16 snapshot) and E2B_MIRROR_ORACLE_DIR (testdata/e2b_mirror_oracle.py dump) to run the sublayer probe")
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

	tr, err := NewLoRATrainer(tm, inference.TrainingConfig{
		LoRA:         inference.LoRAConfig{Rank: 2, Alpha: 4, TargetKeys: []string{ProjQ, ProjV}},
		LearningRate: 0.02,
	})
	if err != nil {
		t.Fatalf("NewLoRATrainer: %v", err)
	}
	defer func() { _ = tr.Close() }()

	// Station -1 — the per-layer PLE weight identity: transpose-invariant element sums
	// of each layer's gate/projection/post-norm plus the layer scalar, printed for a
	// python-side match against the checkpoint tensors. A shifted or misassigned
	// per-layer table shows up here as sums matching a NEIGHBOURING layer's tensors.
	sumBF16 := func(b []byte) float64 {
		var s float64
		for i := 0; i+1 < len(b); i += 2 {
			s += float64(bf16ToF32(b[i], b[i+1]))
		}
		return s
	}
	for li := range tr.sess.state.ple {
		pl := tr.sess.state.ple[li]
		var sc float32
		if buf := tr.sess.state.lb[li].layerScalar; buf != nil {
			sb := tr.sess.state.bufferBytes(buf, bf16Size)
			sc = bf16ToF32(sb[0], sb[1])
		}
		t.Logf("plew layer %2d gateSum=%.6f projSum=%.6f postNormSum=%.6f scalar=%.8f", li, sumBF16(pl.gate.Packed), sumBF16(pl.proj.Packed), sumBF16(pl.postNorm), sc)
	}

	oldICBDisabled := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = oldICBDisabled }()
	prevAttn, prevMLP := capturedAttnHiddens, capturedMLPResHiddens
	capturedAttnHiddens, capturedMLPResHiddens = nil, nil
	defer func() { capturedAttnHiddens, capturedMLPResHiddens = prevAttn, prevMLP }()

	parityIDs := []int32{1204, 2381, 977, 4102, 355, 2048, 613, 1777} // the #42 harness's parity ids
	_, perLayer, err := tr.sess.ForwardCaptureHiddens(parityIDs)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}

	T, H, NL := len(parityIDs), tm.arch.Hidden, len(tm.arch.Layer)
	wantOut := readF32File(t, filepath.Join(oracleDir, "layer_out.f32"), NL*T*H)
	wantAttn := readF32File(t, filepath.Join(oracleDir, "attn_res.f32"), NL*T*H)

	// Station 0 — the per-layer-input tensor itself (the oracle's pli): the host mirror
	// was FED the oracle's pli rows, so the engine's own PerLayerInputs computation has
	// never been compared against ground truth. A dirty row here poisons exactly the
	// layers whose gate consumes it, entering through the layer's SECOND half.
	if tr.sess.perLayerInput != nil {
		PLID := tm.arch.PerLayerInputHidden
		wantPLI := readF32File(t, filepath.Join(oracleDir, "pli.f32"), T*NL*PLID)
		gotPLI := make([][]float32, T) // [T][NL*PLID]
		for tok, id := range parityIDs {
			emb, eerr := tr.sess.embed(id)
			if eerr != nil {
				t.Fatalf("embed(%d): %v", id, eerr)
			}
			row, perr := tr.sess.perLayerInput(id, emb)
			if perr != nil {
				t.Fatalf("perLayerInput(%d): %v", id, perr)
			}
			gotPLI[tok] = bf16ToF32Slice(row)
		}
		for li := range NL {
			got := make([]float32, T*PLID)
			want := make([]float32, T*PLID)
			for tok := range T {
				copy(got[tok*PLID:(tok+1)*PLID], gotPLI[tok][li*PLID:(li+1)*PLID])
				copy(want[tok*PLID:(tok+1)*PLID], wantPLI[tok*NL*PLID+li*PLID:tok*NL*PLID+(li+1)*PLID])
			}
			t.Logf("pli row layer %2d cosine=%.6f", li, cosineF32(got, want))
		}
	}

	if len(capturedAttnHiddens) != T*NL {
		t.Fatalf("capturedAttnHiddens: %d entries, want %d (T=%d NL=%d — serial capture not engaged?)", len(capturedAttnHiddens), T*NL, T, NL)
	}
	if len(capturedMLPResHiddens) != T*NL {
		t.Fatalf("capturedMLPResHiddens: %d entries, want %d", len(capturedMLPResHiddens), T*NL)
	}
	// The captures are token-major [t*NL+l]; re-pack per layer as [T,H].
	repack := func(cap [][]byte, l int) []float32 {
		rows := make([]float32, T*H)
		for tok := range T {
			copy(rows[tok*H:(tok+1)*H], bf16ToF32Slice(cap[tok*NL+l]))
		}
		return rows
	}

	var wantMLP []float32
	if _, serr := os.Stat(filepath.Join(oracleDir, "mlp_res.f32")); serr == nil {
		wantMLP = readF32File(t, filepath.Join(oracleDir, "mlp_res.f32"), NL*T*H)
	}
	worst, worstL, worstStation := 2.0, -1, ""
	for li := range perLayer {
		outCos := cosineF32(bf16ToF32Slice(perLayer[li]), wantOut[li*T*H:(li+1)*T*H])
		attnCos := cosineF32(repack(capturedAttnHiddens, li), wantAttn[li*T*H:(li+1)*T*H])
		mlpCos := -2.0
		if wantMLP != nil {
			mlpCos = cosineF32(repack(capturedMLPResHiddens, li), wantMLP[li*T*H:(li+1)*T*H])
		}
		t.Logf("sublayer layer %2d attn-half=%.6f mlp-half=%.6f layer-out=%.6f", li, attnCos, mlpCos, outCos)
		if attnCos < worst {
			worst, worstL, worstStation = attnCos, li, "attn-half"
		}
		if wantMLP != nil && mlpCos < worst {
			worst, worstL, worstStation = mlpCos, li, "mlp-half"
		}
		if outCos < worst {
			worst, worstL, worstStation = outCos, li, "layer-out"
		}
	}
	t.Logf("sublayer probe: worst %s at layer %d cosine=%.6f", worstStation, worstL, worst)

	// E2B_SUBLAYER_DUMP_DIR: write the engine-side stations as f32 [NL,T,H] beside the
	// oracle's, for offline delta analysis (which sublayer VECTOR moved, not just cosines).
	if dumpDir := os.Getenv("E2B_SUBLAYER_DUMP_DIR"); dumpDir != "" {
		if err := os.MkdirAll(dumpDir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dumpDir, err)
		}
		writeF32 := func(name string, layers func(li int) []float32) {
			buf := make([]byte, 0, NL*T*H*4)
			var scratch [4]byte
			for li := range NL {
				for _, v := range layers(li) {
					binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(v))
					buf = append(buf, scratch[:]...)
				}
			}
			if err := os.WriteFile(filepath.Join(dumpDir, name), buf, 0o644); err != nil {
				t.Fatalf("write %s: %v", name, err)
			}
		}
		writeF32("engine_attn_res.f32", func(li int) []float32 { return repack(capturedAttnHiddens, li) })
		writeF32("engine_mlp_res.f32", func(li int) []float32 { return repack(capturedMLPResHiddens, li) })
		writeF32("engine_layer_out.f32", func(li int) []float32 { return bf16ToF32Slice(perLayer[li]) })
		t.Logf("sublayer stations dumped to %s", dumpDir)
	}
}
