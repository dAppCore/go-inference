// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// composed_quant_test.go gates the PACKED forward path on the host: a quant checkpoint's forward (every
// projection dispatched through matNTQuant → matNTQuantHost, since no engine backend is imported) must
// equal the SAME model with its packed weights dequantised and run dense. matNTQuantHost dequantises each
// weight row and dots in the same ascending-k f64 order matNTCols uses, so the two are BIT-IDENTICAL — any
// mis-wired site (a projection left on the f32 path, a wrong dim, a skipped o_proj) diverges here.

// dequantiseInPlace rewrites a quant ComposedModel into its dense equivalent: every QuantWeight is widened
// to f32 (mlxaffine.DequantizeTensor), its Q field cleared, and Quantised unset — so a second forward runs
// the standard f32 path over the identical weights. In-package: it reaches the unexported mixer weights.
func dequantiseInPlace(t *testing.T, m *ComposedModel) {
	t.Helper()
	deq := func(qw *model.QuantWeight) []float32 {
		if qw == nil {
			return nil
		}
		v, err := mlxaffine.DequantizeTensor(qw.Packed, qw.Scales, qw.Biases, qw.OutDim, qw.InDim, qw.Bits, qw.GroupSize)
		if err != nil {
			t.Fatalf("dequantise: %v", err)
		}
		return v
	}
	if m.EmbedQ != nil {
		m.Embed, m.EmbedQ = deq(m.EmbedQ), nil
	}
	if m.OutputQ != nil {
		m.Output, m.OutputQ = deq(m.OutputQ), nil
	}
	for li := range m.Layers {
		if mlp, ok := m.Layers[li].MLP.(*MLP); ok && mlp.GateQ != nil {
			mlp.Gate, mlp.Up, mlp.Down = deq(mlp.GateQ), deq(mlp.UpQ), deq(mlp.DownQ)
			mlp.GateQ, mlp.UpQ, mlp.DownQ = nil, nil, nil
		}
		switch mx := m.Layers[li].Mixer.(type) {
		case *attnMixer:
			if mx.w.QProjQ != nil {
				mx.w.QProj, mx.w.KProj, mx.w.VProj, mx.w.OProj = deq(mx.w.QProjQ), deq(mx.w.KProjQ), deq(mx.w.VProjQ), deq(mx.w.OProjQ)
				mx.w.QProjQ, mx.w.KProjQ, mx.w.VProjQ, mx.w.OProjQ = nil, nil, nil, nil
			}
		case *gatedDeltaMixer:
			if mx.w.InProjQKVQ != nil {
				mx.w.InProjQKV, mx.w.InProjA, mx.w.InProjB = deq(mx.w.InProjQKVQ), deq(mx.w.InProjAQ), deq(mx.w.InProjBQ)
				mx.w.InProjZ, mx.w.OutProj = deq(mx.w.InProjZQ), deq(mx.w.OutProjQ)
				mx.w.InProjQKVQ, mx.w.InProjAQ, mx.w.InProjBQ, mx.w.InProjZQ, mx.w.OutProjQ = nil, nil, nil, nil, nil
			}
		}
	}
	m.Quantised = false
}

// allProjNames is every 2-D projection in the synthetic hybrid (embed, head, and each layer's MLP +
// mixer projections) — the set quantiseInPlace packs so the forward exercises every quant dispatch site.
func allProjNames() []string {
	names := []string{"model.embed_tokens.weight", "lm_head.weight"}
	for i := range 4 {
		lp := "model.layers." + itoa(i) + "."
		names = append(names, lp+"mlp.gate_proj.weight", lp+"mlp.up_proj.weight", lp+"mlp.down_proj.weight")
		if (i+1)%2 == 0 {
			ap := lp + "self_attn."
			names = append(names, ap+"q_proj.weight", ap+"k_proj.weight", ap+"v_proj.weight", ap+"o_proj.weight")
		} else {
			gp := lp + "linear_attn."
			names = append(names, gp+"in_proj_qkv.weight", gp+"in_proj_a.weight", gp+"in_proj_b.weight", gp+"in_proj_z.weight", gp+"out_proj.weight")
		}
	}
	return names
}

func assertF32Identical(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: quant path %v != dequantised-dense %v", label, i, got[i], want[i])
		}
	}
}

// TestComposedQuantForwardMatchesDequantised is the wiring gate: quantise every projection (gs=8, 8-bit,
// so all inDims 8/16/32 divide cleanly), load PACKED, run a prefill + a head, then dequantise in place and
// re-run dense. The two must be bit-identical — the packed forward touched exactly the weights the dense
// forward does, at every site (embed gather, q/k/v/o, gated-delta in/out projections, gate/up/down, head).
func TestComposedQuantForwardMatchesDequantised(t *testing.T) {
	ts, config := mkHybridCheckpoint()
	for _, name := range allProjNames() {
		quantiseInPlace(t, ts, name, 8, 8)
	}
	config = append(config[:len(config)-1], []byte(`,"quantization":{"group_size":8,"bits":8}}`)...)

	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if !m.Quantised {
		t.Fatal("Quantised flag not set")
	}

	tokens := []int32{1, 5, 3, 0, 7}
	sQ := NewSession(m)
	hQ, err := sQ.Forward(tokens)
	if err != nil {
		t.Fatalf("quant Forward: %v", err)
	}
	lgQ := sQ.headLogits(hQ[(len(tokens)-1)*m.D:])

	dequantiseInPlace(t, m)
	sD := NewSession(m)
	hD, err := sD.Forward(tokens)
	if err != nil {
		t.Fatalf("dense Forward: %v", err)
	}
	lgD := sD.headLogits(hD[(len(tokens)-1)*m.D:])

	assertF32Identical(t, "hidden", hQ, hD)
	assertF32Identical(t, "logits", lgQ, lgD)
}

// TestMatNTQuantHost_MatchesDequantMatNT pins the host quant matvec against the dense reference at the
// helper level: for a synthetic packed weight, matNTQuantHost equals dequantise-then-matNT byte-for-byte
// (the row-dequant and the whole-dequant produce the same values in the same accumulation order).
func TestMatNTQuantHost_MatchesDequantMatNT(t *testing.T) {
	for _, tc := range []struct{ M, K, N, bits, gs int }{
		{1, 64, 24, 4, 64},
		{3, 128, 40, 8, 64},
		{5, 96, 16, 2, 32},
	} {
		w := make([]float32, tc.N*tc.K)
		for i := range w {
			w[i] = float32((i%13)-6) * 0.05
		}
		packed, scales, biases, err := mlxaffine.QuantizeTensor(w, tc.N, tc.K, tc.bits, tc.gs)
		if err != nil {
			t.Fatalf("quantise: %v", err)
		}
		deqW, err := mlxaffine.DequantizeTensor(packed, scales, biases, tc.N, tc.K, tc.bits, tc.gs)
		if err != nil {
			t.Fatalf("dequantise: %v", err)
		}
		x := make([]float32, tc.M*tc.K)
		for i := range x {
			x[i] = float32((i%7)-3) * 0.1
		}
		qw := &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: tc.bits, GroupSize: tc.gs, OutDim: tc.N, InDim: tc.K}
		got := matNTQuantHost(nil, x, qw, tc.M, tc.K, tc.N)
		want := matNT(x, deqW, tc.M, tc.K, tc.N)
		assertF32Identical(t, "matNTQuantHost", got, want)
	}
}
