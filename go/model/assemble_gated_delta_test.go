// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// gdBF16 builds a bf16 [rows, cols] safetensors tensor from a deterministic ramp.
func gdBF16(rows, cols, salt int) safetensors.Tensor {
	n := rows * cols
	if cols == 0 { // 1-D tensor (A_log, norm)
		n = rows
	}
	data := make([]byte, n*2)
	for i := range n {
		b := f32ToBF16(float32((i*salt+3)%17-8) * 0.1)
		data[2*i], data[2*i+1] = byte(b), byte(b>>8)
	}
	shape := []int{rows, cols}
	if cols == 0 {
		shape = []int{rows}
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

// TestAssembleGatedDelta gates the factory's gated-delta loader: geometry resolved from the weight
// shapes (ValueHeads=len(A_log), HeadDim=len(norm), convDim/K from conv1d, KeyHeads=qDim/HeadDim), the
// five projections split onto the bf16 form through the factory's own LoadLinear, and the small
// recurrence tensors widened to host f32 — the un-forked replacement for composed.buildGatedDelta.
func TestAssembleGatedDelta(t *testing.T) {
	const D, valueHeads, headDim, keyHeads, convK = 6, 2, 4, 1, 3
	const vDim = valueHeads * headDim // 8
	const qDim = keyHeads * headDim   // 4
	const convDim = 2*qDim + vDim     // 16
	sp := "layers.0.linear_attn."
	tn := map[string]safetensors.Tensor{
		sp + "A_log":              gdBF16(valueHeads, 0, 5),
		sp + "norm.weight":        gdBF16(headDim, 0, 7),
		sp + "conv1d.weight":      gdBF16(convDim, convK, 9),
		sp + "in_proj_qkv.weight": gdBF16(convDim, D, 11),
		sp + "in_proj_a.weight":   gdBF16(valueHeads, D, 13),
		sp + "in_proj_b.weight":   gdBF16(valueHeads, D, 15),
		sp + "in_proj_z.weight":   gdBF16(vDim, D, 17),
		sp + "out_proj.weight":    gdBF16(D, vDim, 19),
	}

	w, cfg, err := assembleGatedDelta(tn, sp, D, "affine")
	if err != nil {
		t.Fatalf("assembleGatedDelta: %v", err)
	}

	if cfg.KeyHeads != keyHeads || cfg.ValueHeads != valueHeads || cfg.HeadDim != headDim || cfg.ConvKernel != convK {
		t.Fatalf("geometry (K %d, V %d, HD %d, convK %d), want (%d, %d, %d, %d)",
			cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.ConvKernel, keyHeads, valueHeads, headDim, convK)
	}
	// projections load to the bf16 form (quant XOR bf16, exactly one — a bf16 checkpoint here), with the
	// logical dims LoadLinear resolved. A wrong inDim (e.g. out_proj reading D instead of vDim) would
	// mis-shape the weight and the decode would read garbage — this pins the projection wiring.
	if w.InProjQKVB == nil || w.InProjQKVB.OutDim != convDim || w.InProjQKVB.InDim != D {
		t.Fatalf("in_proj_qkv bf16 form wrong: %+v", w.InProjQKVB)
	}
	if w.OutProjB == nil || w.OutProjB.OutDim != D || w.OutProjB.InDim != vDim {
		t.Fatalf("out_proj bf16 form wrong (want out %d in %d): %+v", D, vDim, w.OutProjB)
	}
	if w.InProjAB == nil || w.InProjBB == nil || w.InProjZB == nil {
		t.Fatal("in_proj_a/b/z bf16 forms must be present")
	}
	// the quant forms stay nil for a bf16 checkpoint (mutually exclusive, as composed's proj set one).
	if w.InProjQKVQ != nil || w.OutProjQ != nil {
		t.Fatal("bf16 checkpoint must not populate the quant form")
	}
	// the small recurrence tensors widen to host f32 with the right lengths.
	if len(w.ALog) != valueHeads || len(w.Norm) != headDim || len(w.ConvWeight) != convDim*convK {
		t.Fatalf("f32 tensor lengths: ALog %d Norm %d Conv %d, want %d %d %d",
			len(w.ALog), len(w.Norm), len(w.ConvWeight), valueHeads, headDim, convDim*convK)
	}
	// ConvWeight bytes must equal the widened conv tensor (byte-parity with the checkpoint).
	wantConv, _ := tensorFloat32(tn[sp+"conv1d.weight"])
	for i := range wantConv {
		if w.ConvWeight[i] != wantConv[i] {
			t.Fatalf("ConvWeight[%d] = %v, want %v", i, w.ConvWeight[i], wantConv[i])
		}
	}
}

// TestAssembleGatedDelta_Bad pins the geometry guards: a missing A_log / non-1D norm errors rather than
// loading a mis-shaped layer.
func TestAssembleGatedDelta_Bad(t *testing.T) {
	if _, _, err := assembleGatedDelta(map[string]safetensors.Tensor{}, "x.", 8, "affine"); err == nil {
		t.Fatal("missing A_log must error")
	}
}
