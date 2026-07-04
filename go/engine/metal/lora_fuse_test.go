// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestFuseLoRAIntoModel proves the training→serving bridge: a LoRA delta folded into a BF16Model's
// down-projection matches W + scaling·(B·A) byte-for-byte, genuinely changes the weight, and the fused
// model builds a working session — so a freshly trained adapter goes straight to native serving.
func TestFuseLoRAIntoModel(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen, rank = 64, 2, 64, 4
	scaling := float32(2.0)

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}

	a := scaleSlice(syntheticFloat32(rank*dFF, 1), 0.1)    // A [rank, dFF]
	b := scaleSlice(syntheticFloat32(dModel*rank, 2), 0.1) // B [dModel, rank]

	// independent reference fold of layer 0's wDown.
	baseWDown := append([]byte(nil), g.Layers[0].WDown...)
	ba, err := MatMulF32(b, a, dModel, rank, dFF) // [dModel, dFF]
	if err != nil {
		t.Fatalf("BA: %v", err)
	}
	want := make([]byte, len(baseWDown))
	for i := 0; i < dModel*dFF; i++ {
		v := f32ToBF16(bf16ToF32(baseWDown[2*i], baseWDown[2*i+1]) + scaling*ba[i])
		want[2*i], want[2*i+1] = byte(v), byte(v>>8)
	}

	if err := FuseLoRAIntoModel(g, []LoRADelta{{Layer: 0, Proj: "wdown", A: a, B: b, Rank: rank, Scaling: scaling}}); err != nil {
		t.Fatalf("FuseLoRAIntoModel: %v", err)
	}
	eqBytes(t, "fused wDown == base + scaling·B·A", g.Layers[0].WDown, want)

	// non-vacuous: the fuse actually changed the weight.
	same := true
	for i := range baseWDown {
		if baseWDown[i] != g.Layers[0].WDown[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("fuse left wDown unchanged — the adapter did not apply")
	}

	// the fused model serves.
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession on fused model: %v", err)
	}
	gen, err := sess.Generate([]int32{1, 2, 3}, 4, -1)
	if err != nil {
		t.Fatalf("fused session Generate: %v", err)
	}
	if len(gen) != 4 {
		t.Fatalf("fused session generated %d tokens, want 4", len(gen))
	}
	t.Logf("native LoRA fuse: wDown folded to base+scaling·B·A byte-exact, fused model serves %d tokens — train→fuse→serve, no disk round-trip", len(gen))
}
