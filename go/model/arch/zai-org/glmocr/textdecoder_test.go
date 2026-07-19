// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"fmt"
	"testing"
)

// buildTextWeightsFromGolden maps testdata/block_goldens.json's "text".state_dict (captured
// straight off a REAL torch GlmOcrTextModel's state_dict()) onto this package's TextWeights.
func buildTextWeightsFromGolden(t *testing.T, g textBlockGolden) *TextWeights {
	t.Helper()
	sd := g.StateDict
	get := func(name string) []float32 {
		v, ok := sd[name]
		if !ok {
			t.Fatalf("golden text state_dict missing %q", name)
		}
		return v
	}
	hidden, ff, headDim := g.Config.HiddenSize, g.Config.IntermediateSize, g.Config.HeadDim
	heads, kvHeads := g.Config.NumAttentionHeads, g.Config.NumKeyValueHeads

	layer := func(i int) TextLayerWeights {
		p := func(s string) string { return fmt.Sprintf("layers.%d.%s", i, s) }
		fused := get(p("mlp.gate_up_proj.weight"))
		return TextLayerWeights{
			InputNorm:        RMSNormWeights{Weight: get(p("input_layernorm.weight"))},
			PostAttnNorm:     RMSNormWeights{Weight: get(p("post_attention_layernorm.weight"))},
			PostSelfAttnNorm: RMSNormWeights{Weight: get(p("post_self_attn_layernorm.weight"))},
			PostMLPNorm:      RMSNormWeights{Weight: get(p("post_mlp_layernorm.weight"))},
			Attn: TextAttnWeights{
				Q: LinearWeights{Weight: get(p("self_attn.q_proj.weight")), In: hidden, Out: heads * headDim},
				K: LinearWeights{Weight: get(p("self_attn.k_proj.weight")), In: hidden, Out: kvHeads * headDim},
				V: LinearWeights{Weight: get(p("self_attn.v_proj.weight")), In: hidden, Out: kvHeads * headDim},
				O: LinearWeights{Weight: get(p("self_attn.o_proj.weight")), In: heads * headDim, Out: hidden},
			},
			MLP: TextMLPWeights{
				Gate: LinearWeights{Weight: fused[0 : ff*hidden], In: hidden, Out: ff},
				Up:   LinearWeights{Weight: fused[ff*hidden : 2*ff*hidden], In: hidden, Out: ff},
				Down: LinearWeights{Weight: get(p("mlp.down_proj.weight")), In: ff, Out: hidden},
			},
		}
	}
	layers := make([]TextLayerWeights, g.Config.NumHiddenLayers)
	for i := range layers {
		layers[i] = layer(i)
	}
	return &TextWeights{
		EmbedTokens: get("embed_tokens.weight"),
		Layers:      layers,
		FinalNorm:   RMSNormWeights{Weight: get("norm.weight")},
	}
}

func textConfigFromGolden(g textBlockGolden) *TextConfig {
	return &TextConfig{
		HiddenSize: g.Config.HiddenSize, IntermediateSize: g.Config.IntermediateSize,
		NumHiddenLayers: g.Config.NumHiddenLayers, NumAttentionHeads: g.Config.NumAttentionHeads,
		NumKeyValueHeads: g.Config.NumKeyValueHeads, HeadDim: g.Config.HeadDim, VocabSize: g.Config.VocabSize,
		RMSNormEps: g.Config.RMSNormEps,
		RopeParameters: &RopeParameters{
			RopeTheta: g.Config.RopeTheta, MropeSection: g.Config.MropeSection, PartialRotaryFactor: g.Config.PartialRotaryFactor,
		},
	}
}

// positionAxes unpacks the golden's position_ids [3][batch][seq] into flat per-axis []int for
// batch 0 (this package's toy golden is always batch-1).
func positionAxes(t *testing.T, positionIDs [][][]int) (tPos, hPos, wPos []int) {
	t.Helper()
	if len(positionIDs) != 3 {
		t.Fatalf("golden position_ids has %d axes, want 3", len(positionIDs))
	}
	return positionIDs[0][0], positionIDs[1][0], positionIDs[2][0]
}

func TestTextLayerForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).Text
	w := buildTextWeightsFromGolden(t, g)
	tc := textConfigFromGolden(g)
	T := len(g.InputsEmbeds) / tc.HiddenSize
	tPos, hPos, wPos := positionAxes(t, g.PositionIDs)

	freqsPerPos := make([][]float32, T)
	for i := range T {
		freqsPerPos[i] = textRotaryFreqPos(tPos[i], hPos[i], wPos[i], tc.HeadDim, tc.RopeParameters.RopeTheta, tc.RopeParameters.MropeSection)
	}

	layer0 := textLayerForward(g.InputsEmbeds, T, tc.HiddenSize, tc.IntermediateSize, tc.NumAttentionHeads, tc.NumKeyValueHeads, tc.HeadDim, w.Layers[0], freqsPerPos, tc.RMSNormEps)
	if d := maxAbsDiff32(t, layer0, g.Layer0Out); d > 1e-3 {
		t.Fatalf("textLayerForward layer0 maxAbsDiff = %v, want < 1e-3", d)
	}

	layer1 := textLayerForward(layer0, T, tc.HiddenSize, tc.IntermediateSize, tc.NumAttentionHeads, tc.NumKeyValueHeads, tc.HeadDim, w.Layers[1], freqsPerPos, tc.RMSNormEps)
	if d := maxAbsDiff32(t, layer1, g.Layer1Out); d > 1e-3 {
		t.Fatalf("textLayerForward layer1 maxAbsDiff = %v, want < 1e-3", d)
	}

	normed := rmsNormForward(layer1, w.FinalNorm, T, tc.HiddenSize, tc.RMSNormEps)
	if d := maxAbsDiff32(t, normed, g.NormOut); d > 1e-3 {
		t.Fatalf("final rmsNormForward maxAbsDiff = %v, want < 1e-3", d)
	}
}

func TestTextLayerForward_BlockGoldens_Bad(t *testing.T) {
	// layer0's output must NOT already equal the two-layer golden — proves the stack genuinely
	// keeps transforming, not a tautological pass.
	g := readBlockGoldens(t).Text
	w := buildTextWeightsFromGolden(t, g)
	tc := textConfigFromGolden(g)
	T := len(g.InputsEmbeds) / tc.HiddenSize
	tPos, hPos, wPos := positionAxes(t, g.PositionIDs)
	freqsPerPos := make([][]float32, T)
	for i := range T {
		freqsPerPos[i] = textRotaryFreqPos(tPos[i], hPos[i], wPos[i], tc.HeadDim, tc.RopeParameters.RopeTheta, tc.RopeParameters.MropeSection)
	}
	layer0 := textLayerForward(g.InputsEmbeds, T, tc.HiddenSize, tc.IntermediateSize, tc.NumAttentionHeads, tc.NumKeyValueHeads, tc.HeadDim, w.Layers[0], freqsPerPos, tc.RMSNormEps)
	if d := maxAbsDiff32(t, layer0, g.Layer1Out); d < 1e-3 {
		t.Fatalf("layer0 output unexpectedly already matches the layer1 golden (d=%v)", d)
	}
}

func TestTextLayerForward_Ugly(t *testing.T) {
	// causal masking: position 0's output through a full layer must be UNCHANGED by whatever
	// value sits at position 1 (it cannot attend to the future).
	one := func(n int) []float32 {
		w := make([]float32, n)
		for i := range w {
			w[i] = 0.1
		}
		return w
	}
	w := TextLayerWeights{
		InputNorm: RMSNormWeights{Weight: []float32{1, 1, 1, 1}}, PostAttnNorm: RMSNormWeights{Weight: []float32{1, 1, 1, 1}},
		PostSelfAttnNorm: RMSNormWeights{Weight: []float32{1, 1, 1, 1}}, PostMLPNorm: RMSNormWeights{Weight: []float32{1, 1, 1, 1}},
		Attn: TextAttnWeights{
			Q: LinearWeights{Weight: one(16), In: 4, Out: 4}, K: LinearWeights{Weight: one(16), In: 4, Out: 4},
			V: LinearWeights{Weight: one(16), In: 4, Out: 4}, O: LinearWeights{Weight: one(16), In: 4, Out: 4},
		},
		MLP: TextMLPWeights{
			Gate: LinearWeights{Weight: one(24), In: 4, Out: 6}, Up: LinearWeights{Weight: one(24), In: 4, Out: 6},
			Down: LinearWeights{Weight: one(24), In: 6, Out: 4},
		},
	}
	freqs := [][]float32{{0}, {0}} // headDim=2 -> half=1 frequency per position
	xA := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	xB := []float32{1, 2, 3, 4, 999, 999, 999, 999} // position 1 changed drastically
	outA := textLayerForward(xA, 2, 4, 6, 2, 2, 2, w, freqs, 1e-5)
	outB := textLayerForward(xB, 2, 4, 6, 2, 2, 2, w, freqs, 1e-5)
	if d := maxAbsDiff32(t, outA[:4], outB[:4]); d > 1e-4 {
		t.Fatalf("textLayerForward position 0 leaked from position 1's future token: maxAbsDiff = %v", d)
	}
}

func TestTextForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).Text
	w := buildTextWeightsFromGolden(t, g)
	tc := textConfigFromGolden(g)
	T := len(g.InputsEmbeds) / tc.HiddenSize
	tPos, hPos, wPos := positionAxes(t, g.PositionIDs)

	got, err := TextForward(g.InputsEmbeds, T, tc, w, tPos, hPos, wPos)
	if err != nil {
		t.Fatalf("TextForward: %v", err)
	}
	if d := maxAbsDiff32(t, got, g.NormOut); d > 1e-3 {
		t.Fatalf("TextForward maxAbsDiff = %v, want < 1e-3", d)
	}
	if d := maxAbsDiff32(t, got, g.LastHiddenState); d > 1e-3 {
		t.Fatalf("TextForward vs last_hidden_state maxAbsDiff = %v, want < 1e-3", d)
	}
}

func TestTextForward_Bad(t *testing.T) {
	if _, err := TextForward(nil, 1, nil, nil, nil, nil, nil); err == nil {
		t.Fatal("TextForward accepted nil config/weights")
	}
}

func TestTextForward_Ugly(t *testing.T) {
	tc := &TextConfig{HiddenSize: 2, IntermediateSize: 2, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 2, RMSNormEps: 1e-5, RopeParameters: &RopeParameters{RopeTheta: 10000, MropeSection: []int{1}}}
	w := &TextWeights{Layers: []TextLayerWeights{}, FinalNorm: RMSNormWeights{Weight: []float32{1, 1}}}
	// mismatched position-id slice length must refuse, not index out of range
	if _, err := TextForward([]float32{1, 2, 3, 4}, 2, tc, w, []int{0}, []int{0}, []int{0}); err == nil {
		t.Fatal("TextForward accepted position id slices shorter than T")
	}
}

func TestEmbedTokens_Good(t *testing.T) {
	table := []float32{1, 2, 3, 4, 5, 6} // vocab=3, hidden=2
	got, err := embedTokens([]int32{1, 2}, table, 2)
	if err != nil {
		t.Fatalf("embedTokens: %v", err)
	}
	want := []float32{3, 4, 5, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("embedTokens = %v, want %v", got, want)
		}
	}
}

func TestEmbedTokens_Bad(t *testing.T) {
	table := []float32{1, 2, 3, 4}
	if _, err := embedTokens([]int32{5}, table, 2); err == nil {
		t.Fatal("embedTokens accepted an out-of-vocab id")
	}
}

func TestEmbedTokens_Ugly(t *testing.T) {
	table := []float32{1, 2, 3, 4}
	if _, err := embedTokens([]int32{-1}, table, 2); err == nil {
		t.Fatal("embedTokens accepted a negative id")
	}
}

func TestArgmax32_Good(t *testing.T) {
	if got := argmax32([]float32{1, 5, 3}); got != 1 {
		t.Fatalf("argmax32 = %d, want 1", got)
	}
}

func TestArgmax32_Bad(t *testing.T) {
	if got := argmax32([]float32{0}); got != 0 {
		t.Fatalf("argmax32 single-element = %d, want 0", got)
	}
}

func TestArgmax32_Ugly(t *testing.T) {
	// ties resolve to the FIRST maximal index
	if got := argmax32([]float32{2, 5, 5, 1}); got != 1 {
		t.Fatalf("argmax32 tie-break = %d, want 1 (first max)", got)
	}
}
