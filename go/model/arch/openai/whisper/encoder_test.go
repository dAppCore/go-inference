// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import "testing"

// TestConv1D_Good replays toy_block_goldens.json's conv_block: a real transformers nn.Conv1d(4,8,k=3,
// s=1,p=1) call on a toy hand-built weight/input pair.
func TestConv1D_Good(t *testing.T) {
	g := readToyBlockGoldens(t)
	cb := g.ConvBlock
	geo := g.Geometry
	input := unflattenRows(cb.Input, geo.MelBins, cb.TIn)
	got := conv1D(input, cb.Conv1Weight, cb.Conv1Bias, geo.MelBins, geo.DModel, 3, 1, 1)
	want := unflattenRows(cb.Conv1Out, geo.DModel, cb.TIn)
	if len(got) != len(want) {
		t.Fatalf("conv1 out channels = %d, want %d", len(got), len(want))
	}
	for c := range got {
		if d := maxAbsDiff32(t, got[c], want[c]); d > 1e-4 {
			t.Fatalf("conv1D channel %d max abs diff = %g, want <= 1e-4", c, d)
		}
	}
}

// TestConv1D_Bad replays conv2 — the STRIDE-2 subsample (conv1's stride is 1), a different code path
// through the (ot*stride-padding) indexing.
func TestConv1D_Bad(t *testing.T) {
	g := readToyBlockGoldens(t)
	cb := g.ConvBlock
	geo := g.Geometry
	gelu1 := unflattenRows(cb.Gelu1Out, geo.DModel, cb.TIn)
	got := conv1D(gelu1, cb.Conv2Weight, cb.Conv2Bias, geo.DModel, geo.DModel, 3, 2, 1)
	wantT := cb.Conv2OutShape[2]
	want := unflattenRows(cb.Conv2Out, geo.DModel, wantT)
	if len(got[0]) != wantT {
		t.Fatalf("conv2 (stride 2) output length = %d, want %d", len(got[0]), wantT)
	}
	for c := range got {
		if d := maxAbsDiff32(t, got[c], want[c]); d > 1e-4 {
			t.Fatalf("conv2D (stride 2) channel %d max abs diff = %g, want <= 1e-4", c, d)
		}
	}
}

// TestConv1D_Ugly proves padding actually zero-pads (not wraps/reflects): a kernel=3,pad=1,stride=1 conv
// on a length-1 input must still produce exactly one output position, reading two zero neighbours.
func TestConv1D_Ugly(t *testing.T) {
	input := [][]float32{{5}}
	weight := []float32{1, 1, 1} // sum of a 3-tap window
	bias := []float32{0}
	got := conv1D(input, weight, bias, 1, 1, 3, 1, 1)
	if len(got[0]) != 1 || got[0][0] != 5 {
		t.Fatalf("conv1D(len-1 input, k=3,p=1) = %v, want [5] (both neighbours zero-padded)", got)
	}
}

func TestGeluRow_Good(t *testing.T) {
	got := geluRow([]float32{0, 1})
	if got[0] != 0 {
		t.Fatalf("geluRow[0] = %v, want 0", got[0])
	}
	if got[1] <= 0 || got[1] >= 1 {
		t.Fatalf("geluRow[1] = %v, want in (0,1)", got[1])
	}
}

func TestGeluConv_Good(t *testing.T) {
	got := geluConv([][]float32{{-10, 0, 10}})
	if got[0][0] < -0.01 || got[0][0] > 0.01 {
		t.Fatalf("geluConv[0][0] (gelu(-10)) = %v, want ~0 (gelu saturates to 0 for very negative x)", got[0][0])
	}
	if got[0][1] != 0 {
		t.Fatalf("geluConv[0][1] (gelu(0)) = %v, want exactly 0", got[0][1])
	}
	if got[0][2] < 9.9 || got[0][2] > 10.0 {
		t.Fatalf("geluConv[0][2] (gelu(10)) = %v, want ~10 (gelu saturates to x for very positive x)", got[0][2])
	}
}

// TestEncoderLayer_Good replays toy_block_goldens.json's encoder_layer: a real transformers
// WhisperEncoderLayer(config) forward on toy hand-built weights, exercising the full pre-LN
// self-attention + FFN residual block this file's EncodeAudio loop runs per layer. EncodeAudio's own
// end-to-end wiring (conv → pos → N of these layers → final norm) is proven at REAL scale by
// live_test.go's exact-transcript gate — there is no separate toy-scale EncodeAudio golden because a
// toy WhisperEncoder run would only re-exercise this same per-layer block plus trivial pos-add/transpose
// wiring already covered by the live E2E.
func TestEncoderLayer_Good(t *testing.T) {
	g := readToyBlockGoldens(t)
	el := g.EncoderLayer
	geo := g.Geometry
	w := EncoderLayerWeights{
		SelfAttnNorm: el.Weights.SelfAttnLayerNorm.layerNorm(),
		SelfAttn: AttnWeights{
			Q: el.Weights.QProj.linear(geo.DModel, geo.DModel), K: el.Weights.KProj.linear(geo.DModel, geo.DModel),
			V: el.Weights.VProj.linear(geo.DModel, geo.DModel), Out: el.Weights.OutProj.linear(geo.DModel, geo.DModel),
		},
		FinalNorm: el.Weights.FinalLayerNorm.layerNorm(),
		FC1:       el.Weights.FC1.linear(geo.DModel, geo.FFN),
		FC2:       el.Weights.FC2.linear(geo.FFN, geo.DModel),
	}
	hidden := el.Input
	residual := hidden
	normed := layerNormForward(hidden, w.SelfAttnNorm, el.T, geo.DModel)
	attnOut, err := selfAttentionForward(normed, el.T, geo.DModel, geo.Heads, false, w.SelfAttn)
	if err != nil {
		t.Fatalf("selfAttentionForward: %v", err)
	}
	hidden = addRows(residual, attnOut)
	residual = hidden
	normed = layerNormForward(hidden, w.FinalNorm, el.T, geo.DModel)
	ff := linearForward(geluRow(linearForward(normed, w.FC1, el.T)), w.FC2, el.T)
	hidden = addRows(residual, ff)

	if d := maxAbsDiff32(t, hidden, el.Output); d > 1e-3 {
		t.Fatalf("encoder layer max abs diff vs reference = %g, want <= 1e-3", d)
	}
}

func TestEncodeAudio_Bad(t *testing.T) {
	if _, err := EncodeAudio(nil, nil, nil); err == nil {
		t.Fatal("EncodeAudio accepted nil weights/config")
	}
}

// TestEncodeAudio_Ugly proves a wrong mel-bin count is refused with a specific, checkable message rather
// than an out-of-range panic deep in conv1D.
func TestEncodeAudio_Ugly(t *testing.T) {
	_, cfg := tinyWhisperTensors()
	tensors, _ := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	wrongMel := [][]float32{{1, 2, 3}} // 1 row, cfg wants NumMelBins=2
	if _, err := EncodeAudio(wrongMel, w, cfg); err == nil {
		t.Fatal("EncodeAudio accepted the wrong mel-bin count")
	}
}

// unflattenRows splits a flat channel-major slice into [channels][frames].
func unflattenRows(flat []float32, channels, frames int) [][]float32 {
	out := make([][]float32, channels)
	for c := range out {
		out[c] = flat[c*frames : (c+1)*frames]
	}
	return out
}

func TestAddRows_Good(t *testing.T) {
	got := addRows([]float32{1, 2, 3}, []float32{10, 20, 30})
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("addRows = %v, want %v", got, want)
		}
	}
}
