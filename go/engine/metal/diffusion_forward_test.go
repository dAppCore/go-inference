// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func TestDiffusionDenoiseForwardBF16_Bad(t *testing.T) {
	arch := model.Arch{Layer: []model.LayerSpec{{}}}
	if _, err := DiffusionDenoiseForwardBF16(nil, nil, arch, []int32{1}, nil, nil, nil, nil); err == nil {
		t.Fatal("DiffusionDenoiseForwardBF16(nil model) error = nil")
	}
	if _, err := DiffusionDenoiseForwardBF16(&BF16Model{}, nil, arch, []int32{1}, nil, nil, nil, nil); err == nil {
		t.Fatal("DiffusionDenoiseForwardBF16(layer count mismatch) error = nil")
	}
	if _, err := DiffusionDenoiseForwardBF16(&BF16Model{Layers: []DecodeLayerWeights{{}}}, nil, arch, []int32{1}, nil, nil, nil, nil); err == nil {
		t.Fatal("DiffusionDenoiseForwardBF16(layerKV count mismatch) error = nil")
	}
}

func TestDiffusionDenoiseForwardBF16_EmptyCanvas_Good(t *testing.T) {
	g := &BF16Model{Layers: []DecodeLayerWeights{{}}}
	arch := model.Arch{Layer: []model.LayerSpec{{}}}
	got, err := DiffusionDenoiseForwardBF16(g, nil, arch, nil, nil, []DiffusionLayerKV{{}}, nil, nil)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16(empty canvas): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("DiffusionDenoiseForwardBF16(empty canvas) = %d bytes, want 0", len(got))
	}
}

func TestDiffusionDenoiseForwardQuant_Bad(t *testing.T) {
	arch := model.Arch{Layer: []model.LayerSpec{{}}}
	if _, err := DiffusionDenoiseForwardQuant(nil, nil, arch, []int32{1}, nil, nil, nil, nil); err == nil {
		t.Fatal("DiffusionDenoiseForwardQuant(nil model) error = nil")
	}
	if _, err := DiffusionDenoiseForwardQuant(&QuantModel{}, nil, arch, []int32{1}, nil, nil, nil, nil); err == nil {
		t.Fatal("DiffusionDenoiseForwardQuant(layer count mismatch) error = nil")
	}
	if _, err := DiffusionDenoiseForwardQuant(&QuantModel{Layers: []QuantizedLayerWeights{{}}}, nil, arch, []int32{1}, nil, nil, nil, nil); err == nil {
		t.Fatal("DiffusionDenoiseForwardQuant(layerKV count mismatch) error = nil")
	}
}

func TestDiffusionDenoiseForwardQuant_EmptyCanvas_Good(t *testing.T) {
	g := &QuantModel{Layers: []QuantizedLayerWeights{{}}}
	arch := model.Arch{Layer: []model.LayerSpec{{}}}
	got, err := DiffusionDenoiseForwardQuant(g, nil, arch, nil, nil, []DiffusionLayerKV{{}}, nil, nil)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardQuant(empty canvas): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("DiffusionDenoiseForwardQuant(empty canvas) = %d bytes, want 0", len(got))
	}
}

func TestDiffusionMulLayerScalarBF16_Good(t *testing.T) {
	h := toBF16Bytes([]float32{1, -2, 3, 4})
	got, err := diffusionMulLayerScalarBF16(h, toBF16Bytes([]float32{0.5}), 2, 2)
	if err != nil {
		t.Fatalf("diffusionMulLayerScalarBF16(scalar): %v", err)
	}
	eqBytes(t, "scalar layer scale", got, toBF16Bytes([]float32{0.5, -1, 1.5, 2}))

	got, err = diffusionMulLayerScalarBF16(h, toBF16Bytes([]float32{0.5, -1}), 2, 2)
	if err != nil {
		t.Fatalf("diffusionMulLayerScalarBF16(vector): %v", err)
	}
	eqBytes(t, "per-channel layer scale", got, toBF16Bytes([]float32{0.5, 2, 1.5, -4}))
}

func TestDiffusionMulLayerScalarBF16_Bad(t *testing.T) {
	if _, err := diffusionMulLayerScalarBF16(toBF16Bytes([]float32{1, 2}), toBF16Bytes([]float32{1, 2, 3}), 1, 2); err == nil {
		t.Fatal("diffusionMulLayerScalarBF16(wrong scalar size) error = nil")
	}
}

func TestDiffusionLayerKVGeometry_Good(t *testing.T) {
	rows := toBF16Bytes([]float32{1, 2, 3, 4}) // two K/V rows at kvDim=2
	prefixLen, start, position, err := diffusionLayerKVGeometry(DiffusionLayerKV{K: rows, V: append([]byte(nil), rows...), PrefixStart: 3}, 2)
	if err != nil {
		t.Fatalf("diffusionLayerKVGeometry(default position): %v", err)
	}
	if prefixLen != 2 || start != 3 || position != 5 {
		t.Fatalf("geometry = prefix:%d start:%d position:%d, want 2/3/5", prefixLen, start, position)
	}
	prefixLen, start, position, err = diffusionLayerKVGeometry(DiffusionLayerKV{K: rows, V: append([]byte(nil), rows...), Position: 7}, 2)
	if err != nil {
		t.Fatalf("diffusionLayerKVGeometry(inferred start): %v", err)
	}
	if prefixLen != 2 || start != 5 || position != 7 {
		t.Fatalf("inferred geometry = prefix:%d start:%d position:%d, want 2/5/7", prefixLen, start, position)
	}
}

func TestDiffusionLayerKVGeometry_Bad(t *testing.T) {
	rows := toBF16Bytes([]float32{1, 2})
	if _, _, _, err := diffusionLayerKVGeometry(DiffusionLayerKV{K: rows, V: rows, PrefixStart: -1}, 2); err == nil {
		t.Fatal("diffusionLayerKVGeometry(negative start) error = nil")
	}
	if _, _, _, err := diffusionLayerKVGeometry(DiffusionLayerKV{K: rows, V: rows, PrefixStart: 2, Position: 9}, 2); err == nil {
		t.Fatal("diffusionLayerKVGeometry(mismatched span) error = nil")
	}
}

func TestDiffusionLayerRope_Good(t *testing.T) {
	arch := model.Arch{RopeBase: 0, RotaryDim: 0, RopeLocalBase: 32000, RotaryDimLocal: 6}
	base, rotary := diffusionLayerRope(model.LayerSpec{}, arch, 8)
	if base != 10000 || rotary != 8 {
		t.Fatalf("global rope = base:%v rotary:%d, want 10000/8", base, rotary)
	}
	base, rotary = diffusionLayerRope(model.LayerSpec{Attention: model.SlidingAttention}, arch, 8)
	if base != 32000 || rotary != 6 {
		t.Fatalf("local rope = base:%v rotary:%d, want 32000/6", base, rotary)
	}
}

func TestDiffusionApplySelfConditionLinear_Bad(t *testing.T) {
	h := toBF16Bytes([]float32{1, 2})
	if _, err := diffusionApplySelfConditionLinear(h, toBF16Bytes([]float32{1}), nil, 1, 2, 2, 1e-6, "test"); err == nil {
		t.Fatal("diffusionApplySelfConditionLinear(missing weights) error = nil")
	}
	diffusion := &model.LoadedDiffusion{
		SelfCondGate: &model.Linear{OutDim: 2}, SelfCondUp: &model.Linear{OutDim: 2}, SelfCondDown: &model.Linear{},
	}
	if _, err := diffusionApplySelfConditionLinear(h, toBF16Bytes([]float32{1}), diffusion, 1, 2, 2, 1e-6, "test"); err == nil {
		t.Fatal("diffusionApplySelfConditionLinear(wrong embedding size) error = nil")
	}
}

func TestDiffusionMatRowsQuant_Good(t *testing.T) {
	got, err := diffusionMatRowsQuant(QuantWeight{}, nil, 0, 3, 2, 64, 4)
	if err != nil {
		t.Fatalf("diffusionMatRowsQuant(zero rows): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("diffusionMatRowsQuant(zero rows) = %d bytes, want 0", len(got))
	}
}

func TestDiffusionMatRowsQuant_Bad(t *testing.T) {
	if _, err := diffusionMatRowsQuant(QuantWeight{}, nil, -1, 1, 1, 64, 4); err == nil {
		t.Fatal("diffusionMatRowsQuant(negative rows) error = nil")
	}
	if _, err := diffusionMatRowsQuant(QuantWeight{}, nil, 1, 1, 1, 64, 4); err == nil {
		t.Fatal("diffusionMatRowsQuant(wrong input size) error = nil")
	}
}
