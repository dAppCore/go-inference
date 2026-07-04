// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"context"
	"testing"
)

import core "dappco.re/go"

func TestExport_WritePackedProjectionSafetensors_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "projection.safetensors")
	projection := PackedProjection{
		Tensor: PackTensor{
			Name:        "model.layers.0.self_attn.q_proj.weight",
			Packed:      "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:      "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints:  "model.layers.0.self_attn.q_proj.weight.zeros",
			Bias:        "model.layers.0.self_attn.q_proj.bias",
			Shape:       []int32{1, 4},
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			PackedBytes: 1,
			Groups:      1,
			QMin:        -2,
			QMax:        1,
		},
		Weights: PackedWeights{
			Scheme:     SchemeW2A16,
			Format:     FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     []byte{0b11100100},
			Scales:     []float32{0.5},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
		Bias: []float32{0.25},
	}

	if err := WritePackedProjectionSafetensors(context.Background(), path, projection); err != nil {
		t.Fatalf("WritePackedProjectionSafetensors() error = %v", err)
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors:       []PackTensor{projection.Tensor},
	}
	info.normalise()

	got, err := LoadPackedProjectionFromSafetensors(info, []string{path}, projection.Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(exported) error = %v", err)
	}
	dequantized, err := DequantizePackedWeights(got.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(exported) error = %v", err)
	}

	if got.Weights.Packed[0] != projection.Weights.Packed[0] {
		t.Fatalf("packed byte = %08b, want %08b", got.Weights.Packed[0], projection.Weights.Packed[0])
	}
	assertAutoRoundFloat32SliceClose(t, dequantized, []float32{-1, -0.5, 0, 0.5}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, got.Bias, []float32{0.25}, 1e-6)
}

func TestExport_WritePackedProjectionSafetensors_Bad(t *testing.T) {
	projection := PackedProjection{
		Tensor: PackTensor{
			Name:        "weight",
			Packed:      "weight.packed",
			Scales:      "weight.scales",
			ZeroPoints:  "weight.zeros",
			Shape:       []int32{1, 4},
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			PackedBytes: 1,
			Groups:      1,
			QMin:        -2,
			QMax:        1,
		},
		Weights: PackedWeights{
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     nil,
			Scales:     []float32{1},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
	}
	if err := WritePackedProjectionSafetensors(context.Background(), "", projection); err == nil || !core.Contains(err.Error(), "path") {
		t.Fatalf("WritePackedProjectionSafetensors(empty path) error = %v, want path diagnostic", err)
	}
	if err := WritePackedProjectionSafetensors(context.Background(), core.PathJoin(t.TempDir(), "bad.safetensors"), projection); err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("WritePackedProjectionSafetensors(bad packed) error = %v, want packed length diagnostic", err)
	}
}

func TestExport_WritePackedProjectionSafetensors_Ugly(t *testing.T) {
	// A cancelled context must abort the write before any tensor is encoded,
	// rather than producing a partial file.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	projection := autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil)
	if err := WritePackedProjectionSafetensors(ctx, core.PathJoin(t.TempDir(), "cancelled.safetensors"), projection); err == nil {
		t.Fatal("WritePackedProjectionSafetensors(cancelled ctx) error = nil, want context cancellation")
	}
}

func TestExport_WritePackedProjectionsSafetensors_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	projections := []PackedProjection{
		autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, []float32{0.25}),
		autoRoundTestProjection("model.layers.0.self_attn.k_proj.weight", []byte{0b00011011}, []float32{0.25}, []float32{0}, nil),
	}

	if err := WritePackedProjectionsSafetensors(context.Background(), path, projections); err != nil {
		t.Fatalf("WritePackedProjectionsSafetensors() error = %v", err)
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors:       []PackTensor{projections[0].Tensor, projections[1].Tensor},
	}
	info.normalise()

	qProj, err := LoadPackedProjectionFromSafetensors(info, []string{path}, projections[0].Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(q_proj) error = %v", err)
	}
	kProj, err := LoadPackedProjectionFromSafetensors(info, []string{path}, projections[1].Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(k_proj) error = %v", err)
	}
	qValues, err := DequantizePackedWeights(qProj.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(q_proj) error = %v", err)
	}
	kValues, err := DequantizePackedWeights(kProj.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(k_proj) error = %v", err)
	}

	assertAutoRoundFloat32SliceClose(t, qValues, []float32{-1, -0.5, 0, 0.5}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, qProj.Bias, []float32{0.25}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, kValues, []float32{0.25, 0, -0.25, -0.5}, 1e-6)
	if len(kProj.Bias) != 0 {
		t.Fatalf("k_proj bias len = %d, want no bias", len(kProj.Bias))
	}
}

func TestExport_WritePackedProjectionsSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	first := autoRoundTestProjection("weight", []byte{0}, []float32{1}, []float32{0}, nil)
	second := autoRoundTestProjection("other", []byte{0}, []float32{1}, []float32{0}, nil)
	second.Tensor.Packed = first.Tensor.Packed

	err := WritePackedProjectionsSafetensors(context.Background(), core.PathJoin(dir, "bad.safetensors"), []PackedProjection{first, second})
	if err == nil || !core.Contains(err.Error(), "duplicate") {
		t.Fatalf("WritePackedProjectionsSafetensors(duplicate tensor) error = %v, want duplicate diagnostic", err)
	}
}

func TestExport_WritePackedProjectionsSafetensors_Ugly(t *testing.T) {
	// An empty projection set is the degenerate input: the writer rejects it
	// before opening any file.
	if err := WritePackedProjectionsSafetensors(context.Background(), core.PathJoin(t.TempDir(), "empty.safetensors"), nil); err == nil || !core.Contains(err.Error(), "at least one projection") {
		t.Fatalf("WritePackedProjectionsSafetensors(empty) error = %v, want projection diagnostic", err)
	}
	// An empty path is rejected even when projections are present.
	projections := []PackedProjection{autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil)}
	if err := WritePackedProjectionsSafetensors(context.Background(), "  ", projections); err == nil || !core.Contains(err.Error(), "path") {
		t.Fatalf("WritePackedProjectionsSafetensors(blank path) error = %v, want path diagnostic", err)
	}
}

func TestExport_WriteNativePack_Good(t *testing.T) {
	dir := t.TempDir()
	projections := []PackedProjection{
		autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, []float32{0.25}),
		autoRoundTestProjection("model.layers.0.self_attn.k_proj.weight", []byte{0b00011011}, []float32{0.25}, []float32{0}, nil),
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Scheme:        SchemeW2A16,
		ExportFormat:  FormatAutoRound,
		Iters:         1000,
		NSamples:      512,
		SeqLen:        2048,
	}

	result, err := WriteNativePack(context.Background(), dir, info, projections)
	if err != nil {
		t.Fatalf("WriteNativePack() error = %v", err)
	}
	if result.ConfigPath != core.PathJoin(dir, PackConfigFileAutoRound) || result.WeightPath != core.PathJoin(dir, "model.safetensors") || result.TensorCount != 2 {
		t.Fatalf("WriteNativePack() result = %+v, want config + model.safetensors paths and two tensors", result)
	}

	gotInfo, err := ReadPackInfo(dir)
	if err != nil {
		t.Fatalf("ReadPackInfo(exported) error = %v", err)
	}
	if gotInfo == nil || gotInfo.TensorCount != 2 || gotInfo.Scheme != SchemeW2A16 || gotInfo.ExportFormat != FormatAutoRound {
		t.Fatalf("ReadPackInfo(exported) = %+v, want W2 native tensor map", gotInfo)
	}
	if err := ValidateSafetensorsTensorMap(*gotInfo, []string{result.WeightPath}); err != nil {
		t.Fatalf("ValidateSafetensorsTensorMap(exported) error = %v", err)
	}
	qProj, err := LoadPackedProjectionFromSafetensors(*gotInfo, []string{result.WeightPath}, projections[0].Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(exported pack) error = %v", err)
	}
	values, err := DequantizePackedWeights(qProj.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(exported pack) error = %v", err)
	}
	assertAutoRoundFloat32SliceClose(t, values, []float32{-1, -0.5, 0, 0.5}, 1e-6)
}

func TestExport_WriteNativePack_Bad(t *testing.T) {
	_, err := WriteNativePack(context.Background(), t.TempDir(), PackInfo{Bits: 2, GroupSize: 32, Symmetric: true}, nil)
	if err == nil || !core.Contains(err.Error(), "projection") {
		t.Fatalf("WriteNativePack(empty projections) error = %v, want projection diagnostic", err)
	}
}

func TestExport_WriteNativePack_Ugly(t *testing.T) {
	projections := []PackedProjection{autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil)}
	// An empty root directory is the degenerate target: the writer rejects it
	// before creating any directory.
	if _, err := WriteNativePack(context.Background(), "  ", PackInfo{Bits: 2, GroupSize: 32, Symmetric: true}, projections); err == nil || !core.Contains(err.Error(), "root is empty") {
		t.Fatalf("WriteNativePack(empty root) error = %v, want root diagnostic", err)
	}
	// A pre-cancelled context aborts before any filesystem mutation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := WriteNativePack(ctx, t.TempDir(), PackInfo{Bits: 2, GroupSize: 32, Symmetric: true}, projections); err == nil {
		t.Fatal("WriteNativePack(cancelled ctx) error = nil, want context cancellation")
	}
}
