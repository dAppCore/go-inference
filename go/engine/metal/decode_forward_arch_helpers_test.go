// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestSlideWindowBounds(t *testing.T) {
	tests := []struct {
		name      string
		pos, win  int
		wantStart int
		wantN     int
	}{
		{name: "global first", pos: 0, win: 0, wantStart: 0, wantN: 1},
		{name: "global later", pos: 5, win: 0, wantStart: 0, wantN: 6},
		{name: "inside sliding window", pos: 2, win: 4, wantStart: 0, wantN: 3},
		{name: "at sliding edge", pos: 3, win: 4, wantStart: 0, wantN: 4},
		{name: "past sliding edge", pos: 6, win: 4, wantStart: 3, wantN: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			start, n := slideWindow(tt.pos, tt.win)
			if start != tt.wantStart || n != tt.wantN {
				t.Fatalf("slideWindow(%d, %d) = (%d, %d), want (%d, %d)", tt.pos, tt.win, start, n, tt.wantStart, tt.wantN)
			}
		})
	}
}

func TestArchPLEPayloadSelection(t *testing.T) {
	if got, err := singleArchPLEBF16("bf16", nil); err != nil || got != nil {
		t.Fatalf("singleArchPLEBF16(nil) = (%v, %v), want (nil, nil)", got, err)
	}
	one := ArchPLEBF16{VocabPLI: 8}
	got, err := singleArchPLEBF16("bf16", []ArchPLEBF16{one})
	if err != nil {
		t.Fatalf("singleArchPLEBF16(one): %v", err)
	}
	if got == nil || got.VocabPLI != one.VocabPLI {
		t.Fatalf("singleArchPLEBF16(one) = %+v, want payload", got)
	}
	if _, err := singleArchPLEBF16("bf16", []ArchPLEBF16{{}, {}}); err == nil {
		t.Fatal("singleArchPLEBF16(two) error = nil")
	}

	q := ArchPLEQuant{VocabPLI: 16}
	qgot, err := singleArchPLEQuant("quant", []ArchPLEQuant{q})
	if err != nil {
		t.Fatalf("singleArchPLEQuant(one): %v", err)
	}
	if qgot == nil || qgot.VocabPLI != q.VocabPLI {
		t.Fatalf("singleArchPLEQuant(one) = %+v, want payload", qgot)
	}
	if _, err := singleArchPLEQuant("quant", []ArchPLEQuant{{}, {}}); err == nil {
		t.Fatal("singleArchPLEQuant(two) error = nil")
	}
}

func TestArchPLERuntimeValidation(t *testing.T) {
	if got, dim, err := archPLEBF16Runtime("bf16", nil, 2, 3, 4, 1e-5); err != nil || got != nil || dim != 0 {
		t.Fatalf("archPLEBF16Runtime(nil) = (%v, %d, %v), want nil runtime", got, dim, err)
	}
	if _, _, err := archPLEBF16Runtime("bf16", &ArchPLEBF16{TokenIDs: []int32{1}}, 2, 3, 4, 1e-5); err == nil {
		t.Fatal("archPLEBF16Runtime(token mismatch) error = nil")
	}
	if _, _, err := archPLEBF16Runtime("bf16", &ArchPLEBF16{TokenIDs: []int32{1, 2, 3}}, 2, 3, 4, 1e-5); err == nil {
		t.Fatal("archPLEBF16Runtime(empty geometry) error = nil")
	}
	bf16Payload := &ArchPLEBF16{
		TokenIDs:           []int32{1, 2, 3},
		VocabPLI:           8,
		PliDim:             3,
		PerLayerProjNormW:  make([]byte, 3*bf16Size),
		EmbedPerLayer:      make([]byte, 8*3*bf16Size),
		PerLayerModelProjW: make([]byte, 2*3*4*bf16Size),
	}
	runtime, dim, err := archPLEBF16Runtime("bf16", bf16Payload, 2, 3, 4, 1e-5)
	if err != nil {
		t.Fatalf("archPLEBF16Runtime(valid): %v", err)
	}
	if runtime == nil || dim != 3 {
		t.Fatalf("archPLEBF16Runtime(valid) = (%v, %d), want runtime dim 3", runtime, dim)
	}

	if _, _, err := archPLEQuantRuntime("quant", &ArchPLEQuant{TokenIDs: []int32{1, 2, 3}}, 2, 3, 4, 1e-5); err == nil {
		t.Fatal("archPLEQuantRuntime(empty geometry) error = nil")
	}
	quantPayload := &ArchPLEQuant{
		TokenIDs:          []int32{1, 2, 3},
		VocabPLI:          8,
		PliDim:            4,
		GroupSize:         2,
		Bits:              4,
		ProjGroupSize:     2,
		ProjBits:          4,
		PerLayerProjNormW: make([]byte, 4*bf16Size),
	}
	qruntime, qdim, err := archPLEQuantRuntime("quant", quantPayload, 2, 3, 4, 1e-5)
	if err != nil {
		t.Fatalf("archPLEQuantRuntime(valid): %v", err)
	}
	if qruntime == nil || qdim != 4 {
		t.Fatalf("archPLEQuantRuntime(valid) = (%v, %d), want runtime dim 4", qruntime, qdim)
	}
}

func TestArchPLELayerShapeValidation(t *testing.T) {
	const dModel, pliDim, groupSize, bits = 4, 2, 2, 4
	bf16Layer := DecodeLayerWeights{
		PerLayerGate:           make([]byte, pliDim*dModel*bf16Size),
		PerLayerProjection:     make([]byte, dModel*pliDim*bf16Size),
		PostPerLayerInputNormW: make([]byte, dModel*bf16Size),
	}
	ple, err := bf16PLELayers("bf16", []DecodeLayerWeights{bf16Layer}, dModel, pliDim)
	if err != nil {
		t.Fatalf("bf16PLELayers(valid): %v", err)
	}
	if len(ple) != 1 || len(ple[0].gate.Packed) != len(bf16Layer.PerLayerGate) {
		t.Fatalf("bf16PLELayers(valid) = %+v, want one shaped layer", ple)
	}
	if _, err := bf16PLELayers("bf16", []DecodeLayerWeights{{PerLayerGate: []byte{1}}}, dModel, pliDim); err == nil {
		t.Fatal("bf16PLELayers(invalid) error = nil")
	}

	weight := func(outDim, inDim int) QuantWeight {
		return QuantWeight{
			Packed: make([]byte, outDim*inDim*bits/8),
			Scales: make([]byte, outDim*(inDim/groupSize)*bf16Size),
			Biases: make([]byte, outDim*(inDim/groupSize)*bf16Size),
		}
	}
	if !quantWeightBytesOK(weight(pliDim, dModel), pliDim, dModel, groupSize, bits) {
		t.Fatal("quantWeightBytesOK(valid) = false")
	}
	if quantWeightBytesOK(QuantWeight{Packed: []byte{1}}, pliDim, dModel, groupSize, bits) {
		t.Fatal("quantWeightBytesOK(invalid) = true")
	}
	quantLayer := QuantizedLayerWeights{
		PerLayerGate:           weight(pliDim, dModel),
		PerLayerProjection:     weight(dModel, pliDim),
		PostPerLayerInputNormW: make([]byte, dModel*bf16Size),
	}
	qple, err := quantPLELayers("quant", []QuantizedLayerWeights{quantLayer}, dModel, pliDim, groupSize, bits)
	if err != nil {
		t.Fatalf("quantPLELayers(valid): %v", err)
	}
	if len(qple) != 1 || qple[0].groupSize != groupSize || qple[0].bits != bits {
		t.Fatalf("quantPLELayers(valid) = %+v, want one shaped quant layer", qple)
	}
	if _, err := quantPLELayers("quant", []QuantizedLayerWeights{{PerLayerGate: QuantWeight{Packed: []byte{1}}}}, dModel, pliDim, groupSize, bits); err == nil {
		t.Fatal("quantPLELayers(invalid) error = nil")
	}
}
