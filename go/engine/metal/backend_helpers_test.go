// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func TestNativeBackendDecodeForwardRejectsPLEWholeSequence(t *testing.T) {
	b := &NativeBackend{arch: model.Arch{PerLayerInputHidden: 1}}
	if _, err := b.DecodeForward([][]byte{{0, 1}}); err == nil {
		t.Fatal("DecodeForward(PLE whole sequence) error = nil")
	}
}

func TestNativeBackendDecodeForwardRoutesRejectInvalidInputs(t *testing.T) {
	arch := model.Arch{
		Hidden: 1, Heads: 1, KVHeads: 1, HeadDim: 1, FF: 1,
		RopeBase: 10000, Eps: 1e-5,
	}
	inputs := [][]byte{{0, 0}}

	bf16, err := NewBF16Backend(arch, nil, 1)
	if err != nil {
		t.Fatalf("NewBF16Backend: %v", err)
	}
	if _, err := bf16.DecodeForward(inputs); err == nil {
		t.Fatal("bf16 re-encode route error = nil")
	}

	bf16ICB, err := NewBF16Backend(arch, nil, 1, WithICB())
	if err != nil {
		t.Fatalf("NewBF16Backend(ICB): %v", err)
	}
	if _, err := bf16ICB.DecodeForward(inputs); err == nil {
		t.Fatal("bf16 ICB route error = nil")
	}

	quant, err := NewQuantBackend(arch, nil, 1)
	if err != nil {
		t.Fatalf("NewQuantBackend: %v", err)
	}
	if _, err := quant.DecodeForward(inputs); err == nil {
		t.Fatal("quant re-encode route error = nil")
	}

	quantICB, err := NewQuantBackend(arch, nil, 1, WithICB())
	if err != nil {
		t.Fatalf("NewQuantBackend(ICB): %v", err)
	}
	if _, err := quantICB.DecodeForward(inputs); err == nil {
		t.Fatal("quant ICB route error = nil")
	}
}

func TestNativeBackendDecodeForwardMoEICBFallsBackToReencode(t *testing.T) {
	requireNativeRuntime(t)
	arch := model.Arch{
		Hidden: 1, Heads: 1, KVHeads: 1, HeadDim: 1, FF: 1,
		RopeBase: 10000, Eps: 1e-5,
		Layer: []model.LayerSpec{{
			Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0,
			MoE: true, HeadDim: 1, KVHeads: 1,
		}},
	}
	b, err := NewBF16Backend(arch, []DecodeLayerWeights{{}}, 1, WithICB())
	if err != nil {
		t.Fatalf("NewBF16Backend(MoE ICB): %v", err)
	}
	_, err = b.DecodeForward([][]byte{{0, 0}})
	if err == nil {
		t.Fatal("MoE ICB fallback route error = nil")
	}
	if !core.Contains(err.Error(), "spec.MoE") {
		t.Fatalf("MoE ICB should fall back to re-encode validation, got %v", err)
	}
}
