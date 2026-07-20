// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestParseTurboQuantCacheMode_Good pins the FIXED mode contract: bare
// turboquant = 3.5 (K4/V3 — K is the score-sensitive side), the four explicit
// widths, and the native/empty defaults mapping to nil.
func TestParseTurboQuantCacheMode_Good(t *testing.T) {
	cases := []struct {
		mode  string
		kBits int
		vBits int
		isNil bool
	}{
		{"", 0, 0, true},
		{"native", 0, 0, true},
		{"turboquant", 4, 3, false},
		{"turboquant:3.5", 4, 3, false},
		{"turboquant:4", 4, 4, false},
		{"turboquant:3", 3, 3, false},
		{"turboquant:2", 2, 2, false},
		{" Turboquant:4 ", 4, 4, false}, // trimmed + case-insensitive
	}
	for _, c := range cases {
		tq, err := parseTurboQuantCacheMode(c.mode)
		if err != nil {
			t.Fatalf("mode %q: unexpected error %v", c.mode, err)
		}
		if c.isNil {
			if tq != nil {
				t.Fatalf("mode %q: want nil (native), got %+v", c.mode, tq)
			}
			continue
		}
		if tq == nil || tq.kBits != c.kBits || tq.vBits != c.vBits {
			t.Fatalf("mode %q = %+v, want k%d v%d", c.mode, tq, c.kBits, c.vBits)
		}
	}
}

// TestParseTurboQuantCacheMode_Bad proves unknown modes refuse loudly — an
// unhonoured -kv-cache must never run silently native.
func TestParseTurboQuantCacheMode_Bad(t *testing.T) {
	for _, mode := range []string{"turboquant:5", "turboquant:1", "q8", "kq8vq4", "paged", "fp16"} {
		if _, err := parseTurboQuantCacheMode(mode); err == nil {
			t.Fatalf("mode %q: expected a refusal", mode)
		}
	}
}

// tqTestSpecs builds a minimal arch of one qualifying global owner (+ optional
// extras) for the servability gates.
func tqTestGlobalSpec(hd int) model.LayerSpec {
	return model.LayerSpec{Attention: model.GlobalAttention, HeadDim: hd}
}

// TestTQKVArchServable_Good accepts, per the layer-kind matrix
// (docs/design-tq-moe-hybrid.md): a dense stack with a qualifying global
// layer at each instantiated head dim; a MoE stack (MoE is an FFN property —
// the attention side is standard KV); and a HYBRID mix whose gated-delta
// layers are state-kind beside a qualifying plain global owner.
func TestTQKVArchServable_Good(t *testing.T) {
	tq := &tqKVConfig{kBits: 4, vBits: 3}
	for _, hd := range []int{128, 256, 512} {
		arch := model.Arch{HeadDim: hd, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention},
			tqTestGlobalSpec(hd),
		}}
		if err := tqKVArchServable(arch, nil, tq); err != nil {
			t.Fatalf("hd=%d: unexpected refusal %v", hd, err)
		}
	}
	moe := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{{Attention: model.GlobalAttention, HeadDim: 128, MoE: true}}}
	if err := tqKVArchServable(moe, nil, tq); err != nil {
		t.Fatalf("MoE arch (standard attention KV): unexpected refusal %v", err)
	}
	hybrid := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{{Mixer: model.MixerGatedDelta, CacheIndex: -1}, tqTestGlobalSpec(128)}}
	if err := tqKVArchServable(hybrid, nil, tq); err != nil {
		t.Fatalf("hybrid (gated-delta state + plain global attention): unexpected refusal %v", err)
	}
}

// TestTQKVArchServable_Bad proves each remaining decline of the layer-kind
// matrix: gated full attention (attn_output_gate — its KV lives in the
// gated/fused lane, no TQ wiring), an all-recurrent stack (nothing to
// quantise), an attention-sinks layer, and a stack with no qualifying global
// head dim (plain and MoE alike).
func TestTQKVArchServable_Bad(t *testing.T) {
	tq := &tqKVConfig{kBits: 4, vBits: 4}
	gated := model.Arch{HeadDim: 128, AttnOutputGate: true, Layer: []model.LayerSpec{
		{Mixer: model.MixerGatedDelta, CacheIndex: -1},
		tqTestGlobalSpec(128),
	}}
	if err := tqKVArchServable(gated, nil, tq); err == nil {
		t.Fatal("gated-attention (attn_output_gate) arch: expected a refusal")
	}
	allState := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{
		{Mixer: model.MixerGatedDelta, CacheIndex: -1},
		{Mixer: model.MixerGatedDelta, CacheIndex: -1},
	}}
	if err := tqKVArchServable(allState, nil, tq); err == nil {
		t.Fatal("all-recurrent stack (no KV rows anywhere): expected a refusal")
	}
	sinks := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{tqTestGlobalSpec(128)}}
	if err := tqKVArchServable(sinks, []model.LoadedLayer{{Sinks: []byte{1, 2}}}, tq); err == nil {
		t.Fatal("attention-sinks layer: expected a refusal")
	}
	noQualify := model.Arch{HeadDim: 96, Layer: []model.LayerSpec{tqTestGlobalSpec(96)}}
	if err := tqKVArchServable(noQualify, nil, tq); err == nil {
		t.Fatal("head dim 96 (no instantiation): expected a refusal")
	}
	moeNoQualify := model.Arch{HeadDim: 96, Layer: []model.LayerSpec{{Attention: model.GlobalAttention, HeadDim: 96, MoE: true}}}
	if err := tqKVArchServable(moeNoQualify, nil, tq); err == nil {
		t.Fatal("MoE head dim 96 (no instantiation): expected a refusal")
	}
}

// TestArchSpecsRequireStepToken_Good pins the carrier fork: MoE (host router)
// and gated-delta mixers route TQ onto the STATE carrier; a plain dense stack
// keeps the recorded-ICB carrier.
func TestArchSpecsRequireStepToken_Good(t *testing.T) {
	if archSpecsRequireStepToken([]model.LayerSpec{tqTestGlobalSpec(128), {Attention: model.SlidingAttention}}) {
		t.Fatal("dense stack must keep the recorded carrier")
	}
	if !archSpecsRequireStepToken([]model.LayerSpec{{Attention: model.GlobalAttention, MoE: true}}) {
		t.Fatal("MoE stack must route to the state carrier")
	}
	if !archSpecsRequireStepToken([]model.LayerSpec{{Mixer: model.MixerGatedDelta, CacheIndex: -1}, tqTestGlobalSpec(128)}) {
		t.Fatal("gated-delta hybrid stack must route to the state carrier")
	}
}

// TestTQKVGeometryOK_Good pins the qualification surface: instantiated head
// dims {128, 256, 512} × bit widths {2, 3, 4}.
func TestTQKVGeometryOK_Good(t *testing.T) {
	for _, d := range []int{128, 256, 512} {
		for _, b := range []int{2, 3, 4} {
			if !tqKVGeometryOK(b, b, d) {
				t.Fatalf("(k%d, v%d, hd %d) should qualify", b, b, d)
			}
		}
	}
	if tqKVGeometryOK(4, 4, 96) || tqKVGeometryOK(4, 4, 1024) || tqKVGeometryOK(1, 4, 128) || tqKVGeometryOK(4, 8, 128) {
		t.Fatal("non-instantiated geometry qualified")
	}
}
