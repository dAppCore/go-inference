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

// TestTQKVArchServable_Good accepts a dense stack with a qualifying global
// layer at each instantiated head dim.
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
}

// TestTQKVArchServable_Bad proves each decline: MoE, a hybrid mixer, an
// attention-sinks layer, and a stack with no qualifying global head dim.
func TestTQKVArchServable_Bad(t *testing.T) {
	tq := &tqKVConfig{kBits: 4, vBits: 4}
	moe := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{{Attention: model.GlobalAttention, HeadDim: 128, MoE: true}}}
	if err := tqKVArchServable(moe, nil, tq); err == nil {
		t.Fatal("MoE arch: expected a refusal")
	}
	hybrid := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{{Mixer: model.MixerGatedDelta}, tqTestGlobalSpec(128)}}
	if err := tqKVArchServable(hybrid, nil, tq); err == nil {
		t.Fatal("hybrid (gated-delta) arch: expected a refusal")
	}
	sinks := model.Arch{HeadDim: 128, Layer: []model.LayerSpec{tqTestGlobalSpec(128)}}
	if err := tqKVArchServable(sinks, []model.LoadedLayer{{Sinks: []byte{1, 2}}}, tq); err == nil {
		t.Fatal("attention-sinks layer: expected a refusal")
	}
	noQualify := model.Arch{HeadDim: 96, Layer: []model.LayerSpec{tqTestGlobalSpec(96)}}
	if err := tqKVArchServable(noQualify, nil, tq); err == nil {
		t.Fatal("head dim 96 (no instantiation): expected a refusal")
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
