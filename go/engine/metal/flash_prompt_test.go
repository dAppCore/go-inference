// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"github.com/tmc/apple/metal"
)

func TestSteelAttnPipeline_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if pso, ok := steelAttnPipeline(256, false, false); !ok || pso == nil {
		t.Fatal("steel attention pipeline unavailable")
	}
}

func TestSteelAttnPipeline_Bad(t *testing.T) {
	steelAttnPSOMu.Lock()
	old := steelAttnPSOCache
	steelAttnPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
	steelAttnPSOMu.Unlock()
	t.Cleanup(func() {
		steelAttnPSOMu.Lock()
		steelAttnPSOCache = old
		steelAttnPSOMu.Unlock()
	})
	if pso, ok := steelAttnPipeline(777, false, true); ok || pso != nil {
		t.Fatal("missing pipeline must decline")
	}
}

func TestSplitDAttnPipeline_Bad(t *testing.T) {
	splitDAttnPSOMu.Lock()
	old := splitDAttnPSOCache
	splitDAttnPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
	splitDAttnPSOMu.Unlock()
	t.Cleanup(func() {
		splitDAttnPSOMu.Lock()
		splitDAttnPSOCache = old
		splitDAttnPSOMu.Unlock()
	})
	if pso, ok := splitDAttnPipeline(false, false); ok || pso != nil {
		t.Fatal("missing split-D pipeline must decline")
	}
}

func TestSplitDAttnPipeline_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if pso, ok := splitDAttnPipeline(false, false); !ok || pso == nil {
		t.Fatal("split-D attention pipeline unavailable")
	}
}

func TestFlashQ8Pipeline_Bad(t *testing.T) {
	flashQ8PSOMu.Lock()
	old := flashQ8PSOCache
	flashQ8PSOCache = map[flashQ8Key]metal.MTLComputePipelineState{}
	flashQ8PSOMu.Unlock()
	t.Cleanup(func() {
		flashQ8PSOMu.Lock()
		flashQ8PSOCache = old
		flashQ8PSOMu.Unlock()
	})
	if pso, ok := flashQ8Pipeline(3, true, false); ok || pso != nil {
		t.Fatal("missing q8 pipeline must decline")
	}
}

func TestFlashQ8Pipeline_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if pso, ok := flashQ8Pipeline(1, false, false); !ok || pso == nil {
		t.Fatal("q8 flash pipeline unavailable")
	}
}

func TestFlashWinPipeline_Bad(t *testing.T) {
	flashWinPSOMu.Lock()
	old := flashWinPSOCache
	flashWinPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
	flashWinPSOMu.Unlock()
	t.Cleanup(func() {
		flashWinPSOMu.Lock()
		flashWinPSOCache = old
		flashWinPSOMu.Unlock()
	})
	if pso, ok := flashWinPipeline(false); ok || pso != nil {
		t.Fatal("missing window pipeline must decline")
	}
}

func TestFlashWinPipeline_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if pso, ok := flashWinPipeline(false); !ok || pso == nil {
		t.Fatal("window flash pipeline unavailable")
	}
}

func TestFlashPromptPipeline_Bad(t *testing.T) {
	flashPromptPSOMu.Lock()
	old := flashPromptPSOCache
	flashPromptPSOCache = map[int]metal.MTLComputePipelineState{}
	flashPromptPSOMu.Unlock()
	t.Cleanup(func() {
		flashPromptPSOMu.Lock()
		flashPromptPSOCache = old
		flashPromptPSOMu.Unlock()
	})
	if pso, ok := flashPromptPipeline(13); ok || pso != nil {
		t.Fatal("missing prompt pipeline must decline")
	}
}

func TestFlashPromptPipeline_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if pso, ok := flashPromptPipeline(256); !ok || pso == nil {
		t.Fatal("prompt flash pipeline unavailable")
	}
}

func TestFlashPromptUsable_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !flashPromptUsable(256, splitDAttnMinKV) {
		t.Fatal("supported prompt geometry declined")
	}
}

func TestFlashPromptUsable_Bad(t *testing.T) {
	if flashPromptUsable(13, 65536) {
		t.Fatal("unsupported head dim must not route to flash")
	}
}

func TestFlashPromptUsable_Ugly(t *testing.T) {
	old := flash512Enabled
	flash512Enabled = false
	t.Cleanup(func() { flash512Enabled = old })
	if flashPromptUsable(512, splitDAttnMinKV-1) {
		t.Fatal("disabled shallow split-D must not route to flash")
	}
}

func TestFlashQ8Usable_Bad(t *testing.T) {
	if flashQ8Usable(128, splitDAttnMinKV) {
		t.Fatal("unsupported q8 head dim must not route to flash")
	}
}

func TestFlashQ8Usable_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !flashQ8Usable(256, splitDAttnMinKV) {
		t.Fatal("supported q8 geometry declined")
	}
}

func TestFlashQ8Usable_Ugly(t *testing.T) {
	if flashQ8Usable(512, splitDAttnMinKV-1) {
		t.Fatal("shallow 512 q8 attention must not route to flash")
	}
}

func TestFlashWinUsable_Bad(t *testing.T) {
	if flashWinUsable(512) {
		t.Fatal("unsupported window head dim must not route to flash")
	}
}

func TestFlashWinUsable_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !flashWinUsable(256) {
		t.Fatal("supported window geometry declined")
	}
}

func TestGPUHasFlashPrompt_Bad(t *testing.T) {
	if gpuHasFlashPrompt(128) {
		t.Fatal("unsupported head dim must not have flash prompt")
	}
}

func TestGPUHasFlashPrompt_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasFlashPrompt(128) {
		t.Fatal("supported prompt kernel unavailable")
	}
}

func TestEncFlashPromptSDPA_Bad(t *testing.T) {
	if err := encFlashPromptSDPA(metal.MTLComputeCommandEncoderObject{}, nil, nil, nil, nil, 4, 3, 128, 2, 2, 512, 384, 0.5); err == nil {
		t.Fatal("missing unsupported pipeline must fail")
	}
}

func TestEncFlashPromptQ8_Bad(t *testing.T) {
	if err := encFlashPromptQ8(metal.MTLComputeCommandEncoderObject{}, nil, nil, nil, nil, nil, nil, 4, 2, 128, 3, 3, 512, 256, 0.5); err == nil {
		t.Fatal("unsupported q8 head dim must fail")
	}
}
