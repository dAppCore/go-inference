// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestFlashQ8Usable_UnsupportedHeadDim(t *testing.T) {
	if flashQ8Usable(128, splitDAttnMinKV) {
		t.Fatal("flashQ8Usable accepted an unsupported head dimension")
	}
}

func TestFlashQ8Usable_Short512Depth(t *testing.T) {
	if flashQ8Usable(512, splitDAttnMinKV-1) {
		t.Fatal("flashQ8Usable accepted a 512-wide request below its crossover")
	}
}

func TestFlashWinUsable_UnsupportedHeadDim(t *testing.T) {
	if flashWinUsable(512) {
		t.Fatal("flashWinUsable accepted a non-256 head dimension")
	}
}

func TestGPUHasFlashPrompt_UnsupportedHeadDim(t *testing.T) {
	if gpuHasFlashPrompt(128) {
		t.Fatal("gpuHasFlashPrompt accepted an unsupported head dimension")
	}
}

func TestFlashPromptUsable_UnsupportedHeadDim(t *testing.T) {
	if flashPromptUsable(128, splitDAttnMinKV) {
		t.Fatal("flashPromptUsable accepted an unsupported head dimension")
	}
}
