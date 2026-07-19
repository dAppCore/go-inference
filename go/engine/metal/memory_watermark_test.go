// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/engine"
)

// Both metal TokenModels carry the MemoryReporter capability (mantis #1843).
var (
	_ engine.MemoryReporter = (*NativeTokenModel)(nil)
	_ engine.MemoryReporter = (*sessionTextModel)(nil)
)

// TestMemoryWatermark_Good drives a real synthetic prefill + decode and pins
// the #1843 contract on-device: the operation resets the high-water, the
// device reports non-zero allocation, and the peak is at least the live
// active reading (the fold-at-read guarantee).
func TestMemoryWatermark_Good(t *testing.T) {
	s := newBatchedPLEBF16Fixture(t)
	if err := s.PrefillTokens([]int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if _, err := s.GenerateFromCacheEach(2, -1, nil); err != nil {
		t.Fatalf("GenerateFromCacheEach: %v", err)
	}
	active := deviceAllocatedBytes()
	peak := memWatermarkPeak()
	if active == 0 {
		t.Fatal("device reports zero allocated bytes after a real operation")
	}
	if peak == 0 {
		t.Fatal("watermark reports zero peak after a real operation")
	}
	if peak < active {
		t.Fatalf("peak %d < active %d — the read-time fold must make peak cover now", peak, active)
	}
	t.Logf("active %d MiB, peak %d MiB", active>>20, peak>>20)
}

// TestMemoryWatermark_Bad pins the reset semantics: a new prefill restarts the
// high-water at the current allocation, so a stale larger peak from a previous
// operation does not leak into the next operation's report.
func TestMemoryWatermark_Bad(t *testing.T) {
	s := newBatchedPLEBF16Fixture(t)
	memPeakBytes.Store(1 << 62) // a stale, absurd previous-op watermark
	if err := s.PrefillTokens([]int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if peak := memWatermarkPeak(); peak >= 1<<62 {
		t.Fatalf("stale watermark leaked through the prefill reset: %d", peak)
	}
}
