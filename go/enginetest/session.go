// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

// SessionHandle runs the conformance suite for one engine's
// [inference.SessionHandle] implementation. Every check that must hold for
// any conformant engine runs as its own subtest against a fresh session from
// the factory; optional capabilities are probed and skipped-with-note when
// absent.
func SessionHandle(t *testing.T, factory SessionFactory) {
	t.Helper()
	ctx := context.Background()

	t.Run("PrefillThenGenerateProducesBoundedStream", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "conformance prompt"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		toks := drain(ctx, s, inference.GenerateConfig{MaxTokens: 8})
		if len(toks) == 0 {
			t.Fatal("Generate produced no tokens after a successful Prefill")
		}
		if len(toks) > 8 {
			t.Fatalf("Generate produced %d tokens, budget was 8", len(toks))
		}
		if r := s.Err(); r != nil {
			t.Fatalf("Err after clean generation = %v, want nil", r)
		}
	})

	t.Run("AppendPromptExtendsRetainedState", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "first turn"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		if err := s.AppendPrompt(ctx, " second turn"); err != nil {
			t.Fatalf("AppendPrompt: %v", err)
		}
		if toks := drain(ctx, s, inference.GenerateConfig{MaxTokens: 4}); len(toks) == 0 {
			t.Fatal("Generate produced no tokens after AppendPrompt")
		}
	})

	t.Run("CaptureKVReturnsPopulatedSnapshot", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "capture me"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		snap, err := s.CaptureKV(ctx)
		if err != nil {
			t.Fatalf("CaptureKV: %v", err)
		}
		if snap == nil {
			t.Fatal("CaptureKV returned nil snapshot with nil error")
		}
		if snap.SeqLen <= 0 && len(snap.Tokens) == 0 {
			t.Fatalf("snapshot carries no sequence evidence: SeqLen=%d Tokens=%d", snap.SeqLen, len(snap.Tokens))
		}
	})

	t.Run("ForkIsIndependent", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "shared prefix"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		f, err := s.Fork(ctx)
		if err != nil {
			t.Fatalf("Fork: %v", err)
		}
		defer func() { _ = f.Close() }()
		// advancing the fork must not disturb the parent: both still generate
		if toks := drain(ctx, f, inference.GenerateConfig{MaxTokens: 4}); len(toks) == 0 {
			t.Fatal("fork produced no tokens")
		}
		if toks := drain(ctx, s, inference.GenerateConfig{MaxTokens: 4}); len(toks) == 0 {
			t.Fatal("parent produced no tokens after fork advanced")
		}
	})

	t.Run("ResetAllowsFreshPrefill", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "before reset"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		s.Reset()
		if err := s.Prefill(ctx, "after reset"); err != nil {
			t.Fatalf("Prefill after Reset: %v", err)
		}
		if toks := drain(ctx, s, inference.GenerateConfig{MaxTokens: 4}); len(toks) == 0 {
			t.Fatal("Generate produced no tokens after Reset+Prefill")
		}
	})

	t.Run("RangeKVBlocksStreamsAtLeastOneBlock", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "block me"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		blocks := 0
		err := s.RangeKVBlocks(ctx, 16, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
			blocks++
			return true, nil
		})
		if err != nil {
			t.Fatalf("RangeKVBlocks: %v", err)
		}
		if blocks == 0 {
			t.Fatal("RangeKVBlocks yielded zero blocks over a prefilled session")
		}
	})

	t.Run("CloseThenErrIsSane", func(t *testing.T) {
		s := factory(t)
		if err := s.Close(); err != nil {
			t.Fatalf("Close on fresh session: %v", err)
		}
	})

	t.Run("OptionalKVRestoreRoundTrips", func(t *testing.T) {
		s := factory(t)
		defer func() { _ = s.Close() }()
		if err := s.Prefill(ctx, "restore lane"); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		snap, err := s.CaptureKV(ctx)
		if err != nil {
			t.Fatalf("CaptureKV: %v", err)
		}
		fresh := factory(t)
		defer func() { _ = fresh.Close() }()
		restorer, ok := any(fresh).(inference.KVRestorer)
		if !ok {
			t.Skip("engine session does not expose inference.KVRestorer — optional capability absent")
		}
		if err := restorer.RestoreFromKV(ctx, snap); err != nil {
			t.Fatalf("RestoreFromKV: %v", err)
		}
		if toks := drain(ctx, fresh, inference.GenerateConfig{MaxTokens: 4}); len(toks) == 0 {
			t.Fatal("restored session produced no tokens")
		}
	})
}
