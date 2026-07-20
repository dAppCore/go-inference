// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"strings"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// qwen_mtp_pair_test.go covers the reworded LoadSpeculativePair hybrid-target decline (#59 item 2 —
// see docs/design-qwen-mtp-pair.md) — the refusal-contract receipt: an unimplemented combination
// still declines by name, never crashes. No live pair loads land in this lane (the design doc names
// exactly why); these tests exercise only the decline path, which fires before any checkpoint weight
// is read, so a bare config.json in an otherwise-empty temp dir is enough to drive it.

func writeQwenMTPPairTestConfig(t *testing.T, modelType string) string {
	t.Helper()
	dir := t.TempDir()
	cfg := core.Sprintf(`{"model_type":%q}`, modelType)
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	return dir
}

// TestLoadSpeculativePairDeclinesHybridTarget_Bad proves every gated-delta hybrid model_type declines
// LoadSpeculativePair by name — never a crash, never a silent mis-pair — and that the reworded message
// names both what now exists (the drafter checkpoint parses as a real architecture) and what remains
// (the pair load itself is unwired), rather than the previous vaguer "factory pair route is pending".
func TestLoadSpeculativePairDeclinesHybridTarget_Bad(t *testing.T) {
	for _, modelType := range []string{"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe", "qwen3_next"} {
		t.Run(modelType, func(t *testing.T) {
			targetDir := writeQwenMTPPairTestConfig(t, modelType)
			draftDir := t.TempDir() // never opened — the decline fires before any checkpoint read

			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("LoadSpeculativePair panicked instead of returning an error: %v", r)
				}
			}()
			m, err := LoadSpeculativePair(targetDir, draftDir, 0)
			if err == nil {
				t.Fatalf("LoadSpeculativePair(%s) succeeded, want the gated-delta hybrid decline; got %+v", modelType, m)
			}
			got := err.Error()
			for _, want := range []string{modelType, "gated-delta hybrid", "-draft"} {
				if !strings.Contains(got, want) {
					t.Errorf("LoadSpeculativePair(%s) error = %q, want it to contain %q", modelType, got, want)
				}
			}
		})
	}
}

// TestLoadSpeculativePairHybridDeclineDoesNotFireOnPlainTarget_Good proves the hybrid guard is
// name-scoped: a non-hybrid qwen model_type is not caught by it (the load proceeds to the ordinary
// LoadDir path, which then fails for the ordinary reason — no real checkpoint in the temp dir —
// proving the hybrid-specific message never fires for a target it should not name).
func TestLoadSpeculativePairHybridDeclineDoesNotFireOnPlainTarget_Good(t *testing.T) {
	targetDir := writeQwenMTPPairTestConfig(t, "qwen3") // plain transformer, not a released hybrid id
	draftDir := t.TempDir()

	_, err := LoadSpeculativePair(targetDir, draftDir, 0)
	if err == nil {
		t.Fatal("LoadSpeculativePair on an empty temp dir succeeded, want a load error")
	}
	if strings.Contains(err.Error(), "gated-delta hybrid") {
		t.Errorf("LoadSpeculativePair(qwen3) hit the hybrid decline, want the ordinary load-failure path: %v", err)
	}
}
