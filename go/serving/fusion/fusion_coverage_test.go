// SPDX-Licence-Identifier: EUPL-1.2

package fusion

import (
	"context"
	"sync/atomic"
	"testing"
)

// TestFusion_Disabled_JudgeError covers the bypass-path judge failure (fusion.go
// §6.9 enabled=false): with the panel skipped the judge answers directly, so if
// THAT single call errors the run has no fallback and must surface the error
// (wrapped on the "judge failed on bypass path" branch) rather than returning a
// silent empty answer.
func TestFusion_Disabled_JudgeError(t *testing.T) {
	badJudge := &failModel{id: "judge"}
	// A panel member that must never run on the bypass path.
	p := &fakeModel{id: "p", reply: "should never run"}
	cfg := Config{AnalysisModels: []Model{p}, Judge: badJudge, Enabled: false}

	res, err := Run(context.Background(), "tactical question", cfg)
	if err == nil {
		t.Fatalf("disabled run with a failing judge: want error, got nil")
	}
	// Nothing useful comes back — the result is the zero Result.
	if res.Answer != "" || res.Bypassed {
		t.Fatalf("failed bypass judge should yield an empty, non-bypassed Result, got %+v", res)
	}
	// The panel was correctly skipped even though the judge then failed.
	if p.callCount() != 0 {
		t.Fatalf("panel ran on the bypass path: got %d calls", p.callCount())
	}
	// The judge was the only thing invoked.
	if atomic.LoadInt32(&badJudge.calls) != 1 {
		t.Fatalf("bypass judge: want exactly 1 attempt, got %d", badJudge.calls)
	}
}

// TestFusion_Run_Ugly_JudgeSynthesisError covers the synthesis-path judge failure
// (fusion.go §6.9 steps 4–5): the panel fans out and at least one member
// succeeds, but the judge errors while synthesising. The run must surface that
// error AND still return the assembled panel in the Analysis, so the caller can
// see the deliberation that was gathered before the judge fell over.
func TestFusion_Run_Ugly_JudgeSynthesisError(t *testing.T) {
	ok1 := &fakeModel{id: "gemma-31b", reply: "good answer one"}
	ok2 := &fakeModel{id: "gemma-e4b", reply: "good answer two"}
	badJudge := &failModel{id: "judge"}
	cfg := Config{AnalysisModels: []Model{ok1, ok2}, Judge: badJudge, Enabled: true}

	res, err := Run(context.Background(), "prompt", cfg)
	if err == nil {
		t.Fatalf("judge failing to synthesise: want error, got nil")
	}
	// No final answer, but the gathered panel is preserved for the caller.
	if res.Answer != "" {
		t.Fatalf("failed synthesis should have no answer, got %q", res.Answer)
	}
	if got := len(res.Analysis.Panel); got != 2 {
		t.Fatalf("failed synthesis should still carry the panel: want 2 responses, got %d", got)
	}
	if res.Analysis.Synthesis != "" {
		t.Fatalf("failed synthesis should leave Synthesis empty, got %q", res.Analysis.Synthesis)
	}
	// The panel really did fan out (the failure is at synthesis, not dispatch).
	for _, m := range []*fakeModel{ok1, ok2} {
		if m.callCount() != 1 {
			t.Fatalf("panel %s: want 1 call before synthesis, got %d", m.id, m.callCount())
		}
	}
	// The judge was asked to synthesise exactly once and that is where it failed.
	if atomic.LoadInt32(&badJudge.calls) != 1 {
		t.Fatalf("synthesis judge: want exactly 1 attempt, got %d", badJudge.calls)
	}
}
