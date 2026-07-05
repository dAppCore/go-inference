// SPDX-Licence-Identifier: EUPL-1.2

package fusion

import (
	"context"
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	core "dappco.re/go"
)

// fakeModel is a deterministic stand-in for a routed model (RFC §6.9: the panel
// and judge are just routed models — faked here so the test exercises the
// orchestration, never real inference). It records every prompt it is Run with
// and returns a canned reply, so a test can assert the panel actually fanned
// out and that the judge saw the panel responses.
type fakeModel struct {
	id    string
	reply string

	mu      sync.Mutex
	calls   int32 // how many times Run was invoked (atomic — parallel panel)
	prompts []string
}

// Run returns the canned reply and records the prompt. Concurrency-safe so the
// parallel panel dispatch (RFC §6.9 step 3) can call it from many goroutines.
//
//	m := &fakeModel{id: "gemma-31b", reply: "the sky is blue"}
//	out, _ := m.Run(context.Background(), "why is the sky blue?")
func (m *fakeModel) Run(_ context.Context, prompt string) (string, error) {
	atomic.AddInt32(&m.calls, 1)
	m.mu.Lock()
	m.prompts = append(m.prompts, prompt)
	m.mu.Unlock()
	return m.reply, nil
}

func (m *fakeModel) ID() string { return m.id }

func (m *fakeModel) callCount() int {
	return int(atomic.LoadInt32(&m.calls))
}

func (m *fakeModel) lastPrompt() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.prompts) == 0 {
		return ""
	}
	return m.prompts[len(m.prompts)-1]
}

// failModel always errors — a panel member that can't serve (RFC §6.9: a failed
// panel member is recorded, not fatal, so long as ≥1 succeeds).
type failModel struct {
	id    string
	calls int32
}

func (m *failModel) Run(_ context.Context, _ string) (string, error) {
	atomic.AddInt32(&m.calls, 1)
	return "", core.E("fusion.test", "panel member offline", nil)
}

func (m *failModel) ID() string { return m.id }

// recursiveModel re-enters Run on the SAME fusion config from inside its own
// Run — the recursion attack (RFC §6.9 "Recursion protection"): an analysis
// model trying to fan out a nested fusion. It captures the inner Result so the
// test can assert the nested call was refused.
type recursiveModel struct {
	id        string
	cfg       Config
	innerErr  error
	innerSeen int32
}

func (m *recursiveModel) Run(ctx context.Context, prompt string) (string, error) {
	atomic.AddInt32(&m.innerSeen, 1)
	// A panel member that tries to run fusion again, fanning out unbounded
	// inference. The depth guard must refuse this.
	_, err := Run(ctx, prompt, m.cfg)
	m.innerErr = err
	return "inner-answer", nil
}

func (m *recursiveModel) ID() string { return m.id }

// --- Good ---

// TestFusion_Run_Good is the happy path (RFC §6.9 steps 3–5): the prompt fans
// out to every analysis model IN PARALLEL, the judge receives a synthesis prompt
// carrying every panel response, and the Result carries the assembled Analysis
// plus the final answer.
func TestFusion_Run_Good(t *testing.T) {
	p1 := &fakeModel{id: "gemma-31b", reply: "consensus: photons scatter"}
	p2 := &fakeModel{id: "gemma-26b", reply: "contradiction: it is teal"}
	p3 := &fakeModel{id: "gemma-e4b", reply: "unique: Rayleigh scattering"}
	judge := &fakeModel{id: "judge", reply: "the sky is blue because of Rayleigh scattering"}

	cfg := Config{
		AnalysisModels: []Model{p1, p2, p3},
		Judge:          judge,
		Enabled:        true,
	}

	res, err := Run(context.Background(), "why is the sky blue?", cfg)
	if err != nil {
		t.Fatalf("Run: unexpected error: %v", err)
	}

	// Every panel member ran exactly once — the prompt fanned out to all three.
	for _, m := range []*fakeModel{p1, p2, p3} {
		if got := m.callCount(); got != 1 {
			t.Fatalf("panel %s: want 1 call, got %d", m.id, got)
		}
		if got := m.lastPrompt(); got != "why is the sky blue?" {
			t.Fatalf("panel %s: want original prompt, got %q", m.id, got)
		}
	}

	// The judge ran once (the synthesis call) and that synthesis prompt carried
	// every panel response.
	if got := judge.callCount(); got != 1 {
		t.Fatalf("judge: want 1 call, got %d", got)
	}
	synthesis := judge.lastPrompt()
	for _, want := range []string{p1.reply, p2.reply, p3.reply} {
		if !core.Contains(synthesis, want) {
			t.Fatalf("synthesis prompt missing panel response %q\nprompt was:\n%s", want, synthesis)
		}
	}

	// The Result carries the final answer and an assembled Analysis with one
	// recorded response per panel member.
	if res.Answer != judge.reply {
		t.Fatalf("answer: want %q, got %q", judge.reply, res.Answer)
	}
	if got := len(res.Analysis.Panel); got != 3 {
		t.Fatalf("analysis panel: want 3 responses, got %d", got)
	}
	if res.Analysis.Synthesis != judge.reply {
		t.Fatalf("analysis synthesis: want %q, got %q", judge.reply, res.Analysis.Synthesis)
	}
	// Every panel member must be represented, none marked failed.
	ids := panelIDs(res.Analysis.Panel)
	for _, want := range []string{"gemma-31b", "gemma-26b", "gemma-e4b"} {
		if !contains(ids, want) {
			t.Fatalf("analysis panel missing %s (got %v)", want, ids)
		}
	}
	for _, pr := range res.Analysis.Panel {
		if pr.Err != nil {
			t.Fatalf("panel %s recorded an error on the happy path: %v", pr.ModelID, pr.Err)
		}
	}
}

// TestFusion_Run_Good_ParallelDispatch asserts the panel runs concurrently
// rather than serially (RFC §6.9 step 3 "in parallel"): every member blocks on
// a barrier until all members have started, so the run only completes if they
// all run at once. A serial dispatcher would deadlock.
func TestFusion_Run_Good_ParallelDispatch(t *testing.T) {
	const n = 4
	started := make(chan struct{}, n)
	release := make(chan struct{})

	bar := func() {
		started <- struct{}{}
		<-release // unblocks only once every member has signalled started
	}

	panel := make([]Model, n)
	for i := 0; i < n; i++ {
		panel[i] = &barrierModel{id: idFor(i), enter: bar, reply: "ok"}
	}
	judge := &fakeModel{id: "judge", reply: "final"}

	cfg := Config{AnalysisModels: panel, Judge: judge, Enabled: true}

	done := make(chan resultErr, 1)
	go func() {
		r, e := Run(context.Background(), "prompt", cfg)
		done <- resultErr{r, e}
	}()

	// Wait for every member to have started concurrently, then release them.
	for i := 0; i < n; i++ {
		<-started
	}
	close(release)

	got := <-done
	if got.err != nil {
		t.Fatalf("parallel run: unexpected error: %v", got.err)
	}
	if len(got.res.Analysis.Panel) != n {
		t.Fatalf("parallel run: want %d panel responses, got %d", n, len(got.res.Analysis.Panel))
	}
}

// --- Bad ---

// TestFusion_Run_Bad covers degraded panels (RFC §6.9: a failed panel member is
// recorded, not fatal, as long as ≥1 succeeds). One member errors; the run still
// produces a Result, the failure is recorded in the Analysis, and the judge
// synthesises from the survivors.
func TestFusion_Run_Bad(t *testing.T) {
	ok1 := &fakeModel{id: "gemma-31b", reply: "good answer one"}
	bad := &failModel{id: "gemma-26b"}
	ok2 := &fakeModel{id: "gemma-e4b", reply: "good answer two"}
	judge := &fakeModel{id: "judge", reply: "synthesised from survivors"}

	cfg := Config{
		AnalysisModels: []Model{ok1, bad, ok2},
		Judge:          judge,
		Enabled:        true,
	}

	res, err := Run(context.Background(), "prompt", cfg)
	if err != nil {
		t.Fatalf("Run: a single failed panel member must not be fatal, got: %v", err)
	}
	if res.Answer != judge.reply {
		t.Fatalf("answer: want %q, got %q", judge.reply, res.Answer)
	}

	// All three are recorded; exactly one carries an error.
	if got := len(res.Analysis.Panel); got != 3 {
		t.Fatalf("panel: want 3 recorded responses, got %d", got)
	}
	failures := 0
	for _, pr := range res.Analysis.Panel {
		if pr.Err != nil {
			failures++
			if pr.ModelID != "gemma-26b" {
				t.Fatalf("wrong member recorded as failed: %s", pr.ModelID)
			}
		}
	}
	if failures != 1 {
		t.Fatalf("want exactly 1 recorded failure, got %d", failures)
	}

	// The synthesis prompt carries the survivors' answers, not the failed one.
	synthesis := judge.lastPrompt()
	if !core.Contains(synthesis, ok1.reply) || !core.Contains(synthesis, ok2.reply) {
		t.Fatalf("synthesis prompt should carry both survivor answers, got:\n%s", synthesis)
	}
}

// TestFusion_Run_Bad_NoJudge rejects a config with no judge — there is nothing
// to synthesise the panel or write the final answer (RFC §6.9 steps 4–5).
func TestFusion_Run_Bad_NoJudge(t *testing.T) {
	p := &fakeModel{id: "p", reply: "x"}
	cfg := Config{AnalysisModels: []Model{p}, Judge: nil, Enabled: true}

	_, err := Run(context.Background(), "prompt", cfg)
	if err == nil {
		t.Fatalf("Run with no judge: want error, got nil")
	}
}

// --- Ugly ---

// TestFusion_Run_Ugly is total panel failure: every analysis model errors. With
// no surviving panel response there is nothing to synthesise, so the run errors
// rather than asking the judge to deliberate over an empty panel (RFC §6.9: "as
// long as ≥1 succeeds").
func TestFusion_Run_Ugly(t *testing.T) {
	b1 := &failModel{id: "a"}
	b2 := &failModel{id: "b"}
	b3 := &failModel{id: "c"}
	judge := &fakeModel{id: "judge", reply: "should never be reached"}

	cfg := Config{
		AnalysisModels: []Model{b1, b2, b3},
		Judge:          judge,
		Enabled:        true,
	}

	_, err := Run(context.Background(), "prompt", cfg)
	if err == nil {
		t.Fatalf("all panel members failed: want error, got nil")
	}
	// The judge must not have been asked to synthesise an empty panel.
	if got := judge.callCount(); got != 0 {
		t.Fatalf("judge should not run when the whole panel failed, got %d calls", got)
	}
	// Every member was still attempted (the fan-out happened before the verdict).
	for _, m := range []*failModel{b1, b2, b3} {
		if atomic.LoadInt32(&m.calls) != 1 {
			t.Fatalf("panel %s: want 1 attempt, got %d", m.id, m.calls)
		}
	}
}

// TestFusion_Run_Ugly_EmptyPanel rejects a config with no analysis models — a
// panel of zero can never produce a deliberation.
func TestFusion_Run_Ugly_EmptyPanel(t *testing.T) {
	judge := &fakeModel{id: "judge", reply: "x"}
	cfg := Config{AnalysisModels: nil, Judge: judge, Enabled: true}

	_, err := Run(context.Background(), "prompt", cfg)
	if err == nil {
		t.Fatalf("empty panel: want error, got nil")
	}
}

// --- Recursion guard (RFC §6.9 "Recursion protection") ---

// TestFusion_Recursion_Good confirms a normal single-level fusion is NOT treated
// as recursion: the outer Run succeeds and the depth guard only trips on a
// genuine nested fan-out, not on the first, legitimate level.
func TestFusion_Recursion_Good(t *testing.T) {
	p := &fakeModel{id: "p", reply: "answer"}
	judge := &fakeModel{id: "judge", reply: "final"}
	cfg := Config{AnalysisModels: []Model{p}, Judge: judge, Enabled: true}

	if _, err := Run(context.Background(), "prompt", cfg); err != nil {
		t.Fatalf("single-level fusion must succeed (not be mistaken for recursion): %v", err)
	}
}

// TestFusion_Recursion_Bad is the core guard: an analysis model that tries to
// invoke fusion again from inside its own Run is refused — the nested Run
// returns an error rather than fanning out a second panel (RFC §6.9: "an
// analysis model cannot recursively invoke fusion — the plugin refuses a second
// injection and returns an error rather than fanning out unbounded inference").
func TestFusion_Recursion_Bad(t *testing.T) {
	innerPanel := &fakeModel{id: "inner-panel", reply: "should never run"}
	innerJudge := &fakeModel{id: "inner-judge", reply: "should never run"}
	innerCfg := Config{AnalysisModels: []Model{innerPanel}, Judge: innerJudge, Enabled: true}

	attacker := &recursiveModel{id: "attacker", cfg: innerCfg}
	judge := &fakeModel{id: "judge", reply: "outer final"}
	cfg := Config{AnalysisModels: []Model{attacker}, Judge: judge, Enabled: true}

	res, err := Run(context.Background(), "prompt", cfg)
	if err != nil {
		t.Fatalf("outer fusion should still complete; the recursion is refused INSIDE the panel, not at the top: %v", err)
	}

	// The attacker's nested Run must have been refused.
	if attacker.innerErr == nil {
		t.Fatalf("nested fusion was not refused — the depth guard failed to trip")
	}
	if atomic.LoadInt32(&attacker.innerSeen) != 1 {
		t.Fatalf("attacker should have been dispatched exactly once, got %d", attacker.innerSeen)
	}
	// The nested panel/judge must never have fanned out a second time.
	if innerPanel.callCount() != 0 {
		t.Fatalf("nested panel fanned out — recursion not prevented (got %d calls)", innerPanel.callCount())
	}
	if innerJudge.callCount() != 0 {
		t.Fatalf("nested judge ran — recursion not prevented (got %d calls)", innerJudge.callCount())
	}
	// The outer run still produced its answer from the (recursion-refused) panel.
	if res.Answer != judge.reply {
		t.Fatalf("outer answer: want %q, got %q", judge.reply, res.Answer)
	}
}

// TestFusion_Recursion_Ugly calls Run directly on a context that already carries
// the fusion-depth marker — the guard refuses to fan out regardless of how the
// re-entry arose (defence in depth, RFC §6.9).
func TestFusion_Recursion_Ugly(t *testing.T) {
	p := &fakeModel{id: "p", reply: "x"}
	judge := &fakeModel{id: "judge", reply: "x"}
	cfg := Config{AnalysisModels: []Model{p}, Judge: judge, Enabled: true}

	// Hand Run a context that is already inside a fusion (as a nested call would
	// receive). It must refuse rather than fan out.
	ctx := markFusionActive(context.Background())
	_, err := Run(ctx, "prompt", cfg)
	if err == nil {
		t.Fatalf("Run on an already-active fusion context: want refusal, got nil")
	}
	if p.callCount() != 0 || judge.callCount() != 0 {
		t.Fatalf("guard fanned out on an active context: panel=%d judge=%d", p.callCount(), judge.callCount())
	}
}

// --- Disabled / bypass (RFC §6.9: Enabled=false bypasses the plugin) ---

// TestFusion_Disabled bypasses the panel entirely: with Enabled=false the judge
// answers directly, no panel member runs, and the Result carries the judge's
// answer with an empty Analysis (RFC §6.9 config: `enabled: false`).
func TestFusion_Disabled(t *testing.T) {
	p1 := &fakeModel{id: "p1", reply: "panel one"}
	p2 := &fakeModel{id: "p2", reply: "panel two"}
	judge := &fakeModel{id: "judge", reply: "direct answer"}

	cfg := Config{
		AnalysisModels: []Model{p1, p2},
		Judge:          judge,
		Enabled:        false,
	}

	res, err := Run(context.Background(), "tactical question", cfg)
	if err != nil {
		t.Fatalf("disabled Run: unexpected error: %v", err)
	}
	if res.Answer != judge.reply {
		t.Fatalf("disabled answer: want %q, got %q", judge.reply, res.Answer)
	}
	// No panel member ran.
	if p1.callCount() != 0 || p2.callCount() != 0 {
		t.Fatalf("panel ran while disabled: p1=%d p2=%d", p1.callCount(), p2.callCount())
	}
	// The judge saw the original prompt, not a synthesis prompt.
	if got := judge.lastPrompt(); got != "tactical question" {
		t.Fatalf("disabled judge prompt: want original, got %q", got)
	}
	if len(res.Analysis.Panel) != 0 {
		t.Fatalf("disabled run should have an empty panel, got %d", len(res.Analysis.Panel))
	}
	if !res.Bypassed {
		t.Fatalf("disabled run should be marked Bypassed")
	}
}

// TestFusion_Disabled_NoJudge — even the bypass path needs a judge to answer.
func TestFusion_Disabled_NoJudge(t *testing.T) {
	cfg := Config{AnalysisModels: []Model{&fakeModel{id: "p"}}, Judge: nil, Enabled: false}
	if _, err := Run(context.Background(), "prompt", cfg); err == nil {
		t.Fatalf("disabled Run with no judge: want error, got nil")
	}
}

// --- test helpers ---

type resultErr struct {
	res Result
	err error
}

// barrierModel blocks in Run until a barrier releases it — used to prove the
// panel dispatch is concurrent (TestFusion_Run_Good_ParallelDispatch).
type barrierModel struct {
	id    string
	enter func()
	reply string
}

func (m *barrierModel) Run(_ context.Context, _ string) (string, error) {
	m.enter()
	return m.reply, nil
}

func (m *barrierModel) ID() string { return m.id }

func idFor(i int) string { return "panel-" + string(rune('a'+i)) }

func panelIDs(prs []PanelResponse) []string {
	ids := make([]string, 0, len(prs))
	for _, pr := range prs {
		ids = append(ids, pr.ModelID)
	}
	sort.Strings(ids)
	return ids
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if h == needle {
			return true
		}
	}
	return false
}
