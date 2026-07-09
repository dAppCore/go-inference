// SPDX-Licence-Identifier: EUPL-1.2

// Package fusion is the multi-model deliberation pipeline (RFC.md §6.9 —
// "Fusion — Multi-model Deliberation"). It turns one request into a small panel
// of analysis models that run the prompt IN PARALLEL, a judge that synthesises
// their responses into a structured analysis (consensus, contradictions,
// partial coverage, unique insights, blind spots), and a final answer the judge
// writes from that analysis.
//
// fusion is PURE ORCHESTRATION. It owns no inference: the panel and the judge
// are injected Model values (the real implementation is the provider router,
// §6.2; this package never imports it, so it can be faked in tests). The package
// fans out, collects, guards against recursion, and assembles — the routed
// models do the thinking.
//
//	cfg := fusion.Config{
//		AnalysisModels: []fusion.Model{gemma31b, gemma26b, gemmaE4b},
//		Judge:          gemma31b,
//		Enabled:        true,
//	}
//	res, err := fusion.Run(ctx, "why is the sky blue?", cfg)
//	if err != nil { return err }
//	reply(res.Answer) // res.Analysis carries the panel deliberation
//
// A failed panel member is recorded (PanelResponse.Err), not fatal, so long as
// at least one member succeeds. With Enabled=false the panel is bypassed and the
// judge answers directly (RFC §6.9 config: short tactical prompts).
//
// Recursion is refused (RFC §6.9 "Recursion protection"): every inner call
// carries a fusion-depth marker on the context, so an analysis model that tries
// to invoke fusion again is refused rather than fanning out unbounded inference.
package fusion

import (
	"context"
	"sync"

	core "dappco.re/go"
)

// Model is the minimal contract fusion needs from a routed model: run a prompt,
// get text back. The real implementation is the provider router (RFC §6.2);
// fusion only ever calls Run, so any backend — local go-mlx, a remote provider,
// or a test fake — satisfies it.
//
//	type routed struct{ /* … */ }
//	func (r routed) Run(ctx context.Context, prompt string) (string, error) { … }
//	func (r routed) ID() string { return r.slug }
type Model interface {
	// Run executes the prompt and returns the model's completion. A non-nil
	// error means the model could not serve this call.
	Run(ctx context.Context, prompt string) (string, error)
	// ID is the model's slug (RFC §6.9 "Slugs of the parallel panel") — used to
	// label its PanelResponse in the assembled Analysis.
	ID() string
}

// Config is the fusion panel plus judge (RFC §6.9 "Config"). AnalysisModels is
// the parallel panel; Judge synthesises the panel and writes the final answer.
// Enabled=false bypasses the panel for a single request — the judge answers
// directly (RFC §6.9: `enabled: false`).
//
//	fusion.Config{AnalysisModels: panel, Judge: judge, Enabled: true}
type Config struct {
	// AnalysisModels is the panel run in parallel (RFC §6.9 step 3). Each
	// receives the original prompt. Must hold at least one model when Enabled.
	AnalysisModels []Model
	// Judge synthesises the panel into an Analysis and writes the final answer
	// (RFC §6.9 steps 4–5). Required on both the fused and bypassed paths —
	// without it there is nothing to produce an answer.
	Judge Model
	// Enabled gates the panel. false ⇒ bypass: the judge answers the prompt
	// directly with no fan-out (RFC §6.9 config default is true).
	Enabled bool
}

// PanelResponse is one analysis model's contribution to the deliberation. On
// success Text holds the model's answer and Err is nil; on failure Err is set
// and the member is recorded but not counted toward the survivors (RFC §6.9: a
// failed panel member is recorded, not fatal).
type PanelResponse struct {
	ModelID string `json:"model_id"`
	Text    string `json:"text,omitempty"`
	Err     error  `json:"-"` // recorded for the analysis; nil on success
}

// Analysis is the structured deliberation the judge produces from the panel
// (RFC §6.9 step 4). Panel holds every member's recorded response (successes and
// failures); Synthesis is the judge's combined read across them — its text is
// the raw judge output, the fields below are the §6.9 deliberation dimensions a
// caller can surface. fusion assembles the panel and carries the judge's
// synthesis verbatim; richer structured extraction (consensus vs contradiction
// segmentation) is the judge's own output, parsed downstream.
type Analysis struct {
	// Panel is every member's recorded response, in dispatch order.
	Panel []PanelResponse `json:"panel"`
	// Synthesis is the judge's combined read over the panel (the §6.9
	// "consensus, contradictions, partial coverage, unique insights, blind
	// spots" synthesis) — carried as the judge produced it.
	Synthesis string `json:"synthesis"`
}

// Result is the outcome of a fusion run: the final user-facing answer the judge
// wrote (RFC §6.9 step 5) plus the Analysis it deliberated from. Bypassed is
// true when Enabled was false and the judge answered directly with no panel.
type Result struct {
	// Answer is the final user-facing answer.
	Answer string `json:"answer"`
	// Analysis is the deliberation behind the answer (empty Panel when Bypassed).
	Analysis Analysis `json:"analysis"`
	// Bypassed is true when the panel was skipped (Config.Enabled == false).
	Bypassed bool `json:"bypassed"`
}

// fusionDepthKey is the private context key that marks a fusion as in-flight
// (RFC §6.9 "Recursion protection"). A nested Run sees the marker and refuses to
// fan out a second panel. Unexported so only this package can set or read it —
// the marker can't be spoofed from outside.
type fusionDepthKey struct{}

// markFusionActive returns a context carrying the fusion-depth marker. Every
// panel and judge call is made with this context, so any fusion they attempt to
// invoke sees it and refuses (RFC §6.9).
func markFusionActive(ctx context.Context) context.Context {
	return context.WithValue(ctx, fusionDepthKey{}, true)
}

// fusionActive reports whether ctx is already inside a fusion run.
func fusionActive(ctx context.Context) bool {
	v, ok := ctx.Value(fusionDepthKey{}).(bool)
	return ok && v
}

// Run executes a fusion deliberation over prompt (RFC.md §6.9). It dispatches
// the prompt to every Config.AnalysisModels member in parallel, records each
// response (a failed member is kept but not fatal so long as ≥1 succeeds), then
// asks Config.Judge to synthesise the surviving responses into the final answer
// and the structured Analysis.
//
// Recursion is refused: if ctx already carries the fusion-depth marker, Run
// returns an error rather than fanning out again (RFC §6.9 "Recursion
// protection"). With Config.Enabled false, Run bypasses the panel and the judge
// answers the prompt directly.
//
//	res, err := fusion.Run(ctx, "compare the two designs", cfg)
//	if err != nil { return err }
//	reply(res.Answer)
func Run(ctx context.Context, prompt string, cfg Config) (Result, error) {
	// Recursion guard (RFC §6.9): an analysis model whose own Run re-enters
	// fusion arrives here with the marker already set. Refuse — do not fan out
	// unbounded inference.
	if fusionActive(ctx) {
		return Result{}, core.E("ai.fusion", "recursive fusion refused: an analysis model cannot invoke fusion", nil)
	}

	// The judge writes the answer on every path; without it there is nothing to
	// produce a result (RFC §6.9 steps 4–5).
	if cfg.Judge == nil {
		return Result{}, core.E("ai.fusion", "no judge configured", nil)
	}

	// Mark every downstream call (panel + judge) as inside a fusion, so a nested
	// invocation is refused by the guard above.
	inner := markFusionActive(ctx)

	// Bypass (RFC §6.9: enabled=false) — the judge answers the prompt directly,
	// no panel, empty Analysis.
	if !cfg.Enabled {
		answer, err := cfg.Judge.Run(inner, prompt)
		if err != nil {
			return Result{}, core.E("ai.fusion", "judge failed on bypass path", err)
		}
		return Result{Answer: answer, Bypassed: true}, nil
	}

	// A panel of zero can never deliberate (RFC §6.9: the panel is the
	// deliberation).
	if len(cfg.AnalysisModels) == 0 {
		return Result{}, core.E("ai.fusion", "no analysis models in panel", nil)
	}

	// Fan the prompt out to every panel member in parallel (RFC §6.9 step 3).
	panel := dispatchPanel(inner, prompt, cfg.AnalysisModels)

	// At least one member must have succeeded — otherwise there is nothing to
	// synthesise and the judge is not asked to deliberate over an empty panel
	// (RFC §6.9: "as long as ≥1 succeeds").
	if !anySucceeded(panel) {
		return Result{Analysis: Analysis{Panel: panel}},
			core.E("ai.fusion", "every analysis model failed", nil)
	}

	// The same judge receives a synthesis prompt carrying every surviving panel
	// response and writes the final answer (RFC §6.9 steps 4–5). The final
	// synthesis call is given the prompt only — its freshness lives in the panel
	// responses already.
	synthesis := buildSynthesisPrompt(prompt, panel)
	answer, err := cfg.Judge.Run(inner, synthesis)
	if err != nil {
		return Result{Analysis: Analysis{Panel: panel}},
			core.E("ai.fusion", "judge failed to synthesise the panel", err)
	}

	return Result{
		Answer: answer,
		Analysis: Analysis{
			Panel:     panel,
			Synthesis: answer,
		},
	}, nil
}

// dispatchPanel runs prompt against every model concurrently and returns one
// PanelResponse per model, preserving the input order so the Analysis is
// deterministic (RFC §6.9 step 3 — parallel fan-out). A member that errors is
// recorded with its Err set, not dropped.
func dispatchPanel(ctx context.Context, prompt string, models []Model) []PanelResponse {
	out := make([]PanelResponse, len(models))
	var wg sync.WaitGroup
	wg.Add(len(models))
	for i, m := range models {
		go runPanelMember(ctx, &wg, prompt, out, i, m)
	}
	wg.Wait()
	return out
}

// runPanelMember runs one panel model and records its response at out[i]
// (RFC §6.9 step 3 — one survivor slot per member). It is lifted out of
// dispatchPanel's loop as a non-capturing function taking every dependency as a
// parameter, so each `go runPanelMember(...)` spawns no per-iteration closure
// environment on the heap — the goroutine itself and the shared WaitGroup are
// the only inherent allocations of the fan-out. out[i] is written by exactly one
// goroutine and wg.Wait happens-before dispatchPanel reads out, so the slice
// needs no lock.
func runPanelMember(ctx context.Context, wg *sync.WaitGroup, prompt string, out []PanelResponse, i int, m Model) {
	defer wg.Done()
	text, err := m.Run(ctx, prompt)
	out[i] = PanelResponse{ModelID: m.ID(), Text: text, Err: err}
}

// anySucceeded reports whether at least one panel member returned without error
// (RFC §6.9: ≥1 survivor is required to proceed to synthesis).
func anySucceeded(panel []PanelResponse) bool {
	for _, pr := range panel {
		if pr.Err == nil {
			return true
		}
	}
	return false
}

// buildSynthesisPrompt assembles the judge's synthesis prompt from the original
// prompt and the surviving panel responses (RFC §6.9 step 4). Failed members are
// omitted from the synthesis text — the judge deliberates over answers, not
// errors. Built with core string primitives only (no fmt/strings).
//
// One pre-sized core.Builder, not a per-member core.Concat: the old loop
// re-copied the whole accumulated prompt on every survivor (O(n²) in panel size
// and total text), one heap string per member. A single Grow to the exact final
// length lets WriteString fill in place — one backing allocation, and
// Builder.String() hands it back without a copy.
func buildSynthesisPrompt(prompt string, panel []PanelResponse) string {
	// The fixed framing, folded to compile-time constants (literal + literal is
	// a constant expression) so it costs no runtime concatenation. head/tail are
	// the spans either side of the original prompt.
	const head = "You are the judge in a multi-model deliberation. " +
		"Synthesise the panel responses below into a single grounded answer, " +
		"noting consensus, contradictions, partial coverage, unique insights, and blind spots.\n\n" +
		"Original prompt:\n"
	const tail = "\n\nPanel responses:\n"

	// First pass: the exact final byte length, so the builder allocates its
	// backing array once and never re-grows.
	size := len(head) + len(prompt) + len(tail)
	n := 0
	for _, pr := range panel {
		if pr.Err != nil {
			continue // a failed member contributes nothing to deliberate over
		}
		n++
		// "\n[" + Itoa(n) + "] " + ModelID + ":\n" + Text + "\n"
		size += 2 + decimalDigits(n) + 2 + len(pr.ModelID) + 2 + len(pr.Text) + 1
	}

	var b core.Builder
	b.Grow(size)
	b.WriteString(head)
	b.WriteString(prompt)
	b.WriteString(tail)
	n = 0
	for _, pr := range panel {
		if pr.Err != nil {
			continue
		}
		n++
		b.WriteString("\n[")
		b.WriteString(core.Itoa(n))
		b.WriteString("] ")
		b.WriteString(pr.ModelID)
		b.WriteString(":\n")
		b.WriteString(pr.Text)
		b.WriteString("\n")
	}
	return b.String()
}

// decimalDigits returns the number of decimal digits core.Itoa writes for n
// (n >= 1 in the synthesis loop) — used to pre-size the builder exactly so its
// backing array is allocated once. Pure arithmetic: no allocation, unlike
// measuring len(core.Itoa(n)) which would format a throwaway string per member.
func decimalDigits(n int) int {
	d := 1
	for n >= 10 {
		d++
		n /= 10
	}
	return d
}
