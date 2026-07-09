// SPDX-Licence-Identifier: EUPL-1.2

// Package budget turns a token count into a placement decision (RFC
// §6.13). It counts a request's prompt tokens against a candidate endpoint and
// answers the two questions routing/residency ask before placing the request:
// does prompt + expected completion fit the endpoint's context window (§6.11),
// and does the working set fit the device's memory budget (§6.2/§6.16).
//
// The real tokeniser lives in go-mlx (locally) or the provider's encoding
// (remotely); this package only consumes a count, so a Counter is injected. The
// pure predicates (FitsWindow, FitsMemory) take no Counter at all.
//
//	b := budget.New(mlxCounter)
//	d := b.Decide(messages, "gemma-4-31b", 512, ep)
//	switch d.Decision {
//	case budget.DecisionFits:                place(ep)
//	case budget.DecisionNeedsTransform:      transformThenPlace(ep)   // §6.11
//	case budget.DecisionNeedsLargerEndpoint: routeToRoomierDevice()   // §6.2
//	case budget.DecisionOverflows:           fallOutToProvider()      // §6.2
//	}
package budget

import chat "dappco.re/go/inference/serving/chat"

// Counter returns the prompt-token total for messages under model's tokeniser
// (go-mlx locally, the provider's encoding remotely). It is the only piece
// budgeting borrows from a real model; everything else here is arithmetic.
// Budgeting only needs each turn's role + text to size a prompt, so it consumes
// the canonical chat.Message (multimodal parts, cache-control, §6.1) and reads
// its text via chat.Message.Text.
//
//	type mlxCounter struct{ /* … */ }
//	func (mlxCounter) Count(m []chat.Message, model string) int { /* … */ }
type Counter interface {
	Count(messages []chat.Message, model string) int
}

// Endpoint is the candidate placement budgeting checks against: the model's
// context window, the device's memory budget in bytes, and a rough
// bytes-per-token working-set estimate. Each local runtime is its own endpoint
// with its own budget/quant profile (§6.2) — a 31B bf16 device and a 16 GB-GPU
// q4 device are two Endpoints.
//
//	budget.Endpoint{ContextLen: 8192, MemoryBudget: 16 << 30, BytesPerToken: 2}
type Endpoint struct {
	ContextLen    int // model context window, in tokens
	MemoryBudget  int // device memory budget, in bytes
	BytesPerToken int // rough working-set estimate per token (KV + overhead)
}

// FitsWindow reports whether promptTokens + expectedCompletion fit contextLen
// (§6.11). The boundary is inclusive — a sum exactly equal to the window fits.
// Non-positive context, or negative counts, fit nothing.
//
//	budget.FitsWindow(1000, 512, 8192) // true
//	budget.FitsWindow(7681, 512, 8192) // false (8193 > 8192)
func FitsWindow(promptTokens, expectedCompletion, contextLen int) bool {
	if contextLen <= 0 || promptTokens < 0 || expectedCompletion < 0 {
		return false
	}
	return promptTokens+expectedCompletion <= contextLen
}

// FitsMemory reports whether the working set — workingTokens * bytesPerToken —
// fits deviceBudget bytes (§6.2). The boundary is inclusive. A non-positive
// budget or bytes-per-token holds nothing (fail closed on unusable input).
//
//	budget.FitsMemory(1000, 4, 16<<30) // true
//	budget.FitsMemory(8_000_000_000, 4, 16<<30) // false (32 GB > 16 GB)
func FitsMemory(workingTokens, bytesPerToken, deviceBudget int) bool {
	if deviceBudget <= 0 || bytesPerToken <= 0 || workingTokens < 0 {
		return false
	}
	return workingTokens*bytesPerToken <= deviceBudget
}

// Decision is what routing/residency consult before placement (§6.2/§6.16). It
// is a small closed set, ordered by how recoverable the situation is: Fits →
// NeedsTransform (over window, but compressible §6.11) → NeedsLargerEndpoint
// (fits a window but not this device's memory) → Overflows (no local fix; fall
// out to a provider).
type Decision int

const (
	// DecisionFits — prompt + completion fit the window AND the working set
	// fits the device. Place the request as-is.
	DecisionFits Decision = iota
	// DecisionNeedsTransform — over the context window; compress the middle of
	// the conversation (§6.11) before placing, rather than rejecting it.
	DecisionNeedsTransform
	// DecisionNeedsLargerEndpoint — fits the window but the working set exceeds
	// this device's memory budget; route to a roomier device (§6.2).
	DecisionNeedsLargerEndpoint
	// DecisionOverflows — over BOTH the window and the device budget (or the
	// endpoint is degenerate); a transform alone won't save it, so the caller
	// must fall out to a provider (§6.2 local-first, free-first fallback).
	DecisionOverflows
)

// String renders a Decision as a stable snake_case key for logs and metrics
// (§3.2). The strings are part of the contract — callers may key on them.
//
//	core.Println(d.Decision.String()) // "needs_transform"
func (d Decision) String() string {
	switch d {
	case DecisionFits:
		return "fits"
	case DecisionNeedsTransform:
		return "needs_transform"
	case DecisionNeedsLargerEndpoint:
		return "needs_larger_endpoint"
	case DecisionOverflows:
		return "overflows"
	default:
		return "unknown"
	}
}

// Result carries the placement decision plus the counted total and the two
// underlying fit checks, so a caller can log why a request routed where it did
// without re-running the arithmetic.
type Result struct {
	Decision     Decision
	PromptTokens int  // the count the decision was made from
	FitsWindow   bool // prompt + expected completion fit the context window
	FitsMemory   bool // the working set fits the device memory budget
}

// Budget pairs a Counter with the decision logic. Construct it with New and
// reuse it across requests — it holds no per-request state.
type Budget struct {
	counter Counter
}

// New returns a Budget backed by counter. A nil counter is permitted but makes
// Decide fail closed (DecisionOverflows) — a missing tokeniser must never
// green-light a placement.
//
//	b := budget.New(mlxCounter)
func New(counter Counter) *Budget {
	return &Budget{counter: counter}
}

// Decide counts messages under model and grades the result against ep,
// returning the placement decision routing/residency consult (§6.2/§6.16).
//
// expectedCompletion is the caller's estimate of how many tokens the model will
// generate (max_tokens, §6.1). The working set is prompt + expected completion
// — the tokens that must be held resident — sized by ep.BytesPerToken.
//
// Decisions: fits window AND memory → DecisionFits; over window but memory fine
// → DecisionNeedsTransform; fits window but over memory →
// DecisionNeedsLargerEndpoint; over both (or a degenerate endpoint) →
// DecisionOverflows.
//
//	d := b.Decide(messages, "gemma-4-31b", 512, ep)
//	if d.Decision == budget.DecisionFits { place(ep) }
func (b *Budget) Decide(messages []chat.Message, model string, expectedCompletion int, ep Endpoint) Result {
	// Fail closed: no tokeniser means we can't size the request, so we must not
	// claim it fits anything.
	if b.counter == nil {
		return Result{Decision: DecisionOverflows}
	}

	prompt := b.counter.Count(messages, model)
	working := prompt + expectedCompletion

	res := Result{
		PromptTokens: prompt,
		FitsWindow:   FitsWindow(prompt, expectedCompletion, ep.ContextLen),
		FitsMemory:   FitsMemory(working, ep.BytesPerToken, ep.MemoryBudget),
	}

	switch {
	case res.FitsWindow && res.FitsMemory:
		res.Decision = DecisionFits
	case !res.FitsWindow && res.FitsMemory:
		// Over the window only — a context transform (§6.11) can make it fit.
		res.Decision = DecisionNeedsTransform
	case res.FitsWindow && !res.FitsMemory:
		// Window's fine, this device can't hold the working set — go roomier.
		res.Decision = DecisionNeedsLargerEndpoint
	default:
		// Over both — no local device/transform combination saves it.
		res.Decision = DecisionOverflows
	}
	return res
}
