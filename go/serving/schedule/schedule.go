// SPDX-Licence-Identifier: EUPL-1.2

// Package schedule is the continuous (in-flight) batching scheduler — the
// throughput core of a serving engine. It keeps a RUNNING SET of decoding
// sequences, admits queued requests as slots free, advances every running
// sequence one decode step per iteration, retires finished ones, admits more,
// and repeats until the queue and the running set are both empty.
//
// The package is pure policy/scheduling logic over request ids and token
// counts: it decides WHICH sequences run together and WHEN to admit the next,
// but never runs a model. The single decode step — the real forward pass over
// the batch — is injected as a Stepper, so the heavy work lives in go-mlx /
// go-inference and this package only schedules it (and is faked in tests).
//
//	out, err := schedule.New(schedule.Scheduler{
//		MaxConcurrency: 8,        // at most 8 sequences decoding together
//		MaxBatchTokens: 8192,     // running prompt+generated token budget
//	}).Run(ctx, requests, stepper, func(id string, tok int) {
//		stream(id, tok) // emit each token as it is produced
//	})
//	for _, r := range out {
//		if r.Err != nil { /* per-request typed error (e.g. oversize prompt) */ continue }
//		use(r.Tokens) // r.Finished == true
//	}
//
// Admission honours BOTH limits: a request joins the running set only while the
// set is under MaxConcurrency AND admitting its prompt keeps the running
// prompt+generated total within MaxBatchTokens. A request whose prompt alone
// exceeds MaxBatchTokens can never be admitted — it is retired immediately with
// a typed error in its Result and never blocks the loop. Context cancellation
// aborts the loop between steps (RFC.md §7 — ctx-honouring).
package schedule

import (
	"context"

	core "dappco.re/go"
)

// Request is one generation request to schedule. It is deliberately minimal —
// the prompt itself lives in the caller's world; the scheduler only needs the
// token counts to budget admission and the cap to know when a sequence is done.
//
//	schedule.Request{ID: "chat-42", PromptTokens: 312, MaxNewTokens: 256}
type Request struct {
	ID           string // caller's stable id; keys tokens and the Result
	PromptTokens int    // prompt length in tokens — counts against the batch budget
	MaxNewTokens int    // hard cap on generated tokens (a seq stops here if no EOS)
}

// Seq is the live decode state of an admitted Request inside the running set.
// A Stepper reads it to know how far each sequence has progressed; the
// scheduler owns its lifecycle (admit → step → retire).
//
//	for _, s := range running { emit(s.Request.ID, s.Generated) }
type Seq struct {
	Request   Request // the admitted request
	Generated int     // tokens produced so far (0 on the first step)
	Done      bool    // set true once the sequence has finished (EOS or cap)

	tokens []int // accumulated tokens, copied into the Result on retirement
}

// StepResult reports the outcome of advancing the running set one decode step.
// Tokens maps each running seq id to the token produced this step; Finished
// flags the ids that just completed (model EOS). The scheduler additionally
// retires any sequence that reaches its MaxNewTokens, so a Stepper need only
// signal model-driven EOS.
//
//	StepResult{Tokens: map[string]int{"a": 7}, Finished: map[string]bool{"a": true}}
type StepResult struct {
	Tokens   map[string]int  // seq id -> token produced this step
	Finished map[string]bool // seq id -> true when the model emitted EOS
}

// Stepper advances every sequence in running by exactly one token. It is the
// only model-touching dependency: a local go-mlx batch decode, a remote
// provider, or — in tests — a deterministic fake. An error fails the whole Run
// (a decode-step failure is not recoverable per-sequence).
//
//	type mlxStepper struct{ engine *mlx.Batch }
//	func (m mlxStepper) Step(ctx context.Context, running []*schedule.Seq) (schedule.StepResult, error) { … }
type Stepper interface {
	Step(ctx context.Context, running []*Seq) (StepResult, error)
}

// Scheduler configures one continuous-batching loop. Both limits gate
// admission; non-positive MaxConcurrency is clamped to 1 so the loop always
// makes progress.
type Scheduler struct {
	MaxConcurrency int // max sequences in the running set at once (clamped ≥ 1)
	MaxBatchTokens int // running prompt+generated token budget (≤ 0 ⇒ no prompt fits)
}

// Result is the per-request outcome of a Run. Finished is true when the
// sequence completed normally (EOS or MaxNewTokens); Err is non-nil for a
// request that could not be scheduled (e.g. an oversize prompt) and is mutually
// exclusive with Finished.
//
//	if r.Err != nil { fallBack(r.ID) } else { deliver(r.ID, r.Tokens) }
type Result struct {
	ID       string // the request id
	Tokens   []int  // tokens produced, in generation order
	Finished bool   // true ⇒ completed normally
	Err      error  // non-nil ⇒ never scheduled (typed core.E); excludes Finished
}

// Engine runs a continuous-batching loop. Construct with New.
type Engine struct {
	cap       int
	maxTokens int
}

// New builds an Engine from a Scheduler config, clamping MaxConcurrency to a
// minimum of 1.
//
//	e := schedule.New(schedule.Scheduler{MaxConcurrency: 8, MaxBatchTokens: 8192})
func New(cfg Scheduler) *Engine {
	capN := max(cfg.MaxConcurrency, 1)
	return &Engine{cap: capN, maxTokens: cfg.MaxBatchTokens}
}

// cost is a running sequence's current contribution to the batch token budget:
// its prompt plus everything generated so far.
func cost(s *Seq) int { return s.Request.PromptTokens + s.Generated }

// Run executes the continuous batching loop over requests, calling stepper once
// per decode step and emitting each produced token through onToken (nil onToken
// is allowed). It admits from the FIFO queue while the running set is under both
// the concurrency cap and the token budget, advances all running sequences one
// token, retires those that hit EOS or MaxNewTokens, then admits more — until
// the queue and the running set are both empty. A request whose prompt alone
// exceeds MaxBatchTokens (or MaxNewTokens ≤ 0) is retired up front without ever
// running the model. Results are returned in completion order. A nil stepper or
// a Stepper error fails the whole Run; ctx cancellation aborts between steps.
//
//	out, err := e.Run(ctx, reqs, stepper, func(id string, t int) { stream(id, t) })
func (e *Engine) Run(ctx context.Context, requests []Request, stepper Stepper, onToken func(reqID string, tok int)) ([]Result, error) {
	if stepper == nil {
		return nil, core.E("schedule", "nil stepper", nil)
	}

	// The queue shares the caller's backing array: Run only advances its own
	// slice header (queue = queue[1:]) and never writes a queue element, so the
	// caller's requests slice is observably unchanged and no defensive copy is
	// needed.
	queue := requests
	running := make([]*Seq, 0, e.cap)
	results := make([]Result, 0, len(requests))

	// seqStore backs every admitted Seq from a single slab rather than one heap
	// allocation per admit. Each request is admitted at most once, so
	// len(requests) slots always suffice; the slab is never resliced, so the
	// &seqStore[i] pointers handed into the running set stay stable.
	seqStore := make([]Seq, len(requests))
	nextSeq := 0

	// tokenSlab backs every admitted seq's token buffer from a single slab too,
	// rather than one make per admit. Each admitted seq is carved a
	// non-overlapping window tokenSlab[off:off:off+MaxNewTokens] (a 3-index slice
	// whose cap is exactly its own MaxNewTokens), which is moved into the Result
	// on retirement — caller-owned output, observably identical to a standalone
	// make. Three properties make the shared backing safe:
	//
	//   - No window is ever written past its cap inside Run: a seq appends exactly
	//     one token per step it produces and is retired the moment Generated
	//     reaches MaxNewTokens, so len never exceeds the window cap and the append
	//     never reallocates while decoding.
	//   - No two Results alias the same region: off advances by MaxNewTokens per
	//     admit, so the windows tile the slab end to end without overlap and each
	//     window's spare capacity stops exactly where the next window begins.
	//   - A caller mutating or appending one Result.Tokens cannot reach another:
	//     the 3-index cap confines any in-place write to the window's own region,
	//     and an append past it reallocates (copies out) rather than spilling into
	//     the neighbour.
	//
	// The slab is sized at the sum of every positive MaxNewTokens — an upper bound
	// on what any admission order can consume, since only requests with
	// MaxNewTokens > 0 are ever admitted — so off can never run past the slab
	// regardless of how the budget gate drains the running set.
	slabTokens := 0
	for i := range requests {
		if requests[i].MaxNewTokens > 0 {
			slabTokens += requests[i].MaxNewTokens
		}
	}
	tokenSlab := make([]int, slabTokens)
	slabOff := 0

	// admit pulls from the front of the queue into the running set while both
	// limits allow. A request that can never fit (oversize prompt) or needs no
	// tokens is retired here rather than admitted, so it never blocks the head
	// of the queue.
	admit := func() {
		for len(queue) > 0 && len(running) < e.cap {
			req := queue[0]

			// A request asking for no tokens completes immediately — nothing to
			// decode, no budget consumed.
			if req.MaxNewTokens <= 0 {
				queue = queue[1:]
				results = append(results, Result{ID: req.ID, Tokens: []int{}, Finished: true})
				continue
			}

			// An oversize prompt can never satisfy the token budget — retire it
			// with a typed error and move on (it must not wedge the queue).
			if req.PromptTokens > e.maxTokens {
				queue = queue[1:]
				results = append(results, Result{
					ID:  req.ID,
					Err: core.E("schedule", "prompt exceeds MaxBatchTokens: "+req.ID, nil),
				})
				continue
			}

			// Budget gate: admitting this prompt must keep the running
			// prompt+generated total within MaxBatchTokens. If not, stop
			// admitting for now and let running sequences drain first.
			used := 0
			for _, s := range running {
				used += cost(s)
			}
			if used+req.PromptTokens > e.maxTokens {
				break
			}

			queue = queue[1:]
			// Carve the Seq from its slab (stable pointer) and the token buffer
			// from the token slab as a window capped at the request's hard cap
			// (MaxNewTokens is > 0 here — the ≤ 0 case is retired above), so the
			// per-step append never reallocates as the sequence decodes. The
			// window is moved into the Result on retirement.
			s := &seqStore[nextSeq]
			nextSeq++
			s.Request = req
			s.tokens = tokenSlab[slabOff : slabOff : slabOff+req.MaxNewTokens]
			slabOff += req.MaxNewTokens
			running = append(running, s)
		}
	}

	admit()

	for len(running) > 0 {
		if err := ctx.Err(); err != nil {
			return results, core.E("schedule", "context cancelled", err)
		}

		step, err := stepper.Step(ctx, running)
		if err != nil {
			return results, core.E("schedule", "decode step failed", err)
		}

		// Apply the step to every running sequence: record its token, emit it,
		// and decide whether it has finished (model EOS or its own cap).
		survivors := running[:0]
		for _, s := range running {
			tok, ok := step.Tokens[s.Request.ID]
			if ok {
				s.Generated++
				if onToken != nil {
					onToken(s.Request.ID, tok)
				}
				s.tokens = append(s.tokens, tok)
			}
			finished := step.Finished[s.Request.ID] || s.Generated >= s.Request.MaxNewTokens
			if finished {
				s.Done = true
				results = append(results, Result{
					ID:       s.Request.ID,
					Tokens:   s.tokens,
					Finished: true,
				})
				continue
			}
			survivors = append(survivors, s)
		}
		running = survivors

		// A slot may have freed (a sequence retired) — admit the next queued
		// requests before the next decode step. This is the "continuous" in
		// continuous batching: the running set is topped up every iteration.
		admit()
	}

	return results, nil
}
