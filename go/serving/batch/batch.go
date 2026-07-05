// SPDX-Licence-Identifier: EUPL-1.2

// Package batch is the batch executor (RFC.md §6.3): pure orchestration
// that submits many chat / embedding / completion requests through one call and
// one router. It fans requests out under a configurable concurrency cap,
// throttles every call through a first-class rate Limiter so a provider's
// limits are never exceeded, returns results in INPUT order (Run) or AS each
// completes (RunAsCompleted), each carrying per-item success or a typed error,
// and aggregates token Usage across the whole batch.
//
// The package is transport-agnostic: a Call interface stands in for the actual
// dispatch (the local go-ml expansion pipeline / go-mlx BatchGenerate, or a
// remote provider). Heavy logic stays in those packages — batch only schedules.
//
//	out := batch.Run(ctx, requests, batch.Options{
//		Concurrency: 8,
//		Call:        myCall,                       // does one request
//		Limiter:     batch.NewTokenBucket(10, 5),  // 10/s, burst 5
//	})
//	for _, it := range out.Items {
//		if it.Err != nil { /* per-item typed error */ continue }
//		use(it.Result)
//	}
//	total := out.Usage // aggregated across the batch
package batch

import (
	"context"
	"sync"

	core "dappco.re/go"
)

// Usage is the token accounting for one call, aggregated across a batch in
// BatchResult.Usage (RFC.md §6.6 — reconciled from the responses).
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Add returns the element-wise sum of two usages — the batch aggregator folds
// every successful item's usage into the running total.
//
//	total = total.Add(item.Usage)
func (u Usage) Add(o Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens + o.PromptTokens,
		CompletionTokens: u.CompletionTokens + o.CompletionTokens,
		TotalTokens:      u.TotalTokens + o.TotalTokens,
	}
}

// Call performs one request in the batch. index is the request's position in
// the input slice (so a Call may key per-item state or logging on it); request
// is the caller's opaque request value. It returns the opaque result, that
// call's token Usage, and an error for that item alone — a failed item never
// fails the batch.
//
//	type chatCall struct{ router *ai.Router }
//	func (c chatCall) Do(ctx context.Context, i int, req any) (any, batch.Usage, error) {
//		resp, err := c.router.Chat(ctx, req.(ai.ChatRequest))
//		if err != nil { return nil, batch.Usage{}, err }
//		return resp, batch.Usage{TotalTokens: resp.Usage.TotalTokens}, nil
//	}
type Call interface {
	Do(ctx context.Context, index int, request any) (result any, usage Usage, err error)
}

// Options configures one batch run. Concurrency is the cap on in-flight calls
// (<= 0 clamps to 1 — never unbounded). Call is required; a nil Call fails
// every item closed with a typed error rather than panicking. Limiter is
// optional — nil means no rate limiting (the concurrency cap is the only
// bound).
type Options struct {
	Concurrency int     // max in-flight calls; <= 0 → 1
	Call        Call    // performs each request (required)
	Limiter     Limiter // throttles every call; nil → unthrottled
}

// ItemResult is the outcome of one request, carrying its input Index so callers
// can correlate even in completion order (RunAsCompleted). Exactly one of
// Result / Err is meaningful: on success Err is nil and Result + Usage are set;
// on failure Err is the typed error and Result is nil.
type ItemResult struct {
	Index  int   // position in the input slice
	Result any   // the Call's result (nil on error)
	Usage  Usage // token usage for this call (zero on error)
	Err    error // typed per-item error (nil on success)
}

// BatchResult is the ordered outcome of Run: Items is in INPUT order (Items[i]
// is request i), and Usage is the aggregate over every successful item.
type BatchResult struct {
	Items []ItemResult // one per request, in input order
	Usage Usage        // aggregated across the batch
}

// Run fans the requests out under opts.Concurrency, throttled by opts.Limiter,
// and returns results in INPUT order with aggregated Usage. Each item carries
// its own success or typed error; one failure never aborts the others. An empty
// request slice returns an empty result without dispatching anything.
//
//	out := batch.Run(ctx, reqs, batch.Options{Concurrency: 8, Call: c})
//	answer := out.Items[0].Result // corresponds to reqs[0]
func Run(ctx context.Context, requests []any, opts Options) BatchResult {
	items := make([]ItemResult, len(requests))
	if len(requests) == 0 {
		return BatchResult{Items: items}
	}

	results := dispatch(ctx, requests, opts)
	var agg Usage
	for it := range results {
		items[it.Index] = it // index slot → input order regardless of completion order
		if it.Err == nil {
			agg = agg.Add(it.Usage)
		}
	}
	return BatchResult{Items: items, Usage: agg}
}

// RunAsCompleted fans the requests out the same way as Run but streams each
// ItemResult on the returned channel AS it completes (completion order, not
// input order) — the path for streaming pipelines. Each result still carries
// its input Index, so a consumer can correlate. The channel is closed once the
// final item is delivered; the caller drains it to completion.
//
//	for it := range batch.RunAsCompleted(ctx, reqs, opts) {
//		handle(it.Index, it.Result, it.Err)
//	}
func RunAsCompleted(ctx context.Context, requests []any, opts Options) <-chan ItemResult {
	if len(requests) == 0 {
		ch := make(chan ItemResult)
		close(ch)
		return ch
	}
	return dispatch(ctx, requests, opts)
}

// dispatch is the shared fan-out core: a bounded worker pool draws indices off a
// feed channel, throttles each through the limiter, runs the Call, and emits one
// ItemResult per request on the returned channel (in completion order). The
// channel is closed once every worker has finished. Both Run and
// RunAsCompleted build on it — the only difference is whether the caller
// reorders the stream into input slots.
func dispatch(ctx context.Context, requests []any, opts Options) <-chan ItemResult {
	cap := opts.Concurrency
	if cap < 1 {
		cap = 1 // a non-positive cap is serial, never unbounded
	}
	if cap > len(requests) {
		cap = len(requests) // no point in more workers than work
	}

	feed := make(chan int)
	out := make(chan ItemResult, len(requests))

	// Feed indices in input order; stop early if the context is cancelled.
	go func() {
		defer close(feed)
		for i := range requests {
			select {
			case feed <- i:
			case <-ctx.Done():
				return
			}
		}
	}()

	var wg sync.WaitGroup
	wg.Add(cap)
	for w := 0; w < cap; w++ {
		go func() {
			defer wg.Done()
			for i := range feed {
				out <- runOne(ctx, i, requests[i], opts)
			}
		}()
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

// runOne throttles then performs a single request, translating every failure
// mode into a typed ItemResult (context cancellation, a nil Call, or the Call's
// own error) so the batch never panics and every item is accounted for.
func runOne(ctx context.Context, index int, request any, opts Options) ItemResult {
	if err := ctx.Err(); err != nil {
		return ItemResult{Index: index, Err: core.E("batch", core.Sprintf("item %d cancelled", index), err)}
	}
	if opts.Limiter != nil {
		if err := opts.Limiter.Wait(ctx); err != nil {
			return ItemResult{Index: index, Err: core.E("batch", core.Sprintf("item %d throttle wait", index), err)}
		}
	}
	if opts.Call == nil {
		return ItemResult{Index: index, Err: core.E("batch", core.Sprintf("item %d has no Call configured", index), nil)}
	}

	res, usage, err := opts.Call.Do(ctx, index, request)
	if err != nil {
		return ItemResult{Index: index, Err: core.E("batch", core.Sprintf("item %d", index), err)}
	}
	return ItemResult{Index: index, Result: res, Usage: usage}
}
