// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
	"runtime"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type blockingModel struct {
	started chan string
	release chan struct{}
	metrics inference.GenerateMetrics
}

func newBlockingModel() *blockingModel {
	return &blockingModel{
		started: make(chan string, 8),
		release: make(chan struct{}),
	}
}

func (m *blockingModel) Generate(ctx context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		m.started <- prompt
		select {
		case <-ctx.Done():
			return
		case <-m.release:
		}
		yield(inference.Token{Text: prompt})
	}
}

func (m *blockingModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt := ""
	if len(messages) > 0 {
		prompt = messages[len(messages)-1].Content
	}
	return m.Generate(ctx, prompt, opts...)
}

func (m *blockingModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *blockingModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *blockingModel) ModelType() string { return "blocking" }
func (m *blockingModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3"}
}
func (m *blockingModel) Metrics() inference.GenerateMetrics { return m.metrics }
func (m *blockingModel) Err() core.Result                   { return core.Ok(nil) }
func (m *blockingModel) Close() core.Result                 { return core.Ok(nil) }

func TestModel_Schedule_Good(t *testing.T) {
	base := newBlockingModel()
	var (
		eventsMu sync.Mutex
		events   []inference.ProbeEvent
	)
	snapshotEvents := func() []inference.ProbeEvent {
		eventsMu.Lock()
		defer eventsMu.Unlock()
		out := make([]inference.ProbeEvent, len(events))
		copy(out, events)
		return out
	}
	scheduled, _ := New(base, Config{
		MaxConcurrent:   1,
		MaxQueue:        1,
		StreamBuffer:    1,
		RequestIDPrefix: "test",
		ProbeSink: inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
			eventsMu.Lock()
			events = append(events, event)
			eventsMu.Unlock()
		}),
	})

	first, firstTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "first"})
	if err != nil {
		t.Fatalf("Schedule(first) error = %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "first" {
		t.Fatalf("started = %q, want first", got)
	}
	second, secondTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "second"})
	if err != nil {
		t.Fatalf("Schedule(second) error = %v", err)
	}
	if first.ID == "" || second.ID == "" || first.ID == second.ID {
		t.Fatalf("request IDs = %q/%q, want unique non-empty IDs", first.ID, second.ID)
	}

	assertNoStartedPrompt(t, base.started)
	base.release <- struct{}{}
	firstToken := waitScheduledToken(t, firstTokens)
	if firstToken.RequestID != first.ID || firstToken.Token.Text != "first" {
		t.Fatalf("first token = %+v, want request %q text first", firstToken, first.ID)
	}
	if firstToken.Labels["queue_latency_ms"] == "" || firstToken.Labels["first_token_latency_ms"] == "" {
		t.Fatalf("first token labels = %+v, want latency labels", firstToken.Labels)
	}

	if got := waitStartedPrompt(t, base.started); got != "second" {
		t.Fatalf("started = %q, want second", got)
	}
	base.release <- struct{}{}
	secondToken := waitScheduledToken(t, secondTokens)
	if secondToken.RequestID != second.ID || secondToken.Token.Text != "second" {
		t.Fatalf("second token = %+v, want request %q text second", secondToken, second.ID)
	}
	snap := snapshotEvents()
	if !hasSchedulerProbeEvent(snap, "first_token") || !hasSchedulerProbeEvent(snap, "complete") {
		t.Fatalf("events = %+v, want first_token and complete scheduler probes", snap)
	}
}

func TestModel_Schedule_Bad(t *testing.T) {
	base := newBlockingModel()
	scheduled, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 1})

	_, _, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "active", Prompt: "active"})
	if err != nil {
		t.Fatalf("Schedule(active) error = %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "active" {
		t.Fatalf("started = %q, want active", got)
	}
	_, _, err = scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "queued", Prompt: "queued"})
	if err != nil {
		t.Fatalf("Schedule(queued) error = %v", err)
	}
	_, _, err = scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "overflow", Prompt: "overflow"})
	if err == nil {
		t.Fatal("Schedule(overflow) error = nil, want queue full")
	}
}

func TestModel_CancelRequest_CancelsQueuedRequest_Good(t *testing.T) {
	base := newBlockingModel()
	scheduled, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 1})

	_, activeTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "active", Prompt: "active"})
	if err != nil {
		t.Fatalf("Schedule(active) error = %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "active" {
		t.Fatalf("started = %q, want active", got)
	}
	_, queuedTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "queued", Prompt: "queued"})
	if err != nil {
		t.Fatalf("Schedule(queued) error = %v", err)
	}

	result, err := scheduled.CancelRequest(context.Background(), "queued")
	if err != nil {
		t.Fatalf("CancelRequest() error = %v", err)
	}
	if !result.Cancelled || result.ID != "queued" {
		t.Fatalf("CancelRequest() = %+v, want queued cancellation", result)
	}
	base.release <- struct{}{}
	_ = waitScheduledToken(t, activeTokens)
	if token, ok := <-queuedTokens; ok {
		t.Fatalf("queued token = %+v, want closed channel after cancellation", token)
	}
	assertNoStartedPrompt(t, base.started)
}

type immediateModel struct {
	tokens       []inference.Token
	err          error
	cancelledID  string
	closed       bool
	classified   []string
	batchPrompts []string
	lastPrompt   string
	lastMessages []inference.Message
	metrics      inference.GenerateMetrics
}

func (m *immediateModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.lastPrompt = prompt
	return m.seq()
}

func (m *immediateModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.lastMessages = append([]inference.Message(nil), messages...)
	return m.seq()
}

func (m *immediateModel) Classify(_ context.Context, prompts []string, _ ...inference.GenerateOption) core.Result {
	m.classified = append([]string(nil), prompts...)
	return core.Ok([]inference.ClassifyResult{{Token: inference.Token{Text: "ok"}}})
}

func (m *immediateModel) BatchGenerate(_ context.Context, prompts []string, _ ...inference.GenerateOption) core.Result {
	m.batchPrompts = append([]string(nil), prompts...)
	return core.Ok([]inference.BatchResult{{Tokens: []inference.Token{{Text: "batch"}}}})
}

func (m *immediateModel) ModelType() string { return "immediate" }
func (m *immediateModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3", NumLayers: 2}
}
func (m *immediateModel) Metrics() inference.GenerateMetrics {
	if m.metrics.GeneratedTokens == 0 {
		m.metrics.GeneratedTokens = len(m.tokens)
	}
	return m.metrics
}
func (m *immediateModel) Err() core.Result   { return core.ResultOf(nil, m.err) }
func (m *immediateModel) Close() core.Result { m.closed = true; return core.Ok(nil) }

func (m *immediateModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	m.cancelledID = id
	return inference.RequestCancelResult{ID: id, Cancelled: id != "", Reason: "base_cancelled"}, nil
}

func (m *immediateModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func TestModel_GenerateChatAndDelegates_Good(t *testing.T) {
	base := &immediateModel{tokens: []inference.Token{{Text: "A"}, {Text: "B"}}}
	scheduled, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})

	var generated []string
	for token := range scheduled.Generate(context.Background(), "prompt", inference.WithMaxTokens(2)) {
		generated = append(generated, token.Text)
	}
	if len(generated) != 2 || generated[0] != "A" || generated[1] != "B" || base.lastPrompt != "prompt" {
		t.Fatalf("generated = %v prompt=%q, want A/B from prompt", generated, base.lastPrompt)
	}

	var chat []string
	for token := range scheduled.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		chat = append(chat, token.Text)
	}
	if len(chat) != 2 || len(base.lastMessages) != 1 || base.lastMessages[0].Content != "hi" {
		t.Fatalf("chat = %v messages=%+v, want delegated chat", chat, base.lastMessages)
	}
	if cr := scheduled.Classify(context.Background(), []string{"x"}); !cr.OK || len(cr.Value.([]inference.ClassifyResult)) != 1 || base.classified[0] != "x" {
		t.Fatalf("Classify() = %+v classified=%v", cr, base.classified)
	}
	if br := scheduled.BatchGenerate(context.Background(), []string{"b"}); !br.OK || len(br.Value.([]inference.BatchResult)) != 1 || base.batchPrompts[0] != "b" {
		t.Fatalf("BatchGenerate() = %+v prompts=%v", br, base.batchPrompts)
	}
	if scheduled.ModelType() != "immediate" || scheduled.Info().Architecture != "qwen3" || scheduled.Metrics().GeneratedTokens != 2 {
		t.Fatalf("model delegates = type %q info %+v metrics %+v", scheduled.ModelType(), scheduled.Info(), scheduled.Metrics())
	}
	if cr := scheduled.Close(); !cr.OK || !base.closed {
		t.Fatalf("Close() = %+v closed=%v", cr, base.closed)
	}
}

func TestModel_NilAndErrorPaths_Bad(t *testing.T) {
	var nilScheduler *Model
	if _, _, err := nilScheduler.Schedule(context.Background(), inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(nil scheduler) error = nil")
	}
	if result, err := nilScheduler.CancelRequest(context.Background(), "x"); err != nil || result.Reason != "scheduler_nil" {
		t.Fatalf("CancelRequest(nil scheduler) = %+v/%v", result, err)
	}
	if !nilScheduler.Err().OK || !nilScheduler.Close().OK {
		t.Fatal("nil scheduler Err/Close should be OK")
	}
	nilScheduler.SetProbeSink(nil)
	if nilScheduler.ModelType() != "" || nilScheduler.Info().Architecture != "" || nilScheduler.Metrics().GeneratedTokens != 0 {
		t.Fatalf("nil scheduler delegates returned non-zero values")
	}
	if cr := nilScheduler.Classify(context.Background(), []string{"x"}); cr.OK {
		t.Fatal("Classify(nil scheduler) should fail")
	}
	if br := nilScheduler.BatchGenerate(context.Background(), []string{"x"}); br.OK {
		t.Fatal("BatchGenerate(nil scheduler) should fail")
	}
	var generated []inference.Token
	for token := range nilScheduler.Generate(context.Background(), "prompt") {
		generated = append(generated, token)
	}
	if len(generated) != 0 || !nilScheduler.Err().OK {
		t.Fatalf("nil Generate tokens=%v err=%+v, want no tokens and no stored nil-scheduler err", generated, nilScheduler.Err())
	}

	scheduled, _ := New(nil, Config{})
	if _, _, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(nil base) error = nil")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	base := &immediateModel{tokens: []inference.Token{{Text: "x"}}}
	withBase, _ := New(base, Config{MaxQueue: 1})
	if _, _, err := withBase.Schedule(cancelled, inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(cancelled context) error = nil")
	}
	if result, err := withBase.CancelRequest(context.Background(), ""); err != nil || result.Reason != "missing_id" {
		t.Fatalf("CancelRequest(empty) = %+v/%v", result, err)
	}
	if result, err := withBase.CancelRequest(context.Background(), "unknown"); err != nil || !result.Cancelled || base.cancelledID != "unknown" {
		t.Fatalf("CancelRequest(fallback) = %+v/%v cancelledID=%q", result, err, base.cancelledID)
	}
}

func TestModel_generateOptions_Good(t *testing.T) {
	base := &immediateModel{tokens: []inference.Token{{Text: "x"}}, err: core.NewError("base failed")}
	scheduled, _ := New(base, Config{RequestIDPrefix: "req", MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})
	for range scheduled.Generate(context.Background(), "prompt") {
	}
	if r := scheduled.Err(); r.OK || r.Error() != "base failed" {
		t.Fatalf("Err() = %+v, want base failed", r)
	}
	scheduled.setErr(core.NewError("stored failed"))
	if r := scheduled.Err(); r.OK || r.Error() != "stored failed" {
		t.Fatalf("stored Err() = %+v, want stored failed", r)
	}
	opts := generateOptions(inference.SamplerConfig{
		MaxTokens:     4,
		Temperature:   0.25,
		TopK:          8,
		TopP:          0.9,
		MinP:          0.05,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{1, 2},
		ReturnLogits:  true,
	})
	// generateOptions now returns a single fused option that applies the
	// whole SamplerConfig in one closure — verify by applying and reading
	// the resulting GenerateConfig.
	applied := inference.ApplyGenerateOpts(opts)
	if applied.MaxTokens != 4 || applied.Temperature != 0.25 || applied.TopK != 8 ||
		applied.TopP != 0.9 || applied.MinP != 0.05 || applied.RepeatPenalty != 1.1 || !applied.ReturnLogits ||
		len(applied.StopTokens) != 2 || applied.StopTokens[0] != 1 || applied.StopTokens[1] != 2 {
		t.Fatalf("generateOptions applied = %+v", applied)
	}
	labels := map[string]string{"a": "b"}
	cloned := cloneLabels(labels)
	cloned["a"] = "changed"
	if labels["a"] != "b" {
		t.Fatalf("cloneLabels mutated source = %+v", labels)
	}
	if millis(-time.Millisecond) != 0 || millisString(time.Millisecond) == "" {
		t.Fatal("millis helpers returned unexpected values")
	}
}

func waitStartedPrompt(t *testing.T, started <-chan string) string {
	t.Helper()
	select {
	case prompt := <-started:
		return prompt
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for prompt start")
		return ""
	}
}

func assertNoStartedPrompt(t *testing.T, started <-chan string) {
	t.Helper()
	select {
	case prompt := <-started:
		t.Fatalf("unexpected started prompt %q", prompt)
	case <-time.After(25 * time.Millisecond):
	}
}

func waitScheduledToken(t *testing.T, tokens <-chan inference.ScheduledToken) inference.ScheduledToken {
	t.Helper()
	select {
	case token, ok := <-tokens:
		if !ok {
			t.Fatal("token channel closed before token")
		}
		return token
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for token")
		return inference.ScheduledToken{}
	}
}

func hasSchedulerProbeEvent(events []inference.ProbeEvent, eventName string) bool {
	for _, event := range events {
		if event.Kind == inference.ProbeEventScheduler && event.Scheduler != nil && event.Scheduler.Event == eventName {
			return true
		}
	}
	return false
}

// tokenizingModel is a TextModel that ALSO implements inference.TokenizerModel —
// the capability batch mode (and interleave with a token budget) requires. Its
// Generate/Chat yield a fixed token list so mode-routing tests can assert the
// stream came back through the scheduler.
type tokenizingModel struct {
	tokens  []inference.Token
	metrics inference.GenerateMetrics
}

func (m *tokenizingModel) Generate(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq(ctx)
}

func (m *tokenizingModel) Chat(ctx context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq(ctx)
}

func (m *tokenizingModel) seq(ctx context.Context) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if ctx.Err() != nil {
				return
			}
			if !yield(token) {
				return
			}
		}
	}
}

func (m *tokenizingModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *tokenizingModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *tokenizingModel) ModelType() string { return "tokenizing" }
func (m *tokenizingModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "gemma3"}
}
func (m *tokenizingModel) Metrics() inference.GenerateMetrics { return m.metrics }
func (m *tokenizingModel) Err() core.Result                   { return core.Ok(nil) }
func (m *tokenizingModel) Close() core.Result                 { return core.Ok(nil) }
func (m *tokenizingModel) Encode(text string) []int32         { return make([]int32, len([]rune(text))) }
func (m *tokenizingModel) Decode([]int32) string              { return "" }
func (m *tokenizingModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	out := ""
	for _, msg := range messages {
		out += msg.Content
	}
	return out, nil
}

func TestNew_UnknownMode_Bad(t *testing.T) {
	if _, err := New(&tokenizingModel{}, Config{Mode: "bogus"}); err == nil {
		t.Fatal("New(unknown mode) error = nil, want non-nil")
	}
}

func TestNew_BatchRequiresTokenizer_Bad(t *testing.T) {
	// immediateModel is a TextModel but NOT a TokenizerModel → batch fails closed.
	if _, err := New(&immediateModel{}, Config{Mode: ModeBatch, MaxConcurrent: 2, MaxBatchTokens: 16}); err == nil {
		t.Fatal("New(batch, non-tokenizer) error = nil, want fail-closed error")
	}
	// A TokenizerModel satisfies batch's prompt-token budgeting requirement.
	m, err := New(&tokenizingModel{}, Config{Mode: ModeBatch, MaxConcurrent: 2, MaxBatchTokens: 16})
	if err != nil {
		t.Fatalf("New(batch, tokenizer) error = %v, want nil", err)
	}
	m.Close()
}

func TestNew_InterleaveTokenBudgetRequiresTokenizer_Bad(t *testing.T) {
	// Interleave WITH a token budget needs the tokeniser too — fail closed.
	if _, err := New(&immediateModel{}, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxBatchTokens: 16}); err == nil {
		t.Fatal("New(interleave+budget, non-tokenizer) error = nil, want fail-closed error")
	}
	// Interleave WITHOUT a token budget (count-only) needs only a TextModel.
	m, err := New(&immediateModel{}, Config{Mode: ModeInterleave, MaxConcurrent: 2})
	if err != nil {
		t.Fatalf("New(interleave, no budget) error = %v, want nil", err)
	}
	m.Close()
}

func TestModel_ModeBatch_RoutesThroughBatch_Good(t *testing.T) {
	base := &tokenizingModel{tokens: []inference.Token{{Text: "x0"}, {Text: "x1"}}}
	m, err := New(base, Config{Mode: ModeBatch, MaxConcurrent: 2, MaxQueue: 4, MaxBatchTokens: 1024, StreamBuffer: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer m.Close()
	_, tokens, err := m.Schedule(context.Background(), inference.ScheduledRequest{ID: "r", Prompt: "hello"})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	var got []string
	for tok := range tokens {
		got = append(got, tok.Token.Text)
	}
	if len(got) != 2 || got[0] != "x0" || got[1] != "x1" {
		t.Fatalf("batch-mode tokens = %v, want [x0 x1]", got)
	}
	if s := m.Stats(); s.Submitted != 1 {
		t.Fatalf("batch Stats.Submitted = %d, want 1", s.Submitted)
	}
}

func TestModel_ModeInterleave_RoutesThroughInterleave_Good(t *testing.T) {
	base := &tokenizingModel{tokens: []inference.Token{{Text: "y0"}, {Text: "y1"}, {Text: "y2"}}}
	m, err := New(base, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxQueue: 4, StreamBuffer: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer m.Close()
	_, tokens, err := m.Schedule(context.Background(), inference.ScheduledRequest{ID: "r", Messages: []inference.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	var got []string
	for tok := range tokens {
		got = append(got, tok.Token.Text)
	}
	if len(got) != 3 {
		t.Fatalf("interleave-mode tokens = %v, want 3", got)
	}
	waitStats(t, m.Stats, func(s Stats) bool { return s.Completed == 1 })
}

func TestModel_Stats_Serial_Good(t *testing.T) {
	base := &immediateModel{tokens: []inference.Token{{Text: "z"}}}
	m, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})
	for range m.Generate(context.Background(), "p") {
	}
	if s := m.Stats(); s.Submitted != 1 || s.Completed != 1 {
		t.Fatalf("serial Stats = %+v, want Submitted=1 Completed=1", s)
	}
}

// waitGoroutineCount polls runtime.NumGoroutine until want is satisfied or the
// deadline passes. A worker goroutine's own exit (the runtime's counter
// decrement) trails the workerWG.Done() call that unblocks a joining Wait() by
// a scheduling instant, so an immediate read right after a synchronous
// CloseEngine call can occasionally observe a stale, still-draining count —
// the same reason waitStats polls Stats rather than reading it once.
func waitGoroutineCount(t *testing.T, want func(int) bool) int {
	t.Helper()
	deadline := time.After(2 * time.Second)
	for {
		n := runtime.NumGoroutine()
		if want(n) {
			return n
		}
		select {
		case <-deadline:
			t.Fatalf("goroutine count never satisfied predicate, last = %d", n)
		case <-time.After(2 * time.Millisecond):
		}
	}
}

// TestModel_CloseEngine_Serial_DrainsWorkers_Good is the goroutine-leak guard
// a per-resident-model scheduler depends on: serving.multiModelResolver
// builds a fresh scheduler.Model on every model load and tears it down on
// every eviction, so a serial-mode CloseEngine that does not actually join its
// worker pool would leak MaxConcurrent goroutines per evict, forever, on the
// DEFAULT scheduler mode (serial is Config.Mode's zero value). Before this
// fix, CloseEngine's switch had no serial case at all — worker ranged over
// m.queue, which nothing ever closed.
func TestModel_CloseEngine_Serial_DrainsWorkers_Good(t *testing.T) {
	before := runtime.NumGoroutine()
	base := &immediateModel{tokens: []inference.Token{{Text: "x"}}}
	m, err := New(base, Config{MaxConcurrent: 4, MaxQueue: 4, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	waitGoroutineCount(t, func(n int) bool { return n >= before+4 })

	m.CloseEngine()

	waitGoroutineCount(t, func(n int) bool { return n <= before })
}

// TestModel_CloseEngine_Serial_ActiveAndQueued_NoHang_Good proves CloseEngine
// on serial mode — the mode a per-resident-model scheduler builds by default —
// neither hangs nor panics an in-flight (already running) job or one still
// sitting in the queue: both output channels are guaranteed to close, and a
// Schedule call after CloseEngine fails closed rather than being accepted into
// an abandoned queue. This is the "an in-flight request on the evicting model
// must complete or error cleanly, never hang" contract at the scheduler layer.
func TestModel_CloseEngine_Serial_ActiveAndQueued_NoHang_Good(t *testing.T) {
	base := newBlockingModel()
	m, err := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, activeTokens, err := m.Schedule(context.Background(), inference.ScheduledRequest{ID: "active", Prompt: "active"})
	if err != nil {
		t.Fatalf("Schedule(active): %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "active" {
		t.Fatalf("started = %q, want active", got)
	}
	// active's source is now blocked on base.release, which nobody signals —
	// the only way it unblocks is CloseEngine cancelling its context.
	_, queuedTokens, err := m.Schedule(context.Background(), inference.ScheduledRequest{ID: "queued", Prompt: "queued"})
	if err != nil {
		t.Fatalf("Schedule(queued): %v", err)
	}

	closed := make(chan struct{})
	go func() { m.CloseEngine(); close(closed) }()
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("CloseEngine did not return — an active job blocked on the base model is not being cancelled")
	}

	assertClosedNoTokens(t, activeTokens, "active")
	assertClosedNoTokens(t, queuedTokens, "queued")

	if _, _, err := m.Schedule(context.Background(), inference.ScheduledRequest{ID: "after", Prompt: "after"}); err == nil {
		t.Fatal("Schedule after CloseEngine error = nil, want a closed-engine error")
	}
}

// TestModel_CloseEngine_Serial_Idempotent_Good proves calling CloseEngine
// twice never panics — the shape a belt-and-braces caller (an explicit evict
// plus a deferred teardown) can produce.
func TestModel_CloseEngine_Serial_Idempotent_Good(t *testing.T) {
	base := &immediateModel{tokens: []inference.Token{{Text: "x"}}}
	m, err := New(base, Config{MaxConcurrent: 1, MaxQueue: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	m.CloseEngine()
	m.CloseEngine() // must not panic (close of an already-closed channel)
}

// assertClosedNoTokens drains ch, failing if it yields a token or does not
// close within the timeout — the "no hang" half of the eviction contract.
func assertClosedNoTokens(t *testing.T, ch <-chan inference.ScheduledToken, label string) {
	t.Helper()
	select {
	case tok, ok := <-ch:
		if ok {
			t.Fatalf("%s: got token %+v after CloseEngine, want a closed channel", label, tok)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("%s: channel did not close within 2s of CloseEngine — hang", label)
	}
}
