// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
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

func (m *blockingModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *blockingModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *blockingModel) ModelType() string { return "blocking" }
func (m *blockingModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3"}
}
func (m *blockingModel) Metrics() inference.GenerateMetrics { return m.metrics }
func (m *blockingModel) Err() error                         { return nil }
func (m *blockingModel) Close() error                       { return nil }

func TestModel_QueuesRequestsAndEmitsLatencyProbe_Good(t *testing.T) {
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
	scheduled := New(base, Config{
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

func TestModel_RejectsFullQueue_Bad(t *testing.T) {
	base := newBlockingModel()
	scheduled := New(base, Config{MaxConcurrent: 1, MaxQueue: 1})

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
	scheduled := New(base, Config{MaxConcurrent: 1, MaxQueue: 1})

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

func (m *immediateModel) Classify(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	m.classified = append([]string(nil), prompts...)
	return []inference.ClassifyResult{{Token: inference.Token{Text: "ok"}}}, nil
}

func (m *immediateModel) BatchGenerate(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	m.batchPrompts = append([]string(nil), prompts...)
	return []inference.BatchResult{{Tokens: []inference.Token{{Text: "batch"}}}}, nil
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
func (m *immediateModel) Err() error   { return m.err }
func (m *immediateModel) Close() error { m.closed = true; return nil }

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
	scheduled := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})

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
	if results, err := scheduled.Classify(context.Background(), []string{"x"}); err != nil || len(results) != 1 || base.classified[0] != "x" {
		t.Fatalf("Classify() = %+v/%v classified=%v", results, err, base.classified)
	}
	if batches, err := scheduled.BatchGenerate(context.Background(), []string{"b"}); err != nil || len(batches) != 1 || base.batchPrompts[0] != "b" {
		t.Fatalf("BatchGenerate() = %+v/%v prompts=%v", batches, err, base.batchPrompts)
	}
	if scheduled.ModelType() != "immediate" || scheduled.Info().Architecture != "qwen3" || scheduled.Metrics().GeneratedTokens != 2 {
		t.Fatalf("model delegates = type %q info %+v metrics %+v", scheduled.ModelType(), scheduled.Info(), scheduled.Metrics())
	}
	if err := scheduled.Close(); err != nil || !base.closed {
		t.Fatalf("Close() = %v closed=%v", err, base.closed)
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
	if nilScheduler.Err() != nil || nilScheduler.Close() != nil {
		t.Fatal("nil scheduler Err/Close should be nil")
	}
	nilScheduler.SetProbeSink(nil)
	if nilScheduler.ModelType() != "" || nilScheduler.Info().Architecture != "" || nilScheduler.Metrics().GeneratedTokens != 0 {
		t.Fatalf("nil scheduler delegates returned non-zero values")
	}
	if _, err := nilScheduler.Classify(context.Background(), []string{"x"}); err == nil {
		t.Fatal("Classify(nil scheduler) error = nil")
	}
	if _, err := nilScheduler.BatchGenerate(context.Background(), []string{"x"}); err == nil {
		t.Fatal("BatchGenerate(nil scheduler) error = nil")
	}
	var generated []inference.Token
	for token := range nilScheduler.Generate(context.Background(), "prompt") {
		generated = append(generated, token)
	}
	if len(generated) != 0 || nilScheduler.Err() != nil {
		t.Fatalf("nil Generate tokens=%v err=%v, want no tokens and no stored nil-scheduler err", generated, nilScheduler.Err())
	}

	scheduled := New(nil, Config{})
	if _, _, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(nil base) error = nil")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	base := &immediateModel{tokens: []inference.Token{{Text: "x"}}}
	withBase := New(base, Config{MaxQueue: 1})
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

func TestModel_ErrAndHelpers_Good(t *testing.T) {
	base := &immediateModel{tokens: []inference.Token{{Text: "x"}}, err: core.NewError("base failed")}
	scheduled := New(base, Config{RequestIDPrefix: "req", MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})
	for range scheduled.Generate(context.Background(), "prompt") {
	}
	if err := scheduled.Err(); err == nil || err.Error() != "base failed" {
		t.Fatalf("Err() = %v, want base failed", err)
	}
	scheduled.setErr(core.NewError("stored failed"))
	if err := scheduled.Err(); err == nil || err.Error() != "stored failed" {
		t.Fatalf("stored Err() = %v, want stored failed", err)
	}
	opts := generateOptions(inference.SamplerConfig{
		MaxTokens:     4,
		Temperature:   0.25,
		TopK:          8,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{1, 2},
		ReturnLogits:  true,
	})
	// generateOptions now returns a single fused option that applies the
	// whole SamplerConfig in one closure — verify by applying and reading
	// the resulting GenerateConfig.
	applied := inference.ApplyGenerateOpts(opts)
	if applied.MaxTokens != 4 || applied.Temperature != 0.25 || applied.TopK != 8 ||
		applied.TopP != 0.9 || applied.RepeatPenalty != 1.1 || !applied.ReturnLogits ||
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
