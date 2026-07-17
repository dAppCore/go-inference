// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeTextModel is a deterministic generation probe. Each prompt has a token
// script and may have a gate; a gated call blocks until the gate closes or its
// context is cancelled. The counters are safe to inspect while calls run.
type fakeTextModel struct {
	mu              sync.RWMutex
	scripts         map[string][]string
	gates           map[string]chan struct{}
	afterFirstGates map[string]chan struct{}
	started         chan string
	firstYielded    chan string

	active    atomic.Int64
	maxActive atomic.Int64
	closes    atomic.Int64
}

type requestErrorTextModel struct {
	*fakeTextModel
	errMu   sync.Mutex
	lastErr core.Result
}

func newRequestErrorTextModel() *requestErrorTextModel {
	return &requestErrorTextModel{
		fakeTextModel: newFakeTextModel(nil),
		lastErr:       core.Ok(nil),
	}
}

func (model *requestErrorTextModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt := ""
	if len(messages) > 0 {
		prompt = messages[len(messages)-1].Content
	}
	return func(yield func(inference.Token) bool) {
		model.errMu.Lock()
		if prompt == "fails" {
			model.lastErr = core.Fail(core.E("test.request", "first request failed", nil))
		} else {
			model.lastErr = core.Ok(nil)
		}
		model.errMu.Unlock()
		if prompt != "fails" {
			yield(inference.Token{Text: "second succeeds"})
		}
	}
}

func (model *requestErrorTextModel) Err() core.Result {
	model.errMu.Lock()
	defer model.errMu.Unlock()
	return model.lastErr
}

func newFakeTextModel(scripts map[string][]string) *fakeTextModel {
	return &fakeTextModel{
		scripts:         scripts,
		gates:           make(map[string]chan struct{}),
		afterFirstGates: make(map[string]chan struct{}),
		started:         make(chan string, 16),
		firstYielded:    make(chan string, 16),
	}
}

func (m *fakeTextModel) block(prompt string) chan struct{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	gate := make(chan struct{})
	m.gates[prompt] = gate
	return gate
}

func (m *fakeTextModel) blockAfterFirst(prompt string) chan struct{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	gate := make(chan struct{})
	m.afterFirstGates[prompt] = gate
	return gate
}

func (m *fakeTextModel) Generate(ctx context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.sequence(ctx, prompt)
}

func (m *fakeTextModel) Chat(ctx context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt := ""
	if len(messages) > 0 {
		prompt = messages[len(messages)-1].Content
	}
	return m.sequence(ctx, prompt)
}

func (m *fakeTextModel) sequence(ctx context.Context, prompt string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		active := m.active.Add(1)
		defer m.active.Add(-1)
		for {
			seen := m.maxActive.Load()
			if active <= seen || m.maxActive.CompareAndSwap(seen, active) {
				break
			}
		}
		m.started <- prompt

		m.mu.RLock()
		gate := m.gates[prompt]
		afterFirstGate := m.afterFirstGates[prompt]
		tokens := append([]string(nil), m.scripts[prompt]...)
		m.mu.RUnlock()
		if gate != nil {
			select {
			case <-ctx.Done():
				return
			case <-gate:
			}
		}
		for index, text := range tokens {
			select {
			case <-ctx.Done():
				return
			default:
			}
			if !yield(inference.Token{Text: text}) {
				return
			}
			if index == 0 {
				m.firstYielded <- prompt
				if afterFirstGate != nil {
					select {
					case <-ctx.Done():
						return
					case <-afterFirstGate:
					}
				}
			}
		}
	}
}

func (m *fakeTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *fakeTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *fakeTextModel) ModelType() string                  { return "fake" }
func (m *fakeTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{Architecture: "qwen3"} }
func (m *fakeTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *fakeTextModel) Err() core.Result                   { return core.Ok(nil) }
func (m *fakeTextModel) Close() core.Result                 { m.closes.Add(1); return core.Ok(nil) }

func TestModelLane_Good(t *testing.T) {
	base := newFakeTextModel(map[string][]string{"first": {"one"}, "second": {"two"}})
	firstGate := base.block("first")
	secondGate := base.block("second")
	r := newModelLane(base, "test")
	if !r.OK {
		t.Fatalf("newModelLane error = %s", r.Error())
	}
	lane := r.Value.(*modelLane)
	defer lane.Close()

	done := make(chan string, 2)
	go consumeFakeChat(lane.Model(), "first", done)
	go consumeFakeChat(lane.Model(), "second", done)

	first := waitFakeStarted(t, base.started)
	assertNoFakeStarted(t, base.started)
	if got := base.maxActive.Load(); got != 1 {
		t.Fatalf("max base concurrency = %d, want 1", got)
	}
	if first == "first" {
		close(firstGate)
	} else {
		close(secondGate)
	}
	second := waitFakeStarted(t, base.started)
	if first == second {
		t.Fatalf("started prompts = %q then %q, want both requests", first, second)
	}
	if second == "first" {
		close(firstGate)
	} else {
		close(secondGate)
	}
	waitFakeDone(t, done)
	waitFakeDone(t, done)
	if got := base.maxActive.Load(); got != 1 {
		t.Fatalf("max base concurrency = %d after both requests, want 1", got)
	}
}

func TestModelLane_Bad(t *testing.T) {
	r := newModelLane(nil, "missing")
	if r.OK {
		t.Fatal("newModelLane(nil) succeeded")
	}
}

func TestModelLane_Ugly(t *testing.T) {
	base := newFakeTextModel(map[string][]string{"running": {"one"}, "queued": {"two"}})
	base.block("running")
	base.block("queued")
	r := newModelLane(base, "test")
	if !r.OK {
		t.Fatalf("newModelLane error = %s", r.Error())
	}
	lane := r.Value.(*modelLane)

	done := make(chan string, 2)
	go consumeFakeChat(lane.Model(), "running", done)
	if got := waitFakeStarted(t, base.started); got != "running" {
		t.Fatalf("started = %q, want running", got)
	}
	go consumeFakeChat(lane.Model(), "queued", done)
	assertNoFakeStarted(t, base.started)

	closed := make(chan core.Result, 1)
	go func() { closed <- lane.Close() }()
	waitFakeDone(t, done)
	waitFakeDone(t, done)
	select {
	case result := <-closed:
		if !result.OK {
			t.Fatalf("Close error = %s", result.Error())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close did not drain running and queued requests")
	}
	if r := lane.Close(); !r.OK {
		t.Fatalf("second Close error = %s", r.Error())
	}
	if got := base.closes.Load(); got != 1 {
		t.Fatalf("base Close calls = %d, want 1", got)
	}
}

func TestJobManager_Good(t *testing.T) {
	model := newFakeTextModel(map[string][]string{
		"alpha": {"A1", "A2"},
		"beta":  {"B1", "B2"},
	})
	jobs := newJobManager(context.Background())
	alpha := jobs.Start("session-alpha", "job-alpha", model,
		[]inference.Message{{Role: "user", Content: "alpha"}}, nil)
	beta := jobs.Start("session-beta", "job-beta", model,
		[]inference.Message{{Role: "user", Content: "beta"}}, nil)
	if !alpha.OK || !beta.OK {
		t.Fatalf("Start results = %s / %s", alpha.Error(), beta.Error())
	}

	assertInterleavedGenerations(t, []taggedGenerationExpectation{
		{generation: alpha.Value.(*generation), sessionID: "session-alpha", jobID: "job-alpha", text: "A1A2"},
		{generation: beta.Value.(*generation), sessionID: "session-beta", jobID: "job-beta", text: "B1B2"},
	})
}

func TestJobManager_Bad(t *testing.T) {
	model := newFakeTextModel(map[string][]string{"blocked": {"done"}})
	model.block("blocked")
	jobs := newJobManager(context.Background())
	first := jobs.Start("same-session", "first", model,
		[]inference.Message{{Role: "user", Content: "blocked"}}, nil)
	if !first.OK {
		t.Fatalf("first Start error = %s", first.Error())
	}
	waitFakeStarted(t, model.started)
	duplicate := jobs.Start("same-session", "second", model,
		[]inference.Message{{Role: "user", Content: "blocked"}}, nil)
	if duplicate.OK {
		t.Fatal("duplicate Start for one session succeeded")
	}
	if r := jobs.Cancel("same-session"); !r.OK {
		t.Fatalf("Cancel error = %s", r.Error())
	}
	assertTaggedGeneration(t, first.Value.(*generation), "same-session", "first", "")
}

func TestJobManager_Ugly(t *testing.T) {
	base := newFakeTextModel(map[string][]string{"cancel-me": {"no"}, "continue": {"yes"}})
	base.block("cancel-me")
	continueGate := base.block("continue")
	laneResult := newModelLane(base, "test")
	if !laneResult.OK {
		t.Fatalf("newModelLane error = %s", laneResult.Error())
	}
	lane := laneResult.Value.(*modelLane)
	defer lane.Close()

	jobs := newJobManager(context.Background())
	cancelled := jobs.Start("session-cancel", "job-cancel", lane.Model(),
		[]inference.Message{{Role: "user", Content: "cancel-me"}}, nil)
	if !cancelled.OK {
		t.Fatalf("cancelled Start error = %s", cancelled.Error())
	}
	if got := waitFakeStarted(t, base.started); got != "cancel-me" {
		t.Fatalf("started = %q, want cancel-me", got)
	}
	continued := jobs.Start("session-continue", "job-continue", lane.Model(),
		[]inference.Message{{Role: "user", Content: "continue"}}, nil)
	if !continued.OK {
		t.Fatalf("continued Start error = %s", continued.Error())
	}
	if r := jobs.Cancel("session-cancel"); !r.OK {
		t.Fatalf("Cancel error = %s", r.Error())
	}
	if got := waitFakeStarted(t, base.started); got != "continue" {
		t.Fatalf("started after cancel = %q, want continue", got)
	}
	close(continueGate)

	assertTaggedGeneration(t, cancelled.Value.(*generation), "session-cancel", "job-cancel", "")
	assertTaggedGeneration(t, continued.Value.(*generation), "session-continue", "job-continue", "yes")
}

func TestJobManagerRequestErrorsAreIsolated_Ugly(t *testing.T) {
	base := newRequestErrorTextModel()
	laneResult := newModelLane(base, "request-errors")
	if !laneResult.OK {
		t.Fatalf("newModelLane: %s", laneResult.Error())
	}
	lane := laneResult.Value.(*modelLane)
	defer lane.Close()
	jobs := newJobManager(context.Background())

	first := jobs.Start("session-first", "job-first", lane.Model(), []inference.Message{{Role: "user", Content: "fails"}}, nil)
	if !first.OK {
		t.Fatalf("start first: %s", first.Error())
	}
	firstText, firstErr := collectGeneration(t, first.Value.(*generation))
	if firstText != "" || firstErr == nil || !strings.Contains(firstErr.Error(), "first request failed") {
		t.Fatalf("first result = %q / %v", firstText, firstErr)
	}

	second := jobs.Start("session-second", "job-second", lane.Model(), []inference.Message{{Role: "user", Content: "succeeds"}}, nil)
	if !second.OK {
		t.Fatalf("start second: %s", second.Error())
	}
	secondText, secondErr := collectGeneration(t, second.Value.(*generation))
	if secondText != "second succeeds" || secondErr != nil {
		t.Fatalf("second result = %q / %v", secondText, secondErr)
	}
}

func collectGeneration(t *testing.T, generation *generation) (string, error) {
	t.Helper()
	var text strings.Builder
	var terminalErr error
	for event := range generation.events {
		text.WriteString(event.visible)
		if event.err != nil {
			terminalErr = event.err
		}
	}
	return text.String(), terminalErr
}

func consumeFakeChat(model inference.TextModel, prompt string, done chan<- string) {
	var text strings.Builder
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: prompt}}) {
		text.WriteString(token.Text)
	}
	done <- text.String()
}

func waitFakeStarted(t *testing.T, started <-chan string) string {
	t.Helper()
	select {
	case prompt := <-started:
		return prompt
	case <-time.After(2 * time.Second):
		t.Fatal("model did not start")
		return ""
	}
}

func assertNoFakeStarted(t *testing.T, started <-chan string) {
	t.Helper()
	select {
	case prompt := <-started:
		t.Fatalf("unexpected concurrent start %q", prompt)
	case <-time.After(50 * time.Millisecond):
	}
}

func waitFakeDone(t *testing.T, done <-chan string) string {
	t.Helper()
	select {
	case text := <-done:
		return text
	case <-time.After(2 * time.Second):
		t.Fatal("generation did not finish")
		return ""
	}
}

func assertTaggedGeneration(t *testing.T, generation *generation, sessionID, jobID, wantText string) {
	t.Helper()
	var text strings.Builder
	timeout := time.NewTimer(2 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case event, ok := <-generation.events:
			if !ok {
				if got := text.String(); got != wantText {
					t.Fatalf("stream text = %q, want %q", got, wantText)
				}
				return
			}
			if event.SessionID != sessionID || event.JobID != jobID {
				t.Fatalf("event IDs = %q/%q, want %q/%q", event.SessionID, event.JobID, sessionID, jobID)
			}
			text.WriteString(event.visible)
		case <-timeout.C:
			t.Fatal("timed out draining tagged generation")
		}
	}
}

type taggedGenerationExpectation struct {
	generation *generation
	sessionID  string
	jobID      string
	text       string
}

func assertInterleavedGenerations(t *testing.T, expected []taggedGenerationExpectation) {
	t.Helper()
	texts := make([]strings.Builder, len(expected))
	closed := make([]bool, len(expected))
	remaining := len(expected)
	timeout := time.NewTimer(2 * time.Second)
	defer timeout.Stop()
	for remaining > 0 {
		for i, stream := range expected {
			if closed[i] {
				continue
			}
			select {
			case event, ok := <-stream.generation.events:
				if !ok {
					closed[i] = true
					remaining--
					continue
				}
				if event.SessionID != stream.sessionID || event.JobID != stream.jobID {
					t.Fatalf("event IDs = %q/%q, want %q/%q", event.SessionID, event.JobID, stream.sessionID, stream.jobID)
				}
				texts[i].WriteString(event.visible)
			case <-timeout.C:
				t.Fatal("timed out interleaving tagged generations")
			}
		}
	}
	for i, stream := range expected {
		if got := texts[i].String(); got != stream.text {
			t.Fatalf("stream %q text = %q, want %q", stream.sessionID, got, stream.text)
		}
	}
}
