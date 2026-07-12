// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
	"os"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// simLaneSet is a fake inference.LaneSet: it scripts each admitted lane's tokens
// deterministically from its prompt and advances every active lane by one token
// per Step — so Step's call count is the batched-forward count the coordinator
// drives. It records Prepare calls so a test can prove which requests were
// admitted as lanes (vs falling back to the plain interleave path).
type simLaneSet struct {
	mu           sync.Mutex
	lanes        map[int]*simLane
	order        []int
	nextID       int
	fwd          uint64
	prepareCalls int
	closed       bool
}

type simLane struct {
	script   []int32
	idx      int
	terminal bool
}

func newSimLaneSet() *simLaneSet { return &simLaneSet{lanes: map[int]*simLane{}} }

func scriptFor(spec inference.LaneSpec) []int32 {
	var sum int32
	for _, id := range spec.PromptIDs {
		sum += id
	}
	out := make([]int32, spec.MaxNew)
	for i := range out {
		out[i] = sum*1000 + int32(i) // unique per (prompt, position)
	}
	return out
}

func (s *simLaneSet) Prepare(_ context.Context, spec inference.LaneSpec) (inference.LaneHandle, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prepareCalls++
	s.nextID++
	id := s.nextID
	s.lanes[id] = &simLane{script: scriptFor(spec)}
	s.order = append(s.order, id)
	return inference.LaneHandle{ID: id}, nil
}

func (s *simLaneSet) Step(context.Context) ([]inference.LaneStep, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []inference.LaneStep
	for _, id := range s.order {
		lane := s.lanes[id]
		if lane == nil || lane.terminal {
			continue
		}
		tok := lane.script[lane.idx]
		lane.idx++
		terminal := lane.idx >= len(lane.script)
		if terminal {
			lane.terminal = true
		}
		out = append(out, inference.LaneStep{Lane: inference.LaneHandle{ID: id}, Token: tok, HasToken: true, Terminal: terminal})
	}
	if len(out) == 0 {
		return nil, nil
	}
	s.fwd++ // ONE batched forward advanced every active lane
	return out, nil
}

func (s *simLaneSet) Retire(h inference.LaneHandle) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.lanes, h.ID)
	for i, id := range s.order {
		if id == h.ID {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	return nil
}

func (s *simLaneSet) Active() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.lanes)
}

func (s *simLaneSet) BatchForwardCount() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fwd
}

func (s *simLaneSet) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

// cbCapableModel is a fake TextModel that also exposes the tokenizer and the
// continuous-batching capability, so scheduler.New(interleave) builds a
// cbStepEngine over the sim.
type cbCapableModel struct {
	sim       *simLaneSet
	available bool
	chatSeq   []inference.Token // what the interleave fallback (Chat) yields
}

func (m *cbCapableModel) Generate(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.chat(ctx)
}
func (m *cbCapableModel) Chat(ctx context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.chat(ctx)
}
func (m *cbCapableModel) chat(ctx context.Context) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, t := range m.chatSeq {
			if ctx.Err() != nil || !yield(t) {
				return
			}
		}
	}
}
func (m *cbCapableModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}
func (m *cbCapableModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}
func (m *cbCapableModel) ModelType() string                   { return "cbcapable" }
func (m *cbCapableModel) Info() inference.ModelInfo           { return inference.ModelInfo{Architecture: "gemma4"} }
func (m *cbCapableModel) Metrics() inference.GenerateMetrics  { return inference.GenerateMetrics{} }
func (m *cbCapableModel) Err() core.Result                    { return core.Ok(nil) }
func (m *cbCapableModel) Close() core.Result                  { return core.Ok(nil) }
func (m *cbCapableModel) Encode(text string) []int32 { return []int32{int32(len([]rune(text)))} }
func (m *cbCapableModel) Decode(ids []int32) string  { return core.Sprintf("t%d", ids[0]) }
func (m *cbCapableModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	out := ""
	for _, msg := range messages {
		out += msg.Content
	}
	return out, nil
}

func (m *cbCapableModel) BatchStepAvailable() bool {
	return m.available && os.Getenv("LTHN_CB_STEP") != "0"
}
func (m *cbCapableModel) OpenLaneSet(inference.LaneSetConfig) (inference.LaneSet, error) {
	return m.sim, nil
}

var _ inference.TextModel = (*cbCapableModel)(nil)
var _ inference.TokenizerModel = (*cbCapableModel)(nil)
var _ inference.BatchStepModel = (*cbCapableModel)(nil)

// collectStream drains a scheduled-token channel into its token ids.
func collectStream(ch <-chan inference.ScheduledToken) []int32 {
	var ids []int32
	for st := range ch {
		ids = append(ids, st.Token.ID)
	}
	return ids
}

// TestCBStepCoordinatorBatchesRequests proves interleave mode, over a
// CB-capable model, drives K raw-prompt greedy requests through ONE shared lane
// set: each request gets its full scripted token stream, and the sim's batched
// forwards are far fewer than running the lanes serially (K×maxNew) would take —
// one Step advanced the whole set per round.
func TestCBStepCoordinatorBatchesRequests(t *testing.T) {
	sim := newSimLaneSet()
	model := &cbCapableModel{sim: sim, available: true}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 4, MaxQueue: 16, StreamBuffer: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()
	if sched.cbEngine == nil {
		t.Fatal("cbEngine should be built for an available BatchStepModel with a tokenizer")
	}

	const k, maxNew = 4, 6
	prompts := []string{"a", "bb", "ccc", "dddd"}
	want := make([][]int32, k)
	for i, p := range prompts {
		want[i] = scriptFor(inference.LaneSpec{PromptIDs: model.Encode(p), MaxNew: maxNew})
	}

	var wg sync.WaitGroup
	got := make([][]int32, k)
	for i := range prompts {
		_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
			ID:      core.Sprintf("r%d", i),
			Prompt:  prompts[i],
			Sampler: inference.SamplerConfig{MaxTokens: maxNew},
		})
		if err != nil {
			t.Fatalf("Schedule(%d): %v", i, err)
		}
		wg.Add(1)
		go func(i int, ch <-chan inference.ScheduledToken) {
			defer wg.Done()
			got[i] = collectStream(ch)
		}(i, ch)
	}
	wg.Wait()

	for i := range prompts {
		if len(got[i]) != maxNew {
			t.Fatalf("request %d: got %d tokens, want %d (%v)", i, len(got[i]), maxNew, got[i])
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("request %d token %d: got %d want %d", i, j, got[i][j], want[i][j])
			}
		}
	}
	if sim.prepareCalls != k {
		t.Fatalf("expected %d lanes admitted, got %d", k, sim.prepareCalls)
	}
	fwd := sim.BatchForwardCount()
	if fwd == 0 {
		t.Fatal("no batched forwards ran")
	}
	if fwd >= uint64(k*maxNew) {
		t.Fatalf("batched forwards %d not fewer than serial total %d — lanes were not sharing a forward", fwd, k*maxNew)
	}
}

// TestCBStepFallbackForChatAndUnavailable proves the fallbacks are real, not
// fake batches: a chat request is served by the plain interleave path (no lane
// admitted), and an unavailable capability leaves cbEngine unbuilt entirely.
func TestCBStepFallbackForChatAndUnavailable(t *testing.T) {
	// Chat request → interleave fallback (the CB path needs a raw prompt).
	sim := newSimLaneSet()
	model := &cbCapableModel{sim: sim, available: true, chatSeq: []inference.Token{{ID: 7}, {ID: 8}}}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()

	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:       "chat1",
		Messages: []inference.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Schedule(chat): %v", err)
	}
	ids := collectStream(ch)
	if len(ids) != 2 || ids[0] != 7 || ids[1] != 8 {
		t.Fatalf("chat request should stream via interleave fallback, got %v", ids)
	}
	if sim.prepareCalls != 0 {
		t.Fatalf("chat request must NOT admit a CB lane, got %d prepares", sim.prepareCalls)
	}

	// Unavailable capability → cbEngine never built.
	off := &cbCapableModel{sim: newSimLaneSet(), available: false}
	sched2, err := New(off, Config{Mode: ModeInterleave, MaxConcurrent: 2})
	if err != nil {
		t.Fatalf("New(unavailable): %v", err)
	}
	defer sched2.CloseEngine()
	if sched2.cbEngine != nil {
		t.Fatal("cbEngine must be nil when the capability is unavailable")
	}
}

// TestCBStepKillSwitchUnbindsScheduler proves LTHN_CB_STEP=0 leaves the coordinator
// unbuilt, so interleave mode is the plain per-request engine.
func TestCBStepKillSwitchUnbinds(t *testing.T) {
	t.Setenv("LTHN_CB_STEP", "0")
	model := &cbCapableModel{sim: newSimLaneSet(), available: true}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 2})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()
	if sched.cbEngine != nil {
		t.Fatal("LTHN_CB_STEP=0 must leave cbEngine unbuilt")
	}
}
