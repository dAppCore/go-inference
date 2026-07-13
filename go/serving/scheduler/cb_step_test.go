// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
	"os"
	"slices"
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
	specs        []inference.LaneSpec
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
	s.specs = append(s.specs, spec)
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
func (m *cbCapableModel) ModelType() string { return "cbcapable" }
func (m *cbCapableModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "gemma4"}
}
func (m *cbCapableModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *cbCapableModel) Err() core.Result                   { return core.Ok(nil) }
func (m *cbCapableModel) Close() core.Result                 { return core.Ok(nil) }
func (m *cbCapableModel) Encode(text string) []int32         { return []int32{int32(len([]rune(text)))} }
func (m *cbCapableModel) Decode(ids []int32) string          { return core.Sprintf("t%d", ids[0]) }
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
	// Chat request against a RENDERER-LESS model → interleave fallback (the CB
	// chat route needs the model's FormatChatPrompt capability; this fake has
	// none, so its chat turns keep the plain path).
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

// cbChatCapableModel is cbCapableModel plus the chat renderer + stop resolver
// capabilities the CB chat route probes (engine.TextModel's shape).
type cbChatCapableModel struct {
	cbCapableModel
}

func (m *cbChatCapableModel) FormatChatPrompt(messages []inference.Message) string {
	out := "R"
	for _, msg := range messages {
		out += "|" + msg.Role + ":" + msg.Content
	}
	return out
}

func (m *cbChatCapableModel) ResolvedStopTokens(requestStops []int32) []int32 {
	return append(append([]int32(nil), requestStops...), 99) // 99 = the fake's EOS
}

// TestCBStepChatRequestRidesLaneSet proves a TEXT-ONLY chat turn rides the
// continuous-batching path when the model exposes the chat renderer: the lane's
// prompt is Encode(FormatChatPrompt(messages)) — the model's own template
// string — and its stop set is the FULL resolution (request stops + the
// model's own), so the lane terminates exactly where the plain path would.
func TestCBStepChatRequestRidesLaneSet(t *testing.T) {
	sim := newSimLaneSet()
	model := &cbChatCapableModel{cbCapableModel{sim: sim, available: true}}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()

	msgs := []inference.Message{{Role: "user", Content: "hi"}}
	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:       "chat-cb",
		Messages: msgs,
		Sampler:  inference.SamplerConfig{MaxTokens: 2, StopTokens: []int32{41}},
	})
	if err != nil {
		t.Fatalf("Schedule(chat): %v", err)
	}
	ids := collectStream(ch)
	if len(ids) != 2 {
		t.Fatalf("chat request should stream 2 scripted tokens via the lane set, got %v", ids)
	}
	if sim.prepareCalls != 1 {
		t.Fatalf("text-only chat with a renderer must admit exactly one CB lane, got %d prepares", sim.prepareCalls)
	}
	spec := sim.specs[0]
	wantIDs := model.Encode(model.FormatChatPrompt(msgs))
	if !slices.Equal(spec.PromptIDs, wantIDs) {
		t.Fatalf("lane prompt must be the rendered template's tokens: got %v want %v", spec.PromptIDs, wantIDs)
	}
	if !slices.Equal(spec.StopTokens, []int32{41, 99}) {
		t.Fatalf("lane stops must be the FULL resolution (request + model): got %v want [41 99]", spec.StopTokens)
	}
}

// TestCBStepMultimodalChatFallsBack proves a chat turn carrying media keeps the
// plain interleave path even when the renderer capability is present — the CB
// route serves text-only turns.
func TestCBStepMultimodalChatFallsBack(t *testing.T) {
	sim := newSimLaneSet()
	model := &cbChatCapableModel{cbCapableModel{sim: sim, available: true, chatSeq: []inference.Token{{ID: 7}}}}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()

	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:       "chat-img",
		Messages: []inference.Message{{Role: "user", Content: "what is this", Images: [][]byte{{1, 2, 3}}}},
	})
	if err != nil {
		t.Fatalf("Schedule(multimodal chat): %v", err)
	}
	ids := collectStream(ch)
	if len(ids) != 1 || ids[0] != 7 {
		t.Fatalf("multimodal chat should stream via interleave fallback, got %v", ids)
	}
	if sim.prepareCalls != 0 {
		t.Fatalf("multimodal chat must NOT admit a CB lane, got %d prepares", sim.prepareCalls)
	}
}

// TestCBStepRawStopsResolved proves the raw-prompt CB route ALSO arms the full
// stop resolution when the model exposes it — a raw lane terminates on the
// model's EOS exactly as the plain Generate path does, not only on request
// stops.
func TestCBStepRawStopsResolved(t *testing.T) {
	sim := newSimLaneSet()
	model := &cbChatCapableModel{cbCapableModel{sim: sim, available: true}}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()

	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:      "raw-stops",
		Prompt:  "abc",
		Sampler: inference.SamplerConfig{MaxTokens: 1},
	})
	if err != nil {
		t.Fatalf("Schedule(raw): %v", err)
	}
	collectStream(ch)
	if sim.prepareCalls != 1 {
		t.Fatalf("raw request must admit one CB lane, got %d prepares", sim.prepareCalls)
	}
	if !slices.Equal(sim.specs[0].StopTokens, []int32{99}) {
		t.Fatalf("raw lane stops must carry the model's resolved set: got %v want [99]", sim.specs[0].StopTokens)
	}
}

// TestCBStepSampledRequestRidesLaneSet proves a SAMPLED raw-prompt request now
// routes to the continuous-batching path (the per-lane sampler rung) with its
// SamplerConfig carried intact onto the LaneSpec — sampling is no longer a
// fallback wall; only chat turns keep the plain interleave path.
func TestCBStepSampledRequestRidesLaneSet(t *testing.T) {
	sim := newSimLaneSet()
	model := &cbCapableModel{sim: sim, available: true}
	sched, err := New(model, Config{Mode: ModeInterleave, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()

	cfg := inference.SamplerConfig{Temperature: 0.8, TopK: 40, TopP: 0.95, MaxTokens: 3}
	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:      "sampled1",
		Prompt:  "abc",
		Sampler: cfg,
	})
	if err != nil {
		t.Fatalf("Schedule(sampled): %v", err)
	}
	ids := collectStream(ch)
	if len(ids) != 3 {
		t.Fatalf("sampled request should stream 3 scripted tokens via the lane set, got %v", ids)
	}
	if sim.prepareCalls != 1 {
		t.Fatalf("sampled raw-prompt request must admit exactly one CB lane, got %d prepares", sim.prepareCalls)
	}
	got := sim.specs[0].Sampler
	if got.Temperature != cfg.Temperature || got.TopK != cfg.TopK || got.TopP != cfg.TopP {
		t.Fatalf("LaneSpec.Sampler must carry the request's config intact: got %+v want %+v", got, cfg)
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
