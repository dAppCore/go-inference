// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference"
)

// sinkDeliveringModel is a TextModel double honouring the engine's
// MetricsSink contract: each Generate/Chat applies its opts and, as its
// stream completes, delivers this generation's OWN metrics to the sink —
// exactly what engine.TextModel.decodeFromPrefilled does. It also carries
// the TokenizerModel surface so batch mode constructs over it.
type sinkDeliveringModel struct {
	immediateModel
	delivered inference.GenerateMetrics
}

func (m *sinkDeliveringModel) Generate(_ context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.lastPrompt = prompt
	return m.sinkSeq(opts)
}

func (m *sinkDeliveringModel) Chat(_ context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.lastMessages = append([]inference.Message(nil), messages...)
	return m.sinkSeq(opts)
}

func (m *sinkDeliveringModel) sinkSeq(opts []inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
		if cfg := inference.ApplyGenerateOpts(opts); cfg.MetricsSink != nil {
			cfg.MetricsSink(m.delivered)
		}
	}
}

func (m *sinkDeliveringModel) Encode(text string) []int32 { return make([]int32, len(text)) }
func (m *sinkDeliveringModel) Decode([]int32) string      { return "" }
func (m *sinkDeliveringModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	text := ""
	for _, msg := range messages {
		text += msg.Content
	}
	return text, nil
}

// sinkModeModel builds the shared fixture: two scripted tokens and a
// distinctive delivered-metrics value no global Metrics() read produces.
func sinkModeModel() *sinkDeliveringModel {
	return &sinkDeliveringModel{
		immediateModel: immediateModel{tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}},
		delivered:      inference.GenerateMetrics{PromptTokens: 11, GeneratedTokens: 2},
	}
}

// schedulerModeMetricsSink drives one facade Chat through the given mode with
// a request-scoped sink and asserts the base engine's per-request delivery
// arrives — the seam the openai handler's usage accounting reads. The
// facade's opts→SamplerConfig fold cannot hold a func, so this pins the lift
// onto ScheduledRequest.MetricsSink and the re-arm at dispatch.
func schedulerModeMetricsSink(t *testing.T, mode Mode) {
	t.Helper()
	base := sinkModeModel()
	sched, err := New(base, Config{Mode: mode, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New(%s): %v", mode, err)
	}
	defer sched.CloseEngine()

	var got inference.GenerateMetrics
	fired := 0
	sink := inference.WithMetricsSink(func(gm inference.GenerateMetrics) {
		got = gm
		fired++
	})
	n := 0
	for range sched.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, inference.WithMaxTokens(4), sink) {
		n++
	}
	if n != 2 {
		t.Fatalf("Chat streamed %d tokens, want the 2 scripted", n)
	}
	if fired != 1 {
		t.Fatalf("MetricsSink fired %d times, want exactly once", fired)
	}
	if got != base.delivered {
		t.Fatalf("sink metrics = %+v, want the base engine's per-request delivery %+v", got, base.delivered)
	}
}

// TestModel_Chat_MetricsSink_Serial_Good pins the serial worker pool.
func TestModel_Chat_MetricsSink_Serial_Good(t *testing.T) {
	schedulerModeMetricsSink(t, ModeSerial)
}

// TestModel_Chat_MetricsSink_Batch_Good pins continuous in-flight batching.
func TestModel_Chat_MetricsSink_Batch_Good(t *testing.T) {
	schedulerModeMetricsSink(t, ModeBatch)
}

// TestModel_Chat_MetricsSink_Interleave_Good pins the live admission-budget
// path (the base here exposes no lane set, so this is plain interleave).
func TestModel_Chat_MetricsSink_Interleave_Good(t *testing.T) {
	schedulerModeMetricsSink(t, ModeInterleave)
}

// TestModel_Chat_MetricsSink_Bad pins the no-sink request: every mode
// dispatches without re-arming anything and the stream is unchanged.
func TestModel_Chat_MetricsSink_Bad(t *testing.T) {
	for _, mode := range []Mode{ModeSerial, ModeBatch, ModeInterleave} {
		base := sinkModeModel()
		sched, err := New(base, Config{Mode: mode, MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 4})
		if err != nil {
			t.Fatalf("New(%s): %v", mode, err)
		}
		n := 0
		for range sched.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
			n++
		}
		sched.CloseEngine()
		if n != 2 {
			t.Fatalf("%s: Chat streamed %d tokens, want 2", mode, n)
		}
	}
}
