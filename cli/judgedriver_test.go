// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"iter"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/dataset"
)

// stubJudgeModel is a minimal inference.TextModel fake for judge-driver
// tests: no GPU, no real load — Chat replays one scripted reply (or an
// error via chatErr, surfaced through Err() the way a real engine reports a
// mid-stream failure after the iterator stops). newJudgeDispatcherFromModel
// is the only seam under test here; newJudgeDispatcher's real
// inference.LoadModel call is exercised end-to-end via cli/data_test.go's
// TestRunDataScore_Judge instead (a real GPU model run is the
// orchestrator's merge gate, never a unit test's job).
type stubJudgeModel struct {
	reply   string
	chatErr error
}

func (s *stubJudgeModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return s.Chat(ctx, nil, opts...)
}

func (s *stubJudgeModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if s.chatErr != nil || s.reply == "" {
			return
		}
		yield(inference.Token{Text: s.reply})
	}
}

func (s *stubJudgeModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("stubJudgeModel: Classify not implemented"))
}

func (s *stubJudgeModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("stubJudgeModel: BatchGenerate not implemented"))
}

func (s *stubJudgeModel) ModelType() string                  { return "stub" }
func (s *stubJudgeModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (s *stubJudgeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }

func (s *stubJudgeModel) Err() core.Result {
	if s.chatErr != nil {
		return core.Fail(s.chatErr)
	}
	return core.Ok(nil)
}

func (s *stubJudgeModel) Close() core.Result { return core.Ok(nil) }

// pairItemFixture builds a KindPair dataset.Item carrying (prompt,
// response) as its Content — the fixture judge-driver tests score.
func pairItemFixture(t *testing.T, prompt, response string) dataset.Item {
	t.Helper()
	content := core.JSONMarshal(dataset.PairContent{Prompt: prompt, Response: response})
	if !content.OK {
		t.Fatalf("marshal pair content: %v", content.Value)
	}
	return dataset.Item{ID: dataset.NewID(), Kind: dataset.KindPair, Content: content.Bytes()}
}

// ---- judgeItemPromptResponse ----

func TestJudgeItemPromptResponse_Good(t *testing.T) {
	t.Run("pair content", func(t *testing.T) {
		item := pairItemFixture(t, "hi", "hello")
		prompt, response, err := judgeItemPromptResponse(item)
		if err != nil || prompt != "hi" || response != "hello" {
			t.Fatalf("judgeItemPromptResponse = (%q, %q, %v)", prompt, response, err)
		}
	})
	t.Run("messages content reduces to the last exchange", func(t *testing.T) {
		mc := dataset.MessagesContent{Messages: []dataset.MessageTurn{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		}}
		content := core.JSONMarshal(mc)
		if !content.OK {
			t.Fatalf("marshal messages content: %v", content.Value)
		}
		item := dataset.Item{ID: dataset.NewID(), Kind: dataset.KindMessages, Content: content.Bytes()}
		prompt, response, err := judgeItemPromptResponse(item)
		// LastExchange newline-joins every turn before the last assistant
		// turn (go/dataset/dataset.go), so a single preceding "hi" turn
		// becomes "hi\n" — not go-inference driver behaviour, the frozen
		// contract's own reduction.
		if err != nil || prompt != "hi\n" || response != "hello" {
			t.Fatalf("judgeItemPromptResponse = (%q, %q, %v)", prompt, response, err)
		}
	})
}

func TestJudgeItemPromptResponse_Bad(t *testing.T) {
	t.Run("trace kind is not scorable", func(t *testing.T) {
		item := dataset.Item{ID: dataset.NewID(), Kind: dataset.KindTrace, Content: []byte(`{"x":1}`)}
		if _, _, err := judgeItemPromptResponse(item); err == nil {
			t.Fatalf("judgeItemPromptResponse(trace) = nil error")
		}
	})
	t.Run("malformed pair json", func(t *testing.T) {
		item := dataset.Item{ID: dataset.NewID(), Kind: dataset.KindPair, Content: []byte("not json")}
		if _, _, err := judgeItemPromptResponse(item); err == nil {
			t.Fatalf("judgeItemPromptResponse(malformed pair) = nil error")
		}
	})
	t.Run("messages with no assistant turn", func(t *testing.T) {
		mc := dataset.MessagesContent{Messages: []dataset.MessageTurn{{Role: "user", Content: "hi"}}}
		content := core.JSONMarshal(mc)
		if !content.OK {
			t.Fatalf("marshal messages content: %v", content.Value)
		}
		item := dataset.Item{ID: dataset.NewID(), Kind: dataset.KindMessages, Content: content.Bytes()}
		if _, _, err := judgeItemPromptResponse(item); err == nil {
			t.Fatalf("judgeItemPromptResponse(no assistant turn) = nil error")
		}
	})
}

// ---- newJudgeDispatcherFromModel: the stub-model driver tests ----

func TestNewJudgeDispatcherFromModel_Good(t *testing.T) {
	inRepoDir := t.TempDir()
	writeJudgeTemplateFile(t, inRepoDir, "quality", qualityTemplateFixture(""))
	stub := &stubJudgeModel{reply: "87"}
	dispatcher := newJudgeDispatcherFromModel(stub, "fake-model-path", 32, "", inRepoDir)

	item := pairItemFixture(t, "what is 2+2?", "4")
	verdict, err := dispatcher(context.Background(), "quality", item)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if verdict.Value != 87 {
		t.Errorf("Value = %v, want 87", verdict.Value)
	}
	if verdict.Fingerprint != "fake-model-path" {
		t.Errorf("Fingerprint = %q, want fake-model-path", verdict.Fingerprint)
	}
	if len(verdict.Payload) == 0 || !core.Contains(string(verdict.Payload), "87") {
		t.Errorf("Payload = %s, want it to carry the raw reply", verdict.Payload)
	}
}

// TestNewJudgeDispatcherFromModel_OverrideWins is the resolution-order
// contract exercised through the actual dispatch path (resolveJudgeTemplateFrom
// itself is covered directly in judgetemplate_test.go) — an override in the
// first dir wins even though a differently-scored in-repo default of the
// same name exists in the second.
func TestNewJudgeDispatcherFromModel_OverrideWins(t *testing.T) {
	overrideDir := t.TempDir()
	inRepoDir := t.TempDir()
	writeJudgeTemplateFile(t, overrideDir, "quality", qualityTemplateFixture("Override prompt for {{prompt}} / {{response}}.\n\n"+wantOnlyNumberInstruction))
	writeJudgeTemplateFile(t, inRepoDir, "quality", qualityTemplateFixture("In-repo prompt for {{prompt}} / {{response}}.\n\n"+wantOnlyNumberInstruction))

	stub := &stubJudgeModel{reply: "60"}
	dispatcher := newJudgeDispatcherFromModel(stub, "fp", 32, overrideDir, inRepoDir)
	item := pairItemFixture(t, "p", "r")
	if _, err := dispatcher(context.Background(), "quality", item); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
}

// TestNewJudgeDispatcherFromModel_CachesResolvedTemplate proves a template
// resolved once for a name is cached for the dispatcher's lifetime: scoring
// a second item against the same judge:<name> after the on-disk file has
// been removed still succeeds, because the second call never re-reads it.
func TestNewJudgeDispatcherFromModel_CachesResolvedTemplate(t *testing.T) {
	inRepoDir := t.TempDir()
	writeJudgeTemplateFile(t, inRepoDir, "quality", qualityTemplateFixture(""))
	stub := &stubJudgeModel{reply: "50"}
	dispatcher := newJudgeDispatcherFromModel(stub, "fp", 32, "", inRepoDir)
	item := pairItemFixture(t, "p", "r")

	if _, err := dispatcher(context.Background(), "quality", item); err != nil {
		t.Fatalf("first dispatch: %v", err)
	}
	if err := os.Remove(filepath.Join(inRepoDir, "quality.md")); err != nil {
		t.Fatalf("remove template fixture: %v", err)
	}
	if _, err := dispatcher(context.Background(), "quality", item); err != nil {
		t.Fatalf("second dispatch after the template file was removed: %v (want the cached template to still be used)", err)
	}
}

func TestNewJudgeDispatcherFromModel_Bad(t *testing.T) {
	inRepoDir := t.TempDir()
	writeJudgeTemplateFile(t, inRepoDir, "quality", qualityTemplateFixture(""))
	item := pairItemFixture(t, "p", "r")

	t.Run("unknown template name", func(t *testing.T) {
		dispatcher := newJudgeDispatcherFromModel(&stubJudgeModel{reply: "87"}, "fp", 32, "", inRepoDir)
		if _, err := dispatcher(context.Background(), "does-not-exist", item); err == nil {
			t.Fatalf("dispatch with an unknown template name = nil error")
		}
	})

	t.Run("generation error surfaces", func(t *testing.T) {
		dispatcher := newJudgeDispatcherFromModel(&stubJudgeModel{chatErr: core.NewError("model unreachable")}, "fp", 32, "", inRepoDir)
		if _, err := dispatcher(context.Background(), "quality", item); err == nil {
			t.Fatalf("dispatch with a failing model = nil error")
		}
	})

	t.Run("malformed item content", func(t *testing.T) {
		dispatcher := newJudgeDispatcherFromModel(&stubJudgeModel{reply: "87"}, "fp", 32, "", inRepoDir)
		badItem := dataset.Item{ID: dataset.NewID(), Kind: dataset.KindPair, Content: []byte("not json")}
		if _, err := dispatcher(context.Background(), "quality", badItem); err == nil {
			t.Fatalf("dispatch with malformed item content = nil error")
		}
	})

	t.Run("unscorable item kind", func(t *testing.T) {
		dispatcher := newJudgeDispatcherFromModel(&stubJudgeModel{reply: "87"}, "fp", 32, "", inRepoDir)
		traceItem := dataset.Item{ID: dataset.NewID(), Kind: dataset.KindTrace, Content: []byte(`{"anything":"non-empty"}`)}
		if _, err := dispatcher(context.Background(), "quality", traceItem); err == nil {
			t.Fatalf("dispatch against a trace item = nil error")
		}
	})
}

// TestNewJudgeDispatcherFromModel_Ugly proves malformed judge OUTPUT — a
// model that ignores the bare-number instruction, or answers out of range —
// is a loud per-item error, never a silently-accepted or clamped score.
func TestNewJudgeDispatcherFromModel_Ugly(t *testing.T) {
	inRepoDir := t.TempDir()
	writeJudgeTemplateFile(t, inRepoDir, "quality", qualityTemplateFixture(""))
	item := pairItemFixture(t, "p", "r")

	for _, tc := range []struct {
		name  string
		reply string
	}{
		{"non-numeric prose reply", "I would say this response is quite good, maybe 85."},
		{"out-of-range reply", "500"},
		{"empty reply", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dispatcher := newJudgeDispatcherFromModel(&stubJudgeModel{reply: tc.reply}, "fp", 32, "", inRepoDir)
			verdict, err := dispatcher(context.Background(), "quality", item)
			if err == nil {
				t.Fatalf("dispatch with malformed judge output %q = nil error, verdict=%+v (want a loud failure, never a silent score)", tc.reply, verdict)
			}
		})
	}
}

// ---- newJudgeDispatcher: the production wrapper's fast-fail path ----

// TestNewJudgeDispatcher_Bad proves an empty (or whitespace-only) model
// path is rejected before any inference.LoadModel call is attempted — the
// load-failure path itself (a real, non-empty but unreachable path) is
// exercised end-to-end through the verb in TestRunDataScore_Judge.
func TestNewJudgeDispatcher_Bad(t *testing.T) {
	for _, path := range []string{"", "   "} {
		if _, _, err := newJudgeDispatcher(path, 32); err == nil {
			t.Errorf("newJudgeDispatcher(%q, ...) = nil error, want a failure", path)
		}
	}
}
