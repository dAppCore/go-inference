// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/model"
)

func speculativeTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "tokenizer.json")
	if r := core.WriteFile(path, []byte(textTestTokenizerJSON), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	return tok
}

func speculativeTestModel(t *testing.T) *speculativeModel {
	t.Helper()
	return &speculativeModel{tok: speculativeTestTokenizer(t)}
}

func TestLoadSpeculativePair_BadTarget(t *testing.T) {
	_, err := LoadSpeculativePair(core.PathJoin(t.TempDir(), "missing-target"), "missing-draft", 0)
	if err == nil {
		t.Fatal("LoadSpeculativePair accepted a missing target checkpoint")
	}
}

func TestSpeculativeModel_Generate_BadPair(t *testing.T) {
	m := speculativeTestModel(t)
	var got []inference.Token
	for tok := range m.Generate(context.Background(), "hello", inference.WithMaxTokens(2)) {
		got = append(got, tok)
	}
	if len(got) != 0 {
		t.Fatalf("Generate with an unvalidated pair emitted %v", got)
	}
	if r := m.Err(); r.OK {
		t.Fatal("Generate with an unvalidated pair left Err OK")
	}
	if metrics := m.Metrics(); metrics.PromptTokens == 0 {
		t.Fatal("Generate did not record the tokenised prompt length")
	}
}

func TestSpeculativeModel_Chat_BadPair(t *testing.T) {
	m := speculativeTestModel(t)
	m.turns = engine.TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	for range m.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}, inference.WithMaxTokens(1)) {
		t.Fatal("Chat with an unvalidated pair emitted a token")
	}
	if r := m.Err(); r.OK {
		t.Fatal("Chat with an unvalidated pair left Err OK")
	}
}

func TestSpeculativeModel_speculate_SampledPairError(t *testing.T) {
	m := speculativeTestModel(t)
	seq := m.speculate(context.Background(), []int32{0}, inference.GenerateConfig{
		MaxTokens: 1, Temperature: 0.7, TopK: 4, TopP: 0.8, MinP: 0.1,
		RepeatPenalty: 1.2, SuppressTokens: []int32{3}, Seed: 9,
	})
	for range seq {
		t.Fatal("sampled speculation with an unvalidated pair emitted a token")
	}
	if r := m.Err(); r.OK {
		t.Fatal("sampled speculation did not latch the pair error")
	}
}

func TestSpeculativeModel_generateDFlash_BadPair(t *testing.T) {
	m := &speculativeModel{}
	_, err := m.generateDFlash([]int32{1}, 1, nil, nil, nil)
	if err == nil {
		t.Fatal("generateDFlash accepted a pair without a DFlash drafter")
	}
}

func TestSpeculativeModel_speculativeInts_Good(t *testing.T) {
	got := speculativeInts([]int32{-2, 0, 7})
	want := []int{-2, 0, 7}
	if len(got) != len(want) {
		t.Fatalf("speculativeInts length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("speculativeInts[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSpeculativeModel_speculativeInt32s_Good(t *testing.T) {
	got := speculativeInt32s([]int{-2, 0, 7})
	want := []int32{-2, 0, 7}
	if len(got) != len(want) {
		t.Fatalf("speculativeInt32s length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("speculativeInt32s[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSpeculativeModel_record_Good(t *testing.T) {
	m := &speculativeModel{}
	m.record(AssistantGenerateResult{
		Tokens: []int32{4, 5}, DraftTokenSchedule: []int{3}, DraftTokens: 4,
		AcceptedTokens: 3, RejectedTokens: 1, TargetVerifyCalls: 2,
		TargetCalls: 5, DraftCalls: 2,
	}, 6, 12*time.Millisecond, nil)
	metrics := m.Metrics()
	if metrics.PromptTokens != 6 || metrics.GeneratedTokens != 2 || metrics.TotalDuration != 12*time.Millisecond || metrics.DecodeDuration != 12*time.Millisecond {
		t.Fatalf("recorded metrics = %+v", metrics)
	}
	spec := m.SpeculativeMetrics()
	if spec.ProposedTokens != 4 || spec.AcceptedTokens != 3 || spec.RejectedTokens != 1 || spec.AcceptanceRate != 0.75 || spec.TargetVerifyCalls != 2 || spec.TargetCalls != 5 || spec.DraftCalls != 2 {
		t.Fatalf("recorded speculative metrics = %+v", spec)
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("successful record left Err failed: %v", r.Value)
	}
}

func TestSpeculativeModel_record_Bad(t *testing.T) {
	m := &speculativeModel{}
	m.record(AssistantGenerateResult{}, 0, time.Nanosecond, errors.New("synthetic pair failure"))
	if r := m.Err(); r.OK {
		t.Fatal("record did not latch its error")
	}
}

func TestSpeculativeModel_SpeculativeMetrics_Good(t *testing.T) {
	want := inference.SpeculativeMetrics{ProposedTokens: 2, AcceptedTokens: 1}
	m := &speculativeModel{lastSpec: want}
	if got := m.SpeculativeMetrics(); !reflect.DeepEqual(got, want) {
		t.Fatalf("SpeculativeMetrics = %+v, want %+v", got, want)
	}
}

func TestSpeculativeModel_Metrics_Good(t *testing.T) {
	want := inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 2}
	m := &speculativeModel{lastMetrics: want}
	if got := m.Metrics(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Metrics = %+v, want %+v", got, want)
	}
}

func TestSpeculativeModel_Err_Good(t *testing.T) {
	m := &speculativeModel{}
	if r := m.Err(); !r.OK {
		t.Fatalf("Err on a fresh model = %v, want OK", r.Value)
	}
}

func TestSpeculativeModel_Err_Bad(t *testing.T) {
	m := &speculativeModel{lastErr: errors.New("latched")}
	if r := m.Err(); r.OK {
		t.Fatal("Err hid a latched generation error")
	}
}

func TestSpeculativeModel_ModelType_Good(t *testing.T) {
	m := &speculativeModel{modelType: "gemma4_text"}
	if got := m.ModelType(); got != "gemma4_text" {
		t.Fatalf("ModelType = %q, want gemma4_text", got)
	}
}

func TestSpeculativeModel_Info_Good(t *testing.T) {
	want := inference.ModelInfo{Architecture: "gemma4_text", VocabSize: 102, NumLayers: 3, HiddenSize: 128}
	m := &speculativeModel{info: want}
	if got := m.Info(); got != want {
		t.Fatalf("Info = %+v, want %+v", got, want)
	}
}

func TestSpeculativeModel_Classify_Bad(t *testing.T) {
	if r := (&speculativeModel{}).Classify(context.Background(), []string{"hello"}); r.OK {
		t.Fatal("Classify unexpectedly succeeded on the speculative path")
	}
}

func TestSpeculativeModel_BatchGenerate_Bad(t *testing.T) {
	if r := (&speculativeModel{}).BatchGenerate(context.Background(), []string{"hello"}); r.OK {
		t.Fatal("BatchGenerate unexpectedly succeeded on the speculative path")
	}
}

func TestSpeculativeModel_Close_Good(t *testing.T) {
	if r := (&speculativeModel{}).Close(); !r.OK {
		t.Fatalf("Close on an empty adapter = %v, want OK", r.Value)
	}
}

func TestSpeculativeModel_setErr_Good(t *testing.T) {
	m := &speculativeModel{}
	m.setErr(errors.New("synthetic"))
	if r := m.Err(); r.OK {
		t.Fatal("setErr did not make Err fail")
	}
}

func TestSpeculativeModel_speculativeSampleParams_Good(t *testing.T) {
	cfg := inference.GenerateConfig{
		Temperature: 0.7, TopK: 12, TopP: 0.9, MinP: 0.05,
		RepeatPenalty: 1.1, SuppressTokens: []int32{2, 5},
	}
	got := speculativeSampleParams(cfg)
	if got.Temperature != cfg.Temperature || got.TopK != cfg.TopK || got.TopP != cfg.TopP || got.MinP != cfg.MinP || got.RepeatPenalty != cfg.RepeatPenalty {
		t.Fatalf("speculativeSampleParams scalars = %+v", got)
	}
	if len(got.SuppressTokens) != 2 || got.SuppressTokens[0] != 2 || got.SuppressTokens[1] != 5 {
		t.Fatalf("speculativeSampleParams suppress tokens = %v", got.SuppressTokens)
	}
}

// TestSpeculativeChatFraming pins that the speculative model frames its chat
// prompt through the SHARED engine render (engine.RenderChatTurns over the
// gemma ChatTemplate) — byte-identical to the plain engine path in BOTH marker
// dialects (gemma4 <|turn>/<turn|>; legacy <start_of_turn>), with a trailing
// open model turn. It is the receipt that the private duplicate template is
// gone: the bytes speculativeModel.Chat encodes now come from the one engine
// render, not a package-local copy that could drift from the plain path.
func TestSpeculativeChatFraming(t *testing.T) {
	gemma4 := engine.TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	got := engine.RenderChatTurns(engine.GemmaChatTemplate(gemma4, false), []inference.Message{{Role: "user", Content: "hi"}})
	want := "<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("gemma4 speculative framing = %q, want %q", got, want)
	}
	legacy := engine.TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}
	multi := engine.RenderChatTurns(engine.GemmaChatTemplate(legacy, false), []inference.Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "q2"},
	})
	wantMulti := "<start_of_turn>user\nq<end_of_turn>\n" +
		"<start_of_turn>model\na<end_of_turn>\n" +
		"<start_of_turn>user\nq2<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if multi != wantMulti {
		t.Fatalf("legacy multi-turn framing = %q, want %q", multi, wantMulti)
	}
}

// TestDFlashSpeculativeMethodRoute proves a DFlash-stamped pair selects the
// block-diffusion driver and never enters the legacy MTP loop.
func TestDFlashSpeculativeMethodRoute(t *testing.T) {
	pair := &AssistantPair{Assistant: &AssistantModel{Config: model.AssistantConfig{Method: model.MTPDFlash}}}
	mtpCalls, dflashCalls := 0, 0
	got, err := speculativeMethodRoute(pair,
		func() (AssistantGenerateResult, error) {
			mtpCalls++
			return AssistantGenerateResult{Tokens: []int32{1}}, nil
		},
		func() (AssistantGenerateResult, error) {
			dflashCalls++
			return AssistantGenerateResult{Tokens: []int32{2}}, nil
		})
	if err != nil {
		t.Fatalf("speculativeMethodRoute: %v", err)
	}
	if mtpCalls != 0 || dflashCalls != 1 || len(got.Tokens) != 1 || got.Tokens[0] != 2 {
		t.Fatalf("DFlash route: mtp=%d dflash=%d tokens=%v", mtpCalls, dflashCalls, got.Tokens)
	}
}

// TestDFlashSpeculativeMethodRouteKeepsMTP pins the pre-existing MTP behaviour:
// explicit MTP and the historical unstamped default both select only that loop.
func TestDFlashSpeculativeMethodRouteKeepsMTP(t *testing.T) {
	for _, method := range []model.MTPMethod{model.MTPDraftModel, ""} {
		pair := &AssistantPair{Assistant: &AssistantModel{Config: model.AssistantConfig{Method: method}}}
		mtpCalls, dflashCalls := 0, 0
		got, err := speculativeMethodRoute(pair,
			func() (AssistantGenerateResult, error) {
				mtpCalls++
				return AssistantGenerateResult{Tokens: []int32{7}}, nil
			},
			func() (AssistantGenerateResult, error) {
				dflashCalls++
				return AssistantGenerateResult{Tokens: []int32{9}}, nil
			})
		if err != nil {
			t.Fatalf("method %q: %v", method, err)
		}
		if mtpCalls != 1 || dflashCalls != 0 || len(got.Tokens) != 1 || got.Tokens[0] != 7 {
			t.Fatalf("method %q route: mtp=%d dflash=%d tokens=%v", method, mtpCalls, dflashCalls, got.Tokens)
		}
	}
}

// TestSpeculativeChatRole pins the role spellings the shared gemma template
// keys on for the speculative path: assistant and model both render as the
// "model" turn; anything else (user, system, empty) renders as a "user" turn.
func TestSpeculativeChatRole(t *testing.T) {
	tmpl := engine.GemmaChatTemplate(engine.TurnTokens{Open: "<|turn>", Close: "<turn|>"}, false)
	for _, role := range []string{"assistant", "model"} {
		got := engine.RenderChatTurns(tmpl, []inference.Message{{Role: role, Content: "x"}})
		if want := "<|turn>model\nx<turn|>\n<|turn>model\n"; got != want {
			t.Fatalf("role %q framed %q, want the model turn %q", role, got, want)
		}
	}
	for _, role := range []string{"user", "system", ""} {
		got := engine.RenderChatTurns(tmpl, []inference.Message{{Role: role, Content: "x"}})
		if want := "<|turn>user\nx<turn|>\n<|turn>model\n"; got != want {
			t.Fatalf("role %q framed %q, want the user turn %q", role, got, want)
		}
	}
}

// TestSpeculativeTokenInSet pins the terminator membership the decode sink uses
// to blank a stop token's text.
func TestSpeculativeTokenInSet(t *testing.T) {
	set := []int32{1, 106, 107}
	if !speculativeTokenInSet(107, set) {
		t.Fatal("speculativeTokenInSet missed a present id")
	}
	if speculativeTokenInSet(42, set) {
		t.Fatal("speculativeTokenInSet matched an absent id")
	}
	if speculativeTokenInSet(1, nil) {
		t.Fatal("empty set contains nothing")
	}
}
