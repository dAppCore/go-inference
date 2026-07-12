// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	coreio "dappco.re/go/io"
)

// chatMLFixtureTokenizerJSON is fixtureTokenizerJSON plus the ChatML turn
// markers (<|im_start|>/<|im_end|>, at their real Qwen ids), so a ChatML
// template's Close + declared stop strings resolve through the production
// LoadTokenizer path.
const chatMLFixtureTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"z": 42},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<bos>", "special": true},
    {"id": 2, "content": "<eos>", "special": true},
    {"id": 151644, "content": "<|im_start|>", "special": true},
    {"id": 151645, "content": "<|im_end|>", "special": true}
  ]
}`

func newChatMLFixtureTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, chatMLFixtureTokenizerJSON); err != nil {
		t.Fatalf("write ChatML fixture tokenizer: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("load ChatML fixture tokenizer: %v", err)
	}
	return tok
}

// chatMLTokenModel is a fake engine.TokenModel that DECLARES a ChatML-family
// chat dialect (engine.ChatTemplateDeclarer): <|im_start|>role\n…<|im_end|>
// turns, an "assistant" generation role, system rendered in place (not folded),
// and a Qwen-style no-think block appended when thinking is off. It proves the
// neutral render loop frames a non-gemma dialect purely from the declared
// template — no real qwen model type is wired (that package is another tree),
// the fake proves the mechanism.
type chatMLTokenModel struct {
	TokenModel
}

func (chatMLTokenModel) DeclaredChatTemplate() (ChatTemplate, bool) {
	return ChatTemplate{
		Open:          "<|im_start|>",
		Close:         "<|im_end|>",
		UserRole:      "user",
		AssistantRole: "assistant",
		SystemRole:    "system",
		// ChatML renders a system message in place and spells it "system"
		// (SystemAsLeadingTurn/InlineSystemAsUser both stay false — the gemma
		// leading-fold + collapse-to-user rules do not apply).
		Thinking: &ChatThinking{OffSuffix: "<think>\n\n</think>\n\n"},
		Stops:    []string{"<|im_end|>"},
	}, true
}

// TestModel_ChatMLTemplate_Declared is the fake-model proof: a TokenModel that
// declares a ChatML template renders a full conversation through the neutral
// loop in ChatML — <|im_start|>/<|im_end|> markers, the "assistant" cue, an
// inline "system" turn, and the no-think block on thinking-off / nothing on
// thinking-on — with no gemma spelling leaking through.
func TestModel_ChatMLTemplate_Declared(t *testing.T) {
	tok := newChatMLFixtureTokenizer(t)
	m := NewTextModel(chatMLTokenModel{}, tok, "chatml-fake", inference.ModelInfo{}, 32)
	msgs := []inference.Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "Hi"},
	}

	off := false
	got := m.FormatChatPromptWithThinking(msgs, &off)
	want := "<|im_start|>system\nBe terse.<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if got != want {
		t.Fatalf("ChatML thinking-off render = %q, want %q", got, want)
	}

	on := true
	if gotOn, wantOn := m.FormatChatPromptWithThinking(msgs, &on),
		"<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"; gotOn != wantOn {
		t.Fatalf("ChatML thinking-on render = %q, want %q (no <think> block)", gotOn, wantOn)
	}

	// No gemma spelling leaked: the render carries neither gemma turn marker
	// nor the gemma "model" assistant role.
	if strings.Contains(got, "<|turn>") || strings.Contains(got, "<start_of_turn>") || strings.Contains(got, "<|im_start|>model\n") {
		t.Fatalf("ChatML render leaked a gemma spelling: %q", got)
	}
}

// TestModel_ChatMLTemplate_StopTokens proves the stop-set assembly resolves a
// non-gemma dialect's Close marker and its template-implied stop strings to ids
// against the model's own tokenizer, deduping the marker that both name.
func TestModel_ChatMLTemplate_StopTokens(t *testing.T) {
	tok := newChatMLFixtureTokenizer(t)
	m := NewTextModel(chatMLTokenModel{}, tok, "chatml-fake", inference.ModelInfo{}, 32)
	stop := m.stopTokens(inference.GenerateConfig{})

	imEnd, ok := tok.TokenID("<|im_end|>")
	if !ok {
		t.Fatal("ChatML fixture tokenizer is missing <|im_end|>")
	}
	if !tokenInSet(imEnd, stop) {
		t.Fatalf("ChatML stop set %v missing the <|im_end|> id %d", stop, imEnd)
	}
	// Close and Stops both name <|im_end|>; the dedup keeps it single.
	seen := 0
	for _, id := range stop {
		if id == imEnd {
			seen++
		}
	}
	if seen != 1 {
		t.Fatalf("stop set %v holds <|im_end|> %d times, want exactly once", stop, seen)
	}
}

// TestRenderChatTemplate_ChatMLInlineSystem pins the ChatML system rule through
// the neutral loop directly: a NON-leading system message spells "system"
// (InlineSystemAsUser stays false), the exact contrast with gemma where an
// inline system collapses to a user turn.
func TestRenderChatTemplate_ChatMLInlineSystem(t *testing.T) {
	tmpl := ChatTemplate{
		Open: "<|im_start|>", Close: "<|im_end|>",
		UserRole: "user", AssistantRole: "assistant", SystemRole: "system",
	}
	got := renderChatTemplate(tmpl, []inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "system", Content: "mid"},
		{Role: "assistant", Content: "ok"},
	}, nil)
	want := "<|im_start|>user\nhi<|im_end|>\n" +
		"<|im_start|>system\nmid<|im_end|>\n" +
		"<|im_start|>assistant\nok<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("ChatML inline-system render = %q, want %q", got, want)
	}
}

// TestGemmaChatTemplate_Fallback pins that the gemma template constructor is a
// faithful description of both gemma dialects — the fallback the undeclared
// path and the metal declaration both build, so declared == fallback for gemma.
func TestGemmaChatTemplate_Fallback(t *testing.T) {
	gemma4 := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, true)
	if !gemma4.SystemAsLeadingTurn || gemma4.Thinking == nil {
		t.Fatalf("gemma4 template = %+v, want a leading system turn + thinking hooks", gemma4)
	}
	if gemma4.Thinking.Prelude != "<|think|>\n" || gemma4.Thinking.OffSuffix != "<|channel>thought\n<channel|>" {
		t.Fatalf("gemma4 suppressor thinking = %+v, want the <|think|> switch + pre-closed channel", *gemma4.Thinking)
	}
	if gemma4.AssistantRole != "model" || !gemma4.InlineSystemAsUser {
		t.Fatalf("gemma4 roles = %+v, want assistant->model and inline-system->user", gemma4)
	}
	legacy := GemmaChatTemplate(TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}, true)
	if legacy.SystemAsLeadingTurn || legacy.Thinking != nil {
		t.Fatalf("gemma3-era template = %+v, want no system turn and no thinking channel even with suppressor set", legacy)
	}
}

// TestRenderChatTurns_StripsSystemAndThinking pins the exported plain-turns
// render (the seam engine/metal's speculative pair frames its chat prompt
// through): even on a gemma4 template that DECLARES a leading-system fold and a
// thinking channel, RenderChatTurns emits only plain turns + the generation cue
// — a leading system message spells as an ordinary (user) turn rather than a
// folded system turn, and no thinking prelude/suffix appears. It must be
// byte-identical to the unexported renderChatTurns it delegates to.
func TestRenderChatTurns_StripsSystemAndThinking(t *testing.T) {
	tmpl := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, true)
	msgs := []inference.Message{
		{Role: "system", Content: "Sys"},
		{Role: "user", Content: "Hi"},
	}
	got := RenderChatTurns(tmpl, msgs)
	want := "<|turn>user\nSys<turn|>\n" +
		"<|turn>user\nHi<turn|>\n" +
		"<|turn>model\n"
	if got != want {
		t.Fatalf("RenderChatTurns = %q, want plain turns with no system fold / no thinking %q", got, want)
	}
	if got != renderChatTurns(tmpl, msgs) {
		t.Fatalf("RenderChatTurns diverged from renderChatTurns: %q", got)
	}
}
