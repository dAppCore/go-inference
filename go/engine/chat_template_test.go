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

// TestChatTemplate_ResolveThinking_Good pins the family-default resolution: an
// unset flag takes the dialect's DefaultOn (gemma4 → ON, #1847).
func TestChatTemplate_ResolveThinking_Good(t *testing.T) {
	g4 := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, false)
	if !g4.ResolveThinking(nil) {
		t.Fatal("gemma4 ResolveThinking(nil) = false, want the family default ON")
	}
}

// TestChatTemplate_ResolveThinking_Bad pins the explicit override: a set flag
// always wins over the dialect default, both directions.
func TestChatTemplate_ResolveThinking_Bad(t *testing.T) {
	g4 := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, true)
	off, on := false, true
	if g4.ResolveThinking(&off) {
		t.Fatal("explicit false lost to the dialect default")
	}
	if !g4.ResolveThinking(&on) {
		t.Fatal("explicit true did not resolve on")
	}
}

// TestChatTemplate_ResolveThinking_Ugly pins the no-channel dialects: a
// template without a Thinking framing (gemma3-era) resolves off on an unset
// flag regardless of anything else.
func TestChatTemplate_ResolveThinking_Ugly(t *testing.T) {
	legacy := GemmaChatTemplate(TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}, true)
	if legacy.ResolveThinking(nil) {
		t.Fatal("legacy dialect ResolveThinking(nil) = true, want off (no reasoning channel)")
	}
}

// TestChatTemplate_MergeAdjacentAssistant_Good pins the gemma4 turn-tag-balance
// rule (canonical chat_template.jinja, 2026-07-09): DIRECTLY consecutive
// assistant messages fold into ONE model turn — opened once, contents
// concatenated in order, closed once — instead of the unbalanced close/reopen
// pair the pre-fix template emitted.
func TestChatTemplate_MergeAdjacentAssistant_Good(t *testing.T) {
	tmpl := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, false)
	if !tmpl.MergeAdjacentAssistant {
		t.Fatal("gemma4 dialect must declare MergeAdjacentAssistant")
	}
	got := RenderChatTurns(tmpl, []inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "part one, "},
		{Role: "assistant", Content: "part two"},
	})
	want := "<|turn>user\nhi<turn|>\n" +
		"<|turn>model\npart one, part two<turn|>\n" +
		"<|turn>model\n"
	if got != want {
		t.Fatalf("gemma4 adjacent-assistant render = %q, want %q", got, want)
	}
}

// TestChatTemplate_MergeAdjacentAssistant_Bad pins the contrast: ChatML
// declares no fold, so consecutive assistant messages stay two balanced turns.
func TestChatTemplate_MergeAdjacentAssistant_Bad(t *testing.T) {
	got := RenderChatTurns(ChatMLChatTemplate(), []inference.Message{
		{Role: "assistant", Content: "one"},
		{Role: "assistant", Content: "two"},
	})
	want := "<|im_start|>assistant\none<|im_end|>\n" +
		"<|im_start|>assistant\ntwo<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("ChatML adjacent-assistant render = %q, want %q", got, want)
	}
}

// TestChatTemplate_MergeAdjacentAssistant_Ugly pins the fold's boundaries: a
// user message between assistant turns breaks adjacency, consecutive USER
// messages never fold (the rule is assistant-only), and the gemma3-era dialect
// declares no fold at all.
func TestChatTemplate_MergeAdjacentAssistant_Ugly(t *testing.T) {
	g4 := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, false)
	got := RenderChatTurns(g4, []inference.Message{
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "u"},
		{Role: "user", Content: "v"},
		{Role: "assistant", Content: "b"},
	})
	want := "<|turn>model\na<turn|>\n" +
		"<|turn>user\nu<turn|>\n" +
		"<|turn>user\nv<turn|>\n" +
		"<|turn>model\nb<turn|>\n" +
		"<|turn>model\n"
	if got != want {
		t.Fatalf("gemma4 non-adjacent render = %q, want %q", got, want)
	}
	if g3 := GemmaChatTemplate(TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}, false); g3.MergeAdjacentAssistant {
		t.Fatal("gemma3-era dialect must not declare MergeAdjacentAssistant")
	}
}

// chatMLWithDefaultSystem is the ChatML dialect carrying a DefaultSystem — the
// Qwen2.5-Coder shape (its jinja frames "You are a helpful assistant." as a
// leading system turn when the caller sends none).
func chatMLWithDefaultSystem() ChatTemplate {
	return ChatTemplate{
		Open:          "<|im_start|>",
		Close:         "<|im_end|>",
		UserRole:      "user",
		AssistantRole: "assistant",
		SystemRole:    "system",
		DefaultSystem: "You are a helpful assistant.",
	}
}

// TestRenderChatTemplate_DefaultSystem_Good: a no-system chat frames the
// checkpoint's DefaultSystem as a leading system turn — byte-identical to
// Qwen2.5-Coder's apply_chat_template output for the same messages.
func TestRenderChatTemplate_DefaultSystem_Good(t *testing.T) {
	got := renderChatTemplate(chatMLWithDefaultSystem(),
		[]inference.Message{{Role: "user", Content: "The capital of France is"}}, nil)
	want := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nThe capital of France is<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("no-system render mismatch\n got: %q\nwant: %q", got, want)
	}
}

// TestRenderChatTemplate_DefaultSystem_Bad: a caller-supplied system message
// wins — the default is NOT injected, the provided system renders in place.
func TestRenderChatTemplate_DefaultSystem_Bad(t *testing.T) {
	got := renderChatTemplate(chatMLWithDefaultSystem(), []inference.Message{
		{Role: "system", Content: "Custom system."},
		{Role: "user", Content: "hi"},
	}, nil)
	want := "<|im_start|>system\nCustom system.<|im_end|>\n" +
		"<|im_start|>user\nhi<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("supplied-system render mismatch\n got: %q\nwant: %q", got, want)
	}
}

// TestRenderChatTemplate_DefaultSystem_Ugly: an empty DefaultSystem injects
// nothing — the Qwen3.5/3.6 rule (their jinja emits a system turn ONLY when the
// caller provides one), so a plain ChatML render is unchanged.
func TestRenderChatTemplate_DefaultSystem_Ugly(t *testing.T) {
	tmpl := chatMLWithDefaultSystem()
	tmpl.DefaultSystem = ""
	got := renderChatTemplate(tmpl, []inference.Message{{Role: "user", Content: "hi"}}, nil)
	want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("empty-default render mismatch\n got: %q\nwant: %q", got, want)
	}
}

// TestRenderChatTurns_DefaultSystem: a continuation (woken-session tail) never
// re-injects the default system — it was emitted on the fresh prompt.
func TestRenderChatTurns_DefaultSystem(t *testing.T) {
	got := renderChatTurns(chatMLWithDefaultSystem(),
		[]inference.Message{{Role: "user", Content: "hi"}})
	want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("continuation re-injected the default system\n got: %q\nwant: %q", got, want)
	}
}

// TestExtractDefaultSystem_Good: a Qwen2.5-style template's hardcoded no-system
// default is pulled verbatim — the checkpoint's exact wording, which differs
// Coder-vs-Instruct. The "\n" is the two-character escape a JSON-decoded
// tokenizer_config.json (and a raw chat_template.jinja) both carry.
func TestExtractDefaultSystem_Good(t *testing.T) {
	coder := `{%- else %}{{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{%- endif %}`
	if got := ExtractDefaultSystem(coder); got != "You are a helpful assistant." {
		t.Fatalf("Coder default = %q, want %q", got, "You are a helpful assistant.")
	}
	instruct := `{{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}`
	want := "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
	if got := ExtractDefaultSystem(instruct); got != want {
		t.Fatalf("Instruct default = %q, want %q", got, want)
	}
}

// TestExtractDefaultSystem_Bad: a passthrough template — one that frames a
// system turn only from the caller's own message (Qwen3.5/3.6) — has no
// hardcoded default, so extraction yields "" (the interpolation's leading quote
// is rejected, never mistaken for a literal default).
func TestExtractDefaultSystem_Bad(t *testing.T) {
	qwen35 := `{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}{%- endif %}`
	if got := ExtractDefaultSystem(qwen35); got != "" {
		t.Fatalf("passthrough template default = %q, want \"\"", got)
	}
}

// TestExtractDefaultSystem_Ugly: a non-ChatML template (gemma turn markers, no
// <|im_start|>system literal at all) yields "" — gemma checkpoints inject no
// default and their rendering stays byte-unchanged.
func TestExtractDefaultSystem_Ugly(t *testing.T) {
	gemma := `{{ '<|turn>user\n' + message.content + '<turn|>\n' }}`
	if got := ExtractDefaultSystem(gemma); got != "" {
		t.Fatalf("gemma template default = %q, want \"\"", got)
	}
}
