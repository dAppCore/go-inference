// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	coreio "dappco.re/go/io"
)

// TestChatMLDialect pins the config-driven serve-dialect selection the composed wrap declares from: every
// Qwen composed model_type is ChatML (incl. a future qwenX and any case), and the generic composed/hybrid
// ids plus a non-qwen family keep the gemma fallback.
func TestChatMLDialect(t *testing.T) {
	for _, mt := range []string{"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "qwen3_6", "Qwen3_7"} {
		if !ChatMLDialect(mt) {
			t.Errorf("ChatMLDialect(%q) = false, want true (Qwen hybrids speak ChatML)", mt)
		}
	}
	for _, mt := range []string{"composed", "hybrid", "gemma4", "mamba2", ""} {
		if ChatMLDialect(mt) {
			t.Errorf("ChatMLDialect(%q) = true, want false (non-qwen composed keeps the gemma fallback)", mt)
		}
	}
}

// chatMLFixtureTokenizerJSON carries BOTH the gemma4 <|turn> markers and the ChatML <|im_start|>/<|im_end|>
// markers, so DetectTurnTokens would pick the gemma4 dialect — the declared ChatML template must win.
const chatMLFixtureTokenizerJSON = `{
  "model": {"type": "BPE", "vocab": {"z": 42}, "merges": []},
  "added_tokens": [
    {"id": 1, "content": "<bos>", "special": true},
    {"id": 2, "content": "<eos>", "special": true},
    {"id": 105, "content": "<|turn>", "special": true},
    {"id": 106, "content": "<turn|>", "special": true},
    {"id": 151644, "content": "<|im_start|>", "special": true},
    {"id": 151645, "content": "<|im_end|>", "special": true}
  ]
}`

func loadChatMLFixtureTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, chatMLFixtureTokenizerJSON); err != nil {
		t.Fatalf("write ChatML fixture tokenizer: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("load ChatML fixture tokenizer: %v", err)
	}
	return tok
}

// fakeComposedChatMLModel is a fake engine.TokenModel that declares the SAME ChatML template the composed
// serve wrap declares for a Qwen model_type (engine/metal's chatMLChatTemplate). Mirroring
// engine/chat_template_test.go's fake, it proves the neutral render loop frames ChatML purely from the
// declared template — no real hybrid checkpoint loaded, no metallib needed.
type fakeComposedChatMLModel struct{ engine.TokenModel }

func (fakeComposedChatMLModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	return engine.ChatTemplate{
		Open:          "<|im_start|>",
		Close:         "<|im_end|>",
		UserRole:      "user",
		AssistantRole: "assistant",
		SystemRole:    "system",
		Thinking:      &engine.ChatThinking{OffSuffix: "<think>\n\n</think>\n\n"},
		Stops:         []string{"<|im_end|>"},
	}, true
}

// TestComposedChatMLServeRendering is the composed serve-wrap analogue of the engine's ChatML test: a Qwen
// composed model declares ChatML, so engine.NewTextModel frames <|im_start|>/<|im_end|> turns, the
// "assistant" cue, an inline "system" turn, and the no-think block on thinking-off / nothing on
// thinking-on. The fixture tokenizer carries <|turn>, so this also pins precedence: the DECLARED ChatML
// template wins over the tokenizer-detected gemma dialect — no gemma spelling leaks through.
func TestComposedChatMLServeRendering(t *testing.T) {
	tok := loadChatMLFixtureTokenizer(t)
	m := engine.NewTextModel(fakeComposedChatMLModel{}, tok, "qwen3_5", inference.ModelInfo{}, 32)
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
	gotOn := m.FormatChatPromptWithThinking(msgs, &on)
	wantOn := "<|im_start|>system\nBe terse.<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if gotOn != wantOn {
		t.Fatalf("ChatML thinking-on render = %q, want %q (no <think> block)", gotOn, wantOn)
	}

	// Precedence: the declared ChatML template beats the detected gemma <|turn> dialect.
	if strings.Contains(got, "<|turn>") || strings.Contains(got, "<start_of_turn>") || strings.Contains(got, "<|im_start|>model\n") {
		t.Fatalf("ChatML render leaked a gemma spelling (detection beat the declaration): %q", got)
	}
}
