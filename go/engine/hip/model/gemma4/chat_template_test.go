// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// TestChatTemplate_FormatChatTemplateWithConfig_Good pins the large-variant
// generation cue: with thinking off, a LargeVariant render pre-closes an empty
// thought channel after the trailing model turn — byte-for-byte the
// 12B/26B/31B chat_template.jinja branch
// `{%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}`.
func TestChatTemplate_FormatChatTemplateWithConfig_Good(t *testing.T) {
	msgs := []inference.Message{{Role: "user", Content: "Hi"}}
	got := FormatChatTemplateWithConfig(msgs, ChatTemplateConfig{LargeVariant: true})
	want := "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("large-variant thinking-off = %q, want %q", got, want)
	}
}

// TestChatTemplate_FormatChatTemplateWithConfig_Bad pins the small-variant
// generation cue: the E2B/E4B templates carry NO pre-closed thought channel,
// so a non-LargeVariant thinking-off render must end on the bare model turn —
// appending the channel there ships bytes the checkpoint was never trained on.
func TestChatTemplate_FormatChatTemplateWithConfig_Bad(t *testing.T) {
	msgs := []inference.Message{{Role: "user", Content: "Hi"}}
	got := FormatChatTemplateWithConfig(msgs, ChatTemplateConfig{})
	want := "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("small-variant thinking-off = %q, want the bare generation cue %q", got, want)
	}
}

// TestChatTemplate_FormatChatTemplateWithConfig_Ugly pins the corner cases of
// the suppressor gate: thinking ON never renders the pre-closed channel even
// on a large variant (the jinja branch is `not enable_thinking`), and
// NoGenerationPrompt suppresses the whole cue, channel included.
func TestChatTemplate_FormatChatTemplateWithConfig_Ugly(t *testing.T) {
	msgs := []inference.Message{{Role: "user", Content: "Hi"}}
	thinking := FormatChatTemplateWithConfig(msgs, ChatTemplateConfig{EnableThinking: true, LargeVariant: true})
	if strings.Contains(thinking, "<|channel>thought") {
		t.Fatalf("thinking-on large variant = %q, must not pre-close a thought channel", thinking)
	}
	if !strings.HasSuffix(thinking, "<|turn>model\n") {
		t.Fatalf("thinking-on large variant = %q, want the plain generation cue", thinking)
	}
	noCue := FormatChatTemplateWithConfig(msgs, ChatTemplateConfig{LargeVariant: true, NoGenerationPrompt: true})
	if strings.Contains(noCue, "<|turn>model") || strings.Contains(noCue, "<|channel>thought") {
		t.Fatalf("NoGenerationPrompt render = %q, must carry neither the cue nor the channel", noCue)
	}
}
