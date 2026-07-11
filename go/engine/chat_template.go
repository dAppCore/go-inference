// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"strings"

	"dappco.re/go/inference"
)

// ChatTemplate is the chat dialect a loaded checkpoint was tuned on, described
// declaratively so ONE neutral render loop ([renderChatTemplate]) can frame a
// conversation exactly as the checkpoint's own chat_template.jinja does — the
// turn markers, the role spellings, whether a leading system message folds into
// a dedicated turn, how the reasoning channel renders on and off, and any
// stop strings the template implies. It replaces the gemma-hardcoded rendering
// that used to live in [TextModel], so a second architecture (ChatML-family and
// beyond) follows instructions through the same seam rather than being framed
// in gemma's dialect.
//
// The design expresses BOTH shipped families:
//
//   - gemma: Open/Close "<|turn>"/"<turn|>" (or the gemma3-era
//     "<start_of_turn>"/"<end_of_turn>"), AssistantRole "model", a leading
//     system turn that also carries the "<|think|>" reasoning switch, and — on
//     the large variants — a pre-closed empty thought channel on the generation
//     cue when thinking is off. Built by [GemmaChatTemplate].
//   - ChatML: Open/Close "<|im_start|>"/"<|im_end|>", AssistantRole
//     "assistant", system rendered in place, and a "<think>" reasoning block.
//
// A loaded token model DECLARES its template through [ChatTemplateDeclarer]; a
// model that declares nothing falls back to the tokenizer-detected gemma
// dialect, so gemma rendering is byte-for-byte unchanged.
type ChatTemplate struct {
	// Open and Close bracket a turn: a turn renders as
	// Open + role + "\n" + content + Close + "\n".
	Open  string
	Close string

	// UserRole, AssistantRole and SystemRole are the role spellings the
	// template writes (gemma: user / model / system; ChatML:
	// user / assistant / system). AssistantRole is also the trailing
	// generation cue's role (Open + AssistantRole + "\n").
	UserRole      string
	AssistantRole string
	SystemRole    string

	// SystemAsLeadingTurn folds a leading system message into one dedicated
	// system turn ahead of the history — the gemma rule, whose jinja emits a
	// single "<|turn>system" turn that also hosts the thinking switch. When
	// false a system message renders in place as an ordinary turn (the ChatML
	// rule).
	SystemAsLeadingTurn bool

	// InlineSystemAsUser renders a non-folded (in-place) system message with
	// UserRole rather than SystemRole. gemma sets it (its turn vocabulary keeps
	// "system" for the folded leading turn only and spells any other system
	// message as a user turn); ChatML leaves it false so every system message
	// spells "system".
	InlineSystemAsUser bool

	// Thinking is the reasoning-channel framing, kept as the one dialect-
	// divergent hook. Prelude is written into the leading system turn when
	// thinking is ON (gemma "<|think|>\n"); OffSuffix is appended after the
	// generation cue when thinking is OFF (gemma large variants'
	// "<|channel>thought\n<channel|>"; a ChatML no-think block
	// "<think>\n\n</think>\n\n"). nil means the dialect frames no reasoning
	// channel (the gemma3-era template, which ignores the thinking flag).
	//
	// NOTE Prelude renders only when a leading system turn is emitted; a
	// dialect wanting a thinking prelude without a system turn would key it
	// off the generation cue instead — neither shipped family needs that.
	Thinking *ChatThinking

	// Stops are template-implied stop strings beyond the tokenizer's own EOS
	// and the Close marker (which [TextModel.stopTokens] already folds in).
	// They are carried as text and resolved to ids against the model's
	// tokenizer, so the render side stays token-id-free.
	Stops []string
}

// ChatThinking is a [ChatTemplate]'s reasoning-channel rendering: the strings
// written when a request turns thinking on and off. See [ChatTemplate.Thinking].
type ChatThinking struct {
	// Prelude is written into the leading system turn when thinking is on.
	Prelude string
	// OffSuffix is appended after the generation cue when thinking is off.
	OffSuffix string
}

// ChatTemplateDeclarer is the optional [TokenModel] capability a loaded token
// model implements to declare its chat dialect, mirroring the other capability
// probes ([ThoughtSuppressorDeclarer], [StopTokenDeclarer]): the model package
// owns the family knowledge and DECLARES the template; [TextModel] only renders
// what is declared. A model that does not implement it (or returns ok=false)
// leaves [TextModel] to fall back to the tokenizer-detected gemma dialect.
type ChatTemplateDeclarer interface {
	DeclaredChatTemplate() (ChatTemplate, bool)
}

// GemmaChatTemplate builds the gemma dialect as a declared [ChatTemplate] from
// the detected turn markers and the large-variant thought-suppressor flag. It
// is the single source the fallback (undeclared model) and the metal engine's
// declaration both build from, so declared and fallback gemma rendering are
// identical.
//
// gemma4 (Open "<|turn>") folds a leading system turn and carries the
// "<|think|>" switch; the gemma3-era dialect (any other Open) has neither a
// system turn nor a thinking channel — matching the shipped chat_template.jinja
// of each. suppressor adds the pre-closed empty thought channel to the
// thinking-off cue, and only ever on gemma4 (the gemma3 dialect has no channel
// markers, so suppressor is ignored there).
func GemmaChatTemplate(turns TurnTokens, suppressor bool) ChatTemplate {
	t := ChatTemplate{
		Open:               turns.Open,
		Close:              turns.Close,
		UserRole:           "user",
		AssistantRole:      "model",
		SystemRole:         "system",
		InlineSystemAsUser: true,
	}
	if turns.Open == "<|turn>" {
		th := &ChatThinking{Prelude: "<|think|>\n"}
		if suppressor {
			th.OffSuffix = "<|channel>thought\n<channel|>"
		}
		t.SystemAsLeadingTurn = true
		t.Thinking = th
	}
	return t
}

// turnRole is the ordinary-turn (non-folded) role spelling for role: assistant
// and model spell AssistantRole; a system/developer message spells UserRole
// when InlineSystemAsUser is set (gemma) else SystemRole (ChatML); everything
// else spells UserRole. For a gemma template this is exactly the legacy
// chatTurnRole mapping (assistant/model → model, else → user).
func (t ChatTemplate) turnRole(role string) string {
	switch {
	case role == "assistant" || role == "model":
		return t.AssistantRole
	case role == "system" || role == "developer":
		if t.InlineSystemAsUser {
			return t.UserRole
		}
		return t.SystemRole
	default:
		return t.UserRole
	}
}

// renderChatTemplate is the ONE neutral render loop: it frames messages as a
// fresh chat prompt in template t's dialect and appends the trailing assistant
// generation cue. enableThinking honours the request's thinking flag through
// t.Thinking (nil = the dialect frames no reasoning channel). The gemma dialect
// routed through here is byte-for-byte the old formatChatPrompt output.
func renderChatTemplate(t ChatTemplate, messages []inference.Message, enableThinking *bool) string {
	thinking := enableThinking != nil && *enableThinking
	prelude, offSuffix := "", ""
	if t.Thinking != nil {
		if thinking {
			prelude = t.Thinking.Prelude
		} else {
			offSuffix = t.Thinking.OffSuffix
		}
	}
	sysFirst := t.SystemAsLeadingTurn && len(messages) > 0 && chatSystemRole(messages[0].Role)
	var out strings.Builder
	rest := messages
	// The leading system turn renders when the template folds a leading system
	// message OR the thinking prelude needs a system turn to live in.
	if t.SystemAsLeadingTurn && (sysFirst || prelude != "") {
		out.WriteString(t.Open + t.SystemRole + "\n")
		out.WriteString(prelude)
		if sysFirst {
			out.WriteString(strings.TrimSpace(messages[0].Content))
			rest = messages[1:]
		}
		out.WriteString(t.Close + "\n")
	}
	for _, msg := range rest {
		out.WriteString(t.Open + t.turnRole(msg.Role) + "\n" + msg.Content + t.Close + "\n")
	}
	out.WriteString(t.Open + t.AssistantRole + "\n")
	out.WriteString(offSuffix)
	return out.String()
}

// renderChatTurns renders messages as plain turns in template t's dialect — no
// leading-system fold and no thinking framing — followed by the trailing
// assistant generation cue. It is the continuation/append primitive: the
// woken-session path frames only the new tail this way, so it never re-opens a
// system turn or re-emits the thinking switch mid-conversation.
func renderChatTurns(t ChatTemplate, messages []inference.Message) string {
	t.SystemAsLeadingTurn = false
	t.Thinking = nil
	return renderChatTemplate(t, messages, nil)
}

// RenderChatTurns is the exported plain-turns render (see [renderChatTurns]) —
// the seam engine/metal's speculative pair frames its chat prompt through, so
// there is one turn-rendering implementation rather than a private duplicate.
func RenderChatTurns(t ChatTemplate, messages []inference.Message) string {
	return renderChatTurns(t, messages)
}
