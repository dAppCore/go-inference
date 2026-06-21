// SPDX-Licence-Identifier: EUPL-1.2

package prompt

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
)

// turn is one un-rendered chat turn held by the Builder.
type turn struct {
	role chat.Role
	text string
}

// Builder assembles a multi-turn chat prompt template turn by turn, in the
// ChatPromptBuilder style. System / User / Assistant append turns;
// InputVariables declares the variables every turn is allowed to reference.
// Build flattens the turns into a single Template; BuildMessages renders each
// turn against vars and returns the canonical chat.Message list.
//
//	tpl := prompt.NewBuilder().
//	    System("You are {{persona}}.").
//	    User("Help with {{topic}}.").
//	    InputVariables("persona", "topic").
//	    Build()
type Builder struct {
	turns     []turn
	inputVars []string
}

// NewBuilder returns an empty Builder.
//
//	b := prompt.NewBuilder()
func NewBuilder() *Builder {
	return &Builder{}
}

// System appends a system turn and returns the builder for chaining.
//
//	prompt.NewBuilder().System("You are {{persona}}.")
func (b *Builder) System(text string) *Builder {
	b.turns = append(b.turns, turn{role: chat.System, text: text})
	return b
}

// User appends a user turn and returns the builder for chaining.
//
//	prompt.NewBuilder().User("Help with {{topic}}.")
func (b *Builder) User(text string) *Builder {
	b.turns = append(b.turns, turn{role: chat.User, text: text})
	return b
}

// Assistant appends an assistant turn and returns the builder for chaining.
//
//	prompt.NewBuilder().Assistant("Sure, happy to help.")
func (b *Builder) Assistant(text string) *Builder {
	b.turns = append(b.turns, turn{role: chat.Assistant, text: text})
	return b
}

// InputVariables declares the variables every turn may reference. Calling it
// again replaces the set — the last declaration is the contract.
//
//	prompt.NewBuilder().User("{{topic}}").InputVariables("topic")
func (b *Builder) InputVariables(names ...string) *Builder {
	b.inputVars = append([]string(nil), names...)
	return b
}

// Build flattens the turns into a single Template, joining turn bodies with
// blank lines and carrying the declared input variables. The Template renders
// as a whole.
//
//	tpl := prompt.NewBuilder().System("You are {{p}}.").InputVariables("p").Build()
func (b *Builder) Build() Template {
	parts := make([]string, 0, len(b.turns))
	for _, tn := range b.turns {
		parts = append(parts, tn.text)
	}
	return Template{
		Body:      core.Join("\n\n", parts...),
		InputVars: append([]string(nil), b.inputVars...),
	}
}

// BuildMessages renders each turn's placeholders against vars and returns the
// canonical chat.Message list in turn order. Each turn is rendered as a one-turn
// Template carrying the builder's declared input variables, so a missing or
// undeclared variable surfaces as the same typed error Render produces; the
// rendered body becomes a single text content block.
//
//	msgs, err := prompt.NewBuilder().
//	    System("You are {{p}}.").
//	    InputVariables("p").
//	    BuildMessages(map[string]string{"p": "a coder"})
//	msgs[0].Text() == "You are a coder."
func (b *Builder) BuildMessages(vars map[string]string) ([]chat.Message, error) {
	msgs := make([]chat.Message, 0, len(b.turns))
	for _, tn := range b.turns {
		tpl := Template{Body: tn.text, InputVars: b.varsFor(tn.text)}
		content, err := tpl.Render(vars)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, chat.Message{
			Role:    tn.role,
			Content: []chat.ContentBlock{chat.Text(content)},
		})
	}
	return msgs, nil
}

// varsFor returns the declared input variables that actually appear in one
// turn's text, so each turn is rendered against only the variables it uses —
// a declared variable used by a different turn is not required here, but an
// undeclared placeholder in this turn still errors via Render.
func (b *Builder) varsFor(text string) []string {
	used := placeholders(text)
	declared := make(map[string]bool, len(b.inputVars))
	for _, name := range b.inputVars {
		declared[name] = true
	}
	out := make([]string, 0, len(used))
	for _, name := range used {
		if declared[name] {
			out = append(out, name)
		}
	}
	return out
}
