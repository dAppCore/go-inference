// SPDX-Licence-Identifier: EUPL-1.2

// Package prompt is the prompt store and templating surface — the
// "stored prompt templates and presets" referenced by the inference serving
// layer (RFC §6.10) and the prompt-management row of the inference-stack
// map (RFC.inference-stack §5: versioned templates + templating).
//
// A Template is a versioned body with {{var}} placeholders; Render substitutes
// them. A Builder assembles a multi-turn chat template turn by turn. A Store
// keeps versioned templates addressable by id, with a goroutine-safe in-memory
// implementation.
//
//	tpl := prompt.NewBuilder().
//	    System("You are {{persona}}.").
//	    User("Help me with {{topic}}.").
//	    InputVariables("persona", "topic").
//	    Build()
//	out, _ := tpl.Render(map[string]string{"persona": "a coder", "topic": "Go"})
package prompt

import core "dappco.re/go"

// Template is a versioned prompt body addressable by ID. Body carries {{var}}
// placeholders; InputVars declares the variables the body is allowed to use —
// the declaration and the body must agree, which Render enforces.
//
//	tpl := prompt.Template{
//	    ID:        "greet",
//	    Body:      "Hello {{name}}.",
//	    InputVars: []string{"name"},
//	}
type Template struct {
	ID        string   `json:"id"`
	Version   int      `json:"version"`
	Body      string   `json:"body"`
	InputVars []string `json:"input_vars,omitempty"`
}

// Render substitutes every {{var}} placeholder in Body with its value from
// vars. A declared InputVar absent from vars is a missing-variable error; a
// placeholder present in Body but not declared in InputVars is an
// unknown-placeholder error; extra vars are ignored. On any error the empty
// string is returned alongside it.
//
//	tpl := prompt.Template{Body: "Hi {{name}}", InputVars: []string{"name"}}
//	out, err := tpl.Render(map[string]string{"name": "Nick"})  // "Hi Nick", nil
func (t Template) Render(vars map[string]string) (string, error) {
	found := placeholders(t.Body)

	// Every placeholder in the body must be declared as an InputVar.
	declared := make(map[string]bool, len(t.InputVars))
	for _, name := range t.InputVars {
		declared[name] = true
	}
	for _, name := range found {
		if !declared[name] {
			return "", core.E("prompt", core.Concat("undeclared placeholder {{", name, "}} in template body"), nil)
		}
	}

	// Every declared variable must be supplied.
	for _, name := range t.InputVars {
		if _, ok := vars[name]; !ok {
			return "", core.E("prompt", core.Concat("missing required variable ", name), nil)
		}
	}

	out := t.Body
	for _, name := range found {
		out = core.Replace(out, core.Concat("{{", name, "}}"), vars[name])
	}
	return out, nil
}

// placeholders returns the distinct {{name}} variable names in body, in order
// of first appearance. A {{ with no closing }} and an empty {{}} are literal
// text, not placeholders.
//
//	placeholders("{{a}} and {{b}} and {{a}}")  // ["a", "b"]
func placeholders(body string) []string {
	var names []string
	seen := make(map[string]bool)
	rest := body
	for {
		open := core.Index(rest, "{{")
		if open < 0 {
			break
		}
		after := rest[open+2:]
		close := core.Index(after, "}}")
		if close < 0 {
			break // no closing braces anywhere — the remainder is literal
		}
		name := after[:close]
		// Advance past this "{{" so a malformed token can't loop forever and a
		// nested "{{" inside the name is reconsidered from its own start.
		rest = after[close+2:]
		if name == "" || core.ContainsAny(name, "{}") {
			continue // {{}} or a stray brace run — literal, not a variable
		}
		if !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	return names
}
