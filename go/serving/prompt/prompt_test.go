// SPDX-Licence-Identifier: EUPL-1.2

package prompt

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// --- Render ------------------------------------------------------------------

func TestPrompt_Render_Good(t *core.T) {
	// A template substitutes every declared placeholder from the vars map and
	// ignores any extra vars the caller supplies.
	//
	//	tpl := prompt.Template{Body: "Hi {{name}}", InputVars: []string{"name"}}
	//	out, _ := tpl.Render(map[string]string{"name": "Nick"})  // "Hi Nick"
	tpl := Template{
		ID:        "greet",
		Version:   1,
		Body:      "Hello {{name}}, welcome to {{place}}.",
		InputVars: []string{"name", "place"},
	}
	out, err := tpl.Render(map[string]string{"name": "Nick", "place": "OFM", "extra": "ignored"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "Hello Nick, welcome to OFM.", out)

	// A repeated placeholder is replaced at every occurrence.
	rep := Template{Body: "{{x}}-{{x}}-{{x}}", InputVars: []string{"x"}}
	out, err = rep.Render(map[string]string{"x": "go"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "go-go-go", out)

	// A template with no placeholders and no declared vars renders verbatim.
	plain := Template{Body: "no placeholders here"}
	out, err = plain.Render(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "no placeholders here", out)

	// A nil map is fine when nothing is required.
	empty := Template{Body: ""}
	out, err = empty.Render(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", out)
}

func TestPrompt_Render_Bad(t *core.T) {
	// A declared InputVar that is missing from vars is a typed error.
	tpl := Template{Body: "Hi {{name}}", InputVars: []string{"name"}}
	out, err := tpl.Render(map[string]string{})
	core.AssertError(t, err, "name")
	core.AssertEqual(t, "", out, "a failed render returns the empty string")
	core.AssertEqual(t, "prompt", core.Operation(err))

	// A nil map with a required var is equally a missing-var error.
	out, err = tpl.Render(nil)
	core.AssertError(t, err, "name")
	core.AssertEqual(t, "", out)

	// One missing var among several is still reported.
	multi := Template{Body: "{{a}} {{b}}", InputVars: []string{"a", "b"}}
	_, err = multi.Render(map[string]string{"a": "1"})
	core.AssertError(t, err, "b")
}

func TestPrompt_Render_Ugly(t *core.T) {
	// A placeholder present in the body but NOT declared as an InputVar is an
	// unknown-placeholder error — the body and the declaration disagree.
	tpl := Template{Body: "Hi {{name}} from {{rogue}}", InputVars: []string{"name"}}
	out, err := tpl.Render(map[string]string{"name": "Nick", "rogue": "x"})
	core.AssertError(t, err, "rogue")
	core.AssertEqual(t, "", out)
	core.AssertEqual(t, "prompt", core.Operation(err))

	// An undeclared placeholder is caught even when every declared var is given
	// and the undeclared one happens to be absent from vars too.
	tpl2 := Template{Body: "{{declared}} and {{undeclared}}", InputVars: []string{"declared"}}
	_, err = tpl2.Render(map[string]string{"declared": "ok"})
	core.AssertError(t, err, "undeclared")

	// A lone '{{' with no closing braces is literal text, not a placeholder —
	// it neither substitutes nor errors.
	lone := Template{Body: "use {{ like this"}
	out, err = lone.Render(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "use {{ like this", out)

	// An empty placeholder name {{}} is treated as literal text, never a var.
	emptyName := Template{Body: "a {{}} b"}
	out, err = emptyName.Render(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "a {{}} b", out)
}

// --- Builder -----------------------------------------------------------------

func TestPrompt_Builder_Good(t *core.T) {
	// A Builder assembles a multi-turn template; Build() joins the turns into a
	// single Body and carries the declared input variables.
	//
	//	tpl := prompt.NewBuilder().
	//	    System("You are {{persona}}.").
	//	    User("Help with {{topic}}.").
	//	    InputVariables("persona", "topic").
	//	    Build()
	tpl := NewBuilder().
		System("You are {{persona}}.").
		User("Help me with {{topic}}.").
		Assistant("Sure.").
		InputVariables("persona", "topic").
		Build()
	core.AssertEqual(t, []string{"persona", "topic"}, tpl.InputVars)
	core.AssertContains(t, tpl.Body, "You are {{persona}}.")
	core.AssertContains(t, tpl.Body, "Help me with {{topic}}.")

	// BuildMessages renders each turn's placeholders against vars and returns
	// the typed message list in turn order.
	msgs, err := NewBuilder().
		System("You are {{persona}}.").
		User("Help me with {{topic}}.").
		InputVariables("persona", "topic").
		BuildMessages(map[string]string{"persona": "a coder", "topic": "Go"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, len(msgs))
	core.AssertEqual(t, chat.System, msgs[0].Role)
	core.AssertEqual(t, "You are a coder.", msgs[0].Text())
	core.AssertEqual(t, chat.User, msgs[1].Role)
	core.AssertEqual(t, "Help me with Go.", msgs[1].Text())

	// Each rendered turn carries its body as a single text content block.
	core.AssertEqual(t, 1, len(msgs[0].Content))
	core.AssertEqual(t, chat.KindText, msgs[0].Content[0].Kind)

	// A builder with no input variables and no placeholders builds clean turns.
	plain, err := NewBuilder().User("just text").BuildMessages(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, len(plain))
	core.AssertEqual(t, "just text", plain[0].Text())
}

func TestPrompt_Builder_Bad(t *core.T) {
	// BuildMessages fails when a declared input variable is missing for a turn.
	_, err := NewBuilder().
		User("Help with {{topic}}.").
		InputVariables("topic").
		BuildMessages(map[string]string{})
	core.AssertError(t, err, "topic")
	core.AssertEqual(t, "prompt", core.Operation(err))

	// The error surfaces even if an earlier turn rendered cleanly.
	_, err = NewBuilder().
		System("static system turn").
		User("Help with {{topic}}.").
		InputVariables("topic").
		BuildMessages(nil)
	core.AssertError(t, err, "topic")
}

func TestPrompt_Builder_Ugly(t *core.T) {
	// A turn that uses an undeclared placeholder is an unknown-placeholder error
	// at BuildMessages time (the per-turn Render enforces the declaration).
	_, err := NewBuilder().
		User("Help with {{topic}} and {{rogue}}.").
		InputVariables("topic").
		BuildMessages(map[string]string{"topic": "Go", "rogue": "x"})
	core.AssertError(t, err, "rogue")

	// An empty builder builds an empty template and an empty message list.
	tpl := NewBuilder().Build()
	core.AssertEqual(t, "", tpl.Body)
	core.AssertEqual(t, 0, len(tpl.InputVars))
	msgs, err := NewBuilder().BuildMessages(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 0, len(msgs))

	// InputVariables called twice replaces the set rather than appending, and
	// late calls win — the last declaration is the contract.
	tpl = NewBuilder().
		User("{{a}}").
		InputVariables("wrong").
		InputVariables("a").
		Build()
	core.AssertEqual(t, []string{"a"}, tpl.InputVars)
	out, err := tpl.Render(map[string]string{"a": "ok"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "ok", out)
}

// --- Store -------------------------------------------------------------------

func TestPrompt_Store_Good(t *core.T) {
	// Put auto-assigns version 1 for a fresh id, then Get / Latest / List
	// resolve it.
	//
	//	s := prompt.NewMemoryStore()
	//	stored, _ := s.Put(prompt.Template{ID: "greet", Body: "hi"})  // version 1
	//	got, _ := s.Get("greet", 1)
	s := NewMemoryStore()

	v1, err := s.Put(Template{ID: "greet", Body: "hello {{name}}", InputVars: []string{"name"}})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, v1.Version, "first Put auto-assigns version 1")

	// A second Put for the same id auto-assigns the next version.
	v2, err := s.Put(Template{ID: "greet", Body: "hi {{name}}", InputVars: []string{"name"}})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, v2.Version, "second Put auto-assigns version 2")

	// Get resolves an explicit version.
	got, err := s.Get("greet", 1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "hello {{name}}", got.Body)

	// Latest returns the highest version.
	latest, err := s.Latest("greet")
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, latest.Version)
	core.AssertEqual(t, "hi {{name}}", latest.Body)

	// List returns every version for the id in ascending version order.
	all, err := s.List("greet")
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, len(all))
	core.AssertEqual(t, 1, all[0].Version)
	core.AssertEqual(t, 2, all[1].Version)

	// A caller-set explicit version is honoured rather than auto-assigned, and
	// the next auto-assignment continues above the highest seen.
	pinned, err := s.Put(Template{ID: "greet", Version: 10, Body: "pinned"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 10, pinned.Version)
	next, err := s.Put(Template{ID: "greet", Body: "after pin"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 11, next.Version, "auto-assign continues above the highest version")

	// A second id is independent and starts at version 1.
	other, err := s.Put(Template{ID: "farewell", Body: "bye"})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, other.Version)
}

func TestPrompt_Store_Bad(t *core.T) {
	s := NewMemoryStore()

	// Get / Latest / List on an unknown id are typed errors.
	_, err := s.Get("missing", 1)
	core.AssertError(t, err, "missing")
	core.AssertEqual(t, "prompt", core.Operation(err))

	_, err = s.Latest("missing")
	core.AssertError(t, err, "missing")

	_, err = s.List("missing")
	core.AssertError(t, err, "missing")

	// Get for a known id but an unknown version is an error.
	_, err = s.Put(Template{ID: "greet", Body: "hi"})
	core.AssertNoError(t, err)
	_, err = s.Get("greet", 99)
	core.AssertError(t, err, "99")

	// Put with an empty id is rejected — an id is the storage key.
	_, err = s.Put(Template{Body: "no id"})
	core.AssertError(t, err, "id")
}

func TestPrompt_Store_Ugly(t *core.T) {
	// Re-Putting an already-used explicit version overwrites that version in
	// place rather than creating a duplicate, and List stays sorted and unique.
	s := NewMemoryStore()
	_, err := s.Put(Template{ID: "greet", Version: 1, Body: "first"})
	core.AssertNoError(t, err)
	_, err = s.Put(Template{ID: "greet", Version: 1, Body: "second"})
	core.AssertNoError(t, err)
	got, err := s.Get("greet", 1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "second", got.Body, "explicit re-Put overwrites the version")
	all, err := s.List("greet")
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, len(all), "overwrite does not duplicate the version")

	// The store is goroutine-safe: concurrent Puts and reads to one id do not
	// race and every version lands.
	conc := NewMemoryStore()
	const n = 50
	done := make(chan struct{})
	for range n {
		go func() {
			_, _ = conc.Put(Template{ID: "hot", Body: "x"})
			_, _ = conc.List("hot")
			_, _ = conc.Latest("hot")
			done <- struct{}{}
		}()
	}
	for range n {
		<-done
	}
	all, err = conc.List("hot")
	core.AssertNoError(t, err)
	core.AssertEqual(t, n, len(all), "every concurrent Put is stored exactly once")
	latest, err := conc.Latest("hot")
	core.AssertNoError(t, err)
	core.AssertEqual(t, n, latest.Version, "the highest auto-assigned version is the latest")

	// Get / Latest return copies — mutating a returned template's slice must not
	// corrupt the stored entry.
	iso := NewMemoryStore()
	_, err = iso.Put(Template{ID: "greet", Body: "hi {{name}}", InputVars: []string{"name"}})
	core.AssertNoError(t, err)
	got, err = iso.Get("greet", 1)
	core.AssertNoError(t, err)
	if len(got.InputVars) > 0 {
		got.InputVars[0] = "tampered"
	}
	again, err := iso.Get("greet", 1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "name", again.InputVars[0], "stored InputVars are not aliased to the returned copy")
}
