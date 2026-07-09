// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import core "dappco.re/go"

func ExampleExactMatch() {
	ev := ExactMatch()
	ex := Example{ID: "ex-1", DatasetID: "ds", Reference: map[string]any{"answer": "context-dependent"}}
	key, score, _ := ev.Eval(ex, "context-dependent")
	core.Println(key, score)
	// Output: exact_match 1
}

func ExampleExactMatchOn() {
	ev := ExactMatchOn("gold")
	ex := Example{ID: "ex-1", DatasetID: "ds", Reference: map[string]any{"gold": "42"}}
	key, score, _ := ev.Eval(ex, "42")
	core.Println(key, score)
	// Output: exact_match 1
}

func ExampleContains() {
	ev := Contains()
	ex := Example{ID: "ex-1", DatasetID: "ds", Reference: map[string]any{"answer": "honest"}}
	key, score, _ := ev.Eval(ex, "always be honest")
	core.Println(key, score)
	// Output: contains 1
}

func ExampleContainsOn() {
	ev := ContainsOn("needle")
	ex := Example{ID: "ex-1", DatasetID: "ds", Reference: map[string]any{"needle": "cat"}}
	key, score, _ := ev.Eval(ex, "the cat sat")
	core.Println(key, score)
	// Output: contains 1
}

func ExampleRegexp() {
	r := Regexp(`\d+`)
	if !r.OK {
		return
	}
	ev := r.Value.(Evaluator)
	ex := Example{ID: "ex-1", DatasetID: "ds"}
	key, score, _ := ev.Eval(ex, "build 42 passed")
	core.Println(key, score)
	// Output: regexp 1
}

func ExampleLengthScore() {
	r := LengthScore(10)
	if !r.OK {
		return
	}
	ev := r.Value.(Evaluator)
	ex := Example{ID: "ex-1", DatasetID: "ds"}
	key, score, _ := ev.Eval(ex, "0123456789")
	core.Println(key, score)
	// Output: length 1
}
