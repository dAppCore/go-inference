// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferstate "dappco.re/go/inference/state"
)

func ExampleBookStateContextAssembler() {
	assembler := BookStateContextAssembler{State: BookState{
		Title:   "Meditations",
		Excerpt: "From my grandfather Verus I learned good morals.",
	}}
	contextResult := assembler.AssembleContext(context.Background(), nil)
	contextText := contextResult.Value.(string)

	core.Println(core.Contains(contextText, "grandfather Verus"))
	// Output:
	// true
}

func ExampleBookStateFromWakeResult() {
	state := BookStateFromWakeResult(inferstate.WakeResult{
		Entry:        inferstate.Ref{URI: "memvid://entry", Title: "Meditations"},
		PrefixTokens: 1448,
	})

	core.Println(state.Title)
	core.Println(state.PrefixTokens)
	// Output:
	// Meditations
	// 1448
}

func ExampleBookStateFromRef() {
	state := BookStateFromRef(inferstate.Ref{
		URI:        "memvid://entry",
		BundleURI:  "memvid://bundle",
		Title:      "Meditations",
		TokenCount: 1448,
	})

	core.Println(state.EntryURI)
	core.Println(state.BundleURI)
	// Output:
	// memvid://entry
	// memvid://bundle
}

func ExampleNewBookStateDemo() {
	result := NewBookStateDemo(BookStateDemoConfig{
		State: BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{
			Name:    "teacher",
			ModelID: "teacher",
			Model:   &routerFakeModel{modelType: "teacher", output: "answer"},
		}},
	})

	core.Println(result.OK)
	// Output:
	// true
}

func ExampleBookStateDemo_Ask() {
	result := NewBookStateDemo(BookStateDemoConfig{
		State: BookState{Title: "Meditations", Excerpt: "gentleness and meekness"},
		TeacherRoutes: []ProviderRoute{{
			Name:    "teacher",
			ModelID: "teacher",
			Model:   &routerFakeModel{modelType: "teacher", output: "gentleness"},
		}},
	})
	demo := result.Value.(*BookStateDemo)
	answerResult := demo.Ask(context.Background(), BookStateAskRequest{Question: "What lesson?"})
	response := answerResult.Value.(BookStateAskResponse)

	core.Println(response.TeacherAnswer)
	// Output:
	// gentleness
}

func ExampleBookStateDemo_State() {
	result := NewBookStateDemo(BookStateDemoConfig{
		State: BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{
			Name:    "teacher",
			ModelID: "teacher",
			Model:   &routerFakeModel{modelType: "teacher", output: "answer"},
		}},
	})
	demo := result.Value.(*BookStateDemo)

	core.Println(demo.State().Title)
	// Output:
	// Meditations
}

func ExampleBookStateDemoConfig() {
	cfg := BookStateDemoConfig{
		State: BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{
			Name:    "teacher",
			ModelID: "teacher",
			Model:   &routerFakeModel{modelType: "teacher", output: "answer"},
		}},
	}

	core.Println(cfg.State.Title)
	// Output:
	// Meditations
}

func ExampleBookStateAskRequest() {
	request := BookStateAskRequest{Question: "What lesson?", MaxTokens: 64}

	core.Println(request.MaxTokens)
	// Output:
	// 64
}

func ExampleBookStateAskResponse() {
	response := BookStateAskResponse{
		Question:      "What lesson?",
		TeacherAnswer: "gentleness",
	}

	core.Println(response.TeacherAnswer)
	// Output:
	// gentleness
}

func ExampleBookState() {
	state := BookState{Title: "Meditations", EntryURI: "memvid://aurelius"}

	core.Println(state.EntryURI)
	// Output:
	// memvid://aurelius
}

func ExampleBookStateContextAssembler_AssembleContext() {
	assembler := BookStateContextAssembler{State: BookState{Title: "Meditations"}}
	contextResult := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "hello"}})
	contextText := contextResult.Value.(string)

	core.Println(contextText)
	// Output:
	// title: Meditations
}
