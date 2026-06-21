// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferstate "dappco.re/go/inference/state"
)

func TestBookStateDemo_Ask_Good_TeacherUsesBookState(t *testing.T) {
	student := &routerFakeModel{modelType: "student", output: "Verus taught discipline."}
	teacher := &routerFakeModel{modelType: "teacher", output: "The book says gentleness and meekness."}
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State: BookState{
			Title:        "Meditations",
			Excerpt:      "From my grandfather Verus I learned good morals and the government of my temper.",
			EntryURI:     "mlx://aurelius/full-book/chapter-001",
			PrefixTokens: 1448,
		},
		StudentRoutes: []ProviderRoute{{Name: "student", ModelID: "student-small", Model: student}},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher-state", Model: teacher}},
	})

	result := demo.Ask(context.Background(), BookStateAskRequest{
		Question:  "What did Marcus learn from Verus?",
		MaxTokens: 24,
	})

	if !result.OK {
		t.Fatalf("Ask() error = %s", result.Error())
	}
	response := result.Value.(BookStateAskResponse)
	if response.StudentAnswer != "Verus taught discipline." || response.TeacherAnswer != "The book says gentleness and meekness." {
		t.Fatalf("Ask() = %+v, want student and teacher outputs", response)
	}
	if response.State.Title != "Meditations" || response.State.PrefixTokens != 1448 {
		t.Fatalf("State = %+v, want book state metadata", response.State)
	}
	if len(student.lastMessages) != 1 || core.Contains(student.lastMessages[0].Content, "grandfather Verus") {
		t.Fatalf("student messages = %+v, want unaided student question", student.lastMessages)
	}
	if len(teacher.lastMessages) < 2 || !core.Contains(teacher.lastMessages[0].Content, "grandfather Verus") {
		t.Fatalf("teacher messages = %+v, want book-state context", teacher.lastMessages)
	}
	if !core.Contains(teacher.lastMessages[len(teacher.lastMessages)-1].Content, "Student answer") {
		t.Fatalf("teacher prompt = %+v, want student answer included", teacher.lastMessages)
	}
	if response.Student.ModelID != "student-small" || response.Teacher.ModelID != "teacher-state" {
		t.Fatalf("routes = %+v/%+v, want provider metadata", response.Student, response.Teacher)
	}
}

func TestBookStateDemo_Ask_Good_StudentCanUseBookState(t *testing.T) {
	student := &routerFakeModel{modelType: "student", output: "Gentleness."}
	teacher := &routerFakeModel{modelType: "teacher", output: "Correct."}
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:                BookState{Title: "Meditations", Excerpt: "gentleness and meekness"},
		StudentUsesBookState: true,
		StudentRoutes:        []ProviderRoute{{Name: "student", ModelID: "student", Model: student}},
		TeacherRoutes:        []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: teacher}},
	})

	result := demo.Ask(context.Background(), BookStateAskRequest{Question: "What lesson?", MaxTokens: 8})

	if !result.OK {
		t.Fatalf("Ask() error = %s", result.Error())
	}
	if len(student.lastMessages) < 2 || !core.Contains(student.lastMessages[0].Content, "gentleness and meekness") {
		t.Fatalf("student messages = %+v, want book-state context", student.lastMessages)
	}
}

func TestBookStateDemo_Ask_Bad_RejectsMissingQuestion(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{}}},
	})

	result := demo.Ask(context.Background(), BookStateAskRequest{})

	if result.OK {
		t.Fatal("Ask() OK = true, want missing question failure")
	}
	if !core.Contains(result.Error(), "question is required") {
		t.Fatalf("Ask() error = %q, want question validation", result.Error())
	}
}

func TestBookStateDemo_NewBookStateDemo_Ugly_RejectsMissingTeacher(t *testing.T) {
	result := NewBookStateDemo(BookStateDemoConfig{State: BookState{Title: "Meditations"}})

	if result.OK {
		t.Fatal("NewBookStateDemo() OK = true, want missing teacher failure")
	}
	if !core.Contains(result.Error(), "teacher route") {
		t.Fatalf("NewBookStateDemo() error = %q, want teacher route validation", result.Error())
	}
}

func TestBookStateContextAssembler_Good_FormatsState(t *testing.T) {
	assembler := BookStateContextAssembler{State: BookState{
		Title:        "Meditations",
		Excerpt:      "Verus taught gentleness.",
		EntryURI:     "mlx://entry",
		BundleURI:    "mlx://bundle",
		PrefixTokens: 12,
		Labels:       map[string]string{"source": "state"},
	}}

	result := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "question"}})

	if !result.OK {
		t.Fatalf("AssembleContext() error = %s", result.Error())
	}
	text, _ := result.Value.(string)
	for _, want := range []string{"Meditations", "Verus taught gentleness", "mlx://entry", "prefix_tokens: 12", "source=state"} {
		if !core.Contains(text, want) {
			t.Fatalf("AssembleContext() = %q, want %q", text, want)
		}
	}
}

func TestBookStateFromWakeResult_Good_CopiesInferenceStateMetadata(t *testing.T) {
	wake := inferstate.WakeResult{
		Entry:        inferstate.Ref{URI: "memvid://entry", Title: "Meditations", Labels: map[string]string{"chapter": "one"}},
		Bundle:       inferstate.StateRef{URI: "memvid://bundle"},
		Index:        inferstate.StateRef{URI: "memvid://index"},
		PrefixTokens: 1448,
		BundleTokens: 91732,
		BlockSize:    2048,
		BlocksRead:   45,
		Labels:       map[string]string{"source": "wake"},
	}

	state := BookStateFromWakeResult(wake)

	if state.Title != "Meditations" || state.EntryURI != "memvid://entry" || state.BundleURI != "memvid://bundle" || state.IndexURI != "memvid://index" {
		t.Fatalf("BookStateFromWakeResult() = %+v, want URIs and title copied", state)
	}
	if state.PrefixTokens != 1448 || state.BundleTokens != 91732 || state.BlockSize != 2048 || state.BlocksRead != 45 {
		t.Fatalf("BookStateFromWakeResult() = %+v, want state counters copied", state)
	}
	if state.Labels["source"] != "wake" || state.Labels["chapter"] != "one" {
		t.Fatalf("Labels = %+v, want wake and entry labels merged", state.Labels)
	}
}

func TestBookStateFromRef_Good_CopiesDurableRefMetadata(t *testing.T) {
	ref := inferstate.Ref{
		URI:        "memvid://entry",
		BundleURI:  "memvid://bundle",
		Title:      "Meditations",
		Kind:       "book",
		Hash:       "sha256:test",
		TokenStart: 10,
		TokenCount: 20,
		ByteStart:  30,
		ByteCount:  40,
		Labels:     map[string]string{"source": "ref"},
	}

	state := BookStateFromRef(ref)

	if state.EntryURI != "memvid://entry" || state.BundleURI != "memvid://bundle" || state.PrefixTokens != 20 {
		t.Fatalf("BookStateFromRef() = %+v, want ref URIs and token count", state)
	}
	for _, want := range []string{"book", "sha256:test", "10", "30", "40"} {
		found := false
		for _, value := range state.Metadata {
			if value == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("Metadata = %+v, want value %q", state.Metadata, want)
		}
	}
}

func TestBookStateDemo_BookStateFromWakeResult_Good(t *testing.T) {
	state := BookStateFromWakeResult(inferstate.WakeResult{
		Entry:        inferstate.Ref{URI: "memvid://entry", Title: "Meditations"},
		Bundle:       inferstate.StateRef{URI: "memvid://bundle"},
		PrefixTokens: 12,
	})

	if state.Title != "Meditations" || state.BundleURI != "memvid://bundle" || state.PrefixTokens != 12 {
		t.Fatalf("BookStateFromWakeResult() = %+v, want wake metadata", state)
	}
}

func TestBookStateDemo_BookStateFromWakeResult_Bad(t *testing.T) {
	state := BookStateFromWakeResult(inferstate.WakeResult{})

	if state.Title != "" || state.PrefixTokens != 0 || len(state.Labels) != 0 {
		t.Fatalf("BookStateFromWakeResult() = %+v, want empty state", state)
	}
}

func TestBookStateDemo_BookStateFromWakeResult_Ugly(t *testing.T) {
	state := BookStateFromWakeResult(inferstate.WakeResult{
		Entry:  inferstate.Ref{Labels: map[string]string{"entry": "yes"}},
		Labels: map[string]string{"wake": "yes"},
	})

	if state.Labels["entry"] != "yes" || state.Labels["wake"] != "yes" {
		t.Fatalf("BookStateFromWakeResult() labels = %+v, want merged labels", state.Labels)
	}
}

func TestBookStateDemo_BookStateFromRef_Good(t *testing.T) {
	state := BookStateFromRef(inferstate.Ref{URI: "memvid://entry", BundleURI: "memvid://bundle", TokenCount: 20})

	if state.EntryURI != "memvid://entry" || state.BundleURI != "memvid://bundle" || state.PrefixTokens != 20 {
		t.Fatalf("BookStateFromRef() = %+v, want ref metadata", state)
	}
}

func TestBookStateDemo_BookStateFromRef_Bad(t *testing.T) {
	state := BookStateFromRef(inferstate.Ref{})

	if state.EntryURI != "" || state.PrefixTokens != 0 || len(state.Metadata) != 0 {
		t.Fatalf("BookStateFromRef() = %+v, want empty state", state)
	}
}

func TestBookStateDemo_BookStateFromRef_Ugly(t *testing.T) {
	state := BookStateFromRef(inferstate.Ref{Kind: "book", Hash: "sha256:test", TokenStart: 3, ByteStart: 4, ByteCount: 5})

	for _, want := range []string{"book", "sha256:test", "3", "4", "5"} {
		found := false
		for _, value := range state.Metadata {
			if value == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("BookStateFromRef() metadata = %+v, want %q", state.Metadata, want)
		}
	}
}

func TestBookStateDemo_BookStateContextAssembler_AssembleContext_Good(t *testing.T) {
	assembler := BookStateContextAssembler{State: BookState{Title: "Meditations", Excerpt: "gentleness"}}
	result := assembler.AssembleContext(context.Background(), nil)

	if !result.OK || !core.Contains(result.Value.(string), "gentleness") {
		t.Fatalf("BookStateContextAssembler.AssembleContext() = %#v, want context", result)
	}
}

func TestBookStateDemo_BookStateContextAssembler_AssembleContext_Bad(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	assembler := BookStateContextAssembler{State: BookState{Title: "Meditations"}}
	result := assembler.AssembleContext(ctx, nil)

	if result.OK {
		t.Fatalf("BookStateContextAssembler.AssembleContext() = %#v, want cancelled context failure", result)
	}
}

func TestBookStateDemo_BookStateContextAssembler_AssembleContext_Ugly(t *testing.T) {
	assembler := BookStateContextAssembler{State: BookState{}}
	result := assembler.AssembleContext(context.Background(), nil)

	if !result.OK || result.Value.(string) != "" {
		t.Fatalf("BookStateContextAssembler.AssembleContext() = %#v, want empty context", result)
	}
}

func TestBookStateDemo_NewBookStateDemo_Good(t *testing.T) {
	result := NewBookStateDemo(BookStateDemoConfig{
		State:         BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{modelType: "teacher", output: "ok"}}},
	})

	if !result.OK || result.Value.(*BookStateDemo).State().Title != "Meditations" {
		t.Fatalf("NewBookStateDemo() = %#v, want configured demo", result)
	}
}

func TestBookStateDemo_NewBookStateDemo_Bad(t *testing.T) {
	result := NewBookStateDemo(BookStateDemoConfig{})

	if result.OK || !core.Contains(result.Error(), "teacher route") {
		t.Fatalf("NewBookStateDemo() = %#v, want missing teacher failure", result)
	}
}

func TestBookStateDemo_NewBookStateDemo_Ugly(t *testing.T) {
	result := NewBookStateDemo(BookStateDemoConfig{
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{}}},
		StudentRoutes: []ProviderRoute{{Name: "student"}},
	})

	if result.OK || !core.Contains(result.Error(), "student") {
		t.Fatalf("NewBookStateDemo() = %#v, want invalid student route failure", result)
	}
}

func TestBookStateDemo_BookStateDemo_State_Good(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{}}},
	})

	if state := demo.State(); state.Title != "Meditations" {
		t.Fatalf("BookStateDemo.State() = %+v, want title", state)
	}
}

func TestBookStateDemo_BookStateDemo_State_Bad(t *testing.T) {
	var demo *BookStateDemo

	if state := demo.State(); state.Title != "" || state.EntryURI != "" {
		t.Fatalf("BookStateDemo.State() = %+v, want zero state", state)
	}
}

func TestBookStateDemo_BookStateDemo_State_Ugly(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Labels: map[string]string{"source": "original"}},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{}}},
	})
	state := demo.State()
	state.Labels["source"] = "mutated"

	if again := demo.State(); again.Labels["source"] != "original" {
		t.Fatalf("BookStateDemo.State() leaked labels = %+v", again.Labels)
	}
}

func TestBookStateDemo_BookStateDemo_Ask_Good(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations", Excerpt: "gentleness"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{output: "answer"}}},
	})
	result := demo.Ask(context.Background(), BookStateAskRequest{Question: "What lesson?"})

	if !result.OK || result.Value.(BookStateAskResponse).TeacherAnswer != "answer" {
		t.Fatalf("BookStateDemo.Ask() = %#v, want teacher answer", result)
	}
}

func TestBookStateDemo_BookStateDemo_Ask_Bad(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{}}},
	})
	result := demo.Ask(context.Background(), BookStateAskRequest{})

	if result.OK || !core.Contains(result.Error(), "question") {
		t.Fatalf("BookStateDemo.Ask() = %#v, want missing question failure", result)
	}
}

func TestBookStateDemo_BookStateDemo_Ask_Ugly(t *testing.T) {
	var demo *BookStateDemo
	result := demo.Ask(context.Background(), BookStateAskRequest{Question: "What lesson?"})

	if result.OK || !core.Contains(result.Error(), "demo is nil") {
		t.Fatalf("BookStateDemo.Ask() = %#v, want nil demo failure", result)
	}
}

func mustBookStateDemo(t *testing.T, cfg BookStateDemoConfig) *BookStateDemo {
	t.Helper()
	result := NewBookStateDemo(cfg)
	if !result.OK {
		t.Fatalf("NewBookStateDemo() error = %s", result.Error())
	}
	return result.Value.(*BookStateDemo)
}
