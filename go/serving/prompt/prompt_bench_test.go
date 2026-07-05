// SPDX-Licence-Identifier: EUPL-1.2

package prompt

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// Package-level sinks defeat dead-code elimination — the compiler cannot prove
// the benchmarked results are unused, so it must keep the work.
var (
	sinkString  string
	sinkErr     error
	sinkStrs    []string
	sinkTpl     Template
	sinkTpls    []Template
	sinkMsgs    []chat.Message
	sinkBuilder *Builder
	sinkStore   *MemoryStore
)

// Realistic, shared fixtures: a multi-turn chat prompt with a handful of
// declared variables, one repeated placeholder, mirroring a per-request build.
const (
	benchBody = "You are {{persona}}. Help {{user}} with {{topic}}. " +
		"Remember: {{persona}} stays in character for {{user}} throughout."
)

var (
	benchInputVars = []string{"persona", "user", "topic"}
	benchVars      = map[string]string{
		"persona": "a helpful coder",
		"user":    "Nick",
		"topic":   "Go performance",
	}
)

// --- Render ------------------------------------------------------------------

func BenchmarkRender(b *core.B) {
	tpl := Template{ID: "chat", Version: 1, Body: benchBody, InputVars: benchInputVars}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString, sinkErr = tpl.Render(benchVars)
	}
}

// --- placeholders (hot unexported helper, called by Render and varsFor) ------

func BenchmarkPlaceholders(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkStrs = placeholders(benchBody)
	}
}

// --- Builder -----------------------------------------------------------------

func BenchmarkNewBuilder(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBuilder = NewBuilder()
	}
}

func BenchmarkBuilder_System(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBuilder = NewBuilder().System("You are {{persona}}.")
	}
}

func BenchmarkBuilder_User(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBuilder = NewBuilder().User("Help {{user}} with {{topic}}.")
	}
}

func BenchmarkBuilder_Assistant(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBuilder = NewBuilder().Assistant("Sure, happy to help.")
	}
}

func BenchmarkBuilder_InputVariables(b *core.B) {
	bld := NewBuilder()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBuilder = bld.InputVariables("persona", "user", "topic")
	}
}

func BenchmarkBuilder_Build(b *core.B) {
	bld := NewBuilder().
		System("You are {{persona}}.").
		User("Help {{user}} with {{topic}}.").
		Assistant("Sure.").
		InputVariables("persona", "user", "topic")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTpl = bld.Build()
	}
}

func BenchmarkBuilder_BuildMessages(b *core.B) {
	bld := NewBuilder().
		System("You are {{persona}}.").
		User("Help {{user}} with {{topic}}.").
		Assistant("Sure.").
		InputVariables("persona", "user", "topic")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsgs, sinkErr = bld.BuildMessages(benchVars)
	}
}

// --- Store -------------------------------------------------------------------

func BenchmarkNewMemoryStore(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkStore = NewMemoryStore()
	}
}

func BenchmarkMemoryStore_Put(b *core.B) {
	s := NewMemoryStore()
	// Explicit version overwrites in place, keeping the version map at steady
	// state so the bench measures Put's work, not unbounded map growth.
	tpl := Template{ID: "greet", Version: 1, Body: benchBody, InputVars: benchInputVars}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTpl, sinkErr = s.Put(tpl)
	}
}

func BenchmarkMemoryStore_Get(b *core.B) {
	s := NewMemoryStore()
	_, _ = s.Put(Template{ID: "greet", Body: benchBody, InputVars: benchInputVars})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTpl, sinkErr = s.Get("greet", 1)
	}
}

func BenchmarkMemoryStore_Latest(b *core.B) {
	s := NewMemoryStore()
	for v := 0; v < 5; v++ {
		_, _ = s.Put(Template{ID: "greet", Body: benchBody, InputVars: benchInputVars})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTpl, sinkErr = s.Latest("greet")
	}
}

func BenchmarkMemoryStore_List(b *core.B) {
	s := NewMemoryStore()
	for v := 0; v < 5; v++ {
		_, _ = s.Put(Template{ID: "greet", Body: benchBody, InputVars: benchInputVars})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTpls, sinkErr = s.List("greet")
	}
}
