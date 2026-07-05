// SPDX-Licence-Identifier: EUPL-1.2

// Allocation benchmarks for the canonical chat shape (RFC.md §6.1, §6.12).
// These types are constructed and mapped per request across the serving
// surface, so their steady-state allocation profile is on the hot path:
// every request builds Messages + ContentBlocks, every route reads
// PrimaryModel / FallbackChain, every request is Validate'd, and every
// turn may be flattened via Message.Text. One benchmark per public
// function; realistic multi-block messages and multi-model requests.
//
// Run: go test -bench=. -benchmem -run='^$' ./chat/
package chat_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/serving/chat"
)

// Sinks defeat compiler dead-code elimination — every benchmarked call
// writes its result to a package-level sink of the matching type.
var (
	sinkString string
	sinkBool   bool
	sinkRole   chat.Role
	sinkKind   chat.Kind
	sinkErr    error
	sinkBlock  chat.ContentBlock
	sinkMsg    chat.Message
	sinkChain  []string
)

// --- fixtures: built once, outside the measured loop ---

// benchPNG / benchWAV — small inline media payloads, as a multimodal
// request actually carries (a thumbnail, a short clip).
var (
	benchPNG = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	benchWAV = []byte{0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00}
	benchPDF = []byte{0x25, 0x50, 0x44, 0x46, 0x2D, 0x31, 0x2E, 0x37}
)

// benchMultiBlockMsg — a realistic multimodal user turn: several text
// blocks interleaved with an image and an audio clip. Message.Text must
// walk all blocks, concatenating only the text ones.
func benchMultiBlockMsg() chat.Message {
	return chat.Message{
		Role: chat.User,
		Content: []chat.ContentBlock{
			chat.Text("Here is the chart "),
			chat.Image(benchPNG, "image/png"),
			chat.Text(" and the voice note "),
			chat.Audio(benchWAV, "audio/wav"),
			chat.Text(" please summarise both."),
		},
	}
}

// benchSingleTextMsg — the overwhelmingly common case: one text block.
func benchSingleTextMsg() chat.Message {
	return chat.Message{
		Role:    chat.User,
		Content: []chat.ContentBlock{chat.Text("what is the capital of France?")},
	}
}

// benchMediaOnlyMsg — a turn with no text blocks at all.
func benchMediaOnlyMsg() chat.Message {
	return chat.Message{
		Role: chat.Assistant,
		Content: []chat.ContentBlock{
			chat.Image(benchPNG, "image/png"),
			chat.Audio(benchWAV, "audio/wav"),
		},
	}
}

// benchRequest — a realistic multi-role transcript with a fallback chain,
// the shape Validate / FallbackChain / PrimaryModel see per request.
func benchRequest() chat.Request {
	return chat.Request{
		Model:  "local-metal/gemma-4-31b",
		Models: []string{"local-metal/gemma-4-31b", "nim/qwen-3-32b", "openrouter/claude"},
		Messages: []chat.Message{
			{Role: chat.System, Content: []chat.ContentBlock{chat.Text("You are a helpful assistant. Use UK English.")}},
			{Role: chat.Developer, Content: []chat.ContentBlock{chat.Text("Prefer concise answers.")}},
			{Role: chat.User, Content: []chat.ContentBlock{chat.Text("What's the weather in London?")}},
			{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("Let me check that for you.")}},
			{Role: chat.Tool, ToolCallID: "call_weather_1", Content: []chat.ContentBlock{chat.Text("18C, light rain")}},
		},
	}
}

// --- Role (§6.1) ---

func BenchmarkChat_ParseRole_Clean(b *core.B) {
	// Fast path: already lower-case, no surrounding whitespace.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRole, sinkErr = chat.ParseRole("assistant")
	}
}

func BenchmarkChat_ParseRole_Messy(b *core.B) {
	// Raw request value: surrounding whitespace + upper-case.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRole, sinkErr = chat.ParseRole("  ASSISTANT ")
	}
}

func BenchmarkChat_Role_String(b *core.B) {
	r := chat.Assistant
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = r.String()
	}
}

func BenchmarkChat_Role_Valid(b *core.B) {
	r := chat.Assistant
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = r.Valid()
	}
}

// --- Kind (§6.1, §6.12) ---

func BenchmarkChat_Kind_String(b *core.B) {
	k := chat.KindImage
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = k.String()
	}
}

func BenchmarkChat_Kind_Valid(b *core.B) {
	k := chat.KindImage
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = k.Valid()
	}
}

// --- ContentBlock constructors (§6.1, §6.12) ---

func BenchmarkChat_Text(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = chat.Text("the answer is 42")
	}
}

func BenchmarkChat_Image(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = chat.Image(benchPNG, "image/png")
	}
}

func BenchmarkChat_ImageURL(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = chat.ImageURL("https://cdn.example/x.png", "image/png")
	}
}

func BenchmarkChat_Audio(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = chat.Audio(benchWAV, "audio/wav")
	}
}

func BenchmarkChat_File(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = chat.File(benchPDF, "report.pdf", "application/pdf")
	}
}

func BenchmarkChat_ContentBlock_Cached(b *core.B) {
	blk := chat.Text("system preamble")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = blk.Cached()
	}
}

func BenchmarkChat_ContentBlock_IsEmpty(b *core.B) {
	blk := chat.Text("populated")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = blk.IsEmpty()
	}
}

// --- Message (§6.1) ---

func BenchmarkChat_Message_Text_Single(b *core.B) {
	m := benchSingleTextMsg()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = m.Text()
	}
}

func BenchmarkChat_Message_Text_Multi(b *core.B) {
	m := benchMultiBlockMsg()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = m.Text()
	}
}

func BenchmarkChat_Message_Text_MediaOnly(b *core.B) {
	m := benchMediaOnlyMsg()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = m.Text()
	}
}

func BenchmarkChat_UserText(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsg = chat.UserText("what is 2+2?")
	}
}

// --- Request routing + guard (§6.1, §6.2) ---

func BenchmarkChat_PrimaryModel_ModelSet(b *core.B) {
	r := benchRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = r.PrimaryModel()
	}
}

func BenchmarkChat_PrimaryModel_ModelsOnly(b *core.B) {
	r := chat.Request{Models: []string{"x", "y", "z"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = r.PrimaryModel()
	}
}

func BenchmarkChat_FallbackChain(b *core.B) {
	r := benchRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkChain = r.FallbackChain()
	}
}

func BenchmarkChat_Validate(b *core.B) {
	r := benchRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = r.Validate()
	}
}
