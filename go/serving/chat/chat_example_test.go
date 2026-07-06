// SPDX-Licence-Identifier: EUPL-1.2

package chat

import core "dappco.re/go"

// ExampleParseRole reads a raw request value into a canonical Role.
func ExampleParseRole() {
	r, err := ParseRole(" ASSISTANT ")
	core.Println(r, err)
	// Output:
	// assistant <nil>
}

// ExampleRole_String prints the canonical wire form.
func ExampleRole_String() {
	core.Println(Assistant.String())
	// Output:
	// assistant
}

// ExampleRole_Valid reports whether a role is one of the canonical values.
func ExampleRole_Valid() {
	core.Println(User.Valid(), Role("robot").Valid())
	// Output:
	// true false
}

// ExampleKind_String prints the canonical wire form.
func ExampleKind_String() {
	core.Println(KindImage.String())
	// Output:
	// image
}

// ExampleKind_Valid reports whether a content kind is one of the known values.
func ExampleKind_Valid() {
	core.Println(KindVideo.Valid(), Kind("hologram").Valid())
	// Output:
	// true false
}

// ExampleText builds a text content block.
func ExampleText() {
	b := Text("the answer is 42")
	core.Println(b.Kind, b.Text)
	// Output:
	// text the answer is 42
}

// ExampleImage builds an image block from inline bytes.
func ExampleImage() {
	b := Image([]byte{0x89, 0x50}, "image/png")
	core.Println(b.Kind, b.MIME, len(b.Data))
	// Output:
	// image image/png 2
}

// ExampleImageURL builds an image block that references a URL.
func ExampleImageURL() {
	b := ImageURL("https://cdn.example/x.png", "image/png")
	core.Println(b.Kind, b.URL)
	// Output:
	// image https://cdn.example/x.png
}

// ExampleAudio builds an audio block from inline bytes.
func ExampleAudio() {
	b := Audio([]byte{0x01, 0x02, 0x03}, "audio/wav")
	core.Println(b.Kind, b.MIME, len(b.Data))
	// Output:
	// audio audio/wav 3
}

// ExampleFile builds a file-attachment block.
func ExampleFile() {
	b := File([]byte{0x25, 0x50, 0x44, 0x46}, "report.pdf", "application/pdf")
	core.Println(b.Kind, b.FileName, b.MIME)
	// Output:
	// file report.pdf application/pdf
}

// ExampleContentBlock_Cached marks a block as a cacheable prefix boundary.
func ExampleContentBlock_Cached() {
	preamble := Text("long system prompt").Cached()
	core.Println(preamble.CacheControl)
	// Output:
	// true
}

// ExampleContentBlock_IsEmpty tells a meaningful block from a placeholder.
func ExampleContentBlock_IsEmpty() {
	core.Println(ContentBlock{Kind: KindImage}.IsEmpty(), Text("x").IsEmpty())
	// Output:
	// true false
}

// ExampleMessage_Text flattens a multimodal turn to its text blocks.
func ExampleMessage_Text() {
	m := Message{Role: User, Content: []ContentBlock{
		Text("see "), Image([]byte{1}, "image/png"), Text("and done"),
	}}
	core.Println(m.Text())
	// Output:
	// see and done
}

// ExampleUserText builds the common single-text-message case.
func ExampleUserText() {
	m := UserText("what is 2+2?")
	core.Println(m.Role, m.Text())
	// Output:
	// user what is 2+2?
}

// ExampleRequest_PrimaryModel resolves the first model the router tries.
func ExampleRequest_PrimaryModel() {
	req := Request{Models: []string{"local-metal/gemma-4-31b", "nim/qwen"}}
	core.Println(req.PrimaryModel())
	// Output:
	// local-metal/gemma-4-31b
}

// ExampleRequest_FallbackChain builds the ordered, de-duplicated model list.
func ExampleRequest_FallbackChain() {
	req := Request{Model: "a", Models: []string{"a", "b"}}
	core.Println(req.FallbackChain())
	// Output:
	// [a b]
}

// ExampleRequest_Validate checks a request is well-formed before routing.
func ExampleRequest_Validate() {
	req := Request{Model: "gemma-4-e4b", Messages: []Message{UserText("hi")}}
	core.Println(req.Validate())
	// Output:
	// <nil>
}
