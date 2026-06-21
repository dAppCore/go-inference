// SPDX-Licence-Identifier: EUPL-1.2

package modality

import core "dappco.re/go"

// --- Requested: validate / normalise a requested output set (§6.12) ---

func TestModality_Requested_Good(t *core.T) {
	// A clean, in-order set passes through unchanged.
	got, err := Requested([]Modality{Text, Image, Audio})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 3, len(got), "all three kept")
	core.AssertEqual(t, Text, got[0])
	core.AssertEqual(t, Image, got[1])
	core.AssertEqual(t, Audio, got[2])

	// Duplicates collapse, first-seen order preserved.
	got, err = Requested([]Modality{Image, Image, Text, Image})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, len(got), "image deduped, text kept")
	core.AssertEqual(t, Image, got[0], "first-seen order preserved")
	core.AssertEqual(t, Text, got[1])

	// ParseModality round-trips the wire strings (case-insensitive).
	m, err := ParseModality("AUDIO")
	core.AssertNoError(t, err)
	core.AssertEqual(t, Audio, m)
	core.AssertEqual(t, "audio", m.String())
}

func TestModality_Requested_Bad(t *core.T) {
	// An unknown modality is rejected, not silently dropped. The trailing
	// arg is matched as a substring of err.Error() (core.AssertError).
	_, err := Requested([]Modality{Text, Modality("video")})
	core.AssertError(t, err, "unknown modality")

	// ParseModality rejects junk.
	_, err = ParseModality("hologram")
	core.AssertError(t, err, "unknown modality")

	// The zero value is not a valid modality.
	core.AssertFalse(t, Modality("").Valid(), "empty modality is invalid")
}

func TestModality_Requested_Ugly(t *core.T) {
	// Empty / nil request implies text — never an empty output set.
	got, err := Requested(nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, len(got), "empty implies text")
	core.AssertEqual(t, Text, got[0])

	got, err = Requested([]Modality{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, len(got))
	core.AssertEqual(t, Text, got[0])

	// All-duplicate-text collapses to a single text, still valid.
	got, err = Requested([]Modality{Text, Text, Text})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, len(got))
	core.AssertEqual(t, Text, got[0])

	// Surrounding whitespace and mixed case still parse (tolerant wire input).
	got, err = Requested([]Modality{Modality(" Image "), Modality("text")})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, len(got))
	core.AssertEqual(t, Image, got[0], "whitespace/case normalised")
	core.AssertEqual(t, Text, got[1])
}

// --- Parts: build assistant output parts and assemble them (§6.1) ---

func TestModality_Parts_Good(t *core.T) {
	txt := TextPart("hello")
	core.AssertEqual(t, KindText, txt.Kind)
	core.AssertEqual(t, "hello", txt.Text)

	img := ImagePart([]byte{0x89, 0x50}, "image/png")
	core.AssertEqual(t, KindImage, img.Kind)
	core.AssertEqual(t, "image/png", img.MIME)
	core.AssertEqual(t, 2, len(img.Data))

	aud := AudioPart([]byte{0x01, 0x02, 0x03}, "audio/wav")
	core.AssertEqual(t, KindAudio, aud.Kind)
	core.AssertEqual(t, "audio/wav", aud.MIME)
	core.AssertEqual(t, 3, len(aud.Data))

	// Assemble: text concatenated, media retained in original order.
	msg := Assemble([]ContentPart{
		TextPart("one "),
		img,
		TextPart("two"),
		aud,
	})
	core.AssertEqual(t, RoleAssistant, msg.Role)
	core.AssertEqual(t, "one two", msg.Text, "text parts concatenated in order")
	core.AssertEqual(t, 4, len(msg.Parts), "every part retained")
	core.AssertEqual(t, KindImage, msg.Parts[1].Kind, "media order preserved")
	core.AssertEqual(t, KindAudio, msg.Parts[3].Kind)
}

func TestModality_Parts_Bad(t *core.T) {
	// An image part may carry a URL instead of inline data.
	img := ImageURLPart("https://cdn.example/x.png", "image/png")
	core.AssertEqual(t, KindImage, img.Kind)
	core.AssertEqual(t, "https://cdn.example/x.png", img.URL)
	core.AssertEqual(t, 0, len(img.Data), "URL part carries no inline data")

	// A part with neither data nor URL nor text is empty.
	core.AssertTrue(t, ContentPart{Kind: KindImage}.IsEmpty(), "no payload is empty")
	core.AssertFalse(t, img.IsEmpty(), "URL counts as payload")
	core.AssertFalse(t, TextPart("x").IsEmpty())

	// Assembling a URL image alongside text still concatenates the text and
	// keeps the media part.
	msg := Assemble([]ContentPart{TextPart("see: "), img})
	core.AssertEqual(t, "see: ", msg.Text)
	core.AssertEqual(t, 2, len(msg.Parts))
}

func TestModality_Parts_Ugly(t *core.T) {
	// Assembling nothing yields an empty assistant message, not a nil panic.
	msg := Assemble(nil)
	core.AssertEqual(t, RoleAssistant, msg.Role)
	core.AssertEqual(t, "", msg.Text)
	core.AssertEqual(t, 0, len(msg.Parts))

	// A run of only text parts collapses to one text body, parts still kept.
	msg = Assemble([]ContentPart{TextPart("a"), TextPart("b"), TextPart("c")})
	core.AssertEqual(t, "abc", msg.Text)
	core.AssertEqual(t, 3, len(msg.Parts))

	// Empty-payload parts are tolerated and retained (an empty text adds nothing
	// to the body but stays in the part list for fidelity).
	msg = Assemble([]ContentPart{TextPart(""), TextPart("body")})
	core.AssertEqual(t, "body", msg.Text)
	core.AssertEqual(t, 2, len(msg.Parts))
}

// --- Counts: tally output parts for usage accounting (§6.6) ---

func TestModality_Counts_Good(t *core.T) {
	parts := []ContentPart{
		TextPart("hello"),
		ImagePart([]byte{1, 2}, "image/png"),
		AudioPart([]byte{3, 4, 5}, "audio/wav"),
		ImagePart([]byte{6}, "image/jpeg"),
	}
	c := Counts(parts)
	core.AssertEqual(t, 1, c.TextParts)
	core.AssertEqual(t, 2, c.ImageParts)
	core.AssertEqual(t, 1, c.AudioParts)
}

func TestModality_Counts_Bad(t *core.T) {
	// No media: only a text tally, zero image/audio.
	c := Counts([]ContentPart{TextPart("just text")})
	core.AssertEqual(t, 1, c.TextParts)
	core.AssertEqual(t, 0, c.ImageParts)
	core.AssertEqual(t, 0, c.AudioParts)
	core.AssertEqual(t, 0, c.ImageTokens)
	core.AssertEqual(t, 0, c.AudioTokens)

	// Nil input tallies to a zero value, no panic.
	c = Counts(nil)
	core.AssertEqual(t, 0, c.TextParts)
	core.AssertEqual(t, 0, c.ImageParts)
	core.AssertEqual(t, 0, c.AudioParts)
}

func TestModality_Counts_Ugly(t *core.T) {
	// Per-part token/unit counts (set by the backend) sum into the totals.
	img := ImagePart([]byte{1}, "image/png")
	img.Tokens = 258 // e.g. provider-reported image tokens
	aud := AudioPart([]byte{2}, "audio/wav")
	aud.Tokens = 1200
	txt := TextPart("x")
	txt.Tokens = 4

	c := Counts([]ContentPart{txt, img, aud, img})
	core.AssertEqual(t, 1, c.TextParts)
	core.AssertEqual(t, 2, c.ImageParts)
	core.AssertEqual(t, 1, c.AudioParts)
	core.AssertEqual(t, 4, c.TextTokens, "text tokens summed")
	core.AssertEqual(t, 516, c.ImageTokens, "image tokens summed across both image parts")
	core.AssertEqual(t, 1200, c.AudioTokens, "audio tokens summed")
}
