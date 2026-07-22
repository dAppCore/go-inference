// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	core "dappco.re/go"
)

func TestContentHash_Good(t *core.T) {
	pair := core.JSONMarshal(PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	r := ContentHash(KindPair, pair)
	core.AssertTrue(t, r.OK)
	hash := r.Value.(string)
	core.AssertEqual(t, 64, len(hash), "a hex sha256 digest is 64 characters")

	// Deterministic: hashing the same semantic content twice, even via
	// two independently-marshalled byte slices, produces the same hash.
	pairAgain := core.JSONMarshal(PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	rAgain := ContentHash(KindPair, pairAgain)
	core.AssertTrue(t, rAgain.OK)
	core.AssertEqual(t, hash, rAgain.Value.(string), "identical semantic content must hash identically")

	messages := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}).Value.([]byte)
	r = ContentHash(KindMessages, messages)
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, 64, len(r.Value.(string)))

	r = ContentHash(KindTrace, []byte(`{"logits":[1,2,3]}`))
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, 64, len(r.Value.(string)))
}

func TestContentHash_Bad(t *core.T) {
	// Inequality: different content must (in practice, not by contract)
	// hash differently.
	a := ContentHash(KindPair, core.JSONMarshal(PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte))
	b := ContentHash(KindPair, core.JSONMarshal(PairContent{Prompt: "hi", Response: "goodbye"}).Value.([]byte))
	core.AssertTrue(t, a.OK && b.OK)
	core.AssertNotEqual(t, a.Value.(string), b.Value.(string), "different responses must hash differently")

	// Key order does not matter — the hash is over parsed semantic
	// fields, not raw bytes, so hand-written JSON with fields in the
	// opposite order still matches the struct-marshalled form.
	handWritten := ContentHash(KindPair, []byte(`{"response":"hello","prompt":"hi"}`))
	structMarshalled := ContentHash(KindPair, core.JSONMarshal(PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte))
	core.AssertTrue(t, handWritten.OK && structMarshalled.OK)
	core.AssertEqual(t, structMarshalled.Value.(string), handWritten.Value.(string), "key order must not affect the content hash")
}

func TestContentHash_Ugly(t *core.T) {
	r := ContentHash(KindPair, []byte("not json"))
	core.AssertFalse(t, r.OK, "malformed pair JSON must fail, not panic")

	r = ContentHash(KindMessages, []byte("not json"))
	core.AssertFalse(t, r.OK, "malformed messages JSON must fail, not panic")

	r = ContentHash(KindTrace, nil)
	core.AssertFalse(t, r.OK, "empty trace content must fail")

	r = ContentHash(ItemKind("bogus"), []byte(`{}`))
	core.AssertFalse(t, r.OK, "an unknown kind must fail")
}

func TestManifestHash_Good(t *core.T) {
	h := ManifestHash([]string{"aaa", "bbb", "ccc"})
	core.AssertEqual(t, 64, len(h))

	// Deterministic across repeated calls.
	core.AssertEqual(t, h, ManifestHash([]string{"aaa", "bbb", "ccc"}))
}

func TestManifestHash_Bad(t *core.T) {
	// Order matters — the manifest names exactly what a training run
	// saw, in the order it saw it.
	forward := ManifestHash([]string{"aaa", "bbb"})
	reversed := ManifestHash([]string{"bbb", "aaa"})
	core.AssertNotEqual(t, forward, reversed, "reordering the hashes must change the manifest hash")
}

func TestManifestHash_Ugly(t *core.T) {
	// An empty manifest is a defined edge case, not a panic — the
	// sha256 of the empty string.
	h := ManifestHash(nil)
	core.AssertEqual(t, core.SHA256HexString(""), h)
}
