// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	_ "embed"
	"encoding/hex"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// tokenizer.go is the RWKV "World" tokenizer every RWKV-7 "Goose" checkpoint ships (hf_rwkv_tokenizer.py's
// RWKV_TOKENIZER / TRIE, vocab file rwkv_vocab_v20230424.txt) — a byte-level GREEDY LONGEST-MATCH trie,
// not BPE (no merge rules at all), so it needs none of go/decode/tokenizer's BPE/sentencepiece machinery.
// id 0 (<|rwkv_tokenizer_end_of_text|>, bos=eos=pad) is a special OUTSIDE the trie; ids 1..65529 are the
// trie's byte-string vocabulary.
//
// The upstream vocab file's middle field is a Python str/bytes repr() (single- or double-quoted, `\xHH`/
// `\n`/`\t`/`\\`/quote escapes, a `b'...'` prefix for the 486 entries that are not valid standalone UTF-8)
// — a grammar this port does not hand-parse. data/rwkv_vocab_v20230424.hex is a byte-exact
// re-serialisation instead: line N (1-based) is token id N's raw bytes, hex-encoded. It was produced ONCE,
// offline, by a short throwaway Python snippet run by hand against the upstream checkpoint's own
// rwkv_vocab_v20230424.txt (mirroring the same parse hf_rwkv_tokenizer.RWKV_TOKENIZER.__init__ does,
// literal-for-literal, since Python's repr() grammar has no other standard decoder). That snippet is NOT
// part of this repo, is never invoked by Go code, and runs over a static trusted local file — no data from
// an untrusted or runtime source ever reaches it. LoadWorldTokenizerHex below reads only the derived .hex
// fixture (plain hex-decode, no code execution) and never touches the original .txt or Python at all.
//
// The vocab table is a FIXED artefact shared by the whole World-tokenizer family (every released RWKV-7
// checkpoint — 0.1B/1.5B/2.9B/…), not per-checkpoint data, and a real HF snapshot directory ships only the
// unparseable rwkv_vocab_v20230424.txt (no tokenizer.json, no derived .hex) — so data/rwkv_vocab_v20230424.hex
// is embedded straight into the binary (NewWorldTokenizer) rather than expected to live in a checkpoint
// directory at load time. It lives outside testdata/ deliberately: this file is production data, not a
// test-only fixture (go:embed would still reach into testdata, but the path would misdescribe what ships
// in the binary — AX-3, path is documentation).

// trieNode is one node of the byte-trie: 256 possible next-byte children, and id (0 ⇒ "no vocabulary
// entry ends exactly here") when the byte string ending at this node is itself a complete token.
type trieNode struct {
	children [256]*trieNode
	id       int32
}

// worldVocabHex is the canonical World-tokenizer vocab table, embedded straight into the binary — see the
// package doc comment above for its provenance and why this is production data, not a test fixture.
//
//go:embed data/rwkv_vocab_v20230424.hex
var worldVocabHex string

// WorldTokenizer is the RWKV World byte-trie tokenizer: Encode greedily matches the longest vocabulary
// byte-string at each position (hf_rwkv_tokenizer.TRIE.find_longest); Decode concatenates each id's raw
// bytes and reads the result as UTF-8 (RWKV_TOKENIZER.decodeBytes). Implements the engine's serve-side
// TextTokenizer method set BY SHAPE (Encode/Decode/DecodeToken/DecodeOne/TokenID/EOS) without importing
// the engine package (AX-8: model/arch never imports engine) — see DecodeToken/DecodeOne/TokenID/EOS below.
type WorldTokenizer struct {
	root    *trieNode
	toBytes map[int32][]byte
	toID    map[string]int32
}

// NewWorldTokenizer builds the World tokenizer from its embedded canonical vocab — the zero-config
// constructor: a real checkpoint directory ships only the unparseable rwkv_vocab_v20230424.txt (see the
// package doc comment), so this never reads the checkpoint directory at all. Every released RWKV-7
// checkpoint shares this exact vocabulary.
func NewWorldTokenizer() (*WorldTokenizer, error) {
	return parseWorldTokenizerHex(worldVocabHex)
}

// LoadWorldTokenizerHex loads the tokenizer from an on-disk hex-per-line vocab fixture at path — for
// pointing at an alternate or future vocab revision; NewWorldTokenizer is the production entry point.
func LoadWorldTokenizerHex(path string) (*WorldTokenizer, error) {
	text, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("rwkv7.LoadWorldTokenizerHex", "read vocab", err)
	}
	return parseWorldTokenizerHex(text)
}

// parseWorldTokenizerHex parses the hex-per-line vocab format (line N, 1-based, is token id N's raw
// bytes, hex-encoded) shared by NewWorldTokenizer's embedded table and LoadWorldTokenizerHex's on-disk one.
func parseWorldTokenizerHex(text string) (*WorldTokenizer, error) {
	lines := core.Split(text, "\n")
	tok := &WorldTokenizer{root: &trieNode{}, toBytes: make(map[int32][]byte), toID: make(map[string]int32)}
	id := int32(0)
	for _, line := range lines {
		if line == "" {
			continue
		}
		id++
		b, herr := hex.DecodeString(line)
		if herr != nil {
			return nil, core.E("rwkv7.parseWorldTokenizerHex", core.Sprintf("decode line for id %d", id), herr)
		}
		tok.add(b, id)
	}
	if id == 0 {
		return nil, core.NewError("rwkv7.parseWorldTokenizerHex: empty vocab")
	}
	return tok, nil
}

func (t *WorldTokenizer) add(b []byte, id int32) {
	n := t.root
	for _, c := range b {
		if n.children[c] == nil {
			n.children[c] = &trieNode{}
		}
		n = n.children[c]
	}
	n.id = id
	t.toBytes[id] = append([]byte(nil), b...)
	t.toID[string(b)] = id
}

// Encode greedily matches the longest vocabulary byte-string at each position, exactly
// hf_rwkv_tokenizer.RWKV_TOKENIZER.encodeBytes (TRIE.find_longest): walk the trie as far as bytes allow,
// remembering the deepest node whose id is set, then emit that token and resume from there. text is
// consumed as raw UTF-8 bytes.
func (t *WorldTokenizer) Encode(text string) []int32 {
	src := []byte(text)
	var out []int32
	for i := 0; i < len(src); {
		n := t.root
		bestID := int32(0)
		bestLen := 0
		for j := i; j < len(src); j++ {
			c := n.children[src[j]]
			if c == nil {
				break
			}
			n = c
			if n.id != 0 {
				bestID = n.id
				bestLen = j - i + 1
			}
		}
		if bestLen == 0 {
			// The World vocab covers every single byte 0x00-0xFF as a length-1 token, so this should
			// not trigger on real input; stay well-defined (skip the byte) rather than loop forever.
			i++
			continue
		}
		out = append(out, bestID)
		i += bestLen
	}
	return out
}

// Decode concatenates each token id's raw bytes and reads the result as UTF-8 —
// hf_rwkv_tokenizer.RWKV_TOKENIZER.decodeBytes. An id with no vocabulary entry (out of range, or the
// reserved id 0) is skipped.
func (t *WorldTokenizer) Decode(ids []int32) string {
	var buf []byte
	for _, id := range ids {
		if b, ok := t.toBytes[id]; ok {
			buf = append(buf, b...)
		}
	}
	return string(buf)
}

// DecodeToken decodes a single token id to text — the engine's per-token STREAMING decode
// (engine.TextTokenizer). World is byte-level with no SentencePiece word-boundary marker to preserve, so
// this is Decode([]int32{id}) directly: id 0 and any out-of-range id decode to "" (matching Decode's skip
// rule), and a mid-multibyte-UTF-8 token's raw bytes pass through unchanged — valid once concatenated with
// its neighbours, the same streaming contract the GPT-2 byte-level path (decode/tokenizer.go) already
// relies on.
func (t *WorldTokenizer) DecodeToken(id int32) string {
	if b, ok := t.toBytes[id]; ok {
		return string(b)
	}
	return ""
}

// DecodeOne mirrors Decode([]int32{id}) — engine.TextTokenizer's label-decode entry. Byte-level World has
// no boundary space to strip, so it coincides exactly with DecodeToken.
func (t *WorldTokenizer) DecodeOne(id int32) string { return t.DecodeToken(id) }

// TokenID looks up a token's exact byte string in the vocabulary — the reverse of Decode/DecodeToken
// (engine.TextTokenizer; e.g. resolving a template's stop string to an id).
func (t *WorldTokenizer) TokenID(text string) (int32, bool) {
	id, ok := t.toID[text]
	return id, ok
}

// EOS returns the World tokenizer's single reserved terminator id: 0
// (<|rwkv_tokenizer_end_of_text|>, bos=eos=pad — outside the trie; see the package doc comment above).
func (t *WorldTokenizer) EOS() int32 { return 0 }
