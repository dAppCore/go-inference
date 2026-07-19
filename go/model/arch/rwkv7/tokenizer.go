// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
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
// — a grammar this port does not hand-parse. testdata/rwkv_vocab_v20230424.hex is a byte-exact
// re-serialisation instead: line N (1-based) is token id N's raw bytes, hex-encoded. It was produced ONCE,
// offline, by a short throwaway Python snippet run by hand against the upstream checkpoint's own
// rwkv_vocab_v20230424.txt (mirroring the same parse hf_rwkv_tokenizer.RWKV_TOKENIZER.__init__ does,
// literal-for-literal, since Python's repr() grammar has no other standard decoder). That snippet is NOT
// part of this repo, is never invoked by Go code, and runs over a static trusted local file — no data from
// an untrusted or runtime source ever reaches it. LoadWorldTokenizerHex below reads only the derived .hex
// fixture (plain hex-decode, no code execution) and never touches the original .txt or Python at all.

// trieNode is one node of the byte-trie: 256 possible next-byte children, and id (0 ⇒ "no vocabulary
// entry ends exactly here") when the byte string ending at this node is itself a complete token.
type trieNode struct {
	children [256]*trieNode
	id       int32
}

// WorldTokenizer is the RWKV World byte-trie tokenizer: Encode greedily matches the longest vocabulary
// byte-string at each position (hf_rwkv_tokenizer.TRIE.find_longest); Decode concatenates each id's raw
// bytes and reads the result as UTF-8 (RWKV_TOKENIZER.decodeBytes).
type WorldTokenizer struct {
	root    *trieNode
	toBytes map[int32][]byte
}

// LoadWorldTokenizerHex loads the tokenizer from the hex-per-line derived vocab fixture at path (see the
// package doc comment above for its provenance).
func LoadWorldTokenizerHex(path string) (*WorldTokenizer, error) {
	text, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("rwkv7.LoadWorldTokenizerHex", "read vocab", err)
	}
	lines := core.Split(text, "\n")
	tok := &WorldTokenizer{root: &trieNode{}, toBytes: make(map[int32][]byte)}
	id := int32(0)
	for _, line := range lines {
		if line == "" {
			continue
		}
		id++
		b, herr := hex.DecodeString(line)
		if herr != nil {
			return nil, core.E("rwkv7.LoadWorldTokenizerHex", core.Sprintf("decode line for id %d", id), herr)
		}
		tok.add(b, id)
	}
	if id == 0 {
		return nil, core.NewError("rwkv7.LoadWorldTokenizerHex: empty vocab")
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
