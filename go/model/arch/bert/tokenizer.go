// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"unicode"

	core "dappco.re/go"
	"golang.org/x/text/unicode/norm"
)

// Special token surface strings — the WordPiece vocab lists them verbatim, so
// they resolve to their ids through the same map lookup as any other token.
const (
	tokenCLS = "[CLS]"
	tokenSEP = "[SEP]"
	tokenUNK = "[UNK]"
	tokenPAD = "[PAD]"

	// maxWordPieceChars caps a single word before it collapses to [UNK] — the
	// HF BertTokenizer default. Longer words are almost always URLs or hashes
	// that WordPiece would churn on with no useful sub-tokens.
	maxWordPieceChars = 100
)

// Tokenizer is a BERT WordPiece tokeniser built from a vocab.txt (one token per
// line, id = line index). It reproduces HF's BertTokenizer: a BasicTokenizer
// pass (clean, optional CJK spacing, optional lower-case + accent strip,
// punctuation split) followed by greedy longest-match WordPiece. Encode frames
// the result with [CLS] … [SEP], the shape a BertModel forward expects.
//
//	tk, err := bert.NewTokenizer(vocabBytes, true)
//	ids := tk.Encode("The quick brown fox") // [101 1996 4248 ... 102]
type Tokenizer struct {
	vocab       map[string]int32
	doLowerCase bool
	clsID       int32
	sepID       int32
	unkID       int32
	padID       int32
}

// NewTokenizer parses a vocab.txt blob into a Tokenizer. doLowerCase mirrors the
// snapshot's tokenizer_config.json (bge-small is lower-cased). The four special
// tokens must be present in the vocab or loading fails — a forward pass cannot
// frame a sequence without [CLS]/[SEP].
//
//	tk, err := bert.NewTokenizer(core.ReadFile(path).Bytes(), true)
func NewTokenizer(vocabTxt []byte, doLowerCase bool) (*Tokenizer, error) {
	lines := core.Split(core.AsString(vocabTxt), "\n")
	vocab := make(map[string]int32, len(lines))
	for i, line := range lines {
		// vocab.txt tokens carry no surrounding whitespace; a trailing "\r"
		// from CRLF files or a final empty line must not become a token.
		token := trimCarriageReturn(line)
		if token == "" {
			continue
		}
		if _, exists := vocab[token]; !exists {
			vocab[token] = int32(i)
		}
	}
	tk := &Tokenizer{vocab: vocab, doLowerCase: doLowerCase}
	for _, bind := range []struct {
		name string
		dst  *int32
	}{
		{tokenCLS, &tk.clsID}, {tokenSEP, &tk.sepID},
		{tokenUNK, &tk.unkID}, {tokenPAD, &tk.padID},
	} {
		id, ok := vocab[bind.name]
		if !ok {
			return nil, core.E("bert.NewTokenizer", "vocab is missing special token "+bind.name, nil)
		}
		*bind.dst = id
	}
	return tk, nil
}

// PadID is the [PAD] token id — exposed so callers that batch with padding can
// build attention masks against it.
func (t *Tokenizer) PadID() int32 { return t.padID }

// Encode tokenises text into WordPiece ids framed as [CLS] … [SEP]. The result
// is the input_ids a single-sequence BertModel forward consumes (token_type_ids
// are all zero, attention_mask all one for an unpadded sequence).
//
//	ids := tk.Encode("how do i reset my password?")
func (t *Tokenizer) Encode(text string) []int32 {
	pieces := t.wordPieceAll(t.basicTokenize(text))
	ids := make([]int32, 0, len(pieces)+2)
	ids = append(ids, t.clsID)
	for _, piece := range pieces {
		if id, ok := t.vocab[piece]; ok {
			ids = append(ids, id)
			continue
		}
		ids = append(ids, t.unkID)
	}
	ids = append(ids, t.sepID)
	return ids
}

// EncodePair tokenises a query and passage as [CLS] query [SEP] passage [SEP]
// and returns matching BERT token_type_ids (zero for the first segment and one
// for the passage). This is the sequence shape used by cross-encoder rerankers.
func (t *Tokenizer) EncodePair(query, passage string) ([]int32, []int32) {
	first := t.wordPieceAll(t.basicTokenize(query))
	second := t.wordPieceAll(t.basicTokenize(passage))
	ids := make([]int32, 0, len(first)+len(second)+3)
	types := make([]int32, 0, cap(ids))
	ids = append(ids, t.clsID)
	types = append(types, 0)
	for _, piece := range first {
		ids = append(ids, t.pieceID(piece))
		types = append(types, 0)
	}
	ids = append(ids, t.sepID)
	types = append(types, 0)
	for _, piece := range second {
		ids = append(ids, t.pieceID(piece))
		types = append(types, 1)
	}
	ids = append(ids, t.sepID)
	types = append(types, 1)
	return ids, types
}

func (t *Tokenizer) pieceID(piece string) int32 {
	if id, ok := t.vocab[piece]; ok {
		return id
	}
	return t.unkID
}

// basicTokenize is HF's BasicTokenizer: clean control characters, space out CJK
// ideographs, split on whitespace, then lower-case + strip accents and split
// punctuation per whitespace-delimited word.
func (t *Tokenizer) basicTokenize(text string) []string {
	cleaned := cleanText(text)
	cleaned = spaceChineseChars(cleaned)
	var out []string
	for _, word := range splitWhitespace(cleaned) {
		if t.doLowerCase {
			word = stripAccents(core.Lower(word))
		}
		out = append(out, splitOnPunctuation(word)...)
	}
	return out
}

// wordPieceAll runs greedy longest-match WordPiece over already-basic-tokenised
// words. A word longer than maxWordPieceChars, or one with no matching prefix,
// yields the [UNK] surface string so Encode maps it to unkID.
func (t *Tokenizer) wordPieceAll(words []string) []string {
	var out []string
	for _, word := range words {
		runes := []rune(word)
		if len(runes) > maxWordPieceChars {
			out = append(out, tokenUNK)
			continue
		}
		out = append(out, t.wordPiece(runes)...)
	}
	return out
}

func (t *Tokenizer) wordPiece(runes []rune) []string {
	var pieces []string
	start := 0
	for start < len(runes) {
		end := len(runes)
		matched := ""
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if _, ok := t.vocab[sub]; ok {
				matched = sub
				break
			}
			end--
		}
		if matched == "" {
			// A single unmatchable span makes the whole word [UNK] — HF drops
			// the partial pieces it had already accumulated for this word.
			return []string{tokenUNK}
		}
		pieces = append(pieces, matched)
		start = end
	}
	return pieces
}

// cleanText drops NUL, U+FFFD, and control characters and collapses every
// whitespace rune to a single space, matching BasicTokenizer._clean_text.
func cleanText(text string) string {
	var b []rune
	for _, r := range text {
		if r == 0 || r == 0xFFFD || isControl(r) {
			continue
		}
		if isWhitespace(r) {
			b = append(b, ' ')
			continue
		}
		b = append(b, r)
	}
	return string(b)
}

// spaceChineseChars surrounds every CJK ideograph with spaces so WordPiece
// treats each character as its own token, matching BasicTokenizer.
func spaceChineseChars(text string) string {
	var b []rune
	for _, r := range text {
		if isChineseChar(r) {
			b = append(b, ' ', r, ' ')
			continue
		}
		b = append(b, r)
	}
	return string(b)
}

// stripAccents NFD-decomposes text and drops combining marks (category Mn),
// matching BasicTokenizer._run_strip_accents.
func stripAccents(text string) string {
	decomposed := norm.NFD.String(text)
	var b []rune
	for _, r := range decomposed {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b = append(b, r)
	}
	return string(b)
}

// splitOnPunctuation splits a word so each punctuation rune becomes its own
// token, matching BasicTokenizer._run_split_on_punc.
func splitOnPunctuation(word string) []string {
	if word == "" {
		return nil
	}
	var out []string
	var current []rune
	for _, r := range word {
		if isPunctuation(r) {
			if len(current) > 0 {
				out = append(out, string(current))
				current = current[:0]
			}
			out = append(out, string(r))
			continue
		}
		current = append(current, r)
	}
	if len(current) > 0 {
		out = append(out, string(current))
	}
	return out
}

// splitWhitespace splits on runs of whitespace, dropping empty fields — the
// whitespace_tokenize helper HF applies after cleaning.
func splitWhitespace(text string) []string {
	var out []string
	var current []rune
	for _, r := range text {
		if isWhitespace(r) {
			if len(current) > 0 {
				out = append(out, string(current))
				current = current[:0]
			}
			continue
		}
		current = append(current, r)
	}
	if len(current) > 0 {
		out = append(out, string(current))
	}
	return out
}

func trimCarriageReturn(s string) string {
	if n := len(s); n > 0 && s[n-1] == '\r' {
		return s[:n-1]
	}
	return s
}

func isWhitespace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return unicode.IsSpace(r)
}

func isControl(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	}
	if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
		return true
	}
	return false
}

// isPunctuation matches BERT's treatment: the ASCII punctuation ranges plus any
// Unicode punctuation category. The ASCII ranges are called out explicitly
// because BERT counts characters like '$' '+' '^' as punctuation even though
// Unicode classes them as symbols.
func isPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// isChineseChar covers the CJK Unified Ideograph blocks BERT spaces out — the
// same ranges as BasicTokenizer._is_chinese_char.
func isChineseChar(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF,
		r >= 0x3400 && r <= 0x4DBF,
		r >= 0x20000 && r <= 0x2A6DF,
		r >= 0x2A700 && r <= 0x2B73F,
		r >= 0x2B740 && r <= 0x2B81F,
		r >= 0x2B820 && r <= 0x2CEAF,
		r >= 0xF900 && r <= 0xFAFF,
		r >= 0x2F800 && r <= 0x2FA1F:
		return true
	}
	return false
}
