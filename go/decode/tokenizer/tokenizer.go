// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"slices"
	"sync"

	"dappco.re/go"

	"dappco.re/go/inference/decode/parser"
	coreio "dappco.re/go/io"
)

const (
	tokenizerBPECacheLimit           = 4096
	tokenizerBPECacheMaxSegmentBytes = 64 << 10
	tokenizerBPECacheMaxTokens       = 16 << 10
)

// Tokenizer handles text-to-token and token-to-text conversion.
type Tokenizer struct {
	vocab        map[string]int32
	invVocab     map[int32]string
	merges       []mergePair
	mergeRanks   map[mergeKey]int
	// special holds the special:true added tokens ONLY — the decode-side skip set
	// (Decode/DecodeToken silence these). added holds EVERY added token: the
	// encode-side atomic matcher, because HF matches all added tokens as single
	// ids regardless of the special flag (qwen's <think>/</think> are
	// special:false yet atomic — BPE-splitting them corrupts the reasoning
	// channel the model was trained on).
	special      map[string]int32
	added        map[string]int32
	specialOrder []string

	// specialLeads holds the distinct first bytes of every specialOrder token
	// (all of Gemma/Qwen's specials begin '<', so this is usually one byte).
	// nextSpecialBoundary uses it to IndexAny-jump between candidate start
	// positions instead of scanning the whole remaining text once per special
	// — the boundary scan was O(specials × text), quadratic under a marker-
	// dense stream. Derived once from specialOrder (immutable post-construct)
	// so hand-built tokenizers (tests) and LoadTokenizer share the same path.
	specialLeadsOnce sync.Once
	specialLeads     string

	bosToken int32
	eosToken int32
	hasBOS   bool
	hasEOS   bool

	addPrefixSpace bool

	// GPT-2 byte-level BPE support (used by Qwen, GPT, Llama, etc.)
	isGPT2BPE   bool
	gpt2Decoder map[rune]byte // Unicode char → original byte
	gpt2Encoder map[byte]rune // original byte → Unicode char

	bpeCacheMu sync.RWMutex
	bpeCache   map[string][]int32
	// bpeCacheOrder is a FIFO eviction ring: it grows by append until it holds
	// tokenizerBPECacheLimit keys, then bpeCacheHead marks the oldest slot,
	// overwritten in place on eviction. bpeCacheHead is meaningful only once the
	// ring is full.
	bpeCacheOrder []string
	bpeCacheHead  int
}

type mergePair struct {
	a, b string
	rank int
}

type mergeKey struct {
	a string
	b string
}

// bpeNode tracks one live symbol as a byte window [start,end) into the segment
// text rather than an owned string. Merging two adjacent symbols is then a pure
// window-extend (left.end = right.end) — the merged token is text[left.start:
// right.end], a zero-copy substring view — instead of `left.token += right.token`
// allocating a fresh growing string on every merge. Byte-identical because the
// split symbols are contiguous, non-overlapping substrings of text and merges
// only ever join a node with its immediate successor, so every window stays a
// real contiguous slice of the original segment.
type bpeNode struct {
	start   int
	end     int
	prev    int
	next    int
	alive   bool
	version uint32
}

type bpeCandidate struct {
	rank         int
	left         int
	right        int
	leftVersion  uint32
	rightVersion uint32
}

// bpeCandidateHeap is a min-heap of bpeCandidate ordered by (rank
// ascending, left ascending). The original implementation satisfied
// container/heap.Interface, which forced every Push to box a candidate
// into `any` (one alloc per push) and every Pop to type-assert back —
// pushDirect / popDirect below replace that path with direct typed
// sift-up / sift-down operations on the underlying slice.
type bpeCandidateHeap []bpeCandidate

func (h bpeCandidateHeap) Len() int {
	return len(h)
}

// pushDirect appends c to the heap and sifts it up. Bypasses
// container/heap.Push's `x any` interface boxing — that boxing forces
// every bpeCandidate to escape to the heap (one alloc per push), and
// bpeMerge does ~2N pushes per call. The version-stale-discard
// correctness invariant is preserved (the less ordering — rank then
// left — is identical to the prior heap.Interface path; the wrapper
// just emits the same up-sift without the interface dispatch).
func (h *bpeCandidateHeap) pushDirect(c bpeCandidate) {
	*h = append(*h, c)
	// sift-up
	s := *h
	i := len(s) - 1
	for i > 0 {
		parent := (i - 1) / 2
		// Inline of Less(i, parent): rank then left.
		if s[i].rank < s[parent].rank ||
			(s[i].rank == s[parent].rank && s[i].left < s[parent].left) {
			s[i], s[parent] = s[parent], s[i]
			i = parent
			continue
		}
		break
	}
}

// popDirect removes and returns the minimum candidate. Bypasses
// heap.Pop's `any` return-type boxing.
func (h *bpeCandidateHeap) popDirect() bpeCandidate {
	s := *h
	n := len(s) - 1
	s[0], s[n] = s[n], s[0]
	// sift-down on s[:n]
	i := 0
	for {
		left := 2*i + 1
		if left >= n {
			break
		}
		smallest := left
		right := left + 1
		if right < n {
			// right < left?
			if s[right].rank < s[left].rank ||
				(s[right].rank == s[left].rank && s[right].left < s[left].left) {
				smallest = right
			}
		}
		// smallest < i?
		if s[smallest].rank < s[i].rank ||
			(s[smallest].rank == s[i].rank && s[smallest].left < s[i].left) {
			s[i], s[smallest] = s[smallest], s[i]
			i = smallest
			continue
		}
		break
	}
	out := s[n]
	*h = s[:n]
	return out
}

// tokenizerJSON is the HuggingFace tokenizer.json format.
type tokenizerJSON struct {
	Normalizer struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type     string `json:"type"`
		Behavior string `json:"behavior"`
	} `json:"pre_tokenizer"`
	Model struct {
		Type         string `json:"type"`
		Vocab        any    `json:"vocab"`
		Merges       any    `json:"merges"`
		ByteFallback bool   `json:"byte_fallback"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// IndexIn returns the byte position of substr in s, or -1 if not found.
// Routes through core.Index — stdlib substring search uses Rabin-Karp /
// two-way under the hood, an order of magnitude faster than the naive
// O(n*m) byte-walk this used to do because every iteration constructed
// a fresh `s[i:i+subLen] == substr` slice header for comparison.
//
//	pos := IndexIn("hello world", "world") // → 6
//	pos := IndexIn("hello", "xyz")         // → -1
func IndexIn(s, substr string) int {
	return core.Index(s, substr)
}

// NewForDecode builds a minimal decode-only Tokenizer from an inverse vocabulary (token id
// → piece) — enough for DecodeToken/DecodeOne without loading a full vocab/merges file. A
// nil invVocab yields an empty tokenizer. Handy for lightweight decoders and tests that
// only need to turn ids back into text.
func NewForDecode(invVocab map[int32]string) *Tokenizer {
	return &Tokenizer{invVocab: invVocab}
}

// LoadTokenizer reads a tokenizer.json file and creates a Tokenizer.
//
//	tok, err := metal.LoadTokenizer("/path/to/model/tokenizer.json")
func LoadTokenizer(path string) (*Tokenizer, error) {
	str, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("tokenizer.LoadTokenizer", "read "+path, err)
	}
	data := []byte(str)

	var tj tokenizerJSON
	if r := core.JSONUnmarshal(data, &tj); !r.OK {
		return nil, core.E("tokenizer.LoadTokenizer", "parse", nil)
	}

	tokenizer := &Tokenizer{
		vocab:          make(map[string]int32),
		invVocab:       make(map[int32]string),
		special:        make(map[string]int32),
		added:          make(map[string]int32),
		addPrefixSpace: true,
	}

	// Vocab arrives as any (map[string]interface{} from JSON) — convert
	// to map[string]int32 by re-marshalling through core.JSONMarshal.
	if tj.Model.Vocab != nil {
		vocabBytes := core.JSONMarshal(tj.Model.Vocab)
		if !vocabBytes.OK {
			return nil, core.E("tokenizer.LoadTokenizer", "re-encode vocab", nil)
		}
		var vocab map[string]int32
		if r := core.JSONUnmarshal(vocabBytes.Value.([]byte), &vocab); !r.OK {
			return nil, core.E("tokenizer.LoadTokenizer", "parse vocab", nil)
		}
		tokenizer.vocab = vocab
		for tokenText, tokenID := range vocab {
			tokenizer.invVocab[tokenID] = tokenText
		}
	}

	// Merges arrives as any — supports both ["a b", ...] and [["a","b"], ...]
	if tj.Model.Merges != nil {
		mergeBytes := core.JSONMarshal(tj.Model.Merges)
		if mergeBytes.OK {
			raw := mergeBytes.Value.([]byte)
			var stringMerges []string
			if r := core.JSONUnmarshal(raw, &stringMerges); r.OK {
				for rank, merge := range stringMerges {
					parts := core.SplitN(merge, " ", 2)
					if len(parts) == 2 {
						tokenizer.merges = append(tokenizer.merges, mergePair{a: parts[0], b: parts[1], rank: rank})
					}
				}
			} else {
				var arrayMerges [][]string
				if r := core.JSONUnmarshal(raw, &arrayMerges); r.OK {
					for rank, pair := range arrayMerges {
						if len(pair) == 2 {
							tokenizer.merges = append(tokenizer.merges, mergePair{a: pair[0], b: pair[1], rank: rank})
						}
					}
				}
			}
		}
	}

	tokenizer.mergeRanks = make(map[mergeKey]int, len(tokenizer.merges))
	for _, merge := range tokenizer.merges {
		tokenizer.mergeRanks[mergeKey{a: merge.a, b: merge.b}] = merge.rank
	}

	for _, added := range tj.AddedTokens {
		if added.Special {
			tokenizer.special[added.Content] = added.ID
		}
		tokenizer.added[added.Content] = added.ID
		tokenizer.vocab[added.Content] = added.ID
		tokenizer.invVocab[added.ID] = added.Content
	}
	tokenizer.specialOrder = make([]string, 0, len(tokenizer.added))
	for tokenText := range tokenizer.added {
		tokenizer.specialOrder = append(tokenizer.specialOrder, tokenText)
	}
	slices.SortFunc(tokenizer.specialOrder, func(a, b string) int {
		if len(a) != len(b) {
			return len(b) - len(a)
		}
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	})

	// Detect GPT-2 byte-level BPE (Qwen, GPT, DeepSeek use Ġ for space).
	// Check for "Ġthe" rather than bare "Ġ" — large SentencePiece vocabs
	// (Gemma3 262K) may include Ġ as an obscure character without using
	// GPT-2 byte encoding.
	if _, ok := tokenizer.vocab["Ġthe"]; ok {
		tokenizer.isGPT2BPE = true
		tokenizer.gpt2Decoder, tokenizer.gpt2Encoder = buildGPT2ByteMaps()
	}
	if tj.Normalizer.Type == "Replace" && tj.Normalizer.Content == "▁" &&
		tj.PreTokenizer.Type == "Split" && tj.PreTokenizer.Behavior == "MergedWithPrevious" {
		tokenizer.addPrefixSpace = false
	}

	if id, ok := tokenizer.special["<bos>"]; ok {
		tokenizer.bosToken = id
		tokenizer.hasBOS = true
	}
	if id, ok := tokenizer.special["<eos>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Gemma: <end_of_turn> is the generation stop token
	if id, ok := tokenizer.special["<end_of_turn>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Qwen3: <|im_end|> is the generation stop token
	if id, ok := tokenizer.special["<|im_end|>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// NB: <|im_start|> is deliberately NOT a BOS. The ChatML dialect supplies
	// every <|im_start|> through the template and HF never auto-prepends one
	// (add_bos_token is false for the qwen family) — the old mapping injected a
	// ghost <|im_start|> at the head of every encode that didn't already start
	// with it, corrupting woken-conversation continuations.
	// Llama 3: <|eot_id|> is the turn-end token
	if id, ok := tokenizer.special["<|eot_id|>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Gemma 4: <turn|> is the assistant turn stop token.
	if id, ok := tokenizer.special["<turn|>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Llama 3 BOS: <|begin_of_text|>
	if id, ok := tokenizer.special["<|begin_of_text|>"]; ok {
		tokenizer.bosToken = id
		tokenizer.hasBOS = true
	}

	return tokenizer, nil
}

func (t *Tokenizer) matchSpecialToken(input string) (string, int32, bool) {
	// A special can only be a prefix of input when input's first byte is one of
	// the specials' lead bytes. That single membership test rejects the common
	// "no special here" case (the first token of every clean segment) in O(1)
	// instead of walking HasPrefix over every special — the walk is O(specials)
	// and fires once per Encode segment. Specials are non-empty (LoadTokenizer
	// builds them from added-token content), so a real prefix always shares
	// input's first byte with input.
	if input == "" || !t.isSpecialLead(input[0]) {
		return "", 0, false
	}
	for _, tok := range t.specialOrder {
		if core.HasPrefix(input, tok) {
			// added is the full atomic-match set; hand-built tokenizers (tests)
			// that populate only special still resolve through the fallback.
			if id, ok := t.added[tok]; ok {
				return tok, id, true
			}
			return tok, t.special[tok], true
		}
	}
	return "", 0, false
}

// isSpecialLead reports whether b is the first byte of any special token. The
// lead set is tiny (usually the single byte '<'), so a linear scan is a couple
// of compares — cheaper than a map lookup and allocation-free.
func (t *Tokenizer) isSpecialLead(b byte) bool {
	leads := t.specialLeadBytes()
	for i := 0; i < len(leads); i++ {
		if leads[i] == b {
			return true
		}
	}
	return false
}

// specialLeadBytes returns the distinct first bytes of every special token, as
// a string suitable for core.IndexAny. Computed once from specialOrder, which
// is immutable after construction. An empty result means no special can start
// anywhere (either no specials, or — impossible in practice — an empty-string
// special), so the boundary scan short-circuits to len(input).
func (t *Tokenizer) specialLeadBytes() string {
	t.specialLeadsOnce.Do(func() {
		var seen [256]bool
		leads := make([]byte, 0, 4)
		for _, tok := range t.specialOrder {
			if tok == "" {
				continue
			}
			b := tok[0]
			if !seen[b] {
				seen[b] = true
				leads = append(leads, b)
			}
		}
		t.specialLeads = string(leads)
	})
	return t.specialLeads
}

// nextSpecialBoundary returns the smallest index > 0 at which any special token
// starts in input, or len(input) if none does. Its contract is that input has
// no special token at position 0 (matchSpecialToken already ran and missed), so
// "smallest start > 0" equals the naive "min over specials of first occurrence".
//
// A special can only begin at one of its lead bytes, so IndexAny hops directly
// from one candidate position to the next and only pays the per-special prefix
// check where a lead byte actually lands. That replaces the previous shape —
// one full IndexIn scan of the whole remaining text for every special, i.e.
// O(specials × text) per call and O(specials × segments²) across the Encode
// loop of a marker-dense stream — with O(text + candidates × specials).
func (t *Tokenizer) nextSpecialBoundary(input string) int {
	leads := t.specialLeadBytes()
	if leads == "" {
		return len(input)
	}
	// Position 0 is known clear (matchSpecialToken missed), so scan from 1.
	for i := 1; i < len(input); {
		rel := core.IndexAny(input[i:], leads)
		if rel < 0 {
			return len(input)
		}
		pos := i + rel
		if _, _, ok := t.matchSpecialToken(input[pos:]); ok {
			return pos
		}
		i = pos + 1
	}
	return len(input)
}

func (t *Tokenizer) normalizeSentencePieceSegment(segment string) string {
	if segment == "" {
		return ""
	}
	// Decide upfront whether we need the leading ▁ prefix. The original
	// code called Replace first (allocating a new string), then checked
	// the result for "▁" prefix, then prefixed it (a SECOND alloc). Both
	// can be merged into a single Builder pass:
	//
	//   - Count spaces to compute exact output size (▁ is 3 bytes, ' ' is
	//     1, so each space adds 2 bytes to the output length).
	//   - Decide prefix decision up front: needs ▁ iff addPrefixSpace AND
	//     the segment's first byte is not the ▁-leader (E2). The latter
	//     test is a single byte compare instead of HasPrefix walking 3.
	//   - If no work needed (no spaces, no prefix), return segment as-is
	//     — zero allocations, the input string passes through directly.
	needPrefix := t.addPrefixSpace
	if needPrefix && segment[0] == 0xE2 && len(segment) >= 3 &&
		segment[1] == 0x96 && segment[2] == 0x81 {
		needPrefix = false
	}

	// Count spaces — also tells us if Replace work is needed.
	spaces := 0
	for i := 0; i < len(segment); i++ {
		if segment[i] == ' ' {
			spaces++
		}
	}

	if !needPrefix && spaces == 0 {
		return segment
	}

	// Output size known exactly: prefix (3) + segment + 2 per space.
	outLen := len(segment) + 2*spaces
	if needPrefix {
		outLen += 3
	}
	buf := make([]byte, 0, outLen)
	if needPrefix {
		buf = append(buf, 0xE2, 0x96, 0x81)
	}
	for i := 0; i < len(segment); i++ {
		b := segment[i]
		if b == ' ' {
			buf = append(buf, 0xE2, 0x96, 0x81)
			continue
		}
		buf = append(buf, b)
	}
	return core.AsString(buf)
}

// buildGPT2ByteMaps creates the GPT-2 byte-level BPE encoding/decoding maps.
// GPT-2 maps all 256 bytes to printable Unicode characters to avoid control chars
// in the vocabulary. Printable ASCII + Latin-1 Supplement map to themselves;
// everything else (0-32, 127-160, 173) maps to U+0100 onwards.
func buildGPT2ByteMaps() (decoder map[rune]byte, encoder map[byte]rune) {
	encoder = make(map[byte]rune, 256)
	decoder = make(map[rune]byte, 256)

	// Self-mapping ranges: printable ASCII + Latin-1 Supplement
	// Use int loop variable to avoid byte overflow at 255.
	selfMap := func(lo, hi int) {
		for b := lo; b <= hi; b++ {
			encoder[byte(b)] = rune(b)
			decoder[rune(b)] = byte(b)
		}
	}
	selfMap(33, 126)  // ! through ~
	selfMap(161, 172) // ¡ through ¬
	selfMap(174, 255) // ® through ÿ

	// Non-self-mapping: control chars, space, DEL, and gaps
	nonSelfMapped := 0
	for b := range 256 {
		if _, ok := encoder[byte(b)]; !ok {
			mappedRune := rune(256 + nonSelfMapped)
			encoder[byte(b)] = mappedRune
			decoder[mappedRune] = byte(b)
			nonSelfMapped++
		}
	}
	return
}

// bpeMergePushPair inlines the prior pushPair closure as a free
// function. The closure version captured nodes + candidates + t which
// forced the closure (and its captured slice headers / map) to escape
// to heap on every bpeMerge call. The free-function version takes the
// state explicitly + uses pushDirect to bypass container/heap's `any`
// interface boxing — one alloc per push eliminated. text is the segment the
// node windows index into; the pair's merge key is the two adjacent windows'
// substring views, allocation-free (the map lookup hashes the bytes, not the
// header).
func bpeMergePushPair(text string, nodes []bpeNode, candidates *bpeCandidateHeap, ranks map[mergeKey]int, left int) {
	if left < 0 || left >= len(nodes) || !nodes[left].alive {
		return
	}
	right := nodes[left].next
	if right < 0 || right >= len(nodes) || !nodes[right].alive {
		return
	}
	rank, ok := ranks[mergeKey{a: text[nodes[left].start:nodes[left].end], b: text[nodes[right].start:nodes[right].end]}]
	if !ok {
		return
	}
	candidates.pushDirect(bpeCandidate{
		rank:         rank,
		left:         left,
		right:        right,
		leftVersion:  nodes[left].version,
		rightVersion: nodes[right].version,
	})
}

// bpeMerge applies BPE merges to a sequence of symbols until no more merges apply.
// Uses the standard algorithm: repeatedly find the lowest-rank adjacent pair and merge it.
// text is the segment the symbols were split from (symbols == the in-order rune
// pieces of text); each node tracks a byte window into it, so a merge extends a
// window instead of concatenating strings — the previous shape's
// `left.token += right.token` allocated a fresh growing string on every merge
// (the single biggest allocator on the cache-miss encode path).
func (t *Tokenizer) bpeMerge(text string, symbols []string) []string {
	if len(symbols) <= 1 || len(t.mergeRanks) == 0 {
		return symbols
	}

	nodes := make([]bpeNode, len(symbols))
	offset := 0
	for i, sym := range symbols {
		nodes[i] = bpeNode{
			start: offset,
			end:   offset + len(sym),
			prev:  i - 1,
			next:  i + 1,
			alive: true,
		}
		offset += len(sym)
	}
	nodes[len(nodes)-1].next = -1

	candidates := make(bpeCandidateHeap, 0, len(nodes)-1)
	for i := 0; i < len(nodes)-1; i++ {
		bpeMergePushPair(text, nodes, &candidates, t.mergeRanks, i)
	}
	// pushDirect maintains heap invariant on each insert — no separate
	// heap.Init pass needed.

	for candidates.Len() > 0 {
		candidate := candidates.popDirect()
		left, right := candidate.left, candidate.right
		if left < 0 || right < 0 || left >= len(nodes) || right >= len(nodes) {
			continue
		}
		if !nodes[left].alive || !nodes[right].alive || nodes[left].next != right || nodes[right].prev != left {
			continue
		}
		if nodes[left].version != candidate.leftVersion || nodes[right].version != candidate.rightVersion {
			continue
		}
		if rank, ok := t.mergeRanks[mergeKey{a: text[nodes[left].start:nodes[left].end], b: text[nodes[right].start:nodes[right].end]}]; !ok || rank != candidate.rank {
			continue
		}

		// Window-extend: left now spans to right's end. The merged token is
		// text[left.start:left.end] — a substring view, no allocation. Valid
		// because right is left's immediate successor, so the two windows are
		// contiguous ([left.start,left.end) then [right.start,right.end) with
		// left.end == right.start) and their union is one contiguous window.
		nodes[left].end = nodes[right].end
		nodes[left].next = nodes[right].next
		nodes[left].version++
		nodes[right].alive = false
		nodes[right].version++
		if next := nodes[right].next; next >= 0 {
			nodes[next].prev = left
		}

		bpeMergePushPair(text, nodes, &candidates, t.mergeRanks, nodes[left].prev)
		bpeMergePushPair(text, nodes, &candidates, t.mergeRanks, left)
	}

	merged := symbols[:0]
	for i := 0; i >= 0; i = nodes[i].next {
		merged = append(merged, text[nodes[i].start:nodes[i].end])
	}
	return merged
}

func (t *Tokenizer) cachedBPETokens(key string) ([]int32, bool) {
	t.bpeCacheMu.RLock()
	// Defer-free path — the hot one fires once per Encode segment so
	// the ~7 ns/op `defer t.bpeCacheMu.RUnlock()` cost shows up at the
	// envelope. Explicit RUnlock on both branches keeps the lock
	// discipline visible at the call site.
	if len(t.bpeCache) == 0 {
		t.bpeCacheMu.RUnlock()
		return nil, false
	}
	tokens, ok := t.bpeCache[key]
	t.bpeCacheMu.RUnlock()
	return tokens, ok
}

func (t *Tokenizer) storeBPETokens(key string, tokens []int32) {
	if len(key) > tokenizerBPECacheMaxSegmentBytes || len(tokens) > tokenizerBPECacheMaxTokens {
		return
	}
	t.bpeCacheMu.Lock()
	defer t.bpeCacheMu.Unlock()
	if t.bpeCache == nil {
		t.bpeCache = make(map[string][]int32)
	}
	if _, ok := t.bpeCache[key]; ok {
		t.bpeCache[key] = append([]int32(nil), tokens...)
		return
	}
	// FIFO ring eviction: append while the ring is below the limit, otherwise
	// overwrite the oldest slot in place and advance the head. The previous
	// shape copy-shifted the entire order slice left by one on every eviction —
	// an O(limit) memmove per store once the cache was full, which dominated
	// tokenisation of long, low-repeat inputs (code, mixed scripts, document
	// ingestion) where every distinct segment evicts. Ring eviction is O(1) and
	// preserves the same FIFO order and limit cap.
	if len(t.bpeCacheOrder) < tokenizerBPECacheLimit {
		t.bpeCacheOrder = append(t.bpeCacheOrder, key)
	} else {
		delete(t.bpeCache, t.bpeCacheOrder[t.bpeCacheHead])
		t.bpeCacheOrder[t.bpeCacheHead] = key
		t.bpeCacheHead++
		if t.bpeCacheHead >= tokenizerBPECacheLimit {
			t.bpeCacheHead = 0
		}
	}
	t.bpeCache[key] = append([]int32(nil), tokens...)
}

// splitRunes appends each UTF-8 rune of s to dst as a substring of s
// (zero-alloc per rune — the substring shares the underlying byte
// array). The prior `string(r)` per-rune materialisation allocated a
// fresh 1-4-byte string for every rune; substring slicing reuses the
// input's backing memory and is safe because the input is a string
// (immutable). Returns the appended slice for caller to chain.
func splitRunes(dst []string, s string) []string {
	for i := 0; i < len(s); {
		b := s[i]
		// Fast-path ASCII — single-byte rune, no decode work.
		if b < 0x80 {
			dst = append(dst, s[i:i+1])
			i++
			continue
		}
		// Multi-byte rune — determine length from leading byte.
		var n int
		switch {
		case b&0xE0 == 0xC0:
			n = 2
		case b&0xF0 == 0xE0:
			n = 3
		case b&0xF8 == 0xF0:
			n = 4
		default:
			// Invalid leading byte; emit as single byte and advance.
			n = 1
		}
		if i+n > len(s) {
			n = len(s) - i
		}
		dst = append(dst, s[i:i+n])
		i += n
	}
	return dst
}

func (t *Tokenizer) encodeSentencePieceSegment(segment string) []int32 {
	spText := t.normalizeSentencePieceSegment(segment)
	if spText == "" {
		return nil
	}
	// Key the BPE cache by the segment text directly. isGPT2BPE is fixed for a
	// tokenizer's whole life (set once in LoadTokenizer, never mutated), so only
	// one of encodeSentencePieceSegment / encodeGPT2Segment ever runs on a given
	// tokenizer — a "sp"/"gpt2" kind prefix could never disambiguate two entries,
	// and dropping it saves the per-segment `kind + "\x00" + segment` concat
	// allocation on every cache lookup and store.
	key := spText
	if cached, ok := t.cachedBPETokens(key); ok {
		return cached
	}

	symbols := splitRunes(make([]string, 0, len(spText)), spText)
	symbols = t.bpeMerge(spText, symbols)

	tokens := make([]int32, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			tokens = append(tokens, id)
		}
	}
	t.storeBPETokens(key, tokens)
	return tokens
}

func (t *Tokenizer) encodeGPT2Segment(segment string) []int32 {
	if segment == "" {
		return nil
	}
	encoded := core.NewBuilder()
	// Pre-size the Builder — every input byte maps to one rune (max 4
	// bytes); the worst case is 4*len(segment), but in practice most
	// GPT-2 byte-encoded bytes are 2-byte runes so 2*len(segment) is a
	// fair starting size that avoids a couple of geometric reallocs.
	encoded.Grow(2 * len(segment))
	for _, b := range []byte(segment) {
		if r, ok := t.gpt2Encoder[b]; ok {
			encoded.WriteRune(r)
		}
	}
	encodedText := encoded.String()
	if encodedText == "" {
		return nil
	}
	// Key by the encoded text directly — see encodeSentencePieceSegment: only one
	// encode kind runs per tokenizer, so no kind prefix is needed.
	key := encodedText
	if cached, ok := t.cachedBPETokens(key); ok {
		return cached
	}

	symbols := splitRunes(make([]string, 0, len(encodedText)), encodedText)
	symbols = t.bpeMerge(encodedText, symbols)

	tokens := make([]int32, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			tokens = append(tokens, id)
		}
	}
	t.storeBPETokens(key, tokens)
	return tokens
}

func (t *Tokenizer) shouldPrependBOS(text string) bool {
	if !t.hasBOS {
		return false
	}
	bosText := t.invVocab[t.bosToken]
	return bosText == "" || !core.HasPrefix(text, bosText)
}

// Encode converts text to token IDs (prepends BOS token).
//
//	ids := tok.Encode("Hello world") // → []int32{2, 9906, 1917}
func (t *Tokenizer) Encode(text string) []int32 {
	if t.isGPT2BPE {
		return t.encodeGPT2(text)
	}

	tokens := make([]int32, 0, len(text)+1)
	if t.shouldPrependBOS(text) {
		tokens = append(tokens, t.bosToken)
	}

	// SentencePiece style: split into segments around special tokens, then BPE each segment.
	remaining := text
	for remaining != "" {
		// Check for special tokens at the current position.
		if tok, id, ok := t.matchSpecialToken(remaining); ok {
			tokens = append(tokens, id)
			remaining = remaining[len(tok):]
			continue
		}

		// Find the next special token boundary (or end of string).
		end := t.nextSpecialBoundary(remaining)
		segment := remaining[:end]
		remaining = remaining[end:]

		tokens = append(tokens, t.encodeSentencePieceSegment(segment)...)
	}

	return tokens
}

// encodeGPT2 encodes text using GPT-2 byte-level BPE.
func (t *Tokenizer) encodeGPT2(text string) []int32 {
	tokens := make([]int32, 0, len(text)+1)
	if t.shouldPrependBOS(text) {
		tokens = append(tokens, t.bosToken)
	}

	// Split text around special tokens (matched in original form, not byte-encoded).
	remaining := text
	for remaining != "" {
		// Check for special tokens at the current position.
		if tok, id, ok := t.matchSpecialToken(remaining); ok {
			tokens = append(tokens, id)
			remaining = remaining[len(tok):]
			continue
		}

		// Find the next special token boundary (or end of string).
		end := t.nextSpecialBoundary(remaining)
		segment := remaining[:end]
		remaining = remaining[end:]

		tokens = append(tokens, t.encodeGPT2Segment(segment)...)
	}

	return tokens
}

// Decode converts token IDs back to text (strips SentencePiece leading space).
//
//	text := tok.Decode([]int32{9906, 1917}) // → "Hello world"
func (t *Tokenizer) Decode(tokens []int32) string {
	// GPT-2 byte-level path is handled by walking the raw concatenation
	// through decodeGPT2Bytes — the byte-level decoder strips its own
	// envelope, so the SentencePiece ▁-translation must NOT run on it.
	if t.isGPT2BPE {
		sb := core.NewBuilder()
		for _, id := range tokens {
			if text, ok := t.invVocab[id]; ok {
				if _, isSpecial := t.special[text]; isSpecial {
					continue
				}
				sb.WriteString(text)
			}
		}
		return t.decodeGPT2Bytes(sb.String())
	}

	// SentencePiece path — translate ▁ → space inline while assembling,
	// then strip the single leading space (the prefix-space marker on
	// the first emitted token). Replaces the prior triple walk:
	//   1) Builder.WriteString accumulation → raw
	//   2) core.Replace(raw, "▁", " ")      → result (new alloc)
	//   3) HasPrefix(" ") + slice           → leading-space strip
	// with a single Builder pass that splits on ▁ via indexBytePrefix —
	// the fast-path for tokens without ▁ falls into a single WriteString
	// (memmove), and the only translation work is per-▁-occurrence.
	//
	// A pre-sizing pass (Grow on summed-text length) was tried and
	// reverted — the second map-walk cost outweighs the saved geometric
	// reallocs at every shape from 3 to 64 tokens. Builder's default
	// growth strategy wins here.
	sb := core.NewBuilder()
	for _, id := range tokens {
		text, ok := t.invVocab[id]
		if !ok {
			continue
		}
		if _, isSpecial := t.special[text]; isSpecial {
			continue
		}
		// Bulk-write tokens without ▁ (common case — most vocab tokens
		// are leaf-bytes or non-prefixed merges).
		for {
			idx := indexBytePrefix(text)
			if idx < 0 {
				sb.WriteString(text)
				break
			}
			if idx > 0 {
				sb.WriteString(text[:idx])
			}
			sb.WriteByte(' ')
			text = text[idx+3:]
			if text == "" {
				break
			}
		}
	}
	out := sb.String()
	if len(out) > 0 && out[0] == ' ' {
		return out[1:]
	}
	return out
}

// indexBytePrefix returns the byte offset of the SentencePiece ▁
// marker (U+2581, E2 96 81) in s, or -1 if absent. Inlined so Decode's
// inner loop can branch on a simple int compare instead of the more
// general core.Index three-byte-string-needle call.
func indexBytePrefix(s string) int {
	for i := 0; i+2 < len(s); i++ {
		if s[i] == 0xE2 && s[i+1] == 0x96 && s[i+2] == 0x81 {
			return i
		}
	}
	// Trailing 2 bytes can't contain the 3-byte marker.
	return -1
}

// channelOpenMarker and channelCloseMarker are Gemma 4's reasoning-channel
// delimiters (gpt-oss uses <|channel> as well). Unlike BOS/EOS/turn, these are
// content-bearing control tokens: the reasoning parser needs them in the
// decoded stream to split the thinking span from the visible answer, so
// DecodeToken keeps them. The strings are owned by the marker grammar
// (decode/parser grammar.go) — one source for all three consumers.
const (
	channelOpenMarker  = parser.ChannelOpenMarker
	channelCloseMarker = parser.ChannelCloseMarker
	// The tool-call delimiters are content-bearing the same way: the tool parser
	// needs the whole <|tool_call>…<tool_call|> span (and the <|"|> argument
	// quotes) in the decoded stream to lift a structured call, so DecodeToken
	// keeps them too.
	toolCallOpenMarker  = parser.ToolCallOpenMarker
	toolCallCloseMarker = parser.ToolCallCloseMarker
	toolArgQuoteMarker  = parser.ToolArgQuoteMarker
)

// DecodeToken converts a single token ID to text for streaming.
// Preserves the leading space (word boundary) for correct inter-token spacing.
//
//	text := tok.DecodeToken(1917) // → " world" (note leading space)
func (t *Tokenizer) DecodeToken(id int32) string {
	text, ok := t.invVocab[id]
	if !ok {
		return ""
	}
	if _, isSpecial := t.special[text]; isSpecial {
		// Gemma 4 emits <|channel>thought … <channel|> for its thinking channel
		// (31B/26B can emit a ghost empty channel even with thinking off).
		// Preserve the delimiters so the parser strips the whole span instead of
		// leaking a bare "thought" line into the reply; other specials stay
		// invisible — they terminate generation and never reach the content.
		if text == channelOpenMarker || text == channelCloseMarker ||
			text == toolCallOpenMarker || text == toolCallCloseMarker || text == toolArgQuoteMarker {
			return text
		}
		return ""
	}

	if t.isGPT2BPE {
		return t.decodeGPT2Bytes(text)
	}

	// SentencePiece: translate ▁ → space, keeping it (it's the word boundary).
	// Replaces core.Replace, which allocated a fresh string on every token that
	// carried a marker (1 alloc/8 B per word-leading token in generation).
	// indexBytePrefix lets the no-marker continuation tokens (the common mid-
	// word case) return text unchanged with zero allocations, while marker
	// tokens take a single Builder pass instead of strings.ReplaceAll's
	// internal allocation + scan.
	idx := indexBytePrefix(text)
	if idx < 0 {
		return text
	}
	// Solo marker fast path: a bare "▁" token decodes to exactly " ".
	// The Builder loop below would allocate an 8 B buffer to materialise
	// that single space on every emitted standalone-space token; a const
	// return is byte-identical and zero-alloc. The dominant word-leading
	// shape ("▁word", len > 3) still allocates its output string — that
	// string's bytes differ from the input's (0x20… vs E2 96 81…), so no
	// substring view exists and the alloc is the irreducible output, not
	// the Builder. Only the pure-marker case (idx 0, nothing after) is
	// short-circuitable.
	if idx == 0 && len(text) == 3 {
		return spaceString
	}
	sb := core.NewBuilder()
	for {
		if idx > 0 {
			sb.WriteString(text[:idx])
		}
		sb.WriteByte(' ')
		text = text[idx+3:]
		idx = indexBytePrefix(text)
		if idx < 0 {
			sb.WriteString(text)
			break
		}
	}
	return sb.String()
}

// spaceString is the decode of a bare SentencePiece ▁ marker — a single
// space. Held as a package const so DecodeToken can return it without the
// Builder buffer allocation. Read-only data segment; returning it copies
// only the string header.
const spaceString = " "

// DecodeOne mirrors Decode([]int32{id}) semantics for a single token without
// allocating a one-element slice header at the call site. The hot path is the
// root-package Tokenizer.IDToken wrapper, which fires once per emitted
// generation token. Direct vocab lookup + leading-space strip replaces the
// allocation + Builder + final string() path that Decode([]int32{id}) would
// take.
//
//	text := tok.DecodeOne(1917) // → "world" (leading SP space stripped)
func (t *Tokenizer) DecodeOne(id int32) string {
	text, ok := t.invVocab[id]
	if !ok {
		return ""
	}
	if _, isSpecial := t.special[text]; isSpecial {
		return ""
	}

	if t.isGPT2BPE {
		return t.decodeGPT2Bytes(text)
	}

	// SentencePiece: translate ▁ → space, then strip a single leading space to
	// match Decode([]int32{id}) exactly. A solo "▁" therefore returns "" — the
	// root wrapper substitutes a bare space for that case from its inverse-vocab
	// fallback.
	//
	// Zero-alloc fast paths replace the prior core.Replace (1 alloc/8 B on every
	// marker-bearing token, fired once per emitted generation token):
	//   - no marker            → return text (continuation pieces, unchanged)
	//   - leading marker only  → return text[3:] (drop ▁; the ▁→space→strip
	//                            round-trip is identity on a substring view)
	// Only the rare interior-marker token (e.g. "▁a▁b") takes a Builder pass.
	idx := indexBytePrefix(text)
	if idx < 0 {
		return text
	}
	rest := text[idx+3:]
	next := indexBytePrefix(rest)
	if idx == 0 && next < 0 {
		// Leading "▁" + remainder with no further marker: ▁→space gives
		// " "+rest, and stripping the single leading space yields rest.
		return rest
	}
	if idx > 0 && next < 0 {
		// No leading marker, single interior marker: text[:idx] + " " + rest.
		// HasPrefix(" ") is false (text[0] != ▁), so no leading strip.
		sb := core.NewBuilder()
		sb.WriteString(text[:idx])
		sb.WriteByte(' ')
		sb.WriteString(rest)
		return sb.String()
	}
	// General case: multiple markers. Translate inline then strip a leading
	// space if present.
	sb := core.NewBuilder()
	work := text
	mIdx := idx
	for {
		if mIdx > 0 {
			sb.WriteString(work[:mIdx])
		}
		sb.WriteByte(' ')
		work = work[mIdx+3:]
		mIdx = indexBytePrefix(work)
		if mIdx < 0 {
			sb.WriteString(work)
			break
		}
	}
	out := sb.String()
	if len(out) > 0 && out[0] == ' ' {
		return out[1:]
	}
	return out
}

// decodeGPT2Bytes converts GPT-2 byte-level BPE Unicode back to real bytes.
func (t *Tokenizer) decodeGPT2Bytes(s string) string {
	if s == "" {
		return ""
	}
	// Zero-alloc fast path for self-mapped pure-ASCII pieces — the common
	// per-token continuation case (mid-word fragments like "hello", "ing").
	// GPT-2's byte map sends printable ASCII (33–126) to itself, so a piece
	// composed entirely of those bytes decodes byte-for-byte to itself and
	// the input string can be returned directly, skipping the make([]byte)
	// + per-rune copy below. The scan bails on the FIRST byte that isn't a
	// single-byte rune (>= 0x80) or isn't self-mapped (space → Ġ, control
	// chars, DEL — none of which equal their own decoded byte), so the
	// returned-as-is result is provably identical to the built buffer.
	// gpt2Decoder is only populated when isGPT2BPE is set (this method's
	// sole caller path), so the lookup is always live here.
	fast := true
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b >= 0x80 {
			fast = false
			break
		}
		if mapped, ok := t.gpt2Decoder[rune(b)]; !ok || mapped != b {
			fast = false
			break
		}
	}
	if fast {
		return s
	}
	// Pre-size to the input byte length — GPT-2 maps every rune to exactly
	// one byte (the encoder covers all 256 source bytes), so output bytes
	// ≤ input bytes (every multi-byte rune collapses to 1 byte; ASCII
	// runes stay 1:1). One allocation, no geometric growth.
	//
	// AsString wraps the freshly built buffer in a zero-copy string view —
	// the prior `string(buf)` did a full copy.
	buf := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := t.gpt2Decoder[r]; ok {
			buf = append(buf, b)
			continue
		}
		// Non-mapped runes pass through as UTF-8. Encode the rune
		// directly into buf to avoid the intermediate `[]byte(string(r))`
		// double allocation. utf8.EncodeRune writes up to 4 bytes; grow
		// buf inline rather than detouring through a per-rune string.
		var enc [4]byte
		n := utf8EncodeRune(enc[:], r)
		buf = append(buf, enc[:n]...)
	}
	return core.AsString(buf)
}

// utf8EncodeRune writes the UTF-8 encoding of r into p (which must be
// at least 4 bytes) and returns the byte count. Inlined alternative to
// importing unicode/utf8 in this file — the only caller is
// decodeGPT2Bytes's non-mapped-rune fallback, which is effectively
// unreachable for valid GPT-2 input (the encoder maps all 256 source
// bytes) but kept as a safety net.
func utf8EncodeRune(p []byte, r rune) int {
	switch {
	case r < 0x80:
		p[0] = byte(r)
		return 1
	case r < 0x800:
		p[0] = 0xC0 | byte(r>>6)
		p[1] = 0x80 | (byte(r) & 0x3F)
		return 2
	case r < 0x10000:
		p[0] = 0xE0 | byte(r>>12)
		p[1] = 0x80 | (byte(r>>6) & 0x3F)
		p[2] = 0x80 | (byte(r) & 0x3F)
		return 3
	default:
		p[0] = 0xF0 | byte(r>>18)
		p[1] = 0x80 | (byte(r>>12) & 0x3F)
		p[2] = 0x80 | (byte(r>>6) & 0x3F)
		p[3] = 0x80 | (byte(r) & 0x3F)
		return 4
	}
}

// BOSToken returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOSToken() int32 { return t.bosToken }

// EOSToken returns the end-of-sequence (generation stop) token ID.
func (t *Tokenizer) EOSToken() int32 { return t.eosToken }

// HasBOSToken reports whether the tokenizer explicitly defines a BOS token.
func (t *Tokenizer) HasBOSToken() bool { return t != nil && t.hasBOS }

// HasEOSToken reports whether the tokenizer explicitly defines an EOS/stop token.
func (t *Tokenizer) HasEOSToken() bool { return t != nil && t.hasEOS }

// BOS returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOS() int32 { return t.BOSToken() }

// EOS returns the end-of-sequence (generation stop) token ID.
func (t *Tokenizer) EOS() int32 { return t.EOSToken() }

// TokenID looks up a token string in the vocabulary.
func (t *Tokenizer) TokenID(text string) (int32, bool) {
	id, ok := t.vocab[text]
	return id, ok
}

// IDToken looks up the text for a token ID.
func (t *Tokenizer) IDToken(id int32) string {
	return t.invVocab[id]
}

// FormatGemmaPrompt applies the Gemma 3 chat template.
func FormatGemmaPrompt(prompt string) string {
	return core.Sprintf("<bos><start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}
