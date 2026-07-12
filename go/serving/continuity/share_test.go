// SPDX-Licence-Identifier: EUPL-1.2

package continuity

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/prefixindex"
	"dappco.re/go/inference/model/spine"
	state "dappco.re/go/inference/model/state"
)

// --- fakes: a plain-Go engine surface for the cross-conversation share path ---

// mapTokenizer is the inference.PromptTokenizer the manager probes: it maps a
// framed prompt string to a fixed token slice so a test controls exactly which
// tokens two conversations share.
type mapTokenizer struct{ m map[string][]int32 }

func (t mapTokenizer) Tokenize(text string) ([]int32, error) {
	if toks, ok := t.m[text]; ok {
		return toks, nil
	}
	return nil, core.NewError("mapTokenizer: no tokens seeded for " + text)
}

var _ inference.PromptTokenizer = mapTokenizer{}

// markerFormatter frames prompts as deterministic marker strings the
// mapTokenizer keys on — byte-stable so acquire's match framing and finishTurn's
// publish framing agree.
type markerFormatter struct{}

func (markerFormatter) FormatChatPromptWithThinking(msgs []inference.Message, _ *bool) string {
	return "PROMPT:" + joinMsgs(msgs)
}

func (markerFormatter) FormatChatContinuationWithThinking(msgs []inference.Message, _ *bool) string {
	return "CONT:" + joinMsgs(msgs)
}

func joinMsgs(msgs []inference.Message) string {
	b := core.NewBuilder()
	for _, m := range msgs {
		b.WriteString(m.Role)
		b.WriteString("/")
		b.WriteString(m.Content)
		b.WriteString(";")
	}
	return b.String()
}

// graftHandle records the calls the share path makes and implements the token
// append + KV-block restore capabilities the session machinery probes. A
// non-nil kvSnap makes CaptureKV/RangeKVBlocks succeed (so a real sleep can
// publish); nil makes them decline (a session that only ever consumes).
type graftHandle struct {
	kvSnap         *kv.Snapshot
	genTokens      []inference.Token
	prefillPrompt  string
	prefillCalls   int
	appendPrompt   string
	appendTokens   []int32
	appendCalls    int
	restoredTokens int
	closeCalls     int
}

func (h *graftHandle) Prefill(_ context.Context, prompt string) error {
	h.prefillPrompt = prompt
	h.prefillCalls++
	return nil
}
func (h *graftHandle) AppendPrompt(_ context.Context, prompt string) error {
	h.appendPrompt = prompt
	return nil
}
func (h *graftHandle) AppendTokens(_ context.Context, tokens []int32) error {
	h.appendTokens = append([]int32(nil), tokens...)
	h.appendCalls++
	return nil
}
func (h *graftHandle) RestoreKVBlocks(ctx context.Context, source kv.BlockSource) error {
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(ctx, i)
		if err != nil {
			return err
		}
		if end := block.TokenStart + block.TokenCount; end > h.restoredTokens {
			h.restoredTokens = end
		}
		if h.restoredTokens >= source.PrefixTokens {
			break
		}
	}
	return nil
}
func (h *graftHandle) Generate(_ context.Context, _ inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, tok := range h.genTokens {
			if !yield(tok) {
				return
			}
		}
	}
}
func (h *graftHandle) CaptureKV(context.Context) (*kv.Snapshot, error) {
	if h.kvSnap == nil {
		return nil, core.NewError("graftHandle: no capture")
	}
	return h.kvSnap, nil
}
func (h *graftHandle) RangeKVBlocks(_ context.Context, _ int, _ kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if h.kvSnap == nil {
		return core.NewError("graftHandle: no capture")
	}
	_, err := yield(kv.Block{Index: 0, TokenStart: 0, TokenCount: len(h.kvSnap.Tokens), Snapshot: h.kvSnap})
	return err
}
func (h *graftHandle) Fork(context.Context) (inference.SessionHandle, error) {
	return nil, core.NewError("graftHandle: no fork")
}
func (h *graftHandle) Reset()       {}
func (h *graftHandle) Close() error { h.closeCalls++; return nil }
func (h *graftHandle) Err() error   { return nil }

var _ inference.SessionHandle = (*graftHandle)(nil)

// fixedFactory hands the manager the same handle every NewSession, so a test
// holds a reference to the conversation's session and inspects it after.
type fixedFactory struct{ h inference.SessionHandle }

func (f *fixedFactory) NewSession() inference.SessionHandle { return f.h }

// shareManager builds a Manager wired for (or without) cross-conversation
// sharing directly — mirroring what enable() assembles, without a real model.
func shareManager(store *state.InMemoryStore, handle inference.SessionHandle, tok inference.PromptTokenizer, share bool) *Manager {
	m := &Manager{
		factory:   &fixedFactory{h: handle},
		formatter: markerFormatter{},
		store:     store,
		writer:    store,
		info:      spine.ModelInfo{Architecture: "gemma4_text", ContextLength: 4096},
		max:       defaultMaxResident,
		resident:  make(map[string]*residentConversation),
	}
	if tok != nil {
		m.tokenizer = tok
	}
	if share && tok != nil {
		m.sharePrefix = true
		m.prefixIndex = prefixindex.New(prefixindex.Config{MaxEntries: 64})
	}
	return m
}

func seq(start, n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(start + i)
	}
	return out
}

func concatToks(a, b []int32) []int32 {
	out := make([]int32, 0, len(a)+len(b))
	out = append(out, a...)
	return append(out, b...)
}

func equalToks(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// synthSnapshot builds an n-token float32 KV snapshot in the shape the kv block
// walker accepts (one layer, one head, HeadDim 2), for seeding a durable bundle.
func synthSnapshot(n int) *kv.Snapshot {
	key := make([]float32, n*2)
	val := make([]float32, n*2)
	toks := make([]int32, n)
	for i := 0; i < n; i++ {
		toks[i] = int32(i + 1)
		key[2*i], key[2*i+1] = float32(i), float32(i)+0.5
		val[2*i], val[2*i+1] = float32(i)+10, float32(i)+10.5
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        toks,
		Generated:     []int32{toks[n-1]},
		TokenOffset:   n,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        n,
		HeadDim:       2,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads:      []kv.HeadSnapshot{{Key: key, Value: val}},
		}},
	}
}

// seedBundle writes an n-token block bundle to the store at base+"/bundle" and
// returns that URI — a synthetic conversation's durable KV for the graft path.
func seedBundle(t *testing.T, ctx context.Context, store *state.InMemoryStore, base string, n, blockSize int) string {
	t.Helper()
	bundle, err := synthSnapshot(n).SaveStateBlocks(ctx, store, kv.StateBlockOptions{
		BlockSize:  blockSize,
		KVEncoding: kv.EncodingQ8,
		URI:        base + "/blocks",
	})
	if err != nil {
		t.Fatalf("seed SaveStateBlocks: %v", err)
	}
	bundleURI := base + "/bundle"
	if _, err := kv.SaveStateBlockBundle(ctx, store, bundle, bundleURI); err != nil {
		t.Fatalf("seed SaveStateBlockBundle: %v", err)
	}
	return bundleURI
}

func drain(seq iter.Seq[inference.Token]) {
	for range seq {
	}
}

// --- receipts ---

// TestManagerChat_ShareGraft_Good is the headline serving receipt: conversation
// B, opening with the same system prompt as an already-served conversation A,
// wakes A's shared token span from A's durable bundle and prefills ONLY its
// divergent tail — no second full prefill. The recorded restore span and append
// tail are the token-count receipt.
func TestManagerChat_ShareGraft_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	const blockSize = 4

	// A's durable KV: a 24-token bundle at block size 4.
	bundleA := seedBundle(t, ctx, store, "state://a", 24, blockSize)

	// A and B share a 20-token system+user prefix, then diverge.
	shared := seq(1, 20)
	tokensA := concatToks(shared, seq(101, 4)) // len 24
	tokensB := concatToks(shared, seq(201, 8)) // len 28

	tok := mapTokenizer{m: map[string][]int32{}}
	bHandle := &graftHandle{genTokens: []inference.Token{{Text: "ok"}}}
	m := shareManager(store, bHandle, tok, true)

	// A published its shareable prefix when it finished (seeded directly here).
	m.prefixIndex.Publish(tokensA, prefixindex.Entry{BundleURI: bundleA, BlockSize: blockSize, TokenCount: len(tokensA)})

	msgsB := []inference.Message{{Role: "system", Content: "CONTENT"}, {Role: "user", Content: "BBB"}}
	tok.m["PROMPT:"+joinMsgs(msgsB)] = tokensB

	streamed, ok := m.Chat(ctx, msgsB, inference.WithMaxTokens(4))
	if !ok {
		t.Fatal("B declined; a shared system prefix must be grafted, not re-prefilled")
	}
	drain(streamed)

	if s := m.Stats(); s.SharedGrafts != 1 {
		t.Fatalf("stats = %+v, want exactly one SharedGraft", s)
	}
	if bHandle.prefillCalls != 0 {
		t.Fatalf("B ran %d full prefill(s) (prompt %q); the graft must append only the tail", bHandle.prefillCalls, bHandle.prefillPrompt)
	}
	// span = align_down(min(matched 20, bundle 24), 4) = 20.
	if bHandle.restoredTokens != 20 {
		t.Fatalf("restored %d tokens, want the 20-token shared span", bHandle.restoredTokens)
	}
	wantTail := tokensB[20:]
	if !equalToks(bHandle.appendTokens, wantTail) {
		t.Fatalf("appended %v, want only the divergent tail %v", bHandle.appendTokens, wantTail)
	}
	// The receipt: 8 tail tokens prefilled instead of the whole 28-token prompt.
	if len(bHandle.appendTokens) >= len(tokensB) {
		t.Fatalf("no prefill saving: appended %d of %d prompt tokens", len(bHandle.appendTokens), len(tokensB))
	}
}

// TestManagerChat_ShareGraft_StaleFallback is the safety rail: a matched index
// entry whose backing bundle is gone must fall back to a fresh full prefill —
// never a wrong graft — and evict the dead entry so it is not retried.
func TestManagerChat_ShareGraft_StaleFallback(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	tok := mapTokenizer{m: map[string][]int32{}}
	bHandle := &graftHandle{genTokens: []inference.Token{{Text: "ok"}}}
	m := shareManager(store, bHandle, tok, true)

	// An entry pointing at a bundle that was never written (reclaimed state).
	tokensA := seq(1, 24)
	m.prefixIndex.Publish(tokensA, prefixindex.Entry{BundleURI: "state://missing/bundle", BlockSize: 4, TokenCount: len(tokensA)})

	tokensB := concatToks(seq(1, 20), seq(201, 8))
	msgsB := []inference.Message{{Role: "system", Content: "CONTENT"}, {Role: "user", Content: "BBB"}}
	tok.m["PROMPT:"+joinMsgs(msgsB)] = tokensB

	streamed, ok := m.Chat(ctx, msgsB, inference.WithMaxTokens(4))
	if !ok {
		t.Fatal("a stale entry must fall back to a fresh prefill, not decline to stateless")
	}
	drain(streamed)

	if s := m.Stats(); s.SharedGrafts != 0 || s.FreshConversations != 1 {
		t.Fatalf("stats = %+v, want zero grafts and one fresh conversation", s)
	}
	if bHandle.prefillCalls != 1 || bHandle.prefillPrompt != "PROMPT:"+joinMsgs(msgsB) {
		t.Fatalf("B did not fresh-prefill after the stale fallback: calls=%d prompt=%q", bHandle.prefillCalls, bHandle.prefillPrompt)
	}
	if bHandle.appendCalls != 0 {
		t.Fatalf("B token-appended %d times after a fallback, want 0 (fresh prefill only)", bHandle.appendCalls)
	}
	if _, _, ok := m.prefixIndex.Match(tokensB); ok {
		t.Fatal("the dead index entry survived the failed wake; it must be evicted")
	}
}

// TestManagerChat_ShareOff_Untouched proves the flag-off path is byte-identical:
// with sharing disabled the index is never built, no graft is attempted, and B
// runs a plain full prefill exactly as it does today.
func TestManagerChat_ShareOff_Untouched(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	seedBundle(t, ctx, store, "state://a", 24, 4) // a bundle exists, but must be ignored

	tok := mapTokenizer{m: map[string][]int32{}}
	bHandle := &graftHandle{genTokens: []inference.Token{{Text: "ok"}}}
	m := shareManager(store, bHandle, tok, false) // sharing OFF

	if m.prefixIndex != nil || m.sharePrefix {
		t.Fatal("sharing off must leave the index unbuilt and the flag clear")
	}

	tokensB := concatToks(seq(1, 20), seq(201, 8))
	msgsB := []inference.Message{{Role: "system", Content: "CONTENT"}, {Role: "user", Content: "BBB"}}
	tok.m["PROMPT:"+joinMsgs(msgsB)] = tokensB

	streamed, ok := m.Chat(ctx, msgsB, inference.WithMaxTokens(4))
	if !ok {
		t.Fatal("B declined with sharing off")
	}
	drain(streamed)

	if s := m.Stats(); s.SharedGrafts != 0 || s.FreshConversations != 1 {
		t.Fatalf("stats = %+v, want no grafts and one fresh conversation with sharing off", s)
	}
	if bHandle.prefillCalls != 1 || bHandle.appendCalls != 0 {
		t.Fatalf("with sharing off B must run one plain prefill and no token append: prefill=%d append=%d", bHandle.prefillCalls, bHandle.appendCalls)
	}
}

// TestManagerChat_SharePublish_Good drives a fresh conversation's REAL sleep and
// proves finishTurn publishes its framed prompt as a shareable prefix, keyed to
// the bundle the sleep actually wrote — the produce side the graft consumes.
func TestManagerChat_SharePublish_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	tok := mapTokenizer{m: map[string][]int32{}}
	// A capturing handle so the sleep succeeds and the bundle is durable.
	aHandle := &graftHandle{kvSnap: synthSnapshot(12), genTokens: []inference.Token{{Text: "hi"}}}
	m := shareManager(store, aHandle, tok, true)

	tokensA := seq(1, 12)
	msgsA := []inference.Message{{Role: "system", Content: "CONTENT"}, {Role: "user", Content: "AAA"}}
	tok.m["PROMPT:"+joinMsgs(msgsA)] = tokensA

	streamed, ok := m.Chat(ctx, msgsA, inference.WithMaxTokens(4))
	if !ok {
		t.Fatal("A declined; a fresh [system,user] turn must be served")
	}
	drain(streamed)

	if s := m.Stats(); s.FreshConversations != 1 || s.Sleeps != 1 {
		t.Fatalf("stats = %+v, want one fresh conversation that slept", s)
	}
	entry, matched, ok := m.prefixIndex.Match(tokensA)
	if !ok || matched != len(tokensA) {
		t.Fatalf("A's prefix was not published: match = (%d, %v)", matched, ok)
	}
	if entry.BundleURI == "" || entry.TokenCount != len(tokensA) {
		t.Fatalf("published entry = %+v, want A's bundle URI and its %d prompt tokens", entry, len(tokensA))
	}
}
