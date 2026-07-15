// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"strconv"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	coreio "dappco.re/go/io"
)

// bench_test.go hunts the engine-neutral serving adapter's hot surface: the
// per-token decode emit loop, the per-turn chat render + stop-set assembly, and
// the per-request reuse-lane acquire. Each benchmark drives real inputs (a
// multi-turn conversation with a system turn and thinking on/off, a token
// stream through a fake session) and reports allocations, so a fix is measured
// rather than assumed. The fakes come from model_test.go / prompt_reuse_test.go
// (same package) so a bench exercises the genuine wiring.

// Sinks defeat dead-code elimination — a benchmarked pure function whose result
// is dropped can be optimised away entirely.
var (
	benchStr   string
	benchIDs   []int32
	benchStops []int32
	benchBool  bool
	benchInt   int
	benchTmpl  ChatTemplate
)

// benchLoadTokenizer loads a tokenizer.json fixture through the production
// LoadTokenizer path under a bench temp dir — the TokenID/EOS resolution the
// decode-only NewForDecode path cannot provide.
func benchLoadTokenizer(tb testing.TB, js string) *tokenizer.Tokenizer {
	tb.Helper()
	path := core.JoinPath(tb.TempDir(), "tokenizer.json")
	if err := coreio.Local.Write(path, js); err != nil {
		tb.Fatalf("write bench tokenizer: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		tb.Fatalf("load bench tokenizer: %v", err)
	}
	return tok
}

// benchConversation builds a realistic multi-turn chat: a system turn plus
// exchanges user/assistant pairs of sentence-length content — the shape the
// per-turn render loop frames for every request.
func benchConversation(exchanges int) []inference.Message {
	msgs := make([]inference.Message, 0, 1+exchanges*2)
	msgs = append(msgs, inference.Message{Role: "system", Content: "You are a careful assistant. Answer concisely, show your working, and name the relevant primitive when you refer to one."})
	for i := 0; i < exchanges; i++ {
		msgs = append(msgs, inference.Message{Role: "user", Content: "Walk me through how the budget-triggered context fold keeps a retained conversation inside the window without a hard error."})
		msgs = append(msgs, inference.Message{Role: "assistant", Content: "When an appended turn would leave less than the reply headroom, the handle keeps token zero plus the newest transcript suffix and re-prefills kept plus turn in one engine call."})
	}
	return msgs
}

// --- chat_template.go ------------------------------------------------------

// BenchmarkRenderChatTemplate is the per-turn render loop: frame a whole
// conversation as a fresh chat prompt across the shipped dialects and thinking
// flags. Per request, and the load-bearing serve-path allocator.
func BenchmarkRenderChatTemplate(b *testing.B) {
	msgs := benchConversation(6)
	gemma4 := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, true)
	chatml := ChatTemplate{
		Open: "<|im_start|>", Close: "<|im_end|>",
		UserRole: "user", AssistantRole: "assistant", SystemRole: "system",
		Thinking: &ChatThinking{OffSuffix: "<think>\n\n</think>\n\n"},
	}
	on, off := true, false
	cases := []struct {
		name string
		t    ChatTemplate
		et   *bool
	}{
		{"gemma4_think_on", gemma4, &on},
		{"gemma4_think_off_suppressor", gemma4, &off},
		{"gemma4_no_flag", gemma4, nil},
		{"chatml_think_off", chatml, &off},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				benchStr = renderChatTemplate(tc.t, msgs, tc.et)
			}
		})
	}
}

// BenchmarkRenderChatTurns is the continuation/append render (plain turns, no
// leading-system fold, no thinking framing) — the woken-session tail frame.
func BenchmarkRenderChatTurns(b *testing.B) {
	msgs := benchConversation(6)
	gemma4 := GemmaChatTemplate(TurnTokens{Open: "<|turn>", Close: "<turn|>"}, true)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = renderChatTurns(gemma4, msgs)
	}
}

// BenchmarkGemmaChatTemplate builds the gemma dialect descriptor — the fallback
// path and formatChatPrompt build it fresh (a *ChatThinking on gemma4).
func BenchmarkGemmaChatTemplate(b *testing.B) {
	turns := TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchTmpl = GemmaChatTemplate(turns, true)
	}
}

// --- model.go --------------------------------------------------------------

// BenchmarkTextModel_stopTokens is the per-generation stop-set assembly. The
// empty-cfg sub-bench is the common serve case (no request stop tokens); the
// custom-cfg sub-bench is the caller-supplied-stops slow path.
func BenchmarkTextModel_stopTokens(b *testing.B) {
	tok := benchLoadTokenizer(b, gemma4FixtureTokenizerJSON)
	m := NewTextModel(stopDeclarerTokenModel{stops: []int32{2, 106, 50}}, tok, "gemma4", inference.ModelInfo{}, 4096)
	b.Run("empty_cfg", func(b *testing.B) {
		cfg := inference.GenerateConfig{}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchStops = m.stopTokens(cfg)
		}
	})
	b.Run("custom_cfg", func(b *testing.B) {
		cfg := inference.GenerateConfig{StopTokens: []int32{7, 8, 9}}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchStops = m.stopTokens(cfg)
		}
	})
}

// BenchmarkTextModel_decodeFromPrefilled is the per-token decode emit loop over
// an already-prefilled session: stop-set membership + decode + yield for every
// token, on both the greedy and sampled paths. The stream ids never hit the
// stop set, so each op decodes the full budget.
func BenchmarkTextModel_decodeFromPrefilled(b *testing.B) {
	const n = 128
	inv := make(map[int32]string, n)
	stream := make([]int32, n)
	for i := 0; i < n; i++ {
		id := int32(1000 + i)
		stream[i] = id
		inv[id] = "▁word" + strconv.Itoa(i) // ▁ boundary so DecodeToken does real word work
	}
	m := &TextModel{tok: tokenizer.NewForDecode(inv), tm: &fakeTokenModel{}, maxLen: 1 << 20}
	ctx := context.Background()
	start := time.Now()
	noop := func(inference.Token) bool { return true }
	run := func(b *testing.B, cfg inference.GenerateConfig) {
		sess := &fakeSession{genIDs: stream}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sess.pos = n // reset retained position so each op decodes the full stream
			m.decodeFromPrefilled(ctx, sess, n, cfg, start, noop)
		}
	}
	b.Run("greedy", func(b *testing.B) {
		run(b, inference.GenerateConfig{MaxTokens: n, StopTokens: []int32{2, 106, 50}})
	})
	b.Run("sampled", func(b *testing.B) {
		run(b, inference.GenerateConfig{MaxTokens: n, StopTokens: []int32{2, 106, 50}, Temperature: 0.7, TopK: 40})
	})
}

// BenchmarkTokenInSet measures the per-token stop-set membership scan at
// realistic (3) and adversarial (64) set sizes — the linear-vs-map decision.
// The lookups miss, so each is a full scan (the worst case per token).
func BenchmarkTokenInSet(b *testing.B) {
	small := []int32{2, 106, 50}
	large := make([]int32, 64)
	for i := range large {
		large[i] = int32(i * 3)
	}
	b.Run("small_3_miss", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchBool = tokenInSet(999, small)
		}
	})
	b.Run("large_64_miss", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchBool = tokenInSet(9999, large)
		}
	})
}

// BenchmarkTextModel_FormatChatPrompt is the fresh-prompt render wrapper the
// durable -state loop frames a first turn through.
func BenchmarkTextModel_FormatChatPrompt(b *testing.B) {
	m := NewTextModel(&fakeTokenModel{}, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 4096)
	msgs := benchConversation(6)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = m.FormatChatPrompt(msgs)
	}
}

// BenchmarkTextModel_FormatChatContinuation is the woken-session continuation
// render (close the open model turn, append the new turn, reopen the cue).
func BenchmarkTextModel_FormatChatContinuation(b *testing.B) {
	m := NewTextModel(&fakeTokenModel{}, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 4096)
	msgs := []inference.Message{{Role: "user", Content: "and how does that interact with a fork?"}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = m.FormatChatContinuation(msgs)
	}
}

// BenchmarkTextModel_FormatChatContinuationWithThinking is the durable -state
// per-request continuation render honouring thinking mode. The think_off
// sub-bench runs the suppressor-tail branch (the pre-closed thought channel the
// plain FormatChatContinuation bench never exercises); think_on runs the
// no-tail branch, byte-identical to FormatChatContinuation.
func BenchmarkTextModel_FormatChatContinuationWithThinking(b *testing.B) {
	m := NewTextModel(thoughtSuppressorTokenModel{suppressor: true}, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 4096)
	msgs := []inference.Message{{Role: "user", Content: "and how does that interact with a fork?"}}
	on, off := true, false
	b.Run("think_off_suppressor", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchStr = m.FormatChatContinuationWithThinking(msgs, &off)
		}
	})
	b.Run("think_on", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchStr = m.FormatChatContinuationWithThinking(msgs, &on)
		}
	})
}

// BenchmarkFormatChatPrompt is the gemma-flavoured full-prompt helper the
// multimodal path and the format goldens drive directly.
func BenchmarkFormatChatPrompt(b *testing.B) {
	turns := TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	msgs := benchConversation(6)
	off := false
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = formatChatPrompt(turns, msgs, &off, true)
	}
}

// BenchmarkTextModel_encode is the per-request tokenise of a rendered prompt.
func BenchmarkTextModel_encode(b *testing.B) {
	m := NewTextModel(&fakeTokenModel{}, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 4096)
	prompt := m.FormatChatPrompt(benchConversation(6))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchIDs = m.encode(prompt)
	}
}

// --- session.go ------------------------------------------------------------

// BenchmarkSessionHandle_Prefill is the per-conversation prefill (tokenise +
// copy + store), replacing retained state each call.
func BenchmarkSessionHandle_Prefill(b *testing.B) {
	m := NewTextModel(&fakeTokenModel{}, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 1<<20)
	h := NewSessionHandle(m, &fakeSession{})
	prompt := m.FormatChatPrompt(benchConversation(4))
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = h.Prefill(ctx, prompt)
	}
}

// BenchmarkSessionHandle_AppendPrompt is the per-turn append (tokenise the new
// turn + extend retained state, no replay). The transcript is reset to a fixed
// prefix each op so the fold never triggers and the steady per-turn cost is
// what is measured.
func BenchmarkSessionHandle_AppendPrompt(b *testing.B) {
	m := NewTextModel(&fakeTokenModel{}, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 1<<20)
	sess := &fakeSession{}
	h := NewSessionHandle(m, sess)
	ctx := context.Background()
	if err := h.Prefill(ctx, m.FormatChatPrompt(benchConversation(2))); err != nil {
		b.Fatalf("seed prefill: %v", err)
	}
	seed := append([]int32(nil), h.tokens...)
	seedPos := sess.pos
	turn := m.FormatChatContinuation([]inference.Message{{Role: "user", Content: "one more follow-up question about the fold headroom"}})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		h.tokens = append(h.tokens[:0], seed...)
		sess.pos = seedPos
		sess.appendCalls = sess.appendCalls[:0]
		_ = h.AppendPrompt(ctx, turn)
	}
}

// BenchmarkSessionHandle_Generate is the per-turn retained-session generate
// loop (stop-set assembly + emit over the stream), transcript reset each op.
func BenchmarkSessionHandle_Generate(b *testing.B) {
	stream := make([]int32, 64)
	for i := range stream {
		stream[i] = int32(1 + i%4) // ids spTok decodes to real word pieces
	}
	m := &TextModel{tok: spTok(), tm: &fakeTokenModel{}, maxLen: 1 << 20}
	sess := &fakeSession{genIDs: stream}
	h := NewSessionHandle(m, sess)
	const seedPos = 8
	cfg := inference.GenerateConfig{MaxTokens: len(stream), StopTokens: []int32{99}}
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sess.pos = seedPos
		h.tokens = make([]int32, seedPos)
		h.generated = nil
		for range h.Generate(ctx, cfg) { //nolint:revive // drain the stream
		}
	}
}

// --- session_fold.go ------------------------------------------------------

// BenchmarkSessionHandle_foldForAppendLocked is the budget-triggered fold: keep
// BOS + newest suffix and re-prefill kept+turn (per fold, rare). The transcript
// and the fake's recording slice are reset each op so only the fold's own
// allocation is measured.
func BenchmarkSessionHandle_foldForAppendLocked(b *testing.B) {
	m := &TextModel{tm: &fakeTokenModel{}, maxLen: 4096}
	sess := &fakeSession{}
	h := NewSessionHandle(m, sess)
	base := make([]int32, 4000)
	for i := range base {
		base[i] = int32(i)
	}
	ids := make([]int32, 32)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		h.tokens = append(h.tokens[:0], base...)
		sess.pos = len(base)
		sess.prefillCalls = sess.prefillCalls[:0]
		if _, err := h.foldForAppendLocked(ids); err != nil {
			b.Fatalf("fold: %v", err)
		}
	}
}

// --- prompt_reuse.go ------------------------------------------------------

// BenchmarkTextModel_acquireReuseSession is the per-request reuse-lane acquire
// fast path: the enabled/continuity guards, the capability assertion, the
// TryLock, and the release closure. The resident session is primed first, so
// the loop measures the steady acquire/release, not the one-off session open.
func BenchmarkTextModel_acquireReuseSession(b *testing.B) {
	rs := &reuseFakeSession{fakeSession: fakeSession{genIDs: []int32{10}}}
	tm := &reuseFakeTokenModel{sessions: []*reuseFakeSession{rs}}
	m := NewTextModel(tm, benchLoadTokenizer(b, gemma4FixtureTokenizerJSON), "gemma4", inference.ModelInfo{}, 4096)
	_, release, ok := m.acquireReuseSession()
	if !ok {
		b.Fatal("reuse lane did not engage during priming")
	}
	release()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, rel, ok := m.acquireReuseSession(); ok {
			rel()
		}
	}
}

// --- vision.go -------------------------------------------------------------

// BenchmarkCountTokenID is the placeholder-run check: a linear count over the
// prompt ids (per multimodal request, over a full prompt-length id slice).
func BenchmarkCountTokenID(b *testing.B) {
	ids := make([]int32, 2048)
	for i := range ids {
		ids[i] = int32(i % 300)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchInt = countTokenID(ids, 42)
	}
}
