// SPDX-Licence-Identifier: EUPL-1.2

package model

import "slices"

import core "dappco.re/go"

// The token-loop contract — the rung above Backend that turns a backend's
// hidden-state decode into real tokens. Backend.DecodeForward runs the
// transformer stack (hidden → hidden); the two bookends here close the loop:
// Embedder maps a token id to its input hidden vector, LMHead maps a final
// hidden state to vocab logits. A backend that provides all three is a
// TokenModel, and Generate drives the full token-in → token-out loop over it —
// once, in pure Go, for every backend (native, metal, the future rocm). No
// backend re-hand-rolls the generation loop: it supplies the byte-level pieces
// and inherits generation + sampling from here.
//
// Everything crosses the seam as bf16 []byte — the lingua franca QuantMatVec and
// Backend already use. An embedding is dModel bf16 bytes; logits are vocab bf16
// bytes. The model binds its arch (vocab, hidden size, embed scale, eps,
// soft-cap) at construction, so these methods carry only the per-call data.

// Embedder maps a token id to its input embedding: dModel bf16 bytes, already
// scaled (an arch may scale the table row by sqrt(hidden)). The input bookend.
type Embedder interface {
	Embed(id int32) ([]byte, error)
}

// LMHead maps a final hidden state (dModel bf16 bytes) to vocab logits (vocab
// bf16 bytes): final norm + output projection + the optional monotonic
// soft-cap. The output bookend.
type LMHead interface {
	Head(hidden []byte) ([]byte, error)
}

// TokenModel is a backend that provides the whole token → token path: the two
// bookends plus the hidden-state decode (Backend), and the vocab size that sizes
// the logits Greedy/Sample read. native/metal/rocm each construct one;
// Generate/GenerateSampled drive it. This is the minimal surface a backend MUST
// provide — the whole-sequence path. A backend that can ALSO decode
// incrementally over a persistent cache additionally implements SessionModel.
type TokenModel interface {
	Embedder
	Backend
	LMHead
	Vocab() int
}

// DecodeStepper is stateful single-token decode over a persistent KV cache: the
// cache is built when the stepper is opened and carries across Step calls, so a
// decode costs O(1) per token vs Backend.DecodeForward's whole-sequence O(n²)
// rebuild. Returned by SessionModel.OpenSession.
//
// A stepper MAY additionally implement `Close() error` (io.Closer shape): Generate
// closes the stepper it opens when generation finishes. A manual-memory backend
// (metal/rocm KV caches) implements it to release those resources; a backend whose
// session resources are GC-managed (native's retained buffers) need not.
//
// A stepper whose decode needs the token id itself — not just its embedding — MAY
// implement `StepWithID(id int32, emb []byte) ([]byte, error)`: Generate calls it in
// preference to Step, passing both. Some archs need it (e.g. per-layer inputs that are
// gathered from the id, not derivable from the embedding); for every other model
// StepWithID is just Step with the id ignored.
type DecodeStepper interface {
	// Step decodes one token embedding (the model's activation dtype bytes) at the
	// next cache position, appends its K/V to the persistent cache, and returns the
	// output hidden state (same shape).
	Step(emb []byte) ([]byte, error)
}

// BatchPrefillStepper is the OPTIONAL capability of a DecodeStepper whose backend can forward more
// than one token through the persistent cache in a single call — an L>1 forward path alongside the
// L==1 Step contract (e.g. a recurrent/attention mixer stack that already batches prefill this way
// internally). generateStepwiseWithSession prefers it for the PROMPT: without it, the prefill walk
// calls Step once per prompt token, so any device fusion bound to Step's own last-layer tail (e.g. a
// head GEMM folded onto the final layer's command buffer) fires once per PREFILL token even though
// only the LAST token's result is ever read — bounded waste that is invisible on a short prompt and
// real on a long one. PrefillBatch collapses the whole prompt into the ONE call the backend's L>1
// path already supports, so that fusion runs once for the batch, not once per token. Mutually
// exclusive with StepWithID (PrefillBatch carries no ids): a stepper needing per-token ids keeps the
// per-token loop even if it also implements this. A stepper without an L>1 forward path simply
// doesn't implement BatchPrefillStepper; nothing about it changes.
type BatchPrefillStepper interface {
	// PrefillBatch runs the whole of embs through the persistent cache in one forward pass — the
	// same state transition len(embs) sequential Step calls would produce — and returns the LAST
	// token's output hidden state (same shape Step returns), so the caller's post-prefill tail
	// (head → pick → append) is unchanged.
	PrefillBatch(embs [][]byte) ([]byte, error)
}

// SessionModel is a TokenModel whose backend can decode incrementally over a
// persistent cache — the OPTIONAL fast path. Generate prefers OpenSession's
// incremental stepper (O(1)/token) over the whole-sequence DecodeForward when a
// model provides it; a backend that can't maintain a persistent cache simply
// doesn't implement SessionModel and gets the whole-sequence loop, no contract
// change. Additive by design: the Backend/TokenModel surface a backend must
// implement is unchanged, so this rung never disrupts a backend mid-port.
type SessionModel interface {
	TokenModel
	// OpenSession opens a fresh stateful decode stepper with an empty cache (a
	// new generation starts at position 0).
	OpenSession() (DecodeStepper, error)
}

type greedySessionGenerator interface {
	Generate(promptIDs []int32, maxNew, eosID int) ([]int32, error)
}

type greedySessionOneShotGenerator interface {
	GenerateOneShot(promptIDs []int32, maxNew, eosID int) ([]int32, error)
}

type sampledSessionGenerator interface {
	GenerateSampledEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *Sampler, params SampleParams, transform TokenTransform, yield func(int32) bool) ([]int32, error)
}

type sampledSessionOneShotGenerator interface {
	GenerateSampledOneShotEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *Sampler, params SampleParams, transform TokenTransform, yield func(int32) bool) ([]int32, error)
}

// TokenTransform observes the selected token ID and returns the ID that should
// be committed into the generation state before stop checks and the next
// decode step.
type TokenTransform func(int32) int32

// generate is the shared validation + dispatch: it picks the incremental
// persistent-cache path (SessionModel) when the model offers it, else the
// whole-sequence fallback. pick is the only difference between greedy and
// sampled generation.
func generate(m TokenModel, promptIDs []int32, maxNew, eos int, pick func(logits []byte, vocab int) (int32, error)) ([]int32, error) {
	return generateUntilTransform(m, promptIDs, maxNew, singleStop(eos), nil, nil, pick, nil)
}

func generateUntil(m TokenModel, promptIDs []int32, maxNew int, stop func(int32) bool, pick func(logits []byte, vocab int) (int32, error)) ([]int32, error) {
	return generateUntilTransform(m, promptIDs, maxNew, stop, nil, nil, pick, nil)
}

type logitsTransform func(logits []byte, vocab int, history []int32) ([]byte, error)

// generateUntilTransform threads an optional per-token yield (nil = batch). When
// non-nil, each committed token is handed to yield as it is generated — the
// streaming sibling of the batch return, byte-identical in the tokens produced.
// yield returning false ends generation early (the consumer is done, e.g. ctx
// cancelled), the same early-out the stop set already provides.
func generateUntilTransform(m TokenModel, promptIDs []int32, maxNew int, stop func(int32) bool, transform logitsTransform, tokenTransform TokenTransform, pick func(logits []byte, vocab int) (int32, error), yield func(int32) bool) ([]int32, error) {
	if m == nil {
		return nil, core.NewError("model.Generate: nil model")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("model.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("model.Generate: maxNew must be > 0")
	}
	if sm, ok := m.(SessionModel); ok {
		return generateStepwise(sm, promptIDs, maxNew, stop, transform, tokenTransform, pick, yield)
	}
	return generateWholeSeq(m, promptIDs, maxNew, stop, transform, tokenTransform, pick, yield)
}

func singleStop(eos int) func(int32) bool {
	if eos < 0 {
		return nil
	}
	return func(id int32) bool { return int(id) == eos }
}

func stopSet(tokens []int32) func(int32) bool {
	if len(tokens) == 0 {
		return nil
	}
	return func(id int32) bool {
		return slices.Contains(tokens, id)
	}
}

// generateStepwise is the incremental path: open a persistent-cache session and
// step one token at a time (embed → Step), the cache carrying across steps so
// each token costs O(1). The decode tail (head → pick → append, eos/maxNew stop)
// is shared with the whole-sequence path in shape.
func generateStepwise(m SessionModel, promptIDs []int32, maxNew int, stop func(int32) bool, transform logitsTransform, tokenTransform TokenTransform, pick func(logits []byte, vocab int) (int32, error), yield func(int32) bool) ([]int32, error) {
	sess, err := m.OpenSession()
	if err != nil {
		return nil, err
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }() // release a manual-memory backend's session resources
	}
	return generateStepwiseWithSession(m, sess, promptIDs, maxNew, stop, transform, tokenTransform, pick, yield)
}

func generateStepwiseWithSession(m SessionModel, sess DecodeStepper, promptIDs []int32, maxNew int, stop func(int32) bool, transform logitsTransform, tokenTransform TokenTransform, pick func(logits []byte, vocab int) (int32, error), yield func(int32) bool) ([]int32, error) {
	vocab := m.Vocab()
	// a backend whose decode needs the token id (e.g. per-layer inputs) gets it via
	// StepWithID; everyone else uses Step (the id is already used to compute the embedding).
	stepID, idAware := sess.(interface {
		StepWithID(id int32, emb []byte) ([]byte, error)
	})
	step := func(id int32) ([]byte, error) {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		if idAware {
			return stepID.StepWithID(id, emb)
		}
		return sess.Step(emb)
	}
	// A stepper MAY implement LMHead itself when its last Step call already produced the next logits as
	// a side effect (a fused device fast path folding the head GEMM onto the final layer's own tail
	// command buffer) — headOf prefers that cached fast path over the model's Head, which always
	// recomputes from scratch. Same optional-capability shape as StepWithID/Close.
	headOf := m.Head
	if sh, ok := sess.(LMHead); ok {
		headOf = sh.Head
	}

	var hidden []byte
	if bp, ok := sess.(BatchPrefillStepper); ok && !idAware {
		// Batch the whole prompt into ONE forward call instead of len(promptIDs) Step calls — see
		// BatchPrefillStepper.
		embs := make([][]byte, len(promptIDs))
		for i, id := range promptIDs {
			emb, err := m.Embed(id)
			if err != nil {
				return nil, err
			}
			embs[i] = emb
		}
		var err error
		if hidden, err = bp.PrefillBatch(embs); err != nil {
			return nil, err
		}
	} else {
		for _, id := range promptIDs { // prefill the prompt over the growing cache
			var err error
			if hidden, err = step(id); err != nil {
				return nil, err
			}
		}
	}
	return decodeFromHiddenWithSession(vocab, hidden, maxNew, stop, transform, tokenTransform, pick, yield, step, headOf)
}

// decodeFromHiddenWithSession is the decode loop shared by every stepwise
// session entry: the caller has already prefilled the session (from token ids
// or from spliced embedding rows) and hands over the LAST prefill row's hidden
// state; the loop heads it into logits, picks, streams, and steps the session
// forward one committed token at a time until maxNew/stop/yield ends it.
func decodeFromHiddenWithSession(vocab int, hidden []byte, maxNew int, stop func(int32) bool, transform logitsTransform, tokenTransform TokenTransform, pick func(logits []byte, vocab int) (int32, error), yield func(int32) bool, step func(int32) ([]byte, error), headOf func([]byte) ([]byte, error)) ([]int32, error) {
	gen, _, _, err := decodeResumeFromHidden(vocab, hidden, maxNew, stop, transform, tokenTransform, pick, yield, step, headOf)
	return gen, err
}

// decodeResumeFromHidden is decodeFromHiddenWithSession's tracked body: alongside the
// generated ids it reports the hidden state after the MOST RECENT step call (the input
// hidden when no step ran) and how many of gen were stepped into the session — the loop
// ends after PICKING its final token without stepping it (stop/budget/yield), so stepped
// is len(gen)-1 on every normal end. A stateful session bridge needs both to continue the
// same session later: the un-stepped tail token is prepended to the next forward, and the
// returned hidden is where the next decode resumes.
func decodeResumeFromHidden(vocab int, hidden []byte, maxNew int, stop func(int32) bool, transform logitsTransform, tokenTransform TokenTransform, pick func(logits []byte, vocab int) (int32, error), yield func(int32) bool, step func(int32) ([]byte, error), headOf func([]byte) ([]byte, error)) ([]int32, []byte, int, error) {
	gen := make([]int32, 0, maxNew)
	history := make([]int32, 0, maxNew)
	stepped := 0
	for len(gen) < maxNew {
		logits, err := headOf(hidden) // the last token's state drives the next id
		if err != nil {
			return nil, nil, 0, err
		}
		pickLogits := logits
		if transform != nil {
			pickLogits, err = transform(logits, vocab, history)
			if err != nil {
				return nil, nil, 0, err
			}
		}
		next, err := pick(pickLogits, vocab)
		if err != nil {
			return nil, nil, 0, err
		}
		if tokenTransform != nil {
			next = tokenTransform(next)
		}
		gen = append(gen, next)
		if transform != nil {
			history = append(history, next)
		}
		if yield != nil && !yield(next) { // stream the committed token; consumer may end early
			break
		}
		if stop != nil && stop(next) {
			break
		}
		if len(gen) >= maxNew {
			break
		}
		if hidden, err = step(next); err != nil { // cache the generated token too
			return nil, nil, 0, err
		}
		stepped++
	}
	return gen, hidden, stepped, nil
}

// GenerateSampledFromEmbeddingsEach decodes from ALREADY-COMPUTED embedding
// rows instead of token ids — the multimodal prefill entry: a spliced prompt's
// rows (text embeddings with projected image features over the placeholder
// span, [VisionTokenModel-shaped models produce them]) prefill the session in
// ONE batch, then the decode loop continues exactly as
// [GenerateSampledWithStopTokensTransformEach]. ids carries the prompt's token
// identity purely for length validation — rows cannot replay per-id, so the
// opened session must be a [BatchPrefillStepper] and must not demand
// StepWithID for its prefill.
//
//	gen, err := model.GenerateSampledFromEmbeddingsEach(sm, model.NewSampler(0), model.SampleParams{}, ids, rows, 256, stops, nil, yield)
func GenerateSampledFromEmbeddingsEach(m SessionModel, s *Sampler, p SampleParams, ids []int32, embeddings [][]byte, maxNew int, stopTokens []int32, tokenTransform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if s == nil {
		return nil, core.NewError("model.GenerateSampledFromEmbeddings: nil sampler")
	}
	if m == nil {
		return nil, core.NewError("model.GenerateSampledFromEmbeddings: nil model")
	}
	if len(embeddings) == 0 {
		return nil, core.NewError("model.GenerateSampledFromEmbeddings: empty embedding rows")
	}
	if len(ids) != len(embeddings) {
		return nil, core.NewError("model.GenerateSampledFromEmbeddings: token and embedding counts differ")
	}
	if maxNew <= 0 {
		return nil, core.NewError("model.GenerateSampledFromEmbeddings: maxNew must be > 0")
	}
	sess, err := m.OpenSession()
	if err != nil {
		return nil, err
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	bp, ok := sess.(BatchPrefillStepper)
	if !ok {
		return nil, core.NewError("model.GenerateSampledFromEmbeddings: session cannot prefill embedding rows (no BatchPrefillStepper)")
	}
	step := func(id int32) ([]byte, error) {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		return sess.Step(emb)
	}
	headOf := m.Head
	if sh, ok := sess.(LMHead); ok {
		headOf = sh.Head
	}
	hidden, err := bp.PrefillBatch(embeddings)
	if err != nil {
		return nil, err
	}
	generated := 0
	return decodeFromHiddenWithSession(m.Vocab(), hidden, maxNew, stopSet(stopTokens), repeatPenaltyTransform(p), tokenTransform, func(logits []byte, vocab int) (int32, error) {
		pickParams := p
		if p.MinTokensBeforeStop > 0 && generated < p.MinTokensBeforeStop {
			pickParams.SuppressTokens = appendSuppressionTokens(p.SuppressTokens, stopTokens)
		}
		next, err := s.Sample(logits, vocab, pickParams)
		if err == nil {
			generated++
		}
		return next, err
	}, yield, step, headOf)
}

// SessionResume is the state a stateful session bridge carries between
// [GenerateSampledResumeEach] calls on ONE open stepper: the hidden of the last
// token forwarded through the session, and (when the previous decode ended by
// picking a token it never stepped) that pending token id. PendingID < 0 = none.
type SessionResume struct {
	Hidden    []byte
	PendingID int32
}

// GenerateSampledResumeEach continues sampled decoding over an ALREADY-OPEN,
// already-prefilled stepper — the stateful-session entry: no session open and no
// prompt replay, so a caller that keeps sess and the returned SessionResume pays
// only the new tokens on every turn. r.Hidden is where decode resumes (the last
// forwarded token's output); a PendingID ≥ 0 is stepped first (the previous
// decode's picked-but-unstepped final token). The sampler wiring — repeat
// penalty, MinTokensBeforeStop suppression — matches
// [GenerateSampledWithStopTokensTransformEach] exactly, so a resumed decode and
// a whole-prefix replay pick identical tokens at temp 0. Returns the generated
// ids and the SessionResume for the next call.
func GenerateSampledResumeEach(m SessionModel, sess DecodeStepper, r SessionResume, s *Sampler, p SampleParams, maxNew int, stopTokens []int32, tokenTransform TokenTransform, yield func(int32) bool) ([]int32, SessionResume, error) {
	if s == nil {
		return nil, r, core.NewError("model.GenerateSampledResume: nil sampler")
	}
	if m == nil {
		return nil, r, core.NewError("model.GenerateSampledResume: nil model")
	}
	if sess == nil {
		return nil, r, core.NewError("model.GenerateSampledResume: nil session")
	}
	if len(r.Hidden) == 0 {
		return nil, r, core.NewError("model.GenerateSampledResume: no resume hidden (session not prefilled)")
	}
	if maxNew <= 0 {
		return nil, r, core.NewError("model.GenerateSampledResume: maxNew must be > 0")
	}
	stepID, idAware := sess.(interface {
		StepWithID(id int32, emb []byte) ([]byte, error)
	})
	step := func(id int32) ([]byte, error) {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		if idAware {
			return stepID.StepWithID(id, emb)
		}
		return sess.Step(emb)
	}
	headOf := m.Head
	if sh, ok := sess.(LMHead); ok {
		headOf = sh.Head
	}
	hidden := r.Hidden
	if r.PendingID >= 0 { // the previous decode's final pick enters the session now
		var err error
		if hidden, err = step(r.PendingID); err != nil {
			return nil, r, err
		}
	}
	generated := 0
	gen, lastHidden, stepped, err := decodeResumeFromHidden(m.Vocab(), hidden, maxNew, stopSet(stopTokens), repeatPenaltyTransform(p), tokenTransform, func(logits []byte, vocab int) (int32, error) {
		pickParams := p
		if p.MinTokensBeforeStop > 0 && generated < p.MinTokensBeforeStop {
			pickParams.SuppressTokens = appendSuppressionTokens(p.SuppressTokens, stopTokens)
		}
		next, err := s.Sample(logits, vocab, pickParams)
		if err == nil {
			generated++
		}
		return next, err
	}, yield, step, headOf)
	if err != nil {
		return nil, r, err
	}
	next := SessionResume{Hidden: lastHidden, PendingID: -1}
	if stepped < len(gen) { // the final pick was never stepped — it is the next call's pending token
		next.PendingID = gen[len(gen)-1]
	}
	return gen, next, nil
}

// generateWholeSeq is the fallback for a backend without a persistent-cache
// session: embed the running sequence, run DecodeForward over it (rebuilding the
// KV cache each call → O(n²)), take the last hidden state, head → pick → append,
// re-embed the generated token, repeat. Correct for any backend; the incremental
// path supersedes it whenever a model implements SessionModel.
func generateWholeSeq(m TokenModel, promptIDs []int32, maxNew int, stop func(int32) bool, transform logitsTransform, tokenTransform TokenTransform, pick func(logits []byte, vocab int) (int32, error), yield func(int32) bool) ([]int32, error) {
	vocab := m.Vocab()
	seq := make([][]byte, 0, len(promptIDs)+maxNew)
	for _, id := range promptIDs {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		seq = append(seq, emb)
	}

	gen := make([]int32, 0, maxNew)
	history := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		hidden, err := m.DecodeForward(seq)
		if err != nil {
			return nil, err
		}
		if len(hidden) == 0 {
			return nil, core.NewError("model.Generate: backend returned no hidden states")
		}
		logits, err := m.Head(hidden[len(hidden)-1])
		if err != nil {
			return nil, err
		}
		pickLogits := logits
		if transform != nil {
			pickLogits, err = transform(logits, vocab, history)
			if err != nil {
				return nil, err
			}
		}
		next, err := pick(pickLogits, vocab)
		if err != nil {
			return nil, err
		}
		if tokenTransform != nil {
			next = tokenTransform(next)
		}
		gen = append(gen, next)
		if transform != nil {
			history = append(history, next)
		}
		if yield != nil && !yield(next) { // stream the committed token; consumer may end early
			break
		}
		if stop != nil && stop(next) {
			break
		}
		if len(gen) >= maxNew {
			break
		}
		emb, err := m.Embed(next) // re-embed the generated token for the next step
		if err != nil {
			return nil, err
		}
		seq = append(seq, emb)
	}
	return gen, nil
}

// Generate greedily decodes up to maxNew tokens from a TokenModel, starting from
// promptIDs; eos < 0 disables early stop. Deterministic (no RNG) — the natural
// closer for a correctness gate or a greedy bench. The contract-level token
// loop: backend-agnostic, pure Go, shared by every backend; incremental over a
// persistent cache when the model provides one (SessionModel), else
// whole-sequence.
func Generate(m TokenModel, promptIDs []int32, maxNew, eos int) ([]int32, error) {
	if m == nil {
		return nil, core.NewError("model.Generate: nil model")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("model.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("model.Generate: maxNew must be > 0")
	}
	if sm, ok := m.(SessionModel); ok {
		sess, err := sm.OpenSession()
		if err != nil {
			return nil, err
		}
		if c, ok := sess.(interface{ Close() error }); ok {
			defer func() { _ = c.Close() }()
		}
		if direct, ok := sess.(greedySessionOneShotGenerator); ok {
			return direct.GenerateOneShot(promptIDs, maxNew, eos)
		}
		if direct, ok := sess.(greedySessionGenerator); ok {
			return direct.Generate(promptIDs, maxNew, eos)
		}
		return generateStepwiseWithSession(sm, sess, promptIDs, maxNew, singleStop(eos), nil, nil, Greedy, nil)
	}
	return generate(m, promptIDs, maxNew, eos, Greedy)
}

// GenerateSampled is Generate with stochastic sampling: the same loop, drawing
// each token from the logits via the Sampler + SampleParams (temperature, then
// optional top-k and top-p) instead of greedy. p.Temperature <= 0 falls back to
// greedy per token (so a zero-temp request is deterministic).
func GenerateSampled(m TokenModel, s *Sampler, p SampleParams, promptIDs []int32, maxNew, eos int) ([]int32, error) {
	if s == nil {
		return nil, core.NewError("model.GenerateSampled: nil sampler")
	}
	var stopTokens []int32
	if eos >= 0 {
		stopTokens = []int32{int32(eos)}
		p.MinTokensBeforeStop = 0
	}
	return GenerateSampledWithStopTokensTransformEach(m, s, p, promptIDs, maxNew, stopTokens, nil, nil)
}

// GenerateSampledWithStopTokens is GenerateSampled with a full stop-token set.
// It matches serving engines that accept more than one stop id: generation stops
// immediately after the first sampled token contained in stopTokens.
func GenerateSampledWithStopTokens(m TokenModel, s *Sampler, p SampleParams, promptIDs []int32, maxNew int, stopTokens []int32) ([]int32, error) {
	return GenerateSampledWithStopTokensTransform(m, s, p, promptIDs, maxNew, stopTokens, nil)
}

// GenerateSampledWithStopTokensTransform is GenerateSampledWithStopTokens with
// a committed-token transform applied before stop checks and before the token is
// fed into the next decode step.
func GenerateSampledWithStopTokensTransform(m TokenModel, s *Sampler, p SampleParams, promptIDs []int32, maxNew int, stopTokens []int32, tokenTransform TokenTransform) ([]int32, error) {
	return GenerateSampledWithStopTokensTransformEach(m, s, p, promptIDs, maxNew, stopTokens, tokenTransform, nil)
}

// GenerateSampledWithStopTokensTransformEach is the streaming sibling of
// GenerateSampledWithStopTokensTransform: each committed token is handed to yield
// as it is sampled, so a serving loop emits incrementally instead of waiting for
// the whole batch. yield == nil is exactly the batch path (byte-identical tokens).
// This is the sampled counterpart to the greedy session's GenerateEach — without
// it the temp>0 decode path returns []int32 all at once and an iterator over it
// reports a zero decode interval.
func GenerateSampledWithStopTokensTransformEach(m TokenModel, s *Sampler, p SampleParams, promptIDs []int32, maxNew int, stopTokens []int32, tokenTransform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if s == nil {
		return nil, core.NewError("model.GenerateSampled: nil sampler")
	}
	if m == nil {
		return nil, core.NewError("model.Generate: nil model")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("model.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("model.Generate: maxNew must be > 0")
	}
	if sm, ok := m.(SessionModel); ok {
		sess, err := sm.OpenSession()
		if err != nil {
			return nil, err
		}
		if c, ok := sess.(interface{ Close() error }); ok {
			defer func() { _ = c.Close() }()
		}
		if direct, ok := sess.(sampledSessionOneShotGenerator); ok {
			return direct.GenerateSampledOneShotEach(promptIDs, maxNew, stopTokens, s, p, tokenTransform, yield)
		}
		if direct, ok := sess.(sampledSessionGenerator); ok {
			return direct.GenerateSampledEach(promptIDs, maxNew, stopTokens, s, p, tokenTransform, yield)
		}
		generated := 0
		return generateStepwiseWithSession(sm, sess, promptIDs, maxNew, stopSet(stopTokens), repeatPenaltyTransform(p), tokenTransform, func(logits []byte, vocab int) (int32, error) {
			pickParams := p
			if p.MinTokensBeforeStop > 0 && generated < p.MinTokensBeforeStop {
				pickParams.SuppressTokens = appendSuppressionTokens(p.SuppressTokens, stopTokens)
			}
			next, err := s.Sample(logits, vocab, pickParams)
			if err == nil {
				generated++
			}
			return next, err
		}, yield)
	}
	generated := 0
	return generateUntilTransform(m, promptIDs, maxNew, stopSet(stopTokens), repeatPenaltyTransform(p), tokenTransform, func(logits []byte, vocab int) (int32, error) {
		pickParams := p
		if p.MinTokensBeforeStop > 0 && generated < p.MinTokensBeforeStop {
			pickParams.SuppressTokens = appendSuppressionTokens(p.SuppressTokens, stopTokens)
		}
		next, err := s.Sample(logits, vocab, pickParams)
		if err == nil {
			generated++
		}
		return next, err
	}, yield)
}

func appendSuppressionTokens(base, tokens []int32) []int32 {
	out := base
	for _, token := range tokens {
		if tokenSuppressed(token, out) {
			continue
		}
		out = append(out, token)
	}
	return out
}

func repeatPenaltyTransform(p SampleParams) logitsTransform {
	if p.RepeatPenalty <= 1 {
		return nil
	}
	return func(logits []byte, vocab int, history []int32) ([]byte, error) {
		return applyRepeatPenaltyBF16(logits, vocab, history, p.RepeatPenalty)
	}
}
