// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"slices"
	"sort"

	core "dappco.re/go"
)

// Token sampling — backend-agnostic, the step after the LM head: it turns a vocab of
// logits into the next token id. It operates on bf16 []byte logits (the seam's lingua
// franca — whatever backend produced them), so it lives here in pkg/model, pure-Go and
// all-platforms. Greedy closes a native decode loop deterministically (the right choice
// for a tok/s bench); the temperature/top-k/top-p Sampler is for stochastic generation.
// (The served path's sampler is go-inference's until the reactive engine is pointed at
// model.Backend; this is the contract-native sampler.)

const bf16Size = 2

func bf16ToF32(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// Greedy returns the argmax of vocab bf16 logits; ties resolve to the lowest index.
// Deterministic, no RNG — the natural choice for closing a decode loop in a bench.
func Greedy(logits []byte, vocab int) (int32, error) {
	return greedySuppressed(logits, vocab, nil)
}

func greedySuppressed(logits []byte, vocab int, suppress []int32) (int32, error) {
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("model.Greedy: logits must be vocab bf16 bytes")
	}
	best, bestV := -1, float32(math.Inf(-1))
	for i := range vocab {
		if tokenSuppressed(int32(i), suppress) {
			continue
		}
		if v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]); v > bestV {
			best, bestV = i, v
		}
	}
	if best < 0 {
		return 0, core.NewError("model.Greedy: all tokens are suppressed")
	}
	return int32(best), nil
}

// SampleParams configures stochastic sampling. Temperature <= 0 makes Sample greedy.
// TopK <= 0 disables the top-k cut; TopP <= 0 or >= 1 disables the nucleus cut. The two
// cuts compose (top-k first, then top-p over the kept set), matching the usual order.
type SampleParams struct {
	Temperature         float32
	TopK                int
	TopP                float32
	MinP                float32
	SuppressTokens      []int32
	MinTokensBeforeStop int
	RepeatPenalty       float32
}

// Sampler draws tokens with a reproducible RNG that ADVANCES per Sample call, so a
// generation loop gets a varied sequence from a single seed (vs re-seeding per token).
// Construct with NewSampler; Greedy draws are RNG-free so they don't perturb the state.
//
// A Sampler is NOT safe for concurrent use: its RNG state is mutable, and Sample reuses
// per-call scratch buffers held on it (the softmax/rank workspace, grown once to the vocab
// and reused — so a decode loop pays the vocab-sized allocation once, not per token). The
// served path constructs one Sampler per request (register_native.go), matching this.
type Sampler struct {
	state uint64

	// reusable softmax/rank scratch, grown to the vocab on first Sample and reused. The
	// per-token allocation of these three vocab-sized buffers (≈ the GenerateSampled path's
	// dominant heap bytes) is the AX-11 win: a 256k-vocab decode allocated ~4 MB/token here
	// before reuse. probs stores unnormalised exp weights; the common denominator cancels
	// through ranking, TopP/MinP, and categorical draw, so the hot path skips a full vocab
	// divide pass. scaled is retained for older tests that assert it stays unallocated.
	// order is only needed when a rank filter is active.
	scaled, probs []float32
	order         []int
}

// NewSampler returns a sampler seeded for reproducible draws.
func NewSampler(seed uint64) *Sampler { return &Sampler{state: seed} }

// Draw returns the next reproducible uniform value in [0,1). Backends that keep
// sampling reductions on-device can consume the same RNG stream as Sample while
// avoiding a host logits readback.
func (s *Sampler) Draw() float32 { return s.next() }

// next is splitmix64 → a uniform float32 in [0,1); advances the RNG state.
func (s *Sampler) next() float32 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return float32(z>>40) / float32(1<<24)
}

// Sample picks a token from vocab bf16 logits per p: greedy when Temperature <= 0, else
// temperature-scaled softmax with optional top-k then top-p (nucleus) restriction, drawn
// from the categorical with this sampler's RNG.
func (s *Sampler) Sample(logits []byte, vocab int, p SampleParams) (int32, error) {
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("model.Sample: logits must be vocab bf16 bytes")
	}
	return s.sampleMapped(logits, nil, vocab, p)
}

// SampleCandidates picks from a preselected candidate set. logits is bf16 values
// aligned with ids; the returned token is one of ids. This is the same sampler
// core as Sample, but lets native/Metal backends keep vocab-wide ranking on the
// device and read back only the candidate window.
func (s *Sampler) SampleCandidates(logits []byte, ids []int32, p SampleParams) (int32, error) {
	if len(ids) == 0 {
		return 0, core.NewError("model.SampleCandidates: empty candidates")
	}
	if len(logits) != len(ids)*bf16Size {
		return 0, core.NewError("model.SampleCandidates: logits must be candidate bf16 bytes")
	}
	return s.sampleMapped(logits, ids, len(ids), p)
}

func (s *Sampler) sampleMapped(logits []byte, ids []int32, vocab int, p SampleParams) (int32, error) {
	if p.Temperature <= 0 && p.MinP <= 0 {
		if ids == nil {
			return greedySuppressed(logits, vocab, p.SuppressTokens)
		}
		return greedyMappedSuppressed(logits, ids, p.SuppressTokens)
	}
	if p.TopK == 1 {
		next, err := topMappedSuppressed(logits, ids, vocab, p.SuppressTokens)
		if err != nil {
			return 0, err
		}
		s.next()
		return next, nil
	}
	temp := p.Temperature
	if temp <= 0 {
		temp = 1
	}

	// grow-once, reuse-thereafter scratch (below the greedy guard so a zero-temp request stays
	// zero-alloc): each buffer is grown to the vocab on first need and reused on every later
	// Sample, then sliced to [:vocab] and FULLY overwritten below — so the result is identical
	// to allocating fresh, with the per-token vocab-sized allocations paid once per Sampler.
	rankFilter := sampleRankFilterNeeded(p, vocab)
	if cap(s.probs) < vocab {
		s.probs = make([]float32, vocab)
	}
	if rankFilter && cap(s.order) < vocab {
		s.order = make([]int, vocab)
	}
	probs := s.probs[:vocab]

	// FAST PATH — when top-k caps the kept set, rank the survivors on the raw
	// temperature-scaled logits (exp is monotonic, so the logit order IS the
	// probability order) and exp ONLY those survivors, skipping the full-vocab
	// exp pass — ≈256k math.Exp/token on a Qwen vocab, the dominant host-sampler
	// cost. Byte-identical to the full path: the top-k tokens carry the same
	// exp(v−max) weights in the same order, and any 0-mass tail token whose order
	// differs under an exp-underflow tie is cut by top-p/min-p or never reached
	// by the draw (acc only advances on real mass).
	if rankFilter && p.TopK > 0 && p.TopK < vocab {
		return s.sampleTopKFirst(logits, ids, vocab, temp, p)
	}

	// temperature-scaled logits + their max (for a stable softmax).
	maxL := float32(math.Inf(-1))
	allowed := 0
	for i := range vocab {
		id := int32(i)
		if ids != nil {
			id = ids[i]
		}
		if tokenSuppressed(id, p.SuppressTokens) {
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		allowed++
		if v > maxL {
			maxL = v
		}
	}
	if allowed == 0 {
		return 0, core.NewError("model.Sample: all tokens are suppressed")
	}
	var sum float32
	for i := range vocab {
		id := int32(i)
		if ids != nil {
			id = ids[i]
		}
		if tokenSuppressed(id, p.SuppressTokens) {
			probs[i] = 0
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		e := float32(math.Exp(float64(v - maxL)))
		probs[i] = e
		sum += e
	}

	if !rankFilter {
		return s.sampleMappedInVocabOrder(probs, ids, vocab, sum)
	}

	// top-p / min-p without a top-k cap can walk the whole ordering, so the kept
	// set is the full vocab ranked by a stable descending sort. (A top-k cap took
	// the fast path above.)
	keep := vocab
	order := s.order[:keep]
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })
	return s.drawFromRanked(order, probs, ids, keep, vocab, sum, p)
}

// sampleTopKFirst is Sample's fast path when top-k caps the kept set: it ranks
// the top-k on the raw temperature-scaled logits (monotone in probability) and
// exps only those survivors, avoiding the full-vocab softmax. See the FAST PATH
// note in sampleMapped for the byte-identity argument.
func (s *Sampler) sampleTopKFirst(logits []byte, ids []int32, vocab int, temp float32, p SampleParams) (int32, error) {
	// probs doubles as the vals scratch: it holds the temperature-scaled logits
	// through the select, then the exp weights for the kept survivors.
	vals := s.probs[:vocab]
	allowed := 0
	for i := range vocab {
		id := int32(i)
		if ids != nil {
			id = ids[i]
		}
		if tokenSuppressed(id, p.SuppressTokens) {
			vals[i] = float32(math.Inf(-1)) // ranks below every real token; exps to 0 if ever kept
			continue
		}
		vals[i] = bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		allowed++
	}
	if allowed == 0 {
		return 0, core.NewError("model.Sample: all tokens are suppressed")
	}
	keep := p.TopK
	if keep > vocab {
		keep = vocab
	}
	order := s.order[:keep]
	selectTopKDesc(order, vals, vocab)
	// exp the survivors, stabilised by the global max (the top-ranked survivor).
	// Read the max before overwriting vals[order[0]].
	maxL := vals[order[0]]
	for i := 0; i < keep; i++ {
		vals[order[i]] = float32(math.Exp(float64(vals[order[i]] - maxL)))
	}
	return s.drawFromRanked(order, vals, ids, keep, vocab, 0, p)
}

// drawFromRanked applies the top-p (nucleus) then min-p cut over the kept order
// (descending by probability), renormalises the survivors and draws one with
// this sampler's RNG. sum is the full-vocab mass, used as the top-p base only
// when keep == vocab (the whole vocab is kept); a top-k-capped set recomputes its
// own kept mass. Shared by Sample's full and fast paths so the cut + draw
// arithmetic has one implementation.
func (s *Sampler) drawFromRanked(order []int, probs []float32, ids []int32, keep, vocab int, sum float32, p SampleParams) (int32, error) {
	if p.TopP > 0 && p.TopP < 1 {
		keptMass := sum
		if keep != vocab {
			keptMass = 0
			for i := 0; i < keep; i++ {
				keptMass += probs[order[i]]
			}
		}
		var cum float32
		n := 0
		for n < keep {
			cum += probs[order[n]]
			n++
			if cum >= p.TopP*keptMass {
				break
			}
		}
		keep = n
	}
	if p.MinP > 0 && keep > 0 {
		threshold := probs[order[0]] * p.MinP
		n := 0
		for n < keep && probs[order[n]] >= threshold {
			n++
		}
		if n > 0 {
			keep = n
		}
	}

	// renormalise over the kept set and draw.
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += probs[order[i]]
	}
	target := s.next() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += probs[order[i]]
		if acc >= target {
			if ids != nil {
				return ids[order[i]], nil
			}
			return int32(order[i]), nil
		}
	}
	if ids != nil {
		return ids[order[keep-1]], nil // floating-point fall-through
	}
	return int32(order[keep-1]), nil // floating-point fall-through
}

func sampleRankFilterNeeded(p SampleParams, vocab int) bool {
	if p.TopK > 0 && p.TopK < vocab {
		return true
	}
	if p.TopP > 0 && p.TopP < 1 {
		return true
	}
	return p.MinP > 0
}

// selectTopKDesc fills dst with the indices of the k = len(dst) highest-ranked entries
// of probs[:vocab], ordered exactly as the first k of a descending sort.SliceStable by
// probs (ties broken by ascending index, matching the stable sort's original-order rule).
// It is the bounded equivalent of ranking the whole vocab when only the top k positions
// are read: an O(vocab + k·log k) min-heap select rather than an O(vocab·log vocab) sort.
// Callers must ensure 0 < k <= vocab.
func selectTopKDesc(dst []int, probs []float32, vocab int) {
	k := len(dst)
	for i := 0; i < k; i++ {
		dst[i] = i
	}
	// Min-heap by rank (root = lowest-ranked kept), so the weakest survivor evicts first.
	for i := k/2 - 1; i >= 0; i-- {
		siftDownByRank(dst, i, k, probs)
	}
	for i := k; i < vocab; i++ {
		if rankLess(probs, dst[0], i) { // the kept minimum is outranked by i → i replaces it
			dst[0] = i
			siftDownByRank(dst, 0, k, probs)
		}
	}
	// The kept set is a heap; order it as the stable descending sort would. rankLess is a
	// strict total order over distinct indices, so an unstable sort is deterministic here.
	sort.Slice(dst, func(a, b int) bool { return rankLess(probs, dst[b], dst[a]) })
}

// rankLess reports whether index a ranks below index b under the sampler's stable-descending
// order: strictly lower probability, or equal probability with a higher index (the stable
// sort keeps the lower index ahead).
func rankLess(probs []float32, a, b int) bool {
	if probs[a] != probs[b] {
		return probs[a] < probs[b]
	}
	return a > b
}

// siftDownByRank restores the min-heap property (root = lowest rank) at node i over h[:n].
func siftDownByRank(h []int, i, n int, probs []float32) {
	for {
		lo := i
		if l := 2*i + 1; l < n && rankLess(probs, h[l], h[lo]) {
			lo = l
		}
		if r := 2*i + 2; r < n && rankLess(probs, h[r], h[lo]) {
			lo = r
		}
		if lo == i {
			break
		}
		h[i], h[lo] = h[lo], h[i]
		i = lo
	}
}

func (s *Sampler) sampleMappedInVocabOrder(probs []float32, ids []int32, vocab int, sum float32) (int32, error) {
	if sum == 0 {
		return 0, core.NewError("model.Sample: empty sampled distribution")
	}
	target := s.next() * sum
	var acc float32
	for i := range vocab {
		acc += probs[i]
		if acc >= target {
			if ids != nil {
				return ids[i], nil
			}
			return int32(i), nil
		}
	}
	if ids != nil {
		return ids[vocab-1], nil
	}
	return int32(vocab - 1), nil
}

func tokenSuppressed(id int32, suppress []int32) bool {
	return slices.Contains(suppress, id)
}

func greedyMappedSuppressed(logits []byte, ids []int32, suppress []int32) (int32, error) {
	best, bestV := -1, float32(math.Inf(-1))
	for i, id := range ids {
		if tokenSuppressed(id, suppress) {
			continue
		}
		if v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]); v > bestV {
			best, bestV = i, v
		}
	}
	if best < 0 {
		return 0, core.NewError("model.SampleCandidates: all tokens are suppressed")
	}
	return ids[best], nil
}

func topMappedSuppressed(logits []byte, ids []int32, vocab int, suppress []int32) (int32, error) {
	best, bestV := -1, float32(math.Inf(-1))
	for i := range vocab {
		id := int32(i)
		if ids != nil {
			id = ids[i]
		}
		if tokenSuppressed(id, suppress) {
			continue
		}
		if v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]); v > bestV {
			best, bestV = i, v
		}
	}
	if best < 0 {
		return 0, core.NewError("model.Sample: all tokens are suppressed")
	}
	if ids != nil {
		return ids[best], nil
	}
	return int32(best), nil
}

// applyRepeatPenaltyBF16 divides (positive) or multiplies (non-positive) the
// logit of each already-generated token by penalty, discouraging repetition. It
// returns a fresh buffer, leaving logits untouched — the one-shot convenience
// wrapper; the decode loop's per-token path uses applyRepeatPenaltyBF16Into with
// a reused scratch to avoid a vocab-sized allocation every token.
func applyRepeatPenaltyBF16(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	var scratch []byte
	var idScratch []int32
	return applyRepeatPenaltyBF16Into(&scratch, &idScratch, logits, vocab, history, penalty)
}

// applyRepeatPenaltyBF16Into writes the repeat-penalised logits into *scratch,
// grown to the vocab ONCE and reused on every later call, so a decode loop under
// repeat_penalty pays the vocab-sized buffer once rather than allocating ≈512 KB
// per token on a 256k-vocab model (the AX-11 win). *idScratch is the same
// grow-once reuse for the history working set (which lengthens with the
// generation), so the whole path is zero-alloc after warmup. The input logits
// are left untouched, and a no-op penalty returns logits directly with no copy.
// Only the deduped history positions are rewritten over a fresh full copy, so the
// result is byte-identical to allocating new buffers each call.
func applyRepeatPenaltyBF16Into(scratch *[]byte, idScratch *[]int32, logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("model.applyRepeatPenalty: logits must be vocab bf16 bytes")
	}
	if penalty <= 1 || len(history) == 0 {
		return logits, nil
	}
	ids := (*idScratch)[:0] // reuse the grown backing array — no per-token history copy
	for _, id := range history {
		if id >= 0 && int(id) < vocab {
			ids = append(ids, id)
		}
	}
	*idScratch = ids
	if len(ids) == 0 {
		return logits, nil
	}
	slices.Sort(ids)
	if cap(*scratch) < len(logits) {
		*scratch = make([]byte, len(logits))
	}
	out := (*scratch)[:len(logits)]
	copy(out, logits)
	var prev int32
	for i, id := range ids {
		if i > 0 && id == prev {
			continue
		}
		prev = id
		off := int(id) * bf16Size
		v := bf16ToF32(out[off], out[off+1])
		if v > 0 {
			v /= penalty
		} else {
			v *= penalty
		}
		h := f32ToBF16(v)
		out[off] = byte(h)
		out[off+1] = byte(h >> 8)
	}
	return out, nil
}

func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	return uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
}
