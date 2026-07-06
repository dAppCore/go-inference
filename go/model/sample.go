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

	// rank by probability, descending (top-k + top-p both work over this order).
	order := s.order[:vocab]
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })

	keep := vocab
	if p.TopK > 0 && p.TopK < keep {
		keep = p.TopK
	}
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

func applyRepeatPenaltyBF16(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("model.applyRepeatPenalty: logits must be vocab bf16 bytes")
	}
	if penalty <= 1 || len(history) == 0 {
		return logits, nil
	}
	ids := make([]int32, 0, len(history))
	for _, id := range history {
		if id >= 0 && int(id) < vocab {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return logits, nil
	}
	slices.Sort(ids)
	out := make([]byte, len(logits))
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
