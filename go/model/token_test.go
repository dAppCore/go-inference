// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// f32ToBF16Bytes writes v as the two bf16 bytes the seam uses (the high 16 bits
// of the f32, little-endian within the 16). Small integers are exact in bf16, so
// the counter model round-trips its token ids losslessly.
func f32ToBF16Bytes(v float32) (lo, hi byte) {
	h := uint16(math.Float32bits(v) >> 16)
	return byte(h), byte(h >> 8)
}

// counterModel is a deterministic fake TokenModel: it encodes a token id in
// hidden dim 0, its decode is the identity stack (so the last hidden carries the
// last input id), and its head emits a one-hot logit at (id+1) mod vocab. So
// greedy generation from [k] yields k+1, k+2, … — a sequence that ONLY stays a
// clean count if Generate re-embeds each generated token into the running
// sequence (a broken re-embed breaks the count at the first generated token).
type counterModel struct {
	vocab  int
	dModel int
}

func (m counterModel) Vocab() int { return m.vocab }

func (m counterModel) Embed(id int32) ([]byte, error) {
	emb := make([]byte, m.dModel*bf16Size)
	emb[0], emb[1] = f32ToBF16Bytes(float32(id)) // id in dim 0, rest zero
	return emb, nil
}

// DecodeForward is the identity stack: each output hidden equals its input
// embedding, so the last hidden carries the last token's id.
func (m counterModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return inputs, nil
}

func (m counterModel) Head(hidden []byte) ([]byte, error) {
	id := int(math.Round(float64(bf16ToF32(hidden[0], hidden[1]))))
	target := (id + 1) % m.vocab
	logits := make([]byte, m.vocab*bf16Size)
	logits[target*bf16Size], logits[target*bf16Size+1] = f32ToBF16Bytes(1) // one-hot at id+1
	return logits, nil
}

type repeatPenaltyModel struct{}

func (repeatPenaltyModel) Vocab() int { return 4 }

func (repeatPenaltyModel) Embed(id int32) ([]byte, error) {
	emb := make([]byte, bf16Size)
	emb[0], emb[1] = f32ToBF16Bytes(float32(id))
	return emb, nil
}

func (repeatPenaltyModel) DecodeForward(inputs [][]byte) ([][]byte, error) { return inputs, nil }

func (repeatPenaltyModel) Head([]byte) ([]byte, error) {
	logits := make([]byte, 4*bf16Size)
	logits[1*bf16Size], logits[1*bf16Size+1] = f32ToBF16Bytes(1.0)
	logits[2*bf16Size], logits[2*bf16Size+1] = f32ToBF16Bytes(0.75)
	return logits, nil
}

func idsEqual(a, b []int32) bool {
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

func TestGenerate_CounterLoop(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}

	// greedy from [0] for 5 tokens → 1,2,3,4,5 (only correct if re-embed feeds
	// each generated token back into the next step).
	got, err := Generate(m, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := []int32{1, 2, 3, 4, 5}; !idsEqual(got, want) {
		t.Fatalf("greedy count = %v, want %v", got, want)
	}

	// eos stops the loop the moment that token is generated (3 reached on the
	// third step), so maxNew=10 still ends at [1,2,3].
	got, err = Generate(m, []int32{0}, 10, 3)
	if err != nil {
		t.Fatalf("Generate eos: %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(got, want) {
		t.Fatalf("eos count = %v, want %v", got, want)
	}

	// the prompt's LAST id drives the first generated token (prompt [5,9] → the
	// 9 leads, so 10,11,12).
	got, err = Generate(m, []int32{5, 9}, 3, -1)
	if err != nil {
		t.Fatalf("Generate multi-prompt: %v", err)
	}
	if want := []int32{10, 11, 12}; !idsEqual(got, want) {
		t.Fatalf("multi-prompt count = %v, want %v", got, want)
	}
}

func TestGenerate_Errors(t *testing.T) {
	m := counterModel{vocab: 8, dModel: 2}
	if _, err := Generate(nil, []int32{0}, 4, -1); err == nil {
		t.Fatal("nil model should error")
	}
	if _, err := Generate(m, nil, 4, -1); err == nil {
		t.Fatal("empty prompt should error")
	}
	if _, err := Generate(m, []int32{0}, 0, -1); err == nil {
		t.Fatal("maxNew <= 0 should error")
	}
}

func TestGenerateSampled_ZeroTempIsGreedy(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	greedy, err := Generate(m, []int32{0}, 6, -1)
	if err != nil {
		t.Fatalf("greedy: %v", err)
	}
	// temperature 0 → the sampler falls back to greedy per token, so the
	// stochastic path reproduces the greedy sequence exactly.
	sampled, err := GenerateSampled(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 6, -1)
	if err != nil {
		t.Fatalf("sampled: %v", err)
	}
	if !idsEqual(greedy, sampled) {
		t.Fatalf("zero-temp sampled %v != greedy %v", sampled, greedy)
	}
	if _, err := GenerateSampled(m, nil, SampleParams{}, []int32{0}, 4, -1); err == nil {
		t.Fatal("nil sampler should error")
	}
}

func TestGenerateSampledWithStopTokens_MultipleStops(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}

	got, err := GenerateSampledWithStopTokens(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 10, []int32{4, 2})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens: %v", err)
	}
	if want := []int32{1, 2}; !idsEqual(got, want) {
		t.Fatalf("sampled stop-set count = %v, want %v", got, want)
	}
	if _, err := GenerateSampledWithStopTokens(m, nil, SampleParams{}, []int32{0}, 4, []int32{1}); err == nil {
		t.Fatal("nil sampler should error")
	}
}

func TestGenerateSampledWithStopTokensTransform_CommitsTransformedToken(t *testing.T) {
	var stepped []int32
	m := idSessionModel{counterModel: counterModel{vocab: 16, dModel: 4}, ids: &stepped}

	got, err := GenerateSampledWithStopTokensTransform(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 3, nil, func(id int32) int32 {
		if id == 1 {
			return 5
		}
		return id
	})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransform: %v", err)
	}
	if want := []int32{5, 6, 7}; !idsEqual(got, want) {
		t.Fatalf("transformed sampled ids = %v, want %v", got, want)
	}
	if want := []int32{0, 5, 6}; !idsEqual(stepped, want) {
		t.Fatalf("stepped ids = %v, want %v (transform must feed the next decode step)", stepped, want)
	}
}

func TestGenerateSampledWithStopTokens_SuppressesToken(t *testing.T) {
	got, err := GenerateSampledWithStopTokens(repeatPenaltyModel{}, NewSampler(1), SampleParams{Temperature: 0, SuppressTokens: []int32{1}}, []int32{0}, 1, nil)
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens: %v", err)
	}
	if want := []int32{2}; !idsEqual(got, want) {
		t.Fatalf("suppressed sampled ids = %v, want %v", got, want)
	}
}

func TestGenerateSampledWithStopTokens_MinTokensBeforeStopSuppressesFirstStop(t *testing.T) {
	got, err := GenerateSampledWithStopTokens(repeatPenaltyModel{}, NewSampler(1), SampleParams{Temperature: 0, MinTokensBeforeStop: 1}, []int32{0}, 1, []int32{1})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens: %v", err)
	}
	if want := []int32{2}; !idsEqual(got, want) {
		t.Fatalf("min-stop sampled ids = %v, want %v", got, want)
	}
}

func TestGenerateSampledWithRepeatPenaltyPenalisesGeneratedHistory(t *testing.T) {
	got, err := GenerateSampledWithStopTokens(repeatPenaltyModel{}, NewSampler(1), SampleParams{Temperature: 0, RepeatPenalty: 2}, []int32{0}, 2, nil)
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens: %v", err)
	}
	if want := []int32{1, 2}; !idsEqual(got, want) {
		t.Fatalf("repeat-penalised greedy = %v, want %v", got, want)
	}
}

// counterStepper is the incremental decode of counterModel: the counter is
// memoryless (next = id+1), so the last token's embedding IS its hidden state —
// the identity step. It carries no cache because nothing depends on history. It
// implements the optional Close (recording the call) to gate that Generate
// releases the stepper it opens.
type counterStepper struct{ closed *int }

func (counterStepper) Step(emb []byte) ([]byte, error) { return emb, nil }

func (s counterStepper) Close() error {
	if s.closed != nil {
		*s.closed++
	}
	return nil
}

// sessionCounterModel is counterModel that ALSO offers a persistent-cache
// session — but whose whole-sequence DecodeForward ERRORS, so a passing
// generation proves Generate took the incremental SessionModel path.
type sessionCounterModel struct {
	counterModel
	opened *int
	closed *int
}

func (sessionCounterModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return nil, core.NewError("whole-seq path must not run when a session is available")
}

func (m sessionCounterModel) OpenSession() (DecodeStepper, error) {
	if m.opened != nil {
		*m.opened++
	}
	return counterStepper{closed: m.closed}, nil
}

func TestGenerate_SessionPath(t *testing.T) {
	var _ SessionModel = sessionCounterModel{} // compile-time: it offers the incremental path

	opened, closed := 0, 0
	m := sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}, opened: &opened, closed: &closed}

	// Generate must dispatch to the incremental session path — its DecodeForward
	// errors, so any produced token proves the whole-seq fallback was NOT used.
	got, err := Generate(m, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate (session path): %v", err)
	}
	if want := []int32{1, 2, 3, 4, 5}; !idsEqual(got, want) {
		t.Fatalf("session-path count = %v, want %v", got, want)
	}
	if opened != 1 {
		t.Fatalf("OpenSession called %d times, want exactly 1", opened)
	}
	if closed != 1 {
		t.Fatalf("Close called %d times, want exactly 1 (Generate must release the stepper it opens)", closed)
	}

	// the incremental path is output-identical to the whole-seq fallback on the
	// equivalent session-less model.
	wholeSeq, err := Generate(counterModel{vocab: 16, dModel: 4}, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate (whole-seq): %v", err)
	}
	if !idsEqual(got, wholeSeq) {
		t.Fatalf("session %v != whole-seq %v", got, wholeSeq)
	}
}

// TestGenerateSampledStreamsEachToken locks the sampled streaming path: the
// per-token yield fires once per committed token in order, the streamed sequence
// equals the batch (nil-yield) result byte-for-byte, and a yield returning false
// ends generation early. Regression guard for the native generate-stream fix —
// the temp>0 decode path used to return []int32 all at once, so an iterator over
// it reported a zero decode interval (decode 0.000 tok/s) instead of streaming.
func TestGenerateSampledStreamsEachToken(t *testing.T) {
	m := sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}, opened: new(int), closed: new(int)}
	s := NewSampler(1)
	p := SampleParams{Temperature: 0} // one-hot logits → deterministic counter regardless of seed

	// batch (nil yield) is the reference sequence.
	batch, err := GenerateSampledWithStopTokensTransform(m, s, p, []int32{0}, 5, nil, nil)
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if want := []int32{1, 2, 3, 4, 5}; !idsEqual(batch, want) {
		t.Fatalf("batch = %v, want %v", batch, want)
	}

	// streaming Each: yield once per committed token, in order; the returned slice
	// must match the batch byte-for-byte (streaming changes timing, not tokens).
	var streamed []int32
	got, err := GenerateSampledWithStopTokensTransformEach(m, s, p, []int32{0}, 5, nil, nil, func(id int32) bool {
		streamed = append(streamed, id)
		return true
	})
	if err != nil {
		t.Fatalf("streaming: %v", err)
	}
	if !idsEqual(got, batch) {
		t.Fatalf("streamed return %v != batch %v", got, batch)
	}
	if !idsEqual(streamed, batch) {
		t.Fatalf("yielded %v != returned %v — yield must fire once per committed token", streamed, batch)
	}

	// yield returning false ends generation early — the ctx-cancel / consumer-done
	// path the native iterator relies on to stop mid-stream.
	var partial []int32
	stop, err := GenerateSampledWithStopTokensTransformEach(m, s, p, []int32{0}, 5, nil, nil, func(id int32) bool {
		partial = append(partial, id)
		return len(partial) < 2 // stop after the 2nd token
	})
	if err != nil {
		t.Fatalf("early-stop: %v", err)
	}
	if want := []int32{1, 2}; !idsEqual(stop, want) {
		t.Fatalf("early-stop returned %v, want %v (yield-false must break after the committed token)", stop, want)
	}
}

type directGenerateStepper struct {
	calls        *int
	oneShotCalls *int
	steps        *int
	prompt       *[]int32
	maxNew       *int
	eos          *int
}

func (s directGenerateStepper) Step([]byte) ([]byte, error) {
	if s.steps != nil {
		*s.steps++
	}
	return nil, core.NewError("directGenerateStepper: Step must not run when Generate is available")
}

func (s directGenerateStepper) Generate(promptIDs []int32, maxNew, eos int) ([]int32, error) {
	if s.calls != nil {
		*s.calls++
	}
	return nil, core.NewError("retained Generate must not run when one-shot GenerateOneShot is available")
}

func (s directGenerateStepper) GenerateOneShot(promptIDs []int32, maxNew, eos int) ([]int32, error) {
	if s.oneShotCalls != nil {
		*s.oneShotCalls++
	}
	if s.prompt != nil {
		*s.prompt = append((*s.prompt)[:0], promptIDs...)
	}
	if s.maxNew != nil {
		*s.maxNew = maxNew
	}
	if s.eos != nil {
		*s.eos = eos
	}
	return []int32{7, 8, 9}, nil
}

type directGenerateSessionModel struct {
	counterModel
	calls        *int
	oneShotCalls *int
	steps        *int
	heads        *int
	prompt       *[]int32
	maxNew       *int
	eos          *int
}

func (directGenerateSessionModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return nil, core.NewError("whole-seq path must not run when a direct session generator is available")
}

func (m directGenerateSessionModel) Head([]byte) ([]byte, error) {
	if m.heads != nil {
		*m.heads++
	}
	return nil, core.NewError("Head must not run when a direct session generator is available")
}

func (m directGenerateSessionModel) OpenSession() (DecodeStepper, error) {
	return directGenerateStepper{
		calls: m.calls, oneShotCalls: m.oneShotCalls, steps: m.steps, prompt: m.prompt,
		maxNew: m.maxNew, eos: m.eos,
	}, nil
}

func TestGenerate_DirectSessionGenerate(t *testing.T) {
	calls, oneShotCalls, steps, heads := 0, 0, 0, 0
	maxNew, eos := 0, 0
	var prompt []int32
	m := directGenerateSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		calls:        &calls,
		oneShotCalls: &oneShotCalls,
		steps:        &steps,
		heads:        &heads,
		prompt:       &prompt,
		maxNew:       &maxNew,
		eos:          &eos,
	}

	got, err := Generate(m, []int32{1, 2, 3}, 6, 9)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := []int32{7, 8, 9}; !idsEqual(got, want) {
		t.Fatalf("direct session Generate returned %v, want %v", got, want)
	}
	if oneShotCalls != 1 || calls != 0 {
		t.Fatalf("direct session calls GenerateOneShot=%d Generate=%d, want 1/0", oneShotCalls, calls)
	}
	if steps != 0 || heads != 0 {
		t.Fatalf("fallback path ran: steps=%d heads=%d, want both 0", steps, heads)
	}
	if !idsEqual(prompt, []int32{1, 2, 3}) || maxNew != 6 || eos != 9 {
		t.Fatalf("direct session args prompt=%v maxNew=%d eos=%d", prompt, maxNew, eos)
	}
}

type directSampledStepper struct {
	calls         *int
	oneShotCalls  *int
	steps         *int
	prompt        *[]int32
	maxNew        *int
	stopTokens    *[]int32
	params        *SampleParams
	transformSeen *bool
	yieldSeen     *bool
	sampler       **Sampler
}

func (s directSampledStepper) Step([]byte) ([]byte, error) {
	if s.steps != nil {
		*s.steps++
	}
	return nil, core.NewError("directSampledStepper: Step must not run when GenerateSampledOneShotEach is available")
}

func (s directSampledStepper) GenerateSampledEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *Sampler, params SampleParams, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if s.calls != nil {
		*s.calls++
	}
	return nil, core.NewError("retained GenerateSampledEach must not run when one-shot GenerateSampledOneShotEach is available")
}

func (s directSampledStepper) GenerateSampledOneShotEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *Sampler, params SampleParams, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if s.oneShotCalls != nil {
		*s.oneShotCalls++
	}
	if s.prompt != nil {
		*s.prompt = append((*s.prompt)[:0], promptIDs...)
	}
	if s.maxNew != nil {
		*s.maxNew = maxNew
	}
	if s.stopTokens != nil {
		*s.stopTokens = append((*s.stopTokens)[:0], stopTokens...)
	}
	if s.params != nil {
		*s.params = params
	}
	if s.sampler != nil {
		*s.sampler = sampler
	}
	next := int32(7)
	if transform != nil {
		if s.transformSeen != nil {
			*s.transformSeen = true
		}
		next = transform(next)
	}
	if yield != nil {
		if s.yieldSeen != nil {
			*s.yieldSeen = true
		}
		if !yield(next) {
			return []int32{next}, nil
		}
	}
	return []int32{next, 8}, nil
}

type directSampledSessionModel struct {
	counterModel
	calls         *int
	oneShotCalls  *int
	steps         *int
	heads         *int
	prompt        *[]int32
	maxNew        *int
	stopTokens    *[]int32
	params        *SampleParams
	transformSeen *bool
	yieldSeen     *bool
	sampler       **Sampler
}

func (directSampledSessionModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return nil, core.NewError("whole-seq path must not run when a direct sampled session generator is available")
}

func (m directSampledSessionModel) Head([]byte) ([]byte, error) {
	if m.heads != nil {
		*m.heads++
	}
	return nil, core.NewError("Head must not run when a direct sampled session generator is available")
}

func (m directSampledSessionModel) OpenSession() (DecodeStepper, error) {
	return directSampledStepper{
		calls: m.calls, oneShotCalls: m.oneShotCalls, steps: m.steps, prompt: m.prompt,
		maxNew: m.maxNew, stopTokens: m.stopTokens, params: m.params, transformSeen: m.transformSeen,
		yieldSeen: m.yieldSeen, sampler: m.sampler,
	}, nil
}

func TestGenerateSampled_DirectSessionGenerate(t *testing.T) {
	calls, oneShotCalls, steps, heads := 0, 0, 0, 0
	maxNew := 0
	transformSeen, yieldSeen := false, false
	var prompt, stopTokens, yielded []int32
	var gotParams SampleParams
	var gotSampler *Sampler
	m := directSampledSessionModel{
		counterModel:  counterModel{vocab: 16, dModel: 4},
		calls:         &calls,
		oneShotCalls:  &oneShotCalls,
		steps:         &steps,
		heads:         &heads,
		prompt:        &prompt,
		maxNew:        &maxNew,
		stopTokens:    &stopTokens,
		params:        &gotParams,
		transformSeen: &transformSeen,
		yieldSeen:     &yieldSeen,
		sampler:       &gotSampler,
	}
	sampler := NewSampler(42)
	params := SampleParams{Temperature: 0.7, TopK: 3, TopP: 0.9, MinP: 0.01, MinTokensBeforeStop: 1}

	got, err := GenerateSampledWithStopTokensTransformEach(m, sampler, params, []int32{1, 2, 3}, 6, []int32{4, 5}, func(id int32) int32 {
		return id + 10
	}, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransformEach: %v", err)
	}
	if want := []int32{17, 8}; !idsEqual(got, want) {
		t.Fatalf("direct sampled session returned %v, want %v", got, want)
	}
	if oneShotCalls != 1 || calls != 0 {
		t.Fatalf("direct sampled session calls GenerateSampledOneShotEach=%d GenerateSampledEach=%d, want 1/0", oneShotCalls, calls)
	}
	if steps != 0 || heads != 0 {
		t.Fatalf("fallback path ran: steps=%d heads=%d, want both 0", steps, heads)
	}
	if !idsEqual(prompt, []int32{1, 2, 3}) || maxNew != 6 || !idsEqual(stopTokens, []int32{4, 5}) {
		t.Fatalf("direct sampled args prompt=%v maxNew=%d stopTokens=%v", prompt, maxNew, stopTokens)
	}
	if gotSampler != sampler {
		t.Fatalf("direct sampled sampler = %p, want %p", gotSampler, sampler)
	}
	if gotParams.Temperature != params.Temperature || gotParams.TopK != params.TopK || gotParams.TopP != params.TopP || gotParams.MinP != params.MinP || gotParams.MinTokensBeforeStop != params.MinTokensBeforeStop {
		t.Fatalf("direct sampled params = %+v, want %+v", gotParams, params)
	}
	if !transformSeen || !yieldSeen || !idsEqual(yielded, []int32{17}) {
		t.Fatalf("direct sampled transform/yield seen=%v/%v yielded=%v, want true/true/[17]", transformSeen, yieldSeen, yielded)
	}
}

func TestGenerateSampled_DirectSessionGenerateNoEOS(t *testing.T) {
	calls, oneShotCalls, steps, heads := 0, 0, 0, 0
	maxNew := 0
	var prompt, stopTokens []int32
	var gotParams SampleParams
	var gotSampler *Sampler
	m := directSampledSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		calls:        &calls,
		oneShotCalls: &oneShotCalls,
		steps:        &steps,
		heads:        &heads,
		prompt:       &prompt,
		maxNew:       &maxNew,
		stopTokens:   &stopTokens,
		params:       &gotParams,
		sampler:      &gotSampler,
	}
	sampler := NewSampler(42)
	params := SampleParams{Temperature: 0.7, TopK: 3, TopP: 0.9, MinP: 0.01, MinTokensBeforeStop: 1}

	got, err := GenerateSampled(m, sampler, params, []int32{1, 2, 3}, 6, -1)
	if err != nil {
		t.Fatalf("GenerateSampled: %v", err)
	}
	if want := []int32{7, 8}; !idsEqual(got, want) {
		t.Fatalf("direct sampled session returned %v, want %v", got, want)
	}
	if oneShotCalls != 1 || calls != 0 {
		t.Fatalf("direct sampled session calls GenerateSampledOneShotEach=%d GenerateSampledEach=%d, want 1/0", oneShotCalls, calls)
	}
	if steps != 0 || heads != 0 {
		t.Fatalf("fallback path ran: steps=%d heads=%d, want both 0", steps, heads)
	}
	if !idsEqual(prompt, []int32{1, 2, 3}) || maxNew != 6 || len(stopTokens) != 0 {
		t.Fatalf("direct sampled args prompt=%v maxNew=%d stopTokens=%v", prompt, maxNew, stopTokens)
	}
	if gotSampler != sampler {
		t.Fatalf("direct sampled sampler = %p, want %p", gotSampler, sampler)
	}
	if gotParams.Temperature != params.Temperature || gotParams.TopK != params.TopK || gotParams.TopP != params.TopP || gotParams.MinP != params.MinP || gotParams.MinTokensBeforeStop != params.MinTokensBeforeStop {
		t.Fatalf("direct sampled params = %+v, want %+v", gotParams, params)
	}
}

func TestGenerateSampled_DirectSessionGenerateEOS(t *testing.T) {
	calls, oneShotCalls, steps, heads := 0, 0, 0, 0
	maxNew := 0
	var prompt, stopTokens []int32
	var gotParams SampleParams
	var gotSampler *Sampler
	m := directSampledSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		calls:        &calls,
		oneShotCalls: &oneShotCalls,
		steps:        &steps,
		heads:        &heads,
		prompt:       &prompt,
		maxNew:       &maxNew,
		stopTokens:   &stopTokens,
		params:       &gotParams,
		sampler:      &gotSampler,
	}
	sampler := NewSampler(42)
	params := SampleParams{Temperature: 0.7, TopK: 3, TopP: 0.9, MinP: 0.01, MinTokensBeforeStop: 4}

	got, err := GenerateSampled(m, sampler, params, []int32{1, 2, 3}, 6, 8)
	if err != nil {
		t.Fatalf("GenerateSampled: %v", err)
	}
	if want := []int32{7, 8}; !idsEqual(got, want) {
		t.Fatalf("direct sampled session returned %v, want %v", got, want)
	}
	if oneShotCalls != 1 || calls != 0 {
		t.Fatalf("direct sampled session calls GenerateSampledOneShotEach=%d GenerateSampledEach=%d, want 1/0", oneShotCalls, calls)
	}
	if steps != 0 || heads != 0 {
		t.Fatalf("fallback path ran: steps=%d heads=%d, want both 0", steps, heads)
	}
	if !idsEqual(prompt, []int32{1, 2, 3}) || maxNew != 6 || !idsEqual(stopTokens, []int32{8}) {
		t.Fatalf("direct sampled args prompt=%v maxNew=%d stopTokens=%v", prompt, maxNew, stopTokens)
	}
	if gotSampler != sampler {
		t.Fatalf("direct sampled sampler = %p, want %p", gotSampler, sampler)
	}
	params.MinTokensBeforeStop = 0
	if gotParams.Temperature != params.Temperature || gotParams.TopK != params.TopK || gotParams.TopP != params.TopP || gotParams.MinP != params.MinP || gotParams.MinTokensBeforeStop != params.MinTokensBeforeStop {
		t.Fatalf("direct sampled params = %+v, want %+v", gotParams, params)
	}
}

// idStepper is a stepper that needs the token ID, not just the embedding — the
// StepWithID feature (per-layer inputs gathered from the id). It records
// every id it was stepped with AND ignores the embedding, so a passing count proves
// Generate fed it ids via StepWithID rather than Step. It implements no Close (a
// GC-managed backend), so the no-Close branch is exercised too.
type idStepper struct {
	ids    *[]int32
	vocab  int
	dModel int
}

func (s idStepper) Step(emb []byte) ([]byte, error) {
	return nil, core.NewError("idStepper: Step must not be called when StepWithID is implemented")
}

// StepWithID derives the hidden purely from the id (the same counter identity), so the
// embedding argument is deliberately unused — the point is that the id reaches the step.
func (s idStepper) StepWithID(id int32, emb []byte) ([]byte, error) {
	if s.ids != nil {
		*s.ids = append(*s.ids, id)
	}
	out := make([]byte, s.dModel*bf16Size)
	out[0], out[1] = f32ToBF16Bytes(float32(id))
	return out, nil
}

// idSessionModel offers a StepWithID-aware session (and an erroring DecodeForward so a
// passing generation proves the session path ran).
type idSessionModel struct {
	counterModel
	ids *[]int32
}

func (idSessionModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return nil, core.NewError("whole-seq path must not run when a session is available")
}

func (m idSessionModel) OpenSession() (DecodeStepper, error) {
	return idStepper{ids: m.ids, vocab: m.vocab, dModel: m.dModel}, nil
}

func TestGenerate_StepWithID(t *testing.T) {
	var ids []int32
	m := idSessionModel{counterModel: counterModel{vocab: 16, dModel: 4}, ids: &ids}

	got, err := Generate(m, []int32{0}, 4, -1)
	if err != nil {
		t.Fatalf("Generate (StepWithID path): %v", err)
	}
	if want := []int32{1, 2, 3, 4}; !idsEqual(got, want) {
		t.Fatalf("StepWithID count = %v, want %v", got, want)
	}
	// every prompt id (0) + every generated id except the last (which is produced but
	// not stepped, generation having stopped) must have reached StepWithID: 0,1,2,3.
	if want := []int32{0, 1, 2, 3}; !idsEqual(ids, want) {
		t.Fatalf("StepWithID saw ids %v, want %v (Generate must route through StepWithID, not Step)", ids, want)
	}
}

// batchStepper is counterStepper's batch-prefill-capable sibling: PrefillBatch is the identity on the
// LAST embedding (the counter is memoryless, so that's exactly what len(embs) sequential Step calls would
// leave as the running hidden), and it records every PrefillBatch/Step call it sees — a passing generation
// with the right call counts proves generateStepwiseWithSession's prefill walk used PrefillBatch ONCE for
// the whole prompt and reserved Step for the genuine per-token decode steps that follow, not the other way
// round.
type batchStepper struct {
	prefillCalls *int
	prefillLens  *[]int
	stepCalls    *int
}

func (s batchStepper) Step(emb []byte) ([]byte, error) {
	if s.stepCalls != nil {
		*s.stepCalls++
	}
	return emb, nil // identity, same as counterStepper — the counter is memoryless
}

func (s batchStepper) PrefillBatch(embs [][]byte) ([]byte, error) {
	if s.prefillCalls != nil {
		*s.prefillCalls++
	}
	if s.prefillLens != nil {
		*s.prefillLens = append(*s.prefillLens, len(embs))
	}
	if len(embs) == 0 {
		return nil, core.NewError("batchStepper: PrefillBatch called with an empty batch")
	}
	return embs[len(embs)-1], nil
}

// batchSessionModel is counterModel plus a BatchPrefillStepper-capable session — with an erroring
// DecodeForward so a passing generation proves the incremental session path (not the whole-seq fallback)
// ran.
type batchSessionModel struct {
	counterModel
	prefillCalls *int
	prefillLens  *[]int
	stepCalls    *int
}

func (batchSessionModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return nil, core.NewError("whole-seq path must not run when a session is available")
}

func (m batchSessionModel) OpenSession() (DecodeStepper, error) {
	return batchStepper{prefillCalls: m.prefillCalls, prefillLens: m.prefillLens, stepCalls: m.stepCalls}, nil
}

// TestGenerate_BatchPrefillStepper is BatchPrefillStepper's behavioural receipt: given a 3-token prompt
// and 4 requested tokens, PrefillBatch must be called EXACTLY once (with all 3 prompt embeddings, not one
// call per token) — the fix's whole point, that a per-Step device fusion no longer fires once per prefill
// token. Step must still be called for every GENUINE decode step: maxNew-1 times (the transition from
// each generated token to the next; the last generated token is produced but never stepped, generation
// having stopped — the same off-by-one TestGenerate_StepWithID's own count documents). The generated
// sequence must also be byte-identical to the pre-existing per-token walk (sessionCounterModel), proving
// the batching changes HOW the prompt reaches the cache, not WHAT gets generated.
func TestGenerate_BatchPrefillStepper(t *testing.T) {
	var prefillCalls, stepCalls int
	var prefillLens []int
	m := batchSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		prefillCalls: &prefillCalls, prefillLens: &prefillLens, stepCalls: &stepCalls,
	}
	prompt := []int32{0, 1, 2}
	const maxNew = 4

	got, err := Generate(m, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate (BatchPrefillStepper path): %v", err)
	}
	if want := []int32{3, 4, 5, 6}; !idsEqual(got, want) {
		t.Fatalf("batch-prefill count = %v, want %v", got, want)
	}
	if prefillCalls != 1 {
		t.Fatalf("PrefillBatch called %d times, want exactly 1 (the whole prompt in ONE call)", prefillCalls)
	}
	if len(prefillLens) != 1 || prefillLens[0] != len(prompt) {
		t.Fatalf("PrefillBatch saw batch lengths %v, want a single call of length %d", prefillLens, len(prompt))
	}
	if want := maxNew - 1; stepCalls != want {
		t.Fatalf("Step called %d times, want exactly %d (once per genuine decode transition, zero for prefill)", stepCalls, want)
	}

	// Output-identical to the pre-existing per-token walk over the same prompt — batching the prefill
	// must not change WHAT is generated.
	perTokenWalk, err := Generate(sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}, opened: new(int), closed: new(int)}, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate (per-token walk): %v", err)
	}
	if !idsEqual(got, perTokenWalk) {
		t.Fatalf("batch-prefill result %v != per-token-walk result %v", got, perTokenWalk)
	}
}

// errModel is a TokenModel with one injectable failure point — embed/decode/head — for
// driving every error return in the whole-sequence generate loop. failAt names which call
// errors (and embedAfter delays the embed failure to the RE-embed of a generated token).
type errModel struct {
	counterModel
	failAt     string // "embed" | "decode" | "head" | "nohidden"
	embedAfter int    // for failAt=="embed": fail only once this many embeds have happened
	embeds     int
}

func (m *errModel) Embed(id int32) ([]byte, error) {
	m.embeds++
	if m.failAt == "embed" && m.embeds > m.embedAfter {
		return nil, core.NewError("injected embed error")
	}
	return m.counterModel.Embed(id)
}

func (m *errModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	if m.failAt == "decode" {
		return nil, core.NewError("injected decode error")
	}
	if m.failAt == "nohidden" {
		return [][]byte{}, nil // backend returned no hidden states
	}
	return m.counterModel.DecodeForward(inputs)
}

func (m *errModel) Head(hidden []byte) ([]byte, error) {
	if m.failAt == "head" {
		return nil, core.NewError("injected head error")
	}
	return m.counterModel.Head(hidden)
}

func TestGenerate_WholeSeqErrors(t *testing.T) {
	base := func() counterModel { return counterModel{vocab: 16, dModel: 4} }
	cases := []struct {
		name string
		m    *errModel
	}{
		{"embed (prompt)", &errModel{counterModel: base(), failAt: "embed", embedAfter: 0}},
		{"re-embed (generated token)", &errModel{counterModel: base(), failAt: "embed", embedAfter: 1}},
		{"decode", &errModel{counterModel: base(), failAt: "decode"}},
		{"no hidden states", &errModel{counterModel: base(), failAt: "nohidden"}},
		{"head", &errModel{counterModel: base(), failAt: "head"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Generate(c.m, []int32{0}, 4, -1); err == nil {
				t.Fatalf("expected an error from the %s failure point", c.name)
			}
		})
	}
}

// errStepper is the session counterpart: an injectable failure for the incremental path.
type errStepper struct {
	dModel int
	failAt string // "step" | ""
	steps  int
	failOn int // fail the step at this 1-based count
}

func (s *errStepper) Step(emb []byte) ([]byte, error) {
	s.steps++
	if s.failAt == "step" && s.steps >= s.failOn {
		return nil, core.NewError("injected step error")
	}
	return emb, nil
}

// errSessionModel injects failures into the incremental path: a failing OpenSession, or a
// session whose Embed/Head/Step error.
type errSessionModel struct {
	counterModel
	failAt string // "open" | "embed" | "head" | "step"
	stepOn int
	embeds int
}

func (m *errSessionModel) Embed(id int32) ([]byte, error) {
	m.embeds++
	if m.failAt == "embed" {
		return nil, core.NewError("injected session embed error")
	}
	return m.counterModel.Embed(id)
}

func (m *errSessionModel) Head(hidden []byte) ([]byte, error) {
	if m.failAt == "head" {
		return nil, core.NewError("injected session head error")
	}
	return m.counterModel.Head(hidden)
}

func (m *errSessionModel) OpenSession() (DecodeStepper, error) {
	if m.failAt == "open" {
		return nil, core.NewError("injected OpenSession error")
	}
	return &errStepper{dModel: m.dModel, failAt: m.failAt, failOn: m.stepOn}, nil
}

func TestGenerate_StepwiseErrors(t *testing.T) {
	base := func() counterModel { return counterModel{vocab: 16, dModel: 4} }
	cases := []struct {
		name string
		m    *errSessionModel
	}{
		{"open session", &errSessionModel{counterModel: base(), failAt: "open"}},
		{"embed (prefill)", &errSessionModel{counterModel: base(), failAt: "embed"}},
		{"step (prefill)", &errSessionModel{counterModel: base(), failAt: "step", stepOn: 1}},
		{"head", &errSessionModel{counterModel: base(), failAt: "head"}},
		// a step that fails AFTER prefill + first head — the "cache the generated token" step.
		{"step (generated)", &errSessionModel{counterModel: base(), failAt: "step", stepOn: 2}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var _ SessionModel = c.m // compile-time: drives the incremental path
			if _, err := Generate(c.m, []int32{0}, 4, -1); err == nil {
				t.Fatalf("expected an error from the %s failure point", c.name)
			}
		})
	}
}

// errPickModel pairs a healthy model with a pick func that errors, covering the pick error
// returns in both loops (the sampler/argmax failing mid-generation).
func TestGenerate_PickError(t *testing.T) {
	boom := func(logits []byte, vocab int) (int32, error) {
		return 0, core.NewError("injected pick error")
	}
	// whole-seq path
	if _, err := generate(counterModel{vocab: 16, dModel: 4}, []int32{0}, 4, -1, boom); err == nil {
		t.Fatal("whole-seq: expected a pick error")
	}
	// incremental path (the session model dispatches to generateStepwise)
	sm := sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}}
	if _, err := generate(sm, []int32{0}, 4, -1, boom); err == nil {
		t.Fatal("stepwise: expected a pick error")
	}
}

// TestGenerate_StepwiseEOS exercises the eos early-stop inside the incremental path (the
// whole-seq eos is covered by TestGenerate_CounterLoop; this is its stepwise twin).
func TestGenerate_StepwiseEOS(t *testing.T) {
	sm := sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}}
	got, err := Generate(sm, []int32{0}, 10, 3) // count 1,2,3 then eos at 3
	if err != nil {
		t.Fatalf("Generate stepwise eos: %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(got, want) {
		t.Fatalf("stepwise eos count = %v, want %v", got, want)
	}
}

// TestToken_Generate_Good covers the ordinary greedy whole-sequence loop: each generated
// token is re-embedded and fed into the next decode step (the counterModel's count only
// stays clean if Generate does this correctly).
func TestToken_Generate_Good(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := Generate(m, []int32{0}, 4, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := []int32{1, 2, 3, 4}; !idsEqual(got, want) {
		t.Fatalf("Generate = %v, want %v", got, want)
	}
}

// TestToken_Generate_Bad covers the three input-validation guards: a nil model, an empty
// prompt, and maxNew<=0 are each a clean error before any decode runs.
func TestToken_Generate_Bad(t *testing.T) {
	m := counterModel{vocab: 8, dModel: 2}
	if _, err := Generate(nil, []int32{0}, 4, -1); err == nil {
		t.Fatal("Generate(nil model): expected an error")
	}
	if _, err := Generate(m, nil, 4, -1); err == nil {
		t.Fatal("Generate(empty prompt): expected an error")
	}
	if _, err := Generate(m, []int32{0}, 0, -1); err == nil {
		t.Fatal("Generate(maxNew<=0): expected an error")
	}
}

// TestToken_Generate_Ugly covers the eos-disabled AND the eos-mid-generation edges
// together: eos<0 runs the full maxNew with no early stop, while a reachable eos ends
// generation the instant it's produced, even with a much larger maxNew budget.
func TestToken_Generate_Ugly(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	noEOS, err := Generate(m, []int32{0}, 3, -1)
	if err != nil {
		t.Fatalf("Generate(eos disabled): %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(noEOS, want) {
		t.Fatalf("Generate(eos disabled) = %v, want %v (full maxNew budget)", noEOS, want)
	}
	stopped, err := Generate(m, []int32{0}, 100, 3)
	if err != nil {
		t.Fatalf("Generate(eos=3): %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(stopped, want) {
		t.Fatalf("Generate(eos=3) = %v, want early stop at %v despite maxNew=100", stopped, want)
	}
}

// TestToken_GenerateSampled_Good covers zero-temperature sampling: it must reproduce the
// greedy sequence exactly (the documented fallback), proving the sampler's temp<=0 path
// is truly deterministic.
func TestToken_GenerateSampled_Good(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	greedy, err := Generate(m, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	sampled, err := GenerateSampled(m, NewSampler(9), SampleParams{Temperature: 0}, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("GenerateSampled: %v", err)
	}
	if !idsEqual(greedy, sampled) {
		t.Fatalf("GenerateSampled(temp=0) = %v, want greedy %v", sampled, greedy)
	}
}

// TestToken_GenerateSampled_Bad covers the nil-sampler guard: GenerateSampled requires a
// *Sampler (unlike greedy Generate, which needs none).
func TestToken_GenerateSampled_Bad(t *testing.T) {
	m := counterModel{vocab: 8, dModel: 2}
	if _, err := GenerateSampled(m, nil, SampleParams{}, []int32{0}, 4, -1); err == nil {
		t.Fatal("GenerateSampled(nil sampler): expected an error")
	}
}

// TestToken_GenerateSampled_Ugly covers the eos-reached case: eos>=0 folds into a
// single-element stop-token set AND clears MinTokensBeforeStop (so an eos request never
// gets suppressed by a stale MinTokensBeforeStop), stopping generation the moment eos is
// produced.
func TestToken_GenerateSampled_Ugly(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampled(m, NewSampler(1), SampleParams{Temperature: 0, MinTokensBeforeStop: 50}, []int32{0}, 100, 3)
	if err != nil {
		t.Fatalf("GenerateSampled: %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampled(eos=3) = %v, want early stop at %v (MinTokensBeforeStop must not suppress eos itself)", got, want)
	}
}

// TestToken_GenerateSampledWithStopTokens_Good covers the multi-stop-token set: greedy
// generation stops the instant ANY of the declared stop tokens is produced.
func TestToken_GenerateSampledWithStopTokens_Good(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampledWithStopTokens(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 10, []int32{4, 2})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens: %v", err)
	}
	if want := []int32{1, 2}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampledWithStopTokens = %v, want %v (stops at the first matching stop token, 2)", got, want)
	}
}

// TestToken_GenerateSampledWithStopTokens_Bad covers the nil-sampler guard shared with
// GenerateSampled.
func TestToken_GenerateSampledWithStopTokens_Bad(t *testing.T) {
	m := counterModel{vocab: 8, dModel: 2}
	if _, err := GenerateSampledWithStopTokens(m, nil, SampleParams{}, []int32{0}, 4, []int32{1}); err == nil {
		t.Fatal("GenerateSampledWithStopTokens(nil sampler): expected an error")
	}
}

// TestToken_GenerateSampledWithStopTokens_Ugly covers an empty stop-token set: it must
// behave exactly like no stop tokens at all (run the full maxNew), not error or stop
// immediately on a zero-length slice.
func TestToken_GenerateSampledWithStopTokens_Ugly(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampledWithStopTokens(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 3, []int32{})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens(empty stop set): %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampledWithStopTokens(empty stop set) = %v, want the full maxNew %v", got, want)
	}
}

// TestToken_GenerateSampledWithStopTokensTransform_Good covers the committed-token
// transform: the TRANSFORMED id (not the sampled one) is what gets returned AND fed into
// the next decode step.
func TestToken_GenerateSampledWithStopTokensTransform_Good(t *testing.T) {
	var stepped []int32
	m := idSessionModel{counterModel: counterModel{vocab: 16, dModel: 4}, ids: &stepped}
	got, err := GenerateSampledWithStopTokensTransform(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 2, nil, func(id int32) int32 {
		if id == 1 {
			return 9
		}
		return id
	})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransform: %v", err)
	}
	if want := []int32{9, 10}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampledWithStopTokensTransform = %v, want %v (transformed 1→9, then counting from 9)", got, want)
	}
}

// TestToken_GenerateSampledWithStopTokensTransform_Bad covers a nil transform: it must
// behave identically to GenerateSampledWithStopTokens (the transform is optional).
func TestToken_GenerateSampledWithStopTokensTransform_Bad(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	withNilTransform, err := GenerateSampledWithStopTokensTransform(m, NewSampler(2), SampleParams{Temperature: 0}, []int32{0}, 3, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransform(nil transform): %v", err)
	}
	plain, err := GenerateSampledWithStopTokens(m, NewSampler(2), SampleParams{Temperature: 0}, []int32{0}, 3, nil)
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokens: %v", err)
	}
	if !idsEqual(withNilTransform, plain) {
		t.Fatalf("nil-transform result %v != the no-transform sibling %v", withNilTransform, plain)
	}
}

// TestToken_GenerateSampledWithStopTokensTransform_Ugly covers a transform that maps a
// generated token onto a DECLARED STOP token: the transform runs BEFORE the stop check,
// so generation stops on the transformed id even though the sampler never produced it
// directly.
func TestToken_GenerateSampledWithStopTokensTransform_Ugly(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	got, err := GenerateSampledWithStopTokensTransform(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 10, []int32{5}, func(id int32) int32 {
		if id == 2 {
			return 5 // remap the 2nd generated token onto the stop id
		}
		return id
	})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransform: %v", err)
	}
	if want := []int32{1, 5}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampledWithStopTokensTransform = %v, want %v (stops on the TRANSFORMED id)", got, want)
	}
}

// TestToken_GenerateSampledWithStopTokensTransformEach_Good covers the streaming sibling:
// yield receives each committed token as it is produced, and returning true throughout
// lets generation run to completion with the SAME tokens the batch path returns.
func TestToken_GenerateSampledWithStopTokensTransformEach_Good(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	var streamed []int32
	got, err := GenerateSampledWithStopTokensTransformEach(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 3, nil, nil, func(id int32) bool {
		streamed = append(streamed, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransformEach: %v", err)
	}
	if !idsEqual(got, streamed) {
		t.Fatalf("batch result %v != streamed tokens %v", got, streamed)
	}
	if want := []int32{1, 2, 3}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampledWithStopTokensTransformEach = %v, want %v", got, want)
	}
}

// TestToken_GenerateSampledWithStopTokensTransformEach_Bad covers the nil-sampler guard
// shared by the whole GenerateSampled* family.
func TestToken_GenerateSampledWithStopTokensTransformEach_Bad(t *testing.T) {
	m := counterModel{vocab: 8, dModel: 2}
	if _, err := GenerateSampledWithStopTokensTransformEach(m, nil, SampleParams{}, []int32{0}, 4, nil, nil, nil); err == nil {
		t.Fatal("GenerateSampledWithStopTokensTransformEach(nil sampler): expected an error")
	}
}

// TestToken_GenerateSampledWithStopTokensTransformEach_Ugly covers yield returning FALSE
// mid-generation: it must end generation EARLY (fewer tokens than maxNew), the same
// early-out a cancelled streaming consumer relies on.
func TestToken_GenerateSampledWithStopTokensTransformEach_Ugly(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	seen := 0
	got, err := GenerateSampledWithStopTokensTransformEach(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 10, nil, nil, func(id int32) bool {
		seen++
		return seen < 2 // stop after the second committed token
	})
	if err != nil {
		t.Fatalf("GenerateSampledWithStopTokensTransformEach: %v", err)
	}
	if want := []int32{1, 2}; !idsEqual(got, want) {
		t.Fatalf("GenerateSampledWithStopTokensTransformEach(yield stops early) = %v, want %v", got, want)
	}
}

// TestGenerateSampledFromEmbeddingsEach_Good pins the multimodal prefill entry: ALREADY-COMPUTED rows
// prefill the session in ONE PrefillBatch call and the decode walk continues exactly as the id path —
// so a spliced prompt's rows (image features over placeholder positions) reach the stack without any
// prompt id being re-embedded.
func TestGenerateSampledFromEmbeddingsEach_Good(t *testing.T) {
	var prefillCalls, stepCalls int
	var prefillLens []int
	m := batchSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		prefillCalls: &prefillCalls, prefillLens: &prefillLens, stepCalls: &stepCalls,
	}
	ids := []int32{0, 1, 2}
	rows := make([][]byte, len(ids))
	for i, id := range ids {
		row, err := m.Embed(id)
		if err != nil {
			t.Fatalf("Embed(%d): %v", id, err)
		}
		rows[i] = row
	}
	got, err := GenerateSampledFromEmbeddingsEach(m, NewSampler(0), SampleParams{}, ids, rows, 4, nil, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromEmbeddingsEach: %v", err)
	}
	if want := []int32{3, 4, 5, 6}; !idsEqual(got, want) {
		t.Fatalf("rows-prefill walk = %v, want %v (identical to the id path)", got, want)
	}
	if prefillCalls != 1 || len(prefillLens) != 1 || prefillLens[0] != len(ids) {
		t.Fatalf("PrefillBatch calls %d lens %v, want exactly one call of %d rows", prefillCalls, prefillLens, len(ids))
	}
}

// TestGenerateSampledFromEmbeddingsEach_Bad pins every refusal: nil model, nil sampler, empty rows,
// id/row count mismatch, non-positive maxNew, and a session that cannot batch-prefill rows (no
// BatchPrefillStepper — rows cannot replay per-id, so the entry must fail rather than silently
// re-embedding token ids).
func TestGenerateSampledFromEmbeddingsEach_Bad(t *testing.T) {
	m := batchSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		prefillCalls: new(int), prefillLens: new([]int), stepCalls: new(int),
	}
	ids := []int32{0}
	rows := [][]byte{{0, 0, 0, 0, 0, 0, 0, 0}}
	if _, err := GenerateSampledFromEmbeddingsEach(nil, NewSampler(0), SampleParams{}, ids, rows, 4, nil, nil, nil); err == nil {
		t.Fatal("nil model must refuse")
	}
	if _, err := GenerateSampledFromEmbeddingsEach(m, nil, SampleParams{}, ids, rows, 4, nil, nil, nil); err == nil {
		t.Fatal("nil sampler must refuse")
	}
	if _, err := GenerateSampledFromEmbeddingsEach(m, NewSampler(0), SampleParams{}, nil, nil, 4, nil, nil, nil); err == nil {
		t.Fatal("empty rows must refuse")
	}
	if _, err := GenerateSampledFromEmbeddingsEach(m, NewSampler(0), SampleParams{}, []int32{0, 1}, rows, 4, nil, nil, nil); err == nil {
		t.Fatal("id/row count mismatch must refuse")
	}
	if _, err := GenerateSampledFromEmbeddingsEach(m, NewSampler(0), SampleParams{}, ids, rows, 0, nil, nil, nil); err == nil {
		t.Fatal("maxNew 0 must refuse")
	}
	perToken := sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}, opened: new(int), closed: new(int)}
	if _, err := GenerateSampledFromEmbeddingsEach(perToken, NewSampler(0), SampleParams{}, ids, rows, 4, nil, nil, nil); err == nil {
		t.Fatal("a session without BatchPrefillStepper must refuse rows")
	}
}

// TestGenerateSampledFromEmbeddingsEach_Ugly pins the stream-cut contract: a yield returning false after
// the first token ends generation early with the committed prefix and no error — the same early-exit
// shape as the id path.
func TestGenerateSampledFromEmbeddingsEach_Ugly(t *testing.T) {
	m := batchSessionModel{
		counterModel: counterModel{vocab: 16, dModel: 4},
		prefillCalls: new(int), prefillLens: new([]int), stepCalls: new(int),
	}
	ids := []int32{0, 1, 2}
	rows := make([][]byte, len(ids))
	for i, id := range ids {
		rows[i], _ = m.Embed(id)
	}
	got, err := GenerateSampledFromEmbeddingsEach(m, NewSampler(0), SampleParams{}, ids, rows, 4, nil, nil, func(int32) bool { return false })
	if err != nil {
		t.Fatalf("early-cut generate: %v", err)
	}
	if want := []int32{3}; !idsEqual(got, want) {
		t.Fatalf("early-cut walk = %v, want %v (one committed token)", got, want)
	}
}
