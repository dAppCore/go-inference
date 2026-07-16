// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/mtp"
)

// mtp_test.go is the fixture-level correctness receipt for the composed Qwen 3.5/3.6 MTP pairing — the
// composed twin of engine/metal's gemma4 speculative parity tests, without a giant real checkpoint. It
// proves the load-bearing property the whole port exists for: GREEDY PAIRED OUTPUT IS BYTE-IDENTICAL TO
// GREEDY NON-SPECULATIVE DECODE, for a perfect drafter, an adversarial drafter, and the real MTP head —
// and that speculation actually engages (drafts proposed, accepted/rejected counted).

// mkMTPHead builds a synthetic Qwen MTP head for base: mtp_num_hidden_layers full-attention composed layers
// (D matching the base), the fc input combiner [D,2D], the two pre-fc RMSNorms and the head final norm.
// It shares nothing with the base at construction — the pairing shares the base's embed + LM head at draft
// time (draftHidden/draftLogits), exactly as the real head does (no embed_tokens / lm_head of its own).
func mkMTPHead(base *ComposedModel, nLayers, seed int) *MTPHead {
	D, FF := base.D, 16
	atCfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	layers := make([]Layer, nLayers)
	for li := range layers {
		layers[li] = Layer{
			InputNorm:    syn(D, seed+li*17+1),
			Mixer:        mkAttnMixer(atCfg, D, seed+li*17+2),
			PostAttnNorm: syn(D, seed+li*17+3),
			MLP:          &MLP{Gate: syn(FF*D, seed+li*17+4), Up: syn(FF*D, seed+li*17+5), Down: syn(D*FF, seed+li*17+6), FF: FF},
		}
	}
	return &MTPHead{
		Stack: &ComposedModel{Layers: layers, D: D, Vocab: base.Vocab, Eps: 1e-5},
		FC:    syn(D*2*D, seed+901),
		Enorm: syn(D, seed+902),
		Hnorm: syn(D, seed+903),
		Norm:  syn(D, seed+904),
		D:     D,
		Eps:   1e-5,
	}
}

// oracleDrafter is a PERFECT drafter: it replays the base's own greedy continuation, so every proposal is
// accepted. seq is base.Generate's greedy output; pos counts the committed tokens (one observe per commit),
// so at draft time seq[pos] is the boundary token the loop passes in and the proposal is the continuation
// AFTER it.
type oracleDrafter struct {
	seq []int32
	pos int
}

func (o *oracleDrafter) reset([]int32, []float32) error { return nil }

func (o *oracleDrafter) observe(int32, []float32) error {
	o.pos++
	return nil
}

func (o *oracleDrafter) draftBlock(tok int32, _ []float32, k int) ([]int32, error) {
	start := o.pos + 1 // tok == seq[o.pos] (the token about to commit); the continuation starts after it
	if start >= len(o.seq) {
		return nil, nil
	}
	end := start + k
	if end > len(o.seq) {
		end = len(o.seq)
	}
	return append([]int32(nil), o.seq[start:end]...), nil
}

// adversaryDrafter is a WORST-CASE drafter: every proposal is (greedy+1) mod vocab, so the first draft of
// every block mismatches the base's next greedy and the whole block is rejected — the base commits its own
// token each round. Proves byte-identity survives a drafter that is always wrong. pos counts commits via
// observe, so seq[pos+1] is the greedy the first draft must sabotage.
type adversaryDrafter struct {
	seq   []int32
	pos   int
	vocab int
}

func (a *adversaryDrafter) reset([]int32, []float32) error { return nil }

func (a *adversaryDrafter) observe(int32, []float32) error {
	a.pos++
	return nil
}

func (a *adversaryDrafter) draftBlock(_ int32, _ []float32, k int) ([]int32, error) {
	next := a.pos + 1 // the greedy AFTER the committing boundary token — the one to contradict
	if next >= len(a.seq) {
		return nil, nil
	}
	bad := (a.seq[next] + 1) % int32(a.vocab) // != seq[next], for vocab > 1
	ds := make([]int32, k)
	for i := range ds {
		ds[i] = bad
	}
	return ds, nil
}

func sameSeq(a, b []int32) bool {
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

// pairWith swaps in a test drafter while keeping the pair's validated base + head.
func pairWith(t *testing.T, base *ComposedModel, head *MTPHead, d drafter) *SpeculativePair {
	t.Helper()
	p, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair: %v", err)
	}
	p.drafter = d
	return p
}

// TestMTPHeadDraftBlockGood covers the head forward through its trained-shape driver: a real MTP head,
// prefilled over the prompt's (t_{i+1}, h_i) pairs, chains a full block of in-range draft token ids from
// the boundary pair — the standalone-invalid, paired-valid forward the whole feature turns on. It also
// pins the drafter's speculative discipline: drafting must not advance the live head session (a second
// identical draft is byte-identical).
func TestMTPHeadDraftBlockGood(t *testing.T) {
	base := mkHybridComposedModel(8, 32, 16)
	head := mkMTPHead(base, 1, 500)
	prompt := []int32{1, 5, 9, 2}
	sess := NewSession(base)
	hidden, err := sess.forward(prompt)
	if err != nil {
		t.Fatalf("base prefill: %v", err)
	}
	d := &headDrafter{head: head, base: base}
	if err := d.reset(prompt, hidden); err != nil {
		t.Fatalf("reset: %v", err)
	}
	boundary := hidden[(len(prompt)-1)*base.D:]
	g := argmaxF32(sess.headLogits(boundary)) // the base's next token — the boundary pair's token half
	drafts, err := d.draftBlock(g, boundary, 4)
	if err != nil {
		t.Fatalf("draftBlock: %v", err)
	}
	if len(drafts) != 4 {
		t.Fatalf("draftBlock proposed %d tokens, want 4", len(drafts))
	}
	for i, tok := range drafts {
		if tok < 0 || int(tok) >= base.Vocab {
			t.Fatalf("draft[%d] = %d out of vocab [0,%d)", i, tok, base.Vocab)
		}
	}
	again, err := d.draftBlock(g, boundary, 4)
	if err != nil {
		t.Fatalf("draftBlock (repeat): %v", err)
	}
	if !sameSeq(drafts, again) {
		t.Fatalf("drafting advanced the live head session: first %v, repeat %v", drafts, again)
	}
	t.Logf("MTP head drafted a valid block %v from the boundary pair (repeat identical)", drafts)
}

// TestGenerateSpeculativeFeedsProducingHiddenPairs pins the TRAINED-SHAPE alignment (the vLLM/EAGLE
// "shift input ids by one" contract): the drafter must be fed each committed token WITH the base hidden
// that PRODUCED it — reset with the prompt tokens + all prompt hiddens, then one observe per commit
// pairing token i with the boundary hidden BEFORE token i was forwarded. A recording drafter captures
// every feed; a manual plain-decode replay computes the ground-truth producing hiddens; the two must
// match float-exactly (same math, same order). An off-by-one (pairing a token with the hidden AFTER
// forwarding it) fails this test immediately.
func TestGenerateSpeculativeFeedsProducingHiddenPairs(t *testing.T) {
	base := mkComposedModel(3, 8, 32, 16)
	head := mkMTPHead(base, 1, 950)
	prompt := []int32{7, 1, 4}
	const maxNew = 8
	D := base.D

	rec := &recordingDrafter{}
	p := pairWith(t, base, head, rec)
	got, _, err := p.GenerateSpeculative(prompt, maxNew, -1, 3)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}

	// Ground truth by manual replay: the boundary hidden before each commit is the hidden that
	// produced that commit's token.
	sess := NewSession(base)
	hidden, err := sess.forward(prompt)
	if err != nil {
		t.Fatalf("replay prefill: %v", err)
	}
	if !sameSeq(rec.resetTokens, prompt) {
		t.Fatalf("reset tokens = %v, want the prompt %v", rec.resetTokens, prompt)
	}
	if len(rec.resetHiddens) != len(prompt)*D {
		t.Fatalf("reset hiddens = %d floats, want %d (all prompt rows)", len(rec.resetHiddens), len(prompt)*D)
	}
	for i := range rec.resetHiddens {
		if rec.resetHiddens[i] != hidden[i] {
			t.Fatalf("reset hidden float %d = %g, want %g (the prompt forward's own rows)", i, rec.resetHiddens[i], hidden[i])
		}
	}
	boundary := append([]float32(nil), hidden[(len(prompt)-1)*D:]...)
	// The final token is emitted without an observe (generation stops before the base advances).
	wantPairs := len(got) - 1
	if len(rec.pairs) != wantPairs {
		t.Fatalf("observed %d pairs, want %d (one per commit except the stopping token)", len(rec.pairs), wantPairs)
	}
	for i := 0; i < wantPairs; i++ {
		if rec.pairs[i].tok != got[i] {
			t.Fatalf("pair %d token = %d, want committed token %d", i, rec.pairs[i].tok, got[i])
		}
		for j := range boundary {
			if rec.pairs[i].hidden[j] != boundary[j] {
				t.Fatalf("pair %d hidden float %d = %g, want %g — the drafter was not fed the hidden that PRODUCED token %d",
					i, j, rec.pairs[i].hidden[j], boundary[j], got[i])
			}
		}
		next, ferr := sess.forward([]int32{got[i]})
		if ferr != nil {
			t.Fatalf("replay forward(%d): %v", got[i], ferr)
		}
		boundary = append(boundary[:0], next...)
	}
	t.Logf("trained-shape alignment held over %d committed pairs", wantPairs)
}

// recordingDrafter captures every feed the verify loop makes — the instrument behind
// TestGenerateSpeculativeFeedsProducingHiddenPairs. It never proposes, so the loop degrades to plain
// per-token commits (every commit is a replacement).
type recordingDrafter struct {
	resetTokens  []int32
	resetHiddens []float32
	pairs        []recordedPair
}

type recordedPair struct {
	tok    int32
	hidden []float32
}

func (r *recordingDrafter) reset(tokens []int32, hiddens []float32) error {
	r.resetTokens = append([]int32(nil), tokens...)
	r.resetHiddens = append([]float32(nil), hiddens...)
	return nil
}

func (r *recordingDrafter) observe(tok int32, producingHidden []float32) error {
	r.pairs = append(r.pairs, recordedPair{tok: tok, hidden: append([]float32(nil), producingHidden...)})
	return nil
}

func (r *recordingDrafter) draftBlock(int32, []float32, int) ([]int32, error) { return nil, nil }

// TestSpeculativeGreedyParityRealHead is the headline receipt: the real MTP head paired to its base, run
// speculatively, emits EXACTLY what plain greedy decode emits — byte-identical — while genuinely engaging
// (drafts proposed). Acceptance is whatever a random-weight head earns; correctness does not depend on it.
func TestSpeculativeGreedyParityRealHead(t *testing.T) {
	base := mkHybridComposedModel(8, 32, 16)
	head := mkMTPHead(base, 1, 700)
	prompt := []int32{1, 2, 3, 4, 5}
	const maxNew, block = 24, 5

	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	p, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair: %v", err)
	}
	got, m, err := p.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("speculative output diverged from greedy:\n got  %v\n want %v", got, want)
	}
	if m.ProposedTokens == 0 {
		t.Fatal("speculation did not engage (0 tokens proposed) — the drafter never ran")
	}
	if m.AcceptedTokens+m.RejectedTokens != m.ProposedTokens {
		t.Fatalf("accept/reject bookkeeping: %d + %d != %d", m.AcceptedTokens, m.RejectedTokens, m.ProposedTokens)
	}
	t.Logf("real head: byte-identical over %d tokens; proposed=%d accepted=%d (%.0f%%) draftCalls=%d",
		len(got), m.ProposedTokens, m.AcceptedTokens, m.AcceptanceRate*100, m.DraftCalls)
}

// TestSpeculativeGreedyParityOracle proves the ACCEPT + full-accept BONUS branches: a perfect drafter is
// accepted every time, yet the emitted sequence is still exactly greedy decode (the drafter only decided
// how many of the base's own tokens committed per round, never which).
func TestSpeculativeGreedyParityOracle(t *testing.T) {
	base := mkComposedModel(3, 8, 32, 16)
	head := mkMTPHead(base, 1, 800)
	prompt := []int32{7, 1, 4}
	const maxNew, block = 20, 4

	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	p := pairWith(t, base, head, &oracleDrafter{seq: want})
	got, m, err := p.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("oracle speculative output diverged:\n got  %v\n want %v", got, want)
	}
	if m.ProposedTokens == 0 || m.AcceptedTokens != m.ProposedTokens || m.RejectedTokens != 0 {
		t.Fatalf("oracle drafter should accept every proposal: proposed=%d accepted=%d rejected=%d",
			m.ProposedTokens, m.AcceptedTokens, m.RejectedTokens)
	}
	t.Logf("oracle: byte-identical over %d tokens at 100%% acceptance (proposed=accepted=%d)", len(got), m.AcceptedTokens)
}

// TestSpeculativeGreedyParityAdversary proves the REJECT + REPLACEMENT branch: a drafter that is always
// wrong is rejected every round, and the base's own greedy token replaces each rejected draft — so the
// output is STILL byte-identical to plain greedy decode.
func TestSpeculativeGreedyParityAdversary(t *testing.T) {
	base := mkComposedModel(3, 8, 32, 16)
	head := mkMTPHead(base, 1, 810)
	prompt := []int32{2, 9, 6}
	const maxNew, block = 20, 4

	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	p := pairWith(t, base, head, &adversaryDrafter{seq: want, vocab: base.Vocab})
	got, m, err := p.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("adversary speculative output diverged:\n got  %v\n want %v", got, want)
	}
	if m.ProposedTokens == 0 || m.AcceptedTokens != 0 || m.RejectedTokens != m.ProposedTokens {
		t.Fatalf("adversary drafter should be rejected every time: proposed=%d accepted=%d rejected=%d",
			m.ProposedTokens, m.AcceptedTokens, m.RejectedTokens)
	}
	t.Logf("adversary: byte-identical over %d tokens at 0%% acceptance (proposed=%d, all replaced by base greedy)", len(got), m.ProposedTokens)
}

// TestSpeculativeGreedyParityEOS proves parity holds under early stop: a paired run with an eos token
// terminates on exactly the same token, at the same length, as plain greedy decode with that eos.
func TestSpeculativeGreedyParityEOS(t *testing.T) {
	base := mkHybridComposedModel(8, 32, 16)
	head := mkMTPHead(base, 1, 900)
	prompt := []int32{3, 1, 4, 1, 5}
	const maxNew, block = 32, 5

	full, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	if len(full) < 5 {
		t.Fatalf("need a longer greedy run to pick an eos; got %d tokens", len(full))
	}
	eos := int(full[3]) // stop the run partway through
	want, err := NewSession(base).Generate(prompt, maxNew, eos)
	if err != nil {
		t.Fatalf("base Generate(eos): %v", err)
	}
	p, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair: %v", err)
	}
	got, _, err := p.GenerateSpeculative(prompt, maxNew, eos, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative(eos): %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("speculative eos stop diverged:\n got  %v\n want %v", got, want)
	}
	if int(got[len(got)-1]) != eos {
		t.Fatalf("speculative run did not terminate on eos %d: %v", eos, got)
	}
	t.Logf("eos parity: both runs stopped on token %d at length %d", eos, len(got))
}

// TestNewSpeculativePairBad covers the attachment guard: a head whose hidden width disagrees with the base
// is rejected rather than mis-paired.
func TestNewSpeculativePairBad(t *testing.T) {
	base := mkComposedModel(2, 8, 32, 16)
	wide := mkMTPHead(base, 1, 111)
	wide.D = 16 // deliberately disagree with the base hidden size
	if _, err := NewSpeculativePair(base, wide); err == nil {
		t.Fatal("expected NewSpeculativePair to reject a head whose hidden_size != base hidden_size")
	}
	if _, err := NewSpeculativePair(base, nil); err == nil {
		t.Fatal("expected NewSpeculativePair to reject a nil head")
	}
}

// TestParseMTPAssistantConfigGood covers the reactive registration: the Qwen MTP model_type resolves in the
// neutral assistant registry (mtp.LookupAssistant) — it is a RECOGNISED drafter now, not a bare refusal —
// and its parser carries the base backbone hidden the pair validates against.
func TestParseMTPAssistantConfigGood(t *testing.T) {
	for _, id := range []string{"qwen3_5_mtp", "qwen3_5_mtp_text", "qwen3_6_mtp"} {
		if _, ok := mtp.LookupAssistant(id); !ok {
			t.Fatalf("mtp.LookupAssistant(%q) = not found, want the composed MTP drafter registered", id)
		}
	}
	cfg, err := ParseMTPAssistantConfig([]byte(`{"model_type":"qwen3_5_mtp","block_size":3,"text_config":{"hidden_size":5120,"mtp_num_hidden_layers":1,"num_attention_heads":24,"head_dim":256,"vocab_size":248320}}`))
	if err != nil {
		t.Fatalf("ParseMTPAssistantConfig: %v", err)
	}
	if cfg.ModelType != "qwen3_5_mtp" || cfg.BackboneHidden != 5120 || cfg.Method != mtp.MTPDraftModel {
		t.Fatalf("parsed config = %+v, want qwen3_5_mtp / backbone 5120 / draft-model", cfg)
	}
}

// TestParseMTPAssistantConfigBad covers the parser guards: a wrong model_type, an absent hidden_size and an
// absent mtp_num_hidden_layers are each a clean error rather than a half-built config.
func TestParseMTPAssistantConfigBad(t *testing.T) {
	cases := map[string]string{
		"wrong model_type": `{"model_type":"gemma4_assistant","text_config":{"hidden_size":8,"mtp_num_hidden_layers":1}}`,
		"no hidden_size":   `{"model_type":"qwen3_5_mtp","text_config":{"mtp_num_hidden_layers":1}}`,
		"no head depth":    `{"model_type":"qwen3_5_mtp","text_config":{"hidden_size":8}}`,
		"malformed json":   `{not json`,
	}
	for name, data := range cases {
		if _, err := ParseMTPAssistantConfig([]byte(data)); err == nil {
			t.Fatalf("%s: expected a parse error", name)
		}
	}
}

// TestSpeculativeBlockVerifyGreedyParityRealHead pins the block lane's mechanics on pure host math,
// where a batched forward row is arithmetically identical to a sequential step (per-row dot products,
// same order): BlockVerify output must byte-equal BOTH the per-token verify run and plain greedy
// decode. (With device GEMM hooks bound the lane is token-identity tier — a live-model property,
// documented on the BlockVerify field, deliberately not simulated here.)
func TestSpeculativeBlockVerifyGreedyParityRealHead(t *testing.T) {
	base := mkHybridComposedModel(8, 32, 16)
	head := mkMTPHead(base, 1, 700)
	prompt := []int32{1, 2, 3, 4, 5}
	const maxNew, block = 24, 5

	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	perToken, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair: %v", err)
	}
	gotSeq, _, err := perToken.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative (per-token): %v", err)
	}
	blockPair, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair (block): %v", err)
	}
	blockPair.BlockVerify = true
	gotBlock, m, err := blockPair.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative (block): %v", err)
	}
	if !sameSeq(gotBlock, want) {
		t.Fatalf("block-verify output diverged from plain greedy:\n got  %v\n want %v", gotBlock, want)
	}
	if !sameSeq(gotBlock, gotSeq) {
		t.Fatalf("block-verify output diverged from per-token verify:\n block %v\n seq   %v", gotBlock, gotSeq)
	}
	if m.ProposedTokens == 0 {
		t.Fatal("block-verify did not engage (0 proposed)")
	}
	t.Logf("block lane: byte-identical to plain greedy and per-token over %d tokens; forwards=%d proposed=%d accepted=%d",
		len(gotBlock), m.TargetVerifyCalls, m.ProposedTokens, m.AcceptedTokens)
}

// TestSpeculativeBlockVerifyOracleFewerForwards proves the lane's reason to exist: a perfect drafter
// under block verify commits k+1 tokens per SINGLE base forward — so the forward count comes in well
// under the token count (the per-token lane pays one forward per token). Output still byte-identical.
func TestSpeculativeBlockVerifyOracleFewerForwards(t *testing.T) {
	base := mkComposedModel(3, 8, 32, 16)
	head := mkMTPHead(base, 1, 800)
	prompt := []int32{7, 1, 4}
	const maxNew, block = 20, 4

	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	p := pairWith(t, base, head, &oracleDrafter{seq: want})
	p.BlockVerify = true
	got, m, err := p.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("oracle block-verify diverged:\n got  %v\n want %v", got, want)
	}
	if m.AcceptedTokens != m.ProposedTokens || m.RejectedTokens != 0 {
		t.Fatalf("oracle must be fully accepted: proposed=%d accepted=%d rejected=%d", m.ProposedTokens, m.AcceptedTokens, m.RejectedTokens)
	}
	if m.TargetVerifyCalls >= len(got) {
		t.Fatalf("block verify spent %d base forwards for %d tokens — no better than per-token", m.TargetVerifyCalls, len(got))
	}
	t.Logf("oracle block: %d tokens in %d base forwards (%.2f tokens/forward), byte-identical",
		len(got), m.TargetVerifyCalls, float64(len(got))/float64(m.TargetVerifyCalls))
}

// TestSpeculativeBlockVerifyAdversaryRestores drives the reject path every round: an always-wrong
// drafter forces snapshot → batched verify → restore → committed-prefix re-forward each time, and the
// output must STILL byte-equal plain greedy decode — the Snapshot/Restore rollback receipt.
func TestSpeculativeBlockVerifyAdversaryRestores(t *testing.T) {
	base := mkComposedModel(3, 8, 32, 16)
	head := mkMTPHead(base, 1, 810)
	prompt := []int32{2, 9, 6}
	const maxNew, block = 20, 4

	want, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	p := pairWith(t, base, head, &adversaryDrafter{seq: want, vocab: base.Vocab})
	p.BlockVerify = true
	got, m, err := p.GenerateSpeculative(prompt, maxNew, -1, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("adversary block-verify diverged (restore path broken?):\n got  %v\n want %v", got, want)
	}
	if m.AcceptedTokens != 0 || m.RejectedTokens != m.ProposedTokens {
		t.Fatalf("adversary must be fully rejected: proposed=%d accepted=%d rejected=%d", m.ProposedTokens, m.AcceptedTokens, m.RejectedTokens)
	}
	t.Logf("adversary block: byte-identical over %d tokens with every round restored (proposed=%d, all rejected)", len(got), m.ProposedTokens)
}

// TestSpeculativeBlockVerifyEOS proves the block lane stops on eos exactly as plain greedy does, even
// when the eos lands mid-committed-block.
func TestSpeculativeBlockVerifyEOS(t *testing.T) {
	base := mkHybridComposedModel(8, 32, 16)
	head := mkMTPHead(base, 1, 900)
	prompt := []int32{3, 1, 4, 1, 5}
	const maxNew, block = 32, 5

	full, err := NewSession(base).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("base Generate: %v", err)
	}
	if len(full) < 5 {
		t.Fatalf("need a longer greedy run to pick an eos; got %d tokens", len(full))
	}
	eos := int(full[3])
	want, err := NewSession(base).Generate(prompt, maxNew, eos)
	if err != nil {
		t.Fatalf("base Generate(eos): %v", err)
	}
	p, err := NewSpeculativePair(base, head)
	if err != nil {
		t.Fatalf("NewSpeculativePair: %v", err)
	}
	p.BlockVerify = true
	got, _, err := p.GenerateSpeculative(prompt, maxNew, eos, block)
	if err != nil {
		t.Fatalf("GenerateSpeculative(eos): %v", err)
	}
	if !sameSeq(got, want) {
		t.Fatalf("block-verify eos stop diverged:\n got  %v\n want %v", got, want)
	}
	if int(got[len(got)-1]) != eos {
		t.Fatalf("block-verify run did not terminate on eos %d: %v", eos, got)
	}
	t.Logf("block eos parity: both runs stopped on token %d at length %d", eos, len(got))
}
