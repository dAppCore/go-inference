// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/specctl"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mtp"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// mtp.go is the composed twin of engine/metal's gemma4 MTP pairing (assistant_load.go + speculative_model
// .go), ported to the HOST composed idiom because the Qwen 3.5/3.6 base runs host-side (ComposedModel /
// ComposedSession, model.SessionModel) rather than as a metal ArchSession. The metal AssistantPair shares
// the target's GPU K/V streams keyed by layer type; the composed hybrid's gated-delta layers thread a
// recurrent conv+delta state, not a truncatable K/V cache, so that machinery cannot host a composed target
// — the pairing is realised here, over the composed base's own forward, instead.
//
// The port keeps gemma4's LOGICAL contract exactly:
//   - a small transformer DRAFTS from the base's hidden states (MTPHead, below — the composed mirror of
//     gemma4's MTP-head draft);
//   - the base VERIFIES, accepting the longest draft prefix that matches its own greedy continuation and
//     committing its own token at the divergence (the accept/replacement discipline of
//     verifyDraftBlockFromSession);
//   - greedy paired output is byte-identical to greedy non-speculative decode.
//
// The drafter runs the TRAINED shape (the vLLM/EAGLE-family alignment the Qwen MTP head was trained
// under, vllm/v1/spec_decode — "Shift the input ids by one token"): head row i consumes the pair
// (t_{i+1}, h_i) — each token's embedding with the base hidden that PRODUCED that token — through
// fc([RMSNorm(embed); RMSNorm(hidden)]), and the head keeps a PERSISTENT session over the committed
// pairs at true positions (row i sits at position i because the head sequence always trails the base
// by exactly one row — implicit positions align by construction). Drafting runs on a throwaway clone
// of that session (Snapshot/Restore), so speculative rows never contaminate the live state; within a
// block, draft step j+1 chains on the head's OWN step-j hidden and token, exactly as the reference
// proposer loops.
//
// gemma4's documented parity hazard is that a BATCHED verify forward is not numerically identical to
// sequential decode on a quantised target, so the accepted-boundary token can drift; its answer is to
// re-derive (reforge) the committed boundary from the target's own decode. This host port takes the
// strongest form of that discipline: it advances the base ONE COMMITTED TOKEN AT A TIME through its
// ordinary decode forward, so every committed boundary IS a canonical single decode — the batched-verify
// drift class cannot arise, and byte-identity holds by construction on dense or quantised bases alike.
// (The remaining perf lever — verifying a whole draft block in ONE base forward with recurrent-state
// snapshot/restore on rejection — is the follow-up that converts this acceptance measurement into fewer
// base forwards; it is deliberately out of this correctness-first slice.)

// MTPHead is a loaded Qwen 3.5/3.6 multi-token-prediction drafter head. It has no standalone forward: it
// drafts from a base's hidden states and SHARES the base's token embedding and LM head (the real
// checkpoint carries neither embed_tokens nor lm_head — mtp_use_dedicated_embeddings / tie_word_embeddings
// both false). The head is a small pre-norm transformer:
//
//	x = fc([ RMSNorm(embed(t), Enorm) ; RMSNorm(h, Hnorm) ])   // combine token embed ⊕ the base hidden that produced it
//	h = Stack.forward(x)                                        // mtp_num_hidden_layers composed layers
//	logits = sharedHead( RMSNorm(h, Norm) )                    // head final norm → the BASE's LM head
//
// mirroring gemma4's [token embed ⊕ target hidden] → pre_projection → decode stack → head, with the Qwen
// difference that each half of the fc input is RMS-normed first (pre_fc_norm_embedding / _hidden). Stack is
// a layers-only ComposedModel so the head reuses composed's proven per-layer forward (attention/MLP, the
// device hooks) and threads its own K/V across the committed sequence.
type MTPHead struct {
	Stack *ComposedModel     // Layers only (mtp_num_hidden_layers deep); no Embed/Output/NormF — those are shared/head-local
	FC    []float32          // [D, 2D] input combiner: [normed embed ; normed hidden] → hidden (nil ⇒ FCQ packed)
	FCQ   *model.QuantWeight // packed fc (a quant checkpoint keeps it packed, matching the base's projections)
	Enorm []float32          // [D] pre_fc_norm_embedding
	Hnorm []float32          // [D] pre_fc_norm_hidden
	Norm  []float32          // [D] the head's final RMSNorm (before the shared LM head)
	D     int
	Eps   float32
	// BlockSize is the checkpoint's declared draft depth (top-level block_size) — the depth the head
	// was TRAINED to chain. It is the pair's default draft block; 0 means the checkpoint declared none.
	BlockSize int
}

// fcInput builds the head's stack input for ONE (token, producing-hidden) pair — see fcInputRows.
func (h *MTPHead) fcInput(base *ComposedModel, tok int32, hidden []float32) ([]float32, error) {
	return h.fcInputRows(base, []int32{tok}, hidden)
}

// fcInputRows builds the head's stack inputs for L (token, producing-hidden) pairs: row i =
// fc([ RMSNorm(embed(tokens[i]), Enorm) ; RMSNorm(hiddens[i·D:], Hnorm) ]). The [normed embed ;
// normed hidden] order and the per-half norms match the reference module exactly
// (pre_fc_norm_embedding on the embeds, pre_fc_norm_hidden on the hiddens, embeds FIRST — the
// concatenation order the fc's [D,2D] weight was trained under). base supplies the SHARED token
// embedding. hiddens is consumed synchronously and never retained.
func (h *MTPHead) fcInputRows(base *ComposedModel, tokens []int32, hiddens []float32) ([]float32, error) {
	if h == nil || base == nil {
		return nil, core.NewError("composed.mtp: fc input requires a head and a base")
	}
	L := len(tokens)
	if L == 0 {
		return nil, core.NewError("composed.mtp: fc input requires at least one pair")
	}
	if len(hiddens) != L*h.D {
		return nil, core.NewError("composed.mtp: fc input hidden rows disagree with the token count")
	}
	combined := make([]float32, L*2*h.D)
	embed := make([]float32, h.D)
	for i, tok := range tokens {
		if int(tok) < 0 || int(tok) >= base.Vocab {
			return nil, core.NewError("composed.mtp: fc input token out of range")
		}
		if err := base.embedRow(embed, int(tok)); err != nil {
			return nil, err
		}
		if base.EmbedScale != 0 && base.EmbedScale != 1 {
			for j := range embed {
				embed[j] *= base.EmbedScale
			}
		}
		row := combined[i*2*h.D : (i+1)*2*h.D]
		copy(row[:h.D], rmsNormRowsPlain(embed, h.Enorm, 1, h.D, h.Eps))
		copy(row[h.D:], rmsNormRowsPlain(hiddens[i*h.D:(i+1)*h.D], h.Hnorm, 1, h.D, h.Eps))
	}
	// fc projects [2D]→[D] per row; packed (quant checkpoint) rides the matvec seam, else dense matNT.
	if h.FCQ != nil {
		return matNTQuant(nil, combined, h.FCQ, L, 2*h.D, h.D), nil
	}
	return matNT(combined, h.FC, L, 2*h.D, h.D), nil
}

// draftLogits maps a head output hidden to draft vocab logits through the head's final RMSNorm and the
// SHARED base LM head (untied Output, else tied Embed; the packed quant heads route through the matvec
// seam, matching ComposedSession.headLogits' head selection). The base's logit scaling is applied so a real
// head drafts on the same scale the base decodes; the drafter only proposes, so exact base-arithmetic
// parity here is not required.
func (h *MTPHead) draftLogits(base *ComposedModel, hOut []float32) []float32 {
	y := rmsNormRowsPlain(hOut, h.Norm, 1, h.D, h.Eps)
	var logits []float32
	switch {
	case base.OutputQ != nil:
		logits = matNTQuant(nil, y, base.OutputQ, 1, base.D, base.Vocab)
	case base.Output == nil && base.EmbedQ != nil:
		logits = matNTQuant(nil, y, base.EmbedQ, 1, base.D, base.Vocab)
	default:
		head := base.Output
		if head == nil {
			head = base.Embed
		}
		logits = matNT(y, head, 1, base.D, base.Vocab)
	}
	if base.LogitScale != 0 {
		for i := range logits {
			logits[i] *= base.LogitScale
		}
	}
	if base.LogitsScaling != 0 && base.LogitsScaling != 1 {
		for i := range logits {
			logits[i] /= base.LogitsScaling
		}
	}
	return logits
}

// drafter is the composed pairing's proposal contract, the trained-shape feed: reset begins a
// generation over the prompt's (t_{i+1}, h_i) pairs, observe feeds each committed pair (a token with
// the base hidden that PRODUCED it), and draftBlock proposes up to k tokens continuing after the
// just-produced boundary pair WITHOUT retaining it (speculative context only). Hidden slices passed
// in are valid only for the duration of the call. headDrafter is the shipping implementation; the
// fixture tests drive the identical verify loop with oracle/adversary stubs to prove the accept,
// reject and bonus branches deterministically, independent of any head's draft quality.
type drafter interface {
	reset(promptTokens []int32, promptHiddens []float32) error
	observe(tok int32, producingHidden []float32) error
	draftBlock(tok int32, producingHidden []float32, k int) ([]int32, error)
}

// headDrafter drives an MTPHead as the pairing's drafter: a PERSISTENT head session whose rows are
// the committed (token, producing-hidden) pairs — head row i = (t_{i+1}, h_i), so the head sequence
// always trails the base by exactly one row and every row's implicit position equals the producing
// hidden's true position. Drafting runs on a throwaway clone of that session (Snapshot/Restore from
// the composed session), so speculative rows never leak into the live state; within a block, step
// j+1 chains on the head's OWN step-j output hidden and drafted token.
type headDrafter struct {
	head *MTPHead
	base *ComposedModel
	sess *ComposedSession
}

func (d *headDrafter) reset(promptTokens []int32, promptHiddens []float32) error {
	d.sess = NewSession(d.head.Stack)
	n := len(promptTokens)
	if n < 2 {
		return nil // a 1-token prompt has no (t_{i+1}, h_i) pair yet
	}
	rows, err := d.head.fcInputRows(d.base, promptTokens[1:], promptHiddens[:(n-1)*d.head.D])
	if err != nil {
		return err
	}
	// One batched head forward over the n-1 prompt pairs — the state side-effect is the point.
	_, err = d.sess.forwardEmb(rows, n-1)
	return err
}

func (d *headDrafter) observe(tok int32, producingHidden []float32) error {
	x, err := d.head.fcInput(d.base, tok, producingHidden)
	if err != nil {
		return err
	}
	_, err = d.sess.forwardEmb(x, 1)
	return err
}

func (d *headDrafter) draftBlock(tok int32, producingHidden []float32, k int) ([]int32, error) {
	if k <= 0 {
		return nil, nil
	}
	if d.sess == nil {
		return nil, core.NewError("composed.mtp: draftBlock before reset")
	}
	spec := NewSession(d.head.Stack)
	spec.Restore(d.sess.Snapshot())
	drafts := make([]int32, 0, k)
	curTok := tok
	curHidden := producingHidden
	for range k {
		x, err := d.head.fcInput(d.base, curTok, curHidden)
		if err != nil {
			return drafts, err
		}
		hOut, err := spec.forwardEmb(x, 1)
		if err != nil {
			return drafts, err
		}
		dTok := argmaxF32(d.head.draftLogits(d.base, hOut))
		drafts = append(drafts, dTok)
		curTok, curHidden = dTok, hOut
	}
	return drafts, nil
}

// SpeculativePair is a composed base plus its attached MTP drafter, engaged through the shared speculative
// path. The base owns the verify forward and every committed token; the drafter only proposes how many of
// the base's own greedy tokens can be committed per round.
type SpeculativePair struct {
	Base *ComposedModel
	Head *MTPHead
	// DefaultDraftBlock is the draft depth used when the caller pins none: the checkpoint's declared
	// block_size (the depth the head was trained to chain), falling back to the engine default.
	DefaultDraftBlock int
	drafter           drafter
}

// NewSpeculativePair attaches head to base as a speculative pair, validating the attachment (the head must
// project from the base's hidden width and share a vocab). It is the composed mirror of
// LoadAssistantPairDirs' validateNativeAssistantPair.
func NewSpeculativePair(base *ComposedModel, head *MTPHead) (*SpeculativePair, error) {
	if base == nil {
		return nil, core.NewError("composed.mtp: speculative pair base is nil")
	}
	if head == nil {
		return nil, core.NewError("composed.mtp: speculative pair head is nil")
	}
	if head.D != base.D {
		return nil, core.NewError(core.Sprintf("composed.mtp: head hidden_size = %d, want base hidden_size %d", head.D, base.D))
	}
	if head.Stack == nil || len(head.Stack.Layers) == 0 {
		return nil, core.NewError("composed.mtp: head has no transformer layers")
	}
	if head.Stack.D != base.D {
		return nil, core.NewError("composed.mtp: head stack hidden_size disagrees with the base")
	}
	block := head.BlockSize
	if block <= 0 {
		block = mtpDefaultDraftBlock
	}
	return &SpeculativePair{Base: base, Head: head, DefaultDraftBlock: block, drafter: &headDrafter{head: head, base: base}}, nil
}

// SpeculativeMetrics counts one speculative generation's draft/verify outcome — the composed subset of
// inference.SpeculativeMetrics (the engine adapter maps these onto it). ProposedTokens/AcceptedTokens are
// the true speculative acceptance measurement: how often the head's proposal matched the base's own greedy
// token. AcceptanceRate is Accepted/Proposed.
type SpeculativeMetrics struct {
	ProposedTokens    int
	AcceptedTokens    int
	RejectedTokens    int
	TargetVerifyCalls int // base decode forwards spent committing tokens (verify + bonus)
	DraftCalls        int // drafter block proposals
	AcceptanceRate    float64
}

// GenerateSpeculative greedily decodes up to maxNew tokens from prompt using the MTP head to propose and
// the base to verify. The emitted sequence is byte-identical to ComposedSession.Generate(prompt, maxNew,
// eosID) on the same base — every committed token is the base's own greedy token (an accepted draft equals
// it; a rejected draft is replaced by it; a fully-accepted block adds the base's next greedy token as the
// bonus), so speculation changes only which tokens the drafter got RIGHT, never the output. draftBlock ≤ 0
// falls back to the pair's default (the checkpoint's trained block_size when declared).
func (p *SpeculativePair) GenerateSpeculative(prompt []int32, maxNew, eosID, draftBlock int) ([]int32, SpeculativeMetrics, error) {
	var metrics SpeculativeMetrics
	if p == nil || p.Base == nil || p.drafter == nil {
		return nil, metrics, core.NewError("composed.mtp: GenerateSpeculative requires a validated pair")
	}
	if len(prompt) == 0 || maxNew <= 0 {
		return nil, metrics, core.NewError("composed.mtp: GenerateSpeculative empty prompt or maxNew<=0")
	}
	if draftBlock <= 0 {
		draftBlock = p.DefaultDraftBlock
	}
	if draftBlock <= 0 {
		draftBlock = mtpDefaultDraftBlock
	}
	base := p.Base
	D := base.D
	sess := NewSession(base)
	hidden, err := sess.forward(prompt)
	if err != nil {
		return nil, metrics, err
	}
	if err := p.drafter.reset(prompt, hidden); err != nil {
		return nil, metrics, err
	}
	// boundary is the base hidden at the live decode edge — the hidden that PRODUCES the next greedy
	// token. It is owned here (copied out of the forward's buffer) because the drafter consumes it
	// across a later base forward.
	boundary := append([]float32(nil), hidden[(len(prompt)-1)*D:]...)
	g := argmaxF32(sess.headLogits(boundary)) // the base's canonical next token at this boundary
	out := make([]int32, 0, maxNew)
	// specctl adapts the draft length to recent acceptance (cold-start optimistic at Max), the same policy
	// the metal MTP loop (mtp_draftlen.go) runs — reused, not re-rolled.
	ctrl := specctl.New(specctl.Controller{Min: 1, Max: draftBlock, Window: 8})

	// commit emits id (the base's canonical greedy at the current boundary), feeds the drafter the
	// TRAINED pair — id with the boundary hidden that PRODUCED it — then advances the base one
	// ordinary decode forward to the new boundary. It reports whether generation must stop (maxNew
	// reached or eos); on stop the base is not advanced.
	commit := func(id int32) (bool, error) {
		out = append(out, id)
		if len(out) >= maxNew {
			return true, nil
		}
		if eosID >= 0 && int(id) == eosID {
			return true, nil
		}
		if derr := p.drafter.observe(id, boundary); derr != nil {
			return true, derr
		}
		h1, cerr := sess.forward([]int32{id})
		if cerr != nil {
			return true, cerr
		}
		metrics.TargetVerifyCalls++
		boundary = append(boundary[:0], h1...)
		return false, nil
	}

	for len(out) < maxNew {
		k := ctrl.NextLength()
		// Draft the continuation AFTER g, seeded by the boundary pair (g, the hidden that produced
		// g) — the drafter consumes the pair speculatively; its live state advances only on commit.
		drafts, derr := p.drafter.draftBlock(g, boundary, k)
		if derr != nil {
			return out, metrics, derr
		}
		metrics.DraftCalls++
		metrics.ProposedTokens += len(drafts)

		// Commit g: the base's canonical token at this boundary (round 1: the base's first token;
		// later rounds: the replacement after a reject or the bonus after full acceptance).
		stop, cerr := commit(g)
		if stop || cerr != nil {
			// Drafts of a finished generation were never verified; they bought nothing — rejected.
			metrics.RejectedTokens += len(drafts)
			finishMetrics(&metrics)
			return out, metrics, cerr
		}
		g = argmaxF32(sess.headLogits(boundary))

		accepted := 0
		for _, d := range drafts {
			if d != g {
				break // reject from here — g replaces the draft on the next round's commit
			}
			accepted++
			stop, cerr = commit(d) // d == the base greedy, so committing it is plain decode
			if stop || cerr != nil {
				metrics.AcceptedTokens += accepted
				metrics.RejectedTokens += len(drafts) - accepted
				finishMetrics(&metrics)
				return out, metrics, cerr
			}
			g = argmaxF32(sess.headLogits(boundary)) // next canonical greedy, for the next comparison
		}
		metrics.AcceptedTokens += accepted
		metrics.RejectedTokens += len(drafts) - accepted
		ctrl.Record(len(drafts), accepted)
	}
	finishMetrics(&metrics)
	return out, metrics, nil
}

// mtpDefaultDraftBlock is the composed pairing's fallback draft block when neither the caller nor the
// checkpoint (block_size) pins one — the same engine default as serving.MTPDefaultDraftBlock (kept as a
// local constant so the model package carries no serving import).
const mtpDefaultDraftBlock = 5

// LoadSpeculativePairDirs loads a composed base checkpoint + its Qwen MTP drafter checkpoint as one
// validated speculative pair — the dir-level entry the engine's speculative seam binds for the composed
// family (the mirror of engine/metal's LoadAssistantPairDirs for gemma4 ArchSession targets). The draft
// dir must declare a registered composed MTP model_type (assistant.go's reactive spec); its parsed
// mtp.AssistantConfig validates the attachment BEFORE any base tensor is realised, so a wrong pairing
// fails in milliseconds, not after a 27B load. The base builds ZERO-COPY (packed weights view its own
// checkpoint mapping, retained for the model's lifetime — release via Close); the head is small and
// loads copied, so it threads no second mapping.
func LoadSpeculativePairDirs(baseDir, draftDir string) (*SpeculativePair, error) {
	draftCfgStr, err := coreio.Local.Read(core.PathJoin(draftDir, "config.json"))
	if err != nil {
		return nil, core.E("composed.LoadSpeculativePairDirs", "read draft config.json", err)
	}
	draftCfg := []byte(draftCfgStr)
	acfg, err := mtp.ParseAssistantConfig(draftCfg)
	if err != nil {
		return nil, core.E("composed.LoadSpeculativePairDirs", "parse draft assistant config", err)
	}
	if !isComposedMTPModelType(acfg.ModelType) {
		return nil, core.NewError("composed.LoadSpeculativePairDirs: draft model_type " + acfg.ModelType + " is not a composed MTP head")
	}
	baseCfgStr, err := coreio.Local.Read(core.PathJoin(baseDir, "config.json"))
	if err != nil {
		return nil, core.E("composed.LoadSpeculativePairDirs", "read base config.json", err)
	}
	baseCfg := []byte(baseCfgStr)
	baseDM, err := safetensors.LoadDirMmap(baseDir)
	if err != nil {
		return nil, core.E("composed.LoadSpeculativePairDirs", "map base checkpoint", err)
	}
	base, err := loadComposed(baseDM.Tensors, baseCfg, nil, true)
	if err != nil {
		_ = baseDM.Close()
		return nil, core.E("composed.LoadSpeculativePairDirs", "load base", err)
	}
	if !base.retain(baseDM) {
		_ = baseDM.Close() // nothing aliases the mapping (dense/copied build)
	}
	if acfg.BackboneHidden != base.D {
		_ = base.Close()
		return nil, core.NewError(core.Sprintf("composed.LoadSpeculativePairDirs: draft backbone hidden %d does not match base hidden %d", acfg.BackboneHidden, base.D))
	}
	draftDM, err := safetensors.LoadDirMmap(draftDir)
	if err != nil {
		_ = base.Close()
		return nil, core.E("composed.LoadSpeculativePairDirs", "map draft checkpoint", err)
	}
	head, err := LoadMTPHead(draftDM.Tensors, draftCfg, base)
	_ = draftDM.Close() // the head copies its tensors; the mapping is not aliased
	if err != nil {
		_ = base.Close()
		return nil, core.E("composed.LoadSpeculativePairDirs", "load MTP head", err)
	}
	pair, err := NewSpeculativePair(base, head)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	return pair, nil
}

// Close releases the pair's base checkpoint mapping (the head owns no mapping of its own). After Close
// the pair must not be used — the base's packed weights alias unmapped memory.
func (p *SpeculativePair) Close() error {
	if p == nil || p.Base == nil {
		return nil
	}
	return p.Base.Close()
}

func finishMetrics(m *SpeculativeMetrics) {
	if m.ProposedTokens > 0 {
		m.AcceptanceRate = float64(m.AcceptedTokens) / float64(m.ProposedTokens)
	}
}

// LoadMTPHead builds a Qwen 3.5/3.6 MTP drafter head from an already-mapped checkpoint tensor map + its
// config.json, validated against the base it drafts for (same hidden width). It is the composed twin of
// LoadComposed, and reuses the base loader's OWN tensor helpers (buildAttn / buildFFN / the packed-or-widened
// proj closure), so a quantised head keeps its projections PACKED exactly as the base does and a dense head
// widens — no second copy of that machinery. The head SHARES the base's token embedding + LM head (the real
// checkpoint carries neither embed_tokens nor lm_head — mtp_use_dedicated_embeddings / tie_word_embeddings
// both false); the pair supplies them at draft time. The head's transformer layers are full attention: the
// Qwen MTP module is a standard attention block whatever the base's hybrid schedule, and the real
// checkpoint's tensors confirm it (layers.N.self_attn.*).
//
// The real tensor layout (mlx-community/Qwen3.6-27B-MTP-4bit): pre_fc_norm_embedding / pre_fc_norm_hidden
// (the two pre-fc RMSNorms), fc (the [D,2D] input combiner), layers.N.{input_layernorm, self_attn.*,
// post_attention_layernorm, mlp.*}, norm (the head's final RMSNorm). The top-level block_size is the
// trained draft depth (3 on the real checkpoint) — loaded into MTPHead.BlockSize as the pair's default.
func LoadMTPHead(tensors map[string]safetensors.Tensor, configJSON []byte, base *ComposedModel) (*MTPHead, error) {
	if base == nil {
		return nil, core.NewError("composed.mtp: LoadMTPHead requires the base model the head drafts for")
	}
	var raw loaderConfig
	if r := core.JSONUnmarshal(configJSON, &raw); !r.OK {
		return nil, core.NewError("composed.mtp: LoadMTPHead config.json parse failed")
	}
	cfg := raw.effective()
	D := cfg.HiddenSize
	if D <= 0 {
		return nil, core.NewError("composed.mtp: LoadMTPHead config has no hidden_size")
	}
	if D != base.D {
		return nil, core.NewError(core.Sprintf("composed.mtp: head hidden_size = %d, want base hidden_size %d", D, base.D))
	}
	nLayers := cfg.MTPNumHiddenLayers
	if nLayers <= 0 {
		return nil, core.NewError("composed.mtp: LoadMTPHead config has no mtp_num_hidden_layers")
	}
	eps := cfg.RMSNormEps
	if eps == 0 {
		eps = 1e-6
	}
	var top struct {
		BlockSize int `json:"block_size"`
	}
	// Best-effort: an absent block_size leaves BlockSize 0 and the pair falls back to the engine default.
	_ = core.JSONUnmarshal(configJSON, &top)
	quant := quantBlock(configJSON)

	// The base loader's helper closures over the head's tensor map. zeroCopy=false: the head is small and
	// loads once, and a copied head aliases no checkpoint mapping — so the pair has no mmap lifetime to
	// thread (the base owns its own).
	get := func(name string) (safetensors.Tensor, bool) { t, ok := tensors[name]; return t, ok }
	f32 := func(name string) ([]float32, error) {
		t, ok := get(name)
		if !ok {
			return nil, core.NewError("composed.mtp: LoadMTPHead missing " + name)
		}
		return tensorAsF32(tensors, name, t, quant)
	}
	f32opt := func(name string) []float32 {
		if t, ok := get(name); ok {
			if v, e := tensorAsF32(tensors, name, t, quant); e == nil {
				return v
			}
		}
		return nil
	}
	proj := func(name string) ([]float32, *model.QuantWeight, error) {
		t, ok := get(name)
		if !ok {
			return nil, nil, core.NewError("composed.mtp: LoadMTPHead missing " + name)
		}
		qw, _, err := tensorAsQuant(tensors, name, t, quant, false)
		if err != nil {
			return nil, nil, err
		}
		if qw != nil {
			return nil, qw, nil
		}
		fv, err := tensorF32(t)
		return fv, nil, err
	}

	fcF, fcQ, err := proj("fc.weight")
	if err != nil {
		return nil, core.E("composed.mtp: LoadMTPHead", "fc", err)
	}
	enorm, err := f32("pre_fc_norm_embedding.weight")
	if err != nil {
		return nil, err
	}
	hnorm, err := f32("pre_fc_norm_hidden.weight")
	if err != nil {
		return nil, err
	}
	finalNorm, err := f32("norm.weight")
	if err != nil {
		return nil, err
	}

	layers := make([]Layer, nLayers)
	for i := range layers {
		lp := core.Sprintf("layers.%d.", i)
		inNorm, ferr := f32(lp + "input_layernorm.weight")
		if ferr != nil {
			return nil, ferr
		}
		postNorm, ferr := f32(lp + "post_attention_layernorm.weight")
		if ferr != nil {
			return nil, ferr
		}
		ffn, ferr := buildFFN(get, proj, f32, lp+"mlp.", cfg, nil, D)
		if ferr != nil {
			return nil, core.E("composed.mtp: LoadMTPHead", core.Sprintf("layer %d ffn", i), ferr)
		}
		mixer, ferr := buildAttn(proj, f32opt, lp+"self_attn.", cfg, nil, i, D, "full_attention")
		if ferr != nil {
			return nil, core.E("composed.mtp: LoadMTPHead", core.Sprintf("layer %d attn", i), ferr)
		}
		layers[i] = Layer{InputNorm: inNorm, Mixer: mixer, PostAttnNorm: postNorm, MLP: ffn}
	}

	return &MTPHead{
		Stack:     &ComposedModel{Layers: layers, D: D, Vocab: base.Vocab, Eps: eps, Quantised: quant != nil},
		FC:        fcF,
		FCQ:       fcQ,
		Enorm:     enorm,
		Hnorm:     hnorm,
		Norm:      finalNorm,
		D:         D,
		Eps:       eps,
		BlockSize: top.BlockSize,
	}, nil
}
