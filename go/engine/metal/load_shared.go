// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// load_shared.go consumes the backend-agnostic model.LoadedModel (pkg/model — where the
// per-weight quant decision is made ONCE, quant-agnostically, by reading the tensor shapes) and
// maps it onto the native decode structs. The hand-coded per-weight fetchQuant/fetchNorm walk that
// used to live in the native assembler is gone: this is a mechanical translation, not a second
// loader. A weight that one quant leaves bf16 while another quantises (e4b's per_layer_model_
// projection) is handled by the shared loader's .scales decision, so native never re-bugs it.

// loadedToQuant maps a LoadedModel onto the native 4-bit QuantModel. The model-wide gs/bits are
// the native structs' single quant geometry (gemma4 quant packs are uniform across the projections;
// the per-weight geometry the shared loader read from shapes agrees with it). MoE (26B-A4B) is not
// yet routed here — it errors clearly rather than mis-assembling.
func loadedToQuant(m *model.LoadedModel, gs, bits int) (*QuantModel, error) {
	if m == nil || m.Embed == nil {
		return nil, core.NewError("native.loadedToQuant: nil model or embedding")
	}
	g := &QuantModel{GroupSize: gs, Bits: bits, FinalNorm: m.FinalNorm}
	g.Embed, g.EmbedScales, g.EmbedBiases = m.Embed.Weight, m.Embed.Scales, m.Embed.Biases
	if m.LMHead != nil {
		g.LMHead, g.LMHeadScales, g.LMHeadBiases = m.LMHead.Weight, m.LMHead.Scales, m.LMHead.Biases
	} else { // tied: the head reuses the embedding triple
		g.LMHead, g.LMHeadScales, g.LMHeadBiases, g.Tied = m.Embed.Weight, m.Embed.Scales, m.Embed.Biases, true
	}
	if m.EmbedPerLayer != nil { // PLE tower (E2B/E4B)
		g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases = m.EmbedPerLayer.Weight, m.EmbedPerLayer.Scales, m.EmbedPerLayer.Biases
		g.PerLayerProjNormW = m.PerLayerProjNorm
	}
	if p := m.PerLayerModelProj; p != nil {
		// PerLayerModelProjW holds the packed weight (qat: e4b) or the bf16 weight (regular: e2b);
		// the scales (set only when quantised) tell PerLayerInputs which matvec to run.
		g.PerLayerModelProjW = p.Weight
		if p.Quantised() {
			g.PerLayerModelProjScales, g.PerLayerModelProjBiases = p.Scales, p.Biases
			g.PerLayerModelProjGS, g.PerLayerModelProjBits = p.GroupSize, p.Bits
		}
	}
	g.Layers = make([]QuantizedLayerWeights, len(m.Layers))
	for i := range m.Layers {
		L := &m.Layers[i]
		ql := &g.Layers[i]
		ql.AttnNormW, ql.PostAttnNormW = L.AttnNorm, L.PostAttnNorm
		ql.QNormW, ql.KNormW, ql.LayerScalarW = L.QNorm, L.KNorm, L.LayerScalar
		ql.GroupSize, ql.Bits = gs, bits
		ql.Q, ql.K, ql.V, ql.O = qw(L.Q), qw(L.K), qw(L.V), qw(L.O)
		ql.PerLayerGate, ql.PerLayerProjection = qw(L.PerLayerGate), qw(L.PerLayerProjection)
		ql.PostPerLayerInputNormW = L.PostPerLayerInputNorm
		if L.MoE != nil {
			ql.MoE = moeToQuant(L.MoE, m.Arch.Experts, m.Arch.TopK, m.Arch.ExpertFF, m.Arch.Hidden)
		} else {
			ql.MLPNormW, ql.PostFFNormW = L.MLPNorm, L.PostFFNorm
			ql.Gate, ql.Up, ql.Down = qw(L.Gate), qw(L.Up), qw(L.Down)
			if L.Gate != nil { // per-layer MatFormer FFN width, read from the gate's output rows
				ql.DFF = L.Gate.OutDim
			}
		}
	}
	return g, nil
}

// moeToQuant maps the shared loader's MoE block onto the native MoEQuantLayerWeights. The
// per-component quant geometry (experts vs local MLP vs router) is read from each weight's own
// shape — gemma4 26B-A4B keeps the experts 4-bit while the local MLP + router are 8-bit — and the
// router norm is pre-folded by RootSize (matching metal's cached Router.ScaleScaled).
func moeToQuant(e *model.LoadedMoE, experts, topK, expertFF, dModel int) *MoEQuantLayerWeights {
	q := &MoEQuantLayerWeights{
		NumExperts: experts, TopK: topK, ExpertDFF: expertFF,
		PreFFNormW: e.PreFFNorm, PreFFNorm2W: e.PreFFNorm2,
		PostFFNorm1W: e.PostFFNorm1, PostFFNorm2W: e.PostFFNorm2, PostFFNormW: e.PostFFNorm,
		LocalGate:         qw(e.LocalGate),
		LocalUp:           qw(e.LocalUp),
		LocalDown:         qw(e.LocalDown),
		RouterNormWScaled: foldRootSize(e.RouterScale, dModel),
		Router:            qw(e.Router),
		PerExpertScale:    e.PerExpertScale,
		ExpGate:           qw(e.ExpGate),
		ExpUp:             qw(e.ExpUp),
		ExpGateUp:         qw(e.ExpGateUp),
		ExpDown:           qw(e.ExpDown),
	}
	if e.ExpGate != nil {
		q.ExpertGroupSize, q.ExpertBits = e.ExpGate.GroupSize, e.ExpGate.Bits
	} else if e.ExpGateUp != nil {
		q.ExpertGroupSize, q.ExpertBits = e.ExpGateUp.GroupSize, e.ExpGateUp.Bits
	}
	if e.LocalGate != nil {
		q.LocalGroupSize, q.LocalBits = e.LocalGate.GroupSize, e.LocalGate.Bits
	}
	if e.Router != nil {
		q.RouterGroupSize, q.RouterBits = e.Router.GroupSize, e.Router.Bits
	}
	return q
}

// qw maps a shared model.Linear to the native quant-weight triple (packed codes + bf16 scales +
// biases). A nil Linear (an absent optional weight — a K==V layer's v_proj, a KV-shared layer's
// k_proj) yields the zero QuantWeight, which the projector treats as "skip".
func qw(lin *model.Linear) QuantWeight {
	if lin == nil {
		return QuantWeight{}
	}
	// GroupSize/Bits are the weight's OWN geometry (read from shapes by the shared loader) — this is
	// what carries e4b-qat's per-layer mixed precision (the 8-bit MLP beside the 4-bit attention)
	// through to the qmv kernel, instead of a single model-wide width.
	return QuantWeight{Packed: lin.Weight, Scales: lin.Scales, Biases: lin.Biases, GroupSize: lin.GroupSize, Bits: lin.Bits}
}

// loadedToBF16 maps a dense LoadedModel onto the native bf16 BF16Model — the bf16 sibling of
// loadedToQuant. Routing the bf16 path through the SAME shared loader means it inherits the per-layer
// FFN width (MatFormer), KV-share and the PLE tower from the SHAPES, instead of the hand-coded
// assembler's fixed-dim "dense only" subset (which choked on E2B's per-layer FFN). bw takes a dense
// Linear's bf16 weight bytes (nil for an absent optional weight).
func loadedToBF16(m *model.LoadedModel) *BF16Model {
	bw := func(lin *model.Linear) []byte {
		if lin == nil {
			return nil
		}
		return lin.Weight
	}
	g := &BF16Model{FinalNorm: m.FinalNorm, Embed: bw(m.Embed)}
	if m.LMHead != nil {
		g.LMHead = bw(m.LMHead)
	} else {
		g.LMHead, g.Tied = bw(m.Embed), true
	}
	if m.EmbedPerLayer != nil { // PLE tower (E2B/E4B)
		g.EmbedPerLayer = m.EmbedPerLayer.Weight
		g.PerLayerProjNormW = m.PerLayerProjNorm
	}
	if m.PerLayerModelProj != nil {
		g.PerLayerModelProjW = m.PerLayerModelProj.Weight
	}
	g.Layers = make([]DecodeLayerWeights, len(m.Layers))
	for i := range m.Layers {
		L, l := &m.Layers[i], &g.Layers[i]
		l.AttnNormW, l.PostAttnNormW = L.AttnNorm, L.PostAttnNorm
		l.QNormW, l.KNormW, l.LayerScalarW = L.QNorm, L.KNorm, L.LayerScalar
		l.MLPNormW, l.PostFFNormW = L.MLPNorm, L.PostFFNorm
		l.WQ, l.WK, l.WV, l.WO = bw(L.Q), bw(L.K), bw(L.V), bw(L.O)
		l.WGate, l.WUp, l.WDown = bw(L.Gate), bw(L.Up), bw(L.Down)
		if L.Gate != nil { // per-layer MatFormer FFN width, read from the gate's output rows
			l.DFF = L.Gate.OutDim
		}
		l.PerLayerGate, l.PerLayerProjection = bw(L.PerLayerGate), bw(L.PerLayerProjection)
		l.PostPerLayerInputNormW = L.PostPerLayerInputNorm
	}
	return g
}
