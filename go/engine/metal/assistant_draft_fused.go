// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// assistant_draft_fused.go is the single-command-buffer drafter step. The
// unfused path runs every drafter op (norms, projections, rope, SDPA, MLP)
// through its standalone wrapper, each paying a full commit+wait GPU
// round-trip — ~60 synchronisations per draft step, which made one ~10ms on
// hardware where the actual GPU work is well under a millisecond, and left
// the MTP pair slower than plain decode. Here the whole transformer (input
// projection, four layers, final norm, output projection) encodes into ONE
// command buffer with ONE wait; only the centroid logits head (which needs a
// CPU top-k between its two matmuls) stays on the wrapper path. Kernels,
// order and operands are identical to the unfused step, so the outputs are
// byte-identical — gated by TestAssistantFusedDraftParity.

// assistantFusedDraftEnabled routes draft steps through the fused encoder.
// SetAssistantFusedDraft(false) forces the legacy per-op path (A/B traces,
// parity tests).
var assistantFusedDraftEnabled = true

// SetAssistantFusedDraft toggles the fused single-command-buffer drafter step.
func SetAssistantFusedDraft(enabled bool) { assistantFusedDraftEnabled = enabled }

// fusedDraftLayer is one drafter layer's resolved weights + geometry: every
// tensor pre-wrapped as a resident MTLBuffer, the rope spectrum pre-built, so
// the hot step only encodes.
type fusedDraftLayer struct {
	layerType string
	nHeads    int
	headDim   int
	rotaryDim int
	ropeBase  float32
	ropeFreqs metal.MTLBuffer // proportional periods (full_attention); nil = base-derived rope

	inputNormW, postAttnNormW, preFFNormW, postFFNormW metal.MTLBuffer
	qProjW, qNormW, oProjW                             metal.MTLBuffer
	gateW, upW, downW                                  metal.MTLBuffer
	layerScalar                                        metal.MTLBuffer // [1] bf16; nil when absent
}

// fusedDraftKV is one layer type's target KV slab resident on the GPU for the
// current draft block, refreshed by loadKV each round.
type fusedDraftKV struct {
	k, v     metal.MTLBuffer
	capBytes int
	kvHeads  int
	headDim  int
	length   int
	qPos     int32
}

// assistantFusedDraft is the fused drafter: resolved layer weights plus the
// step's GPU scratch. One instance per AssistantPair, built lazily on the
// first draft block and reused for the pair's lifetime.
type assistantFusedDraft struct {
	hidden, backbone, dFF int
	eps                   float32
	attnScale             float32
	ropeScale             float32

	layers                 []fusedDraftLayer
	preProjW, postProjW    metal.MTLBuffer
	finalNormW             metal.MTLBuffer
	kv                     map[string]*fusedDraftKV
	inConcat               metal.MTLBuffer // [2*backbone] bf16 — concat(emb, hidden)
	h, resid, normed, ffIn metal.MTLBuffer // [hidden]
	q, qr, attn            metal.MTLBuffer // [nHeads*maxHeadDim]
	gate, up, gated        metal.MTLBuffer // [dFF]
	ff                     metal.MTLBuffer // [hidden]
	outNormed              metal.MTLBuffer // [hidden]
	outHidden              metal.MTLBuffer // [backbone]
}

// fusedTensorBuf resolves a named bf16 tensor to a resident buffer, failing
// on quantised or missing tensors (the fused path serves bf16 drafters only —
// exactly the set the unfused loader accepts today).
func fusedTensorBuf(m *AssistantModel, name string, wantElems int) (metal.MTLBuffer, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return nil, core.NewError("native.assistant fused draft missing tensor " + name)
	}
	if t.Dtype != "BF16" {
		return nil, core.NewError("native.assistant fused draft tensor " + name + " dtype = " + t.Dtype + ", want BF16")
	}
	if len(t.Data) != wantElems*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant fused draft tensor %s bytes = %d, want %d", name, len(t.Data), wantElems*bf16Size))
	}
	buf := residentBytes(t.Data)
	if buf == nil {
		return nil, core.NewError("native.assistant fused draft tensor " + name + " did not wrap as a Metal buffer")
	}
	return buf, nil
}

// newAssistantFusedDraft resolves the drafter's full geometry + weights for
// the fused step. Any unsupported shape (quantised tensors, missing gelu
// kernel, incomplete dims) returns an error and the caller stays on the
// legacy per-op path.
func newAssistantFusedDraft(m *AssistantModel) (*assistantFusedDraft, error) {
	if m == nil {
		return nil, core.NewError("native.assistant fused draft model is nil")
	}
	if !gpuHasGeluKernel() {
		return nil, core.NewError("native.assistant fused draft needs the fused gelu kernel")
	}
	hidden, dFF, backbone := m.Arch.Hidden, m.Arch.FF, m.BackboneHiddenSize
	nHeads := m.Arch.Heads
	if hidden <= 0 || dFF <= 0 || backbone <= 0 || nHeads <= 0 || len(m.Arch.Layer) == 0 {
		return nil, core.NewError("native.assistant fused draft has incomplete dimensions")
	}
	f := &assistantFusedDraft{
		hidden: hidden, backbone: backbone, dFF: dFF,
		eps:       m.Arch.Eps,
		attnScale: nativeAssistantAttentionScale(m),
		ropeScale: m.Arch.RopeScale,
		kv:        map[string]*fusedDraftKV{},
	}
	if f.ropeScale == 0 {
		f.ropeScale = 1
	}
	var err error
	if f.preProjW, err = fusedTensorBuf(m, "pre_projection.weight", hidden*2*backbone); err != nil {
		return nil, err
	}
	if f.postProjW, err = fusedTensorBuf(m, "post_projection.weight", backbone*hidden); err != nil {
		return nil, err
	}
	if f.finalNormW, err = fusedTensorBuf(m, "model.norm.weight", hidden); err != nil {
		return nil, err
	}
	maxHeadDim := 0
	for i := range m.Arch.Layer {
		spec := m.Arch.Layer[i]
		headDim := spec.HeadDim
		if headDim <= 0 {
			headDim = m.Arch.HeadDim
		}
		if headDim > maxHeadDim {
			maxHeadDim = headDim
		}
		l := fusedDraftLayer{
			layerType: m.Config.LayerType(i),
			nHeads:    nHeads,
			headDim:   headDim,
			rotaryDim: nativeAssistantLayerRotaryDim(m, spec, headDim),
			ropeBase:  nativeAssistantLayerRopeBase(m, spec),
		}
		if len(m.Arch.RopeFreqs) > 0 {
			return nil, core.NewError("native.assistant fused draft does not carry the YaRN freqs path")
		}
		if l.rotaryDim < headDim {
			// gemma4 proportional partial rotary — the same full-head period
			// spectrum nativeAssistantRoPEInto ropes with (see that function).
			l.ropeFreqs = cachedRawRopePeriodsBuffer(globalRopePeriodsFromFolded(headDim, l.rotaryDim, l.ropeBase))
			l.rotaryDim = headDim
		}
		prefix := core.Sprintf("model.layers.%d.", i)
		if l.inputNormW, err = fusedTensorBuf(m, prefix+"input_layernorm.weight", hidden); err != nil {
			return nil, err
		}
		if l.postAttnNormW, err = fusedTensorBuf(m, prefix+"post_attention_layernorm.weight", hidden); err != nil {
			return nil, err
		}
		if l.preFFNormW, err = fusedTensorBuf(m, prefix+"pre_feedforward_layernorm.weight", hidden); err != nil {
			return nil, err
		}
		if l.postFFNormW, err = fusedTensorBuf(m, prefix+"post_feedforward_layernorm.weight", hidden); err != nil {
			return nil, err
		}
		if l.qProjW, err = fusedTensorBuf(m, prefix+"self_attn.q_proj.weight", nHeads*headDim*hidden); err != nil {
			return nil, err
		}
		if l.qNormW, err = fusedTensorBuf(m, prefix+"self_attn.q_norm.weight", headDim); err != nil {
			return nil, err
		}
		if l.oProjW, err = fusedTensorBuf(m, prefix+"self_attn.o_proj.weight", hidden*nHeads*headDim); err != nil {
			return nil, err
		}
		if l.gateW, err = fusedTensorBuf(m, prefix+"mlp.gate_proj.weight", dFF*hidden); err != nil {
			return nil, err
		}
		if l.upW, err = fusedTensorBuf(m, prefix+"mlp.up_proj.weight", dFF*hidden); err != nil {
			return nil, err
		}
		if l.downW, err = fusedTensorBuf(m, prefix+"mlp.down_proj.weight", hidden*dFF); err != nil {
			return nil, err
		}
		scalar, serr := nativeAssistantLayerScalar(m, core.Sprintf("model.layers.%d", i), hidden)
		if serr != nil {
			return nil, serr
		}
		if len(scalar) == bf16Size {
			l.layerScalar = residentBytes(scalar)
		} else if len(scalar) != 0 {
			return nil, core.NewError("native.assistant fused draft layer_scalar is not a single bf16 scalar")
		}
		f.layers = append(f.layers, l)
	}
	f.inConcat = scratchBF16(2 * backbone)
	f.h = scratchBF16(hidden)
	f.resid = scratchBF16(hidden)
	f.normed = scratchBF16(hidden)
	f.ffIn = scratchBF16(hidden)
	f.q = scratchBF16(nHeads * maxHeadDim)
	f.qr = scratchBF16(nHeads * maxHeadDim)
	f.attn = scratchBF16(nHeads * maxHeadDim)
	f.gate = scratchBF16(dFF)
	f.up = scratchBF16(dFF)
	f.gated = scratchBF16(dFF)
	f.ff = scratchBF16(hidden)
	f.outNormed = scratchBF16(hidden)
	f.outHidden = scratchBF16(backbone)
	for _, buf := range []metal.MTLBuffer{f.inConcat, f.h, f.resid, f.normed, f.ffIn, f.q, f.qr, f.attn, f.gate, f.up, f.gated, f.ff, f.outNormed, f.outHidden} {
		if buf == nil {
			return nil, core.NewError("native.assistant fused draft scratch allocation failed")
		}
	}
	return f, nil
}

// loadKV uploads the round's target KV slabs into the fused scratch — once
// per draft BLOCK (the slabs are fixed across a block's steps). Dedicated
// buffers with an explicit copy: the slabs are pooled Go scratch reused with
// new content each round, so a pointer-keyed resident cache would go stale.
func (f *assistantFusedDraft) loadKV(targetKVs AssistantTargetKVByType) error {
	for _, e := range targetKVs.entries {
		slot := f.kv[e.LayerType]
		if slot == nil {
			slot = &fusedDraftKV{}
			f.kv[e.LayerType] = slot
		}
		need := len(e.KV.Key)
		if len(e.KV.Value) > need {
			need = len(e.KV.Value)
		}
		if slot.capBytes < need {
			slot.k = device.NewBufferWithLengthOptions(uint(need), metal.MTLResourceStorageModeShared)
			slot.v = device.NewBufferWithLengthOptions(uint(need), metal.MTLResourceStorageModeShared)
			if slot.k == nil || slot.v == nil {
				return core.NewError("native.assistant fused draft KV buffer allocation failed")
			}
			slot.capBytes = need
		}
		copy(unsafe.Slice((*byte)(slot.k.Contents()), slot.capBytes), e.KV.Key)
		copy(unsafe.Slice((*byte)(slot.v.Contents()), slot.capBytes), e.KV.Value)
		slot.kvHeads = e.KV.KVHeads
		slot.headDim = e.KV.HeadDim
		slot.length = e.KV.Length
		slot.qPos = int32(max(e.KV.Offset+e.KV.Length-1, 0))
	}
	return nil
}

// step runs one fused drafter step: concat(emb, hidden) → pre_projection →
// four layers → final norm → post_projection, all in one command buffer with
// one wait. normedOut receives the final-normed drafter hidden (the logits
// head's input); hiddenOut the backbone-space recursion hidden.
func (f *assistantFusedDraft) step(tokenEmbedding, previousHidden, normedOut, hiddenOut []byte) ([]byte, []byte, error) {
	bb := f.backbone * bf16Size
	if len(tokenEmbedding) != bb || len(previousHidden) != bb {
		return nil, nil, core.NewError("native.assistant fused draft input bytes mismatch")
	}
	in := unsafe.Slice((*byte)(f.inConcat.Contents()), 2*bb)
	copy(in, tokenEmbedding)
	copy(in[bb:], previousHidden)

	var encErr error
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emit := func(err error) {
			if err != nil && encErr == nil {
				encErr = err
			}
		}
		emit(encGemvBF16(enc, f.preProjW, f.inConcat, f.h, f.hidden, 2*f.backbone))
		for i := range f.layers {
			l := &f.layers[i]
			kv := f.kv[l.layerType]
			if kv == nil || kv.length <= 0 {
				emit(core.NewError("native.assistant fused draft missing KV for " + l.layerType))
				break
			}
			emit(encRMSNormBF16(enc, f.h, l.inputNormW, f.normed, 0, f.hidden, f.eps))
			emit(encGemvBF16(enc, l.qProjW, f.normed, f.q, l.nHeads*l.headDim, f.hidden))
			emit(encRMSNormRowsBF16(enc, f.q, l.qNormW, f.q, 0, 0, 0, l.nHeads, l.headDim, f.eps))
			emit(encRopeDecodeAt(enc, f.q, f.qr, 0, 0, scalarI32(kv.qPos), 0, l.ropeFreqs, l.nHeads, l.headDim, l.rotaryDim, l.ropeBase, f.ropeScale))
			emit(encSDPA(enc, f.qr, kv.k, kv.v, f.attn, l.nHeads, kv.kvHeads, l.headDim, kv.length, f.attnScale))
			emit(encGemvBF16(enc, l.oProjW, f.attn, f.ff, f.hidden, l.nHeads*l.headDim))
			emit(encRMSNormBF16(enc, f.ff, l.postAttnNormW, f.normed, 0, f.hidden, f.eps))
			emit(encAddBF16(enc, f.h, f.normed, f.resid, f.hidden))
			emit(encRMSNormBF16(enc, f.resid, l.preFFNormW, f.ffIn, 0, f.hidden, f.eps))
			emit(encGemvBF16(enc, l.gateW, f.ffIn, f.gate, f.dFF, f.hidden))
			emit(encGemvBF16(enc, l.upW, f.ffIn, f.up, f.dFF, f.hidden))
			emit(encGeluGateMulFused(enc, f.gate, f.up, f.gated, f.dFF))
			emit(encGemvBF16(enc, l.downW, f.gated, f.ff, f.hidden, f.dFF))
			emit(encRMSNormBF16(enc, f.ff, l.postFFNormW, f.normed, 0, f.hidden, f.eps))
			emit(encAddBF16(enc, f.resid, f.normed, f.h, f.hidden))
			if l.layerScalar != nil {
				emit(encMulScalarBF16(enc, f.h, l.layerScalar, f.h, 0, f.hidden))
			}
		}
		emit(encRMSNormBF16(enc, f.h, f.finalNormW, f.outNormed, 0, f.hidden, f.eps))
		emit(encGemvBF16(enc, f.postProjW, f.outNormed, f.outHidden, f.backbone, f.hidden))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	if encErr != nil {
		return nil, nil, encErr
	}
	normedOut = normedOut[:f.hidden*bf16Size]
	hiddenOut = hiddenOut[:f.backbone*bf16Size]
	copy(normedOut, unsafe.Slice((*byte)(f.outNormed.Contents()), len(normedOut)))
	copy(hiddenOut, unsafe.Slice((*byte)(f.outHidden.Contents()), len(hiddenOut)))
	return normedOut, hiddenOut, nil
}

// fusedDraft lazily builds (once) and returns the pair's fused drafter, or
// nil when the geometry is unsupported / the switch is off — callers fall
// back to the legacy per-op step.
func (pair *AssistantPair) fusedDraft() *assistantFusedDraft {
	if !assistantFusedDraftEnabled || pair == nil || pair.Assistant == nil {
		return nil
	}
	if pair.fusedInit {
		return pair.fused
	}
	pair.fusedInit = true
	f, err := newAssistantFusedDraft(pair.Assistant)
	if err != nil {
		pair.fused = nil
		return nil
	}
	pair.fused = f
	return f
}
