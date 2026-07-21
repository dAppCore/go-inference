// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// MoELayerWeights holds the bf16 weights AND the MoE-specific shape of one MoE feed-forward block.
// gemma4's shape populates every field: five independent RMSNorm weights sandwiching a dual-branch
// FFN (an always-on local dense MLP alongside the routed experts), plus a router with its OWN
// internal norm. Norm weights are dModel bf16. RouterNormWScaled is the router's own norm weight
// ALREADY scaled by RootSize (folded at load like metal's cached ScaleScaled — see MoERouter).
// PerExpertScale is optional (nil to skip). The local MLP runs at the model-wide dFF; the experts
// run at ExpertDFF (gemma4 gives them a distinct MoEIntermediateSize).
//
// A llama-family zoo layer (mixtral/dbrx/olmoe — #59) declares a SIMPLER shape: ONE pre-FFN norm
// (post_attention_layernorm), no local dense MLP, no post-combine sandwich norms, and no router-own
// norm (HF's router is a plain nn.Linear on the SAME normed hidden state the experts consume, not a
// second independent RMSNorm). It populates PreFFNormW only, leaving PreFFNorm2W/PostFFNorm1W/
// PostFFNorm2W/PostFFNormW/RouterNormWScaled/WGate/WUp/WDown nil. Every executor consuming this
// struct (moeBlockBF16AfterRouterWithBufferPooled and the router-norm callers below) treats each of
// those absent fields as identity/skip — nil local MLP ⇒ no local branch; nil PreFFNorm2W ⇒ the
// expert branch normalises on PreFFNormW instead (the one norm the zoo arch declares); nil
// PostFFNorm1W/PostFFNorm2W/PostFFNormW ⇒ that norm step is skipped, so a local-less layer with none
// of the three sandwich norms reduces to out = h + expertAcc, exactly HF's llama-family MoE residual
// add; nil RouterNormWScaled ⇒ routerNormWeight() falls back to PreFFNormW so the router scores the
// SAME normed input the experts consume, with no second norm. gemma4's checkpoint always populates
// every field, so none of these fallbacks ever engage for it — zero behaviour change (TestMoEBlock
// gates that byte-for-byte). The MoE-specific dims (NumExperts/TopK/ExpertDFF) live here so a MoE
// layer is self-describing — model-wide dModel/dFF/eps stay executor parameters shared by dense and
// MoE layers alike.
type MoELayerWeights struct {
	NumExperts, TopK, ExpertDFF int // MoE shape (model-wide dModel/dFF/eps are args)

	PreFFNormW   []byte // local-branch input norm (gemma4); the ARCH's one pre-FFN norm (zoo)
	PreFFNorm2W  []byte // expert-branch input norm; nil (zoo) ⇒ falls back to PreFFNormW
	PostFFNorm1W []byte // post local-MLP norm; nil (zoo, no local branch) ⇒ skipped
	PostFFNorm2W []byte // post-expert norm; nil (zoo) ⇒ skipped
	PostFFNormW  []byte // final combined-branch norm; nil (zoo) ⇒ skipped

	WGate, WUp, WDown []byte // local dense MLP (dFF); all nil (zoo) ⇒ no local branch at all

	RouterNormWScaled []byte // router internal norm (pre-scaled by RootSize); nil (zoo) ⇒ see routerNormWeight
	RouterW           []byte // [NumExperts × dModel] expert-score projection
	PerExpertScale    []byte // [NumExperts] optional (nil to skip)

	ExpGateW, ExpUpW, ExpDownW []byte // experts ([NumExperts × …] at ExpertDFF) — split gate/up layout
	// ExpGateUpW is the checkpoint-native FUSED [gate‖up] expert tensor (granitemoe's
	// block_sparse_moe.input_linear: [NumExperts·2·ExpertDFF, dModel] — expert e's gate rows at
	// [e·2·ExpertDFF, e·2·ExpertDFF+ExpertDFF), up immediately after). Mirrors
	// MoEQuantLayerWeights.ExpGateUp's layout exactly (mtp_rows_moe.go's moeExpertQuantOffsets doc:
	// "gate's packed/scales/biases ahead of up's"). Bound XOR with ExpGateW/ExpUpW: len(ExpGateUpW) != 0
	// selects the offset-split read in moeBlockBF16AfterRouterWithBufferPooled instead of two separate
	// weight tensors (#59).
	ExpGateUpW []byte

	// Shared expert (qwen2_moe/qwen3_5_moe): a single always-on dense SwiGLU (SharedGateW/SharedUpW/
	// SharedDownW) + an optional σ gate (SharedSigmoidW, [1×dModel]; unbound ⇒ σ≡1) added to the routed
	// output — out += σ(SharedSigmoidW·normed) · SwiGLU(normed) — mirroring MoEQuantLayerWeights' Shared*
	// quad (arch_qwen_moe.go's encQwenMoEHalf is the quant reference for both the maths and the combine
	// point: the shared contribution sums into the SAME accumulator as the routed experts, before any
	// post-combine norm). A bound SharedGateW marks this bf16 layer qwen-shaped (#59) — every OTHER
	// executor here treats it as absent (nil/empty ⇒ no shared branch), matching the zoo/gemma4
	// tolerance the rest of this struct already documents.
	SharedGateW, SharedUpW, SharedDownW, SharedSigmoidW []byte
	// SharedDFF is the shared expert's OWN intermediate size — may differ from ExpertDFF (real
	// Qwen1.5-MoE-A2.7B ships moe_intermediate_size=1408 vs shared_expert_intermediate_size=5632, a 4x
	// mismatch). Zero falls back to ExpertDFF at the moeBlockBF16AfterRouterWithBufferPooled call site,
	// mirroring MoEQuantLayerWeights.SharedDFF (#61).
	SharedDFF int

	// UsesSiLU selects the expert-combine gate nonlinearity: false (the zero value) is gemma4's GELU
	// (encGeluGateMul/the fused gelu kernel — byte-identical to every load before #63); true is
	// SiLU/SwiGLU (encSiLUGateMulBF16 — mixtral/dbrx/olmoe's llama-family shape, and qwen-shaped bf16
	// layers' shared expert). Set from arch.Activation via ffnUsesSiLU at load (moeLoadedToBF16) — see
	// moeBlockBF16AfterRouterWithBufferPooled's emitGelu closure and its shared-expert branch, the two
	// places this flag is consulted. gemma4 never sets Arch.Activation, so this is always false for it.
	UsesSiLU bool

	// NormaliseTopK is the arch's declared router policy (model.Arch.NormaliseMoETopK, #65): true —
	// softmax over ALL experts then renormalise the gathered top-K to sum to one (mathematically the
	// SAME value as softmax over just the selected K, MoERouter's shipping shape — mixtral/granitemoe/
	// gpt-oss/qwenmoe with norm_topk_prob=true); false — softmax over ALL experts then gather the
	// top-K WITHOUT renormalising (OLMoE's norm_topk_prob=false shape; the combine weights do not sum
	// to 1). Set from arch.NormaliseMoETopK at load (moeLoadedToBF16), the same DECLARES discipline as
	// UsesSiLU/arch.Activation. gemma4 always sets it true, so this is byte-unchanged for every
	// existing gemma4 load.
	NormaliseTopK bool
}

// routerNormWeight is the norm weight MoERouter applies internally before the router's score
// projection. gemma4 declares a DEDICATED router norm (RouterNormWScaled — see the field doc); a
// llama-family zoo layer (mixtral/dbrx/olmoe/qwenmoe — #59) has none: HF's router is a plain
// nn.Linear consuming the SAME already-normed hidden state the expert branch does, with no second
// norm of its own. Falling back to PreFFNormW reproduces exactly that — MoERouter(h, PreFFNormW, …)
// computes RMSNorm(h, PreFFNormW) internally, the identical value the expert branch normalises to
// (see moeBlockBF16AfterRouterWithBufferPooled's pre2W fallback). gemma4 always sets
// RouterNormWScaled, so this fallback never engages for it.
func (w MoELayerWeights) routerNormWeight() []byte {
	if len(w.RouterNormWScaled) != 0 {
		return w.RouterNormWScaled
	}
	return w.PreFFNormW
}

// moeLoadedToBF16 maps the shared loader's MoE block (model.LoadedMoE, model.Assemble's neutral
// output) onto the native bf16 MoELayerWeights — the bf16 sibling of load_shared.go's moeToQuant,
// for any MoE arch model.Assemble can build: gemma4's dual-branch shape (every field populated), a
// llama-family zoo layer's single-norm shape (mixtral/dbrx/olmoe — #59; only PreFFNorm/Router/
// ExpGate/ExpUp/ExpDown set, everything else nil), a qwen-shaped layer's always-on shared expert
// (qwen2_moe/qwen3_moe — #59; SharedGate/Up/Down/Sigmoid additionally set), or a checkpoint-native
// fused ExpGateUp (granitemoe — #59; ExpGate/ExpUp nil, ExpGateUp set instead). Every field maps
// straight across; an absent LoadedMoE field (nil *Linear or nil/empty []byte) yields the matching
// nil MoELayerWeights field — the identity/skip convention moeBlockBF16AfterRouterWithBufferPooled
// honours (see MoELayerWeights' doc). NumExperts/TopK/ExpertDFF come from arch (model-wide);
// SharedDFF falls back to ExpertDFF when the arch declares no distinct shared width, mirroring
// moeToQuant's own sharedFF resolution (#61).
//
// WIRED into load_shared.go's loadedToBF16 (`l.MoE = moeLoadedToBF16(L.MoE, m.Arch)`, commit
// 6721442d) — the factory bf16 MoE route no longer drops expert weights on the floor. What that
// wiring did NOT yet close (closed here, #59): MoELayerWeights itself carried no shared-expert
// fields at all (a bf16 qwenmoe Generate silently dropped the shared-expert contribution), and
// moeBlockBF16AfterRouterWithBufferPooled only read the split ExpGateW/ExpUpW layout (a
// checkpoint-native fused ExpGateUp, granitemoe's block_sparse_moe.input_linear, declined with a
// shape error). See moe_zoo_generate_test.go's TestFactoryLoadQwenMoEBF16_Generate_Good and
// TestFactoryLoadGraniteMoEBF16_Generate_Good.
func moeLoadedToBF16(e *model.LoadedMoE, arch model.Arch) *MoELayerWeights {
	if e == nil {
		return nil
	}
	bw := func(lin *model.Linear) []byte {
		if lin == nil {
			return nil
		}
		return lin.Weight
	}
	sharedFF := arch.SharedExpertFF
	if sharedFF == 0 {
		sharedFF = arch.ExpertFF
	}
	return &MoELayerWeights{
		NumExperts: arch.Experts, TopK: arch.TopK, ExpertDFF: arch.ExpertFF, SharedDFF: sharedFF,
		PreFFNormW: e.PreFFNorm, PreFFNorm2W: e.PreFFNorm2,
		PostFFNorm1W: e.PostFFNorm1, PostFFNorm2W: e.PostFFNorm2, PostFFNormW: e.PostFFNorm,
		WGate: bw(e.LocalGate), WUp: bw(e.LocalUp), WDown: bw(e.LocalDown),
		RouterNormWScaled: e.RouterScale, RouterW: bw(e.Router), PerExpertScale: e.PerExpertScale,
		ExpGateW: bw(e.ExpGate), ExpUpW: bw(e.ExpUp), ExpGateUpW: bw(e.ExpGateUp), ExpDownW: bw(e.ExpDown),
		SharedGateW: bw(e.SharedGate), SharedUpW: bw(e.SharedUp), SharedDownW: bw(e.SharedDown), SharedSigmoidW: bw(e.SharedSigmoid),
		// #63: the expert combine's gate nonlinearity, resolved the same way the dense-FFN/ICB paths
		// already resolve theirs (projector.go's ffnUsesSiLU). gemma4 never sets Activation, so this
		// stays false (GELU) for every existing gemma4 load — zero behaviour change.
		UsesSiLU: ffnUsesSiLU(arch.Activation),
		// #65: gemma4 always declares NormaliseMoETopK true, so this is byte-unchanged for it.
		NormaliseTopK: arch.NormaliseMoETopK,
	}
}

// mlpTransformBF16 is the gemma SwiGLU MLP transform on an ALREADY-normed input:
// WDown·(gelu(WGate·x)·(WUp·x)) — no input norm, no residual (the MoE block applies
// those around it). Structurally one expert's computation; composed from the
// parity-proven bf16 ops encoded as one resident sequence. The per-token input is
// transient; the local dense weights are fixed per layer and stay resident like the
// selected expert weights.
func mlpTransformBF16(x, wGate, wUp, wDown []byte, dModel, dFF int) ([]byte, error) {
	return mlpTransformBF16Into(nil, x, wGate, wUp, wDown, dModel, dFF)
}

func mlpTransformBF16Into(out []byte, x, wGate, wUp, wDown []byte, dModel, dFF int) ([]byte, error) {
	return mlpTransformActivationBF16Into(out, x, wGate, wUp, wDown, dModel, dFF, false)
}

// mlpTransformActivationBF16Into is mlpTransformBF16Into with an explicit gate-activation choice:
// GELU (useSiLU false — gemma4's local dense MLP; mlpTransformBF16Into's existing callers all resolve
// here, byte-identical) or SiLU/SwiGLU (useSiLU true —
// moeBlockBF16AfterRouterWithBufferPooled's qwen-shaped shared-expert branch, #63).
func mlpTransformActivationBF16Into(out []byte, x, wGate, wUp, wDown []byte, dModel, dFF int, useSiLU bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformBF16: x must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformBF16: wGate/wUp must be dFF*dModel bf16 bytes")
	}
	if len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.mlpTransformBF16: wDown must be dModel*dFF bf16 bytes")
	}
	outLen := dModel * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMLPTransformScratch(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		defer putMLPTransformScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		wgBuf, wuBuf, wdBuf := residentBytes(wGate), residentBytes(wUp), residentBytes(wDown)
		msc := scratch.mlp
		outBuf := msc.down
		directOut := false
		if tmp, ok := scratch.outputView(out); ok {
			outBuf = tmp
			directOut = true
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encGemvBF16(enc, wgBuf, xBuf, msc.gate, dFF, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encGemvBF16(enc, wuBuf, xBuf, msc.up, dFF, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if useSiLU {
			encErr = encSiLUGateMulBF16(enc, msc.gate, msc.up, msc.gated, dFF)
		} else {
			encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF)
		}
		if encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encGemvBF16(enc, wdBuf, msc.gated, outBuf, dModel, dFF); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(msc.down.Contents()), len(out)))
		}
	})
	return out, encErr
}

type moeBlockPostCombineScratch struct {
	dModel                       int
	h, h1, h2, out               *pinnedNoCopyBytes
	hPinned, h1Pinned, h2Pinned  *pinnedNoCopyBytes
	h1Normed, h2Normed, combined metal.MTLBuffer
	ffResidual                   metal.MTLBuffer
}

type scratchLIFOPool[T any] struct {
	mu    sync.Mutex
	items []T
}

func (p *scratchLIFOPool[T]) Get() T {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.items)
	if n == 0 {
		var zero T
		return zero
	}
	item := p.items[n-1]
	var zero T
	p.items[n-1] = zero
	p.items = p.items[:n-1]
	return item
}

func (p *scratchLIFOPool[T]) Put(item T) {
	p.mu.Lock()
	p.items = append(p.items, item)
	p.mu.Unlock()
}

var moeBlockPostCombineScratchPools sync.Map

func newMoEBlockPostCombineScratch(dModel int) (*moeBlockPostCombineScratch, error) {
	size := dModel * bf16Size
	h, err := newPinnedNoCopyBytes(size)
	if err != nil {
		return nil, err
	}
	h1, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		return nil, err
	}
	h2, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		h1.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		h1.Close()
		h2.Close()
		return nil, err
	}
	return &moeBlockPostCombineScratch{
		dModel:     dModel,
		h:          h,
		h1:         h1,
		h2:         h2,
		out:        out,
		h1Normed:   scratchBF16(dModel),
		h2Normed:   scratchBF16(dModel),
		combined:   scratchBF16(dModel),
		ffResidual: scratchBF16(dModel),
	}, nil
}

func getMoEBlockPostCombineScratch(dModel int) (*moeBlockPostCombineScratch, error) {
	pool := moeBlockPostCombineScratchPoolFor(dModel)
	if s := pool.Get(); s != nil {
		if s != nil &&
			s.dModel == dModel &&
			s.h != nil && s.h.buf != nil &&
			s.h1 != nil && s.h1.buf != nil &&
			s.h2 != nil && s.h2.buf != nil &&
			s.out != nil && s.out.buf != nil &&
			s.h1Normed != nil &&
			s.h2Normed != nil &&
			s.combined != nil &&
			s.ffResidual != nil {
			return s, nil
		}
		s.Close()
	}
	return newMoEBlockPostCombineScratch(dModel)
}

func moeBlockPostCombineScratchPoolFor(dModel int) *scratchLIFOPool[*moeBlockPostCombineScratch] {
	if v, ok := moeBlockPostCombineScratchPools.Load(dModel); ok {
		return v.(*scratchLIFOPool[*moeBlockPostCombineScratch])
	}
	pool := &scratchLIFOPool[*moeBlockPostCombineScratch]{}
	if v, loaded := moeBlockPostCombineScratchPools.LoadOrStore(dModel, pool); loaded {
		return v.(*scratchLIFOPool[*moeBlockPostCombineScratch])
	}
	return pool
}

func putMoEBlockPostCombineScratch(s *moeBlockPostCombineScratch) {
	if s != nil &&
		s.h != nil && s.h.buf != nil &&
		s.h1 != nil && s.h1.buf != nil &&
		s.h2 != nil && s.h2.buf != nil &&
		s.out != nil && s.out.buf != nil &&
		s.h1Normed != nil &&
		s.h2Normed != nil &&
		s.combined != nil &&
		s.ffResidual != nil {
		moeBlockPostCombineScratchPoolFor(s.dModel).Put(s)
	}
}

func (s *moeBlockPostCombineScratch) Close() {
	if s == nil {
		return
	}
	if s.h != nil {
		s.h.Close()
		s.h = nil
	}
	if s.hPinned != nil {
		s.hPinned.Close()
		s.hPinned = nil
	}
	if s.h1 != nil {
		s.h1.Close()
		s.h1 = nil
	}
	if s.h1Pinned != nil {
		s.h1Pinned.Close()
		s.h1Pinned = nil
	}
	if s.h2 != nil {
		s.h2.Close()
		s.h2 = nil
	}
	if s.h2Pinned != nil {
		s.h2Pinned.Close()
		s.h2Pinned = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.dModel = 0
}

func postCombineInputView(slot **pinnedNoCopyBytes, x []byte) (metal.MTLBuffer, bool) {
	if len(x) == 0 {
		return nil, false
	}
	if pinned := *slot; pinned != nil && len(pinned.bytes) == len(x) && &pinned.bytes[0] == &x[0] {
		return pinned.buf, true
	}
	if *slot != nil {
		(*slot).Close()
		*slot = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(x); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(x)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: x, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	*slot = pinned
	return buf, true
}

func (s *moeBlockPostCombineScratch) residualView(h []byte) (metal.MTLBuffer, bool) {
	if s == nil {
		return nil, false
	}
	return postCombineInputView(&s.hPinned, h)
}

func (s *moeBlockPostCombineScratch) branch1View(h1 []byte) (metal.MTLBuffer, bool) {
	if s == nil {
		return nil, false
	}
	return postCombineInputView(&s.h1Pinned, h1)
}

func (s *moeBlockPostCombineScratch) branch2View(h2 []byte) (metal.MTLBuffer, bool) {
	if s == nil {
		return nil, false
	}
	return postCombineInputView(&s.h2Pinned, h2)
}

func moeBlockPostCombineBF16(h, h1, h2 []byte, post1 []byte, post1View bufView, post2 []byte, post2View bufView, post []byte, postView bufView, dModel int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	size := dModel * bf16Size
	if len(h) != size || len(h1) != size || len(h2) != size {
		return nil, core.NewError("native.moeBlockPostCombineBF16: h/h1/h2 must be dModel bf16 bytes")
	}
	if len(post1) != size || len(post2) != size || len(post) != size {
		return nil, core.NewError("native.moeBlockPostCombineBF16: post norm weights must be dModel bf16 bytes")
	}
	out := make([]byte, size)
	if dModel == 0 {
		return out, nil
	}
	post1Buf := bf16WeightView(post1, post1View)
	post2Buf := bf16WeightView(post2, post2View)
	postBuf := bf16WeightView(post, postView)

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMoEBlockPostCombineScratch(dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEBlockPostCombineScratch(scratch)
		hBuf, ok := scratch.residualView(h)
		if !ok {
			hBuf, err = scratch.h.copyBuffer(h)
			if err != nil {
				encErr = err
				return
			}
		}
		h1Buf, ok := scratch.branch1View(h1)
		if !ok {
			h1Buf, err = scratch.h1.copyBuffer(h1)
			if err != nil {
				encErr = err
				return
			}
		}
		h2Buf, ok := scratch.branch2View(h2)
		if !ok {
			h2Buf, err = scratch.h2.copyBuffer(h2)
			if err != nil {
				encErr = err
				return
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encRMSNormBF16(enc, h1Buf, post1Buf.buf, scratch.h1Normed, post1Buf.off, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encRMSNormBF16(enc, h2Buf, post2Buf.buf, scratch.h2Normed, post2Buf.off, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encAddBF16(enc, scratch.h1Normed, scratch.h2Normed, scratch.combined, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encRMSNormBF16(enc, scratch.combined, postBuf.buf, scratch.ffResidual, postBuf.off, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encAddBF16(enc, hBuf, scratch.ffResidual, scratch.out.buf, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, scratch.out.bytes[:size])
	})
	return out, encErr
}

type moeBlockBF16Scratch struct {
	dModel, dFF, expertDFF, topK int
	h, weights, idx, out         *pinnedNoCopyBytes
	hPinned                      *pinnedNoCopyBytes
	weightsPinned                *pinnedNoCopyBytes
	idxPinned                    *pinnedNoCopyBytes
	outPinned                    *pinnedNoCopyBytes
	mlp                          mlpScratch
	localIn, expertIn            metal.MTLBuffer
	localOut                     metal.MTLBuffer
	expertScaled, expertAcc      metal.MTLBuffer
	localNormed, expertNormed    metal.MTLBuffer
	combined, ffResidual         metal.MTLBuffer
	localMegaGated               metal.MTLBuffer
	localMegaArrive              metal.MTLBuffer
	localMegaArrivePtr           *uint32
	// all-routes expert slabs (the encoded lane's single-dispatch expert projections):
	// [topK × expertDFF] gate/up/gated + [topK × dModel] down, plus the constant index
	// buffers the MLX gather batch dimension consumes (zeros for the shared-x lhs, iota
	// for the per-route gated rows).
	expertGateAll, expertUpAll, expertGatedAll, expertDownAll metal.MTLBuffer
	routeZeros, routeIota                                     metal.MTLBuffer
}

// ensureAllRoutesScratch sizes the all-routes expert slabs (idempotent; the scratch's
// dims are fixed at construction so a single build serves every layer).
func (s *moeBlockBF16Scratch) ensureAllRoutesScratch() error {
	if s.expertGateAll != nil {
		return nil
	}
	if s.topK <= 0 || s.expertDFF <= 0 || s.dModel <= 0 {
		return core.NewError("native.moeBlockScratch: all-routes scratch needs topK/expertDFF/dModel")
	}
	ffBytes := uint(s.topK * s.expertDFF * bf16Size)
	s.expertGateAll = device.NewBufferWithLengthOptions(ffBytes, metal.MTLResourceStorageModeShared)
	s.expertUpAll = device.NewBufferWithLengthOptions(ffBytes, metal.MTLResourceStorageModeShared)
	s.expertGatedAll = device.NewBufferWithLengthOptions(ffBytes, metal.MTLResourceStorageModeShared)
	s.expertDownAll = device.NewBufferWithLengthOptions(uint(s.topK*s.dModel*bf16Size), metal.MTLResourceStorageModeShared)
	zeros := make([]int32, s.topK)
	iota := make([]int32, s.topK)
	for i := range iota {
		iota[i] = int32(i)
	}
	s.routeZeros = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&zeros[0]), uint(s.topK*4), metal.MTLResourceStorageModeShared)
	s.routeIota = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&iota[0]), uint(s.topK*4), metal.MTLResourceStorageModeShared)
	if s.expertGateAll == nil || s.expertUpAll == nil || s.expertGatedAll == nil || s.expertDownAll == nil || s.routeZeros == nil || s.routeIota == nil {
		s.expertGateAll, s.expertUpAll, s.expertGatedAll, s.expertDownAll = nil, nil, nil, nil
		s.routeZeros, s.routeIota = nil, nil
		return core.NewError("native.moeBlockScratch: all-routes scratch unavailable")
	}
	return nil
}

type moeBlockBF16ScratchKey struct {
	dModel, dFF, expertDFF, topK int
}

var moeBlockBF16ScratchPools sync.Map

func newMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK int) (*moeBlockBF16Scratch, error) {
	size := dModel * bf16Size
	h, err := newPinnedNoCopyBytes(size)
	if err != nil {
		return nil, err
	}
	weightsSize := topK * bf16Size
	if weightsSize <= 0 {
		weightsSize = bf16Size
	}
	weights, err := newPinnedNoCopyBytes(weightsSize)
	if err != nil {
		h.Close()
		return nil, err
	}
	idxSize := topK * 4
	if idxSize <= 0 {
		idxSize = 4
	}
	idx, err := newPinnedNoCopyBytes(idxSize)
	if err != nil {
		h.Close()
		weights.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		weights.Close()
		idx.Close()
		return nil, err
	}
	scratchDFF := max(expertDFF, dFF)
	return &moeBlockBF16Scratch{
		dModel:       dModel,
		dFF:          dFF,
		expertDFF:    expertDFF,
		topK:         topK,
		h:            h,
		weights:      weights,
		idx:          idx,
		out:          out,
		mlp:          newMLPScratch(dModel, scratchDFF),
		localIn:      scratchBF16(dModel),
		expertIn:     scratchBF16(dModel),
		localOut:     scratchBF16(dModel),
		expertScaled: scratchBF16(dModel),
		expertAcc:    scratchBF16(dModel),
		localNormed:  scratchBF16(dModel),
		expertNormed: scratchBF16(dModel),
		combined:     scratchBF16(dModel),
		ffResidual:   scratchBF16(dModel),
	}, nil
}

func moeBlockBF16ScratchPoolFor(dModel, dFF, expertDFF, topK int) *scratchLIFOPool[*moeBlockBF16Scratch] {
	key := moeBlockBF16ScratchKey{dModel: dModel, dFF: dFF, expertDFF: expertDFF, topK: topK}
	if v, ok := moeBlockBF16ScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*moeBlockBF16Scratch])
	}
	pool := &scratchLIFOPool[*moeBlockBF16Scratch]{}
	if v, loaded := moeBlockBF16ScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*moeBlockBF16Scratch])
	}
	return pool
}

func getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK int) (*moeBlockBF16Scratch, error) {
	pool := moeBlockBF16ScratchPoolFor(dModel, dFF, expertDFF, topK)
	if s := pool.Get(); s != nil {
		wantWeights := topK * bf16Size
		if wantWeights <= 0 {
			wantWeights = bf16Size
		}
		wantIdx := topK * 4
		if wantIdx <= 0 {
			wantIdx = 4
		}
		if s != nil &&
			s.dModel == dModel &&
			s.dFF == dFF &&
			s.expertDFF == expertDFF &&
			s.topK == topK &&
			s.h != nil && s.h.buf != nil &&
			s.weights != nil && s.weights.buf != nil && len(s.weights.bytes) == wantWeights &&
			s.idx != nil && s.idx.buf != nil && len(s.idx.bytes) == wantIdx &&
			s.out != nil && s.out.buf != nil &&
			s.mlp.gate != nil &&
			s.mlp.up != nil &&
			s.mlp.gated != nil &&
			s.mlp.down != nil &&
			s.localIn != nil &&
			s.expertIn != nil &&
			s.localOut != nil &&
			s.expertScaled != nil &&
			s.expertAcc != nil &&
			s.localNormed != nil &&
			s.expertNormed != nil &&
			s.combined != nil &&
			s.ffResidual != nil {
			return s, nil
		}
		s.Close()
	}
	return newMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
}

func putMoEBlockBF16Scratch(s *moeBlockBF16Scratch) {
	if s != nil &&
		s.h != nil && s.h.buf != nil &&
		s.weights != nil && s.weights.buf != nil &&
		s.idx != nil && s.idx.buf != nil &&
		s.out != nil && s.out.buf != nil &&
		s.mlp.gate != nil &&
		s.mlp.up != nil &&
		s.mlp.gated != nil &&
		s.mlp.down != nil &&
		s.localIn != nil &&
		s.expertIn != nil &&
		s.localOut != nil &&
		s.expertScaled != nil &&
		s.expertAcc != nil &&
		s.localNormed != nil &&
		s.expertNormed != nil &&
		s.combined != nil &&
		s.ffResidual != nil {
		moeBlockBF16ScratchPoolFor(s.dModel, s.dFF, s.expertDFF, s.topK).Put(s)
	}
}

func (s *moeBlockBF16Scratch) ensureLocalMegaScratch() error {
	if s.localMegaGated != nil && s.localMegaArrive != nil && s.localMegaArrivePtr != nil {
		return nil
	}
	s.localMegaGated = device.NewBufferWithLengthOptions(uint(s.dFF*4), metal.MTLResourceStorageModeShared)
	s.localMegaArrive = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	if s.localMegaGated == nil || s.localMegaGated.GetID() == 0 || s.localMegaArrive == nil || s.localMegaArrive.GetID() == 0 {
		s.localMegaGated = nil
		s.localMegaArrive = nil
		s.localMegaArrivePtr = nil
		return core.NewError("native.moeBlockScratch: local megakernel scratch unavailable")
	}
	s.localMegaArrivePtr = (*uint32)(s.localMegaArrive.Contents())
	return nil
}

func (s *moeBlockBF16Scratch) Close() {
	if s == nil {
		return
	}
	if s.h != nil {
		s.h.Close()
		s.h = nil
	}
	if s.hPinned != nil {
		s.hPinned.Close()
		s.hPinned = nil
	}
	if s.weights != nil {
		s.weights.Close()
		s.weights = nil
	}
	if s.weightsPinned != nil {
		s.weightsPinned.Close()
		s.weightsPinned = nil
	}
	if s.idx != nil {
		s.idx.Close()
		s.idx = nil
	}
	if s.idxPinned != nil {
		s.idxPinned.Close()
		s.idxPinned = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	if s.outPinned != nil {
		s.outPinned.Close()
		s.outPinned = nil
	}
	s.localMegaGated = nil
	s.localMegaArrive = nil
	s.localMegaArrivePtr = nil
	s.dModel, s.dFF, s.expertDFF, s.topK = 0, 0, 0, 0
}

func (s *moeBlockBF16Scratch) inputView(h []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(h) == 0 {
		return nil, false
	}
	if s.hPinned != nil && len(s.hPinned.bytes) == len(h) && &s.hPinned.bytes[0] == &h[0] {
		return s.hPinned.buf, true
	}
	if s.hPinned != nil {
		s.hPinned.Close()
		s.hPinned = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(h); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(h)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: h, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.hPinned = pinned
	return buf, true
}

func (s *moeBlockBF16Scratch) weightsView(weights []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(weights) == 0 {
		return nil, false
	}
	if s.weightsPinned != nil && len(s.weightsPinned.bytes) == len(weights) && &s.weightsPinned.bytes[0] == &weights[0] {
		return s.weightsPinned.buf, true
	}
	if s.weightsPinned != nil {
		s.weightsPinned.Close()
		s.weightsPinned = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(weights); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(weights)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: weights, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.weightsPinned = pinned
	return buf, true
}

func (s *moeBlockBF16Scratch) indexView(idx []int32) (metal.MTLBuffer, bool) {
	if s == nil || len(idx) == 0 {
		return nil, false
	}
	idxBytes := unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), len(idx)*4)
	if s.idxPinned != nil && len(s.idxPinned.bytes) == len(idxBytes) && &s.idxPinned.bytes[0] == &idxBytes[0] {
		return s.idxPinned.buf, true
	}
	if s.idxPinned != nil {
		return nil, false
	}
	if buf, ok := registeredPinnedNoCopyBytes(idxBytes); ok {
		runtime.KeepAlive(idx)
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(idxBytes)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: idxBytes, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.idxPinned = pinned
	runtime.KeepAlive(idx)
	return buf, true
}

func (s *moeBlockBF16Scratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	if s.outPinned != nil && len(s.outPinned.bytes) == len(out) && &s.outPinned.bytes[0] == &out[0] {
		return s.outPinned.buf, true
	}
	if s.outPinned != nil {
		s.outPinned.Close()
		s.outPinned = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outPinned = pinned
	return buf, true
}

func moeBlockBF16AfterRouter(h []byte, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBuffer(h, nil, idx, weights, weightBuf, w, dModel, dFF, eps)
}

func moeBlockBF16AfterRouterWithBuffer(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, false)
}

func moeBlockBF16AfterRouterWithBufferInPool(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
}

func moeBlockBF16AfterRouterWithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockBF16AfterRouterWithBufferIntoInPool(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, true)
}

func moeBlockBF16AfterRouterWithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.moeBlockBF16AfterRouter: output buffer is nil")
	}
	_, err := moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, nil, outputBuf, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
	return err
}

// sigmoidF32 is the plain host logistic 1/(1+e^-x) — the shared-expert σ gate's nonlinearity,
// mirroring arch_qwen_moe.go's sharedGateSigmoid exactly (same maths; float32 in/out here since the
// bf16 shared branch never needs sharedGateSigmoid's float64 accumulation for a single scalar gate).
func sigmoidF32(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

func moeBlockBF16AfterRouterWithBufferPooled(h []byte, hBuf metal.MTLBuffer, out []byte, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	expertDFF, numExperts, topK := w.ExpertDFF, w.NumExperts, w.TopK
	size := dModel * bf16Size
	if len(h) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: h must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: idx/weights length must equal topK")
	}
	// PreFFNormW is always required: gemma4's local-branch input norm, and — for a llama-family zoo
	// layer with no local branch at all (mixtral/dbrx/olmoe: see MoELayerWeights doc) — the single
	// pre-FFN norm shared by the router and the expert branch. Every OTHER norm here is OPTIONAL:
	// nil/empty = identity/skip, matching what the arch actually declares. gemma4's checkpoint always
	// populates all five, so this is zero behaviour change for it (TestMoEBlock gates that).
	if len(w.PreFFNormW) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: PreFFNormW must be dModel bf16 bytes")
	}
	if len(w.PreFFNorm2W) != 0 && len(w.PreFFNorm2W) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: PreFFNorm2W must be dModel bf16 bytes or absent")
	}
	if len(w.PostFFNorm1W) != 0 && len(w.PostFFNorm1W) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: PostFFNorm1W must be dModel bf16 bytes or absent")
	}
	if len(w.PostFFNorm2W) != 0 && len(w.PostFFNorm2W) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: PostFFNorm2W must be dModel bf16 bytes or absent")
	}
	if len(w.PostFFNormW) != 0 && len(w.PostFFNormW) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: PostFFNormW must be dModel bf16 bytes or absent")
	}
	// hasLocal marks gemma4's always-on dense MLP branch running alongside the routed experts; a zoo
	// layer (mixtral/dbrx/olmoe) declares none of the three (see MoELayerWeights doc), so the local
	// branch — its pipelines, its pre/post norms — is skipped entirely below.
	hasLocal := len(w.WGate) != 0 || len(w.WUp) != 0 || len(w.WDown) != 0
	if hasLocal {
		if len(w.WGate) != dFF*dModel*bf16Size || len(w.WUp) != dFF*dModel*bf16Size {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: local gate/up weights must be dFF*dModel bf16 bytes")
		}
		if len(w.WDown) != dModel*dFF*bf16Size {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: local down weight must be dModel*dFF bf16 bytes")
		}
	}
	gateSz, downSz := expertDFF*dModel*bf16Size, dModel*expertDFF*bf16Size
	// fusedExperts marks a checkpoint-native FUSED [gate‖up] expert tensor (granitemoe's
	// block_sparse_moe.input_linear — see ExpGateUpW's doc); the split ExpGateW/ExpUpW pair is the
	// default (gemma4, mixtral/dbrx/olmoe, qwenmoe). Exactly one layout is validated — never both.
	fusedExperts := len(w.ExpGateUpW) != 0
	if fusedExperts {
		if len(w.ExpGateUpW) != numExperts*2*gateSz {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: fused expert gate_up weight size mismatch")
		}
	} else if len(w.ExpGateW) != numExperts*gateSz || len(w.ExpUpW) != numExperts*gateSz {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: expert weight size mismatch")
	}
	if len(w.ExpDownW) != numExperts*downSz {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: expert weight size mismatch")
	}
	// hasShared marks a qwen-shaped layer's always-on shared expert (see SharedGateW's doc) — gated on
	// SharedGateW alone, mirroring the quant reference's OWN marker convention (arch_qwen_moe.go: "A
	// bound SharedGate MARKS a qwen MoE layer"). sharedFF falls back to ExpertDFF when the arch declares
	// no distinct shared width, mirroring MoEQuantLayerWeights.SharedDFF's own fallback (#61) and
	// moeLoadedToBF16's resolution.
	hasShared := len(w.SharedGateW) != 0
	sharedFF := w.SharedDFF
	if sharedFF == 0 {
		sharedFF = expertDFF
	}
	if hasShared {
		sharedGateSz := sharedFF * dModel * bf16Size
		if len(w.SharedGateW) != sharedGateSz || len(w.SharedUpW) != sharedGateSz {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: shared expert gate/up weights must be SharedDFF*dModel bf16 bytes")
		}
		if len(w.SharedDownW) != dModel*sharedFF*bf16Size {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: shared expert down weight must be dModel*SharedDFF bf16 bytes")
		}
		if len(w.SharedSigmoidW) != 0 && len(w.SharedSigmoidW) != size {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: shared expert sigmoid gate must be dModel bf16 bytes or absent")
		}
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: expert index out of range")
		}
	}
	bufferOut := outputBuf != nil
	callerOut := !bufferOut && useCallerOut && cap(out) >= size
	if bufferOut {
		out = nil
	} else if callerOut {
		out = out[:size]
	} else {
		out = make([]byte, size)
	}
	if dModel == 0 || expertDFF == 0 || (hasLocal && dFF == 0) {
		if bufferOut && size > 0 {
			clear(unsafe.Slice((*byte)(outputBuf.Contents()), size))
			return nil, nil
		}
		if !bufferOut {
			clear(out)
		}
		return out, nil
	}

	// pre2W is the expert branch's input norm. A zoo layer sets only PreFFNormW (the single pre-FFN
	// norm shared by router + experts — see MoEWeightNames doc); fall back to it when PreFFNorm2W is
	// absent, so the expert branch normalises on exactly the value HF's llama-family MoE feeds its
	// router AND its experts. gemma4 always sets PreFFNorm2W distinctly, so this fallback never
	// engages for it.
	pre2W := w.PreFFNorm2W
	if len(pre2W) == 0 {
		pre2W = w.PreFFNormW
	}
	hasPost1 := len(w.PostFFNorm1W) != 0
	hasPost2 := len(w.PostFFNorm2W) != 0
	hasPostCombine := len(w.PostFFNormW) != 0

	pre1Buf := bf16WeightView(w.PreFFNormW, bufView{})
	pre2Buf := bf16WeightView(pre2W, bufView{})
	var post1Buf, post2Buf, postBuf bufView
	if hasPost1 {
		post1Buf = bf16WeightView(w.PostFFNorm1W, bufView{})
	}
	if hasPost2 {
		post2Buf = bf16WeightView(w.PostFFNorm2W, bufView{})
	}
	if hasPostCombine {
		postBuf = bf16WeightView(w.PostFFNormW, bufView{})
	}
	var localGate, localUp, localDown metal.MTLBuffer
	if hasLocal {
		localGate, localUp, localDown = residentBytes(w.WGate), residentBytes(w.WUp), residentBytes(w.WDown)
	}
	var expertGate, expertUp, expertGateUp metal.MTLBuffer
	if fusedExperts {
		expertGateUp = residentBytes(w.ExpGateUpW)
	} else {
		expertGate, expertUp = residentBytes(w.ExpGateW), residentBytes(w.ExpUpW)
	}
	expertDown := residentBytes(w.ExpDownW)
	var localInBM, localInBN, localInSM, localInSN, localInTM, localInTN int
	var localDownBM, localDownBN, localDownSM, localDownSN, localDownTM, localDownTN int
	var localInPSO, localDownPSO metal.MTLComputePipelineState
	if hasLocal {
		var err error
		localInBM, localInBN, localInSM, localInSN, localInTM, localInTN = gemvTiles(dModel, dFF)
		localInPSO, err = pipelineFor(gemvKernelName("bfloat16", localInBM, localInBN, localInSM, localInSN, localInTM, localInTN))
		if err != nil {
			return nil, err
		}
		localDownBM, localDownBN, localDownSM, localDownSN, localDownTM, localDownTN = gemvTiles(dFF, dModel)
		localDownPSO, err = pipelineFor(gemvKernelName("bfloat16", localDownBM, localDownBN, localDownSM, localDownSN, localDownTM, localDownTN))
		if err != nil {
			return nil, err
		}
	}
	expertInBM, expertInBN, expertInSM, expertInSN, expertInTM, expertInTN := gemvTiles(dModel, expertDFF)
	expertInPSO, err := pipelineFor(gemvKernelName("bfloat16", expertInBM, expertInBN, expertInSM, expertInSN, expertInTM, expertInTN))
	if err != nil {
		return nil, err
	}
	expertDownBM, expertDownBN, expertDownSM, expertDownSN, expertDownTM, expertDownTN := gemvTiles(expertDFF, dModel)
	expertDownPSO, err := pipelineFor(gemvKernelName("bfloat16", expertDownBM, expertDownBN, expertDownSM, expertDownSN, expertDownTM, expertDownTN))
	if err != nil {
		return nil, err
	}
	// Shared expert (qwen-shaped, see SharedGateW's doc): a THIRD independent branch, sized at sharedFF
	// (may differ from both dFF and expertDFF), computed here via the self-contained
	// mlpTransformActivationBF16Into (its own pooled scratch keyed on (dModel, sharedFF) — never the
	// fixed-width msc scratch run() uses below, which is sized max(dFF, expertDFF) and would silently
	// truncate a wider shared branch) rather than folded into the live encoder. normedHost is
	// rms(h, pre2W) — the SAME value the routed experts normalise on (scratch.expertIn inside run()) —
	// matching the quant reference (encQwenMoEHalf), which feeds ONE normed value to both the routed
	// and the shared expert. sharedContribution (host bf16, dModel) is added into scratch.expertAcc
	// inside run(), after the routed-expert loop and before any post-combine norm — exactly where
	// encQwenMoEHalf sums the shared expert into the SAME accumulator as the routed experts,
	// pre-residual. The shared branch's gate uses w.UsesSiLU — the SAME per-layer activation as the
	// routed experts (#63): every real arch that ships a shared expert (qwen's family) declares SiLU
	// for both branches uniformly, never a mix.
	var sharedContribution []byte
	if hasShared {
		normedHost, nerr := RMSNormBF16Into(nil, h, pre2W, 1, dModel, eps)
		if nerr != nil {
			return nil, nerr
		}
		mlpOut, merr := mlpTransformActivationBF16Into(nil, normedHost, w.SharedGateW, w.SharedUpW, w.SharedDownW, dModel, sharedFF, w.UsesSiLU)
		if merr != nil {
			return nil, merr
		}
		sigmoidGate := float32(1)
		if len(w.SharedSigmoidW) != 0 {
			gl, gerr := MatVecBF16(w.SharedSigmoidW, normedHost, 1, dModel)
			if gerr != nil {
				return nil, gerr
			}
			sigmoidGate = sigmoidF32(bf16ToF32Slice(gl)[0])
		}
		mlpOutF32 := bf16ToF32Slice(mlpOut)
		for i := range mlpOutF32 {
			mlpOutF32[i] *= sigmoidGate
		}
		sharedContribution = f32ToBf16Slice(mlpOutF32)
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
	}
	// #63: the layer's declared expert-combine activation — gemma4 (Activation unset) stays GELU,
	// byte-identical; a llama-family zoo layer (mixtral/dbrx/olmoe) or a qwen-shaped layer's routed
	// experts (Activation "silu") switch to SiLU. Applies uniformly to BOTH branches emitGelu serves
	// below (gemma4's local dense MLP, every arch's routed experts) — the two never coexist with a
	// mismatched activation on any registered arch.
	useSiLU := w.UsesSiLU
	scalePSO, scaleErr := bf16MulScalarPipeline()

	var encErr error
	run := func() {
		scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEBlockBF16Scratch(scratch)
		inputBuf := hBuf
		if inputBuf == nil {
			var ok bool
			inputBuf, ok = scratch.inputView(h)
			if !ok {
				inputBuf, err = scratch.h.copyBuffer(h)
				if err != nil {
					encErr = err
					return
				}
			}
		}
		weightsBuf := weightBuf
		if topK > 0 {
			if weightsBuf == nil {
				var ok bool
				weightsBuf, ok = scratch.weightsView(weights)
				if !ok {
					weightsBuf, err = scratch.weights.copyBuffer(weights)
					if err != nil {
						encErr = err
						return
					}
				}
			}
		} else {
			clear(unsafe.Slice((*byte)(scratch.expertAcc.Contents()), size))
		}
		msc := scratch.mlp
		finalOutBuf := scratch.out.buf
		directOut := false
		if bufferOut {
			finalOutBuf = outputBuf
			directOut = true
		} else if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				finalOutBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMS := func(x, weight, out metal.MTLBuffer, wOff uint) {
			emitRMSNorm(sink, rmsPSO, x, weight, out, wOff, dModel, eps, rmsTG)
		}
		emitLocalInGemv := func(mat, vec, out metal.MTLBuffer, matOff uint) {
			emitGemv(sink, localInPSO, mat, matOff, vec, out, 0, dModel, dFF, localInBM, localInBN, localInSM, localInTM)
		}
		emitLocalDownGemv := func(mat, vec, out metal.MTLBuffer) {
			emitGemv(sink, localDownPSO, mat, 0, vec, out, 0, dFF, dModel, localDownBM, localDownBN, localDownSM, localDownTM)
		}
		emitExpertInGemv := func(mat, vec, out metal.MTLBuffer, matOff uint) {
			emitGemv(sink, expertInPSO, mat, matOff, vec, out, 0, dModel, expertDFF, expertInBM, expertInBN, expertInSM, expertInTM)
		}
		emitExpertDownGemv := func(mat, vec, out metal.MTLBuffer, matOff uint) {
			emitGemv(sink, expertDownPSO, mat, matOff, vec, out, 0, expertDFF, dModel, expertDownBM, expertDownBN, expertDownSM, expertDownTM)
		}
		// emitGelu, despite its name (kept for callers below), is the layer's whole gate-activation
		// choice (#63): useSiLU routes to the SwiGLU gate first — gemma4 (useSiLU false) falls through
		// to the original GELU dispatch unchanged.
		emitGelu := func(gate, up, out metal.MTLBuffer, n int) error {
			if useSiLU {
				return encSiLUGateMulBF16(enc, gate, up, out, n)
			}
			if useFusedGelu {
				emitBinary(sink, geluPSO, gate, 0, up, 0, out, 0, n)
				return nil
			}
			return encGeluGateMul(enc, gate, up, out, msc, n)
		}
		emitScale := func(in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
			if scaleErr != nil {
				return encScaleBF16(enc, in, scalar, out, scalarOffset, scalarBytes, n)
			}
			sink.setPSO(scalePSO)
			sink.setBuf(in, 0, 0)
			sink.setBuf(scalar, scalarOffset, 1)
			sink.setBuf(out, 0, 2)
			sink.setI32(int32(n), 3)
			group := min(uint(n), uint(256))
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
				metal.MTLSize{Width: group, Height: 1, Depth: 1},
			)
			return nil
		}
		emitAdd := func(a, b, out metal.MTLBuffer) {
			emitBinary(sink, addPSO, a, 0, b, 0, out, 0, dModel)
		}
		if hasLocal {
			emitRMS(inputBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
			emitLocalInGemv(localGate, scratch.localIn, msc.gate, 0)
			emitLocalInGemv(localUp, scratch.localIn, msc.up, 0)
			if encErr = emitGelu(msc.gate, msc.up, msc.gated, dFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			emitLocalDownGemv(localDown, msc.gated, scratch.localOut)
		}
		emitRMS(inputBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
		for i := range topK {
			e := int(idx[i])
			downOff := uint(e * downSz)
			if fusedExperts {
				// expert e's fused block is rows [e·2·expertDFF, (e+1)·2·expertDFF) of ExpGateUpW: gate is
				// the block's own start, up is the very next expertDFF-sized slice — two separate GEMV
				// dispatches against the SAME resident tensor, never a single gemv-then-split kernel
				// (mirrors encMoEBlockQuantDevice/mtp_rows_moe.go's own fused handling).
				base := uint(e * 2 * gateSz)
				emitExpertInGemv(expertGateUp, scratch.expertIn, msc.gate, base)
				emitExpertInGemv(expertGateUp, scratch.expertIn, msc.up, base+uint(gateSz))
			} else {
				gateOff := uint(e * gateSz)
				emitExpertInGemv(expertGate, scratch.expertIn, msc.gate, gateOff)
				emitExpertInGemv(expertUp, scratch.expertIn, msc.up, gateOff)
			}
			if encErr = emitGelu(msc.gate, msc.up, msc.gated, expertDFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			emitExpertDownGemv(expertDown, msc.gated, msc.down, downOff)
			if i == 0 {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertAcc, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertScaled, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
		// Shared expert (qwen-shaped, #59): sum the pre-computed sharedContribution into the SAME
		// accumulator as the routed experts, before any post-combine norm — exactly where the quant host
		// path (encQwenMoEHalf) adds it. No-op (sharedContribution nil, hasShared false) for every
		// non-qwen-shaped layer — gemma4/zoo output is byte-for-byte untouched.
		if hasShared {
			sharedBuf := sharedBytes(sharedContribution)
			emitAdd(scratch.expertAcc, sharedBuf, scratch.expertAcc)
		}
		// Combine: gemma4 sums BOTH branches (each independently post-normed) then post-norms the sum.
		// A zoo layer with no local branch (hasLocal false) skips straight to the expert accumulator —
		// HF's llama-family MoE adds the routed output directly to the residual, no sandwich norm at
		// all (see MoELayerWeights doc) — and any INDIVIDUAL norm this arch omits is skipped in place
		// (nil/empty = identity), so a hybrid shape degrades gracefully too. Dispatch order matches the
		// original unconditional gemma4 sequence exactly (localNormed, then expertNormed, then the
		// combine add, then the post-combine norm) so gemma4's byte-for-byte output is untouched.
		var localBranch metal.MTLBuffer
		if hasLocal {
			localBranch = scratch.localOut
			if hasPost1 {
				emitRMS(scratch.localOut, post1Buf.buf, scratch.localNormed, post1Buf.off)
				localBranch = scratch.localNormed
			}
		}
		ffResidualSrc := scratch.expertAcc
		if hasPost2 {
			emitRMS(scratch.expertAcc, post2Buf.buf, scratch.expertNormed, post2Buf.off)
			ffResidualSrc = scratch.expertNormed
		}
		if hasLocal {
			emitAdd(localBranch, ffResidualSrc, scratch.combined)
			ffResidualSrc = scratch.combined
		}
		if hasPostCombine {
			emitRMS(ffResidualSrc, postBuf.buf, scratch.ffResidual, postBuf.off)
			ffResidualSrc = scratch.ffResidual
		}
		emitAdd(inputBuf, ffResidualSrc, finalOutBuf)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:size])
		}
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	return out, encErr
}

// MoEBlockBF16 runs the dual-branch feed-forward of a gemma4 MoE layer on the
// post-attention residual h and returns h + ffResidual. BOTH branches run: the local
// dense MLP on rms(h, PreFFNorm), and the expert branch (router → topK experts) on
// rms(h, PreFFNorm2). Each branch output is independently normed (PostFFNorm1 /
// PostFFNorm2), summed, post-normed (PostFFNorm), then added back to the residual
// once. Mirrors pkg/metal/model/gemma4 decoder_layer.go's MoE branch op-for-op.
//
// The router operates on the RAW residual h (it applies its own internal norm); the
// experts operate on the separately-normed h2In. The router runs host top-k (see
// MoERouter) so this block is not a single command buffer; everything else is the
// parity-proven bf16 ops composed. Byte-for-byte against an independent reference
// that rebuilds both branches from primitives (TestMoEBlock). The per-layer-input
// gate, the LayerScalar, and the FFN-memory augmenter are out of scope (later
// slices / nil for standard gemma4) — this block ends at residual + ffResidual.
// NumExperts/TopK/ExpertDFF come from w; dModel/dFF/eps are the model-wide args.
func MoEBlockBF16(h []byte, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBuffer(h, nil, w, dModel, dFF, eps)
}

func MoEBlockBF16Into(out []byte, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferInto(out, h, nil, w, dModel, dFF, eps)
}

func moeBlockBF16WithBuffer(h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferPooled(h, hBuf, w, dModel, dFF, eps, true)
}

func moeBlockBF16WithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockBF16WithBufferInPool(h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferPooled(h, hBuf, w, dModel, dFF, eps, false)
}

func moeBlockBF16WithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.MoEBlockBF16: output buffer is nil")
	}
	if err := ensureInit(); err != nil {
		return err
	}
	if len(h) != dModel*bf16Size {
		return core.NewError("native.MoEBlockBF16: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK
	routerNormW := w.routerNormWeight() // zoo layers (no RouterNormWScaled) fall back to PreFFNormW

	if idx, weights, weightBuf, routerScratch, ok, err := moeRouterBF16DeviceTopKNoCopyWithBufferInPool(h, hBuf, routerNormW, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps, w.NormaliseTopK); ok || err != nil {
		if err != nil {
			return err
		}
		err = moeBlockBF16AfterRouterWithBufferOutputInPool(h, hBuf, outputBuf, idx, weights, weightBuf, w, dModel, dFF, eps)
		putRouterDeviceScratch(routerScratch)
		return err
	}
	idx, weights, err := MoERouter(h, routerNormW, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps, w.NormaliseTopK)
	if err != nil {
		return err
	}
	return moeBlockBF16AfterRouterWithBufferOutputInPool(h, hBuf, outputBuf, idx, weights, nil, w, dModel, dFF, eps)
}

func moeBlockBF16WithBufferPooled(h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool) ([]byte, error) {
	return moeBlockBF16WithBufferPooledInto(nil, h, hBuf, w, dModel, dFF, eps, useAutoreleasePool, false)
}

func moeBlockBF16WithBufferPooledInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != dModel*bf16Size {
		return nil, core.NewError("native.MoEBlockBF16: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK

	if useAutoreleasePool {
		var blockOut []byte
		var blockErr error
		withAutoreleasePool(func() {
			blockOut, blockErr = moeBlockBF16WithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, false, useCallerOut)
		})
		return blockOut, blockErr
	}

	// router decision on the raw residual (the router applies its own norm) — or, for a zoo layer
	// with no router-own norm, on routerNormWeight()'s PreFFNormW fallback (see MoELayerWeights doc).
	routerNormW := w.routerNormWeight()
	if idx, weights, weightBuf, routerScratch, ok, err := moeRouterBF16DeviceTopKNoCopyWithBufferInPool(h, hBuf, routerNormW, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps, w.NormaliseTopK); ok || err != nil {
		if err != nil {
			return nil, err
		}
		var blockOut []byte
		if useCallerOut {
			blockOut, err = moeBlockBF16AfterRouterWithBufferIntoInPool(out, h, hBuf, idx, weights, weightBuf, w, dModel, dFF, eps)
		} else {
			blockOut, err = moeBlockBF16AfterRouterWithBufferInPool(h, hBuf, idx, weights, weightBuf, w, dModel, dFF, eps)
		}
		putRouterDeviceScratch(routerScratch)
		return blockOut, err
	}
	idx, weights, err := MoERouter(h, routerNormW, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps, w.NormaliseTopK)
	if err != nil {
		return nil, err
	}
	if useCallerOut {
		return moeBlockBF16AfterRouterWithBufferIntoInPool(out, h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	}
	return moeBlockBF16AfterRouterWithBufferInPool(h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
}

// MoEQuantLayerWeights is MoELayerWeights for a 4-bit MoE layer (gemma4 26B-A4B): the local
// dense MLP, the router score projection, and the batched SwitchGLU experts are all affine-
// quantised; the five norms stay bf16. RouterNormWScaled is the router norm pre-folded by
// RootSize (as MoERouter expects); PerExpertScale is optional. Local dFF (IntermediateSize) and
// expert dFF (ExpertDFF / MoEIntermediateSize) differ, as in the bf16 block.
type MoEQuantLayerWeights struct {
	NumExperts, TopK, ExpertDFF int
	// per-component quant (mixed-precision QAT: gemma4 26B-A4B keeps the experts 4-bit but the
	// local MLP + router 8-bit). Uniform packs set all three the same.
	ExpertGroupSize, ExpertBits int
	LocalGroupSize, LocalBits   int
	RouterGroupSize, RouterBits int

	PreFFNormW, PreFFNorm2W                 []byte
	PostFFNorm1W, PostFFNorm2W, PostFFNormW []byte
	preFFNormView, preFFNorm2View           bufView
	postFFNorm1View, postFFNorm2View        bufView
	postFFNormView                          bufView

	LocalGate, LocalUp, LocalDown QuantWeight // local dense MLP (dFF)

	RouterNormWScaled  []byte
	Router             QuantWeight // [NumExperts × dModel] expert-score projection
	PerExpertScale     []byte      // [NumExperts] (nil to skip)
	routerNormView     bufView
	perExpertScaleView bufView

	ExpGate, ExpUp, ExpGateUp, ExpDown QuantWeight // batched SwitchGLU experts (ExpertDFF)
	// Shared expert (qwen3_5_moe): a single always-on dense SwiGLU (SharedGate/Up/Down) + a σ gate
	// (SharedSigmoid, [1×dModel]) added to the routed output. A bound SharedGate MARKS a qwen MoE layer,
	// which decodes on the host (encQwenMoEHalf) rather than the gemma device MoE (#18).
	SharedGate, SharedUp, SharedDown, SharedSigmoid QuantWeight
	// SharedDFF (#61) is the shared expert's OWN intermediate size — moeToQuant resolves it from
	// arch.SharedExpertFF with an arch.ExpertFF fallback (load_shared.go), mirroring
	// model.assembleMoE's SharedDown InDim resolution (#57). encQwenMoEHalf sizes its shared-expert
	// MoEExpertsQuantSiLU dispatch from THIS field, not ExpertDFF (the ROUTED width) — a checkpoint
	// whose shared and routed widths genuinely differ (real llama4 Scout: 16384 shared vs 8192
	// routed; Qwen1.5-MoE-A2.7B: 5632 vs 1408) previously sized the shared dispatch off the wrong
	// width, which MoEExpertsQuantSiLU's own packed-length check turns into a hard decode-time error
	// rather than silently wrong output. Zero (an unpopulated field, e.g. a pre-#61 hand-built test
	// fixture) falls back to ExpertDFF at the encQwenMoEHalf call site — see its "sharedFF" doc.
	SharedDFF int

	// gpt_oss (#37): ClampedSwiGLU MARKS a gpt_oss MoE layer — clamped-sigmoid SwiGLU experts
	// (MoEExpertsQuantClampedSiLU at SwigluLimit) + an additive router bias, no local dense MLP, no
	// router norm, no sandwich norms — which decodes on the host (encGptOssMoEHalf) rather than the
	// gemma device MoE or the qwen half. RouterBias (bf16 [NumExperts], mlx-lm gpt_oss.py: router =
	// nn.Linear(..., bias=True)) is added to the router logits BEFORE top-k. ExpGateBias/ExpUpBias
	// (bf16 [NumExperts×ExpertDFF]) and ExpDownBias (bf16 [NumExperts×dModel]) are the per-expert
	// additive projection biases (SwitchGLU(..., bias=True)), added after each expert matvec. All
	// zero/nil on every other arch — the gemma/qwen paths never read them.
	ClampedSwiGLU                       bool
	SwigluLimit                         float32
	RouterBias                          []byte
	ExpGateBias, ExpUpBias, ExpDownBias []byte

	// UsesSiLU selects the expert-combine gate nonlinearity: false (the zero value) is gemma4's GELU
	// (byte-identical to every load before #63); true is SiLU/SwiGLU (encSiLUGateMulBF16 —
	// mixtral/dbrx/olmoe's llama-family shape). Set from arch.Activation via ffnUsesSiLU at load
	// (moeToQuant, load_shared.go) — see encMoEBlockQuantDevice and
	// moeBlockQuantAfterRouterWithDeviceIndexBufferPooled's emitGelu/emitGate closures, where this flag
	// is consulted. gemma4 never sets Arch.Activation, so this is always false for it. A qwen-shaped
	// layer (SharedGate bound) never reaches either function — it decodes on the host (encQwenMoEHalf,
	// already SiLU-correct) before either combine sees it — so this field is moot but still correctly
	// resolved for one.
	UsesSiLU bool

	// NormaliseTopK is MoELayerWeights.NormaliseTopK's quant twin — the arch's declared router policy
	// (model.Arch.NormaliseMoETopK, #65), set from arch.NormaliseMoETopK at load (moeToQuant,
	// load_shared.go). gemma4 always declares it true, so this is byte-unchanged for every existing
	// gemma4 load.
	NormaliseTopK bool
}

// routerNorm is MoELayerWeights.routerNormWeight() for the quant struct: the norm weight (bytes
// AND its pre-resolved view, paired) the quant router applies internally before the score
// projection. gemma4 declares a dedicated router norm (RouterNormWScaled); a llama-family zoo layer
// (mixtral/dbrx/olmoe/qwenmoe — #59) has none, so it falls back to PreFFNormW/preFFNormView — the
// SAME value the expert branch normalises to (see moeBlockQuantAfterRouterWithDeviceIndexBufferPooled's
// pre2W fallback). gemma4 always sets RouterNormWScaled, so this fallback never engages for it.
func (w MoEQuantLayerWeights) routerNorm() ([]byte, bufView) {
	if len(w.RouterNormWScaled) != 0 {
		return w.RouterNormWScaled, w.routerNormView
	}
	return w.PreFFNormW, w.preFFNormView
}

type mlpTransformScratch struct {
	dModel, dFF int
	x           *pinnedNoCopyBytes
	mlp         mlpScratch
	inViewPtr   uintptr
	inViewLen   int
	inView      metal.MTLBuffer
	inPinned    *pinnedNoCopyBytes
	outViewPtr  uintptr
	outViewLen  int
	outView     metal.MTLBuffer
	outPinned   *pinnedNoCopyBytes
}

type mlpTransformScratchKey struct {
	dModel, dFF int
}

var mlpTransformScratchPools sync.Map

func newMLPTransformScratch(dModel, dFF int) (*mlpTransformScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	return &mlpTransformScratch{
		dModel: dModel,
		dFF:    dFF,
		x:      x,
		mlp:    newMLPScratch(dModel, dFF),
	}, nil
}

func mlpTransformScratchPoolFor(dModel, dFF int) *scratchLIFOPool[*mlpTransformScratch] {
	key := mlpTransformScratchKey{dModel: dModel, dFF: dFF}
	if v, ok := mlpTransformScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*mlpTransformScratch])
	}
	pool := &scratchLIFOPool[*mlpTransformScratch]{}
	if v, loaded := mlpTransformScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*mlpTransformScratch])
	}
	return pool
}

func getMLPTransformScratch(dModel, dFF int) (*mlpTransformScratch, error) {
	pool := mlpTransformScratchPoolFor(dModel, dFF)
	if s := pool.Get(); s != nil {
		if s != nil &&
			s.dModel == dModel &&
			s.dFF == dFF &&
			s.x != nil &&
			s.x.buf != nil &&
			s.mlp.gate != nil &&
			s.mlp.up != nil &&
			s.mlp.gated != nil &&
			s.mlp.down != nil {
			return s, nil
		}
		s.Close()
	}
	return newMLPTransformScratch(dModel, dFF)
}

func putMLPTransformScratch(s *mlpTransformScratch) {
	if s != nil && s.x != nil && s.x.buf != nil && s.mlp.gate != nil && s.mlp.up != nil && s.mlp.gated != nil && s.mlp.down != nil {
		mlpTransformScratchPoolFor(s.dModel, s.dFF).Put(s)
	}
}

func (s *mlpTransformScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	s.closeInputView()
	s.closeOutputView()
	s.dModel, s.dFF = 0, 0
}

func (s *mlpTransformScratch) closeInputView() {
	if s == nil {
		return
	}
	if s.inPinned != nil {
		s.inPinned.Close()
	}
	s.inViewPtr = 0
	s.inViewLen = 0
	s.inView = nil
	s.inPinned = nil
}

func (s *mlpTransformScratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outPinned != nil {
		s.outPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outPinned = nil
}

func (s *mlpTransformScratch) inputView(x []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(x) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&x[0]))
	if s.inView != nil && s.inViewPtr == ptr && s.inViewLen == len(x) {
		return s.inView, true
	}
	s.closeInputView()
	if buf, ok := registeredPinnedNoCopyBytes(x); ok {
		s.inViewPtr = ptr
		s.inViewLen = len(x)
		s.inView = buf
		s.inPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(x)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: x, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.inViewPtr = ptr
	s.inViewLen = len(x)
	s.inView = buf
	s.inPinned = pinned
	return buf, true
}

func (s *mlpTransformScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outPinned = pinned
	return buf, true
}

type mlpTransformMegaScratch struct {
	dModel, dFF        int
	x                  *pinnedNoCopyBytes
	gated, out, arrive metal.MTLBuffer
	outBytes           []byte
	arrivePtr          *uint32
	inViewPtr          uintptr
	inViewLen          int
	inView             metal.MTLBuffer
	inPinned           *pinnedNoCopyBytes
	outViewPtr         uintptr
	outViewLen         int
	outView            metal.MTLBuffer
	outPinned          *pinnedNoCopyBytes
}

var mlpTransformMegaScratchPools sync.Map

func newMLPTransformMegaScratch(dModel, dFF int) (*mlpTransformMegaScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	gated := device.NewBufferWithLengthOptions(uint(dFF*4), metal.MTLResourceStorageModeShared)
	out := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	return &mlpTransformMegaScratch{
		dModel:    dModel,
		dFF:       dFF,
		x:         x,
		gated:     gated,
		out:       out,
		arrive:    arrive,
		outBytes:  unsafe.Slice((*byte)(out.Contents()), dModel*bf16Size),
		arrivePtr: (*uint32)(arrive.Contents()),
	}, nil
}

func mlpTransformMegaScratchPoolFor(dModel, dFF int) *scratchLIFOPool[*mlpTransformMegaScratch] {
	key := mlpTransformScratchKey{dModel: dModel, dFF: dFF}
	if v, ok := mlpTransformMegaScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*mlpTransformMegaScratch])
	}
	pool := &scratchLIFOPool[*mlpTransformMegaScratch]{}
	if v, loaded := mlpTransformMegaScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*mlpTransformMegaScratch])
	}
	return pool
}

func getMLPTransformMegaScratch(dModel, dFF int) (*mlpTransformMegaScratch, error) {
	pool := mlpTransformMegaScratchPoolFor(dModel, dFF)
	if s := pool.Get(); s != nil {
		if s != nil && s.dModel == dModel && s.dFF == dFF && s.x != nil && s.x.buf != nil && s.gated != nil && s.out != nil && s.arrive != nil && len(s.outBytes) == dModel*bf16Size && s.arrivePtr != nil {
			return s, nil
		}
		s.Close()
	}
	return newMLPTransformMegaScratch(dModel, dFF)
}

func putMLPTransformMegaScratch(s *mlpTransformMegaScratch) {
	if s != nil && s.x != nil && s.x.buf != nil && s.gated != nil && s.out != nil && s.arrive != nil && len(s.outBytes) == s.dModel*bf16Size && s.arrivePtr != nil {
		mlpTransformMegaScratchPoolFor(s.dModel, s.dFF).Put(s)
	}
}

func (s *mlpTransformMegaScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	s.gated = nil
	s.out = nil
	s.arrive = nil
	s.outBytes = nil
	s.arrivePtr = nil
	s.closeInputView()
	s.closeOutputView()
	s.dModel, s.dFF = 0, 0
}

func (s *mlpTransformMegaScratch) closeInputView() {
	if s == nil {
		return
	}
	if s.inPinned != nil {
		s.inPinned.Close()
	}
	s.inViewPtr = 0
	s.inViewLen = 0
	s.inView = nil
	s.inPinned = nil
}

func (s *mlpTransformMegaScratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outPinned != nil {
		s.outPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outPinned = nil
}

func (s *mlpTransformMegaScratch) inputView(x []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(x) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&x[0]))
	if s.inView != nil && s.inViewPtr == ptr && s.inViewLen == len(x) {
		return s.inView, true
	}
	s.closeInputView()
	if buf, ok := registeredPinnedNoCopyBytes(x); ok {
		s.inViewPtr = ptr
		s.inViewLen = len(x)
		s.inView = buf
		s.inPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(x)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: x, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.inViewPtr = ptr
	s.inViewLen = len(x)
	s.inView = buf
	s.inPinned = pinned
	return buf, true
}

func (s *mlpTransformMegaScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outPinned = pinned
	return buf, true
}

type quantMLPProjView struct {
	packed, scales, biases bufView
	groupSize, bits        int
}

// ffnMegaDefaultGeometry admits the FFN megakernel only on the shape family it
// earned its receipt on: dense-style MLPs with dFF well above dModel (E2B
// 1536×6144, E4B 2560×10240, 12B — all ratio 4). The grid-barrier design
// parallelises over dFF rows, so an INVERTED shape starves it: the 26B-A4B's
// 4-bit local expert (dModel 2816 × dFF 2112, first exposed by the non-QAT
// checkpoint — the QAT local is 8-bit and routed away) measured 12.43 ms/token
// on the mega vs ~1 ms on the qmv trio, a 9× family blowup that took the whole
// decode from 137 to 42 tok/s. Route a new shape only with a receipt.
func ffnMegaDefaultGeometry(dModel, dFF int) bool {
	return dModel >= 256 && dFF >= 512 && dFF >= 2*dModel
}

var (
	moeArriveZeroOnce sync.Once
	moeArriveZeroBuf  metal.MTLBuffer
)

// moeArriveZeroBuffer returns a shared 4-byte zero buffer — the copy source that resets the FFN
// megakernel's grid-barrier arrive counter INSIDE the encoder (no host write, no wait).
func moeArriveZeroBuffer() metal.MTLBuffer {
	moeArriveZeroOnce.Do(func() {
		moeArriveZeroBuf = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	})
	return moeArriveZeroBuf
}

// moeConcurrentBlocks counts concurrent-pass MoE block encodes — the engagement receipt for
// the lane (a silent gate regression reads as zero, not as a perf blur).
var moeConcurrentBlocks atomic.Int64

// geluFoldDispatches counts MoE blocks that ran the gelu-fused down kernels
// (#341 phase 1) instead of the gelu-dispatch + plain-down chain. Engagement
// counter for the A/B tests.
var geluFoldDispatches atomic.Int64

type lthnGeluQMVKey struct {
	groupSize, bits int
}

var (
	lthnGeluQMVPSOMu    sync.Mutex
	lthnGeluQMVPSOCache = map[lthnGeluQMVKey]metal.MTLComputePipelineState{}
)

// lthnGeluQMVPipeline resolves (and caches, including failures) the local-down
// variant with the MLP gate fused into its x-load (lthn_gelu_qmv, #341 phase
// 1). A miss — custom library absent, or a gs/bits pair outside the
// instantiated set — caches nil so the block falls back to the gelu-dispatch +
// plain-qmv chain without re-probing.
func lthnGeluQMVPipeline(groupSize, bits int) (metal.MTLComputePipelineState, bool) {
	key := lthnGeluQMVKey{groupSize: groupSize, bits: bits}
	lthnGeluQMVPSOMu.Lock()
	defer lthnGeluQMVPSOMu.Unlock()
	if pso, ok := lthnGeluQMVPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		lthnGeluQMVPSOCache[key] = nil
		return nil, false
	}
	fn := customLibrary.NewFunctionWithName(core.Sprintf("lthn_gelu_qmv_bfloat16_t_gs_%d_b_%d", groupSize, bits))
	if fn == nil || fn.GetID() == 0 {
		lthnGeluQMVPSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		lthnGeluQMVPSOCache[key] = nil
		return nil, false
	}
	lthnGeluQMVPSOCache[key] = pso
	return pso, true
}

// ffnMegaKernelCompatible is the KERNEL truth: the widths the megakernel is specialised for
// (4/8-bit byte-aligned codes via ffnMegaPipelineBits, parity-proven at both) with all three
// projections agreeing so one PSO serves the dispatch.
func ffnMegaKernelCompatible(gate, up, down quantMLPProjView, dModel, dFF int) bool {
	return (gate.bits == 4 || gate.bits == 8) && up.bits == gate.bits && down.bits == gate.bits &&
		gate.groupSize == up.groupSize && gate.groupSize == down.groupSize &&
		gate.groupSize > 0 && dModel%gate.groupSize == 0 && dFF%gate.groupSize == 0
}

// ffnMegaSupported is the ROUTING policy: 8-bit stays unrouted by receipt — on the 26B-A4B's
// b8 local MLP the mega measured 26.9 tok/s vs the qmv trio's 51.9 on the fully-encoded decode
// (#338); its per-byte scalar gemv loses to the steel qmv far more than the 4->1 dispatch
// saving returns. Route a new width only with a geometry receipt that says otherwise.
func ffnMegaSupported(gate, up, down quantMLPProjView, dModel, dFF int) bool {
	return gate.bits == 4 && ffnMegaKernelCompatible(gate, up, down, dModel, dFF)
}

func emitFFNMega[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x metal.MTLBuffer, xOff uint, gate, up, down quantMLPProjView, gated, out metal.MTLBuffer, outOff uint, arrive metal.MTLBuffer, dModel, dFF int) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(gate.packed.buf, gate.packed.off, 1)
	sink.setBuf(gate.scales.buf, gate.scales.off, 2)
	sink.setBuf(gate.biases.buf, gate.biases.off, 3)
	sink.setBuf(up.packed.buf, up.packed.off, 4)
	sink.setBuf(up.scales.buf, up.scales.off, 5)
	sink.setBuf(up.biases.buf, up.biases.off, 6)
	sink.setBuf(down.packed.buf, down.packed.off, 7)
	sink.setBuf(down.scales.buf, down.scales.off, 8)
	sink.setBuf(down.biases.buf, down.biases.off, 9)
	sink.setBuf(gated, 0, 10)
	sink.setBuf(out, outOff, 11)
	sink.setBuf(arrive, 0, 12)
	sink.setI32(int32(dModel), 13)
	sink.setI32(int32(dFF), 14)
	sink.setI32(int32(gate.groupSize), 15)
	sink.setI32(ffnMegaNumThreadgroups, 16)
	sink.setI32(ffnMegaMaxSpinIterations, 17)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: ffnMegaNumThreadgroups, Height: 1, Depth: 1},
		metal.MTLSize{Width: ffnMegaThreadsPerGroup, Height: 1, Depth: 1},
	)
}

// encMoEBlockQuantDevice encodes the WHOLE MoE block — device router top-K, the local MLP, the
// gathered expert MLPs and the norm/combine tail — into the CALLER's live encoder: zero command
// buffers, zero waits, zero host bytes. This is the fully-encoded decode lane (26B-A4B): the
// break-out flow it replaces cost ~3 command buffers per layer per token. handled=false declines
// (non-gather geometry, missing kernels, mega-local, host-only shapes) to the proven break-out
// path, byte-identically. The caller owns BOTH scratches' single-flight lifetime (session-owned;
// recycling across layers/tokens is GPU-GPU in commit order) and pre-validated the router
// geometry (routerTopKUsable + quantMoEDeviceRouterBuffersUsable).
//
// The device-relevant prelude is deliberately a sibling of moeBlockQuantAfterRouterWith
// DeviceIndexBufferPooled's — keep the two in step when weight layouts change.
// encConc marks enc as an OPEN CONCURRENT encoder carried from the previous
// pass (#341 phase 1.5): the concurrent fork joins it behind one buffer barrier
// instead of paying an encoder seam, and returns encConc=true itself when it
// leaves its encoder open for the next pass. Declines return enc untouched
// with the caller's encConc passed through; the serial path always normalises
// to a tracked serial encoder first (a carried concurrent encoder has no
// hazard tracking).
func encMoEBlockQuantDevice(enc metal.MTLComputeCommandEncoderObject, cb metal.MTLCommandBufferObject, prof *gpuCounterProfiler, encConc bool, routerScratch *routerDeviceScratch, scratch *moeBlockBF16Scratch, hBuf, outputBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) (metal.MTLComputeCommandEncoderObject, bool, bool, error) {
	expertDFF, numExperts, topK := w.ExpertDFF, w.NumExperts, w.TopK
	size := dModel * bf16Size
	if hBuf == nil || outputBuf == nil || routerScratch == nil || scratch == nil || topK <= 0 || dModel == 0 || dFF == 0 || expertDFF == 0 {
		return enc, encConc, false, nil
	}
	if len(w.PreFFNormW) != size || len(w.PreFFNorm2W) != size || len(w.PostFFNorm1W) != size || len(w.PostFFNorm2W) != size || len(w.PostFFNormW) != size {
		return enc, encConc, false, nil
	}
	localGatePacked, localGateScales, localGateBiases, localGateGroupSize, localGateBits, err := quantWeightViewsForShape("native.encMoEBlockQuantDevice: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return enc, encConc, false, nil
	}
	localUpPacked, localUpScales, localUpBiases, localUpGroupSize, localUpBits, err := quantWeightViewsForShape("native.encMoEBlockQuantDevice: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return enc, encConc, false, nil
	}
	localDownPacked, localDownScales, localDownBiases, localDownGroupSize, localDownBits, err := quantWeightViewsForShape("native.encMoEBlockQuantDevice: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return enc, encConc, false, nil
	}
	// (the break-out path may run the local MLP through the megakernel; this lane always
	// encodes the qmv trio instead — the megakernel's arrive counter is a host write, and
	// three chained qmvs inside the live encoder still beat a broken-out mega dispatch)
	fusedExperts := len(w.ExpGateUp.Packed) > 0
	var expGateUpPacked, expGateUpScales, expGateUpBiases bufView
	var expGatePacked, expGateScales, expGateBiases bufView
	var expUpPacked, expUpScales, expUpBiases bufView
	var expDownPacked, expDownScales, expDownBiases bufView
	var expGateGroupSize, expGateBits, expUpGroupSize, expUpBits, expGateUpGroupSize, expGateUpBits, expDownGroupSize, expDownBits int
	if fusedExperts {
		expGateUpPacked, expGateUpScales, expGateUpBiases, expGateUpGroupSize, expGateUpBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return enc, encConc, false, nil
		}
	} else {
		expGatePacked, expGateScales, expGateBiases, expGateGroupSize, expGateBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return enc, encConc, false, nil
		}
		expUpPacked, expUpScales, expUpBiases, expUpGroupSize, expUpBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return enc, encConc, false, nil
		}
	}
	expDownPacked, expDownScales, expDownBiases, expDownGroupSize, expDownBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if err != nil {
		return enc, encConc, false, nil
	}
	if !affineBitsSupported(expDownBits) {
		return enc, encConc, false, nil
	}
	inGroup, inBits, inRows := expGateGroupSize, expGateBits, expertDFF
	if fusedExperts {
		if !affineBitsSupported(expGateUpBits) {
			return enc, encConc, false, nil
		}
		inGroup, inBits, inRows = expGateUpGroupSize, expGateUpBits, 2*expertDFF
	} else if !affineBitsSupported(expGateBits) || expGateBits != expUpBits || expGateGroupSize != expUpGroupSize {
		// gate + up share one gather PSO — width and group size must agree between them.
		return enc, encConc, false, nil
	}
	gatherExpertInPSO, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, inGroup, inBits)
	if err != nil {
		return enc, encConc, false, nil
	}
	gatherExpertDownPSO, err := gatherQMVBF16SteelPipeline(dModel, expertDFF, expDownGroupSize, expDownBits)
	if err != nil {
		return enc, encConc, false, nil
	}
	gatherExpertInMeta, err := gatherQMVBF16Metadata(numExperts, expertDFF, dModel, inGroup, inBits, inRows)
	if err != nil {
		return enc, encConc, false, nil
	}
	gatherExpertDownMeta, err := gatherQMVBF16Metadata(numExperts, dModel, expertDFF, expDownGroupSize, expDownBits, dModel)
	if err != nil {
		return enc, encConc, false, nil
	}
	localGatePSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, localGateGroupSize, localGateBits))
	if err != nil {
		return enc, encConc, false, nil
	}
	localUpPSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, localUpGroupSize, localUpBits))
	if err != nil {
		return enc, encConc, false, nil
	}
	localDownPSO, err := pipelineFor(qmvBF16KernelName(dModel, dFF, localDownGroupSize, localDownBits))
	if err != nil {
		return enc, encConc, false, nil
	}
	// #63: the layer's declared expert-combine activation — see moeBlockBF16AfterRouterWithBufferPooled's
	// identical useSiLU doc. gemma4 (the only arch shaped to reach this fully-encoded lane today, since
	// eligibility above requires the local dense MLP + all five sandwich norms) never sets
	// Arch.Activation, so this stays false for it — byte-identical.
	useSiLU := w.UsesSiLU
	// gelu fold (#341 phase 1): both down projections read gate/up directly and
	// compute gelu(gate)·up at load — the two gelu dispatches (and the expert
	// gelu's barrier) never encode. Decided ONCE for the whole block so the
	// gated scratch stays coherent; either PSO missing (stale metallib, exotic
	// width) or the lever falls back to the chain unchanged. The fused kernel is
	// GELU-specific (lthn_gelu_qmv/lthn_gather_qmv's gelu:true variant) — a SiLU layer
	// always falls back to the plain (already SiLU-aware) chain below, never the fold.
	geluFold := false
	var localDownGeluPSO, gatherDownGeluPSO metal.MTLComputePipelineState
	if geluFoldEnabled && !useSiLU {
		if p1, ok1 := lthnGeluQMVPipeline(localDownGroupSize, localDownBits); ok1 {
			if p2, ok2 := lthnGatherQMVPipeline(lthnGatherQMVKey{groupSize: expDownGroupSize, bits: expDownBits, expertRows: dModel, batchedX: true, gelu: true}); ok2 {
				localDownGeluPSO, gatherDownGeluPSO, geluFold = p1, p2, true
			}
		}
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return enc, encConc, false, nil
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return enc, encConc, false, nil
	}
	if !gpuHasGeluKernel() {
		return enc, encConc, false, nil
	}
	geluPSO, err := geluPipeline()
	if err != nil {
		return enc, encConc, false, nil
	}
	scalePSO, err := bf16MulScalarPipeline()
	if err != nil {
		return enc, encConc, false, nil
	}
	pre1Buf := bf16WeightView(w.PreFFNormW, w.preFFNormView)
	pre2Buf := bf16WeightView(w.PreFFNorm2W, w.preFFNorm2View)
	post1Buf := bf16WeightView(w.PostFFNorm1W, w.postFFNorm1View)
	post2Buf := bf16WeightView(w.PostFFNorm2W, w.postFFNorm2View)
	postBuf := bf16WeightView(w.PostFFNormW, w.postFFNormView)

	// the device router's plan — idx/weights land in routerScratch on device; the emits are
	// staged below (serial: back to back with the profiler seams; concurrent: interleaved
	// with the local/expert stages between dependency barriers).
	routerPlan, rerr := buildRouterEncodePlan(routerScratch, hBuf, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
	if rerr != nil {
		return enc, encConc, false, nil
	}
	routeIdxBuf, weightsBuf := routerScratch.idxBuf, routerScratch.weightBuf
	msc := scratch.mlp
	sink := encSink{enc}
	// seam splits the live encoder at a stage boundary when the GPU profiler is armed (tests
	// only — prof nil in production): enc AND sink are rebuilt so every emit closure below
	// follows the new encoder.
	seam := func(label string) {
		if prof == nil {
			return
		}
		endEncodingFast(enc)
		enc = prof.encoderFor(cb, label)
		sink = encSink{enc}
	}

	// stage 2: the block body — the break-out run()'s device subset, same kernels, same order.
	emitRMS := func(x, weight, out metal.MTLBuffer, wOff uint) {
		emitRMSNorm(sink, rmsPSO, x, weight, out, wOff, dModel, eps, rmsTG)
	}
	emitQ := func(pso metal.MTLComputePipelineState, wq, scales, biases bufView, x, out metal.MTLBuffer, inDim, outDim int) {
		emitQMV(sink, pso, wq.buf, wq.off, scales.buf, scales.off, biases.buf, biases.off, x, out, 0, inDim, outDim)
	}
	emitGatherQ := func(pso metal.MTLComputePipelineState, meta *gatherQMVBF16Meta, wq, scales, biases bufView, x, out metal.MTLBuffer, route, inDim, outDim, groupSize, bits, rowBase int) {
		emitGatherQMVBF16Steel(sink, pso, meta, x, wq.buf, wq.off, scales.buf, scales.off, biases.buf, biases.off, routeIdxBuf, uint(route*4), out, 0, outDim, inDim, groupSize, bits, rowBase)
	}
	emitScaleAt := func(in metal.MTLBuffer, scalarOff uint, out metal.MTLBuffer, n int) {
		sink.setPSO(scalePSO)
		sink.setBuf(in, 0, 0)
		sink.setBuf(weightsBuf, scalarOff, 1)
		sink.setBuf(out, 0, 2)
		sink.setI32(int32(n), 3)
		group := min(uint(n), uint(256))
		sink.dispatchThreads(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
	}
	emitScaleFromAt := func(in metal.MTLBuffer, inOff, scalarOff uint, out metal.MTLBuffer, n int) {
		sink.setPSO(scalePSO)
		sink.setBuf(in, inOff, 0)
		sink.setBuf(weightsBuf, scalarOff, 1)
		sink.setBuf(out, 0, 2)
		sink.setI32(int32(n), 3)
		group := min(uint(n), uint(256))
		sink.dispatchThreads(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
	}
	emitAdd := func(a, b, out metal.MTLBuffer) {
		emitBinary(sink, addPSO, a, 0, b, 0, out, 0, dModel)
	}
	// emitGate is the layer's whole gate-activation dispatch (#63): useSiLU routes to the composed
	// SwiGLU chain (no fused SiLU kernel exists, mirroring the bf16/quant break-out funnels' identical
	// choice); gemma4 (useSiLU false) keeps the original one-dispatch fused gelu kernel this lane
	// requires (gpuHasGeluKernel is asserted above). encSiLUGateMulBF16's own pipelines
	// (vv_Multiplybfloat16, v_SigmoidBFloat16BFloat16) are base MLX kernels already relied on
	// unchecked elsewhere in this lane (addPSO); its error is not expected to fire in practice — same
	// tolerance as every other post-setup emit closure here — so the caller need not thread it through.
	emitGate := func(gate, up, out metal.MTLBuffer, n int) {
		if useSiLU {
			_ = encSiLUGateMulBF16(enc, gate, up, out, n)
			return
		}
		emitBinary(sink, geluPSO, gate, 0, up, 0, out, 0, n)
	}
	// ---- stage decisions, hoisted above the emits so the concurrent fork can gate ----
	localGateViewQ := quantMLPProjView{packed: localGatePacked, scales: localGateScales, biases: localGateBiases, groupSize: localGateGroupSize, bits: localGateBits}
	localUpViewQ := quantMLPProjView{packed: localUpPacked, scales: localUpScales, biases: localUpBiases, groupSize: localUpGroupSize, bits: localUpBits}
	localDownViewQ := quantMLPProjView{packed: localDownPacked, scales: localDownScales, biases: localDownBiases, groupSize: localDownGroupSize, bits: localDownBits}
	megaLocal := ffnMegaDefaultGeometry(dModel, dFF) && ffnMegaSupported(localGateViewQ, localUpViewQ, localDownViewQ, dModel, dFF)
	var localMegaPSO metal.MTLComputePipelineState
	if megaLocal {
		if localMegaPSO, err = ffnMegaPipelineBits(localGateViewQ.bits); err != nil {
			megaLocal = false
		}
	}
	if megaLocal {
		if err = scratch.ensureLocalMegaScratch(); err != nil {
			megaLocal = false
		}
	}
	// expert MLPs, ALL routes per dispatch: the MLX gather batch dimension carries every
	// selected expert (grid.z = topK, rhs = the router's device idxBuf), collapsing the
	// old per-route loop's topK×3 matvec dispatches to 3. gate/up share the one normed
	// input (lhs zeros, stride 0); down walks each route's gated row (lhs iota, stride 1).
	// Falls back to the per-route loop when the slabs or metadata are unavailable.
	allRoutes := scratch.ensureAllRoutesScratch() == nil
	var inKeyShared, downKeyBatched gatherQMVAllRoutesMetaKey
	var inAllMeta, downAllMeta *gatherQMVBF16Meta
	if allRoutes {
		inGroupSize, inBitsSel, inRowsSel := expGateGroupSize, expGateBits, expertDFF
		if fusedExperts {
			inGroupSize, inBitsSel, inRowsSel = expGateUpGroupSize, expGateUpBits, 2*expertDFF
		}
		inKeyShared = gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: expertDFF, inDim: dModel, groupSize: inGroupSize, bits: inBitsSel, expertRows: inRowsSel, routes: topK, xRows: 1, batchedX: false}
		downKeyBatched = gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: dModel, inDim: expertDFF, groupSize: expDownGroupSize, bits: expDownBits, expertRows: dModel, routes: topK, xRows: topK, batchedX: true}
		if inAllMeta, err = gatherQMVAllRoutesMetadata(numExperts, expertDFF, dModel, inGroupSize, inBitsSel, inRowsSel, topK, 1, false); err != nil {
			allRoutes = false
		} else if downAllMeta, err = gatherQMVAllRoutesMetadata(numExperts, dModel, expertDFF, expDownGroupSize, expDownBits, dModel, topK, topK, true); err != nil {
			allRoutes = false
		}
	}
	wsumPSO, wsumErr := moeWeightedSumPipeline()
	// The norm/combine tail — out = h + rms(rms(local)·w1 + rms(expert)·w2)·w3 — as ONE
	// fused dispatch (byte-identical rounding to the chain; single-row rms only), with the
	// five-kernel chain as the fallback for unavailable PSO / looped-rms widths.
	var combinePSO metal.MTLComputePipelineState
	combineTG, combineOK := uint(0), false
	if dModel <= rmsLoopedLimit {
		if fusedPSO, ferr := moeCombineNormsPipeline(); ferr == nil {
			tg := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
			if tg <= fusedPSO.MaxTotalThreadsPerThreadgroup() {
				combinePSO, combineTG, combineOK = fusedPSO, tg, true
			}
		}
	}
	emitWSum := func() {
		sink.setPSO(wsumPSO)
		sink.setBuf(scratch.expertDownAll, 0, 0)
		sink.setBuf(weightsBuf, 0, 1)
		sink.setBuf(scratch.expertAcc, 0, 2)
		sink.setI32(int32(dModel), 3)
		sink.setI32(int32(topK), 4)
		group := min(uint(dModel), uint(256))
		sink.dispatchThreads(
			metal.MTLSize{Width: uint(dModel), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
	}
	emitCombineNorms := func() {
		sink.setPSO(combinePSO)
		sink.setBuf(scratch.localOut, 0, 0)
		sink.setBuf(post1Buf.buf, post1Buf.off, 1)
		sink.setBuf(scratch.expertAcc, 0, 2)
		sink.setBuf(post2Buf.buf, post2Buf.off, 3)
		sink.setBuf(postBuf.buf, postBuf.off, 4)
		sink.setBuf(hBuf, 0, 5)
		sink.setBuf(outputBuf, 0, 6)
		sink.setF32(eps, 7)
		sink.setI32(int32(dModel), 8)
		sink.dispatchThreads(
			metal.MTLSize{Width: combineTG, Height: 1, Depth: 1},
			metal.MTLSize{Width: combineTG, Height: 1, Depth: 1},
		)
	}
	emitGatherInAll := func() {
		if fusedExperts {
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertGateAll, 0, expertDFF, dModel, expGateUpGroupSize, expGateUpBits, 0, topK)
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertUpAll, 0, expertDFF, dModel, expGateUpGroupSize, expGateUpBits, expertDFF, topK)
		} else {
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expGatePacked.buf, expGatePacked.off, expGateScales.buf, expGateScales.off, expGateBiases.buf, expGateBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertGateAll, 0, expertDFF, dModel, expGateGroupSize, expGateBits, 0, topK)
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expUpPacked.buf, expUpPacked.off, expUpScales.buf, expUpScales.off, expUpBiases.buf, expUpBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertUpAll, 0, expertDFF, dModel, expUpGroupSize, expUpBits, 0, topK)
		}
	}
	emitGatherDownAll := func() {
		if geluFold {
			// the fused down reads each route's gate/up rows and gelus at load —
			// the expert gelu dispatch never encoded.
			emitLthnGatherQMVGeluRoutes(sink, gatherDownGeluPSO, scratch.expertGateAll, 0, scratch.expertUpAll, 0, expDownPacked.buf, expDownPacked.off, expDownScales.buf, expDownScales.off, expDownBiases.buf, expDownBiases.off, scratch.routeIota, routeIdxBuf, 0, scratch.expertDownAll, 0, dModel, expertDFF, expDownGroupSize, expDownBits, 0, topK)
			return
		}
		emitGatherQMVAllRoutes(sink, gatherExpertDownPSO, downAllMeta, downKeyBatched, scratch.expertGatedAll, 0, expDownPacked.buf, expDownPacked.off, expDownScales.buf, expDownScales.off, expDownBiases.buf, expDownBiases.off, scratch.routeIota, routeIdxBuf, 0, scratch.expertDownAll, 0, dModel, expertDFF, expDownGroupSize, expDownBits, 0, topK)
	}
	emitLocalDown := func() {
		if geluFold {
			geluFoldDispatches.Add(1)
			emitGeluQMV(sink, localDownGeluPSO, localDownPacked.buf, localDownPacked.off, localDownScales.buf, localDownScales.off, localDownBiases.buf, localDownBiases.off, msc.gate, msc.up, scratch.localOut, 0, dFF, dModel)
			return
		}
		emitQ(localDownPSO, localDownPacked, localDownScales, localDownBiases, msc.gated, scratch.localOut, dFF, dModel)
	}

	// ---- concurrent pass: dispatches overlap; barriers mark the true dependency edges.
	// The router, local-MLP, and expert branches are independent until the combine, so the
	// serial path's 13 dependent dispatch gaps collapse to 7 barrier stages with 3-way
	// overlap at the top. Values are unchanged — every kernel and rounding point is the
	// serial path's; only the schedule differs.
	if prof == nil && !moeConcurrentDisabled && !megaLocal && allRoutes && wsumErr == nil && combineOK {
		moeConcurrentBlocks.Add(1)
		if encConc && !encCarryDisabled {
			// Carry the attention pass's open concurrent encoder: one buffer barrier
			// orders its writes ahead of this block's reads, no encoder seam paid.
			concEncoderCarries.Add(1)
			memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			sink = encSink{enc}
		} else {
			endEncodingFast(enc)
			enc = concurrentComputeEncoderFast(cb)
			sink = encSink{enc}
		}
		barrier := func() { memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers) }
		// stage 1: router ∥ the two local input norms (all read hBuf only). The fused
		// single-dispatch router runs its whole rms→qmv→topk here, overlapping the
		// local gate/up/gelu stages instead of interleaving with them; its indices are
		// ready three barriers before the stage-4 gathers either way.
		routerFused := routerPlan.emitFused(sink)
		if !routerFused {
			routerPlan.emitRMS(sink)
		}
		emitRMS(hBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
		emitRMS(hBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
		barrier()
		// stage 2: router scores ∥ local gate ∥ local up
		if !routerFused {
			routerPlan.emitQMV(sink)
		}
		emitQ(localGatePSO, localGatePacked, localGateScales, localGateBiases, scratch.localIn, msc.gate, dModel, dFF)
		emitQ(localUpPSO, localUpPacked, localUpScales, localUpBiases, scratch.localIn, msc.up, dModel, dFF)
		barrier()
		// stage 3: top-k select ∥ local gelu (folded into the local down when the
		// fused kernels are available — the dispatch never encodes)
		if !routerFused {
			routerPlan.emitTopK(sink)
		}
		if !geluFold {
			emitGate(msc.gate, msc.up, msc.gated, dFF)
		}
		barrier()
		// stage 4: expert gate/up gathers (need the top-k indices) ∥ local down
		// (gelu-fused: reads msc.gate/msc.up from stage 2, two barriers upstream)
		emitGatherInAll()
		emitLocalDown()
		barrier()
		// stage 5: expert gelu — folded into the down gather's load when fused
		// (the stage and its barrier disappear; the stage-4 barrier already
		// orders the gate/up slabs ahead of the down gather)
		if !geluFold {
			emitGate(scratch.expertGateAll, scratch.expertUpAll, scratch.expertGatedAll, topK*expertDFF)
			barrier()
		}
		// stage 6: expert down gather
		emitGatherDownAll()
		barrier()
		// stage 7: route combine (weights from stage 3)
		emitWSum()
		barrier()
		// stage 8: the fused norm/combine tail -> residual out
		emitCombineNorms()
		if encCarryDisabled {
			endEncodingFast(enc)
			enc = computeCommandEncoderFast(cb)
			return enc, false, true, nil
		}
		// Leave the concurrent encoder OPEN — the caller carries it into the next
		// pass (a barrier at that pass's entry orders this block's writes).
		return enc, true, true, nil
	}

	// ---- serial path: the single-encoder order (hazard tracking serialises every edge) ----
	// A carried concurrent encoder has no hazard tracking — close it and reopen a
	// tracked serial encoder (the emit closures follow sink) before dispatching.
	if encConc {
		endEncodingFast(enc)
		enc = computeCommandEncoderFast(cb)
		sink = encSink{enc}
		encConc = false
	}
	if routerPlan.fusedPSO != nil && routerFusedEnabled {
		seam("router.fused")
		routerPlan.emitFused(sink)
	} else {
		seam("router.rms")
		routerPlan.emitRMS(sink)
		seam("router.qmv")
		routerPlan.emitQMV(sink)
		seam("router.topk")
		routerPlan.emitTopK(sink)
	}
	seam("moe.local")
	emitRMS(hBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
	// local MLP: the FFN megakernel when its geometry holds — its grid barrier needs the
	// arrive counter zeroed per dispatch, which the break-out path did with a HOST write;
	// here a 2-element bf16 copy from a device zero buffer resets it INSIDE the encoder
	// (hazard-ordered before the mega's atomics), keeping the whole lane host-free. The
	// qmv trio stays as the fallback for unsupported geometry.
	if megaLocal {
		if cerr := encCopyBF16Contig(enc, moeArriveZeroBuffer(), scratch.localMegaArrive, 0, 0, 2); cerr != nil {
			megaLocal = false
		}
	}
	if megaLocal {
		emitFFNMega(sink, localMegaPSO, scratch.localIn, 0, localGateViewQ, localUpViewQ, localDownViewQ, scratch.localMegaGated, scratch.localOut, 0, scratch.localMegaArrive, dModel, dFF)
	} else {
		emitQ(localGatePSO, localGatePacked, localGateScales, localGateBiases, scratch.localIn, msc.gate, dModel, dFF)
		emitQ(localUpPSO, localUpPacked, localUpScales, localUpBiases, scratch.localIn, msc.up, dModel, dFF)
		if !geluFold {
			emitGate(msc.gate, msc.up, msc.gated, dFF)
		}
		emitLocalDown()
	}
	seam("moe.expert")
	emitRMS(hBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
	if allRoutes {
		emitGatherInAll()
		if !geluFold {
			emitGate(scratch.expertGateAll, scratch.expertUpAll, scratch.expertGatedAll, topK*expertDFF)
		}
		emitGatherDownAll()
		seam("moe.tail")
		// combine the routes: acc = Σ_r w_r · down_r — one fused dispatch (byte-identical
		// rounding to the scale+add chain it replaces); the chain stays as the fallback.
		if wsumErr == nil {
			emitWSum()
		} else {
			for i := range topK {
				if i == 0 {
					emitScaleFromAt(scratch.expertDownAll, uint(i*dModel*bf16Size), uint(i*bf16Size), scratch.expertAcc, dModel)
				} else {
					emitScaleFromAt(scratch.expertDownAll, uint(i*dModel*bf16Size), uint(i*bf16Size), scratch.expertScaled, dModel)
					emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
				}
			}
		}
	} else {
		for i := range topK {
			if fusedExperts {
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked, expGateUpScales, expGateUpBiases, scratch.expertIn, msc.gate, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, 0)
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked, expGateUpScales, expGateUpBiases, scratch.expertIn, msc.up, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, expertDFF)
			} else {
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGatePacked, expGateScales, expGateBiases, scratch.expertIn, msc.gate, i, dModel, expertDFF, expGateGroupSize, expGateBits, 0)
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expUpPacked, expUpScales, expUpBiases, scratch.expertIn, msc.up, i, dModel, expertDFF, expUpGroupSize, expUpBits, 0)
			}
			emitGate(msc.gate, msc.up, msc.gated, expertDFF)
			emitGatherQ(gatherExpertDownPSO, gatherExpertDownMeta, expDownPacked, expDownScales, expDownBiases, msc.gated, msc.down, i, expertDFF, dModel, expDownGroupSize, expDownBits, 0)
			if i == 0 {
				emitScaleAt(msc.down, uint(i*bf16Size), scratch.expertAcc, dModel)
			} else {
				emitScaleAt(msc.down, uint(i*bf16Size), scratch.expertScaled, dModel)
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
	}
	if combineOK {
		emitCombineNorms()
	} else {
		emitRMS(scratch.localOut, post1Buf.buf, scratch.localNormed, post1Buf.off)
		emitRMS(scratch.expertAcc, post2Buf.buf, scratch.expertNormed, post2Buf.off)
		emitAdd(scratch.localNormed, scratch.expertNormed, scratch.combined)
		emitRMS(scratch.combined, postBuf.buf, scratch.ffResidual, postBuf.off)
		emitAdd(hBuf, scratch.ffResidual, outputBuf)
	}
	return enc, false, true, nil
}

func quantWeightViewsForShape(fn string, w QuantWeight, outDim, inDim, groupSize, bits int) (bufView, bufView, bufView, int, int, error) {
	groupSize, bits = quantWeightGeometryForShape(w, outDim, inDim, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || inDim%groupSize != 0 {
		return bufView{}, bufView{}, bufView{}, 0, 0, core.NewError(fn + ": invalid quant geometry")
	}
	wantPacked := outDim * inDim * bits / 8
	wantScales := outDim * (inDim / groupSize) * bf16Size
	if len(w.Packed) != wantPacked || len(w.Scales) != wantScales || len(w.Biases) != wantScales {
		return bufView{}, bufView{}, bufView{}, 0, 0, core.NewError(fn + ": quant weight size mismatch")
	}
	packed, scales, biases := quantWeightViews(w)
	return packed, scales, biases, groupSize, bits, nil
}

func moeBlockQuantAfterRouter(h []byte, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBuffer(h, nil, idx, weights, weightBuf, w, dModel, dFF, eps)
}

func moeBlockQuantAfterRouterWithBuffer(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, false)
}

func moeBlockQuantAfterRouterWithBufferInPool(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
}

func moeBlockQuantAfterRouterWithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockQuantAfterRouterWithBufferIntoInPool(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, true)
}

func moeBlockQuantAfterRouterWithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.moeBlockQuantAfterRouter: output buffer is nil")
	}
	_, err := moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, nil, outputBuf, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
	return err
}

func moeBlockQuantAfterRouterWithBufferPooled(h []byte, hBuf metal.MTLBuffer, out []byte, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	return moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, out, outputBuf, idx, nil, weights, weightBuf, w, dModel, dFF, eps, useAutoreleasePool, useCallerOut, nil)
}

func moeBlockQuantAfterRouterWithDeviceIndexBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, idx []int32, idxBuf metal.MTLBuffer, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, ownedScratch *moeBlockBF16Scratch) error {
	if outputBuf == nil {
		return core.NewError("native.moeBlockQuantAfterRouter: output buffer is nil")
	}
	_, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, nil, outputBuf, idx, idxBuf, weights, weightBuf, w, dModel, dFF, eps, false, false, ownedScratch)
	return err
}

func moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h []byte, hBuf metal.MTLBuffer, out []byte, outputBuf metal.MTLBuffer, idx []int32, idxBuf metal.MTLBuffer, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool, ownedScratch *moeBlockBF16Scratch) ([]byte, error) {
	expertDFF, numExperts, topK := w.ExpertDFF, w.NumExperts, w.TopK
	size := dModel * bf16Size
	if hBuf == nil && h == nil {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: h bytes or hBuf required")
	}
	if h != nil && len(h) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: h must be dModel bf16 bytes")
	}
	idxOnDevice := idxBuf != nil
	weightsOnDevice := weightBuf != nil
	if (!idxOnDevice && len(idx) != topK) || (idxOnDevice && idx != nil && len(idx) != topK) || (!weightsOnDevice && len(weights) != topK*bf16Size) || (weightsOnDevice && weights != nil && len(weights) != topK*bf16Size) {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: idx/weights length must equal topK")
	}
	// PreFFNormW is always required (see MoELayerWeights doc's bf16 rationale — the SAME zoo-vs-gemma4
	// shape split applies here); every OTHER norm is OPTIONAL: nil/empty = identity/skip. gemma4's
	// checkpoint always populates all five, so this is zero behaviour change for it.
	if len(w.PreFFNormW) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: PreFFNormW must be dModel bf16 bytes")
	}
	if len(w.PreFFNorm2W) != 0 && len(w.PreFFNorm2W) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: PreFFNorm2W must be dModel bf16 bytes or absent")
	}
	if len(w.PostFFNorm1W) != 0 && len(w.PostFFNorm1W) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: PostFFNorm1W must be dModel bf16 bytes or absent")
	}
	if len(w.PostFFNorm2W) != 0 && len(w.PostFFNorm2W) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: PostFFNorm2W must be dModel bf16 bytes or absent")
	}
	if len(w.PostFFNormW) != 0 && len(w.PostFFNormW) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: PostFFNormW must be dModel bf16 bytes or absent")
	}
	// hasLocal marks gemma4's always-on dense MLP branch (see MoELayerWeights doc); a zoo layer
	// (mixtral/dbrx/olmoe) declares none of the three, so the local branch's quant views/pipelines are
	// skipped entirely below — resolving quantWeightViewsForShape against an absent LocalGate/Up/Down
	// would otherwise fail the shape check (0 packed bytes) before hasLocal is even consulted.
	hasLocal := len(w.LocalGate.Packed) != 0 || len(w.LocalUp.Packed) != 0 || len(w.LocalDown.Packed) != 0
	var localGateView, localUpView, localDownView quantMLPProjView
	if hasLocal {
		localGatePacked, localGateScales, localGateBiases, localGateGroupSize, localGateBits, lerr := quantWeightViewsForShape("native.moeBlockQuantAfterRouter: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits)
		if lerr != nil {
			return nil, lerr
		}
		localUpPacked, localUpScales, localUpBiases, localUpGroupSize, localUpBits, lerr := quantWeightViewsForShape("native.moeBlockQuantAfterRouter: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits)
		if lerr != nil {
			return nil, lerr
		}
		localDownPacked, localDownScales, localDownBiases, localDownGroupSize, localDownBits, lerr := quantWeightViewsForShape("native.moeBlockQuantAfterRouter: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits)
		if lerr != nil {
			return nil, lerr
		}
		localGateView = quantMLPProjView{packed: localGatePacked, scales: localGateScales, biases: localGateBiases, groupSize: localGateGroupSize, bits: localGateBits}
		localUpView = quantMLPProjView{packed: localUpPacked, scales: localUpScales, biases: localUpBiases, groupSize: localUpGroupSize, bits: localUpBits}
		localDownView = quantMLPProjView{packed: localDownPacked, scales: localDownScales, biases: localDownBiases, groupSize: localDownGroupSize, bits: localDownBits}
	}

	fusedExperts := len(w.ExpGateUp.Packed) > 0
	expertGatePackedPer, expertGateScalePer := 0, 0
	expertDownPackedPer, expertDownScalePer := 0, 0
	var expGatePacked, expGateScales, expGateBiases bufView
	var expUpPacked, expUpScales, expUpBiases bufView
	var expGateUpPacked, expGateUpScales, expGateUpBiases bufView
	var expDownPacked, expDownScales, expDownBiases bufView
	var expGateGroupSize, expGateBits, expUpGroupSize, expUpBits, expGateUpGroupSize, expGateUpBits, expDownGroupSize, expDownBits int
	var err error
	if fusedExperts {
		expGateUpPacked, expGateUpScales, expGateUpBiases, expGateUpGroupSize, expGateUpBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
		expDownPacked, expDownScales, expDownBiases, expDownGroupSize, expDownBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
	} else {
		expGatePacked, expGateScales, expGateBiases, expGateGroupSize, expGateBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
		expUpPacked, expUpScales, expUpBiases, expUpGroupSize, expUpBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
		expDownPacked, expDownScales, expDownBiases, expDownGroupSize, expDownBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
	}
	if expGateGroupSize > 0 {
		expertGatePackedPer = expertDFF * dModel * expGateBits / 8
		expertGateScalePer = expertDFF * (dModel / expGateGroupSize) * bf16Size
	}
	if expGateUpGroupSize > 0 {
		expertGatePackedPer = expertDFF * dModel * expGateUpBits / 8
		expertGateScalePer = expertDFF * (dModel / expGateUpGroupSize) * bf16Size
	}
	if expDownGroupSize > 0 {
		expertDownPackedPer = dModel * expertDFF * expDownBits / 8
		expertDownScalePer = dModel * (expertDFF / expDownGroupSize) * bf16Size
	}
	if !idxOnDevice {
		for i := range idx {
			if idx[i] < 0 || int(idx[i]) >= numExperts {
				return nil, core.NewError("native.moeBlockQuantAfterRouter: expert index out of range")
			}
		}
	}

	bufferOut := outputBuf != nil
	callerOut := !bufferOut && useCallerOut && cap(out) >= size
	if bufferOut {
		out = nil
	} else if callerOut {
		out = out[:size]
	} else {
		out = make([]byte, size)
	}
	if dModel == 0 || expertDFF == 0 || (hasLocal && dFF == 0) {
		if bufferOut && size > 0 {
			clear(unsafe.Slice((*byte)(outputBuf.Contents()), size))
			return nil, nil
		}
		if !bufferOut {
			clear(out)
		}
		return out, nil
	}
	qmvPSO := func(outDim, inDim, groupSize, bits int) (metal.MTLComputePipelineState, error) {
		return pipelineFor(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	}
	useLocalMega := hasLocal && ffnMegaDefaultGeometry(dModel, dFF) && ffnMegaSupported(localGateView, localUpView, localDownView, dModel, dFF)
	var localMegaPSO metal.MTLComputePipelineState
	if useLocalMega {
		localMegaPSO, err = ffnMegaPipelineBits(localGateView.bits)
		if err != nil {
			useLocalMega = false
		}
	}
	var localGatePSO, localUpPSO, localDownPSO metal.MTLComputePipelineState
	if hasLocal && !useLocalMega {
		localGatePSO, err = qmvPSO(dFF, dModel, localGateView.groupSize, localGateView.bits)
		if err != nil {
			return nil, err
		}
		localUpPSO, err = qmvPSO(dFF, dModel, localUpView.groupSize, localUpView.bits)
		if err != nil {
			return nil, err
		}
		localDownPSO, err = qmvPSO(dModel, dFF, localDownView.groupSize, localDownView.bits)
		if err != nil {
			return nil, err
		}
	}
	hostIdxAvailable := len(idx) == topK
	useGatherExperts := (idxBuf != nil || hostIdxAvailable) && topK > 0 && affineBitsSupported(expDownBits)
	if fusedExperts {
		useGatherExperts = useGatherExperts && affineBitsSupported(expGateUpBits)
	} else {
		useGatherExperts = useGatherExperts && affineBitsSupported(expGateBits) && expGateBits == expUpBits && expGateGroupSize == expUpGroupSize
	}
	var gatherExpertInPSO, gatherExpertDownPSO metal.MTLComputePipelineState
	var gatherExpertInMeta, gatherExpertDownMeta *gatherQMVBF16Meta
	if useGatherExperts {
		inGroup, inBits := expGateGroupSize, expGateBits
		inRows := expertDFF
		if fusedExperts {
			inGroup, inBits = expGateUpGroupSize, expGateUpBits
			inRows = 2 * expertDFF
		}
		gatherExpertInPSO, err = gatherQMVBF16SteelPipeline(expertDFF, dModel, inGroup, inBits)
		if err == nil {
			gatherExpertDownPSO, err = gatherQMVBF16SteelPipeline(dModel, expertDFF, expDownGroupSize, expDownBits)
		}
		if err == nil {
			gatherExpertInMeta, err = gatherQMVBF16Metadata(numExperts, expertDFF, dModel, inGroup, inBits, inRows)
		}
		if err == nil {
			gatherExpertDownMeta, err = gatherQMVBF16Metadata(numExperts, dModel, expertDFF, expDownGroupSize, expDownBits, dModel)
		}
		if err != nil {
			useGatherExperts = false
		}
	}
	if !useGatherExperts && len(idx) != topK {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: host idx required when gathered device expert routing is unavailable")
	}
	var expGatePSO, expUpPSO, expGateUpPSO, expDownPSO metal.MTLComputePipelineState
	if !useGatherExperts {
		if fusedExperts {
			expGateUpPSO, err = qmvPSO(expertDFF, dModel, expGateUpGroupSize, expGateUpBits)
			if err != nil {
				return nil, err
			}
		} else {
			expGatePSO, err = qmvPSO(expertDFF, dModel, expGateGroupSize, expGateBits)
			if err != nil {
				return nil, err
			}
			expUpPSO, err = qmvPSO(expertDFF, dModel, expUpGroupSize, expUpBits)
			if err != nil {
				return nil, err
			}
		}
		expDownPSO, err = qmvPSO(dModel, expertDFF, expDownGroupSize, expDownBits)
		if err != nil {
			return nil, err
		}
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
	}
	// #63: the layer's declared expert-combine activation — see moeBlockBF16AfterRouterWithBufferPooled's
	// identical useSiLU doc (its bf16 sibling; the two funnels are kept in step).
	useSiLU := w.UsesSiLU
	scalePSO, scaleErr := bf16MulScalarPipeline()
	if scaleErr != nil && len(weights) != topK*bf16Size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: host weights required when device scalar scaling is unavailable")
	}
	// pre2W/pre2View are the expert branch's input norm. A zoo layer sets only PreFFNormW (see
	// MoELayerWeights doc's bf16 rationale — the same convention applies here); fall back to it
	// (bytes AND its pre-resolved view together) when PreFFNorm2W is absent. post1Buf/post2Buf/
	// postBuf stay the zero bufView (never resolved — resolving an absent norm's empty bytes would
	// panic in residentBytes) when their norm is absent; hasPost1/2/Combine below gate every emit
	// that would otherwise read them.
	pre2W, pre2View := w.PreFFNorm2W, w.preFFNorm2View
	if len(pre2W) == 0 {
		pre2W, pre2View = w.PreFFNormW, w.preFFNormView
	}
	hasPost1 := len(w.PostFFNorm1W) != 0
	hasPost2 := len(w.PostFFNorm2W) != 0
	hasPostCombine := len(w.PostFFNormW) != 0
	pre1Buf := bf16WeightView(w.PreFFNormW, w.preFFNormView)
	pre2Buf := bf16WeightView(pre2W, pre2View)
	var post1Buf, post2Buf, postBuf bufView
	if hasPost1 {
		post1Buf = bf16WeightView(w.PostFFNorm1W, w.postFFNorm1View)
	}
	if hasPost2 {
		post2Buf = bf16WeightView(w.PostFFNorm2W, w.postFFNorm2View)
	}
	if hasPostCombine {
		postBuf = bf16WeightView(w.PostFFNormW, w.postFFNormView)
	}

	var encErr error
	run := func() {
		var err error
		scratch := ownedScratch
		if scratch == nil {
			scratch, err = getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
			if err != nil {
				encErr = err
				return
			}
			defer putMoEBlockBF16Scratch(scratch)
		}
		routeIdxBuf := idxBuf
		if useGatherExperts && routeIdxBuf == nil {
			var ok bool
			routeIdxBuf, ok = scratch.indexView(idx)
			if !ok {
				idxBytes := unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), len(idx)*4)
				routeIdxBuf, err = scratch.idx.copyBuffer(idxBytes)
				runtime.KeepAlive(idx)
				if err != nil {
					encErr = err
					return
				}
			}
		}
		inputBuf := hBuf
		if inputBuf == nil {
			var ok bool
			inputBuf, ok = scratch.inputView(h)
			if !ok {
				inputBuf, err = scratch.h.copyBuffer(h)
				if err != nil {
					encErr = err
					return
				}
			}
		}
		weightsBuf := weightBuf
		if topK > 0 {
			if weightsBuf == nil {
				var ok bool
				weightsBuf, ok = scratch.weightsView(weights)
				if !ok {
					weightsBuf, err = scratch.weights.copyBuffer(weights)
					if err != nil {
						encErr = err
						return
					}
				}
			}
		} else {
			clear(unsafe.Slice((*byte)(scratch.expertAcc.Contents()), size))
		}
		msc := scratch.mlp
		if useLocalMega {
			if err = scratch.ensureLocalMegaScratch(); err != nil {
				encErr = err
				return
			}
			*scratch.localMegaArrivePtr = 0
		}
		finalOutBuf := scratch.out.buf
		directOut := false
		if bufferOut {
			finalOutBuf = outputBuf
			directOut = true
		} else if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				finalOutBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMS := func(x, weight, out metal.MTLBuffer, wOff uint) {
			emitRMSNorm(sink, rmsPSO, x, weight, out, wOff, dModel, eps, rmsTG)
		}
		emitQ := func(pso metal.MTLComputePipelineState, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, outOff uint, inDim, outDim int) {
			emitQMV(sink, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, out, outOff, inDim, outDim)
		}
		emitGatherQ := func(pso metal.MTLComputePipelineState, meta *gatherQMVBF16Meta, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff uint, route int, inDim, outDim, groupSize, bits, rowBase int) {
			emitGatherQMVBF16Steel(sink, pso, meta, x, wq, wqOff, scales, scalesOff, biases, biasesOff, routeIdxBuf, uint(route*4), out, 0, outDim, inDim, groupSize, bits, rowBase)
		}
		// emitGelu, despite its name (kept for callers below), is the layer's whole gate-activation
		// choice (#63): useSiLU routes to the SwiGLU gate first — gemma4 (useSiLU false) falls through
		// to the original GELU dispatch unchanged.
		emitGelu := func(gate, up, out metal.MTLBuffer, n int) error {
			if useSiLU {
				return encSiLUGateMulBF16(enc, gate, up, out, n)
			}
			if useFusedGelu {
				emitBinary(sink, geluPSO, gate, 0, up, 0, out, 0, n)
				return nil
			}
			return encGeluGateMul(enc, gate, up, out, msc, n)
		}
		emitScale := func(in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
			if scaleErr != nil {
				return encScaleBF16(enc, in, scalar, out, scalarOffset, scalarBytes, n)
			}
			sink.setPSO(scalePSO)
			sink.setBuf(in, 0, 0)
			sink.setBuf(scalar, scalarOffset, 1)
			sink.setBuf(out, 0, 2)
			sink.setI32(int32(n), 3)
			group := min(uint(n), uint(256))
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
				metal.MTLSize{Width: group, Height: 1, Depth: 1},
			)
			return nil
		}
		emitAdd := func(a, b, out metal.MTLBuffer) {
			emitBinary(sink, addPSO, a, 0, b, 0, out, 0, dModel)
		}
		if hasLocal {
			emitRMS(inputBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
			if useLocalMega {
				emitFFNMega(sink, localMegaPSO, scratch.localIn, 0, localGateView, localUpView, localDownView, scratch.localMegaGated, scratch.localOut, 0, scratch.localMegaArrive, dModel, dFF)
			} else {
				emitQ(localGatePSO, localGateView.packed.buf, localGateView.scales.buf, localGateView.biases.buf, scratch.localIn, msc.gate, localGateView.packed.off, localGateView.scales.off, localGateView.biases.off, 0, dModel, dFF)
				emitQ(localUpPSO, localUpView.packed.buf, localUpView.scales.buf, localUpView.biases.buf, scratch.localIn, msc.up, localUpView.packed.off, localUpView.scales.off, localUpView.biases.off, 0, dModel, dFF)
				if encErr = emitGelu(msc.gate, msc.up, msc.gated, dFF); encErr != nil {
					endEncodingFast(enc)
					return
				}
				emitQ(localDownPSO, localDownView.packed.buf, localDownView.scales.buf, localDownView.biases.buf, msc.gated, scratch.localOut, localDownView.packed.off, localDownView.scales.off, localDownView.biases.off, 0, dFF, dModel)
			}
		}
		emitRMS(inputBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
		for i := range topK {
			if useGatherExperts {
				if fusedExperts {
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.gate, expGateUpPacked.off, expGateUpScales.off, expGateUpBiases.off, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, 0)
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.up, expGateUpPacked.off, expGateUpScales.off, expGateUpBiases.off, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, expertDFF)
				} else {
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGatePacked.buf, expGateScales.buf, expGateBiases.buf, scratch.expertIn, msc.gate, expGatePacked.off, expGateScales.off, expGateBiases.off, i, dModel, expertDFF, expGateGroupSize, expGateBits, 0)
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expUpPacked.buf, expUpScales.buf, expUpBiases.buf, scratch.expertIn, msc.up, expUpPacked.off, expUpScales.off, expUpBiases.off, i, dModel, expertDFF, expUpGroupSize, expUpBits, 0)
				}
			} else {
				e := int(idx[i])
				if fusedExperts {
					gatePackedOff, gateScaleOff := uint(e*2*expertGatePackedPer), uint(e*2*expertGateScalePer)
					upPackedOff, upScaleOff := gatePackedOff+uint(expertGatePackedPer), gateScaleOff+uint(expertGateScalePer)
					emitQ(expGateUpPSO, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.gate, expGateUpPacked.off+gatePackedOff, expGateUpScales.off+gateScaleOff, expGateUpBiases.off+gateScaleOff, 0, dModel, expertDFF)
					emitQ(expGateUpPSO, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.up, expGateUpPacked.off+upPackedOff, expGateUpScales.off+upScaleOff, expGateUpBiases.off+upScaleOff, 0, dModel, expertDFF)
				} else {
					gatePackedOff, gateScaleOff := uint(e*expertGatePackedPer), uint(e*expertGateScalePer)
					emitQ(expGatePSO, expGatePacked.buf, expGateScales.buf, expGateBiases.buf, scratch.expertIn, msc.gate, expGatePacked.off+gatePackedOff, expGateScales.off+gateScaleOff, expGateBiases.off+gateScaleOff, 0, dModel, expertDFF)
					emitQ(expUpPSO, expUpPacked.buf, expUpScales.buf, expUpBiases.buf, scratch.expertIn, msc.up, expUpPacked.off+gatePackedOff, expUpScales.off+gateScaleOff, expUpBiases.off+gateScaleOff, 0, dModel, expertDFF)
				}
			}
			if encErr = emitGelu(msc.gate, msc.up, msc.gated, expertDFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			if useGatherExperts {
				emitGatherQ(gatherExpertDownPSO, gatherExpertDownMeta, expDownPacked.buf, expDownScales.buf, expDownBiases.buf, msc.gated, msc.down, expDownPacked.off, expDownScales.off, expDownBiases.off, i, expertDFF, dModel, expDownGroupSize, expDownBits, 0)
			} else {
				e := int(idx[i])
				downPackedOff, downScaleOff := uint(e*expertDownPackedPer), uint(e*expertDownScalePer)
				emitQ(expDownPSO, expDownPacked.buf, expDownScales.buf, expDownBiases.buf, msc.gated, msc.down, expDownPacked.off+downPackedOff, expDownScales.off+downScaleOff, expDownBiases.off+downScaleOff, 0, expertDFF, dModel)
			}
			var weightBytes []byte
			if len(weights) >= (i+1)*bf16Size {
				weightBytes = weights[i*bf16Size : (i+1)*bf16Size]
			}
			if i == 0 {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertAcc, uint(i*bf16Size), weightBytes, dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertScaled, uint(i*bf16Size), weightBytes, dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
		// Combine: gemma4 sums BOTH branches (each independently post-normed) then post-norms the sum.
		// A zoo layer with no local branch (hasLocal false) skips straight to the expert accumulator —
		// HF's llama-family MoE adds the routed output directly to the residual, no sandwich norm at
		// all (see MoELayerWeights doc) — and any INDIVIDUAL norm this arch omits is skipped in place
		// (nil/empty = identity). Dispatch order matches the original unconditional gemma4 sequence
		// exactly so gemma4's byte-for-byte output is untouched.
		var localBranch metal.MTLBuffer
		if hasLocal {
			localBranch = scratch.localOut
			if hasPost1 {
				emitRMS(scratch.localOut, post1Buf.buf, scratch.localNormed, post1Buf.off)
				localBranch = scratch.localNormed
			}
		}
		ffResidualSrc := scratch.expertAcc
		if hasPost2 {
			emitRMS(scratch.expertAcc, post2Buf.buf, scratch.expertNormed, post2Buf.off)
			ffResidualSrc = scratch.expertNormed
		}
		if hasLocal {
			emitAdd(localBranch, ffResidualSrc, scratch.combined)
			ffResidualSrc = scratch.combined
		}
		if hasPostCombine {
			emitRMS(ffResidualSrc, postBuf.buf, scratch.ffResidual, postBuf.off)
			ffResidualSrc = scratch.ffResidual
		}
		emitAdd(inputBuf, ffResidualSrc, finalOutBuf)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		// The decode loop's fully-device path (session-OWNED scratch, device h/idx/weights,
		// buffer output, no host clears or mega arrive-counter) skips the completion wait:
		// the queue orders every later consumer, the owner guarantees single-flight scratch
		// reuse (GPU-GPU, commit-ordered), and no host bytes are read after this point. Every
		// other shape keeps the wait — pooled scratch lifetimes and host readbacks assume it.
		skipWait := ownedScratch != nil && bufferOut && hBuf != nil && idxBuf != nil && weightBuf != nil &&
			topK > 0 && !useLocalMega && scaleErr == nil
		if !skipWait {
			waitUntilCompletedFast(cb)
		}
		if !directOut {
			copy(out, scratch.out.bytes[:size])
		}
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	return out, encErr
}

// mlpTransformQuant is mlpTransformBF16 for a 4-bit MLP: gate/up (dModel→dFF) and down
// (dFF→dModel) via resident quant QMVBF16, with the SwiGLU activation between — no
// residual. The local quant weights are fixed per layer, so their packed/scales/biases
// buffers follow the same resident route as selected quant expert slices.
func mlpTransformQuant(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformQuant: x must be dModel bf16 bytes")
	}
	if dModel == 0 || dFF == 0 {
		return make([]byte, dModel*bf16Size), nil
	}
	gateView, upView, downView, err := mlpTransformQuantViews("native.mlpTransformQuant", gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if ffnMegaDefaultGeometry(dModel, dFF) {
		if out, ok, err := mlpTransformQuantMegaWithViews(x, gateView, upView, downView, dModel, dFF); ok || err != nil {
			return out, err
		}
	}
	return mlpTransformQuantComposedWithViews(x, gateView, upView, downView, dModel, dFF)
}

func mlpTransformQuantComposed(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantComposedIntoInternal(nil, x, gate, up, down, dModel, dFF, groupSize, bits, false)
}

func mlpTransformQuantComposedInto(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantComposedIntoInternal(out, x, gate, up, down, dModel, dFF, groupSize, bits, true)
}

func mlpTransformQuantComposedIntoInternal(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformQuant: x must be dModel bf16 bytes")
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}
	gateView, upView, downView, err := mlpTransformQuantViews("native.mlpTransformQuant", gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		return nil, err
	}
	return mlpTransformQuantComposedWithViewsInto(out, x, gateView, upView, downView, dModel, dFF, callerOut)
}

func mlpTransformQuantMega(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantMegaIntoInternal(nil, x, gate, up, down, dModel, dFF, groupSize, bits, false)
}

func mlpTransformQuantMegaInto(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantMegaIntoInternal(out, x, gate, up, down, dModel, dFF, groupSize, bits, true)
}

func mlpTransformQuantMegaIntoInternal(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformQuantMega: x must be dModel bf16 bytes")
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}
	gateView, upView, downView, err := mlpTransformQuantViews("native.mlpTransformQuantMega", gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		return nil, err
	}
	out, ok, err := mlpTransformQuantMegaWithViewsInto(out, x, gateView, upView, downView, dModel, dFF, callerOut)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, core.NewError("native.mlpTransformQuantMega: unsupported quant geometry or megakernel unavailable")
	}
	return out, nil
}

func mlpTransformQuantViews(fn string, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) (quantMLPProjView, quantMLPProjView, quantMLPProjView, error) {
	gatePacked, gateScales, gateBiases, gateGroupSize, gateBits, err := quantWeightViewsForShape(fn+": gate", gate, dFF, dModel, groupSize, bits)
	if err != nil {
		return quantMLPProjView{}, quantMLPProjView{}, quantMLPProjView{}, err
	}
	upPacked, upScales, upBiases, upGroupSize, upBits, err := quantWeightViewsForShape(fn+": up", up, dFF, dModel, groupSize, bits)
	if err != nil {
		return quantMLPProjView{}, quantMLPProjView{}, quantMLPProjView{}, err
	}
	downPacked, downScales, downBiases, downGroupSize, downBits, err := quantWeightViewsForShape(fn+": down", down, dModel, dFF, groupSize, bits)
	if err != nil {
		return quantMLPProjView{}, quantMLPProjView{}, quantMLPProjView{}, err
	}
	return quantMLPProjView{packed: gatePacked, scales: gateScales, biases: gateBiases, groupSize: gateGroupSize, bits: gateBits},
		quantMLPProjView{packed: upPacked, scales: upScales, biases: upBiases, groupSize: upGroupSize, bits: upBits},
		quantMLPProjView{packed: downPacked, scales: downScales, biases: downBiases, groupSize: downGroupSize, bits: downBits},
		nil
}

func mlpTransformQuantMegaWithViews(x []byte, gate, up, down quantMLPProjView, dModel, dFF int) ([]byte, bool, error) {
	return mlpTransformQuantMegaWithViewsInto(nil, x, gate, up, down, dModel, dFF, false)
}

func mlpTransformQuantMegaWithViewsInto(out []byte, x []byte, gate, up, down quantMLPProjView, dModel, dFF int, useCallerOut bool) ([]byte, bool, error) {
	if !ffnMegaSupported(gate, up, down, dModel, dFF) {
		return nil, false, nil
	}
	return mlpTransformQuantMegaRun(out, x, gate, up, down, dModel, dFF, useCallerOut)
}

// mlpTransformQuantMegaRun executes the megakernel for any KERNEL-compatible width, bypassing
// the routing receipt in ffnMegaSupported — the width parity tests drive 8-bit through here.
func mlpTransformQuantMegaRun(out []byte, x []byte, gate, up, down quantMLPProjView, dModel, dFF int, useCallerOut bool) ([]byte, bool, error) {
	if !ffnMegaKernelCompatible(gate, up, down, dModel, dFF) {
		return nil, false, nil
	}
	pso, err := ffnMegaPipelineBits(gate.bits)
	if err != nil {
		return nil, false, nil
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMLPTransformMegaScratch(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		defer putMLPTransformMegaScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		*scratch.arrivePtr = 0
		outBuf := scratch.out
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitFFNMega(sink, pso, xBuf, 0, gate, up, down, scratch.gated, outBuf, 0, scratch.arrive, dModel, dFF)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.outBytes[:len(out)])
		}
	})
	return out, true, encErr
}

func mlpTransformQuantComposedWithViews(x []byte, gate, up, down quantMLPProjView, dModel, dFF int) ([]byte, error) {
	return mlpTransformQuantComposedWithViewsInto(nil, x, gate, up, down, dModel, dFF, false)
}

func mlpTransformQuantComposedWithViewsInto(out []byte, x []byte, gate, up, down quantMLPProjView, dModel, dFF int, useCallerOut bool) ([]byte, error) {
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	gatePSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, gate.groupSize, gate.bits))
	if err != nil {
		return nil, err
	}
	upPSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, up.groupSize, up.bits))
	if err != nil {
		return nil, err
	}
	downPSO, err := pipelineFor(qmvBF16KernelName(dModel, dFF, down.groupSize, down.bits))
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMLPTransformScratch(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		defer putMLPTransformScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		msc := scratch.mlp
		outBuf := msc.down
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitQMV(sink, gatePSO, gate.packed.buf, gate.packed.off, gate.scales.buf, gate.scales.off, gate.biases.buf, gate.biases.off, xBuf, msc.gate, 0, dModel, dFF)
		emitQMV(sink, upPSO, up.packed.buf, up.packed.off, up.scales.buf, up.scales.off, up.biases.buf, up.biases.off, xBuf, msc.up, 0, dModel, dFF)
		if useFusedGelu {
			emitBinary(sink, geluPSO, msc.gate, 0, msc.up, 0, msc.gated, 0, dFF)
		} else {
			encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF)
		}
		if encErr != nil {
			endEncodingFast(enc)
			return
		}
		emitQMV(sink, downPSO, down.packed.buf, down.packed.off, down.scales.buf, down.scales.off, down.biases.buf, down.biases.off, msc.gated, outBuf, 0, dFF, dModel)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(msc.down.Contents()), len(out)))
		}
	})
	return out, encErr
}

// MoEBlockQuant is MoEBlockBF16 for a 4-bit MoE layer — the same dual-branch feed-forward
// (local dense MLP + router→topK experts, each independently normed, summed, post-normed,
// residual added once), with QMVBF16 / MoERouterQuant / MoEExpertsQuant in place of the bf16
// ops. The router runs on the raw residual; the local MLP uses dFF, the experts ExpertDFF.
func MoEBlockQuant(h []byte, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBuffer(h, nil, w, dModel, dFF, eps)
}

func MoEBlockQuantInto(out []byte, h []byte, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferInto(out, h, nil, w, dModel, dFF, eps)
}

func moeBlockQuantWithBuffer(h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferPooled(h, hBuf, w, dModel, dFF, eps, true)
}

func moeBlockQuantWithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockQuantWithBufferInPool(h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferPooled(h, hBuf, w, dModel, dFF, eps, false)
}

func moeBlockQuantWithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, ownedScratch *moeBlockBF16Scratch) error {
	if outputBuf == nil {
		return core.NewError("native.MoEBlockQuant: output buffer is nil")
	}
	if err := ensureInit(); err != nil {
		return err
	}
	// h may be nil when hBuf carries the hidden (the decode loop's no-wait handoff: the live
	// command buffer commits without a completion wait and the MoE stages queue behind it —
	// nothing on the happy path reads host bytes). Host-bytes callers still validate.
	if hBuf == nil && h == nil {
		return core.NewError("native.MoEBlockQuant: h bytes or hBuf required")
	}
	if h != nil && len(h) != dModel*bf16Size {
		return core.NewError("native.MoEBlockQuant: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK
	routerNormW, routerNormView := w.routerNorm() // zoo layers (no RouterNormWScaled) fall back to PreFFNormW

	if quantMoEDeviceRouterBuffersUsable(w, dModel) {
		weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKBuffersWithBufferInPool(h, hBuf, routerNormW, routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps, w.NormaliseTopK)
		if ok || err != nil {
			if err != nil {
				return err
			}
			var idxBuf metal.MTLBuffer
			if routerScratch != nil {
				idxBuf = routerScratch.idxBuf
			}
			err = moeBlockQuantAfterRouterWithDeviceIndexBufferOutputInPool(h, hBuf, outputBuf, nil, idxBuf, nil, weightBuf, w, dModel, dFF, eps, ownedScratch)
			putRouterDeviceScratch(routerScratch)
			return err
		}
	}
	// moeRouterQuantDeviceTopKNoCopyWithBufferInPool's signature is FROZEN (see its own doc,
	// router.go) — it always assumes normalise=true (the GPU kernel's only order), so a
	// NormaliseMoETopK=false arch (#65) must skip this tier entirely rather than call it, falling
	// straight to the host-capable moeRouterQuantWithViews below.
	if w.NormaliseTopK {
		if idx, weights, weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKNoCopyWithBufferInPool(h, hBuf, routerNormW, routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps); ok || err != nil {
			if err != nil {
				return err
			}
			var idxBuf metal.MTLBuffer
			if routerScratch != nil {
				idxBuf = routerScratch.idxBuf
			}
			idxView, weightView := quantMoEHostRouterViewsForDeviceBuffers(idx, weights, idxBuf, weightBuf, w, dModel)
			err = moeBlockQuantAfterRouterWithDeviceIndexBufferOutputInPool(h, hBuf, outputBuf, idxView, idxBuf, weightView, weightBuf, w, dModel, dFF, eps, ownedScratch)
			putRouterDeviceScratch(routerScratch)
			return err
		}
	}
	idx, weights, err := moeRouterQuantWithViews(h, routerNormW, routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps, w.NormaliseTopK)
	if err != nil {
		return err
	}
	return moeBlockQuantAfterRouterWithBufferOutputInPool(h, hBuf, outputBuf, idx, weights, nil, w, dModel, dFF, eps)
}

func moeBlockQuantWithBufferPooled(h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool) ([]byte, error) {
	return moeBlockQuantWithBufferPooledInto(nil, h, hBuf, w, dModel, dFF, eps, useAutoreleasePool, false)
}

func moeBlockQuantWithBufferPooledInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != dModel*bf16Size {
		return nil, core.NewError("native.MoEBlockQuant: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK

	if useAutoreleasePool {
		var blockOut []byte
		var blockErr error
		withAutoreleasePool(func() {
			blockOut, blockErr = moeBlockQuantWithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, false, useCallerOut)
		})
		return blockOut, blockErr
	}

	routerNormW, routerNormView := w.routerNorm() // zoo layers (no RouterNormWScaled) fall back to PreFFNormW
	if quantMoEDeviceRouterBuffersUsable(w, dModel) {
		weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKBuffersWithBufferInPool(h, hBuf, routerNormW, routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps, w.NormaliseTopK)
		if ok || err != nil {
			if err != nil {
				return nil, err
			}
			var idxBuf metal.MTLBuffer
			if routerScratch != nil {
				idxBuf = routerScratch.idxBuf
			}
			blockOut, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, out, nil, nil, idxBuf, nil, weightBuf, w, dModel, dFF, eps, false, useCallerOut, nil)
			putRouterDeviceScratch(routerScratch)
			return blockOut, err
		}
	}
	// see moeBlockQuantWithBufferOutputInPool's identical note: this tier's function has a FROZEN
	// signature that always assumes normalise=true, so skip it entirely when the arch wants false.
	if w.NormaliseTopK {
		if idx, weights, weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKNoCopyWithBufferInPool(h, hBuf, routerNormW, routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps); ok || err != nil {
			if err != nil {
				return nil, err
			}
			var idxBuf metal.MTLBuffer
			if routerScratch != nil {
				idxBuf = routerScratch.idxBuf
			}
			idxView, weightView := quantMoEHostRouterViewsForDeviceBuffers(idx, weights, idxBuf, weightBuf, w, dModel)
			blockOut, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, out, nil, idxView, idxBuf, weightView, weightBuf, w, dModel, dFF, eps, false, useCallerOut, nil)
			putRouterDeviceScratch(routerScratch)
			return blockOut, err
		}
	}
	idx, weights, err := moeRouterQuantWithViews(h, routerNormW, routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps, w.NormaliseTopK)
	if err != nil {
		return nil, err
	}
	if useCallerOut {
		return moeBlockQuantAfterRouterWithBufferIntoInPool(out, h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	}
	return moeBlockQuantAfterRouterWithBufferInPool(h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
}

func quantMoEHostRouterViewsForDeviceBuffers(idx []int32, weights []byte, idxBuf, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel int) ([]int32, []byte) {
	if idxBuf == nil || weightBuf == nil || !quantMoEDeviceRouterBuffersUsable(w, dModel) {
		return idx, weights
	}
	return nil, nil
}

// quantMoEDeviceRouterBuffersUsable is the keystone predicate every device-router lane gates on
// (decode_forward_arch.go's sharedEncodeEligible/fully-encoded-lane/break-out-flow checks,
// moe_batch.go, and this file's own tiers): a false here forces the SAFE host-capable router path
// everywhere, uniformly. !w.NormaliseTopK (#65) declines it because the GPU topK kernel can only
// implement the always-renormalise order (see moeRouterQuantDeviceTopKWithBufferPooled's doc) —
// OLMoE's norm_topk_prob=false checkpoint needs the host's softmax-over-all-then-gather-no-renorm
// instead.
func quantMoEDeviceRouterBuffersUsable(w MoEQuantLayerWeights, dModel int) bool {
	if w.TopK <= 0 || w.NumExperts <= 0 || w.ExpertDFF <= 0 || !w.NormaliseTopK {
		return false
	}
	if _, err := bf16MulScalarPipeline(); err != nil {
		return false
	}
	expertDFF, numExperts := w.ExpertDFF, w.NumExperts
	downGroup, downBits := quantWeightGeometryForShape(w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if downGroup <= 0 || !affineBitsSupported(downBits) || expertDFF%downGroup != 0 {
		return false
	}
	if len(w.ExpGateUp.Packed) > 0 {
		gateUpGroup, gateUpBits := quantWeightGeometryForShape(w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		return gateUpGroup > 0 && affineBitsSupported(gateUpBits) && dModel%gateUpGroup == 0
	}
	gateGroup, gateBits := quantWeightGeometryForShape(w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
	upGroup, upBits := quantWeightGeometryForShape(w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
	return gateGroup > 0 && upGroup > 0 && affineBitsSupported(gateBits) && gateBits == upBits && dModel%gateGroup == 0 && dModel%upGroup == 0 && gateGroup == upGroup
}
