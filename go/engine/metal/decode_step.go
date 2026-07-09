// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// This file completes the decode step DecodeLayer deferred: the cache-WRITE half.
// A real autoregressive step projects K and V from the current token, RoPEs the
// new K, APPENDS them to a growing KV cache, then attends over the grown window —
// where DecodeLayer attended a fixed handed-in cache (Q-only).
//
// The cache is SEQ-MAJOR [maxLen, nKVHeads, headDim] (headDim innermost). That
// makes a token's K (and V) for all heads one contiguous row at byte offset
// pos*nKVHeads*headDim*2, so the projection writes STRAIGHT into the cache via a
// bound-buffer offset — no copy kernel (the static metallib has none; copies are
// JIT). The sdpa_vector kernel indexes keys as kv_head*k_head_stride +
// seq*k_seq_stride + d, so seq-major just sets k_head_stride=headDim,
// k_seq_stride=nKVHeads*headDim and N=pos+1 (the live window). bf16 throughout.

// attnScratch holds the attention-half intermediates, allocated once so the
// decode loop reuses them every token (no per-token buffer churn).
type attnScratch struct {
	dModel, qDim, kvDim, nHeads, maxLen int
	normed, q, qr, kProj, attn, attnOut metal.MTLBuffer
	// vProj stages the V row for q8 paged landings (#357): the projection
	// cannot write int8 pages directly, so K/V project into scratch, the
	// norms run there, and the quantise-store hop writes the page. bf16
	// landings keep writing the page row and never touch it.
	vProj metal.MTLBuffer
	// 2-pass long-context SDPA intermediates — nil unless the path opted in (maxLen
	// reaches the knee), so the router falls back to single-pass when absent. Sized to
	// the largest layer's qDim × the maxLen block count, allocated once (no per-token
	// churn): partials [blocks·qDim] bf16, sums/maxs [blocks·nHeads] float32.
	p2Partials, p2Sums, p2Maxs metal.MTLBuffer
}

type attnScratchKey struct {
	dModel, qDim, kvDim, nHeads, maxLen int
}

type attnScratchPool struct {
	core.Pool[*attnScratch]
}

var attnScratchPools sync.Map

func attnScratchPoolFor(key attnScratchKey) *attnScratchPool {
	if v, ok := attnScratchPools.Load(key); ok {
		return v.(*attnScratchPool)
	}
	pool := new(attnScratchPool)
	if v, loaded := attnScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*attnScratchPool)
	}
	return pool
}

func attnScratchReady(sc *attnScratch, key attnScratchKey) bool {
	if sc == nil || sc.dModel != key.dModel || sc.qDim != key.qDim || sc.kvDim != key.kvDim || sc.nHeads != key.nHeads || sc.maxLen != key.maxLen ||
		sc.normed == nil || sc.q == nil || sc.qr == nil || sc.kProj == nil || sc.vProj == nil || sc.attn == nil || sc.attnOut == nil {
		return false
	}
	if key.maxLen >= sdpa2PassMinKV && key.nHeads > 0 {
		return sc.p2Partials != nil && sc.p2Sums != nil && sc.p2Maxs != nil
	}
	return true
}

// newAttnScratch allocates the reusable attention-half scratch. When maxLen reaches
// the 2-pass knee (and nHeads is known), it also allocates the once-per-session 2-pass
// SDPA intermediates so long-context decode routes to the 2-pass kernels with no
// per-token allocation; pass maxLen=0 to keep a path single-pass only. qDim should be
// the LARGEST layer's q dimension (the scratch is shared across all layers).
func newAttnScratch(dModel, qDim, kvDim, nHeads, maxLen int) attnScratch {
	sc := attnScratch{
		dModel: dModel, qDim: qDim, kvDim: kvDim, nHeads: nHeads, maxLen: maxLen,
		normed:  scratchBF16(dModel),
		q:       scratchBF16(qDim),
		qr:      scratchBF16(qDim),
		kProj:   scratchBF16(kvDim),
		vProj:   scratchBF16(kvDim),
		attn:    scratchBF16(qDim),
		attnOut: scratchBF16(dModel),
	}
	if maxLen >= sdpa2PassMinKV && nHeads > 0 {
		blocks := int(sdpa2PassBlocks(maxLen))
		sc.p2Partials = scratchBF16(blocks * qDim)
		sc.p2Sums = scratchF32(blocks * nHeads)
		sc.p2Maxs = scratchF32(blocks * nHeads)
	}
	return sc
}

func getAttnScratch(dModel, qDim, kvDim, nHeads, maxLen int) *attnScratch {
	key := attnScratchKey{dModel: dModel, qDim: qDim, kvDim: kvDim, nHeads: nHeads, maxLen: maxLen}
	pool := attnScratchPoolFor(key)
	if sc := pool.Get(); sc != nil {
		if attnScratchReady(sc, key) {
			return sc
		}
	}
	sc := newAttnScratch(dModel, qDim, kvDim, nHeads, maxLen)
	return &sc
}

func putAttnScratch(sc *attnScratch) {
	if sc == nil {
		return
	}
	key := attnScratchKey{dModel: sc.dModel, qDim: sc.qDim, kvDim: sc.kvDim, nHeads: sc.nHeads, maxLen: sc.maxLen}
	if attnScratchReady(sc, key) {
		attnScratchPoolFor(key).Put(sc)
	}
}

// mlpScratch holds the MLP-half intermediates (the gelu chain), allocated once.
type mlpScratch struct {
	dModel, dFF                 int
	mlpNormed, gate, up         metal.MTLBuffer
	x2, x3, x3s, inner          metal.MTLBuffer
	scaled, tnh, onePlus, halfG metal.MTLBuffer
	gelu, gated, down           metal.MTLBuffer
	c044, c079, c1, c05         metal.MTLBuffer
}

type mlpScratchKey struct {
	dModel, dFF int
}

type mlpScratchPool struct {
	core.Pool[*mlpScratch]
}

var mlpScratchPools sync.Map

func mlpScratchPoolFor(key mlpScratchKey) *mlpScratchPool {
	if v, ok := mlpScratchPools.Load(key); ok {
		return v.(*mlpScratchPool)
	}
	pool := new(mlpScratchPool)
	if v, loaded := mlpScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*mlpScratchPool)
	}
	return pool
}

func mlpScratchReady(sc *mlpScratch, key mlpScratchKey) bool {
	if sc == nil || sc.dModel != key.dModel || sc.dFF != key.dFF ||
		sc.mlpNormed == nil || sc.gate == nil || sc.up == nil || sc.gated == nil || sc.down == nil {
		return false
	}
	if gpuHasGeluKernel() {
		return true
	}
	return sc.x2 != nil && sc.x3 != nil && sc.x3s != nil && sc.inner != nil &&
		sc.scaled != nil && sc.tnh != nil && sc.onePlus != nil && sc.halfG != nil &&
		sc.gelu != nil && sc.c044 != nil && sc.c079 != nil && sc.c1 != nil && sc.c05 != nil
}

func newMLPScratch(dModel, dFF int) mlpScratch {
	sc := mlpScratch{
		dModel: dModel, dFF: dFF,
		mlpNormed: scratchBF16(dModel),
		gate:      scratchBF16(dFF), up: scratchBF16(dFF),
		gated: scratchBF16(dFF), down: scratchBF16(dModel),
	}
	if gpuHasGeluKernel() {
		return sc
	}
	sc.x2, sc.x3, sc.x3s, sc.inner = scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
	sc.scaled, sc.tnh, sc.onePlus, sc.halfG = scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
	sc.gelu = scratchBF16(dFF)
	sc.c044 = bf16ConstBuffer(dFF, 0.044715)
	sc.c079 = bf16ConstBuffer(dFF, 0.7978845608028654)
	sc.c1 = bf16ConstBuffer(dFF, 1.0)
	sc.c05 = bf16ConstBuffer(dFF, 0.5)
	return sc
}

func getMLPScratch(dModel, dFF int) *mlpScratch {
	key := mlpScratchKey{dModel: dModel, dFF: dFF}
	pool := mlpScratchPoolFor(key)
	if sc := pool.Get(); sc != nil {
		if mlpScratchReady(sc, key) {
			return sc
		}
	}
	sc := newMLPScratch(dModel, dFF)
	return &sc
}

func putMLPScratch(sc *mlpScratch) {
	if sc == nil {
		return
	}
	key := mlpScratchKey{dModel: sc.dModel, dFF: sc.dFF}
	if mlpScratchReady(sc, key) {
		mlpScratchPoolFor(key).Put(sc)
	}
}

// encResidualMaybeNorm encodes out = x + v, or out = x + RMSNorm(v, norm) when norm is
// non-nil (the gemma4 post-attention / post-feed-forward norm, applied to the branch
// output before the residual add). norm is a bufView so the weight can be a no-copy shard view
// at an offset; a nil norm.buf skips the norm. scratch holds the normed value; pass a buffer the
// caller no longer needs (sc.normed after the attention projections, sc.mlpNormed after
// the MLP projections). Bf16, dModel-wide.
func encResidualMaybeNorm(enc metal.MTLComputeCommandEncoder, x, v, scratch, out metal.MTLBuffer, norm bufView, dModel int, eps float32) error {
	return encResidualMaybeNormAt(enc, x, 0, v, 0, scratch, out, 0, norm, dModel, eps)
}

// encResidualRowsMaybeNorm is encResidualMaybeNormAt across `rows` contiguous rows in two
// dispatches — one norm-rows + one add over rows·dModel — or ONE add when norm is nil. Per-row
// bytes match the per-row calls: the rows kernel norms each row independently and the add is
// elementwise (and the fused row-0 variant rounds identically to the composed pair — the parity
// the batched pass has leaned on since the fold landed).
func encResidualRowsMaybeNorm(enc metal.MTLComputeCommandEncoder, x metal.MTLBuffer, xOff uint, v metal.MTLBuffer, vOff uint, scratch, out metal.MTLBuffer, outOff uint, norm bufView, rows, dModel int, eps float32) error {
	if norm.buf == nil {
		return encAddBF16To(enc, x, v, out, xOff, vOff, outOff, rows*dModel)
	}
	if err := encRMSNormRowsBF16(enc, v, norm.buf, scratch, vOff, norm.off, 0, rows, dModel, eps); err != nil {
		return err
	}
	return encAddBF16To(enc, x, scratch, out, xOff, 0, outOff, rows*dModel)
}

func encResidualMaybeNormAt(enc metal.MTLComputeCommandEncoder, x metal.MTLBuffer, xOff uint, v metal.MTLBuffer, vOff uint, scratch, out metal.MTLBuffer, outOff uint, norm bufView, dModel int, eps float32) error {
	if norm.buf == nil {
		return encAddBF16To(enc, x, v, out, xOff, vOff, outOff, dModel)
	}
	// Lockstep with the ICB's setRMSResidual: when the custom library is present the ICB fuses
	// out = res + rms(branch) into one kernel, so the re-encode must use the SAME fused kernel to
	// stay byte-equal (the ICB-vs-re-encode parity tests) — bound at the row's offsets when it
	// lives inside a shared K-row buffer. Same gpuHasGeluKernel gate as the recorder.
	if gpuHasGeluKernel() {
		return encRMSNormResidualBF16At(enc, v, norm.buf, x, out, vOff, norm.off, xOff, outOff, dModel, eps)
	}
	if vOff == 0 {
		if err := encRMSNormBF16(enc, v, norm.buf, scratch, norm.off, dModel, eps); err != nil {
			return err
		}
	} else if err := encRMSNormRowsBF16(enc, v, norm.buf, scratch, vOff, norm.off, 0, 1, dModel, eps); err != nil {
		return err
	}
	return encAddBF16To(enc, x, scratch, out, xOff, 0, outOff, dModel)
}

// encAttnHalfKV encodes the real attention half — projections, K-RoPE into the
// cache, V into the cache, attention over the grown window, output projection,
// residual — into enc. The new token's K/V are written into kCacheBuf/vCacheBuf
// (seq-major) at row pos via the projection's bound-buffer offset; offBuf must
// already hold int32(pos). attends over rows [0..pos]; writes x + Wo·attn -> h (with
// the gemma4 post-attention norm on Wo·attn first when postAttnNorm is non-nil).
func encAttnHalfKV(
	enc metal.MTLComputeCommandEncoder,
	x, kCacheBuf, vCacheBuf, offBuf, h metal.MTLBuffer,
	attnNormW, postAttnNorm, qNorm, kNorm bufView, valueNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	return encAttnHalfKVAt(enc, x, kCacheBuf, vCacheBuf, offBuf, h, 0,
		attnNormW, postAttnNorm, qNorm, kNorm, valueNorm, sc, proj,
		dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim, base, scale, eps, ropeFreqs)
}

func encAttnHalfKVAt(
	enc metal.MTLComputeCommandEncoder,
	x, kCacheBuf, vCacheBuf, offBuf, h metal.MTLBuffer, offOff uint,
	attnNormW, postAttnNorm, qNorm, kNorm bufView, valueNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	return encAttnHalfKVInputAt(enc, x, 0, kCacheBuf, vCacheBuf, offBuf, h, 0, offOff,
		attnNormW, postAttnNorm, qNorm, kNorm, valueNorm, sc, proj,
		dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim, base, scale, eps, ropeFreqs)
}

func encAttnHalfKVInputAt(
	enc metal.MTLComputeCommandEncoder,
	x metal.MTLBuffer, xOff uint, kCacheBuf, vCacheBuf, offBuf, h metal.MTLBuffer, hOff, offOff uint,
	attnNormW, postAttnNorm, qNorm, kNorm bufView, valueNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	// the cache is a RING of size slideW for sliding layers (slideW>0): write this token's row at
	// pos%slideW (evicting pos-slideW, which has just left the window) and attend the whole live ring
	// [0..n). Global layers (slideW==0) keep the seq-major cache: write at pos, attend [0..pos]. The
	// ring reads in slot order, not absolute order — but the softmax is permutation-invariant and each
	// cached K carries its OWN baked-in RoPE (rotated by the absolute pos at write), so the attention
	// output is identical bar the ~1e-6 fp32 sum-order rounding. This is what lets a sliding layer
	// allocate slideW rows instead of maxLen (the full-context KV-cache memory fix).
	slot, n := pos, pos+1
	if slideW > 0 {
		slot = pos % slideW
		if n > slideW {
			n = slideW
		}
	}
	rowOff := uint(slot * kvDim * bf16Size) // byte offset of this token's cache ring slot
	// entry rms via the size-specialised single-row kernel at the row's offset — the batched
	// interleave's rows must norm bit-identically to the sequential step (the generic rows
	// kernel reduces in a different order and drifts the whole layer by ulps).
	if err := encRMSNormBF16At(enc, x, attnNormW.buf, sc.normed, xOff, attnNormW.off, 0, dModel, eps); err != nil {
		return err
	}
	// query: project, (gemma4 per-head QK-norm), rotate IN PLACE (so partial rotary's tail keeps the projected value)
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		// fused: sc.q = RoPE(RMSNorm(sc.q, qNorm)) in one op — lockstep with the ICB setQKNormRope
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	// key: project STRAIGHT into the cache row, then (gemma4 per-head QK-norm) + rotate IN PLACE
	// there — partial rotary leaves the tail as the projected+normed value already in the cache.
	if err := proj.project(enc, sc.normed, kCacheBuf, rowOff, projK); err != nil {
		return err
	}
	if gpuHasGeluKernel() && kNorm.buf != nil {
		// fused: kCache row = RoPE(RMSNorm(kCache row, kNorm)) in one op — lockstep with the ICB setQKNormRope
		if err := encQKNormRopeAt(enc, kCacheBuf, kNorm.buf, kCacheBuf, rowOff, kNorm.off, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if kNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, kCacheBuf, kNorm.buf, kCacheBuf, rowOff, kNorm.off, rowOff, nKVHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, kCacheBuf, kCacheBuf, rowOff, rowOff, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	// value: project STRAIGHT into the cache row (no rotation). gemma4 K==V layers carry
	// no v_proj — V is the k-proj output (pre-knorm/rope), so project via wK from the same
	// normed input (proj.hasV()==false); otherwise the dedicated v_proj.
	vIdx := projV
	if !proj.hasV() {
		vIdx = projK
	}
	if err := proj.project(enc, sc.normed, vCacheBuf, rowOff, vIdx); err != nil {
		return err
	}
	// gemma4 value RMSNorm — a no-scale per-head RMSNorm on V (metal's RMSNormNoScale),
	// expressed with a ones weight through the proven rows kernel. valueNorm is nil for
	// non-gemma4 paths (Mistral, the generic step helpers) ⇒ skipped, byte-identical.
	if valueNorm != nil {
		if err := encRMSNormRowsBF16(enc, vCacheBuf, valueNorm, vCacheBuf, rowOff, 0, rowOff, nKVHeads, headDim, eps); err != nil {
			return err
		}
	}
	// attend the n live rows from offset 0 — the whole seq-major cache (global) or the whole ring
	// (sliding). n + the ring write above replace the old seq-major slideWindow(pos, slideW).
	if err := encSDPADecode(enc, sc, sc.q, kCacheBuf, vCacheBuf, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, 0); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	// h = x + Wo·attn  (gemma4: post-attention norm on Wo·attn first; sc.normed is free)
	return encResidualMaybeNormAt(enc, x, xOff, sc.attnOut, 0, sc.normed, h, hOff, postAttnNorm, dModel, eps)
}

// encMLPHalfBF16 encodes the gemma MLP half — rms, gate/up projections, the tanh
// gelu approximation, gate·up, down projection, residual — into enc, exactly as
// DecodeLayer's MLP half. Reads h, writes h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h)))
// -> out.
func encMLPHalfBF16(
	enc metal.MTLComputeCommandEncoder,
	h, out metal.MTLBuffer, mlpNormW, postFFNorm bufView,
	sc mlpScratch, proj projector,
	dModel, dFF int, eps float32,
) error {
	return encMLPHalfBF16At(enc, h, out, 0, mlpNormW, postFFNorm, sc, proj, dModel, dFF, eps)
}

func encMLPHalfBF16At(
	enc metal.MTLComputeCommandEncoder,
	h, out metal.MTLBuffer, outOff uint, mlpNormW, postFFNorm bufView,
	sc mlpScratch, proj projector,
	dModel, dFF int, eps float32,
) error {
	if err := encRMSNormBF16(enc, h, mlpNormW.buf, sc.mlpNormed, mlpNormW.off, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.mlpNormed, sc.gate, 0, projGate); err != nil {
		return err
	}
	if err := proj.project(enc, sc.mlpNormed, sc.up, 0, projUp); err != nil {
		return err
	}
	// gelu(gate)·up — fused kernel (1 dispatch, fp32-internal) when loaded, composed bf16 chain otherwise
	if gpuHasGeluKernel() {
		if err := encGeluGateMulFused(enc, sc.gate, sc.up, sc.gated, dFF); err != nil {
			return err
		}
	} else {
		_ = encMulBF16(enc, sc.gate, sc.gate, sc.x2, dFF)
		_ = encMulBF16(enc, sc.x2, sc.gate, sc.x3, dFF)
		_ = encMulBF16(enc, sc.x3, sc.c044, sc.x3s, dFF)
		_ = encAddBF16(enc, sc.gate, sc.x3s, sc.inner, dFF)
		_ = encMulBF16(enc, sc.inner, sc.c079, sc.scaled, dFF)
		_ = encTanhBF16(enc, sc.scaled, sc.tnh, dFF)
		_ = encAddBF16(enc, sc.tnh, sc.c1, sc.onePlus, dFF)
		_ = encMulBF16(enc, sc.gate, sc.c05, sc.halfG, dFF)
		_ = encMulBF16(enc, sc.halfG, sc.onePlus, sc.gelu, dFF)
		_ = encMulBF16(enc, sc.gelu, sc.up, sc.gated, dFF)
	}
	if err := proj.project(enc, sc.gated, sc.down, 0, projDown); err != nil {
		return err
	}
	// out = h + Wdown·…  (gemma4: post-feed-forward norm on Wdown·… first; sc.mlpNormed is free)
	return encResidualMaybeNormAt(enc, h, 0, sc.down, 0, sc.mlpNormed, out, outOff, postFFNorm, dModel, eps)
}

// validateStepKV checks the shared shape contract for the KV-cache decode entries.
func validateStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, maxLen, pos int) error {
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return core.NewError("native.DecodeStepKV: nHeads must be a multiple of nKVHeads")
	}
	if pos < 0 || pos >= maxLen {
		return core.NewError("native.DecodeStepKV: pos out of [0,maxLen)")
	}
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size {
		return core.NewError("native.DecodeStepKV: x/attnNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return core.NewError("native.DecodeStepKV: wQ/wO size mismatch")
	}
	if len(wK) != kvDim*dModel*bf16Size || len(wV) != kvDim*dModel*bf16Size {
		return core.NewError("native.DecodeStepKV: wK/wV size mismatch")
	}
	if len(kCache) != maxLen*kvDim*bf16Size || len(vCache) != maxLen*kvDim*bf16Size {
		return core.NewError("native.DecodeStepKV: kCache/vCache must be maxLen*nKVHeads*headDim bf16 bytes")
	}
	return nil
}

// AttentionStepKV runs the attention half of one REAL decode step: it projects
// q/k/v from x, RoPEs q and the new k, appends k,v to the seq-major caches at row
// pos, attends over rows [0..pos], and returns x + Wo·attn. kCache/vCache are
// updated in place (the caller's backing arrays grow by one row). This is the
// piece DecodeLayer's "cache-write half is a follow-up" referred to. All raw bf16.
func AttentionStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, maxLen, pos int, base, scale, eps float32) ([]byte, error) {
	return AttentionStepKVInto(nil, x, attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, pos, base, scale, eps)
}

// AttentionStepKVInto runs AttentionStepKV and writes into caller-owned bf16 output when possible.
func AttentionStepKVInto(out []byte, x, attnNormW, wQ, wK, wV, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, maxLen, pos int, base, scale, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if err := validateStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, pos); err != nil {
		return nil, err
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	outLen := dModel * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVBF16Scratch(dModel, dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(ioScratch)
		xBuf, hBuf, err := ioScratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := ioScratch.outputView(out); ok {
				hBuf = tmp
				directOut = true
			}
		}
		nwBuf := residentBytes(attnNormW)
		proj := bf16Projector{
			wQ: bufView{buf: residentBytes(wQ)}, wK: bufView{buf: residentBytes(wK)}, wV: bufView{buf: residentBytes(wV)}, wO: bufView{buf: residentBytes(wO)},
			dModel: dModel, qDim: qDim, kvDim: kvDim,
		}
		kvScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockKVScratch(kvScratch)
		var kBuf, vBuf metal.MTLBuffer
		directKV := false
		if callerOut {
			kBuf, vBuf, directKV, err = kvScratch.buffersNoCopy(kCache, vCache)
			if err != nil {
				encErr = err
				return
			}
		}
		if !directKV {
			kBuf, vBuf, err = kvScratch.buffers(kCache, vCache)
			if err != nil {
				encErr = err
				return
			}
		}
		offBuf := scalarI32(int32(pos))
		sc := getAttnScratch(dModel, qDim, kvDim, nHeads, 0)
		defer putAttnScratch(sc)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encAttnHalfKV(enc, xBuf, kBuf, vBuf, offBuf, hBuf, bufView{buf: nwBuf}, bufView{}, bufView{}, bufView{}, nil, *sc, proj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(hBuf.Contents()), len(out)))
		}
		if !directKV {
			// reflect the grown cache rows back to the caller's slices
			copy(kCache, unsafe.Slice((*byte)(kBuf.Contents()), len(kCache)))
			copy(vCache, unsafe.Slice((*byte)(vBuf.Contents()), len(vCache)))
		}
	})
	return out, encErr
}

// DecodeStepKV runs one full REAL decode-layer step — the AttentionStepKV half
// then the gemma MLP half, both residuals — in one command buffer, growing the
// seq-major KV cache at row pos. kCache/vCache are updated in place. With the same
// inputs it equals AttentionStepKV fed through the MLP half; gated byte-for-byte
// against a reference built from the parity-proven ops. All raw bf16.
func DecodeStepKV(
	x, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, pos int,
	base, scale, eps float32,
) ([]byte, error) {
	return DecodeStepKVInto(nil, x, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, pos, base, scale, eps)
}

// DecodeStepKVInto runs DecodeStepKV and writes into caller-owned bf16 output when possible.
func DecodeStepKVInto(
	out []byte,
	x, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, pos int,
	base, scale, eps float32,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if err := validateStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, pos); err != nil {
		return nil, err
	}
	if len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeStepKV: mlpNormW must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeStepKV: MLP weight size mismatch")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	outLen := dModel * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVBF16Scratch(dModel, dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(ioScratch)
		xBuf, outBuf, err := ioScratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := ioScratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		nwBuf := residentBytes(attnNormW)
		proj := bf16Projector{
			wQ: bufView{buf: residentBytes(wQ)}, wK: bufView{buf: residentBytes(wK)}, wV: bufView{buf: residentBytes(wV)}, wO: bufView{buf: residentBytes(wO)},
			wGate: bufView{buf: residentBytes(wGate)}, wUp: bufView{buf: residentBytes(wUp)}, wDown: bufView{buf: residentBytes(wDown)},
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
		}
		kvScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockKVScratch(kvScratch)
		var kBuf, vBuf metal.MTLBuffer
		directKV := false
		if callerOut {
			kBuf, vBuf, directKV, err = kvScratch.buffersNoCopy(kCache, vCache)
			if err != nil {
				encErr = err
				return
			}
		}
		if !directKV {
			kBuf, vBuf, err = kvScratch.buffers(kCache, vCache)
			if err != nil {
				encErr = err
				return
			}
		}
		mnwBuf := residentBytes(mlpNormW)
		offBuf := scalarI32(int32(pos))
		asc := getAttnScratch(dModel, qDim, kvDim, nHeads, 0)
		defer putAttnScratch(asc)
		msc := getMLPScratch(dModel, dFF)
		defer putMLPScratch(msc)
		layerScratch := getDecodeLayerResidualScratch(dModel)
		defer putDecodeLayerResidualScratch(layerScratch)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encAttnHalfKV(enc, xBuf, kBuf, vBuf, offBuf, layerScratch.h, bufView{buf: nwBuf}, bufView{}, bufView{}, bufView{}, nil, *asc, proj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encMLPHalfBF16(enc, layerScratch.h, outBuf, bufView{buf: mnwBuf}, bufView{}, *msc, proj, dModel, dFF, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
		}
		if !directKV {
			copy(kCache, unsafe.Slice((*byte)(kBuf.Contents()), len(kCache)))
			copy(vCache, unsafe.Slice((*byte)(vBuf.Contents()), len(vCache)))
		}
	})
	return out, encErr
}
