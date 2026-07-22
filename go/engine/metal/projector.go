// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// A decode layer's seven matmuls — the Q/K/V/O attention projections and the
// gate/up/down MLP projections — are the ONLY ops that differ between a bf16-weight
// layer and a 4-bit-quantised one; rms, rope, sdpa and the gelu elementwise chain
// are identical. So the half-encoders (encAttnHalfKV / encMLPHalfBF16) and the
// whole forward are written once against the `projector` interface and run either
// way: bf16Projector drives the bf16 gemv, qmvProjector drives the bf16-activation
// 4-bit qmv. (This is also the projection seam pkg/model will extract.)

// encProjBias adds the additive projection bias into out[outOff:outOff+outDim] (bf16),
// after the projection matvec — Qwen2/2.5's q/k/v bias (bias=True on q_proj/k_proj/v_proj;
// o_proj and the MLP carry none). A zero bufView (a bias-free projection) is a no-op, so
// bias-free arches (llama/gemma/mistral/qwen3) pay only one nil check. outOff is a BYTE
// offset matching the matvec's own write (K/V project straight into their seq-major cache
// row), so the bias lands on exactly the elements the matvec just produced. The add reads
// then writes out on the same encoder as the matvec, hazard-ordered like the project→RoPE
// chain the caller already relies on.
func encProjBias(enc metal.MTLComputeCommandEncoder, bias bufView, out metal.MTLBuffer, outOff uint, outDim int) error {
	if bias.buf == nil {
		return nil
	}
	return encAddBF16To(enc, out, bias.buf, out, outOff, bias.off, outOff, outDim)
}

// encProjBiasRows broadcasts the additive projection bias over rows contiguous [outDim] rows
// at outOff — the batched prefill fold, one bf16 add per row (before RoPE). A zero bufView is
// a no-op. Prefill-only, so the per-row loop is off the per-token decode hot path.
func encProjBiasRows(enc metal.MTLComputeCommandEncoder, bias bufView, out metal.MTLBuffer, outOff uint, rows, outDim int) error {
	if bias.buf == nil {
		return nil
	}
	rowBytes := uint(outDim * bf16Size)
	for r := 0; r < rows; r++ {
		ro := outOff + uint(r)*rowBytes
		if err := encAddBF16To(enc, out, bias.buf, out, ro, bias.off, ro, outDim); err != nil {
			return err
		}
	}
	return nil
}

type projIndex int

const (
	projQ    projIndex = iota // dModel → nHeads·headDim
	projK                     // dModel → nKVHeads·headDim
	projV                     // dModel → nKVHeads·headDim
	projO                     // nHeads·headDim → dModel
	projGate                  // dModel → dFF
	projUp                    // dModel → dFF
	projDown                  // dFF → dModel

	// The per-layer-input (PLE) matmuls ride the SAME per-layer projection seam as the seven
	// decode projections, so the ICB record path has ONE polymorphism point (the recordProj
	// hook) for every per-layer matmul — bf16 gemv, 4-bit qmv and dense-in-quant all plug in
	// there. They are ICB-record-path values only: the projector interface below serves the
	// seven decode projections and is never called with these.
	projPLEGate // dModel → pliDim (per-layer-input gate)
	projPLEProj // pliDim → dModel (per-layer-input projection)
)

// projector encodes one projection — out[outOff:] = W_p · vec — into enc. outOff
// lets the V projection write straight into its seq-major KV-cache row. The
// per-projection dims are baked into the concrete projector at construction.
type projector interface {
	project(enc metal.MTLComputeCommandEncoder, vec, out metal.MTLBuffer, outOff uint, p projIndex) error
	// projectRows encodes the BATCHED projection — out rows [rows,outDim] = in rows
	// [rows,inDim] @ W_pᵀ, contiguous rows at byte offsets — reading the weight ONCE
	// for all rows (the prompt-prefill / K-row fold). Each implementation owns its
	// batched dispatch (bf16 → batched gemv, quant → MLX qmm_t at the weight's own
	// gs/bits), so the fold never names a concrete projector type: a new weight
	// scheme lands here once and every batched lane inherits it. ok=false means this
	// projector (or this weight's geometry) has no batched kernel — the caller keeps
	// its per-row path.
	projectRows(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) (bool, error)
	// projectRowsByteTier is projectRows on the BYTE tier — identical routing up
	// to lthnQMVRowsMaxM rows, the chunked tiled route past it (quant), so the
	// laneSet GEMM fold stays byte-identical to the per-lane replay at any K the
	// byte tier admits. Throughput callers (the MTP verify) keep projectRows.
	projectRowsByteTier(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) (bool, error)
	// rowsCapable reports whether EVERY present projection has a batched dispatch —
	// the fold's upfront eligibility probe (slab sizing happens before any encode).
	rowsCapable() bool
	// rowsByteTier reports whether projectRows reproduces the per-lane replay
	// projection BYTE for byte for every present weight — the laneSet GEMM fold's
	// eligibility tier. The bf16 batched gemv qualifies at any row count (z-slices
	// run the single-row tile loop unchanged); a quant weight qualifies when the
	// tiled plan admits it — the register-tiled lthn_qmv_rows is qmv_fast_impl's
	// M-variant, byte-identical on fast-twin dims (outDim%8==0 && inDim%512==0)
	// under the -fno-fast-math metallib. qmm_t (simdgroup-MMA) never qualifies.
	rowsByteTier(rows int) bool
	// foldProfitable reports whether the weight-read-once fold is a WALL win for
	// this projector, not just byte-correct: bf16/dense weights are weight-bound
	// (fold receipted 1.51× vs batched replay); 4-bit quant streams are 4× thinner
	// and the fold's re-encode tax outweighs the saving at K ≤ lthnQMVRowsMaxM
	// (live E2B K=4: fold ~114 vs replay ~118 tok/s). AUTO mode
	// consults this; forced mode (receipts) does not.
	foldProfitable() bool
	// hasV reports whether a distinct V projection weight exists. gemma4 K==V layers
	// (12B/31B: attention_k_eq_v) carry NO v_proj — V is the k-proj output (pre-knorm/
	// rope) value-normed — so the decode projects V via wK; hasV()==false signals that.
	hasV() bool
	// usesSiLU reports whether the feed-forward gate activation is SiLU/swish (llama/mistral/
	// qwen SwiGLU) rather than the gemma-family GELU. Set per model from Arch.Activation at
	// build; the MLP encode picks silu(gate)·up vs gelu(gate)·up. The zero value is GELU, so
	// every existing (gemma) projector stays byte-identical.
	usesSiLU() bool
	// withSiLU returns a copy of the projector with the feed-forward SiLU flag set — the session
	// constructor stamps every layer's projector with the model's activation once, off the shared
	// builder path (which stays activation-agnostic). A false argument is a no-op-equivalent copy.
	withSiLU(bool) projector
}

// ffnUsesSiLU maps a checkpoint's declared feed-forward activation to the gate the decode encodes:
// SiLU/swish (llama/mistral/qwen SwiGLU) vs the gemma-family GELU. An unset/unknown activation
// keeps GELU so the gemma path — the only one exercised before — stays byte-identical.
func ffnUsesSiLU(activation string) bool {
	return activation == "silu" || activation == "swish"
}

// ffnUsesClampedSwiGLU recognises gpt_oss's clamped-sigmoid SwiGLU marker — the deliberately
// non-standard string gptoss.Config.Arch declares so ffnUsesSiLU can NEVER mis-route it onto the
// plain (uncapped, unshifted) SiLU kernel (see model/arch/openai/gptoss/config.go). It selects the
// MoEExpertsQuantClampedSiLU expert path via MoEQuantLayerWeights.ClampedSwiGLU.
func ffnUsesClampedSwiGLU(activation string) bool {
	return activation == "gpt_oss_clamped_swiglu"
}

// bf16Projector drives a bf16 gemv per projection (the original weight path). Each weight is a
// bufView — a Metal buffer plus a byte offset — so the projection binds either an uploaded copy
// (off 0) or a no-copy view into a shared shard mmap at its offset, transparently.
type bf16Projector struct {
	wQ, wK, wV, wO, wGate, wUp, wDown bufView
	bQ, bK, bV                        bufView // Qwen2/2.5 additive q/k/v projection bias (zero bufView ⇒ none)
	bO                                bufView // gpt_oss additive o_proj bias (attention_bias=true reaches o_proj); zero bufView ⇒ none
	dModel, qDim, kvDim, dFF          int
	mlpUsesSiLU                       bool // SwiGLU (llama/mistral/qwen); zero ⇒ gemma GELU (byte-identical)
}

func (b bf16Projector) hasV() bool     { return b.wV.buf != nil }
func (b bf16Projector) usesSiLU() bool { return b.mlpUsesSiLU }
func (b bf16Projector) withSiLU(v bool) projector {
	b.mlpUsesSiLU = v
	return b
}

// biasView returns the additive projection bias for q/k/v/o (a zero bufView for every other
// projection — the MLP never carries one in the supported arches; o_proj's is gpt_oss's
// attention_bias=true, zero for every other arch so their dispatch stream is unchanged).
func (b bf16Projector) biasView(p projIndex) bufView {
	switch p {
	case projQ:
		return b.bQ
	case projK:
		return b.bK
	case projV:
		return b.bV
	case projO:
		return b.bO
	}
	return bufView{}
}

// foldProfitable: bf16 decode is weight-bandwidth-bound — reading each weight
// once for K lanes is the receipted 1.51× win.
func (b bf16Projector) foldProfitable() bool { return true }

// projectRowsByteTier: the batched bf16 gemv is byte-identical at any row
// count — the byte tier IS the plain route.
func (b bf16Projector) projectRowsByteTier(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) (bool, error) {
	return b.projectRows(enc, in, out, inOff, outOff, rows, p)
}

func (b bf16Projector) weightDims(p projIndex) (bufView, int, int, bool) {
	switch p {
	case projQ:
		return b.wQ, b.qDim, b.dModel, true
	case projK:
		return b.wK, b.kvDim, b.dModel, true
	case projV:
		return b.wV, b.kvDim, b.dModel, true
	case projO:
		return b.wO, b.dModel, b.qDim, true
	case projGate:
		return b.wGate, b.dFF, b.dModel, true
	case projUp:
		return b.wUp, b.dFF, b.dModel, true
	case projDown:
		return b.wDown, b.dModel, b.dFF, true
	}
	return bufView{}, 0, 0, false
}

func (b bf16Projector) projectRows(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) (bool, error) {
	w, outDim, inDim, ok := b.weightDims(p)
	if !ok {
		return false, core.NewError("native: bad projIndex")
	}
	if w.buf == nil {
		return false, nil
	}
	if err := encGemvBF16BatchedAt(enc, w.buf, in, out, w.off, inOff, outOff, outDim, inDim, rows); err != nil {
		return true, err
	}
	return true, encProjBiasRows(enc, b.biasView(p), out, outOff, rows, outDim)
}

func (b bf16Projector) rowsCapable() bool { return true } // batched gemv covers every dim

// rowsByteTier: the batched gemv's z-slices run the single-row tile loop
// unchanged, so every row count is byte-identical to the per-row decode gemv.
func (b bf16Projector) rowsByteTier(int) bool { return true }

func (b bf16Projector) project(enc metal.MTLComputeCommandEncoder, vec, out metal.MTLBuffer, outOff uint, p projIndex) error {
	w, outDim, inDim, ok := b.weightDims(p)
	if !ok {
		return core.NewError("native: bad projIndex")
	}
	if err := encGemvBF16To(enc, w.buf, vec, out, w.off, outOff, outDim, inDim); err != nil {
		return err
	}
	return encProjBias(enc, b.biasView(p), out, outOff, outDim)
}

// qmvWeight is one affine-quantised projection weight: packed codes + bf16 scales + bf16
// biases (MLX's quantiser output), each a bufView (buffer + offset) so the triple can be bound
// as no-copy views into the shard mmap(s) — the three tensors may sit in different shards. gs/bits
// are the weight's OWN affine geometry (mixed-precision packs vary it per weight); 0 ⇒ the
// projector's layer-default groupSize/bits.
type qmvWeight struct {
	wq, scales, biases bufView
	gs, bits           int
}

func (w qmvWeight) present() bool { return w.wq.buf != nil }

func (w qmvWeight) dense() bool { return w.present() && w.scales.buf == nil && w.biases.buf == nil }

// qmvProjector drives a bf16-activation 4-bit qmv per projection.
type qmvProjector struct {
	q, k, v, o, gate, up, down qmvWeight
	bQ, bK, bV                 bufView // Qwen2/2.5 additive q/k/v bias (plain bf16; zero bufView ⇒ none)
	bO                         bufView // gpt_oss additive o_proj bias (plain bf16; zero bufView ⇒ none)
	dModel, qDim, kvDim, dFF   int
	groupSize, bits            int
	mlpUsesSiLU                bool // SwiGLU (llama/mistral/qwen); zero ⇒ gemma GELU (byte-identical)
}

func (m qmvProjector) hasV() bool     { return m.v.present() }
func (m qmvProjector) usesSiLU() bool { return m.mlpUsesSiLU }
func (m qmvProjector) withSiLU(v bool) projector {
	m.mlpUsesSiLU = v
	return m
}

// biasView returns the additive projection bias for q/k/v/o (a zero bufView otherwise) — the
// same bias seam as the bf16 path; the affine pack still ships each as a plain bf16 vector.
// o_proj's bias is gpt_oss's attention_bias=true; zero for every other arch (byte-identical).
func (m qmvProjector) biasView(p projIndex) bufView {
	switch p {
	case projQ:
		return m.bQ
	case projK:
		return m.bK
	case projV:
		return m.bV
	case projO:
		return m.bO
	}
	return bufView{}
}

func (m qmvProjector) weightDims(p projIndex) (qmvWeight, int, int, bool) {
	switch p {
	case projQ:
		return m.q, m.qDim, m.dModel, true
	case projK:
		return m.k, m.kvDim, m.dModel, true
	case projV:
		return m.v, m.kvDim, m.dModel, true
	case projO:
		return m.o, m.dModel, m.qDim, true
	case projGate:
		return m.gate, m.dFF, m.dModel, true
	case projUp:
		return m.up, m.dFF, m.dModel, true
	case projDown:
		return m.down, m.dModel, m.dFF, true
	}
	return qmvWeight{}, 0, 0, false
}

func (m qmvProjector) projectRows(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) (bool, error) {
	w, outDim, inDim, ok := m.weightDims(p)
	if !ok {
		return false, core.NewError("native: bad projIndex")
	}
	if !w.present() {
		return false, nil
	}
	if w.dense() { // mixed packs carry the odd bf16 weight: batch it as a plain gemv
		if err := encGemvBF16BatchedAt(enc, w.wq.buf, in, out, w.wq.off, inOff, outOff, outDim, inDim, rows); err != nil {
			return true, err
		}
		return true, encProjBiasRows(enc, m.biasView(p), out, outOff, rows, outDim)
	}
	gs, bits := m.groupSize, m.bits
	if w.bits > 0 {
		gs, bits = w.gs, w.bits
	}
	// Small row counts (the MTP verify's draft blocks) take the multi-row qmv:
	// weight streamed once at qmv occupancy. On the tiled plan (fast-twin dims,
	// outDim%8 && inDim%512) each row's bytes are identical to the per-row
	// decode qmv; the gather fallback is throughput-tier only (the qmm_t below
	// reads the weight once too, but at small-M GEMM occupancy — ~5× off the
	// qmv floor on dense 12B/31B verifies).
	if handled, err := encQMVRowsBF16At(enc, w.wq.buf, w.scales.buf, w.biases.buf, in, out, w.wq.off, w.scales.off, w.biases.off, inOff, outOff, rows, outDim, inDim, gs, bits); handled || err != nil {
		if err != nil {
			return handled, err
		}
		return true, encProjBiasRows(enc, m.biasView(p), out, outOff, rows, outDim)
	}
	// MLX's qmm_t needs K%group_size==0 (whole groups per row); anything else keeps per-row.
	if inDim <= 0 || gs <= 0 || inDim%gs != 0 {
		return false, nil
	}
	if _, perr := pipelineFor(qmmTKernelName(outDim, gs, bits)); perr != nil {
		return false, nil // older metallib without this gs/bits variant — per-row fallback
	}
	if err := encQMMTBF16At(enc, w.wq.buf, w.scales.buf, w.biases.buf, in, out, w.wq.off, w.scales.off, w.biases.off, inOff, outOff, rows, outDim, inDim, gs, bits); err != nil {
		return true, err
	}
	return true, encProjBiasRows(enc, m.biasView(p), out, outOff, rows, outDim)
}

func (m qmvProjector) rowsCapable() bool {
	for p := projQ; p <= projDown; p++ {
		w, outDim, inDim, _ := m.weightDims(p)
		if !w.present() {
			continue // absent V (K==V layers) / MoE-owned MLP weights: not this fold's problem
		}
		if w.dense() {
			continue // batched gemv covers the odd bf16 weight in a mixed pack
		}
		gs, bits := m.groupSize, m.bits
		if w.bits > 0 {
			gs, bits = w.gs, w.bits
		}
		if inDim <= 0 || gs <= 0 || inDim%gs != 0 {
			return false
		}
		if _, err := pipelineFor(qmmTKernelName(outDim, gs, bits)); err != nil {
			return false
		}
	}
	return true
}

// rowsByteTier: the bf16 batched gemv reproduces the per-lane replay byte for
// byte at any row count. A quantised weight qualifies only when EVERY present
// quant weight routes to the register-tiled lthn_qmv_rows — qmv_fast_impl's
// M-variant, byte-identical to the per-row decode qmv precisely where the
// per-row oracle itself routes fast (outDim%8==0 && inDim%512==0; the plan
// gate carries the rule). Any weight the plan would send to the gather or
// qmm_t fallback declines the byte tier and the fold keeps the byte-identical
// merged ICB replay. History: the packs=1 predecessor claimed qmv_impl parity
// for ALL 256-multiples and was refuted (value-dependent ~1 ulp
// accumulation drift — surfaced as the hd-256 fold failure at step 6); the
// fast-twin match + this per-weight plan check replaced the blanket decline
// that closed that exposure.
// foldProfitable: a genuinely-quantised weight keeps the replay in AUTO mode —
// the 4-bit stream is 4× thinner than bf16, so weight-read-once saves little
// while the fold pays the re-encode tax the recorded replay avoids (live E2B
// K=4: fold ~114 vs replay ~118 tok/s aggregate). An all-dense
// mixed pack is bf16 in practice and keeps the win. Revisit at K>4 tiled-M.
func (m qmvProjector) foldProfitable() bool {
	for p := projQ; p <= projDown; p++ {
		w, _, _, _ := m.weightDims(p)
		if !w.present() {
			continue
		}
		if !w.dense() {
			return false
		}
	}
	return true
}

func (m qmvProjector) rowsByteTier(rows int) bool {
	for p := projQ; p <= projDown; p++ {
		w, outDim, inDim, _ := m.weightDims(p)
		if !w.present() {
			continue
		}
		if w.dense() {
			continue // batched gemv: byte-identical at any row count
		}
		gs, bits := m.groupSize, m.bits
		if w.bits > 0 {
			gs, bits = w.gs, w.bits
		}
		if rows > qmvRowsTiledCap() {
			// The chunked byte-tier route (projectRowsByteTier): every chunk size
			// must resolve a tiled PSO on the fast-twin envelope.
			if rows > qmvRowsMax || outDim%8 != 0 || inDim%512 != 0 || !qmvChunksEnabled() {
				return false
			}
			for _, cm := range qmvRowsChunks(rows) {
				if _, ok := lthnQMVRowsPipeline(lthnQMVRowsKey{groupSize: gs, bits: bits, m: cm}); !ok {
					return false
				}
			}
			continue
		}
		plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits)
		if !ok || !plan.tiled {
			return false // gather/qmm_t route: throughput tier, not byte tier
		}
	}
	return true
}

// projectRowsByteTier is projectRows on the BYTE tier: identical routing for
// rows ≤ lthnQMVRowsMaxM, and the chunked tiled route (encQMVRowsBF16ChunkedAt)
// past it — the laneSet GEMM fold's projection entry, where byte identity with
// the per-lane replay outranks the gather's throughput. Callers gate on
// rowsByteTier first; a false return here means the byte route fell through.
func (m qmvProjector) projectRowsByteTier(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) (bool, error) {
	if rows <= qmvRowsTiledCap() {
		return m.projectRows(enc, in, out, inOff, outOff, rows, p)
	}
	w, outDim, inDim, ok := m.weightDims(p)
	if !ok {
		return false, core.NewError("native: bad projIndex")
	}
	if !w.present() {
		return false, nil
	}
	if w.dense() {
		if err := encGemvBF16BatchedAt(enc, w.wq.buf, in, out, w.wq.off, inOff, outOff, outDim, inDim, rows); err != nil {
			return true, err
		}
		return true, encProjBiasRows(enc, m.biasView(p), out, outOff, rows, outDim)
	}
	gs, bits := m.groupSize, m.bits
	if w.bits > 0 {
		gs, bits = w.gs, w.bits
	}
	handled, err := encQMVRowsBF16ChunkedAt(enc, w.wq.buf, w.scales.buf, w.biases.buf, in, out, w.wq.off, w.scales.off, w.biases.off, inOff, outOff, rows, outDim, inDim, gs, bits)
	if !handled || err != nil {
		return handled, err
	}
	return true, encProjBiasRows(enc, m.biasView(p), out, outOff, rows, outDim)
}

func (m qmvProjector) project(enc metal.MTLComputeCommandEncoder, vec, out metal.MTLBuffer, outOff uint, p projIndex) error {
	var w qmvWeight
	var outDim, inDim int
	switch p {
	case projQ:
		w, outDim, inDim = m.q, m.qDim, m.dModel
	case projK:
		w, outDim, inDim = m.k, m.kvDim, m.dModel
	case projV:
		w, outDim, inDim = m.v, m.kvDim, m.dModel
	case projO:
		w, outDim, inDim = m.o, m.dModel, m.qDim
	case projGate:
		w, outDim, inDim = m.gate, m.dFF, m.dModel
	case projUp:
		w, outDim, inDim = m.up, m.dFF, m.dModel
	case projDown:
		w, outDim, inDim = m.down, m.dModel, m.dFF
	default:
		return core.NewError("native: bad projIndex")
	}
	if w.dense() {
		if err := encGemvBF16To(enc, w.wq.buf, vec, out, w.wq.off, outOff, outDim, inDim); err != nil {
			return err
		}
		return encProjBias(enc, m.biasView(p), out, outOff, outDim)
	}
	gs, bits := m.groupSize, m.bits // per-weight geometry (mixed-precision packs); fall back to the layer default
	if w.bits > 0 {
		gs, bits = w.gs, w.bits
	}
	if err := encQMVBF16(enc, w.wq.buf, w.scales.buf, w.biases.buf, vec, out, w.wq.off, w.scales.off, w.biases.off, outOff, outDim, inDim, gs, bits); err != nil {
		return err
	}
	return encProjBias(enc, m.biasView(p), out, outOff, outDim)
}
