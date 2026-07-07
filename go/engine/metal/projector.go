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

type projIndex int

const (
	projQ    projIndex = iota // dModel → nHeads·headDim
	projK                     // dModel → nKVHeads·headDim
	projV                     // dModel → nKVHeads·headDim
	projO                     // nHeads·headDim → dModel
	projGate                  // dModel → dFF
	projUp                    // dModel → dFF
	projDown                  // dFF → dModel
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
	// hasV reports whether a distinct V projection weight exists. gemma4 K==V layers
	// (12B/31B: attention_k_eq_v) carry NO v_proj — V is the k-proj output (pre-knorm/
	// rope) value-normed — so the decode projects V via wK; hasV()==false signals that.
	hasV() bool
}

// bf16Projector drives a bf16 gemv per projection (the original weight path). Each weight is a
// bufView — a Metal buffer plus a byte offset — so the projection binds either an uploaded copy
// (off 0) or a no-copy view into a shared shard mmap at its offset, transparently.
type bf16Projector struct {
	wQ, wK, wV, wO, wGate, wUp, wDown bufView
	dModel, qDim, kvDim, dFF          int
}

func (b bf16Projector) hasV() bool { return b.wV.buf != nil }

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
	return true, encGemvBF16BatchedAt(enc, w.buf, in, out, w.off, inOff, outOff, outDim, inDim, rows)
}

func (b bf16Projector) project(enc metal.MTLComputeCommandEncoder, vec, out metal.MTLBuffer, outOff uint, p projIndex) error {
	switch p {
	case projQ:
		return encGemvBF16To(enc, b.wQ.buf, vec, out, b.wQ.off, outOff, b.qDim, b.dModel)
	case projK:
		return encGemvBF16To(enc, b.wK.buf, vec, out, b.wK.off, outOff, b.kvDim, b.dModel)
	case projV:
		return encGemvBF16To(enc, b.wV.buf, vec, out, b.wV.off, outOff, b.kvDim, b.dModel)
	case projO:
		return encGemvBF16To(enc, b.wO.buf, vec, out, b.wO.off, outOff, b.dModel, b.qDim)
	case projGate:
		return encGemvBF16To(enc, b.wGate.buf, vec, out, b.wGate.off, outOff, b.dFF, b.dModel)
	case projUp:
		return encGemvBF16To(enc, b.wUp.buf, vec, out, b.wUp.off, outOff, b.dFF, b.dModel)
	case projDown:
		return encGemvBF16To(enc, b.wDown.buf, vec, out, b.wDown.off, outOff, b.dModel, b.dFF)
	}
	return core.NewError("native: bad projIndex")
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
	dModel, qDim, kvDim, dFF   int
	groupSize, bits            int
}

func (m qmvProjector) hasV() bool { return m.v.present() }

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
		return true, encGemvBF16BatchedAt(enc, w.wq.buf, in, out, w.wq.off, inOff, outOff, outDim, inDim, rows)
	}
	gs, bits := m.groupSize, m.bits
	if w.bits > 0 {
		gs, bits = w.gs, w.bits
	}
	// MLX's qmm_t needs K%group_size==0 (whole groups per row); anything else keeps per-row.
	if inDim <= 0 || gs <= 0 || inDim%gs != 0 {
		return false, nil
	}
	if _, perr := pipelineFor(qmmTKernelName(outDim, gs, bits)); perr != nil {
		return false, nil // older metallib without this gs/bits variant — per-row fallback
	}
	return true, encQMMTBF16At(enc, w.wq.buf, w.scales.buf, w.biases.buf, in, out, w.wq.off, w.scales.off, w.biases.off, inOff, outOff, rows, outDim, inDim, gs, bits)
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
		return encGemvBF16To(enc, w.wq.buf, vec, out, w.wq.off, outOff, outDim, inDim)
	}
	gs, bits := m.groupSize, m.bits // per-weight geometry (mixed-precision packs); fall back to the layer default
	if w.bits > 0 {
		gs, bits = w.gs, w.bits
	}
	return encQMVBF16(enc, w.wq.buf, w.scales.buf, w.biases.buf, vec, out, w.wq.off, w.scales.off, w.biases.off, outOff, outDim, inDim, gs, bits)
}
