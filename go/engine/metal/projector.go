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
