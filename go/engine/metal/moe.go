// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// scalarFillBF16 returns an n-element bf16 buffer with every element set to the
// single bf16 value in val (2 bytes) — used to broadcast a router weight across a
// column for the weighted expert combine.
func scalarFillBF16(val []byte, n int) []byte {
	out := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		out[i*bf16Size] = val[0]
		out[i*bf16Size+1] = val[1]
	}
	return out
}

// encGeluGateMul encodes the tanh-approx SwiGLU activation gelu(gate)·up into enc —
// the same inline chain as encMLPHalfBF16, factored so the MoE experts reuse it.
// Reads gate/up, writes out; sc supplies the gelu scratch + constant buffers.
func encGeluGateMul(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, sc mlpScratch, dFF int) error {
	if gpuHasGeluKernel() { // fused kernel (1 dispatch, fp32-internal) when loaded, composed bf16 chain otherwise
		return encGeluGateMulFused(enc, gate, up, out, dFF)
	}
	_ = encMulBF16(enc, gate, gate, sc.x2, dFF)
	_ = encMulBF16(enc, sc.x2, gate, sc.x3, dFF)
	_ = encMulBF16(enc, sc.x3, sc.c044, sc.x3s, dFF)
	_ = encAddBF16(enc, gate, sc.x3s, sc.inner, dFF)
	_ = encMulBF16(enc, sc.inner, sc.c079, sc.scaled, dFF)
	_ = encTanhBF16(enc, sc.scaled, sc.tnh, dFF)
	_ = encAddBF16(enc, sc.tnh, sc.c1, sc.onePlus, dFF)
	_ = encMulBF16(enc, gate, sc.c05, sc.halfG, dFF)
	_ = encMulBF16(enc, sc.halfG, sc.onePlus, sc.gelu, dFF)
	_ = encMulBF16(enc, sc.gelu, up, out, dFF)
	return nil
}

type moeExpertsScratch struct {
	dModel, dFF, topK int
	x, weights        *pinnedNoCopyBytes
	xPinned           *pinnedNoCopyBytes
	weightsPinned     *pinnedNoCopyBytes
	outPinned         *pinnedNoCopyBytes
	mlp               mlpScratch
	scaled, acc       metal.MTLBuffer
}

type moeExpertsScratchKey struct {
	dModel, dFF, topK int
}

var moeExpertsScratchPools sync.Map

type moeExpertsScratchPool struct {
	core.Pool[*moeExpertsScratch]
}

func newMoEExpertsScratch(dModel, dFF, topK int) (*moeExpertsScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	weights, err := newPinnedNoCopyBytes(topK * bf16Size)
	if err != nil {
		x.Close()
		return nil, err
	}
	return &moeExpertsScratch{
		dModel:  dModel,
		dFF:     dFF,
		topK:    topK,
		x:       x,
		weights: weights,
		mlp:     newMLPScratch(dModel, dFF),
		scaled:  scratchBF16(dModel),
		acc:     scratchBF16(dModel),
	}, nil
}

func moeExpertsScratchPoolFor(dModel, dFF, topK int) *moeExpertsScratchPool {
	key := moeExpertsScratchKey{dModel: dModel, dFF: dFF, topK: topK}
	if v, ok := moeExpertsScratchPools.Load(key); ok {
		return v.(*moeExpertsScratchPool)
	}
	pool := &moeExpertsScratchPool{}
	if v, loaded := moeExpertsScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*moeExpertsScratchPool)
	}
	return pool
}

func getMoEExpertsScratch(dModel, dFF, topK int) (*moeExpertsScratch, error) {
	pool := moeExpertsScratchPoolFor(dModel, dFF, topK)
	if s := pool.Get(); s != nil {
		if s != nil &&
			s.dModel == dModel &&
			s.dFF == dFF &&
			s.topK == topK &&
			s.x != nil &&
			s.x.buf != nil &&
			s.weights != nil &&
			s.weights.buf != nil &&
			s.mlp.gate != nil &&
			s.mlp.up != nil &&
			s.mlp.gated != nil &&
			s.mlp.down != nil &&
			s.scaled != nil &&
			s.acc != nil {
			return s, nil
		}
		s.Close()
	}
	return newMoEExpertsScratch(dModel, dFF, topK)
}

func putMoEExpertsScratch(s *moeExpertsScratch) {
	if s != nil && s.x != nil && s.x.buf != nil && s.weights != nil && s.weights.buf != nil && s.mlp.gate != nil && s.mlp.up != nil && s.mlp.gated != nil && s.mlp.down != nil && s.scaled != nil && s.acc != nil {
		moeExpertsScratchPoolFor(s.dModel, s.dFF, s.topK).Put(s)
	}
}

func (s *moeExpertsScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	if s.xPinned != nil {
		s.xPinned.Close()
		s.xPinned = nil
	}
	if s.weights != nil {
		s.weights.Close()
		s.weights = nil
	}
	if s.weightsPinned != nil {
		s.weightsPinned.Close()
		s.weightsPinned = nil
	}
	if s.outPinned != nil {
		s.outPinned.Close()
		s.outPinned = nil
	}
}

func (s *moeExpertsScratch) inputView(x []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(x) == 0 {
		return nil, false
	}
	if s.xPinned != nil && len(s.xPinned.bytes) == len(x) && &s.xPinned.bytes[0] == &x[0] {
		return s.xPinned.buf, true
	}
	if s.xPinned != nil {
		s.xPinned.Close()
		s.xPinned = nil
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
	s.xPinned = pinned
	return buf, true
}

func (s *moeExpertsScratch) weightsView(weights []byte) (metal.MTLBuffer, bool) {
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

func (s *moeExpertsScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
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

// MoEExperts runs the expert branch of a gemma4 MoE layer: for each of the topK
// selected experts (idx) it runs that expert's SwiGLU MLP on x and accumulates the
// router-weighted result —  out = Σ_i weights[i] · Wdown_e( gelu(Wgate_e·x)·(Wup_e·x) ).
// Given the routing decision (idx, weights from the router); the routing itself is a
// separate sub-slice. It binds each batched expert tensor once and addresses selected
// experts by byte offset, matching the no-copy residency shape used by loader-backed
// weights without creating one retained Metal buffer per selected expert slice.
// gateW/upW are [numExperts × dFF × dModel] row-major bf16, downW is
// [numExperts × dModel × dFF]; x is dModel bf16, idx topK int32, weights topK bf16.
// Byte-for-byte against a composed reference of the parity-proven ops in the tests.
func MoEExperts(x []byte, idx []int32, weights, gateW, upW, downW []byte, numExperts, topK, dModel, dFF int) ([]byte, error) {
	return moeExpertsInto(nil, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF, false)
}

func MoEExpertsInto(out []byte, x []byte, idx []int32, weights, gateW, upW, downW []byte, numExperts, topK, dModel, dFF int) ([]byte, error) {
	return moeExpertsInto(out, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF, true)
}

func moeExpertsInto(out []byte, x []byte, idx []int32, weights, gateW, upW, downW []byte, numExperts, topK, dModel, dFF int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	gateSz, downSz := dFF*dModel*bf16Size, dModel*dFF*bf16Size
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.MoEExperts: x must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.MoEExperts: idx/weights length must equal topK")
	}
	if len(gateW) != numExperts*gateSz || len(upW) != numExperts*gateSz || len(downW) != numExperts*downSz {
		return nil, core.NewError("native.MoEExperts: expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.MoEExperts: expert index out of range")
		}
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if topK == 0 {
		clear(out)
		return out, nil
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEExpertsScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		weightsBuf, ok := scratch.weightsView(weights)
		if !ok {
			weightsBuf, err = scratch.weights.copyBuffer(weights)
			if err != nil {
				encErr = err
				return
			}
		}
		msc := scratch.mlp
		downE, scaled, acc := msc.down, scratch.scaled, scratch.acc
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				acc = tmp
				directOut = true
			}
		}
		gateBuf, upBuf, downBuf := residentBytes(gateW), residentBytes(upW), residentBytes(downW)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for i := 0; i < topK; i++ {
			e := int(idx[i])
			gateOff, downOff := uint(e*gateSz), uint(e*downSz)
			if encErr = encGemvBF16To(enc, gateBuf, xBuf, msc.gate, gateOff, 0, dFF, dModel); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encGemvBF16To(enc, upBuf, xBuf, msc.up, gateOff, 0, dFF, dModel)
			if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encGemvBF16To(enc, downBuf, msc.gated, downE, downOff, 0, dModel, dFF)
			if i == 0 {
				if encErr = encScaleBF16(enc, downE, weightsBuf, acc, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = encScaleBF16(enc, downE, weightsBuf, scaled, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				_ = encAddBF16(enc, acc, scaled, acc, dModel) // acc += wi·downi
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(scratch.acc.Contents()), len(out)))
		}
	})
	return out, encErr
}

// MoEExpertsQuant is MoEExperts for 4-bit experts: the gemma4 26B-A4B SwitchGLU stores all
// experts batched (experts.switch_glu.{gate,up,down}_proj as [numExperts × out × in] affine-
// quant tensors), so gate/up/down are QuantWeights whose Packed/Scales/Biases hold every
// expert's slice. For each of the topK selected experts it runs the SwiGLU via QMVBF16
// (gate/up: dModel→dFF, down: dFF→dModel) and accumulates weights[i]·downᵢ — the quant sibling
// of MoEExperts, encQMVBF16 in place of encGemvBF16. groupSize/bits are the checkpoint's quant.
func MoEExpertsQuant(x []byte, idx []int32, weights []byte, gate, up, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return moeExpertsQuantInto(nil, x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits, false)
}

func MoEExpertsQuantInto(out []byte, x []byte, idx []int32, weights []byte, gate, up, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return moeExpertsQuantInto(out, x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits, true)
}

func moeExpertsQuantInto(out []byte, x []byte, idx []int32, weights []byte, gate, up, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuant: x must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuant: idx/weights length must equal topK")
	}
	if dModel%groupSize != 0 || dFF%groupSize != 0 {
		return nil, core.NewError("native.MoEExpertsQuant: dModel and dFF must be multiples of groupSize")
	}
	gatePacked, gateScale := dFF*dModel*bits/8, dFF*(dModel/groupSize)*bf16Size // per expert (gate, up)
	downPacked, downScale := dModel*dFF*bits/8, dModel*(dFF/groupSize)*bf16Size // per expert (down)
	if len(gate.Packed) != numExperts*gatePacked || len(up.Packed) != numExperts*gatePacked || len(down.Packed) != numExperts*downPacked ||
		len(gate.Scales) != numExperts*gateScale || len(up.Scales) != numExperts*gateScale || len(down.Scales) != numExperts*downScale ||
		len(gate.Biases) != numExperts*gateScale || len(up.Biases) != numExperts*gateScale || len(down.Biases) != numExperts*downScale {
		return nil, core.NewError("native.MoEExpertsQuant: batched expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.MoEExpertsQuant: expert index out of range")
		}
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if topK == 0 {
		clear(out)
		return out, nil
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEExpertsScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		weightsBuf, ok := scratch.weightsView(weights)
		if !ok {
			weightsBuf, err = scratch.weights.copyBuffer(weights)
			if err != nil {
				encErr = err
				return
			}
		}
		msc := scratch.mlp
		downE, scaled, acc := msc.down, scratch.scaled, scratch.acc
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				acc = tmp
				directOut = true
			}
		}
		// Bind each batched [numExperts x ...] expert tensor once and address selected experts by
		// qmv byte offsets. This preserves the resident/no-copy shape needed by loader-backed MoE
		// weights and avoids creating one retained Metal buffer per selected expert slice.
		gatePackedBuf, gateScalesBuf, gateBiasesBuf := quantWeightViews(gate)
		upPackedBuf, upScalesBuf, upBiasesBuf := quantWeightViews(up)
		downPackedBuf, downScalesBuf, downBiasesBuf := quantWeightViews(down)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for i := 0; i < topK; i++ {
			e := int(idx[i])
			gatePackedOff, gateScaleOff := uint(e*gatePacked), uint(e*gateScale)
			downPackedOff, downScaleOff := uint(e*downPacked), uint(e*downScale)
			if encErr = encQMVBF16(enc, gatePackedBuf.buf, gateScalesBuf.buf, gateBiasesBuf.buf, xBuf, msc.gate, gatePackedBuf.off+gatePackedOff, gateScalesBuf.off+gateScaleOff, gateBiasesBuf.off+gateScaleOff, 0, dFF, dModel, groupSize, bits); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encQMVBF16(enc, upPackedBuf.buf, upScalesBuf.buf, upBiasesBuf.buf, xBuf, msc.up, upPackedBuf.off+gatePackedOff, upScalesBuf.off+gateScaleOff, upBiasesBuf.off+gateScaleOff, 0, dFF, dModel, groupSize, bits)
			if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encQMVBF16(enc, downPackedBuf.buf, downScalesBuf.buf, downBiasesBuf.buf, msc.gated, downE, downPackedBuf.off+downPackedOff, downScalesBuf.off+downScaleOff, downBiasesBuf.off+downScaleOff, 0, dModel, dFF, groupSize, bits)
			if i == 0 {
				if encErr = encScaleBF16(enc, downE, weightsBuf, acc, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = encScaleBF16(enc, downE, weightsBuf, scaled, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				_ = encAddBF16(enc, acc, scaled, acc, dModel)
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(scratch.acc.Contents()), len(out)))
		}
	})
	return out, encErr
}

// MoEExpertsQuantFusedGateUp is MoEExpertsQuant for checkpoints that store
// experts.switch_glu.gate_up_proj as [numExperts x 2*dFF x dModel] instead of
// separate gate/up expert tensors. It keeps the fused tensor resident and addresses
// gate/up halves by byte offset, avoiding loader-time split copies.
func MoEExpertsQuantFusedGateUp(x []byte, idx []int32, weights []byte, gateUp, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return moeExpertsQuantFusedGateUpInto(nil, x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits, false)
}

func MoEExpertsQuantFusedGateUpInto(out []byte, x []byte, idx []int32, weights []byte, gateUp, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return moeExpertsQuantFusedGateUpInto(out, x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits, true)
}

func moeExpertsQuantFusedGateUpInto(out []byte, x []byte, idx []int32, weights []byte, gateUp, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantFusedGateUp: x must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantFusedGateUp: idx/weights length must equal topK")
	}
	if dModel%groupSize != 0 || dFF%groupSize != 0 {
		return nil, core.NewError("native.MoEExpertsQuantFusedGateUp: dModel and dFF must be multiples of groupSize")
	}
	gatePacked, gateScale := dFF*dModel*bits/8, dFF*(dModel/groupSize)*bf16Size
	downPacked, downScale := dModel*dFF*bits/8, dModel*(dFF/groupSize)*bf16Size
	if len(gateUp.Packed) != numExperts*2*gatePacked || len(down.Packed) != numExperts*downPacked ||
		len(gateUp.Scales) != numExperts*2*gateScale || len(down.Scales) != numExperts*downScale ||
		len(gateUp.Biases) != numExperts*2*gateScale || len(down.Biases) != numExperts*downScale {
		return nil, core.NewError("native.MoEExpertsQuantFusedGateUp: batched expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.MoEExpertsQuantFusedGateUp: expert index out of range")
		}
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if topK == 0 {
		clear(out)
		return out, nil
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEExpertsScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		weightsBuf, ok := scratch.weightsView(weights)
		if !ok {
			weightsBuf, err = scratch.weights.copyBuffer(weights)
			if err != nil {
				encErr = err
				return
			}
		}
		msc := scratch.mlp
		downE, scaled, acc := msc.down, scratch.scaled, scratch.acc
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				acc = tmp
				directOut = true
			}
		}
		gateUpPackedBuf, gateUpScalesBuf, gateUpBiasesBuf := quantWeightViews(gateUp)
		downPackedBuf, downScalesBuf, downBiasesBuf := quantWeightViews(down)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for i := 0; i < topK; i++ {
			e := int(idx[i])
			gatePackedOff, gateScaleOff := uint(e*2*gatePacked), uint(e*2*gateScale)
			upPackedOff, upScaleOff := gatePackedOff+uint(gatePacked), gateScaleOff+uint(gateScale)
			downPackedOff, downScaleOff := uint(e*downPacked), uint(e*downScale)
			if encErr = encQMVBF16(enc, gateUpPackedBuf.buf, gateUpScalesBuf.buf, gateUpBiasesBuf.buf, xBuf, msc.gate, gateUpPackedBuf.off+gatePackedOff, gateUpScalesBuf.off+gateScaleOff, gateUpBiasesBuf.off+gateScaleOff, 0, dFF, dModel, groupSize, bits); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encQMVBF16(enc, gateUpPackedBuf.buf, gateUpScalesBuf.buf, gateUpBiasesBuf.buf, xBuf, msc.up, gateUpPackedBuf.off+upPackedOff, gateUpScalesBuf.off+upScaleOff, gateUpBiasesBuf.off+upScaleOff, 0, dFF, dModel, groupSize, bits)
			if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encQMVBF16(enc, downPackedBuf.buf, downScalesBuf.buf, downBiasesBuf.buf, msc.gated, downE, downPackedBuf.off+downPackedOff, downScalesBuf.off+downScaleOff, downBiasesBuf.off+downScaleOff, 0, dModel, dFF, groupSize, bits)
			if i == 0 {
				if encErr = encScaleBF16(enc, downE, weightsBuf, acc, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = encScaleBF16(enc, downE, weightsBuf, scaled, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				_ = encAddBF16(enc, acc, scaled, acc, dModel)
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(scratch.acc.Contents()), len(out)))
		}
	})
	return out, encErr
}
