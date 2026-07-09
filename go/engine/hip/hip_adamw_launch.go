// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
)

const (
	hipAdamWUpdateLaunchArgsVersion uint32 = 1
	hipAdamWUpdateLaunchArgsBytes          = 128
)

type hipAdamWUpdateRequest struct {
	State     *NativeAdamWState
	Gradients [][]float32
}

type hipAdamWUpdateDeviceBuffers struct {
	Parameters  hipDeviceByteBuffer
	MomentM     hipDeviceByteBuffer
	MomentV     hipDeviceByteBuffer
	Gradients   hipDeviceByteBuffer
	ParamCount  int
	TensorCount int
	Step        int
}

type hipAdamWUpdateLaunchArgs struct {
	ParameterPointer nativeDevicePointer
	MomentMPointer   nativeDevicePointer
	MomentVPointer   nativeDevicePointer
	GradientPointer  nativeDevicePointer
	ParamCount       int
	TensorCount      int
	Step             int
	ParameterBytes   uint64
	MomentBytes      uint64
	GradientBytes    uint64
	LearningRate     float64
	Beta1            float64
	Beta2            float64
	Eps              float64
	WeightDecay      float64
}

type hipAdamWPayloadPool struct {
	sync.Mutex
	payloads [][]byte
}

var hipAdamWPayloadPools sync.Map

const (
	hipAdamWPayloadPoolMaxBytes   = 2 << 20
	hipAdamWPayloadPoolMaxPerSize = 128
)

func (req hipAdamWUpdateRequest) validate() error {
	state := req.State
	if state == nil {
		return core.E("rocm.hip.AdamWUpdateLaunch", "AdamW state is required", nil)
	}
	if err := validateNativeAdamWConfig(state.Config); err != nil {
		return core.E("rocm.hip.AdamWUpdateLaunch", "AdamW config", err)
	}
	total := stateTotalLen(state)
	if total <= 0 || len(state.Slab) != total*3 {
		return core.E("rocm.hip.AdamWUpdateLaunch", "packed AdamW slab shape is invalid", nil)
	}
	if len(req.Gradients) != len(state.Layout) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "gradient count must match AdamW layout", nil)
	}
	for index, gradient := range req.Gradients {
		desc := state.Layout[index]
		if len(gradient) != desc.Length {
			return core.E("rocm.hip.AdamWUpdateLaunch", "gradient length must match parameter layout", nil)
		}
		if !rocmFloat32SliceFinite(gradient) {
			return core.E("rocm.hip.AdamWUpdateLaunch", "gradient values must be finite", nil)
		}
	}
	if state.Step < 0 {
		return core.E("rocm.hip.AdamWUpdateLaunch", "AdamW step must be non-negative", nil)
	}
	return nil
}

func (req hipAdamWUpdateRequest) deviceBuffers(driver nativeHIPDriver) (*hipAdamWUpdateDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffersValidatedValue(driver)
	if err != nil {
		return nil, err
	}
	return &buffers, nil
}

func (req hipAdamWUpdateRequest) deviceBuffersValidatedValue(driver nativeHIPDriver) (hipAdamWUpdateDeviceBuffers, error) {
	state := req.State
	total := stateTotalLen(state)
	payload := hipBorrowAdamWPayload(total * 4)
	defer hipReleaseAdamWPayload(payload)
	params, err := hipFloat32PayloadInto(payload, state.Parameters())
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, core.E("rocm.hip.AdamWUpdateLaunch", "encode parameters", err)
	}
	paramBuffer, err := hipAdamWUploadByteBufferValue(driver, "AdamW parameters", params, len(state.Parameters()))
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, err
	}
	buffers := hipAdamWUpdateDeviceBuffers{
		Parameters:  paramBuffer,
		ParamCount:  len(state.Parameters()),
		TensorCount: len(state.Layout),
		Step:        state.Step + 1,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	momentMPayload, err := hipFloat32PayloadInto(payload, state.FirstMoment())
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, core.E("rocm.hip.AdamWUpdateLaunch", "encode first moments", err)
	}
	momentM, err := hipAdamWUploadByteBufferValue(driver, "AdamW first moments", momentMPayload, len(state.FirstMoment()))
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, err
	}
	buffers.MomentM = momentM

	momentVPayload, err := hipFloat32PayloadInto(payload, state.SecondMoment())
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, core.E("rocm.hip.AdamWUpdateLaunch", "encode second moments", err)
	}
	momentV, err := hipAdamWUploadByteBufferValue(driver, "AdamW second moments", momentVPayload, len(state.SecondMoment()))
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, err
	}
	buffers.MomentV = momentV

	gradientPayload, err := hipAdamWGradientPayloadInto(payload, state, req.Gradients)
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, core.E("rocm.hip.AdamWUpdateLaunch", "encode gradients", err)
	}
	gradients, err := hipAdamWUploadByteBufferValue(driver, "AdamW gradients", gradientPayload, total)
	if err != nil {
		return hipAdamWUpdateDeviceBuffers{}, err
	}
	buffers.Gradients = gradients
	success = true
	return buffers, nil
}

func (req hipAdamWUpdateRequest) launchArgs(buffers *hipAdamWUpdateDeviceBuffers) (hipAdamWUpdateLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipAdamWUpdateLaunchArgs{}, err
	}
	return req.launchArgsValidated(buffers)
}

func (req hipAdamWUpdateRequest) launchArgsValidated(buffers *hipAdamWUpdateDeviceBuffers) (hipAdamWUpdateLaunchArgs, error) {
	if buffers == nil || buffers.Parameters.Pointer() == 0 || buffers.MomentM.Pointer() == 0 || buffers.MomentV.Pointer() == 0 || buffers.Gradients.Pointer() == 0 {
		return hipAdamWUpdateLaunchArgs{}, core.E("rocm.hip.AdamWUpdateLaunch", "AdamW device buffers are required", nil)
	}
	total := stateTotalLen(req.State)
	if buffers.ParamCount != total || buffers.TensorCount != len(req.State.Layout) || buffers.Step != req.State.Step+1 ||
		buffers.Parameters.Count() != total || buffers.MomentM.Count() != total ||
		buffers.MomentV.Count() != total || buffers.Gradients.Count() != total ||
		buffers.Parameters.SizeBytes() != uint64(total*4) ||
		buffers.MomentM.SizeBytes() != uint64(total*4) ||
		buffers.MomentV.SizeBytes() != uint64(total*4) ||
		buffers.Gradients.SizeBytes() != uint64(total*4) {
		return hipAdamWUpdateLaunchArgs{}, core.E("rocm.hip.AdamWUpdateLaunch", "AdamW device buffer shape mismatch", nil)
	}
	return hipAdamWUpdateLaunchArgs{
		ParameterPointer: buffers.Parameters.Pointer(),
		MomentMPointer:   buffers.MomentM.Pointer(),
		MomentVPointer:   buffers.MomentV.Pointer(),
		GradientPointer:  buffers.Gradients.Pointer(),
		ParamCount:       total,
		TensorCount:      len(req.State.Layout),
		Step:             req.State.Step + 1,
		ParameterBytes:   buffers.Parameters.SizeBytes(),
		MomentBytes:      buffers.MomentM.SizeBytes(),
		GradientBytes:    buffers.Gradients.SizeBytes(),
		LearningRate:     req.State.Config.LearningRate,
		Beta1:            req.State.Config.Beta1,
		Beta2:            req.State.Config.Beta2,
		Eps:              req.State.Config.Eps,
		WeightDecay:      req.State.Config.WeightDecay,
	}, nil
}

func (args hipAdamWUpdateLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipAdamWUpdateLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.ParameterPointer == 0 || args.MomentMPointer == 0 || args.MomentVPointer == 0 || args.GradientPointer == 0 {
		return nil, core.E("rocm.hip.AdamWUpdateLaunch", "parameter, moment, and gradient pointers are required", nil)
	}
	paramCount, err := rocmDeviceKVPositiveUint32("AdamW parameter count", args.ParamCount)
	if err != nil {
		return nil, err
	}
	tensorCount, err := rocmDeviceKVPositiveUint32("AdamW tensor count", args.TensorCount)
	if err != nil {
		return nil, err
	}
	step, err := rocmDeviceKVPositiveUint32("AdamW step", args.Step)
	if err != nil {
		return nil, err
	}
	parameterBytes, err := hipAlignedFloat32Bytes("AdamW parameters", args.ParameterBytes, paramCount)
	if err != nil {
		return nil, core.E("rocm.hip.AdamWUpdateLaunch", "parameter byte count", err)
	}
	momentBytes, err := hipAlignedFloat32Bytes("AdamW moments", args.MomentBytes, paramCount)
	if err != nil {
		return nil, core.E("rocm.hip.AdamWUpdateLaunch", "moment byte count", err)
	}
	gradientBytes, err := hipAlignedFloat32Bytes("AdamW gradients", args.GradientBytes, paramCount)
	if err != nil {
		return nil, core.E("rocm.hip.AdamWUpdateLaunch", "gradient byte count", err)
	}
	if err := validateHIPAdamWHyperparameters(args); err != nil {
		return nil, err
	}
	if cap(payload) < hipAdamWUpdateLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipAdamWUpdateLaunchArgsBytes)
	} else {
		payload = payload[:hipAdamWUpdateLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipAdamWUpdateLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.ParameterPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.MomentMPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.MomentVPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.GradientPointer))
	binary.LittleEndian.PutUint32(payload[40:], paramCount)
	binary.LittleEndian.PutUint32(payload[44:], tensorCount)
	binary.LittleEndian.PutUint32(payload[48:], step)
	binary.LittleEndian.PutUint32(payload[52:], parameterBytes)
	binary.LittleEndian.PutUint32(payload[56:], momentBytes)
	binary.LittleEndian.PutUint32(payload[60:], gradientBytes)
	binary.LittleEndian.PutUint64(payload[64:], math.Float64bits(args.LearningRate))
	binary.LittleEndian.PutUint64(payload[72:], math.Float64bits(args.Beta1))
	binary.LittleEndian.PutUint64(payload[80:], math.Float64bits(args.Beta2))
	binary.LittleEndian.PutUint64(payload[88:], math.Float64bits(args.Eps))
	binary.LittleEndian.PutUint64(payload[96:], math.Float64bits(args.WeightDecay))
	return payload, nil
}

func (buffers *hipAdamWUpdateDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{&buffers.Gradients, &buffers.MomentV, &buffers.MomentM, &buffers.Parameters} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipAdamWUpdateDeviceBuffers) ReadBack(state *NativeAdamWState) error {
	if buffers == nil || buffers.Parameters.Pointer() == 0 || buffers.MomentM.Pointer() == 0 || buffers.MomentV.Pointer() == 0 {
		return core.E("rocm.hip.AdamWUpdateLaunch", "AdamW result buffers are required", nil)
	}
	if state == nil {
		return core.E("rocm.hip.AdamWUpdateLaunch", "AdamW state is required", nil)
	}
	total := stateTotalLen(state)
	if total <= 0 || len(state.Slab) != total*3 ||
		buffers.ParamCount != total || buffers.Step != state.Step+1 ||
		buffers.Parameters.Count() != total || buffers.MomentM.Count() != total || buffers.MomentV.Count() != total ||
		buffers.Parameters.SizeBytes() != uint64(total*4) ||
		buffers.MomentM.SizeBytes() != uint64(total*4) ||
		buffers.MomentV.SizeBytes() != uint64(total*4) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "AdamW readback shape mismatch", nil)
	}
	params := state.Parameters()
	if err := buffers.Parameters.driver.CopyDeviceToHost(buffers.Parameters.Pointer(), hipAdamWFloat32Bytes(params)); err != nil {
		return core.E("rocm.hip.AdamWUpdateLaunch", "copy updated parameters", err)
	}
	momentsM := state.FirstMoment()
	if err := buffers.MomentM.driver.CopyDeviceToHost(buffers.MomentM.Pointer(), hipAdamWFloat32Bytes(momentsM)); err != nil {
		return core.E("rocm.hip.AdamWUpdateLaunch", "copy updated first moments", err)
	}
	momentsV := state.SecondMoment()
	if err := buffers.MomentV.driver.CopyDeviceToHost(buffers.MomentV.Pointer(), hipAdamWFloat32Bytes(momentsV)); err != nil {
		return core.E("rocm.hip.AdamWUpdateLaunch", "copy updated second moments", err)
	}
	state.Step = buffers.Step
	return nil
}

func hipRunAdamWUpdateKernel(ctx context.Context, driver nativeHIPDriver, req hipAdamWUpdateRequest) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if err := req.validate(); err != nil {
		return err
	}
	buffers, err := req.deviceBuffersValidatedValue(driver)
	if err != nil {
		return err
	}
	defer buffers.Close()
	launch, err := req.launchArgsValidated(&buffers)
	if err != nil {
		return err
	}
	var launchScratch [hipAdamWUpdateLaunchArgsBytes]byte
	launchBytes, err := launch.BinaryInto(launchScratch[:])
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAdamWUpdate, launchBytes, launch.ParamCount)
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return buffers.ReadBack(req.State)
}

func hipAdamWGradientPayloadInto(payload []byte, state *NativeAdamWState, gradients [][]float32) ([]byte, error) {
	total := stateTotalLen(state)
	if len(payload) < total*4 {
		return nil, core.E("rocm.hip.AdamWUpdateLaunch", "gradient payload buffer is too small", nil)
	}
	payload = payload[:total*4]
	clear(payload)
	for index, gradient := range gradients {
		desc := state.Layout[index]
		for valueIndex, value := range gradient {
			offset := (desc.Offset + valueIndex) * 4
			binary.LittleEndian.PutUint32(payload[offset:], math.Float32bits(value))
		}
	}
	return payload, nil
}

func hipAdamWUploadByteBufferValue(driver nativeHIPDriver, label string, payload []byte, count int) (hipDeviceByteBuffer, error) {
	const operation = "rocm.hip.AdamWUpdateLaunch"
	if len(payload) == 0 {
		return hipDeviceByteBuffer{}, core.E(operation, label+" payload is empty", nil)
	}
	buffer, err := hipAllocateByteBufferValue(driver, operation, label, uint64(len(payload)), count)
	if err != nil {
		return hipDeviceByteBuffer{}, err
	}
	if err := hipCopyHostToDeviceLabeled(driver, buffer.pointer, payload, operation, label); err != nil {
		_ = buffer.Close()
		return hipDeviceByteBuffer{}, core.E(operation, "copy "+label, err)
	}
	return buffer, nil
}

func hipAdamWFloat32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
}

func hipBorrowAdamWPayload(size int) []byte {
	if size <= 0 {
		return nil
	}
	if size > hipAdamWPayloadPoolMaxBytes {
		return make([]byte, size)
	}
	poolValue, ok := hipAdamWPayloadPools.Load(size)
	if !ok {
		pool := &hipAdamWPayloadPool{}
		poolValue, _ = hipAdamWPayloadPools.LoadOrStore(size, pool)
	}
	pool := poolValue.(*hipAdamWPayloadPool)
	pool.Lock()
	if index := len(pool.payloads) - 1; index >= 0 {
		payload := pool.payloads[index]
		pool.payloads[index] = nil
		pool.payloads = pool.payloads[:index]
		pool.Unlock()
		return payload[:size]
	}
	pool.Unlock()
	return make([]byte, size)
}

func hipReleaseAdamWPayload(payload []byte) {
	if len(payload) == 0 || cap(payload) != len(payload) || len(payload) > hipAdamWPayloadPoolMaxBytes {
		return
	}
	clear(payload)
	poolValue, ok := hipAdamWPayloadPools.Load(len(payload))
	if !ok {
		pool := &hipAdamWPayloadPool{}
		poolValue, _ = hipAdamWPayloadPools.LoadOrStore(len(payload), pool)
	}
	pool := poolValue.(*hipAdamWPayloadPool)
	pool.Lock()
	if len(pool.payloads) < hipAdamWPayloadPoolMaxPerSize {
		pool.payloads = append(pool.payloads, payload[:0])
	}
	pool.Unlock()
}

func hipPrewarmAdamWUpdateBuffers(driver nativeHIPDriver, paramCount, depth int) {
	if driver == nil || !driver.Available() || paramCount <= 0 || depth <= 0 {
		return
	}
	size := paramCount * 4
	if size <= 0 {
		return
	}
	hipPrewarmDeviceByteBufferPool(driver, uint64(size), depth*4)
	payloads := make([][]byte, 0, depth)
	for i := 0; i < depth; i++ {
		payloads = append(payloads, hipBorrowAdamWPayload(size))
	}
	for i := len(payloads) - 1; i >= 0; i-- {
		hipReleaseAdamWPayload(payloads[i])
	}
}

func validateHIPAdamWHyperparameters(args hipAdamWUpdateLaunchArgs) error {
	if args.LearningRate <= 0 || math.IsNaN(args.LearningRate) || math.IsInf(args.LearningRate, 0) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "learning rate must be positive and finite", nil)
	}
	if args.Beta1 < 0 || args.Beta1 >= 1 || math.IsNaN(args.Beta1) || math.IsInf(args.Beta1, 0) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "beta1 must be in [0,1)", nil)
	}
	if args.Beta2 < 0 || args.Beta2 >= 1 || math.IsNaN(args.Beta2) || math.IsInf(args.Beta2, 0) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "beta2 must be in [0,1)", nil)
	}
	if args.Eps <= 0 || math.IsNaN(args.Eps) || math.IsInf(args.Eps, 0) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "epsilon must be positive and finite", nil)
	}
	if args.WeightDecay < 0 || math.IsNaN(args.WeightDecay) || math.IsInf(args.WeightDecay, 0) {
		return core.E("rocm.hip.AdamWUpdateLaunch", "weight decay must be non-negative and finite", nil)
	}
	return nil
}
