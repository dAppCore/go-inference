// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	hipCrossEntropyLossLaunchArgsVersion   uint32 = 1
	hipCrossEntropyLossLaunchArgsBytes            = 64
	hipCrossEntropyLossOutputBytes                = 16
	hipDistillationKLLossLaunchArgsVersion uint32 = 1
	hipDistillationKLLossLaunchArgsBytes          = 64
	hipDistillationKLLossOutputBytes              = 8
	hipGRPOAdvantageLaunchArgsVersion      uint32 = 1
	hipGRPOAdvantageLaunchArgsBytes               = 64
)

type hipCrossEntropyLossRequest struct {
	Logits  []float32
	Targets []int32
	Batch   int
	Vocab   int
}

type hipCrossEntropyLossDeviceBuffers struct {
	Logits  *hipDeviceByteBuffer
	Targets *hipDeviceByteBuffer
	Output  *hipDeviceByteBuffer
	Batch   int
	Vocab   int
}

type hipCrossEntropyLossLaunchArgs struct {
	LogitPointer  nativeDevicePointer
	TargetPointer nativeDevicePointer
	OutputPointer nativeDevicePointer
	Batch         int
	Vocab         int
	LogitBytes    uint64
	TargetBytes   uint64
	OutputBytes   uint64
}

type hipCrossEntropyLossResult struct {
	Loss       float64
	Perplexity float64
}

type hipDistillationKLLossRequest struct {
	StudentLogits []float32
	TeacherLogits []float32
	Batch         int
	Vocab         int
	Temperature   float64
}

type hipDistillationKLLossDeviceBuffers struct {
	StudentLogits *hipDeviceByteBuffer
	TeacherLogits *hipDeviceByteBuffer
	Output        *hipDeviceByteBuffer
	Batch         int
	Vocab         int
}

type hipDistillationKLLossLaunchArgs struct {
	StudentPointer nativeDevicePointer
	TeacherPointer nativeDevicePointer
	OutputPointer  nativeDevicePointer
	Batch          int
	Vocab          int
	StudentBytes   uint64
	TeacherBytes   uint64
	OutputBytes    uint64
	Temperature    float64
}

type hipDistillationKLLossResult struct {
	KL float64
}

type hipGRPOAdvantageRequest struct {
	Rewards []float64
	Count   int
}

type hipGRPOAdvantageDeviceBuffers struct {
	Rewards *hipDeviceByteBuffer
	Output  *hipDeviceByteBuffer
	Count   int
}

type hipGRPOAdvantageLaunchArgs struct {
	RewardPointer nativeDevicePointer
	OutputPointer nativeDevicePointer
	Count         int
	RewardBytes   uint64
	OutputBytes   uint64
}

func (req hipCrossEntropyLossRequest) validate() error {
	if req.Batch <= 0 || req.Vocab <= 0 {
		return core.E("rocm.hip.CrossEntropyLossLaunch", "batch and vocabulary must be positive", nil)
	}
	if len(req.Logits) != req.Batch*req.Vocab {
		return core.E("rocm.hip.CrossEntropyLossLaunch", "logit length must match batch*vocab", nil)
	}
	if len(req.Targets) != req.Batch {
		return core.E("rocm.hip.CrossEntropyLossLaunch", "target length must match batch", nil)
	}
	if !rocmFloat32SliceFinite(req.Logits) {
		return core.E("rocm.hip.CrossEntropyLossLaunch", "logit values must be finite", nil)
	}
	for _, target := range req.Targets {
		if target < 0 || int(target) >= req.Vocab {
			return core.E("rocm.hip.CrossEntropyLossLaunch", "target is outside vocabulary", nil)
		}
	}
	return nil
}

func (req hipCrossEntropyLossRequest) deviceBuffers(driver nativeHIPDriver) (*hipCrossEntropyLossDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	logitPayload, err := hipFloat32Payload(req.Logits)
	if err != nil {
		return nil, core.E("rocm.hip.CrossEntropyLossLaunch", "encode logits", err)
	}
	logits, err := hipUploadByteBuffer(driver, "rocm.hip.CrossEntropyLossLaunch", "cross entropy logits", logitPayload, len(req.Logits))
	if err != nil {
		return nil, err
	}
	buffers := &hipCrossEntropyLossDeviceBuffers{Logits: logits, Batch: req.Batch, Vocab: req.Vocab}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	targetPayload, err := hipTokenIDsPayload(req.Targets)
	if err != nil {
		return nil, core.E("rocm.hip.CrossEntropyLossLaunch", "encode targets", err)
	}
	targets, err := hipUploadByteBuffer(driver, "rocm.hip.CrossEntropyLossLaunch", "cross entropy targets", targetPayload, len(req.Targets))
	if err != nil {
		return nil, err
	}
	buffers.Targets = targets
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.CrossEntropyLossLaunch", "cross entropy output", hipCrossEntropyLossOutputBytes, 2)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipCrossEntropyLossRequest) launchArgs(buffers *hipCrossEntropyLossDeviceBuffers) (hipCrossEntropyLossLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipCrossEntropyLossLaunchArgs{}, err
	}
	if buffers == nil || buffers.Logits == nil || buffers.Targets == nil || buffers.Output == nil {
		return hipCrossEntropyLossLaunchArgs{}, core.E("rocm.hip.CrossEntropyLossLaunch", "cross entropy device buffers are required", nil)
	}
	if buffers.Logits.Count() != req.Batch*req.Vocab ||
		buffers.Targets.Count() != req.Batch ||
		buffers.Output.Count() != 2 ||
		buffers.Output.SizeBytes() != hipCrossEntropyLossOutputBytes ||
		buffers.Batch != req.Batch ||
		buffers.Vocab != req.Vocab {
		return hipCrossEntropyLossLaunchArgs{}, core.E("rocm.hip.CrossEntropyLossLaunch", "cross entropy device buffer shape mismatch", nil)
	}
	return hipCrossEntropyLossLaunchArgs{
		LogitPointer:  buffers.Logits.Pointer(),
		TargetPointer: buffers.Targets.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Batch:         req.Batch,
		Vocab:         req.Vocab,
		LogitBytes:    buffers.Logits.SizeBytes(),
		TargetBytes:   buffers.Targets.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (args hipCrossEntropyLossLaunchArgs) Binary() ([]byte, error) {
	if args.LogitPointer == 0 || args.TargetPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.CrossEntropyLossLaunch", "logit, target, and output pointers are required", nil)
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	vocab, err := rocmDeviceKVPositiveUint32("vocabulary", args.Vocab)
	if err != nil {
		return nil, err
	}
	logitCount, err := hipUint32Product("cross entropy logits", batch, vocab)
	if err != nil {
		return nil, err
	}
	logitBytes, err := hipAlignedFloat32Bytes("cross entropy logits", args.LogitBytes, logitCount)
	if err != nil {
		return nil, core.E("rocm.hip.CrossEntropyLossLaunch", "logit byte count", err)
	}
	targetBytes, err := hipExactUint32Bytes("cross entropy targets", args.TargetBytes, uint64(batch)*4)
	if err != nil {
		return nil, core.E("rocm.hip.CrossEntropyLossLaunch", "target byte count", err)
	}
	outputBytes, err := hipExactUint32Bytes("cross entropy output", args.OutputBytes, hipCrossEntropyLossOutputBytes)
	if err != nil {
		return nil, core.E("rocm.hip.CrossEntropyLossLaunch", "output byte count", err)
	}
	payload := make([]byte, hipCrossEntropyLossLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipCrossEntropyLossLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LogitPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.TargetPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], batch)
	binary.LittleEndian.PutUint32(payload[36:], vocab)
	binary.LittleEndian.PutUint32(payload[40:], logitBytes)
	binary.LittleEndian.PutUint32(payload[44:], targetBytes)
	binary.LittleEndian.PutUint32(payload[48:], outputBytes)
	return payload, nil
}

func (buffers *hipCrossEntropyLossDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Targets, buffers.Logits} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipCrossEntropyLossDeviceBuffers) ReadOutput() (hipCrossEntropyLossResult, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return hipCrossEntropyLossResult{}, core.E("rocm.hip.CrossEntropyLossLaunch", "cross entropy output buffer is required", nil)
	}
	if buffers.Output.Count() != 2 || buffers.Output.SizeBytes() != hipCrossEntropyLossOutputBytes {
		return hipCrossEntropyLossResult{}, core.E("rocm.hip.CrossEntropyLossLaunch", "cross entropy output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return hipCrossEntropyLossResult{}, core.E("rocm.hip.CrossEntropyLossLaunch", "copy cross entropy output", err)
	}
	result := hipCrossEntropyLossResult{
		Loss:       math.Float64frombits(binary.LittleEndian.Uint64(payload[0:])),
		Perplexity: math.Float64frombits(binary.LittleEndian.Uint64(payload[8:])),
	}
	if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) || result.Loss < 0 ||
		math.IsNaN(result.Perplexity) || math.IsInf(result.Perplexity, 0) || result.Perplexity <= 0 {
		return hipCrossEntropyLossResult{}, core.E("rocm.hip.CrossEntropyLossLaunch", "cross entropy output values must be finite and valid", nil)
	}
	return result, nil
}

func hipRunCrossEntropyLossKernel(ctx context.Context, driver nativeHIPDriver, req hipCrossEntropyLossRequest) (hipCrossEntropyLossResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipCrossEntropyLossResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipCrossEntropyLossResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipCrossEntropyLossResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipCrossEntropyLossResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameCrossEntropy, launchBytes, 1)
	if err != nil {
		return hipCrossEntropyLossResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipCrossEntropyLossResult{}, err
	}
	return buffers.ReadOutput()
}

func (req hipDistillationKLLossRequest) validate() error {
	if req.Batch <= 0 || req.Vocab <= 0 {
		return core.E("rocm.hip.DistillationKLLossLaunch", "batch and vocabulary must be positive", nil)
	}
	if len(req.StudentLogits) != req.Batch*req.Vocab || len(req.TeacherLogits) != req.Batch*req.Vocab {
		return core.E("rocm.hip.DistillationKLLossLaunch", "student and teacher logit lengths must match batch*vocab", nil)
	}
	if req.Temperature <= 0 || math.IsNaN(req.Temperature) || math.IsInf(req.Temperature, 0) {
		return core.E("rocm.hip.DistillationKLLossLaunch", "temperature must be positive and finite", nil)
	}
	if !rocmFloat32SliceFinite(req.StudentLogits) || !rocmFloat32SliceFinite(req.TeacherLogits) {
		return core.E("rocm.hip.DistillationKLLossLaunch", "student and teacher logits must be finite", nil)
	}
	return nil
}

func (req hipDistillationKLLossRequest) deviceBuffers(driver nativeHIPDriver) (*hipDistillationKLLossDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	studentPayload, err := hipFloat32Payload(req.StudentLogits)
	if err != nil {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "encode student logits", err)
	}
	student, err := hipUploadByteBuffer(driver, "rocm.hip.DistillationKLLossLaunch", "distillation student logits", studentPayload, len(req.StudentLogits))
	if err != nil {
		return nil, err
	}
	buffers := &hipDistillationKLLossDeviceBuffers{StudentLogits: student, Batch: req.Batch, Vocab: req.Vocab}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	teacherPayload, err := hipFloat32Payload(req.TeacherLogits)
	if err != nil {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "encode teacher logits", err)
	}
	teacher, err := hipUploadByteBuffer(driver, "rocm.hip.DistillationKLLossLaunch", "distillation teacher logits", teacherPayload, len(req.TeacherLogits))
	if err != nil {
		return nil, err
	}
	buffers.TeacherLogits = teacher
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.DistillationKLLossLaunch", "distillation output", hipDistillationKLLossOutputBytes, 1)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipDistillationKLLossRequest) launchArgs(buffers *hipDistillationKLLossDeviceBuffers) (hipDistillationKLLossLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipDistillationKLLossLaunchArgs{}, err
	}
	if buffers == nil || buffers.StudentLogits == nil || buffers.TeacherLogits == nil || buffers.Output == nil {
		return hipDistillationKLLossLaunchArgs{}, core.E("rocm.hip.DistillationKLLossLaunch", "distillation device buffers are required", nil)
	}
	if buffers.StudentLogits.Count() != req.Batch*req.Vocab ||
		buffers.TeacherLogits.Count() != req.Batch*req.Vocab ||
		buffers.Output.Count() != 1 ||
		buffers.Output.SizeBytes() != hipDistillationKLLossOutputBytes ||
		buffers.Batch != req.Batch ||
		buffers.Vocab != req.Vocab {
		return hipDistillationKLLossLaunchArgs{}, core.E("rocm.hip.DistillationKLLossLaunch", "distillation device buffer shape mismatch", nil)
	}
	return hipDistillationKLLossLaunchArgs{
		StudentPointer: buffers.StudentLogits.Pointer(),
		TeacherPointer: buffers.TeacherLogits.Pointer(),
		OutputPointer:  buffers.Output.Pointer(),
		Batch:          req.Batch,
		Vocab:          req.Vocab,
		StudentBytes:   buffers.StudentLogits.SizeBytes(),
		TeacherBytes:   buffers.TeacherLogits.SizeBytes(),
		OutputBytes:    buffers.Output.SizeBytes(),
		Temperature:    req.Temperature,
	}, nil
}

func (args hipDistillationKLLossLaunchArgs) Binary() ([]byte, error) {
	if args.StudentPointer == 0 || args.TeacherPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "student, teacher, and output pointers are required", nil)
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	vocab, err := rocmDeviceKVPositiveUint32("vocabulary", args.Vocab)
	if err != nil {
		return nil, err
	}
	logitCount, err := hipUint32Product("distillation logits", batch, vocab)
	if err != nil {
		return nil, err
	}
	studentBytes, err := hipAlignedFloat32Bytes("distillation student logits", args.StudentBytes, logitCount)
	if err != nil {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "student byte count", err)
	}
	teacherBytes, err := hipAlignedFloat32Bytes("distillation teacher logits", args.TeacherBytes, logitCount)
	if err != nil {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "teacher byte count", err)
	}
	outputBytes, err := hipExactUint32Bytes("distillation output", args.OutputBytes, hipDistillationKLLossOutputBytes)
	if err != nil {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "output byte count", err)
	}
	if args.Temperature <= 0 || math.IsNaN(args.Temperature) || math.IsInf(args.Temperature, 0) {
		return nil, core.E("rocm.hip.DistillationKLLossLaunch", "temperature must be positive and finite", nil)
	}
	payload := make([]byte, hipDistillationKLLossLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipDistillationKLLossLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.StudentPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.TeacherPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[32:], batch)
	binary.LittleEndian.PutUint32(payload[36:], vocab)
	binary.LittleEndian.PutUint32(payload[40:], studentBytes)
	binary.LittleEndian.PutUint32(payload[44:], teacherBytes)
	binary.LittleEndian.PutUint32(payload[48:], outputBytes)
	binary.LittleEndian.PutUint64(payload[56:], math.Float64bits(args.Temperature))
	return payload, nil
}

func (buffers *hipDistillationKLLossDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.TeacherLogits, buffers.StudentLogits} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipDistillationKLLossDeviceBuffers) ReadOutput() (hipDistillationKLLossResult, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return hipDistillationKLLossResult{}, core.E("rocm.hip.DistillationKLLossLaunch", "distillation output buffer is required", nil)
	}
	if buffers.Output.Count() != 1 || buffers.Output.SizeBytes() != hipDistillationKLLossOutputBytes {
		return hipDistillationKLLossResult{}, core.E("rocm.hip.DistillationKLLossLaunch", "distillation output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return hipDistillationKLLossResult{}, core.E("rocm.hip.DistillationKLLossLaunch", "copy distillation output", err)
	}
	result := hipDistillationKLLossResult{KL: math.Float64frombits(binary.LittleEndian.Uint64(payload[0:]))}
	if math.IsNaN(result.KL) || math.IsInf(result.KL, 0) || result.KL < 0 {
		return hipDistillationKLLossResult{}, core.E("rocm.hip.DistillationKLLossLaunch", "distillation output value must be finite and valid", nil)
	}
	return result, nil
}

func hipRunDistillationKLLossKernel(ctx context.Context, driver nativeHIPDriver, req hipDistillationKLLossRequest) (hipDistillationKLLossResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipDistillationKLLossResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipDistillationKLLossResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipDistillationKLLossResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipDistillationKLLossResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameDistillKL, launchBytes, 1)
	if err != nil {
		return hipDistillationKLLossResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipDistillationKLLossResult{}, err
	}
	return buffers.ReadOutput()
}

func (req hipGRPOAdvantageRequest) validate() error {
	if req.Count <= 0 {
		return core.E("rocm.hip.GRPOAdvantageLaunch", "reward count must be positive", nil)
	}
	if len(req.Rewards) != req.Count {
		return core.E("rocm.hip.GRPOAdvantageLaunch", "reward length must match count", nil)
	}
	if !hipFloat64SliceFinite(req.Rewards) {
		return core.E("rocm.hip.GRPOAdvantageLaunch", "reward values must be finite", nil)
	}
	return nil
}

func (req hipGRPOAdvantageRequest) deviceBuffers(driver nativeHIPDriver) (*hipGRPOAdvantageDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	rewardPayload, err := hipFloat64Payload(req.Rewards)
	if err != nil {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "encode rewards", err)
	}
	rewards, err := hipUploadByteBuffer(driver, "rocm.hip.GRPOAdvantageLaunch", "GRPO rewards", rewardPayload, len(req.Rewards))
	if err != nil {
		return nil, err
	}
	buffers := &hipGRPOAdvantageDeviceBuffers{Rewards: rewards, Count: req.Count}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	outputBytes := uint64(req.Count) * 8
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.GRPOAdvantageLaunch", "GRPO advantages output", outputBytes, req.Count)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipGRPOAdvantageRequest) launchArgs(buffers *hipGRPOAdvantageDeviceBuffers) (hipGRPOAdvantageLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipGRPOAdvantageLaunchArgs{}, err
	}
	if buffers == nil || buffers.Rewards == nil || buffers.Output == nil {
		return hipGRPOAdvantageLaunchArgs{}, core.E("rocm.hip.GRPOAdvantageLaunch", "GRPO advantage device buffers are required", nil)
	}
	outputBytes := uint64(req.Count) * 8
	if buffers.Rewards.Count() != req.Count ||
		buffers.Rewards.SizeBytes() != outputBytes ||
		buffers.Output.Count() != req.Count ||
		buffers.Output.SizeBytes() != outputBytes ||
		buffers.Count != req.Count {
		return hipGRPOAdvantageLaunchArgs{}, core.E("rocm.hip.GRPOAdvantageLaunch", "GRPO advantage device buffer shape mismatch", nil)
	}
	return hipGRPOAdvantageLaunchArgs{
		RewardPointer: buffers.Rewards.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Count:         req.Count,
		RewardBytes:   buffers.Rewards.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (args hipGRPOAdvantageLaunchArgs) Binary() ([]byte, error) {
	if args.RewardPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "reward and output pointers are required", nil)
	}
	count, err := rocmDeviceKVPositiveUint32("reward count", args.Count)
	if err != nil {
		return nil, err
	}
	outputBytes := uint64(count) * 8
	rewardBytes, err := hipExactUint32Bytes("GRPO rewards", args.RewardBytes, outputBytes)
	if err != nil {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "reward byte count", err)
	}
	resultBytes, err := hipExactUint32Bytes("GRPO advantages output", args.OutputBytes, outputBytes)
	if err != nil {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "output byte count", err)
	}
	payload := make([]byte, hipGRPOAdvantageLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipGRPOAdvantageLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.RewardPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], count)
	binary.LittleEndian.PutUint32(payload[28:], rewardBytes)
	binary.LittleEndian.PutUint32(payload[32:], resultBytes)
	return payload, nil
}

func (buffers *hipGRPOAdvantageDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Rewards} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipGRPOAdvantageDeviceBuffers) ReadOutput() ([]float64, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "GRPO advantage output buffer is required", nil)
	}
	if buffers.Count <= 0 || buffers.Output.Count() != buffers.Count || buffers.Output.SizeBytes() != uint64(buffers.Count)*8 {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "GRPO advantage output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "copy GRPO advantage output", err)
	}
	values, err := hipFloat64PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !hipFloat64SliceFinite(values) {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "GRPO advantage output values must be finite", nil)
	}
	return values, nil
}

func hipRunGRPOAdvantageKernel(ctx context.Context, driver nativeHIPDriver, req hipGRPOAdvantageRequest) ([]float64, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGRPOAdvantage, launchBytes, 1)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipFloat64Payload(values []float64) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "float64 payload is empty", nil)
	}
	payload := make([]byte, len(values)*8)
	for index, value := range values {
		binary.LittleEndian.PutUint64(payload[index*8:], math.Float64bits(value))
	}
	return payload, nil
}

func hipFloat64PayloadValues(payload []byte) ([]float64, error) {
	if len(payload) == 0 || len(payload)%8 != 0 {
		return nil, core.E("rocm.hip.GRPOAdvantageLaunch", "float64 payload byte length must be positive and aligned", nil)
	}
	values := make([]float64, len(payload)/8)
	for index := range values {
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(payload[index*8:]))
	}
	return values, nil
}

func hipFloat64SliceFinite(values []float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (model *hipLoadedModel) RunEvalCrossEntropyLoss(ctx context.Context, logits [][]float32, targets []int) (hipCrossEntropyLossResult, bool, error) {
	if model == nil || model.driver == nil {
		return hipCrossEntropyLossResult{}, false, nil
	}
	if normalizeHIPKernelStatus(model.KernelStatus()).CrossEntropy != hipKernelStatusLinked {
		return hipCrossEntropyLossResult{}, false, nil
	}
	if len(logits) == 0 || len(logits) != len(targets) {
		return hipCrossEntropyLossResult{}, false, nil
	}
	flat, vocab, ok, err := hipFlattenFloat32Rows("rocm.hip.EvalCrossEntropyLoss", logits)
	if err != nil {
		return hipCrossEntropyLossResult{}, ok, err
	}
	if !ok {
		return hipCrossEntropyLossResult{}, false, nil
	}
	targetIDs := make([]int32, len(targets))
	for index, target := range targets {
		targetIDs[index] = int32(target)
	}
	result, err := hipRunCrossEntropyLossKernel(ctx, model.driver, hipCrossEntropyLossRequest{
		Logits:  flat,
		Targets: targetIDs,
		Batch:   len(logits),
		Vocab:   vocab,
	})
	return result, true, err
}

func (model *hipLoadedModel) RunDistillationKLLoss(ctx context.Context, studentLogits, teacherLogits [][]float32, temperature float64) (hipDistillationKLLossResult, bool, error) {
	if model == nil || model.driver == nil {
		return hipDistillationKLLossResult{}, false, nil
	}
	if normalizeHIPKernelStatus(model.KernelStatus()).Distillation != hipKernelStatusLinked {
		return hipDistillationKLLossResult{}, false, nil
	}
	if len(studentLogits) == 0 || len(studentLogits) != len(teacherLogits) {
		return hipDistillationKLLossResult{}, false, nil
	}
	studentFlat, vocab, ok, err := hipFlattenFloat32Rows("rocm.hip.DistillationKLLoss", studentLogits)
	if err != nil {
		return hipDistillationKLLossResult{}, ok, err
	}
	if !ok {
		return hipDistillationKLLossResult{}, false, nil
	}
	teacherFlat, teacherVocab, ok, err := hipFlattenFloat32Rows("rocm.hip.DistillationKLLoss", teacherLogits)
	if err != nil {
		return hipDistillationKLLossResult{}, ok, err
	}
	if !ok || teacherVocab != vocab {
		return hipDistillationKLLossResult{}, false, nil
	}
	result, err := hipRunDistillationKLLossKernel(ctx, model.driver, hipDistillationKLLossRequest{
		StudentLogits: studentFlat,
		TeacherLogits: teacherFlat,
		Batch:         len(studentLogits),
		Vocab:         vocab,
		Temperature:   temperature,
	})
	return result, true, err
}

func (model *hipLoadedModel) RunGRPOAdvantage(ctx context.Context, rewards []float64) ([]float64, bool, error) {
	if model == nil || model.driver == nil {
		return nil, false, nil
	}
	if normalizeHIPKernelStatus(model.KernelStatus()).GRPO != hipKernelStatusLinked {
		return nil, false, nil
	}
	if len(rewards) == 0 {
		return nil, false, nil
	}
	result, err := hipRunGRPOAdvantageKernel(ctx, model.driver, hipGRPOAdvantageRequest{
		Rewards: append([]float64(nil), rewards...),
		Count:   len(rewards),
	})
	return result, true, err
}

func (model *hipLoadedModel) RunAdamWUpdate(ctx context.Context, state *NativeAdamWState, gradients [][]float32) (bool, error) {
	if model == nil || model.driver == nil {
		return false, nil
	}
	if normalizeHIPKernelStatus(model.KernelStatus()).Optimizer != hipKernelStatusLinked {
		return false, nil
	}
	if state == nil || len(gradients) == 0 {
		return false, nil
	}
	err := hipRunAdamWUpdateKernel(ctx, model.driver, hipAdamWUpdateRequest{
		State:     state,
		Gradients: gradients,
	})
	return true, err
}

func hipFlattenFloat32Rows(scope string, rows [][]float32) ([]float32, int, bool, error) {
	if len(rows) == 0 {
		return nil, 0, false, nil
	}
	vocab := len(rows[0])
	if vocab == 0 {
		return nil, 0, true, core.E(scope, "logit row must be non-empty", nil)
	}
	flat := make([]float32, 0, len(rows)*vocab)
	for _, row := range rows {
		if len(row) != vocab {
			return nil, 0, false, nil
		}
		flat = append(flat, row...)
	}
	return flat, vocab, true, nil
}
