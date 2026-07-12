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
	hipMoERouterLaunchArgsVersion uint32 = 1
	hipMoERouterLaunchArgsBytes          = 64
	hipMoERouterLaunchStatusOK    uint32 = 0x4d4f4552
	hipMoERouterBlockSize         uint32 = 256
	hipMoELazyLaunchArgsVersion   uint32 = 1
	hipMoELazyLaunchArgsBytes            = 64
)

type hipMoERouterRequest struct {
	Logits []float32
	TopK   int
	Layer  int
}

type hipMoERouterDeviceBuffers struct {
	Logits         *hipDeviceByteBuffer
	Output         *hipDeviceByteBuffer
	IDs            *hipDeviceByteBuffer
	Probs          *hipDeviceByteBuffer
	Status         *hipDeviceByteBuffer
	InputLogits    []float32
	ExpertCount    int
	TopK           int
	Layer          int
	BorrowedLogits bool
}

type hipMoERouterLaunchArgs struct {
	LogitPointer  nativeDevicePointer
	IDPointer     nativeDevicePointer
	ProbPointer   nativeDevicePointer
	StatusPointer nativeDevicePointer
	ExpertCount   int
	TopK          int
	Layer         int
	LogitBytes    uint64
	IDBytes       uint64
	ProbBytes     uint64
}

type hipMoERouterResult struct {
	Routes []rocmExpertRoute
	Layer  int
	Status uint32
}

type hipMoELazyExpertRequest struct {
	ExpertIDs    []int32
	TotalExperts int
}

type hipMoELazyExpertDeviceBuffers struct {
	IDs          *hipDeviceByteBuffer
	Resident     *hipDeviceByteBuffer
	Selected     int
	TotalExperts int
}

type hipMoELazyExpertLaunchArgs struct {
	IDPointer       nativeDevicePointer
	ResidentPointer nativeDevicePointer
	SelectedCount   int
	TotalExperts    int
	IDBytes         uint64
	ResidentBytes   uint64
}

type hipMoELazyExpertResult struct {
	Resident []bool
}

func (req hipMoERouterRequest) validate() error {
	if len(req.Logits) == 0 {
		return core.E("rocm.hip.MoERouterLaunch", "router logits are required", nil)
	}
	if req.TopK <= 0 || req.TopK > len(req.Logits) {
		return core.E("rocm.hip.MoERouterLaunch", "top-k must be within the expert count", nil)
	}
	if !rocmFloat32SliceFinite(req.Logits) {
		return core.E("rocm.hip.MoERouterLaunch", "router logits must be finite", nil)
	}
	if req.Layer < 0 {
		return core.E("rocm.hip.MoERouterLaunch", "layer must be non-negative", nil)
	}
	return nil
}

func (req hipMoERouterRequest) deviceBuffers(driver nativeHIPDriver) (*hipMoERouterDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	logitPayload, err := hipFloat32Payload(req.Logits)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "encode router logits", err)
	}
	logits, err := hipUploadByteBuffer(driver, "rocm.hip.MoERouterLaunch", "router logits", logitPayload, len(req.Logits))
	if err != nil {
		return nil, err
	}
	buffers := &hipMoERouterDeviceBuffers{
		Logits:      logits,
		InputLogits: append([]float32(nil), req.Logits...),
		ExpertCount: len(req.Logits),
		TopK:        req.TopK,
		Layer:       req.Layer,
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	if err := buffers.allocateOutput(driver); err != nil {
		return nil, err
	}
	success = true
	return buffers, nil
}

func (buffers *hipMoERouterDeviceBuffers) allocateOutput(driver nativeHIPDriver) error {
	if buffers == nil || buffers.TopK <= 0 {
		return core.E("rocm.hip.MoERouterLaunch", "router output geometry is required", nil)
	}
	idBytes := uint64(buffers.TopK * 4)
	probBytes := uint64(buffers.TopK * 4)
	totalBytes := idBytes + probBytes + 4
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MoERouterLaunch", "packed router output", totalBytes, int(totalBytes))
	if err != nil {
		return err
	}
	buffers.Output = output
	ids := hipBorrowDeviceByteBufferValue(driver, "router id output", output.Pointer(), idBytes, buffers.TopK)
	probs := hipBorrowDeviceByteBufferValue(driver, "router probability output", output.Pointer()+nativeDevicePointer(idBytes), probBytes, buffers.TopK)
	status := hipBorrowDeviceByteBufferValue(driver, "router status", output.Pointer()+nativeDevicePointer(idBytes+probBytes), 4, 1)
	buffers.IDs = &ids
	buffers.Probs = &probs
	buffers.Status = &status
	return nil
}

func (req hipMoERouterRequest) launchArgs(buffers *hipMoERouterDeviceBuffers) (hipMoERouterLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipMoERouterLaunchArgs{}, err
	}
	if buffers == nil || buffers.Logits == nil || buffers.IDs == nil || buffers.Probs == nil || buffers.Status == nil {
		return hipMoERouterLaunchArgs{}, core.E("rocm.hip.MoERouterLaunch", "router device buffers are required", nil)
	}
	if buffers.ExpertCount != len(req.Logits) || buffers.TopK != req.TopK ||
		buffers.Logits.Count() != len(req.Logits) || buffers.IDs.Count() != req.TopK || buffers.Probs.Count() != req.TopK ||
		buffers.Status.Count() != 1 || buffers.Logits.SizeBytes() != uint64(len(req.Logits)*4) ||
		buffers.IDs.SizeBytes() != uint64(req.TopK*4) || buffers.Probs.SizeBytes() != uint64(req.TopK*4) ||
		buffers.Status.SizeBytes() != 4 {
		return hipMoERouterLaunchArgs{}, core.E("rocm.hip.MoERouterLaunch", "router device buffer shape mismatch", nil)
	}
	return hipMoERouterLaunchArgs{
		LogitPointer:  buffers.Logits.Pointer(),
		IDPointer:     buffers.IDs.Pointer(),
		ProbPointer:   buffers.Probs.Pointer(),
		StatusPointer: buffers.Status.Pointer(),
		ExpertCount:   len(req.Logits),
		TopK:          req.TopK,
		Layer:         req.Layer,
		LogitBytes:    buffers.Logits.SizeBytes(),
		IDBytes:       buffers.IDs.SizeBytes(),
		ProbBytes:     buffers.Probs.SizeBytes(),
	}, nil
}

func (args hipMoERouterLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipMoERouterLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipMoERouterLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.LogitPointer == 0 || args.IDPointer == 0 || args.ProbPointer == 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "router logits and output pointers are required", nil)
	}
	if len(payload) < hipMoERouterLaunchArgsBytes {
		return nil, core.E("rocm.hip.MoERouterLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipMoERouterLaunchArgsBytes]
	expertCount, err := rocmDeviceKVPositiveUint32("expert count", args.ExpertCount)
	if err != nil {
		return nil, err
	}
	topK, err := rocmDeviceKVPositiveUint32("top-k", args.TopK)
	if err != nil {
		return nil, err
	}
	if topK > expertCount {
		return nil, core.E("rocm.hip.MoERouterLaunch", "top-k must be within the expert count", nil)
	}
	if args.Layer < 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "layer must be non-negative", nil)
	}
	logitBytes, err := hipAlignedFloat32Bytes("router logits", args.LogitBytes, expertCount)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "logit byte count", err)
	}
	idBytes, err := hipAlignedFloat32Bytes("router ids", args.IDBytes, topK)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "id byte count", err)
	}
	probBytes, err := hipAlignedFloat32Bytes("router probabilities", args.ProbBytes, topK)
	if err != nil {
		return nil, core.E("rocm.hip.MoERouterLaunch", "probability byte count", err)
	}
	if args.StatusPointer == 0 {
		return nil, core.E("rocm.hip.MoERouterLaunch", "router status pointer is required", nil)
	}
	if uint64(args.Layer) > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.MoERouterLaunch", "layer exceeds uint32", nil)
	}
	layer := uint32(args.Layer)
	binary.LittleEndian.PutUint32(payload[0:], hipMoERouterLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.LogitPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.IDPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ProbPointer))
	binary.LittleEndian.PutUint32(payload[32:], expertCount)
	binary.LittleEndian.PutUint32(payload[36:], topK)
	binary.LittleEndian.PutUint32(payload[40:], logitBytes)
	binary.LittleEndian.PutUint32(payload[44:], idBytes)
	binary.LittleEndian.PutUint32(payload[48:], probBytes)
	binary.LittleEndian.PutUint32(payload[52:], layer)
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.StatusPointer))
	return payload, nil
}

func (buffers *hipMoERouterDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	if err := buffers.Output.Close(); err != nil {
		lastErr = err
	}
	if !buffers.BorrowedLogits {
		if err := buffers.Logits.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipMoERouterDeviceBuffers) ReadOutput() (hipMoERouterResult, error) {
	if buffers == nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output buffers are required", nil)
	}
	if buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "packed router output buffer is required", nil)
	}
	payload := make([]byte, int(buffers.Output.SizeBytes()))
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy packed router output", err)
	}
	idBytes := buffers.TopK * 4
	probBytes := buffers.TopK * 4
	idPayload := payload[:idBytes]
	probPayload := payload[idBytes : idBytes+probBytes]
	statusPayload := payload[idBytes+probBytes:]
	routes := make([]rocmExpertRoute, buffers.TopK)
	return buffers.parseOutput(routes, idPayload, probPayload, statusPayload)
}

func (buffers *hipMoERouterDeviceBuffers) ReadOutputInto(routes []rocmExpertRoute, idPayload, probPayload, statusPayload []byte) (hipMoERouterResult, error) {
	if buffers == nil || buffers.IDs == nil || buffers.Probs == nil || buffers.Status == nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output buffers are required", nil)
	}
	if len(routes) < buffers.TopK {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router result buffer is too small", nil)
	}
	idBytes := int(buffers.IDs.SizeBytes())
	probBytes := int(buffers.Probs.SizeBytes())
	statusBytes := int(buffers.Status.SizeBytes())
	if len(idPayload) < idBytes || len(probPayload) < probBytes || len(statusPayload) < statusBytes {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output payload buffer is too small", nil)
	}
	idPayload = idPayload[:idBytes]
	probPayload = probPayload[:probBytes]
	statusPayload = statusPayload[:statusBytes]
	if err := buffers.IDs.driver.CopyDeviceToHost(buffers.IDs.Pointer(), idPayload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy router id output", err)
	}
	if err := buffers.Probs.driver.CopyDeviceToHost(buffers.Probs.Pointer(), probPayload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy router probability output", err)
	}
	if err := buffers.Status.driver.CopyDeviceToHost(buffers.Status.Pointer(), statusPayload); err != nil {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "copy router status", err)
	}
	return buffers.parseOutput(routes, idPayload, probPayload, statusPayload)
}

func (buffers *hipMoERouterDeviceBuffers) parseOutput(routes []rocmExpertRoute, idPayload, probPayload, statusPayload []byte) (hipMoERouterResult, error) {
	if len(idPayload) != buffers.TopK*4 || len(probPayload) != buffers.TopK*4 || len(statusPayload) != 4 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router output byte count mismatch", nil)
	}
	status := binary.LittleEndian.Uint32(statusPayload)
	if status != hipMoERouterLaunchStatusOK {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", core.Sprintf("router status marker mismatch: got 0x%08x want 0x%08x", status, hipMoERouterLaunchStatusOK), nil)
	}
	routes = routes[:buffers.TopK]
	for index := range routes {
		id := int(int32(binary.LittleEndian.Uint32(idPayload[index*4:])))
		if id < 0 || id >= buffers.ExpertCount {
			return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", core.Sprintf("router expert id %d outside expert count %d", id, buffers.ExpertCount), nil)
		}
		prob := math.Float32frombits(binary.LittleEndian.Uint32(probPayload[index*4:]))
		if math.IsNaN(float64(prob)) || math.IsInf(float64(prob), 0) || prob < 0 || prob > 1 {
			return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router probability must be finite and within [0,1]", nil)
		}
		score := float32(0)
		if len(buffers.InputLogits) == buffers.ExpertCount {
			score = buffers.InputLogits[id]
		}
		routes[index] = rocmExpertRoute{ID: id, Score: score, Prob: prob}
	}
	return hipMoERouterResult{
		Routes: routes,
		Layer:  buffers.Layer,
		Status: status,
	}, nil
}

func hipRunMoERouterKernel(ctx context.Context, driver nativeHIPDriver, req hipMoERouterRequest) (hipMoERouterResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipMoERouterResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipMoERouterResult{}, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameMoERouter, launchBytes, hipMoERouterBlockSize)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipMoERouterResult{}, err
	}
	return buffers.ReadOutput()
}

func hipRunMoERouterKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, topK, layer int) (hipMoERouterResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipMoERouterResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 || logits.Count() <= 0 || logits.SizeBytes() != uint64(logits.Count()*4) {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "router logits device buffer shape mismatch", nil)
	}
	if topK <= 0 || topK > logits.Count() || layer < 0 {
		return hipMoERouterResult{}, core.E("rocm.hip.MoERouterLaunch", "top-k and layer must fit the router geometry", nil)
	}
	buffers := &hipMoERouterDeviceBuffers{
		Logits: logits, ExpertCount: logits.Count(), TopK: topK, Layer: layer, BorrowedLogits: true,
	}
	defer buffers.Close()
	if err := buffers.allocateOutput(driver); err != nil {
		return hipMoERouterResult{}, err
	}
	launchBytes, err := (hipMoERouterLaunchArgs{
		LogitPointer:  logits.Pointer(),
		IDPointer:     buffers.IDs.Pointer(),
		ProbPointer:   buffers.Probs.Pointer(),
		StatusPointer: buffers.Status.Pointer(),
		ExpertCount:   logits.Count(),
		TopK:          topK,
		Layer:         layer,
		LogitBytes:    logits.SizeBytes(),
		IDBytes:       buffers.IDs.SizeBytes(),
		ProbBytes:     buffers.Probs.SizeBytes(),
	}).Binary()
	if err != nil {
		return hipMoERouterResult{}, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameMoERouter, launchBytes, hipMoERouterBlockSize)
	if err != nil {
		return hipMoERouterResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipMoERouterResult{}, err
	}
	return buffers.ReadOutput()
}

func (req hipMoELazyExpertRequest) validate() error {
	if req.TotalExperts <= 0 {
		return core.E("rocm.hip.MoELazyLaunch", "expert count must be positive", nil)
	}
	if len(req.ExpertIDs) == 0 {
		return core.E("rocm.hip.MoELazyLaunch", "selected expert IDs are required", nil)
	}
	routes := make([]rocmExpertRoute, len(req.ExpertIDs))
	for index, id := range req.ExpertIDs {
		routes[index] = rocmExpertRoute{ID: int(id)}
	}
	if _, err := rocmReferenceLazyExpertResidency(routes, req.TotalExperts); err != nil {
		return err
	}
	return nil
}

func (req hipMoELazyExpertRequest) deviceBuffers(driver nativeHIPDriver) (*hipMoELazyExpertDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	idPayload := make([]byte, len(req.ExpertIDs)*4)
	for index, id := range req.ExpertIDs {
		binary.LittleEndian.PutUint32(idPayload[index*4:], uint32(id))
	}
	ids, err := hipUploadByteBuffer(driver, "rocm.hip.MoELazyLaunch", "selected expert IDs", idPayload, len(req.ExpertIDs))
	if err != nil {
		return nil, err
	}
	buffers := &hipMoELazyExpertDeviceBuffers{IDs: ids, Selected: len(req.ExpertIDs), TotalExperts: req.TotalExperts}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	resident, err := hipAllocateByteBuffer(driver, "rocm.hip.MoELazyLaunch", "resident expert output", uint64(req.TotalExperts), req.TotalExperts)
	if err != nil {
		return nil, err
	}
	buffers.Resident = resident
	success = true
	return buffers, nil
}

func (req hipMoELazyExpertRequest) launchArgs(buffers *hipMoELazyExpertDeviceBuffers) (hipMoELazyExpertLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipMoELazyExpertLaunchArgs{}, err
	}
	if buffers == nil || buffers.IDs == nil || buffers.Resident == nil {
		return hipMoELazyExpertLaunchArgs{}, core.E("rocm.hip.MoELazyLaunch", "lazy expert device buffers are required", nil)
	}
	if buffers.Selected != len(req.ExpertIDs) || buffers.TotalExperts != req.TotalExperts ||
		buffers.IDs.Count() != len(req.ExpertIDs) || buffers.Resident.Count() != req.TotalExperts {
		return hipMoELazyExpertLaunchArgs{}, core.E("rocm.hip.MoELazyLaunch", "lazy expert device buffer shape mismatch", nil)
	}
	return hipMoELazyExpertLaunchArgs{
		IDPointer:       buffers.IDs.Pointer(),
		ResidentPointer: buffers.Resident.Pointer(),
		SelectedCount:   len(req.ExpertIDs),
		TotalExperts:    req.TotalExperts,
		IDBytes:         buffers.IDs.SizeBytes(),
		ResidentBytes:   buffers.Resident.SizeBytes(),
	}, nil
}

func (args hipMoELazyExpertLaunchArgs) Binary() ([]byte, error) {
	payload := make([]byte, hipMoELazyLaunchArgsBytes)
	return args.BinaryInto(payload)
}

func (args hipMoELazyExpertLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.IDPointer == 0 || args.ResidentPointer == 0 {
		return nil, core.E("rocm.hip.MoELazyLaunch", "expert ID and resident output pointers are required", nil)
	}
	if len(payload) < hipMoELazyLaunchArgsBytes {
		return nil, core.E("rocm.hip.MoELazyLaunch", "launch arg payload buffer is too small", nil)
	}
	payload = payload[:hipMoELazyLaunchArgsBytes]
	selected, err := rocmDeviceKVPositiveUint32("selected expert count", args.SelectedCount)
	if err != nil {
		return nil, err
	}
	total, err := rocmDeviceKVPositiveUint32("expert count", args.TotalExperts)
	if err != nil {
		return nil, err
	}
	if args.IDBytes != uint64(selected)*4 {
		return nil, core.E("rocm.hip.MoELazyLaunch", "expert ID byte count mismatch", nil)
	}
	if args.ResidentBytes != uint64(total) {
		return nil, core.E("rocm.hip.MoELazyLaunch", "resident byte count mismatch", nil)
	}
	if args.IDBytes > uint64(^uint32(0)) || args.ResidentBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.MoELazyLaunch", "lazy expert byte counts are out of uint32 range", nil)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipMoELazyLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.IDPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ResidentPointer))
	binary.LittleEndian.PutUint32(payload[24:], selected)
	binary.LittleEndian.PutUint32(payload[28:], total)
	binary.LittleEndian.PutUint32(payload[32:], uint32(args.IDBytes))
	binary.LittleEndian.PutUint32(payload[36:], uint32(args.ResidentBytes))
	return payload, nil
}

func (buffers *hipMoELazyExpertDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Resident, buffers.IDs} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipMoELazyExpertDeviceBuffers) ReadOutput() (hipMoELazyExpertResult, error) {
	if buffers == nil {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output buffer is required", nil)
	}
	payload := make([]byte, buffers.Resident.SizeBytes())
	resident := make([]bool, buffers.TotalExperts)
	return buffers.ReadOutputInto(resident, payload)
}

func (buffers *hipMoELazyExpertDeviceBuffers) ReadOutputInto(resident []bool, payload []byte) (hipMoELazyExpertResult, error) {
	if buffers == nil || buffers.Resident == nil || buffers.Resident.Pointer() == 0 {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output buffer is required", nil)
	}
	if buffers.TotalExperts <= 0 || buffers.Resident.Count() != buffers.TotalExperts || buffers.Resident.SizeBytes() != uint64(buffers.TotalExperts) {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output byte count mismatch", nil)
	}
	payloadBytes := int(buffers.Resident.SizeBytes())
	if len(resident) < buffers.TotalExperts || len(payload) < payloadBytes {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output buffer is too small", nil)
	}
	payload = payload[:payloadBytes]
	if err := buffers.Resident.driver.CopyDeviceToHost(buffers.Resident.Pointer(), payload); err != nil {
		return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "copy resident expert output", err)
	}
	resident = resident[:buffers.TotalExperts]
	for index, value := range payload {
		if value != 0 && value != 1 {
			return hipMoELazyExpertResult{}, core.E("rocm.hip.MoELazyLaunch", "resident expert output must contain binary flags", nil)
		}
		resident[index] = value != 0
	}
	return hipMoELazyExpertResult{Resident: resident}, nil
}

func hipRunMoELazyExpertKernel(ctx context.Context, driver nativeHIPDriver, req hipMoELazyExpertRequest) (hipMoELazyExpertResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipMoELazyExpertResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMoELazy, launchBytes, req.TotalExperts)
	if err != nil {
		return hipMoELazyExpertResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipMoELazyExpertResult{}, err
	}
	return buffers.ReadOutput()
}
