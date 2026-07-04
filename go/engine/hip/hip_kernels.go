// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type hipKernelKind string

const (
	hipKernelDecode     hipKernelKind = "decode"
	hipKernelEmbedding  hipKernelKind = "embedding"
	hipKernelLoRA       hipKernelKind = "lora"
	hipKernelPrefill    hipKernelKind = "prefill"
	hipKernelProjection hipKernelKind = "projection"
	hipKernelRerank     hipKernelKind = "rerank"
	hipKernelKVCache    hipKernelKind = "kv_cache"
)

const (
	hipKernelStatusLinked    = "linked"
	hipKernelStatusNotLinked = "not_linked"
	hipKernelStatusPlanned   = "planned"
)

const (
	hipPrefillLaunchArgsVersion uint32 = 1
	hipPrefillLaunchArgsBytes          = 64
	hipPrefillLaunchStatusOK    uint32 = 0x5052464c

	hipDecodeLaunchArgsVersion     uint32 = 1
	hipDecodeLaunchArgsHeaderBytes        = 32
	hipDecodeLaunchArgsBytes              = hipDecodeLaunchArgsHeaderBytes + rocmDeviceKVLaunchDescriptorBytes
	hipDecodeLaunchStatusOK        uint32 = 0x4445434f
)

type hipKernelStatus struct {
	CrossEntropy string
	Decode       string
	Distillation string
	Embedding    string
	GRPO         string
	LoRA         string
	Optimizer    string
	Prefill      string
	Projection   string
	Rerank       string
	KVCache      string
	Reason       string
}

type hipProjectionRequest struct {
	Input     []float32
	F32       []float32
	FP16      []uint16
	BF16      []uint16
	Q8        []int8
	Q8Scale   float32
	Rows      int
	Cols      int
	Bias      []float32
	TensorKey string
}

type hipPrefillRequest struct {
	TokenIDs   []int32
	Prompt     string
	CacheMode  string
	KeyWidth   int
	ValueWidth int
}

type hipPrefillLaunchArgs struct {
	TokenPointer  nativeDevicePointer
	TokenCount    int
	TokenBytes    uint64
	CacheMode     string
	ModeCode      uint32
	BlockSize     int
	KeyWidth      int
	ValueWidth    int
	StatusPointer nativeDevicePointer
	StatusValue   uint32
}

type hipPrefillResult struct {
	Logits              []float32
	PromptTokens        int
	KV                  *rocmKVCache
	DeviceKV            *rocmDeviceKVCache
	DescriptorTable     *rocmDeviceKVDescriptorTable
	Gemma4Q4State       hipGemma4Q4DecodeState
	Gemma4Q4DeviceState *hipGemma4Q4DeviceDecodeState
	Labels              map[string]string
}

type hipDecodeRequest struct {
	TokenID             int32
	KV                  *rocmKVCache
	DeviceKV            *rocmDeviceKVCache
	DescriptorTable     *rocmDeviceKVDescriptorTable
	KeyWidth            int
	ValueWidth          int
	DeviceKVMode        string
	Position            int
	Gemma4Q4State       hipGemma4Q4DecodeState
	Gemma4Q4DeviceState *hipGemma4Q4DeviceDecodeState
}

type hipDecodeLaunchArgs struct {
	TokenID  int32
	Position int
	KV       rocmDeviceKVLaunchDescriptor
}

type hipDecodeResult struct {
	Token               inference.Token
	Logits              []float32
	KV                  *rocmKVCache
	DeviceKV            *rocmDeviceKVCache
	DescriptorTable     *rocmDeviceKVDescriptorTable
	Gemma4Q4State       hipGemma4Q4DecodeState
	Gemma4Q4DeviceState *hipGemma4Q4DeviceDecodeState
	Labels              map[string]string
}

func defaultHIPKernelStatus() hipKernelStatus {
	return hipKernelStatus{
		CrossEntropy: hipKernelStatusNotLinked,
		Decode:       hipKernelStatusNotLinked,
		Distillation: hipKernelStatusNotLinked,
		Embedding:    hipKernelStatusNotLinked,
		GRPO:         hipKernelStatusNotLinked,
		LoRA:         hipKernelStatusNotLinked,
		Optimizer:    hipKernelStatusNotLinked,
		Prefill:      hipKernelStatusNotLinked,
		Projection:   hipKernelStatusNotLinked,
		Rerank:       hipKernelStatusNotLinked,
		KVCache:      hipKernelStatusPlanned,
		Reason:       "native HIP kernels are not linked into this build",
	}
}

func normalizeHIPKernelStatus(status hipKernelStatus) hipKernelStatus {
	defaultStatus := defaultHIPKernelStatus()
	if status.CrossEntropy == "" {
		status.CrossEntropy = defaultStatus.CrossEntropy
	}
	if status.Decode == "" {
		status.Decode = defaultStatus.Decode
	}
	if status.Distillation == "" {
		status.Distillation = defaultStatus.Distillation
	}
	if status.Embedding == "" {
		status.Embedding = defaultStatus.Embedding
	}
	if status.GRPO == "" {
		status.GRPO = defaultStatus.GRPO
	}
	if status.LoRA == "" {
		status.LoRA = defaultStatus.LoRA
	}
	if status.Optimizer == "" {
		status.Optimizer = defaultStatus.Optimizer
	}
	if status.Prefill == "" {
		status.Prefill = defaultStatus.Prefill
	}
	if status.Projection == "" {
		status.Projection = defaultStatus.Projection
	}
	if status.Rerank == "" {
		status.Rerank = defaultStatus.Rerank
	}
	if status.KVCache == "" {
		status.KVCache = defaultStatus.KVCache
	}
	if status.Reason == "" && status.Overall() != hipKernelStatusLinked {
		status.Reason = defaultStatus.Reason
	}
	return status
}

func (status hipKernelStatus) Overall() string {
	status = normalizeHIPKernelStatusFields(status)
	if status.CrossEntropy == hipKernelStatusLinked || status.Decode == hipKernelStatusLinked || status.Distillation == hipKernelStatusLinked || status.Embedding == hipKernelStatusLinked || status.GRPO == hipKernelStatusLinked || status.LoRA == hipKernelStatusLinked || status.Optimizer == hipKernelStatusLinked || status.Prefill == hipKernelStatusLinked || status.Projection == hipKernelStatusLinked || status.Rerank == hipKernelStatusLinked {
		return hipKernelStatusLinked
	}
	if status.CrossEntropy == hipKernelStatusNotLinked || status.Decode == hipKernelStatusNotLinked || status.Distillation == hipKernelStatusNotLinked || status.Embedding == hipKernelStatusNotLinked || status.GRPO == hipKernelStatusNotLinked || status.LoRA == hipKernelStatusNotLinked || status.Optimizer == hipKernelStatusNotLinked || status.Prefill == hipKernelStatusNotLinked || status.Projection == hipKernelStatusNotLinked || status.Rerank == hipKernelStatusNotLinked {
		return hipKernelStatusNotLinked
	}
	return hipKernelStatusPlanned
}

func (status hipKernelStatus) Labels() map[string]string {
	status = normalizeHIPKernelStatus(status)
	labels := map[string]string{
		"cross_entropy_kernel": firstNonEmptyString(status.CrossEntropy, hipKernelStatusPlanned),
		"decode_kernel":        firstNonEmptyString(status.Decode, hipKernelStatusPlanned),
		"distillation_kernel":  firstNonEmptyString(status.Distillation, hipKernelStatusPlanned),
		"embedding_kernel":     firstNonEmptyString(status.Embedding, hipKernelStatusPlanned),
		"grpo_kernel":          firstNonEmptyString(status.GRPO, hipKernelStatusPlanned),
		"kernel_status":        status.Overall(),
		"kv_cache_kernel":      firstNonEmptyString(status.KVCache, hipKernelStatusPlanned),
		"lora_kernel":          firstNonEmptyString(status.LoRA, hipKernelStatusPlanned),
		"optimizer_kernel":     firstNonEmptyString(status.Optimizer, hipKernelStatusPlanned),
		"prefill_kernel":       firstNonEmptyString(status.Prefill, hipKernelStatusPlanned),
		"projection_kernel":    firstNonEmptyString(status.Projection, hipKernelStatusPlanned),
		"rerank_kernel":        firstNonEmptyString(status.Rerank, hipKernelStatusPlanned),
	}
	if status.Reason != "" {
		labels["kernel_detail"] = status.Reason
	}
	return labels
}

func normalizeHIPKernelStatusFields(status hipKernelStatus) hipKernelStatus {
	if status.CrossEntropy == "" {
		status.CrossEntropy = hipKernelStatusPlanned
	}
	if status.Decode == "" {
		status.Decode = hipKernelStatusPlanned
	}
	if status.Distillation == "" {
		status.Distillation = hipKernelStatusPlanned
	}
	if status.Embedding == "" {
		status.Embedding = hipKernelStatusPlanned
	}
	if status.GRPO == "" {
		status.GRPO = hipKernelStatusPlanned
	}
	if status.LoRA == "" {
		status.LoRA = hipKernelStatusPlanned
	}
	if status.Optimizer == "" {
		status.Optimizer = hipKernelStatusPlanned
	}
	if status.Prefill == "" {
		status.Prefill = hipKernelStatusPlanned
	}
	if status.Projection == "" {
		status.Projection = hipKernelStatusPlanned
	}
	if status.Rerank == "" {
		status.Rerank = hipKernelStatusPlanned
	}
	return status
}

type hipKernelSet interface {
	Status() hipKernelStatus
	Generate(ctx context.Context, model *hipLoadedModel, prompt string, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error)
	Chat(ctx context.Context, model *hipLoadedModel, messages []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error)
	Classify(ctx context.Context, model *hipLoadedModel, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error)
	BatchGenerate(ctx context.Context, model *hipLoadedModel, prompts []string, cfg inference.GenerateConfig) ([]inference.BatchResult, error)
	Project(ctx context.Context, model *hipLoadedModel, req hipProjectionRequest) ([]float32, error)
	Prefill(ctx context.Context, model *hipLoadedModel, req hipPrefillRequest) (hipPrefillResult, error)
	Decode(ctx context.Context, model *hipLoadedModel, req hipDecodeRequest) (hipDecodeResult, error)
}

func (model *hipLoadedModel) kernelSet() hipKernelSet {
	if model != nil && model.kernels != nil {
		return model.kernels
	}
	return hipKernelStub{}
}

func hipKernelNotLinkedError(operation string, kind hipKernelKind, status hipKernelStatus) error {
	message := "native " + string(kind) + " kernels are not linked yet"
	if status.Reason != "" {
		message += ": " + status.Reason
	}
	return core.E(operation, message, nil)
}

func (req hipProjectionRequest) validate() error {
	encodings := 0
	if len(req.F32) > 0 {
		encodings++
	}
	if len(req.FP16) > 0 {
		encodings++
	}
	if len(req.BF16) > 0 {
		encodings++
	}
	if len(req.Q8) > 0 {
		encodings++
	}
	if encodings > 1 {
		return core.E("rocm.hip.Projection", "only one projection weight encoding may be supplied", nil)
	}
	switch {
	case len(req.F32) > 0:
		return validateHIPProjectionShape(len(req.Input), len(req.F32), len(req.Bias), req.Rows, req.Cols)
	case len(req.FP16) > 0:
		return validateHIPProjectionShape(len(req.Input), len(req.FP16), len(req.Bias), req.Rows, req.Cols)
	case len(req.BF16) > 0:
		return validateHIPProjectionShape(len(req.Input), len(req.BF16), len(req.Bias), req.Rows, req.Cols)
	case len(req.Q8) > 0:
		if !hipQ8ScaleIsPositiveFinite(req.Q8Scale) {
			return core.E("rocm.hip.Projection", "q8 scale must be positive and finite", nil)
		}
		return validateHIPProjectionShape(len(req.Input), len(req.Q8), len(req.Bias), req.Rows, req.Cols)
	default:
		return core.E("rocm.hip.Projection", "projection weights are required", nil)
	}
}

func (req hipPrefillRequest) validate() error {
	if len(req.TokenIDs) == 0 && core.Trim(req.Prompt) == "" {
		return core.E("rocm.hip.Prefill", "prompt or token IDs are required", nil)
	}
	if req.CacheMode != "" && !isROCmKVCacheMode(req.CacheMode) {
		return core.E("rocm.hip.Prefill", core.Sprintf("unsupported cache mode %q", req.CacheMode), nil)
	}
	if _, _, err := hipKVVectorWidths(req.KeyWidth, req.ValueWidth); err != nil {
		return core.E("rocm.hip.Prefill", "invalid KV vector widths", err)
	}
	for _, id := range req.TokenIDs {
		if id < 0 {
			return core.E("rocm.hip.Prefill", "token IDs must be non-negative", nil)
		}
	}
	return nil
}

func validateROCmPromptBatch(operation string, prompts []string) error {
	if len(prompts) == 0 {
		return core.E(operation, "prompts are required", nil)
	}
	for index, prompt := range prompts {
		if core.Trim(prompt) == "" {
			return core.E(operation, core.Sprintf("prompt %d is empty", index), nil)
		}
	}
	return nil
}

func validateROCmChatMessages(operation string, messages []inference.Message) error {
	if len(messages) == 0 {
		return core.E(operation, "messages are required", nil)
	}
	hasContent := false
	for index, message := range messages {
		role := core.Lower(core.Trim(message.Role))
		switch role {
		case "system", "developer", "user", "assistant", "tool":
		default:
			return core.E(operation, core.Sprintf("message %d role must be system, developer, user, assistant, or tool", index), nil)
		}
		if core.Trim(message.Content) != "" {
			hasContent = true
		}
	}
	if !hasContent {
		return core.E(operation, "at least one message must contain content", nil)
	}
	return nil
}

func (req hipPrefillRequest) resolvedTokenIDs(model *hipLoadedModel) ([]int32, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	if len(req.TokenIDs) > 0 {
		tokens := append([]int32(nil), req.TokenIDs...)
		if _, err := hipTokenIDsPayload(tokens); err != nil {
			return nil, err
		}
		return tokens, nil
	}
	var tokens []int32
	if model != nil {
		tokens = model.Encode(req.Prompt)
	} else {
		tokens = approximateTokenIDs(req.Prompt)
	}
	if _, err := hipTokenIDsPayload(tokens); err != nil {
		return nil, err
	}
	return tokens, nil
}

func (req hipDecodeRequest) validate() error {
	if req.TokenID < 0 {
		return core.E("rocm.hip.Decode", "token ID must be non-negative", nil)
	}
	if req.KV == nil || req.KV.TokenCount() == 0 {
		return core.E("rocm.hip.Decode", "prefill KV cache is required", nil)
	}
	if req.DeviceKV != nil {
		if err := req.DeviceKV.CompatibleWith(req.KV); err != nil {
			return core.E("rocm.hip.Decode", "device KV cache does not match prefill KV cache", err)
		}
		if req.DescriptorTable == nil {
			return core.E("rocm.hip.Decode", "device KV cache requires descriptor table", nil)
		}
	}
	if req.DescriptorTable != nil {
		if req.DeviceKV == nil {
			return core.E("rocm.hip.Decode", "descriptor table requires device KV cache", nil)
		}
		if err := req.DescriptorTable.CompatibleWith(req.DeviceKV); err != nil {
			return core.E("rocm.hip.Decode", "descriptor table does not match device KV cache", err)
		}
	}
	if _, _, err := req.kvVectorWidths(); err != nil {
		return core.E("rocm.hip.Decode", "invalid KV vector widths", err)
	}
	return nil
}

func (req hipDecodeRequest) kvLaunchDescriptor() (rocmDeviceKVLaunchDescriptor, error) {
	if err := req.validate(); err != nil {
		return rocmDeviceKVLaunchDescriptor{}, err
	}
	if req.DeviceKV == nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.hip.DecodeLaunch", "device KV cache is required for kernel launch", nil)
	}
	return req.DeviceKV.KernelLaunchDescriptor(req.DescriptorTable)
}

func (req hipDecodeRequest) kvLaunchDescriptorBytes() ([]byte, error) {
	launch, err := req.kvLaunchDescriptor()
	if err != nil {
		return nil, err
	}
	return launch.Binary()
}

func (req hipPrefillRequest) prefillLaunchArgs(tokens *hipDeviceTokenBuffer) (hipPrefillLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipPrefillLaunchArgs{}, err
	}
	if tokens == nil || tokens.Pointer() == 0 {
		return hipPrefillLaunchArgs{}, core.E("rocm.hip.PrefillLaunch", "token buffer is required for kernel launch", nil)
	}
	if len(req.TokenIDs) > 0 && tokens.Count() != len(req.TokenIDs) {
		return hipPrefillLaunchArgs{}, core.E("rocm.hip.PrefillLaunch", "token buffer count does not match request", nil)
	}
	mode, keyWidth, valueWidth, err := req.kvConfig()
	if err != nil {
		return hipPrefillLaunchArgs{}, err
	}
	modeCode, err := rocmDeviceKVModeCode(mode)
	if err != nil {
		return hipPrefillLaunchArgs{}, err
	}
	return hipPrefillLaunchArgs{
		TokenPointer: tokens.Pointer(),
		TokenCount:   tokens.Count(),
		TokenBytes:   tokens.SizeBytes(),
		CacheMode:    mode,
		ModeCode:     modeCode,
		BlockSize:    defaultROCmKVBlockSize,
		KeyWidth:     keyWidth,
		ValueWidth:   valueWidth,
	}, nil
}

func (args hipPrefillLaunchArgs) Binary() ([]byte, error) {
	if args.TokenPointer == 0 {
		return nil, core.E("rocm.hip.PrefillLaunch", "token pointer is nil", nil)
	}
	tokenCount, err := rocmDeviceKVUint64("token count", args.TokenCount)
	if err != nil {
		return nil, err
	}
	if tokenCount == 0 {
		return nil, core.E("rocm.hip.PrefillLaunch", "token count must be positive", nil)
	}
	if args.TokenBytes == 0 {
		return nil, core.E("rocm.hip.PrefillLaunch", "token bytes must be positive", nil)
	}
	if args.TokenBytes != tokenCount*4 {
		return nil, core.E("rocm.hip.PrefillLaunch", "token byte count mismatch", nil)
	}
	if err := rocmDeviceKVValidateModeCode(args.ModeCode); err != nil {
		return nil, err
	}
	if args.CacheMode != "" {
		modeCode, err := rocmDeviceKVModeCode(args.CacheMode)
		if err != nil {
			return nil, err
		}
		if modeCode != args.ModeCode {
			return nil, core.E("rocm.hip.PrefillLaunch", "mode code mismatch", nil)
		}
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", args.BlockSize)
	if err != nil {
		return nil, err
	}
	keyWidth, err := rocmDeviceKVPositiveUint32("key width", args.KeyWidth)
	if err != nil {
		return nil, err
	}
	valueWidth, err := rocmDeviceKVPositiveUint32("value width", args.ValueWidth)
	if err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipPrefillLaunchArgsBytes)
	statusValue := args.StatusValue
	if args.StatusPointer != 0 && statusValue == 0 {
		statusValue = hipPrefillLaunchStatusOK
	}
	binary.LittleEndian.PutUint32(payload[0:], hipPrefillLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.TokenPointer))
	binary.LittleEndian.PutUint64(payload[16:], tokenCount)
	binary.LittleEndian.PutUint64(payload[24:], args.TokenBytes)
	binary.LittleEndian.PutUint32(payload[32:], args.ModeCode)
	binary.LittleEndian.PutUint32(payload[36:], blockSize)
	binary.LittleEndian.PutUint32(payload[40:], keyWidth)
	binary.LittleEndian.PutUint32(payload[44:], valueWidth)
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.StatusPointer))
	binary.LittleEndian.PutUint32(payload[56:], statusValue)
	return payload, nil
}

func (req hipDecodeRequest) decodeLaunchArgs() (hipDecodeLaunchArgs, error) {
	launch, err := req.kvLaunchDescriptor()
	if err != nil {
		return hipDecodeLaunchArgs{}, err
	}
	return hipDecodeLaunchArgs{
		TokenID:  req.TokenID,
		Position: req.KV.TokenCount(),
		KV:       launch,
	}, nil
}

func (req hipDecodeRequest) decodeLaunchArgsBytes() ([]byte, error) {
	args, err := req.decodeLaunchArgs()
	if err != nil {
		return nil, err
	}
	return args.Binary()
}

func (args hipDecodeLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipDecodeLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.TokenID < 0 {
		return nil, core.E("rocm.hip.DecodeLaunch", "token ID must be non-negative", nil)
	}
	if args.Position < 0 {
		return nil, core.E("rocm.hip.DecodeLaunch", "decode position must be non-negative", nil)
	}
	if args.Position != args.KV.TokenCount {
		return nil, core.E("rocm.hip.DecodeLaunch", "decode position must match KV token count", nil)
	}
	if cap(payload) < hipDecodeLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipDecodeLaunchArgsBytes)
	} else {
		payload = payload[:hipDecodeLaunchArgsBytes]
		clear(payload)
	}
	kvPayload, err := args.KV.BinaryInto(payload[hipDecodeLaunchArgsHeaderBytes:])
	if err != nil {
		return nil, core.E("rocm.hip.DecodeLaunch", "KV launch descriptor", err)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipDecodeLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipDecodeLaunchArgsHeaderBytes))
	binary.LittleEndian.PutUint32(payload[8:], uint32(len(payload)))
	binary.LittleEndian.PutUint32(payload[12:], uint32(args.TokenID))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.Position))
	binary.LittleEndian.PutUint32(payload[24:], uint32(len(kvPayload)))
	copy(payload[hipDecodeLaunchArgsHeaderBytes:], kvPayload)
	return payload, nil
}

func (req hipPrefillRequest) kvConfig() (string, int, int, error) {
	keyWidth, valueWidth, err := hipKVVectorWidths(req.KeyWidth, req.ValueWidth)
	if err != nil {
		return "", 0, 0, err
	}
	return firstNonEmptyString(req.CacheMode, rocmKVCacheModeFP16), keyWidth, valueWidth, nil
}

func (req hipDecodeRequest) kvVectorWidths() (int, int, error) {
	if req.KeyWidth > 0 || req.ValueWidth > 0 {
		return hipKVVectorWidths(req.KeyWidth, req.ValueWidth)
	}
	if keyWidth, valueWidth, ok := req.KV.LastVectorWidths(); ok {
		return keyWidth, valueWidth, nil
	}
	return hipKVVectorWidths(0, 0)
}

func hipKVVectorWidths(keyWidth, valueWidth int) (int, int, error) {
	if keyWidth < 0 || valueWidth < 0 {
		return 0, 0, core.E("rocm.hip.KVShape", "key and value widths must be non-negative", nil)
	}
	if keyWidth == 0 {
		keyWidth = 1
	}
	if valueWidth == 0 {
		valueWidth = keyWidth
	}
	return keyWidth, valueWidth, nil
}
