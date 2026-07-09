// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPHardwareAvailabilitySmoke_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}

	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	device := runtime.DeviceInfo()
	if device.Name == "" || device.MemoryBytes == 0 {
		t.Fatalf("device = %+v, want populated HIP device info", device)
	}

	report := newROCmBackendWithRuntime(runtime).Capabilities()
	if !report.Available ||
		report.Labels["runtime_status"] != "available" ||
		report.Labels["decode_kernel"] != hipKernelStatusNotLinked ||
		report.Labels["prefill_kernel"] != hipKernelStatusNotLinked {
		t.Fatalf("report = %+v, want available runtime with production prefill/decode not-linked status", report)
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") != "" && report.Labels["projection_kernel"] != hipKernelStatusLinked {
		t.Fatalf("report = %+v, want linked projection fixture kernel when GO_ROCM_KERNEL_HSACO is set", report)
	}
}

func TestNativeDecodeSmokeKernelStatus_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	modelPath := os.Getenv("GO_ROCM_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a local GGUF model or safetensors model pack for ROCm model smoke tests")
	}

	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModel(modelPath, inference.WithContextLen(128)))
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer model.Close()

	linkedGemma4Generate := false
	if rocmLoaded, ok := model.(*rocmModel); ok {
		if hipLoaded, ok := rocmLoaded.native.(*hipLoadedModel); ok {
			linkedGemma4Generate = hipLoadedGemma4Q4GenerateLinked(hipLoaded)
		}
	}
	if !linkedGemma4Generate {
		for range model.Generate(context.Background(), "hello", inference.WithMaxTokens(1)) {
		}
		err = resultError(model.Err())
		if err == nil || !core.Contains(err.Error(), "native decode kernels are not linked yet") {
			t.Fatalf("Generate error = %v, want explicit native decode kernel status", err)
		}
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") != "" {
		rocmLoaded, ok := model.(*rocmModel)
		if ok {
			hipLoaded, ok := rocmLoaded.native.(*hipLoadedModel)
			if ok {
				q4Embedding := assertLoadedGemma4MLXQ4EmbeddingLookupSmoke(t, hipLoaded)
				embedding := assertLoadedGemma4BF16EmbeddingLookupSmoke(t, hipLoaded)
				var layerInput []float32
				if hidden := hipLoaded.modelInfo.HiddenSize; hidden > 0 && len(embedding) >= hidden {
					scaledEmbedding := assertLoadedGemma4EmbeddingScaleSmoke(t, hipLoaded, "bf16 embedding scale", embedding[:hidden])
					layerInput = assertLoadedGemma4BF16RMSNormSmoke(t, hipLoaded, scaledEmbedding)
					projections := assertLoadedGemma4BF16ProjectionSmoke(t, hipLoaded, layerInput)
					projections = assertLoadedGemma4QKNormSmoke(t, hipLoaded, projections)
					rope := assertLoadedGemma4RoPESmoke(t, hipLoaded, projections)
					attentionOutput := assertLoadedGemma4AttentionSmoke(t, hipLoaded, rope, projections.Value)
					attentionProjection := assertLoadedGemma4OutputProjectionSmoke(t, hipLoaded, attentionOutput)
					attentionNorm := assertLoadedGemma4BF16RMSNormTensorSmoke(t, hipLoaded, "language_model.model.layers.0.post_attention_layernorm.weight", "post_attention_layernorm", attentionProjection)
					attentionResidual := assertLoadedGemma4VectorAddSmoke(t, hipLoaded, "attention residual", scaledEmbedding, attentionNorm)
					preFeedforward := assertLoadedGemma4BF16RMSNormTensorSmoke(t, hipLoaded, "language_model.model.layers.0.pre_feedforward_layernorm.weight", "pre_feedforward_layernorm", attentionResidual)
					mlpOutput := assertLoadedGemma4MLPSmoke(t, hipLoaded, preFeedforward)
					mlpNorm := assertLoadedGemma4BF16RMSNormTensorSmoke(t, hipLoaded, "language_model.model.layers.0.post_feedforward_layernorm.weight", "post_feedforward_layernorm", mlpOutput)
					mlpResidual := assertLoadedGemma4VectorAddSmoke(t, hipLoaded, "mlp residual", attentionResidual, mlpNorm)
					assertLoadedGemma4BF16LogitSmoke(t, hipLoaded, mlpResidual)
				}
				assertLoadedGemma4MLXQ4ProjectionSmoke(t, hipLoaded)
				assertLoadedGemma4MLXQ4Layer0Smoke(t, hipLoaded, q4Embedding)
				assertLoadedGemma4Q4PackagePrefillDecodeSmoke(t, hipLoaded)
				assertLoadedGemma4Q4PublicGenerateSmoke(t, model, hipLoaded)
			}
		}
	}
}

func TestNativeAttachedDrafterGenerateSmoke_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		targetPath = strings.TrimSpace(os.Getenv("GO_ROCM_PRODUCTION_MODEL_PATH"))
	}
	if targetPath == "" {
		targetPath = strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	}
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH, GO_ROCM_PRODUCTION_MODEL_PATH, or GO_ROCM_MODEL_PATH to a local Gemma4 QAT target pack")
	}
	draftPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_PATH"))
	if draftPath == "" {
		draftPath = strings.TrimSpace(os.Getenv("GO_ROCM_DRAFT_MODEL_PATH"))
	}
	if draftPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_DRAFT_PATH or GO_ROCM_DRAFT_MODEL_PATH to a local Gemma4 MTP-QAT assistant pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}
	maxTokens := 16
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		maxTokens = value
	}

	backend := newROCmBackendWithRuntime(newSystemNativeRuntime())
	pair, err := backend.LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetOptions: []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		DraftOptions:  []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
	})
	if err != nil {
		t.Fatalf("LoadAttachedDrafterPair(%q, %q): %v", targetPath, draftPath, err)
	}
	pairClosed := false
	defer func() {
		if !pairClosed {
			_ = pair.Close()
		}
	}()
	if !pair.NativeReady() {
		t.Fatalf("attached drafter native ready = false labels=%+v error=%q", pair.Attachment.Labels, pair.NativeError)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil {
		t.Fatalf("pair target = %T, want *rocmModel", pair.Target)
	}

	draftTokens := pair.Plan.DraftTokens
	result, err := pair.GenerateNative(context.Background(), prompt, AttachedDrafterGenerateConfig{
		MaxTokens:   maxTokens,
		DraftTokens: draftTokens,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("GenerateNative(%q): %v", prompt, err)
	}
	if result.Text == "" {
		t.Fatalf("GenerateNative(%q) returned empty text; metrics=%+v", prompt, result.Metrics)
	}
	if result.Metrics.DraftCalls == 0 {
		t.Fatalf("GenerateNative metrics = %+v, want assistant draft calls", result.Metrics)
	}
	if result.Metrics.TargetCalls == 0 {
		t.Fatalf("GenerateNative metrics = %+v, want target verification calls", result.Metrics)
	}
	if result.Metrics.AcceptedTokens+result.Metrics.RejectedTokens != result.Metrics.DraftTokens {
		t.Fatalf("GenerateNative metrics = %+v, want accepted+rejected to match draft tokens", result.Metrics)
	}
	if err := pair.Close(); err != nil {
		t.Fatalf("close attached drafter pair before reference target load: %v", err)
	}
	pairClosed = true

	referenceModel, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModel(targetPath, inference.WithContextLen(defaultContextLengthCap)))
	if err != nil {
		t.Fatalf("LoadModel reference target %q: %v", targetPath, err)
	}
	defer referenceModel.Close()
	targetText := strings.Join(collectTokenText(referenceModel.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens), inference.WithTemperature(0))), "")
	if err := resultError(referenceModel.Err()); err != nil {
		t.Fatalf("reference target Generate(%q): %v", prompt, err)
	}
	if targetText == "" {
		t.Fatalf("reference target Generate(%q) returned empty text", prompt)
	}
	if result.Text == targetText {
		t.Logf("native attached smoke exact-match: text=%q metrics=%+v", result.Text, result.Metrics)
	} else {
		assertNativeAttachedDrafterTargetARMatchStable(t, targetPath, draftPath, prompt, maxTokens, draftTokens, result.Text, targetText)
	}
}

func assertNativeAttachedDrafterTargetARMatchStable(t *testing.T, targetPath, draftPath, prompt string, maxTokens, draftTokens int, nativeText, targetText string) {
	t.Helper()
	targetAgain := loadNativeAttachedDrafterReferenceText(t, targetPath, prompt, maxTokens)
	if targetAgain != targetText {
		t.Skipf("reference target Generate(%q) shifted between runs (%q -> %q); attached-drafter equivalence comparison is not stable", prompt, targetText, targetAgain)
	}
	nativeAgain := loadNativeAttachedDrafterText(t, targetPath, draftPath, prompt, maxTokens, draftTokens)
	if nativeAgain != nativeText {
		t.Skipf("native attached Generate(%q) shifted between runs (%q -> %q); attached-drafter equivalence comparison is not stable", prompt, nativeText, nativeAgain)
	}
	t.Fatalf("native attached drafter text differs from stable target AR route: native=%q target=%q", nativeText, targetText)
}

func loadNativeAttachedDrafterReferenceText(t *testing.T, targetPath, prompt string, maxTokens int) string {
	t.Helper()
	referenceModel, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModel(targetPath, inference.WithContextLen(defaultContextLengthCap)))
	if err != nil {
		t.Fatalf("LoadModel reference target %q: %v", targetPath, err)
	}
	defer referenceModel.Close()
	targetText := strings.Join(collectTokenText(referenceModel.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens), inference.WithTemperature(0))), "")
	if err := resultError(referenceModel.Err()); err != nil {
		t.Fatalf("reference target Generate(%q): %v", prompt, err)
	}
	if targetText == "" {
		t.Fatalf("reference target Generate(%q) returned empty text", prompt)
	}
	return targetText
}

func loadNativeAttachedDrafterText(t *testing.T, targetPath, draftPath, prompt string, maxTokens, draftTokens int) string {
	t.Helper()
	pair, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetOptions: []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		DraftOptions:  []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
	})
	if err != nil {
		t.Fatalf("LoadAttachedDrafterPair(%q, %q): %v", targetPath, draftPath, err)
	}
	defer pair.Close()
	if !pair.NativeReady() {
		t.Fatalf("attached drafter native ready = false labels=%+v error=%q", pair.Attachment.Labels, pair.NativeError)
	}
	result, err := pair.GenerateNative(context.Background(), prompt, AttachedDrafterGenerateConfig{
		MaxTokens:   maxTokens,
		DraftTokens: draftTokens,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("GenerateNative(%q): %v", prompt, err)
	}
	if result.Text == "" {
		t.Fatalf("GenerateNative(%q) returned empty text; metrics=%+v", prompt, result.Metrics)
	}
	return result.Text
}

func assertLoadedGemma4BF16EmbeddingLookupSmoke(t *testing.T, model *hipLoadedModel) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		model.modelInfo.QuantBits != 0 {
		return nil
	}
	tensor, ok := model.tensors["language_model.model.embed_tokens.weight"]
	if !ok {
		t.Fatalf("loaded Gemma4 BF16 model is missing embed_tokens tensor")
	}
	if tensor.info.TypeName != "BF16" ||
		len(tensor.info.Dimensions) != 2 ||
		tensor.info.Dimensions[0] != 262144 ||
		tensor.info.Dimensions[1] != 1536 ||
		tensor.info.ByteSize != uint64(262144*1536*2) {
		t.Fatalf("embed_tokens tensor = %+v, want Gemma4 BF16 [262144,1536]", tensor.info)
	}
	vocab := int(tensor.info.Dimensions[0])
	hidden := int(tensor.info.Dimensions[1])
	tokenIDs := []int32{0, 1, 257}
	got, err := hipRunEmbeddingLookupKernelWithDeviceTable(context.Background(), model.driver, tokenIDs, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: tensor.pointer,
		EmbeddingBytes:   tensor.info.ByteSize,
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	})
	core.RequireNoError(t, err)
	want := readLoadedBF16EmbeddingRows(t, tensor, tokenIDs, hidden)
	assertFloat32SlicesNear(t, want, got, 0)
	return got
}

func assertLoadedGemma4MLXQ4EmbeddingLookupSmoke(t *testing.T, model *hipLoadedModel) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return nil
	}
	bits := hipMLXQ4ProjectionBitsOrDefault(model.modelInfo.QuantBits)
	weight, ok := model.tensors["language_model.model.embed_tokens.weight"]
	if !ok {
		t.Fatalf("loaded Gemma4 q%d model is missing embed_tokens packed weight tensor", bits)
	}
	scales, ok := model.tensors["language_model.model.embed_tokens.scales"]
	if !ok {
		t.Fatalf("loaded Gemma4 q%d model is missing embed_tokens scales tensor", bits)
	}
	biases, ok := model.tensors["language_model.model.embed_tokens.biases"]
	if !ok {
		t.Fatalf("loaded Gemma4 q%d model is missing embed_tokens biases tensor", bits)
	}
	vocab := model.modelInfo.VocabSize
	hidden := model.modelInfo.HiddenSize
	groupSize := model.modelInfo.QuantGroup
	if groupSize == 0 {
		groupSize = 64
	}
	packedPerRow, err := hipMLXAffinePackedCols(hidden, bits)
	core.RequireNoError(t, err)
	groups := hidden / groupSize
	if vocab != 262144 || hidden != 1536 || groupSize != 64 {
		t.Fatalf("loaded Gemma4 q%d dimensions vocab=%d hidden=%d group=%d, want 262144/1536/64", bits, vocab, hidden, groupSize)
	}
	if weight.info.TypeName != "U32" ||
		len(weight.info.Dimensions) != 2 ||
		weight.info.Dimensions[0] != uint64(vocab) ||
		weight.info.Dimensions[1] != uint64(packedPerRow) ||
		weight.info.ByteSize != uint64(vocab*packedPerRow*4) {
		t.Fatalf("q%d embed_tokens weight tensor = %+v, want Gemma4 q%d [%d,%d]", bits, weight.info, bits, vocab, packedPerRow)
	}
	for label, tensor := range map[string]hipTensor{"scales": scales, "biases": biases} {
		if tensor.info.TypeName != "BF16" ||
			len(tensor.info.Dimensions) != 2 ||
			tensor.info.Dimensions[0] != uint64(vocab) ||
			tensor.info.Dimensions[1] != uint64(groups) ||
			tensor.info.ByteSize != uint64(vocab*groups*2) {
			t.Fatalf("q%d embed_tokens %s tensor = %+v, want Gemma4 q%d [%d,%d]", bits, label, tensor.info, bits, vocab, groups)
		}
	}
	tokenIDs := []int32{0, 1, 257}
	got, err := hipRunEmbeddingLookupKernelWithDeviceTable(context.Background(), model.driver, tokenIDs, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: weight.pointer,
		EmbeddingBytes:   weight.info.ByteSize,
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        vocab,
		HiddenSize:       hidden,
		GroupSize:        groupSize,
		ScalePointer:     scales.pointer,
		BiasPointer:      biases.pointer,
		ScaleBytes:       scales.info.ByteSize,
		BiasBytes:        biases.info.ByteSize,
		QuantBits:        bits,
	})
	core.RequireNoError(t, err)
	wantWeights := readLoadedUint32EmbeddingRows(t, weight, tokenIDs, packedPerRow)
	wantScales := readLoadedBF16TensorRowsByID(t, scales, tokenIDs, groups)
	wantBiases := readLoadedBF16TensorRowsByID(t, biases, tokenIDs, groups)
	want, err := hipReferenceMLXAffineEmbeddingLookup(wantWeights, wantScales, wantBiases, len(tokenIDs), hidden, groupSize, []int32{0, 1, 2}, bits)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.01)
	return got
}

func assertLoadedGemma4Q4PackagePrefillDecodeSmoke(t *testing.T, model *hipLoadedModel) {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return
	}
	prefill, err := model.Prefill(context.Background(), hipPrefillRequest{
		TokenIDs:  []int32{0},
		CacheMode: rocmKVCacheModeKQ8VQ4,
	})
	if err != nil {
		t.Fatalf("Gemma4 q4 package Prefill: %v", err)
	}
	defer prefill.Gemma4Q4DeviceState.Close()
	if prefill.PromptTokens != 1 ||
		len(prefill.Logits) != model.modelInfo.VocabSize ||
		len(prefill.Gemma4Q4State.Layers) != model.modelInfo.NumLayers ||
		prefill.Gemma4Q4DeviceState == nil ||
		prefill.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_prefill" ||
		prefill.Labels["gemma4_q4_prefill_kernel"] != hipKernelStatusLinked ||
		prefill.Labels["prefill_kernel"] != hipKernelStatusNotLinked ||
		prefill.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		prefill.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		prefill.Labels["production_decode"] != hipKernelStatusNotLinked ||
		prefill.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
		t.Fatalf("Gemma4 q4 package Prefill result labels=%+v prompt=%d logits=%d layers=%d device=%v, want experimental q4 package state",
			prefill.Labels, prefill.PromptTokens, len(prefill.Logits), len(prefill.Gemma4Q4State.Layers), prefill.Gemma4Q4DeviceState != nil)
	}
	nextToken, _, err := hipReferenceGreedySample(prefill.Logits)
	if err != nil {
		t.Fatalf("Gemma4 q4 package Prefill greedy sample: %v", err)
	}
	decode, err := model.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:             int32(nextToken),
		DeviceKVMode:        rocmKVCacheModeKQ8VQ4,
		Gemma4Q4State:       prefill.Gemma4Q4State,
		Gemma4Q4DeviceState: prefill.Gemma4Q4DeviceState,
	})
	if err != nil {
		t.Fatalf("Gemma4 q4 package DecodeToken: %v", err)
	}
	defer decode.Gemma4Q4DeviceState.Close()
	if decode.Token.ID < 0 ||
		int(decode.Token.ID) >= model.modelInfo.VocabSize ||
		len(decode.Logits) != model.modelInfo.VocabSize ||
		len(decode.Gemma4Q4State.Layers) != model.modelInfo.NumLayers ||
		decode.Gemma4Q4DeviceState == nil ||
		prefill.Gemma4Q4DeviceState.closed != true ||
		decode.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_decode" ||
		decode.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
		decode.Labels["decode_kernel"] != hipKernelStatusNotLinked ||
		decode.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
		decode.Labels["production_prefill"] != hipKernelStatusNotLinked ||
		decode.Labels["production_decode"] != hipKernelStatusNotLinked ||
		decode.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
		t.Fatalf("Gemma4 q4 package DecodeToken result token=%+v labels=%+v logits=%d layers=%d device=%v priorClosed=%v, want experimental q4 package decode state",
			decode.Token, decode.Labels, len(decode.Logits), len(decode.Gemma4Q4State.Layers), decode.Gemma4Q4DeviceState != nil, prefill.Gemma4Q4DeviceState.closed)
	}
	t.Logf("Gemma4 q4 package Prefill/Decode prompt=[0] next=%d decoded=%d text=%q", nextToken, decode.Token.ID, decode.Token.Text)
}

func assertLoadedGemma4BF16RMSNormSmoke(t *testing.T, model *hipLoadedModel, input []float32) []float32 {
	t.Helper()
	return assertLoadedGemma4BF16RMSNormTensorSmoke(t, model, "language_model.model.layers.0.input_layernorm.weight", "input_layernorm", input)
}

func assertLoadedGemma4BF16RMSNormTensorSmoke(t *testing.T, model *hipLoadedModel, tensorName, label string, input []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if len(input) != hidden {
		t.Fatalf("%s rms input length = %d, want hidden size %d", label, len(input), hidden)
	}
	return assertLoadedGemma4BF16RMSNormVectorSmoke(t, model, tensorName, label, input)
}

func assertLoadedGemma4BF16RMSNormVectorSmoke(t *testing.T, model *hipLoadedModel, tensorName, label string, input []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	if len(input) == 0 {
		t.Fatalf("%s rms input is empty", label)
	}
	tensor, ok := model.tensors[tensorName]
	if !ok {
		t.Fatalf("loaded Gemma4 model is missing %s tensor", label)
	}
	if tensor.info.TypeName != "BF16" ||
		len(tensor.info.Dimensions) != 1 ||
		tensor.info.Dimensions[0] != uint64(len(input)) ||
		tensor.info.ByteSize != uint64(len(input)*2) {
		t.Fatalf("%s tensor = %+v, want Gemma4 BF16 [%d]", label, tensor.info, len(input))
	}
	got, err := hipRunRMSNormKernelWithDeviceWeightConfig(context.Background(), model.driver, input, hipRMSNormDeviceWeightConfig{
		WeightPointer:  tensor.pointer,
		WeightBytes:    tensor.info.ByteSize,
		Count:          len(input),
		Epsilon:        1e-6,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	})
	core.RequireNoError(t, err)

	bf16Weights := readLoadedBF16TensorRows(t, tensor, 1, len(input))
	weights := make([]float32, len(bf16Weights))
	for index, value := range bf16Weights {
		weights[index] = hipBFloat16ToFloat32(value)
	}
	want, err := hipReferenceRMSNorm(input, weights, 1e-6)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.001)
	return got
}

type gemma4BF16ProjectionSpec struct {
	label      string
	tensorName string
	rows       int
	cols       int
}

type gemma4BF16AttentionProjectionOutputs struct {
	Query []float32
	Key   []float32
	Value []float32
}

type gemma4RoPEOutputs struct {
	QueryHeads [][]float32
	Key        []float32
}

func assertLoadedGemma4BF16ProjectionSmoke(t *testing.T, model *hipLoadedModel, layerInput []float32) gemma4BF16AttentionProjectionOutputs {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		model.modelInfo.QuantBits != 0 {
		return gemma4BF16AttentionProjectionOutputs{}
	}
	hidden := model.modelInfo.HiddenSize
	input := layerInput
	if len(input) != hidden {
		input = make([]float32, hidden)
		for index := range input {
			input[index] = float32((index%7)-3) * 0.125
		}
		input[0] = 1
	}

	qOutput := assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "q_proj",
		tensorName: "language_model.model.layers.0.self_attn.q_proj.weight",
		rows:       2048,
		cols:       hidden,
	}, input)
	kOutput := assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "k_proj",
		tensorName: "language_model.model.layers.0.self_attn.k_proj.weight",
		rows:       256,
		cols:       hidden,
	}, input)
	vOutput := assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "v_proj",
		tensorName: "language_model.model.layers.0.self_attn.v_proj.weight",
		rows:       256,
		cols:       hidden,
	}, input)
	return gemma4BF16AttentionProjectionOutputs{Query: qOutput, Key: kOutput, Value: vOutput}
}

func assertLoadedGemma4QKNormSmoke(t *testing.T, model *hipLoadedModel, projections gemma4BF16AttentionProjectionOutputs) gemma4BF16AttentionProjectionOutputs {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return projections
	}
	const headDim = 256
	const queryHeads = 8
	if len(projections.Query) != queryHeads*headDim || len(projections.Key) != headDim || len(projections.Value) != headDim {
		t.Fatalf("Gemma4 q/k norm inputs q=%d k=%d v=%d, want %d query heads and one %d-dim k/v", len(projections.Query), len(projections.Key), len(projections.Value), queryHeads, headDim)
	}
	query := make([]float32, 0, len(projections.Query))
	for head := 0; head < queryHeads; head++ {
		start := head * headDim
		end := start + headDim
		query = append(query, assertLoadedGemma4BF16RMSNormVectorSmoke(t, model, "language_model.model.layers.0.self_attn.q_norm.weight", core.Sprintf("q_norm head%d", head), projections.Query[start:end])...)
	}
	key := assertLoadedGemma4BF16RMSNormVectorSmoke(t, model, "language_model.model.layers.0.self_attn.k_norm.weight", "k_norm", projections.Key)
	return gemma4BF16AttentionProjectionOutputs{Query: query, Key: key, Value: projections.Value}
}

func assertLoadedGemma4BF16ProjectionTensorSmoke(t *testing.T, model *hipLoadedModel, spec gemma4BF16ProjectionSpec, input []float32) []float32 {
	t.Helper()
	if len(input) != spec.cols {
		t.Fatalf("%s input length = %d, want cols %d", spec.label, len(input), spec.cols)
	}
	tensor, ok := model.tensors[spec.tensorName]
	if !ok {
		t.Fatalf("loaded Gemma4 BF16 model is missing layer-0 %s tensor", spec.label)
	}
	if tensor.info.TypeName != "BF16" ||
		len(tensor.info.Dimensions) != 2 ||
		tensor.info.Dimensions[0] != uint64(spec.rows) ||
		tensor.info.Dimensions[1] != uint64(spec.cols) ||
		tensor.info.ByteSize != uint64(spec.rows*spec.cols*2) {
		t.Fatalf("%s tensor = %+v, want Gemma4 BF16 [%d,%d]", spec.label, tensor.info, spec.rows, spec.cols)
	}

	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(model.driver, "rocm.hip.Gemma4BF16ProjectionSmoke", "gemma4 "+spec.label+" input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	outputBuffer, err := hipAllocateByteBuffer(model.driver, "rocm.hip.Gemma4BF16ProjectionSmoke", "gemma4 "+spec.label+" output", uint64(spec.rows*4), spec.rows)
	core.RequireNoError(t, err)
	defer outputBuffer.Close()

	launch, err := (hipProjectionLaunchArgs{
		InputPointer:   inputBuffer.Pointer(),
		InputCount:     spec.cols,
		InputBytes:     inputBuffer.SizeBytes(),
		WeightPointer:  tensor.pointer,
		WeightBytes:    tensor.info.ByteSize,
		OutputPointer:  outputBuffer.Pointer(),
		OutputBytes:    outputBuffer.SizeBytes(),
		Rows:           spec.rows,
		Cols:           spec.cols,
		WeightEncoding: hipProjectionWeightEncodingBF16,
	}).Binary()
	core.RequireNoError(t, err)
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, launch, spec.rows)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(model.driver, config))
	output, err := (&hipProjectionDeviceBuffers{Output: outputBuffer, Rows: spec.rows}).ReadOutput()
	core.RequireNoError(t, err)

	compareRows := 8
	if spec.rows < compareRows {
		compareRows = spec.rows
	}
	expectedWeights := readLoadedBF16TensorRows(t, tensor, compareRows, spec.cols)
	expected, err := hipReferenceBF16Projection(input, expectedWeights, compareRows, spec.cols, nil)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, expected, output[:compareRows], 0.05)
	return output
}

func assertLoadedGemma4MLXQ4Layer0Smoke(t *testing.T, model *hipLoadedModel, embedding []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if hidden <= 0 || len(embedding) < hidden {
		t.Fatalf("Gemma4 q4 embedding length = %d, want at least hidden size %d", len(embedding), hidden)
	}
	var allLayers hipGemma4Q4ForwardConfig
	if model.modelInfo.NumLayers > 1 {
		var err error
		allLayers, err = model.loadedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
		core.RequireNoError(t, err)
		if len(allLayers.Layers) != model.modelInfo.NumLayers {
			t.Fatalf("Gemma4 q4 loaded layer configs = %d, want %d", len(allLayers.Layers), model.modelInfo.NumLayers)
		}
	}
	cfg, err := model.loadedGemma4Q4ForwardConfig(1)
	core.RequireNoError(t, err)
	result, err := hipRunGemma4Q4SingleTokenForward(context.Background(), model.driver, cfg, hipGemma4Q4ForwardRequest{
		TokenID:  0,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, embedding[:hidden], result.Embedding, 0)
	if len(result.FinalHidden) != hidden || len(result.Logits) != model.modelInfo.VocabSize {
		t.Fatalf("Gemma4 q4 layer0 result final=%d logits=%d, want hidden=%d vocab=%d", len(result.FinalHidden), len(result.Logits), hidden, model.modelInfo.VocabSize)
	}
	wantToken, wantScore, err := hipReferenceGreedySample(result.Logits)
	core.RequireNoError(t, err)
	if result.Greedy.TokenID != wantToken || math.Abs(float64(result.Greedy.Score-wantScore)) > 0.0001 {
		t.Fatalf("Gemma4 q4 layer0 greedy output = %+v, want token %d score %f", result.Greedy, wantToken, wantScore)
	}
	forwardLayers := allLayers.Layers
	if len(forwardLayers) == 0 {
		forwardLayers = cfg.Layers
	}
	if layerCount, ok := gemma4Q4ForwardLayerCountFromEnv(t, len(forwardLayers)); ok {
		forwardCfg := hipGemma4Q4ForwardConfig{Layers: forwardLayers[:layerCount]}
		forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), model.driver, forwardCfg, hipGemma4Q4ForwardRequest{
			TokenID:  0,
			Position: 1,
			Epsilon:  1e-6,
		})
		core.RequireNoError(t, err)
		if len(forward.LayerResults) != layerCount ||
			len(forward.FinalHidden) != hidden ||
			len(forward.Logits) != model.modelInfo.VocabSize ||
			forward.Labels["decode_layers"] != strconv.Itoa(layerCount) ||
			forward.Labels["production_decode"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 %d-layer forward result layers=%d final=%d logits=%d labels=%+v, want single-token smoke with production decode still not linked",
				layerCount, len(forward.LayerResults), len(forward.FinalHidden), len(forward.Logits), forward.Labels)
		}
		t.Logf("Gemma4 q4 %d-layer single-token forward greedy token=%d score=%f", layerCount, forward.Greedy.TokenID, forward.Greedy.Score)
	}
	if decodeLayers, decodeTokens, ok := gemma4Q4DecodeEnv(t, len(forwardLayers)); ok {
		decodeCfg := hipGemma4Q4ForwardConfig{Layers: forwardLayers[:decodeLayers]}
		promptTokens := gemma4Q4DecodePromptTokensEnv(t, model.modelInfo.VocabSize)
		decode, err := hipRunGemma4Q4GreedyDecode(context.Background(), model.driver, decodeCfg, hipGemma4Q4GreedyDecodeRequest{
			PromptTokenIDs:    promptTokens,
			MaxNewTokens:      decodeTokens,
			Position:          1,
			Epsilon:           1e-6,
			MirrorDeviceKV:    true,
			DeviceKVAttention: true,
			DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		})
		core.RequireNoError(t, err)
		defer decode.DeviceState.Close()
		wantSteps := len(promptTokens) + decodeTokens - 1
		if len(decode.Generated) != decodeTokens ||
			len(decode.StepResults) != wantSteps ||
			len(decode.State.Layers) != decodeLayers ||
			decode.Labels["decode_layers"] != strconv.Itoa(decodeLayers) ||
			decode.Labels["decode_prompt_tokens"] != strconv.Itoa(len(promptTokens)) ||
			decode.Labels["decode_generated_tokens"] != strconv.Itoa(decodeTokens) ||
			decode.Labels["decode_forward_steps"] != strconv.Itoa(wantSteps) ||
			decode.Labels["decode_state_tokens"] != strconv.Itoa(wantSteps) ||
			decode.Labels["production_decode"] != hipKernelStatusNotLinked ||
			decode.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 greedy decode labels/results generated=%d steps=%d state_layers=%d labels=%+v, want experimental token decode with production decode/cache still not linked",
				len(decode.Generated), len(decode.StepResults), len(decode.State.Layers), decode.Labels)
		}
		deviceState := decode.DeviceState
		if deviceState == nil {
			t.Fatalf("Gemma4 q4 decode device state is nil, want carried HIP mirror")
		}
		deviceLabels := deviceState.Labels()
		if deviceLabels["gemma4_q4_device_kv_backing"] != "hip_device_mirror" ||
			deviceLabels["gemma4_q4_device_kv_layers"] != strconv.Itoa(decodeLayers) ||
			deviceLabels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 device KV labels = %+v, want HIP mirror labels with production cache pending", deviceLabels)
		}
		if len(decode.StepResults) == 0 ||
			decode.StepResults[0].Labels["attention_kv_backing"] != "hip_device_descriptor" ||
			decode.StepResults[0].Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 {
			t.Fatalf("Gemma4 q4 decode step labels = %+v, want descriptor-backed attention labels", decode.StepResults)
		}
		restoredState, err := deviceState.HostState()
		core.RequireNoError(t, err)
		assertGemma4Q4DeviceStateMatchesQuantizedHost(t, decodeCfg, decode.State, restoredState, deviceState, rocmKVCacheModeKQ8VQ4)
		core.RequireNoError(t, deviceState.Close())
		t.Logf("Gemma4 q4 %d-layer greedy decode prompt=%v generated tokens=%v", decodeLayers, promptTokens, hipGemma4Q4GreedyTokenIDs(decode.Generated))
	}
	if len(allLayers.Layers) > 15 {
		for _, check := range []struct {
			layer        int
			headDim      int
			intermediate int
		}{
			{layer: 4, headDim: 512, intermediate: 6144},
			{layer: 15, headDim: 256, intermediate: 12288},
		} {
			layerCfg := allLayers.Layers[check.layer]
			wantSlidingWindow := hipGemma4Q4EffectiveSlidingWindow(check.headDim, model.contextSize)
			if layerCfg.HeadDim != check.headDim ||
				layerCfg.QueryHeads != 8 ||
				layerCfg.IntermediateSize != check.intermediate ||
				layerCfg.RoPEBase != hipGemma4Q4LayerRoPEBase(check.headDim) ||
				layerCfg.RoPERotaryDim != hipGemma4Q4LayerRoPERotaryDim(check.headDim) ||
				layerCfg.SlidingWindow != wantSlidingWindow {
				t.Fatalf("Gemma4 q4 layer %d config head=%d qheads=%d intermediate=%d rope=%f rotary=%d sliding=%d, want head=%d qheads=8 intermediate=%d rope=%f rotary=%d sliding=%d",
					check.layer, layerCfg.HeadDim, layerCfg.QueryHeads, layerCfg.IntermediateSize, layerCfg.RoPEBase, layerCfg.RoPERotaryDim, layerCfg.SlidingWindow, check.headDim, check.intermediate, hipGemma4Q4LayerRoPEBase(check.headDim), hipGemma4Q4LayerRoPERotaryDim(check.headDim), wantSlidingWindow)
			}
			layerOutput, err := hipRunGemma4Q4DecoderLayer(context.Background(), model.driver, layerCfg, result.ScaledEmbedding, hipGemma4Q4DecoderLayerRequest{
				Position: 1,
				Epsilon:  1e-6,
			})
			core.RequireNoError(t, err)
			if len(layerOutput.AttentionOutput) != layerCfg.QueryHeads*layerCfg.HeadDim ||
				len(layerOutput.FinalHidden) != hidden {
				t.Fatalf("Gemma4 q4 layer %d output attention=%d final=%d, want attention=%d final=%d",
					check.layer, len(layerOutput.AttentionOutput), len(layerOutput.FinalHidden), layerCfg.QueryHeads*layerCfg.HeadDim, hidden)
			}
		}
	}
	if result.Labels["production_decode"] != hipKernelStatusNotLinked ||
		result.Labels["decode_layers"] != "1" ||
		!core.Contains(result.Labels["decode_primitives"], "mlx_q4_projection") {
		t.Fatalf("Gemma4 q4 layer0 labels = %+v, want q4 primitive smoke with production decode still not linked", result.Labels)
	}
	return result.FinalHidden
}

func assertLoadedGemma4Q4PublicGenerateSmoke(t *testing.T, textModel inference.TextModel, loaded *hipLoadedModel) {
	t.Helper()
	if textModel == nil ||
		loaded == nil ||
		!isROCmGemma4Architecture(loaded.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(loaded.modelInfo.QuantBits) {
		return
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT"))
	if prompt == "" {
		return
	}
	promptTokens, tokenPrompt, err := hipGemma4Q4TokenPromptIDs(prompt, loaded.modelInfo.VocabSize)
	if err != nil {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT=%q token prompt parse failed: %v", prompt, err)
	}
	if !tokenPrompt {
		promptTokens, tokenPrompt, err = hipGemma4Q4TextPromptIDs(prompt, loaded)
		if err != nil || !tokenPrompt {
			t.Fatalf("GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT=%q must be a valid tokens: or text: prompt: %v", prompt, err)
		}
	}
	if tokenizer, ok := any(textModel).(inference.TokenizerModel); ok {
		encoded := tokenizer.Encode("Hello world")
		core.AssertEqual(t, []int32{2, 9259, 1902}, encoded)
		core.AssertEqual(t, "Hello world", tokenizer.Decode(encoded))
	}
	if reporter, ok := any(textModel).(inference.CapabilityReporter); ok {
		report := reporter.Capabilities()
		generate, ok := report.Capability(inference.CapabilityGenerate)
		if !ok || generate.Status != inference.CapabilityStatusExperimental ||
			generate.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_generate" ||
			generate.Labels["gemma4_q4_decode_kernel"] != hipKernelStatusLinked ||
			generate.Labels["attention_kv_backing"] != "hip_device_descriptor" ||
			generate.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
			generate.Labels["gemma4_q4_device_kv_state"] != "forward_returned_device_state" ||
			generate.Labels["production_prefill"] != hipKernelStatusNotLinked ||
			generate.Labels["production_decode"] != hipKernelStatusNotLinked ||
			generate.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 Generate capability = %+v ok=%v, want experimental q4 route with production prefill/decode pending", generate, ok)
		}
		batch, ok := report.Capability(inference.CapabilityBatchGenerate)
		if !ok || batch.Status != inference.CapabilityStatusExperimental ||
			batch.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_batch_generate" ||
			batch.Labels["batch_generate_kernel"] != hipKernelStatusLinked ||
			batch.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
			batch.Labels["production_prefill"] != hipKernelStatusNotLinked ||
			batch.Labels["production_decode"] != hipKernelStatusNotLinked ||
			batch.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 BatchGenerate capability = %+v ok=%v, want experimental q4 route with production prefill/decode pending", batch, ok)
		}
		chat, ok := report.Capability(inference.CapabilityChat)
		if !ok || chat.Status != inference.CapabilityStatusExperimental ||
			chat.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_chat" ||
			chat.Labels["chat_kernel"] != hipKernelStatusLinked ||
			chat.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
			chat.Labels["production_prefill"] != hipKernelStatusNotLinked ||
			chat.Labels["production_decode"] != hipKernelStatusNotLinked ||
			chat.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 Chat capability = %+v ok=%v, want experimental q4 route with production prefill/decode pending", chat, ok)
		}
		classify, ok := report.Capability(inference.CapabilityClassify)
		if !ok || classify.Status != inference.CapabilityStatusExperimental ||
			classify.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_classify" ||
			classify.Labels["classify_kernel"] != hipKernelStatusLinked ||
			classify.Labels["classify_logits_source"] != "gemma4_mlx_affine_package_prefill" ||
			classify.Labels["attention_kv_mode"] != rocmKVCacheModeKQ8VQ4 ||
			classify.Labels["production_prefill"] != hipKernelStatusNotLinked ||
			classify.Labels["production_decode"] != hipKernelStatusNotLinked ||
			classify.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 Classify capability = %+v ok=%v, want experimental q4 route with production prefill pending", classify, ok)
		}
		speculative, ok := report.Capability(inference.CapabilitySpeculativeDecode)
		if !ok || speculative.Status != inference.CapabilityStatusExperimental ||
			speculative.Labels["attached_drafter_helper"] != hipKernelStatusLinked ||
			speculative.Labels["attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
			speculative.Labels["attached_drafter_role"] != "gemma4_assistant" ||
			speculative.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_speculative_decode" ||
			speculative.Labels["speculative_decode_helper"] != hipKernelStatusLinked ||
			speculative.Labels["speculative_decode_source"] != "gemma4_q4_generate" ||
			speculative.Labels["production_prefill"] != hipKernelStatusNotLinked ||
			speculative.Labels["production_decode"] != hipKernelStatusNotLinked ||
			speculative.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 SpeculativeDecode capability = %+v ok=%v, want experimental q4 helper with production prefill/decode pending", speculative, ok)
		}
		promptLookup, ok := report.Capability(inference.CapabilityPromptLookupDecode)
		if !ok || promptLookup.Status != inference.CapabilityStatusExperimental ||
			promptLookup.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_prompt_lookup_decode" ||
			promptLookup.Labels["prompt_lookup_decode_helper"] != hipKernelStatusLinked ||
			promptLookup.Labels["prompt_lookup_decode_source"] != "gemma4_q4_generate" ||
			promptLookup.Labels["production_prefill"] != hipKernelStatusNotLinked ||
			promptLookup.Labels["production_decode"] != hipKernelStatusNotLinked ||
			promptLookup.Labels["production_kv_cache_backing"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 PromptLookupDecode capability = %+v ok=%v, want experimental q4 helper with production prefill/decode pending", promptLookup, ok)
		}
	}
	tokenCount := 2
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		tokenCount = value
	}
	var generated []inference.Token
	for token := range textModel.Generate(context.Background(), prompt, inference.WithMaxTokens(tokenCount)) {
		generated = append(generated, token)
	}
	if err := resultError(textModel.Err()); err != nil {
		t.Fatalf("Gemma4 q4 public Generate(%q) error = %v", prompt, err)
	}
	if len(generated) != tokenCount {
		t.Fatalf("Gemma4 q4 public Generate(%q) emitted %d tokens, want %d: %+v", prompt, len(generated), tokenCount, generated)
	}
	ids := make([]int32, len(generated))
	texts := make([]string, len(generated))
	for index, token := range generated {
		if token.ID < 0 || int(token.ID) >= loaded.modelInfo.VocabSize {
			t.Fatalf("Gemma4 q4 public Generate token[%d]=%+v outside vocab size %d", index, token, loaded.modelInfo.VocabSize)
		}
		if token.Text == "" || strings.HasPrefix(token.Text, "<token:") {
			t.Fatalf("Gemma4 q4 public Generate token[%d] text=%q, want decoded tokenizer text for ID %d", index, token.Text, token.ID)
		}
		ids[index] = token.ID
		texts[index] = token.Text
	}
	metrics := textModel.Metrics()
	if metrics.GeneratedTokens != tokenCount {
		t.Fatalf("Gemma4 q4 public Generate metrics generated=%d, want %d", metrics.GeneratedTokens, tokenCount)
	}
	if !strings.Contains(prompt, ":") && metrics.PromptTokens != len(promptTokens) {
		t.Fatalf("Gemma4 q4 public Generate metrics prompt=%d, want tokenizer prompt length %d", metrics.PromptTokens, len(promptTokens))
	}
	batch, err := resultValue[[]inference.BatchResult](textModel.BatchGenerate(context.Background(), []string{prompt}, inference.WithMaxTokens(1)))
	if err != nil {
		t.Fatalf("Gemma4 q4 public BatchGenerate: %v", err)
	}
	if len(batch) != 1 || batch[0].Err != nil || len(batch[0].Tokens) != 1 ||
		batch[0].Tokens[0].ID < 0 ||
		int(batch[0].Tokens[0].ID) >= loaded.modelInfo.VocabSize {
		t.Fatalf("Gemma4 q4 public BatchGenerate = %+v, want one generated in-vocab token without per-prompt error", batch)
	}
	batchMetrics := textModel.Metrics()
	if batchMetrics.GeneratedTokens != 1 || batchMetrics.PromptTokens != len(promptTokens) {
		t.Fatalf("Gemma4 q4 public BatchGenerate metrics = %+v, want one generated token and %d prompt tokens", batchMetrics, len(promptTokens))
	}
	badBatch, err := resultValue[[]inference.BatchResult](textModel.BatchGenerate(context.Background(), []string{"text:"}, inference.WithMaxTokens(1)))
	if err != nil {
		t.Fatalf("Gemma4 q4 public BatchGenerate invalid text prompt top-level error = %v, want per-prompt error", err)
	}
	if len(badBatch) != 1 || badBatch[0].Err == nil || !strings.Contains(badBatch[0].Err.Error(), "text prompt must contain prompt text") {
		t.Fatalf("Gemma4 q4 public BatchGenerate invalid text prompt = %+v, want per-prompt text prompt error", badBatch)
	}
	if resultError(textModel.Err()) == nil || !strings.Contains(resultError(textModel.Err()).Error(), "text prompt must contain prompt text") {
		t.Fatalf("Gemma4 q4 public BatchGenerate Err() = %v, want per-prompt text prompt error", resultError(textModel.Err()))
	}
	chatMessages := []inference.Message{{Role: "user", Content: "Hi"}}
	chatPrompt, err := loaded.ApplyChatTemplate(chatMessages)
	if err != nil {
		t.Fatalf("Gemma4 q4 chat template: %v", err)
	}
	chatPromptTokens, chatPromptOK, err := hipGemma4Q4TextPromptIDs("text:"+chatPrompt, loaded)
	if err != nil || !chatPromptOK {
		t.Fatalf("Gemma4 q4 chat prompt tokenization failed: tokens=%v ok=%v err=%v", chatPromptTokens, chatPromptOK, err)
	}
	var chatTokens []inference.Token
	for token := range textModel.Chat(context.Background(), chatMessages, inference.WithMaxTokens(1)) {
		chatTokens = append(chatTokens, token)
	}
	if err := resultError(textModel.Err()); err != nil {
		t.Fatalf("Gemma4 q4 public Chat: %v", err)
	}
	if len(chatTokens) != 1 ||
		chatTokens[0].ID < 0 ||
		int(chatTokens[0].ID) >= loaded.modelInfo.VocabSize {
		t.Fatalf("Gemma4 q4 public Chat tokens = %+v, want one generated in-vocab token", chatTokens)
	}
	chatMetrics := textModel.Metrics()
	if chatMetrics.GeneratedTokens != 1 || chatMetrics.PromptTokens != len(chatPromptTokens) {
		t.Fatalf("Gemma4 q4 public Chat metrics = %+v, want one generated token and %d prompt tokens", chatMetrics, len(chatPromptTokens))
	}
	var classifyEvents []inference.ProbeEvent
	probeable, probeableOK := any(textModel).(inference.ProbeableModel)
	if probeableOK {
		probeable.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
			classifyEvents = append(classifyEvents, event)
		}))
	}
	classify, err := resultValue[[]inference.ClassifyResult](textModel.Classify(context.Background(), []string{"Hi"}, inference.WithLogits()))
	if err != nil {
		t.Fatalf("Gemma4 q4 public Classify: %v", err)
	}
	if len(classify) != 1 ||
		classify[0].Token.ID < 0 ||
		int(classify[0].Token.ID) >= loaded.modelInfo.VocabSize ||
		len(classify[0].Logits) != loaded.modelInfo.VocabSize {
		t.Fatalf("Gemma4 q4 public Classify = %+v, want one in-vocab token and vocab-sized logits", classify)
	}
	if probeableOK {
		logitEvent, ok := nativeContractProbeEvent(classifyEvents, inference.ProbeEventLogits)
		if !ok || logitEvent.Logits == nil || len(logitEvent.Logits.Top) == 0 || logitEvent.Labels["source"] != "classification" {
			t.Fatalf("Gemma4 q4 public Classify probe events = %+v, want classification logit probe", classifyEvents)
		}
		entropyEvent, ok := nativeContractProbeEvent(classifyEvents, inference.ProbeEventEntropy)
		if !ok || entropyEvent.Entropy == nil || entropyEvent.Labels["classify_prompt_index"] != "0" {
			t.Fatalf("Gemma4 q4 public Classify probe events = %+v, want classification entropy probe", classifyEvents)
		}
		probeable.SetProbeSink(nil)
	}
	speculative, err := SpeculativeDecode(context.Background(), textModel, textModel, SpeculativeDecodeConfig{
		Prompt:      prompt,
		MaxTokens:   1,
		DraftTokens: 1,
	})
	if err != nil {
		t.Fatalf("Gemma4 q4 public SpeculativeDecode: %v", err)
	}
	if speculative.Mode != "speculative" ||
		speculative.Metrics.TargetCalls != 1 ||
		speculative.Metrics.DraftCalls != 1 ||
		speculative.Metrics.AcceptedTokens != 1 ||
		len(speculative.Tokens) != 1 ||
		speculative.Tokens[0].ID < 0 ||
		int(speculative.Tokens[0].ID) >= loaded.modelInfo.VocabSize {
		t.Fatalf("Gemma4 q4 public SpeculativeDecode = %+v, want one accepted in-vocab token", speculative)
	}
	promptLookup, err := PromptLookupDecode(context.Background(), textModel, PromptLookupDecodeConfig{
		Prompt:       prompt,
		MaxTokens:    1,
		LookupTokens: []int32{generated[0].ID},
	})
	if err != nil {
		t.Fatalf("Gemma4 q4 public PromptLookupDecode: %v", err)
	}
	if promptLookup.Mode != "prompt_lookup" ||
		promptLookup.Metrics.TargetCalls != 1 ||
		promptLookup.Metrics.LookupTokens != 1 ||
		promptLookup.Metrics.AcceptedTokens != 1 ||
		len(promptLookup.Tokens) != 1 ||
		promptLookup.Tokens[0].ID != generated[0].ID {
		t.Fatalf("Gemma4 q4 public PromptLookupDecode = %+v, want one accepted lookup token %d", promptLookup, generated[0].ID)
	}
	if benchable, ok := any(textModel).(inference.BenchableModel); ok {
		bench, err := benchable.Benchmark(context.Background(), inference.BenchConfig{
			Prompts:      []string{"Hi"},
			MaxTokens:    1,
			MeasuredRuns: 1,
		})
		if err != nil {
			t.Fatalf("Gemma4 q4 benchmark over explicit text prompt: %v", err)
		}
		if bench.GeneratedTokens != 1 ||
			bench.PromptTokens == 0 ||
			bench.Labels["attached.drafter.decode"] != "experimental" ||
			bench.Labels["attached.drafter.native_attachment"] != hipKernelStatusNotLinked ||
			bench.Labels["attached.drafter.role"] != "gemma4_assistant" ||
			bench.Labels["kernel_scope"] != "loaded_gemma4_q4_experimental_benchmark" ||
			bench.Labels["benchmark_prompt_mode"] != "explicit_text" ||
			bench.Labels["benchmark_retained_state_book"] != "BenchmarkInferenceGemma4Q4Book10Turn_RetainedState" ||
			bench.Labels["benchmark_retained_state_required"] != "true" ||
			bench.Labels["benchmark_prompt_replay_fallback"] != "forbidden" ||
			bench.Labels["benchmark_state_source"] != "rocm_state_session_runtime_kv" ||
			bench.Labels["production_book_policy"] != "retained_state_required" ||
			bench.Labels["production_book_decision_source"] != "benchmark_metrics" ||
			bench.Labels["production_book_gate_wall_seconds"] != strconv.Itoa(ProductionLaneBookWallSeconds) ||
			bench.Labels["production_book_gate_turns"] != strconv.Itoa(ProductionLaneBookTurnCount) ||
			bench.Labels["production_book_gate_raw_decode_tokens_per_sec"] != strconv.Itoa(DefaultProductionQuantizationPolicy().MinimumVisibleTokensPerSec) ||
			bench.Labels["production_book_gate_metrics"] == "" ||
			bench.Labels["production_book_gate_reason_codes"] != productionBookGateReasonCodesLabel ||
			bench.Labels["production_book_retained_route_metrics"] == "" ||
			bench.Labels["production_book_retained_artifact_labels"] == "" ||
			bench.Labels["production_model_source"] != "model_identity_or_pack" ||
			bench.Labels["production_mtp_required_metrics"] == "" ||
			bench.Labels["production_quant_decision_source"] != "gemma4_family_matrix" ||
			bench.Labels["speculative.decode"] != "experimental" ||
			bench.Labels["speculative.decode.affine_source"] != "gemma4_mlx_affine_generate" ||
			bench.Labels["speculative.decode.source"] != "gemma4_q4_generate" ||
			bench.Labels["prompt.lookup.decode"] != "experimental" ||
			bench.Labels["prompt.lookup.decode.affine_source"] != "gemma4_mlx_affine_generate" ||
			bench.Labels["prompt.lookup.decode.source"] != "gemma4_q4_generate" ||
			bench.Labels["decode_kernel"] != hipKernelStatusNotLinked ||
			bench.Labels["prefill_kernel"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 benchmark = %+v labels=%+v, want MLX affine benchmark/helper labels plus not-linked production prefill/decode labels", bench, bench.Labels)
		}
		for _, metric := range DefaultProductionQuantizationPolicy().RequiredBenchmarkMetrics {
			if !strings.Contains(bench.Labels["production_book_required_metrics"], metric) {
				t.Fatalf("Gemma4 q4 benchmark required metrics = %q, missing %q", bench.Labels["production_book_required_metrics"], metric)
			}
		}
		assertCSVLabelContainsAll(t, "production_book_gate_metrics", bench.Labels["production_book_gate_metrics"], productionBookGateMetrics)
		assertCSVLabelContainsAll(t, "production_book_retained_route_metrics", bench.Labels["production_book_retained_route_metrics"], productionBookRetainedRouteMetrics)
		assertCSVLabelContainsAll(t, "production_book_retained_artifact_labels", bench.Labels["production_book_retained_artifact_labels"], productionBookRetainedArtifactLabels)
		assertCSVLabelContainsAll(t, "production_mtp_required_metrics", bench.Labels["production_mtp_required_metrics"], defaultProductionMTPRequiredMetrics)
	}
	if evaluator, ok := any(textModel).(inference.Evaluator); ok {
		eval, err := evaluator.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
			Text:   "Hi",
			Prompt: "Hi",
			Labels: map[string]string{"target_token_id": "0"},
		}}, inference.EvalConfig{
			MaxSamples: 1,
			MaxSeqLen:  2,
			Probes:     []inference.QualityProbe{{Name: "q4-eval", Prompt: "Hi"}},
		})
		if err != nil {
			t.Fatalf("Gemma4 q4 eval quality probe over explicit text prompt: %v", err)
		}
		if len(eval.Probes) != 1 ||
			!eval.Probes[0].Passed ||
			eval.Metrics.Tokens != len(loaded.Encode("Hi")) ||
			eval.Labels["eval.tokens"] != core.Sprintf("%d", len(loaded.Encode("Hi"))) ||
			eval.Labels["quality_probe_status"] != "passed" ||
			eval.Labels["eval.loss_logits_source"] != "gemma4_mlx_affine_package_prefill" ||
			eval.Labels["loss_backend"] != "hip" ||
			eval.Labels["loss_status"] != "experimental" ||
			eval.Labels["decode_kernel"] != hipKernelStatusNotLinked ||
			eval.Labels["prefill_kernel"] != hipKernelStatusNotLinked {
			t.Fatalf("Gemma4 q4 eval = %+v metrics=%+v labels=%+v, want q4 prefill loss, passed quality probe, and not-linked production prefill/decode labels", eval.Probes, eval.Metrics, eval.Labels)
		}
	}
	t.Logf("Gemma4 q4 public Generate prompt=%q prompt_tokens=%v generated tokens=%v text=%q", prompt, promptTokens, ids, texts)
}

func gemma4Q4ForwardLayerCountFromEnv(t *testing.T, max int) (int, bool) {
	t.Helper()
	raw := os.Getenv("GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS")
	if raw == "" {
		return 0, false
	}
	if max <= 0 {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=%q requires loaded Gemma4 q4 layer configs", raw)
	}
	layerCount, err := strconv.Atoi(raw)
	if err != nil || layerCount <= 0 {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=%q, want positive integer", raw)
	}
	if layerCount > max {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=%d exceeds loaded layer count %d", layerCount, max)
	}
	return layerCount, true
}

func gemma4Q4DecodeEnv(t *testing.T, maxLayers int) (int, int, bool) {
	t.Helper()
	rawLayers := os.Getenv("GO_ROCM_GEMMA4_Q4_DECODE_LAYERS")
	rawTokens := os.Getenv("GO_ROCM_GEMMA4_Q4_DECODE_TOKENS")
	if rawLayers == "" && rawTokens == "" {
		return 0, 0, false
	}
	if rawLayers == "" || rawTokens == "" {
		t.Fatalf("set both GO_ROCM_GEMMA4_Q4_DECODE_LAYERS and GO_ROCM_GEMMA4_Q4_DECODE_TOKENS for q4 decode smoke")
	}
	layerCount, err := strconv.Atoi(rawLayers)
	if err != nil || layerCount <= 0 {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=%q, want positive integer", rawLayers)
	}
	if layerCount > maxLayers {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=%d exceeds loaded layer count %d", layerCount, maxLayers)
	}
	tokenCount, err := strconv.Atoi(rawTokens)
	if err != nil || tokenCount <= 1 {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=%q, want integer greater than 1 to exercise cached decode", rawTokens)
	}
	return layerCount, tokenCount, true
}

func gemma4Q4DecodePromptTokensEnv(t *testing.T, vocabSize int) []int32 {
	t.Helper()
	raw := os.Getenv("GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS")
	if raw == "" {
		return []int32{0}
	}
	parts := strings.Split(raw, ",")
	tokens := make([]int32, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			t.Fatalf("GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS=%q contains an empty token", raw)
		}
		value, err := strconv.Atoi(part)
		if err != nil || value < 0 || value >= vocabSize {
			t.Fatalf("GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS=%q has invalid token %q for vocab size %d", raw, part, vocabSize)
		}
		tokens = append(tokens, int32(value))
	}
	if len(tokens) == 0 {
		t.Fatalf("GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS=%q, want at least one token", raw)
	}
	return tokens
}

func hipGemma4Q4GreedyTokenIDs(tokens []hipGreedySampleResult) []int {
	ids := make([]int, len(tokens))
	for index, token := range tokens {
		ids[index] = token.TokenID
	}
	return ids
}

func assertLoadedGemma4MLXQ4AttentionProjectionSmoke(t *testing.T, model *hipLoadedModel, layerInput []float32) gemma4BF16AttentionProjectionOutputs {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return gemma4BF16AttentionProjectionOutputs{}
	}
	hidden := model.modelInfo.HiddenSize
	if len(layerInput) != hidden {
		t.Fatalf("Gemma4 q4 attention input length = %d, want %d", len(layerInput), hidden)
	}
	qOutput := assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "q_proj",
		tensorBase: "language_model.model.layers.0.self_attn.q_proj",
		rows:       2048,
		cols:       hidden,
	}, layerInput)
	kOutput := assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "k_proj",
		tensorBase: "language_model.model.layers.0.self_attn.k_proj",
		rows:       256,
		cols:       hidden,
	}, layerInput)
	vOutput := assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "v_proj",
		tensorBase: "language_model.model.layers.0.self_attn.v_proj",
		rows:       256,
		cols:       hidden,
	}, layerInput)
	return gemma4BF16AttentionProjectionOutputs{Query: qOutput, Key: kOutput, Value: vOutput}
}

func assertLoadedGemma4MLXQ4ProjectionSmoke(t *testing.T, model *hipLoadedModel) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	input := make([]float32, hidden)
	for index := range input {
		input[index] = float32((index%11)-5) * 0.0625
	}
	return assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "q_proj",
		tensorBase: "language_model.model.layers.0.self_attn.q_proj",
		rows:       2048,
		cols:       hidden,
	}, input)
}

type gemma4MLXQ4ProjectionSpec struct {
	label      string
	tensorBase string
	rows       int
	cols       int
}

func assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t *testing.T, model *hipLoadedModel, spec gemma4MLXQ4ProjectionSpec, input []float32) []float32 {
	t.Helper()
	if len(input) != spec.cols {
		t.Fatalf("%s q4 input length = %d, want cols %d", spec.label, len(input), spec.cols)
	}
	bits := hipMLXQ4ProjectionBitsOrDefault(model.modelInfo.QuantBits)
	weight, ok := model.tensors[spec.tensorBase+".weight"]
	if !ok {
		t.Fatalf("loaded Gemma4 q%d model is missing %s packed weight tensor", bits, spec.label)
	}
	scales, ok := model.tensors[spec.tensorBase+".scales"]
	if !ok {
		t.Fatalf("loaded Gemma4 q%d model is missing %s scales tensor", bits, spec.label)
	}
	biases, ok := model.tensors[spec.tensorBase+".biases"]
	if !ok {
		t.Fatalf("loaded Gemma4 q%d model is missing %s biases tensor", bits, spec.label)
	}
	groupSize := model.modelInfo.QuantGroup
	if groupSize == 0 {
		groupSize = 64
	}
	packedPerRow, err := hipMLXAffinePackedCols(spec.cols, bits)
	core.RequireNoError(t, err)
	groups := spec.cols / groupSize
	if weight.info.TypeName != "U32" ||
		len(weight.info.Dimensions) != 2 ||
		weight.info.Dimensions[0] != uint64(spec.rows) ||
		weight.info.Dimensions[1] != uint64(packedPerRow) ||
		weight.info.ByteSize != uint64(spec.rows*packedPerRow*4) {
		t.Fatalf("q%d %s weight tensor = %+v, want Gemma4 q%d [%d,%d]", bits, spec.label, weight.info, bits, spec.rows, packedPerRow)
	}
	for label, tensor := range map[string]hipTensor{"scales": scales, "biases": biases} {
		if tensor.info.TypeName != "BF16" ||
			len(tensor.info.Dimensions) != 2 ||
			tensor.info.Dimensions[0] != uint64(spec.rows) ||
			tensor.info.Dimensions[1] != uint64(groups) ||
			tensor.info.ByteSize != uint64(spec.rows*groups*2) {
			t.Fatalf("q%d %s %s tensor = %+v, want Gemma4 q%d [%d,%d]", bits, spec.label, label, tensor.info, bits, spec.rows, groups)
		}
	}
	got, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(context.Background(), model.driver, input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: weight.pointer,
		ScalePointer:  scales.pointer,
		BiasPointer:   biases.pointer,
		WeightBytes:   weight.info.ByteSize,
		ScaleBytes:    scales.info.ByteSize,
		BiasBytes:     biases.info.ByteSize,
		Rows:          spec.rows,
		Cols:          spec.cols,
		GroupSize:     groupSize,
		Bits:          bits,
	})
	core.RequireNoError(t, err)
	compareRows := 8
	if spec.rows < compareRows {
		compareRows = spec.rows
	}
	wantWeights := readLoadedUint32TensorRows(t, weight, compareRows, packedPerRow)
	wantScales := readLoadedBF16TensorRows(t, scales, compareRows, groups)
	wantBiases := readLoadedBF16TensorRows(t, biases, compareRows, groups)
	want, err := hipReferenceMLXAffineProjection(input, wantWeights, wantScales, wantBiases, compareRows, spec.cols, groupSize, bits)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got[:compareRows], 0.05)
	return got
}

func assertLoadedGemma4RoPESmoke(t *testing.T, model *hipLoadedModel, projections gemma4BF16AttentionProjectionOutputs) gemma4RoPEOutputs {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return gemma4RoPEOutputs{}
	}
	const headDim = 256
	const queryHeads = 8
	if len(projections.Query) != queryHeads*headDim || len(projections.Key) != headDim {
		t.Fatalf("Gemma4 projection outputs q=%d k=%d, want %d query heads and one %d-dim key head", len(projections.Query), len(projections.Key), queryHeads, headDim)
	}
	heads := make([][]float32, 0, queryHeads)
	for head := 0; head < queryHeads; head++ {
		start := head * headDim
		end := start + headDim
		heads = append(heads, assertLoadedGemma4RoPEVectorSmoke(t, model.driver, core.Sprintf("q_head%d", head), projections.Query[start:end]))
	}
	key := assertLoadedGemma4RoPEVectorSmoke(t, model.driver, "k_head0", projections.Key)
	return gemma4RoPEOutputs{QueryHeads: heads, Key: key}
}

func assertLoadedGemma4RoPEVectorSmoke(t *testing.T, driver nativeHIPDriver, label string, input []float32) []float32 {
	t.Helper()
	req := hipRoPERequest{Input: append([]float32(nil), input...), Position: 1, Base: 10000}
	output, err := hipRunRoPEKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)
	expected, err := hipReferenceRoPE(req.Input, req.Position, float64(req.Base))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, expected, output, 0.005)
	if len(output) != len(input) {
		t.Fatalf("%s RoPE output length = %d, want %d", label, len(output), len(input))
	}
	return output
}

func assertLoadedGemma4AttentionSmoke(t *testing.T, model *hipLoadedModel, rope gemma4RoPEOutputs, value []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	const headDim = 256
	const queryHeads = 8
	if len(rope.QueryHeads) != queryHeads || len(rope.Key) != headDim || len(value) != headDim {
		t.Fatalf("Gemma4 attention inputs query_heads=%d k=%d v=%d, want %d heads and %d-dim k/v", len(rope.QueryHeads), len(rope.Key), len(value), queryHeads, headDim)
	}
	concat := make([]float32, 0, queryHeads*headDim)
	for head, query := range rope.QueryHeads {
		if len(query) != headDim {
			t.Fatalf("Gemma4 attention query head %d length = %d, want %d", head, len(query), headDim)
		}
		req := hipAttentionRequest{
			Query:  append([]float32(nil), query...),
			Keys:   append([]float32(nil), rope.Key...),
			Values: append([]float32(nil), value...),
		}
		output, err := hipRunAttentionKernel(context.Background(), model.driver, req)
		core.RequireNoError(t, err)
		expectedOutput, expectedWeights, err := hipReferenceSingleHeadAttention(req.Query, [][]float32{req.Keys}, [][]float32{req.Values})
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, expectedOutput, output.Output, 0.005)
		assertFloat32SlicesNear(t, expectedWeights, output.Weights, 0.0001)
		concat = append(concat, output.Output...)
	}
	return concat
}

func assertLoadedGemma4OutputProjectionSmoke(t *testing.T, model *hipLoadedModel, attentionOutput []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		model.modelInfo.QuantBits != 0 {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if len(attentionOutput) != 2048 {
		t.Fatalf("Gemma4 attention concat length = %d, want 2048", len(attentionOutput))
	}
	return assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "o_proj",
		tensorName: "language_model.model.layers.0.self_attn.o_proj.weight",
		rows:       hidden,
		cols:       2048,
	}, attentionOutput)
}

func assertLoadedGemma4MLXQ4OutputProjectionSmoke(t *testing.T, model *hipLoadedModel, attentionOutput []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if len(attentionOutput) != 2048 {
		t.Fatalf("Gemma4 q4 attention concat length = %d, want 2048", len(attentionOutput))
	}
	return assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "o_proj",
		tensorBase: "language_model.model.layers.0.self_attn.o_proj",
		rows:       hidden,
		cols:       2048,
	}, attentionOutput)
}

func assertLoadedGemma4VectorAddSmoke(t *testing.T, model *hipLoadedModel, label string, left, right []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if len(left) != hidden || len(right) != hidden {
		t.Fatalf("Gemma4 %s vector add lengths left=%d right=%d, want %d", label, len(left), len(right), hidden)
	}
	req := hipVectorAddRequest{Left: append([]float32(nil), left...), Right: append([]float32(nil), right...)}
	output, err := hipRunVectorAddKernel(context.Background(), model.driver, req)
	core.RequireNoError(t, err)
	expected := make([]float32, hidden)
	for index := range expected {
		expected[index] = left[index] + right[index]
	}
	assertFloat32SlicesNear(t, expected, output, 0.0001)
	return output
}

func assertLoadedGemma4EmbeddingScaleSmoke(t *testing.T, model *hipLoadedModel, label string, embedding []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if hidden <= 0 || len(embedding) != hidden {
		t.Fatalf("Gemma4 %s input length = %d, want hidden size %d", label, len(embedding), hidden)
	}
	scale := float32(math.Sqrt(float64(hidden)))
	req := hipVectorScaleRequest{Input: append([]float32(nil), embedding...), Scale: scale}
	output, err := hipRunVectorScaleKernel(context.Background(), model.driver, req)
	core.RequireNoError(t, err)
	expected := make([]float32, len(embedding))
	for index := range expected {
		expected[index] = embedding[index] * scale
	}
	assertFloat32SlicesNear(t, expected, output, 0.0001)
	return output
}

func assertLoadedGemma4MLPSmoke(t *testing.T, model *hipLoadedModel, input []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		model.modelInfo.QuantBits != 0 {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if len(input) != hidden {
		t.Fatalf("Gemma4 MLP input length = %d, want %d", len(input), hidden)
	}
	const intermediate = 6144
	gate := assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "mlp.gate_proj",
		tensorName: "language_model.model.layers.0.mlp.gate_proj.weight",
		rows:       intermediate,
		cols:       hidden,
	}, input)
	up := assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "mlp.up_proj",
		tensorName: "language_model.model.layers.0.mlp.up_proj.weight",
		rows:       intermediate,
		cols:       hidden,
	}, input)
	activated := assertLoadedGemma4SwiGLUSmoke(t, model, gate, up)
	return assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "mlp.down_proj",
		tensorName: "language_model.model.layers.0.mlp.down_proj.weight",
		rows:       hidden,
		cols:       intermediate,
	}, activated)
}

func assertLoadedGemma4MLXQ4MLPSmoke(t *testing.T, model *hipLoadedModel, input []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return nil
	}
	hidden := model.modelInfo.HiddenSize
	if len(input) != hidden {
		t.Fatalf("Gemma4 q4 MLP input length = %d, want %d", len(input), hidden)
	}
	const intermediate = 6144
	gate := assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "mlp.gate_proj",
		tensorBase: "language_model.model.layers.0.mlp.gate_proj",
		rows:       intermediate,
		cols:       hidden,
	}, input)
	up := assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "mlp.up_proj",
		tensorBase: "language_model.model.layers.0.mlp.up_proj",
		rows:       intermediate,
		cols:       hidden,
	}, input)
	activated := assertLoadedGemma4SwiGLUSmoke(t, model, gate, up)
	return assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "mlp.down_proj",
		tensorBase: "language_model.model.layers.0.mlp.down_proj",
		rows:       hidden,
		cols:       intermediate,
	}, activated)
}

func assertLoadedGemma4MLXQ4LogitSmoke(t *testing.T, model *hipLoadedModel, input []float32) hipGreedySampleResult {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		!hipMLXAffineSupportedBits(model.modelInfo.QuantBits) {
		return hipGreedySampleResult{}
	}
	hidden := model.modelInfo.HiddenSize
	vocab := model.modelInfo.VocabSize
	if len(input) != hidden {
		t.Fatalf("Gemma4 q4 logit input length = %d, want %d", len(input), hidden)
	}
	finalNorm := assertLoadedGemma4BF16RMSNormTensorSmoke(t, model, "language_model.model.norm.weight", "q4 final_norm", input)
	logits := assertLoadedGemma4MLXQ4ProjectionTensorSmoke(t, model, gemma4MLXQ4ProjectionSpec{
		label:      "embed_tokens_lm_head",
		tensorBase: "language_model.model.embed_tokens",
		rows:       vocab,
		cols:       hidden,
	}, finalNorm)
	greedyOutput, err := hipRunGreedyKernel(context.Background(), model.driver, hipGreedySampleRequest{Logits: logits})
	core.RequireNoError(t, err)
	wantToken, wantScore, err := hipReferenceGreedySample(logits)
	core.RequireNoError(t, err)
	if greedyOutput.TokenID != wantToken || math.Abs(float64(greedyOutput.Score-wantScore)) > 0.0001 {
		t.Fatalf("Gemma4 q4 greedy output = %+v, want token %d score %f", greedyOutput, wantToken, wantScore)
	}
	return greedyOutput
}

func assertLoadedGemma4BF16LogitSmoke(t *testing.T, model *hipLoadedModel, input []float32) hipGreedySampleResult {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		model.modelInfo.QuantBits != 0 {
		return hipGreedySampleResult{}
	}
	hidden := model.modelInfo.HiddenSize
	vocab := model.modelInfo.VocabSize
	if len(input) != hidden {
		t.Fatalf("Gemma4 BF16 logit input length = %d, want %d", len(input), hidden)
	}
	finalNorm := assertLoadedGemma4BF16RMSNormTensorSmoke(t, model, "language_model.model.norm.weight", "bf16 final_norm", input)
	logits := assertLoadedGemma4BF16ProjectionTensorSmoke(t, model, gemma4BF16ProjectionSpec{
		label:      "embed_tokens_lm_head",
		tensorName: "language_model.model.embed_tokens.weight",
		rows:       vocab,
		cols:       hidden,
	}, finalNorm)
	logits, err := hipGemma4Q4SoftcapLogits(logits, hipGemma4Q4FinalLogitSoftcap())
	core.RequireNoError(t, err)
	greedyOutput, err := hipRunGreedyKernel(context.Background(), model.driver, hipGreedySampleRequest{Logits: logits})
	core.RequireNoError(t, err)
	wantToken, wantScore, err := hipReferenceGreedySample(logits)
	core.RequireNoError(t, err)
	if greedyOutput.TokenID != wantToken || math.Abs(float64(greedyOutput.Score-wantScore)) > 0.0001 {
		t.Fatalf("Gemma4 BF16 greedy output = %+v, want token %d score %f", greedyOutput, wantToken, wantScore)
	}
	t.Logf("Gemma4 BF16 layer0 tied LM-head greedy token=%d score=%f", greedyOutput.TokenID, greedyOutput.Score)
	return greedyOutput
}

func assertLoadedGemma4SwiGLUSmoke(t *testing.T, model *hipLoadedModel, gate, up []float32) []float32 {
	t.Helper()
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	if len(gate) != 6144 || len(up) != len(gate) {
		t.Fatalf("Gemma4 SwiGLU inputs gate=%d up=%d, want 6144", len(gate), len(up))
	}
	req := hipSwiGLURequest{Gate: append([]float32(nil), gate...), Up: append([]float32(nil), up...)}
	output, err := hipRunSwiGLUKernel(context.Background(), model.driver, req)
	core.RequireNoError(t, err)
	expected := make([]float32, len(gate))
	for index := range expected {
		expected[index] = gate[index] / (1 + float32(math.Exp(float64(-gate[index])))) * up[index]
	}
	assertFloat32SlicesNear(t, expected, output, 0.001)
	return output
}

func readLoadedBF16TensorRows(t *testing.T, tensor hipTensor, rows, cols int) []uint16 {
	t.Helper()
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		t.Fatalf("loaded tensor %s has no source path", tensor.info.Name)
	}
	file, err := os.Open(sourcePath)
	core.RequireNoError(t, err)
	defer file.Close()

	payload := make([]byte, rows*cols*2)
	start := tensor.info.DataOffset + int64(tensor.info.Offset)
	n, err := file.ReadAt(payload, start)
	if err != nil || n != len(payload) {
		t.Fatalf("read tensor rows from %s at %d: n=%d err=%v", sourcePath, start, n, err)
	}
	values := make([]uint16, rows*cols)
	for index := range values {
		values[index] = binary.LittleEndian.Uint16(payload[index*2:])
	}
	return values
}

func readLoadedUint32TensorRows(t *testing.T, tensor hipTensor, rows, cols int) []uint32 {
	t.Helper()
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		t.Fatalf("loaded tensor %s has no source path", tensor.info.Name)
	}
	file, err := os.Open(sourcePath)
	core.RequireNoError(t, err)
	defer file.Close()

	payload := make([]byte, rows*cols*4)
	start := tensor.info.DataOffset + int64(tensor.info.Offset)
	n, err := file.ReadAt(payload, start)
	if err != nil || n != len(payload) {
		t.Fatalf("read tensor rows from %s at %d: n=%d err=%v", sourcePath, start, n, err)
	}
	values := make([]uint32, rows*cols)
	for index := range values {
		values[index] = binary.LittleEndian.Uint32(payload[index*4:])
	}
	return values
}

func readLoadedUint32EmbeddingRows(t *testing.T, tensor hipTensor, tokenIDs []int32, cols int) []uint32 {
	t.Helper()
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		t.Fatalf("loaded tensor %s has no source path", tensor.info.Name)
	}
	file, err := os.Open(sourcePath)
	core.RequireNoError(t, err)
	defer file.Close()

	rowBytes := cols * 4
	values := make([]uint32, 0, len(tokenIDs)*cols)
	payload := make([]byte, rowBytes)
	for _, id := range tokenIDs {
		if id < 0 {
			t.Fatalf("token ID %d is negative", id)
		}
		start := tensor.info.DataOffset + int64(tensor.info.Offset) + int64(id)*int64(rowBytes)
		n, err := file.ReadAt(payload, start)
		if err != nil || n != len(payload) {
			t.Fatalf("read q4 embedding row %d from %s at %d: n=%d err=%v", id, sourcePath, start, n, err)
		}
		for index := 0; index < cols; index++ {
			values = append(values, binary.LittleEndian.Uint32(payload[index*4:]))
		}
	}
	return values
}

func readLoadedBF16TensorRowsByID(t *testing.T, tensor hipTensor, tokenIDs []int32, cols int) []uint16 {
	t.Helper()
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		t.Fatalf("loaded tensor %s has no source path", tensor.info.Name)
	}
	file, err := os.Open(sourcePath)
	core.RequireNoError(t, err)
	defer file.Close()

	rowBytes := cols * 2
	values := make([]uint16, 0, len(tokenIDs)*cols)
	payload := make([]byte, rowBytes)
	for _, id := range tokenIDs {
		if id < 0 {
			t.Fatalf("token ID %d is negative", id)
		}
		start := tensor.info.DataOffset + int64(tensor.info.Offset) + int64(id)*int64(rowBytes)
		n, err := file.ReadAt(payload, start)
		if err != nil || n != len(payload) {
			t.Fatalf("read bf16 tensor row %d from %s at %d: n=%d err=%v", id, sourcePath, start, n, err)
		}
		for index := 0; index < cols; index++ {
			values = append(values, binary.LittleEndian.Uint16(payload[index*2:]))
		}
	}
	return values
}

func readLoadedBF16EmbeddingRows(t *testing.T, tensor hipTensor, tokenIDs []int32, hidden int) []float32 {
	t.Helper()
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		t.Fatalf("loaded tensor %s has no source path", tensor.info.Name)
	}
	file, err := os.Open(sourcePath)
	core.RequireNoError(t, err)
	defer file.Close()

	rowBytes := hidden * 2
	values := make([]float32, 0, len(tokenIDs)*hidden)
	payload := make([]byte, rowBytes)
	for _, id := range tokenIDs {
		if id < 0 {
			t.Fatalf("token ID %d is negative", id)
		}
		start := tensor.info.DataOffset + int64(tensor.info.Offset) + int64(id)*int64(rowBytes)
		n, err := file.ReadAt(payload, start)
		if err != nil || n != len(payload) {
			t.Fatalf("read embedding row %d from %s at %d: n=%d err=%v", id, sourcePath, start, n, err)
		}
		for index := 0; index < hidden; index++ {
			values = append(values, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:])))
		}
	}
	return values
}

func TestHIPHardwareKVCacheSmoke_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_CACHE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_CACHE_TESTS=1 to run ROCm cache hardware tests")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: hipRuntime.driver})
	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_cache_block_size": "2",
			"kv_key_width":        "2",
			"kv_value_width":      "2",
		},
	})
	core.RequireNoError(t, err)
	if warmed.Labels["kv_device_backing"] != "mirrored" || warmed.Labels["kv_device_pages"] != "2" || warmed.Stats.Labels["kv_device_tokens"] != "3" {
		t.Fatalf("cache warm labels=%+v stats=%+v, want block-cache HIP device remirror", warmed.Labels, warmed.Stats.Labels)
	}
	if _, err := service.ClearCache(context.Background(), nil); err != nil {
		t.Fatalf("clear remirrored cache: %v", err)
	}

	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))

	device, err := cache.MirrorToDevice(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer device.Close()

	stats := device.Stats()
	if stats.Blocks != 1 || stats.Labels["kv_device_backing"] != "mirrored" || stats.Labels["kv_key_width"] != "2" || stats.Labels["kv_value_width"] != "3" {
		t.Fatalf("device KV stats = %+v, want mirrored toy cache labels", stats)
	}
	descriptor, err := device.KernelDescriptor()
	core.RequireNoError(t, err)
	if len(descriptor.Pages) != 1 || descriptor.Pages[0].KeyPointer == 0 || descriptor.Pages[0].ValuePointer == 0 {
		t.Fatalf("device KV descriptor = %+v, want non-zero key/value pointers", descriptor)
	}
	descriptorBytes, err := device.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	if len(descriptorBytes) != rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVDescriptorPageBytes {
		t.Fatalf("descriptor byte length = %d, want one fixed-width page table", len(descriptorBytes))
	}
	if binary.LittleEndian.Uint32(descriptorBytes[0:]) != rocmDeviceKVDescriptorVersion ||
		binary.LittleEndian.Uint32(descriptorBytes[12:]) != rocmDeviceKVDescriptorModeKQ8VQ4 ||
		binary.LittleEndian.Uint64(descriptorBytes[24:]) != uint64(cache.TokenCount()) {
		t.Fatalf("descriptor header bytes = %+v, want v1 k-q8-v-q4 token table", descriptorBytes[:rocmDeviceKVDescriptorHeaderBytes])
	}
	table, err := device.KernelDescriptorTable()
	core.RequireNoError(t, err)
	if table.Pointer() == 0 || table.SizeBytes() != uint64(len(descriptorBytes)) {
		t.Fatalf("descriptor table pointer=%d size=%d, want device-resident descriptor bytes", table.Pointer(), table.SizeBytes())
	}
	core.RequireNoError(t, table.Close())

	tokenBuffer, err := hipUploadTokenIDs(hipRuntime.driver, []int32{1, 2})
	core.RequireNoError(t, err)
	defer tokenBuffer.Close()
	prefillLaunch, err := (hipPrefillRequest{
		TokenIDs:   []int32{1, 2},
		CacheMode:  rocmKVCacheModeKQ8VQ4,
		KeyWidth:   2,
		ValueWidth: 3,
	}).prefillLaunchArgs(tokenBuffer)
	core.RequireNoError(t, err)
	prefillLaunchBytes, err := prefillLaunch.Binary()
	core.RequireNoError(t, err)
	if len(prefillLaunchBytes) != hipPrefillLaunchArgsBytes || binary.LittleEndian.Uint64(prefillLaunchBytes[16:]) != 2 {
		t.Fatalf("prefill launch bytes length=%d token_count=%d, want fixed launch packet", len(prefillLaunchBytes), binary.LittleEndian.Uint64(prefillLaunchBytes[16:]))
	}

	projectionReq := hipProjectionRequest{
		Input: []float32{1, 2},
		FP16:  []uint16{0x3c00, 0x4000},
		Rows:  1,
		Cols:  2,
	}
	projectionBuffers, err := projectionReq.projectionDeviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer projectionBuffers.Close()
	projectionLaunch, err := projectionReq.projectionLaunchArgs(projectionBuffers)
	core.RequireNoError(t, err)
	projectionLaunchBytes, err := projectionLaunch.Binary()
	core.RequireNoError(t, err)
	if len(projectionLaunchBytes) != hipProjectionLaunchArgsBytes || binary.LittleEndian.Uint32(projectionLaunchBytes[80:]) != hipProjectionWeightEncodingFP16 {
		t.Fatalf("projection launch bytes length=%d encoding=%d, want fixed fp16 launch packet", len(projectionLaunchBytes), binary.LittleEndian.Uint32(projectionLaunchBytes[80:]))
	}
}

func TestHIPHardwareKVEncodeRowsKernel_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_CACHE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_CACHE_TESTS=1 to run ROCm cache hardware tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	keyRows := []float32{
		100, -100,
		0.5, -0.5,
	}
	valueRows := []float32{
		7, -7,
		0.25, -0.25,
	}
	keyInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.KVCache.HardwareTest", "row-scaled key rows", mustHIPFloat32Payload(t, keyRows), len(keyRows))
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.KVCache.HardwareTest", "row-scaled value rows", mustHIPFloat32Payload(t, valueRows), len(valueRows))
	core.RequireNoError(t, err)
	defer valueInput.Close()

	key, value, err := hipRunKVEncodeRowsKernel(context.Background(), hipRuntime.driver, keyInput, valueInput, 2, 2, 2, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer hipRuntime.driver.Free(key.pointer)
	defer hipRuntime.driver.Free(value.pointer)

	core.AssertEqual(t, rocmKVEncodingQ8Rows, key.encoding)
	core.AssertEqual(t, rocmKVEncodingQ4Rows, value.encoding)
	core.AssertEqual(t, uint64(12), key.sizeBytes)
	core.AssertEqual(t, uint64(10), value.sizeBytes)
	keyDecoded, err := copyROCmDeviceKVTensorRowsToHost(hipRuntime.driver, key, len(keyRows), 2)
	core.RequireNoError(t, err)
	valueDecoded, err := copyROCmDeviceKVTensorRowsToHost(hipRuntime.driver, value, len(valueRows), 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, keyRows, keyDecoded.decodeRows(2), 0.02)
	assertFloat32SlicesNear(t, valueRows, valueDecoded.decodeRows(2), 0.02)

	cache := &rocmDeviceKVCache{driver: hipRuntime.driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 2}
	deviceKV, err := cache.withAppendedDeviceRowsWindow(context.Background(), keyInput, valueInput, 2, 2, 2, 0)
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	table, err := deviceKV.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer table.Close()
	attentionOutput, err := hipRunAttentionKernel(context.Background(), hipRuntime.driver, hipAttentionRequest{
		Query:           []float32{1, 0},
		DeviceKV:        deviceKV,
		DescriptorTable: table,
	})
	core.RequireNoError(t, err)
	hostCache, err := deviceKV.hostCache()
	core.RequireNoError(t, err)
	restoredKeys, restoredValues, err := hostCache.Restore(0, deviceKV.TokenCount())
	core.RequireNoError(t, err)
	referenceKeys, err := splitHIPReferenceVectors(restoredKeys, 2)
	core.RequireNoError(t, err)
	referenceValues, err := splitHIPReferenceVectors(restoredValues, 2)
	core.RequireNoError(t, err)
	wantOutput, wantWeights, err := hipReferenceSingleHeadAttention([]float32{1, 0}, referenceKeys, referenceValues)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, wantOutput, attentionOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, wantWeights, attentionOutput.Weights, 0.0001)
}

func TestHIPHardwareProjectionKernelSource_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	req := hipProjectionRequest{
		Input: []float32{1, 2},
		FP16:  []uint16{0x3c00, 0x4000},
		Rows:  1,
		Cols:  2,
		Bias:  []float32{0.5},
	}
	buffers, err := req.projectionDeviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	launch, err := req.projectionLaunchArgs(buffers)
	core.RequireNoError(t, err)
	launchBytes, err := launch.Binary()
	core.RequireNoError(t, err)
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, launchBytes, req.Rows)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, config))
	output, err := buffers.ReadOutput()
	core.RequireNoError(t, err)
	if len(output) != 1 || math.Abs(float64(output[0]-5.5)) > 0.0001 {
		t.Fatalf("projection output = %+v, want [5.5]", output)
	}
	q8Req := hipProjectionRequest{
		Input:   []float32{3, -2},
		Q8:      []int8{2, -4, -1, 3},
		Q8Scale: 0.25,
		Rows:    2,
		Cols:    2,
		Bias:    []float32{0.5, -0.25},
	}
	q8Buffers, err := q8Req.projectionDeviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer q8Buffers.Close()
	q8Launch, err := q8Req.projectionLaunchArgs(q8Buffers)
	core.RequireNoError(t, err)
	q8LaunchBytes, err := q8Launch.Binary()
	core.RequireNoError(t, err)
	q8Config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, q8LaunchBytes, q8Req.Rows)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, q8Config))
	q8Output, err := q8Buffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{4, -2.5}, q8Output, 0.0001)

	bf16Req := hipProjectionRequest{
		Input: []float32{1.5, -2},
		BF16:  []uint16{0x3f80, 0xc000, 0x4000, 0x3f00},
		Rows:  2,
		Cols:  2,
		Bias:  []float32{0.25, -0.5},
	}
	bf16Buffers, err := bf16Req.projectionDeviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer bf16Buffers.Close()
	bf16Launch, err := bf16Req.projectionLaunchArgs(bf16Buffers)
	core.RequireNoError(t, err)
	bf16LaunchBytes, err := bf16Launch.Binary()
	core.RequireNoError(t, err)
	bf16Config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, bf16LaunchBytes, bf16Req.Rows)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, bf16Config))
	bf16Output, err := bf16Buffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.75, 1.5}, bf16Output, 0.0001)

	t.Run("mlx-q4-projection", func(t *testing.T) {
		q4Req := hipMLXQ4ProjectionRequest{
			Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
			Weight:    []uint32{0x76543210, 0xfedcba98},
			Scales:    []uint16{0x3f80, 0x3f00},
			Biases:    []uint16{0x0000, 0xbf80},
			Rows:      2,
			Cols:      8,
			GroupSize: 8,
		}
		q4Want, err := hipReferenceMLXQ4Projection(q4Req.Input, q4Req.Weight, q4Req.Scales, q4Req.Biases, q4Req.Rows, q4Req.Cols, q4Req.GroupSize)
		core.RequireNoError(t, err)
		q4Output, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, q4Req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, q4Want, q4Output, 0.0001)

		q4Buffers, err := q4Req.deviceBuffers(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer q4Buffers.Close()
		batchInput := append(append([]float32(nil), q4Req.Input...), []float32{2, 2, 2, 2, 2, 2, 2, 2}...)
		batchPayload, err := hipFloat32Payload(batchInput)
		core.RequireNoError(t, err)
		batchInputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch input", batchPayload, len(batchInput))
		core.RequireNoError(t, err)
		defer batchInputBuffer.Close()
		batchOutputBuffer, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, batchInputBuffer, hipMLXQ4DeviceWeightConfig{
			WeightPointer: q4Buffers.Weight.Pointer(),
			ScalePointer:  q4Buffers.Scales.Pointer(),
			BiasPointer:   q4Buffers.Biases.Pointer(),
			WeightBytes:   q4Buffers.Weight.SizeBytes(),
			ScaleBytes:    q4Buffers.Scales.SizeBytes(),
			BiasBytes:     q4Buffers.Biases.SizeBytes(),
			Rows:          q4Req.Rows,
			Cols:          q4Req.Cols,
			GroupSize:     q4Req.GroupSize,
		}, 2)
		core.RequireNoError(t, err)
		defer batchOutputBuffer.Close()
		batchOutput, err := hipReadFloat32DeviceOutput(batchOutputBuffer, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch output", q4Req.Rows*2)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, []float32{q4Want[0], q4Want[1], q4Want[0] * 2, q4Want[1] * 2}, batchOutput, 0.0001)

		batchActivated, err := hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, batchInputBuffer, hipMLXQ4DeviceWeightConfig{
			WeightPointer: q4Buffers.Weight.Pointer(),
			ScalePointer:  q4Buffers.Scales.Pointer(),
			BiasPointer:   q4Buffers.Biases.Pointer(),
			WeightBytes:   q4Buffers.Weight.SizeBytes(),
			ScaleBytes:    q4Buffers.Scales.SizeBytes(),
			BiasBytes:     q4Buffers.Biases.SizeBytes(),
			Rows:          q4Req.Rows,
			Cols:          q4Req.Cols,
			GroupSize:     q4Req.GroupSize,
		}, hipMLXQ4DeviceWeightConfig{
			WeightPointer: q4Buffers.Weight.Pointer(),
			ScalePointer:  q4Buffers.Scales.Pointer(),
			BiasPointer:   q4Buffers.Biases.Pointer(),
			WeightBytes:   q4Buffers.Weight.SizeBytes(),
			ScaleBytes:    q4Buffers.Scales.SizeBytes(),
			BiasBytes:     q4Buffers.Biases.SizeBytes(),
			Rows:          q4Req.Rows,
			Cols:          q4Req.Cols,
			GroupSize:     q4Req.GroupSize,
		}, 2)
		core.RequireNoError(t, err)
		defer batchActivated.Close()
		activatedOutput, err := hipReadFloat32DeviceOutput(batchActivated, "rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch output", q4Req.Rows*2)
		core.RequireNoError(t, err)
		secondReq := q4Req
		secondReq.Input = []float32{2, 2, 2, 2, 2, 2, 2, 2}
		wantActivated := append(
			expectedGELUTanhMultiplyFromQ4(t, q4Req, q4Req),
			expectedGELUTanhMultiplyFromQ4(t, secondReq, secondReq)...,
		)
		assertFloat32SlicesNear(t, wantActivated, activatedOutput, 0.0001)

		batchMultiplierPayload, err := hipFloat32Payload([]float32{2, 3, 4, 5})
		core.RequireNoError(t, err)
		batchMultiplier, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch multiplier", batchMultiplierPayload, q4Req.Rows*2)
		core.RequireNoError(t, err)
		defer batchMultiplier.Close()
		batchProjected, err := hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(context.Background(), hipRuntime.driver, batchInputBuffer, batchMultiplier, hipMLXQ4DeviceWeightConfig{
			WeightPointer: q4Buffers.Weight.Pointer(),
			ScalePointer:  q4Buffers.Scales.Pointer(),
			BiasPointer:   q4Buffers.Biases.Pointer(),
			WeightBytes:   q4Buffers.Weight.SizeBytes(),
			ScaleBytes:    q4Buffers.Scales.SizeBytes(),
			BiasBytes:     q4Buffers.Biases.SizeBytes(),
			Rows:          q4Req.Rows,
			Cols:          q4Req.Cols,
			GroupSize:     q4Req.GroupSize,
		}, 2)
		core.RequireNoError(t, err)
		defer batchProjected.Close()
		batchProjectedOutput, err := hipReadFloat32DeviceOutput(batchProjected, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch output", q4Req.Rows*2)
		core.RequireNoError(t, err)
		wantProjected := append(
			expectedGELUTanhProjectionFromQ4(t, q4Req, []float32{2, 3}),
			expectedGELUTanhProjectionFromQ4(t, secondReq, []float32{4, 5})...,
		)
		assertFloat32SlicesNear(t, wantProjected, batchProjectedOutput, 0.0001)
	})

	t.Run("mlx-q8-projection", func(t *testing.T) {
		q8Req := hipMLXQ4ProjectionRequest{
			Input: []float32{1, 1, 1, 1, 1, 1, 1, 1},
			Weight: hipPackMLXAffineValuesForTest([]uint32{
				0, 1, 2, 3, 4, 5, 6, 7,
				8, 9, 10, 11, 12, 13, 14, 15,
			}, 8, 8),
			Scales:    []uint16{0x3f80, 0x3f00},
			Biases:    []uint16{0x0000, 0xbf80},
			Rows:      2,
			Cols:      8,
			GroupSize: 8,
			Bits:      8,
		}
		q8Want, err := hipReferenceMLXAffineProjection(q8Req.Input, q8Req.Weight, q8Req.Scales, q8Req.Biases, q8Req.Rows, q8Req.Cols, q8Req.GroupSize, q8Req.Bits)
		core.RequireNoError(t, err)
		q8Output, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, q8Req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, q8Want, q8Output, 0.0001)

		q8Buffers, err := q8Req.deviceBuffers(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer q8Buffers.Close()
		q8Config := hipMLXQ4DeviceWeightConfig{
			WeightPointer: q8Buffers.Weight.Pointer(),
			ScalePointer:  q8Buffers.Scales.Pointer(),
			BiasPointer:   q8Buffers.Biases.Pointer(),
			WeightBytes:   q8Buffers.Weight.SizeBytes(),
			ScaleBytes:    q8Buffers.Scales.SizeBytes(),
			BiasBytes:     q8Buffers.Biases.SizeBytes(),
			Rows:          q8Req.Rows,
			Cols:          q8Req.Cols,
			GroupSize:     q8Req.GroupSize,
			Bits:          q8Req.Bits,
		}
		batchInput := append(append([]float32(nil), q8Req.Input...), []float32{2, 2, 2, 2, 2, 2, 2, 2}...)
		batchPayload, err := hipFloat32Payload(batchInput)
		core.RequireNoError(t, err)
		batchInputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q8 projection batch input", batchPayload, len(batchInput))
		core.RequireNoError(t, err)
		defer batchInputBuffer.Close()
		batchOutputBuffer, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, batchInputBuffer, q8Config, 2)
		core.RequireNoError(t, err)
		defer batchOutputBuffer.Close()
		batchOutput, err := hipReadFloat32DeviceOutput(batchOutputBuffer, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q8 projection batch output", q8Req.Rows*2)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, []float32{q8Want[0], q8Want[1], q8Want[0] * 2, q8Want[1] * 2}, batchOutput, 0.0001)

		batchActivated, err := hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, batchInputBuffer, q8Config, q8Config, 2)
		core.RequireNoError(t, err)
		defer batchActivated.Close()
		activatedOutput, err := hipReadFloat32DeviceOutput(batchActivated, "rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q8 GELU tanh multiply batch output", q8Req.Rows*2)
		core.RequireNoError(t, err)
		secondReq := q8Req
		secondReq.Input = []float32{2, 2, 2, 2, 2, 2, 2, 2}
		wantActivated := append(
			expectedGELUTanhMultiplyFromMLXAffine(t, q8Req, q8Req, 8),
			expectedGELUTanhMultiplyFromMLXAffine(t, secondReq, secondReq, 8)...,
		)
		assertFloat32SlicesNear(t, wantActivated, activatedOutput, 0.0001)

		batchMultiplierPayload, err := hipFloat32Payload([]float32{2, 3, 4, 5})
		core.RequireNoError(t, err)
		batchMultiplier, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q8 GELU tanh projection batch multiplier", batchMultiplierPayload, q8Req.Rows*2)
		core.RequireNoError(t, err)
		defer batchMultiplier.Close()
		batchProjected, err := hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(context.Background(), hipRuntime.driver, batchInputBuffer, batchMultiplier, q8Config, 2)
		core.RequireNoError(t, err)
		defer batchProjected.Close()
		batchProjectedOutput, err := hipReadFloat32DeviceOutput(batchProjected, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q8 GELU tanh projection batch output", q8Req.Rows*2)
		core.RequireNoError(t, err)
		wantProjected := append(
			expectedGELUTanhProjectionFromMLXAffine(t, q8Req, []float32{2, 3}, 8),
			expectedGELUTanhProjectionFromMLXAffine(t, secondReq, []float32{4, 5}, 8)...,
		)
		assertFloat32SlicesNear(t, wantProjected, batchProjectedOutput, 0.0001)
	})

	t.Run("jangtq-projection", func(t *testing.T) {
		jangReq := hipJANGTQProjectionRequest{
			Input:         []float32{2, 4},
			PackedWeights: []byte{0x8d},
			Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
			Rows:          2,
			Cols:          2,
			Scale:         0.5,
			Bias:          []float32{0, 1},
		}
		jangWant, err := rocmReferenceJANGTQProjection(jangReq.Input, jangReq.PackedWeights, jangReq.Descriptor, jangReq.Rows, jangReq.Cols, jangReq.Scale, jangReq.Bias)
		core.RequireNoError(t, err)
		jangOutput, err := hipRunJANGTQProjectionKernel(context.Background(), hipRuntime.driver, jangReq)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, jangWant, jangOutput, 0.0001)
	})

	t.Run("codebook-lookup", func(t *testing.T) {
		codebookReq := hipCodebookLookupRequest{
			Codes:    []uint8{2, 0},
			Codebook: []float32{1, 2, 3, 4, 5, 6},
			CodeDim:  2,
		}
		codebookWant, err := rocmReferenceCodebookLookup(codebookReq.Codes, codebookReq.Codebook, codebookReq.CodeDim)
		core.RequireNoError(t, err)
		codebookOutput, err := hipRunCodebookLookupKernel(context.Background(), hipRuntime.driver, codebookReq)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, codebookWant, codebookOutput, 0.0001)
	})

	t.Run("lora-projection", func(t *testing.T) {
		loraReq := hipLoRAProjectionRequest{
			Input:      []float32{2, 3},
			BaseWeight: []float32{1, 0, 0, 1},
			LoRAA:      []float32{1, 1},
			LoRAB:      []float32{2, -1},
			Rows:       2,
			Cols:       2,
			Rank:       1,
			Alpha:      0.5,
			Bias:       []float32{0.25, -0.5},
		}
		loraWant, err := rocmReferenceLoRAProjection(loraReq.Input, loraReq.BaseWeight, loraReq.LoRAA, loraReq.LoRAB, loraReq.Rows, loraReq.Cols, loraReq.Rank, loraReq.Alpha, loraReq.Bias)
		core.RequireNoError(t, err)
		loraOutput, err := hipRunLoRAProjectionKernel(context.Background(), hipRuntime.driver, loraReq)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, loraWant, loraOutput, 0.0001)
	})

	path, dataOffset := nativeHIPTensorGGUF(t)
	model, err := hipRuntime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 16,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 16,
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	if loaded.KernelStatus().Projection != hipKernelStatusLinked {
		t.Fatalf("kernel status = %+v, want linked projection kernel", loaded.KernelStatus())
	}
	loadedOutput, err := loaded.Project(context.Background(), req)
	core.RequireNoError(t, err)
	if len(loadedOutput) != 1 || math.Abs(float64(loadedOutput[0]-5.5)) > 0.0001 {
		t.Fatalf("loaded projection output = %+v, want [5.5]", loadedOutput)
	}
	loadedQ8Output, err := loaded.Project(context.Background(), q8Req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{4, -2.5}, loadedQ8Output, 0.0001)
}

func TestHIPHardwareEmbeddingKernelSource_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	t.Run("embedding-mean-pool", func(t *testing.T) {
		req := hipEmbeddingMeanPoolRequest{Tokens: []float32{1, 3, 3, 5}, TokenCount: 2, Dim: 2, Normalize: true}
		want, err := rocmReferenceMeanPoolEmbedding(splitFloat32Vectors(req.Tokens, req.Dim), req.Normalize)
		core.RequireNoError(t, err)
		got, err := hipRunEmbeddingMeanPoolKernel(context.Background(), hipRuntime.driver, req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, want, got, 0.0001)
	})

	t.Run("embedding-lookup", func(t *testing.T) {
		f32Req := hipEmbeddingLookupRequest{
			TokenIDs:     []int32{2, 0},
			EmbeddingF32: []float32{1, -2, 0.5, 2, -1, 3},
			VocabSize:    3,
			HiddenSize:   2,
		}
		got, err := hipRunEmbeddingLookupKernel(context.Background(), hipRuntime.driver, f32Req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, []float32{-1, 3, 1, -2}, got, 0.0001)

		bf16Req := hipEmbeddingLookupRequest{
			TokenIDs:      []int32{2, 0},
			EmbeddingBF16: []uint16{0x3f80, 0xc000, 0x3f00, 0x4000, 0xbf80, 0x4040},
			VocabSize:     3,
			HiddenSize:    2,
		}
		got, err = hipRunEmbeddingLookupKernel(context.Background(), hipRuntime.driver, bf16Req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, []float32{-1, 3, 1, -2}, got, 0.0001)

		q4Req := hipEmbeddingLookupRequest{
			TokenIDs:    []int32{2, 0},
			EmbeddingQ4: []uint32{0x76543210, 0x11111111, 0xfedcba98},
			Q4Scales:    []uint16{0x3f80, 0x3f80, 0x3f00},
			Q4Biases:    []uint16{0x0000, 0x0000, 0xbf80},
			Q4GroupSize: 8,
			VocabSize:   3,
			HiddenSize:  8,
		}
		q4Want, err := hipReferenceMLXQ4EmbeddingLookup(q4Req.EmbeddingQ4, q4Req.Q4Scales, q4Req.Q4Biases, q4Req.VocabSize, q4Req.HiddenSize, q4Req.Q4GroupSize, q4Req.TokenIDs)
		core.RequireNoError(t, err)
		got, err = hipRunEmbeddingLookupKernel(context.Background(), hipRuntime.driver, q4Req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, q4Want, got, 0.0001)
	})

	t.Run("rerank-cosine", func(t *testing.T) {
		req := hipRerankCosineRequest{
			Query:         []float32{1, 0},
			Documents:     []float32{0, 1, 1, 1, 1, 0},
			DocumentCount: 3,
			Dim:           2,
		}
		got, err := hipRunRerankCosineKernel(context.Background(), hipRuntime.driver, req)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, []float32{0, 0.70710677, 1}, got, 0.0001)
	})
}

func TestHIPHardwareTransformerKernelSource_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	rmsReq := hipRMSNormRequest{Input: []float32{3, 4}, Weight: []float32{1, 0.5}}
	rmsBuffers, err := rmsReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer rmsBuffers.Close()
	rmsLaunch, err := rmsReq.launchArgs(rmsBuffers)
	core.RequireNoError(t, err)
	rmsLaunchBytes, err := rmsLaunch.Binary()
	core.RequireNoError(t, err)
	rmsConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, rmsLaunchBytes, rmsBuffers.Count)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, rmsConfig))
	rmsOutput, err := rmsBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, rmsOutput, 0.0001)

	bf16RMSReq := hipRMSNormRequest{Input: []float32{3, 4}, WeightBF16: []uint16{0x3f80, 0x3f00}}
	bf16RMSBuffers, err := bf16RMSReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer bf16RMSBuffers.Close()
	bf16RMSLaunch, err := bf16RMSReq.launchArgs(bf16RMSBuffers)
	core.RequireNoError(t, err)
	bf16RMSLaunchBytes, err := bf16RMSLaunch.Binary()
	core.RequireNoError(t, err)
	bf16RMSConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, bf16RMSLaunchBytes, bf16RMSBuffers.Count)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, bf16RMSConfig))
	bf16RMSOutput, err := bf16RMSBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, bf16RMSOutput, 0.0001)

	ropeReq := hipRoPERequest{Input: []float32{1, 0}, Position: 1, Base: 1}
	ropeBuffers, err := ropeReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer ropeBuffers.Close()
	ropeLaunch, err := ropeReq.launchArgs(ropeBuffers)
	core.RequireNoError(t, err)
	ropeLaunchBytes, err := ropeLaunch.Binary()
	core.RequireNoError(t, err)
	ropeConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, ropeLaunchBytes, ropeBuffers.Count)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, ropeConfig))
	ropeOutput, err := ropeBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(1)), float32(math.Sin(1))}, ropeOutput, 0.0001)

	ropeBatchInputValues := []float32{1, 0, 3, 4, 2, 0, 1, 1}
	ropeBatchInputPayload, err := hipFloat32Payload(ropeBatchInputValues)
	core.RequireNoError(t, err)
	ropeBatchInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "hardware rms norm rope heads batch input", ropeBatchInputPayload, len(ropeBatchInputValues))
	core.RequireNoError(t, err)
	defer ropeBatchInput.Close()
	ropeBatchOutput, err := hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(context.Background(), hipRuntime.driver, ropeBatchInput, hipRMSNormDeviceWeightConfig{
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}, 1, 2, 1, 1, 4, 2)
	core.RequireNoError(t, err)
	defer ropeBatchOutput.Close()
	ropeBatchValues, err := hipReadFloat32DeviceOutput(ropeBatchOutput, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "hardware rms norm rope heads batch output", len(ropeBatchInputValues))
	core.RequireNoError(t, err)
	var ropeBatchWant []float32
	unitWeight := []float32{1, 1, 1, 1}
	for batch := 0; batch < 2; batch++ {
		start := batch * 4
		normalized, err := hipReferenceRMSNorm(ropeBatchInputValues[start:start+4], unitWeight, 0)
		core.RequireNoError(t, err)
		rotated, err := hipReferenceRoPEWithFrequencyDim(normalized[:2], 1+batch, 1, 4)
		core.RequireNoError(t, err)
		normalized[0] = rotated[0]
		normalized[1] = rotated[1]
		ropeBatchWant = append(ropeBatchWant, normalized...)
	}
	assertFloat32SlicesNear(t, ropeBatchWant, ropeBatchValues, 0.0001)

	neoxBatchWeightPayload, err := hipUint16Payload([]uint16{0x0000, 0x3f00, 0xbf00, 0x3f80})
	core.RequireNoError(t, err)
	neoxBatchWeight, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "hardware rms norm rope heads batch neox bf16 weight", neoxBatchWeightPayload, 4)
	core.RequireNoError(t, err)
	defer neoxBatchWeight.Close()
	neoxBatchOutput, err := hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(context.Background(), hipRuntime.driver, ropeBatchInput, hipRMSNormDeviceWeightConfig{
		WeightPointer:  neoxBatchWeight.Pointer(),
		WeightBytes:    neoxBatchWeight.SizeBytes(),
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
		Flags:          hipRMSNormLaunchFlagAddUnitWeight | hipRMSNormLaunchFlagRoPENeoX,
	}, 1, 2, 1, 1, 4, 2)
	core.RequireNoError(t, err)
	defer neoxBatchOutput.Close()
	neoxBatchValues, err := hipReadFloat32DeviceOutput(neoxBatchOutput, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "hardware rms norm rope heads batch neox output", len(ropeBatchInputValues))
	core.RequireNoError(t, err)
	neoxBatchWeights := []float32{1, 1.5, 0.5, 2}
	var neoxBatchWant []float32
	for batch := 0; batch < 2; batch++ {
		start := batch * 4
		normalized, err := hipReferenceRMSNorm(ropeBatchInputValues[start:start+4], neoxBatchWeights, 0)
		core.RequireNoError(t, err)
		rotated, err := hipReferenceRoPENeoXWithFrequencyDim(normalized, 1+batch, 1, 4, 2)
		core.RequireNoError(t, err)
		neoxBatchWant = append(neoxBatchWant, rotated...)
	}
	assertFloat32SlicesNear(t, neoxBatchWant, neoxBatchValues, 0.0001)

	greedyReq := hipGreedySampleRequest{Logits: []float32{-1, 0.25, 0.2}}
	greedyBuffers, err := greedyReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer greedyBuffers.Close()
	greedyLaunch, err := greedyReq.launchArgs(greedyBuffers)
	core.RequireNoError(t, err)
	greedyLaunchBytes, err := greedyLaunch.Binary()
	core.RequireNoError(t, err)
	greedyConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameGreedy, greedyLaunchBytes, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, greedyConfig))
	greedyOutput, err := greedyBuffers.ReadOutput()
	core.RequireNoError(t, err)
	if greedyOutput.TokenID != 1 || math.Abs(float64(greedyOutput.Score-0.25)) > 0.0001 {
		t.Fatalf("greedy output = %+v, want token 1 score 0.25", greedyOutput)
	}

	crossEntropyOutput, err := hipRunCrossEntropyLossKernel(context.Background(), hipRuntime.driver, hipCrossEntropyLossRequest{
		Logits:  []float32{2, 0, 0, 2},
		Targets: []int32{0, 1},
		Batch:   2,
		Vocab:   2,
	})
	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.1269, crossEntropyOutput.Loss, 0.0001)
	assertFloat64Near(t, 1.1353, crossEntropyOutput.Perplexity, 0.0001)

	distillationOutput, err := hipRunDistillationKLLossKernel(context.Background(), hipRuntime.driver, hipDistillationKLLossRequest{
		StudentLogits: []float32{1, 0},
		TeacherLogits: []float32{2, 0},
		Batch:         1,
		Vocab:         2,
		Temperature:   1,
	})
	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.0671, distillationOutput.KL, 0.0001)

	grpoOutput, err := hipRunGRPOAdvantageKernel(context.Background(), hipRuntime.driver, hipGRPOAdvantageRequest{
		Rewards: []float64{1, 2, 3},
		Count:   3,
	})
	core.RequireNoError(t, err)
	assertFloat64Near(t, -1.2247, grpoOutput[0], 0.0001)
	assertFloat64Near(t, 0, grpoOutput[1], 0.0001)
	assertFloat64Near(t, 1.2247, grpoOutput[2], 0.0001)

	t.Run("moe-router", func(t *testing.T) {
		moeReq := hipMoERouterRequest{Logits: []float32{0.1, 2, 1, -1}, TopK: 2, Layer: 7}
		moeWant, err := rocmReferenceRouteExperts(moeReq.Logits, moeReq.TopK, moeReq.Layer, nil)
		core.RequireNoError(t, err)
		moeOutput, err := hipRunMoERouterKernel(context.Background(), hipRuntime.driver, moeReq)
		core.RequireNoError(t, err)
		core.AssertEqual(t, hipMoERouterLaunchStatusOK, moeOutput.Status)
		core.AssertEqual(t, len(moeWant), len(moeOutput.Routes))
		for index := range moeWant {
			core.AssertEqual(t, moeWant[index].ID, moeOutput.Routes[index].ID)
			assertFloat32Near(t, moeWant[index].Prob, moeOutput.Routes[index].Prob)
		}
	})

	t.Run("moe-lazy-experts", func(t *testing.T) {
		lazyReq := hipMoELazyExpertRequest{ExpertIDs: []int32{3, 1}, TotalExperts: 5}
		lazyWant, err := rocmReferenceLazyExpertResidency([]rocmExpertRoute{{ID: 3}, {ID: 1}}, lazyReq.TotalExperts)
		core.RequireNoError(t, err)
		lazyOutput, err := hipRunMoELazyExpertKernel(context.Background(), hipRuntime.driver, lazyReq)
		core.RequireNoError(t, err)
		core.AssertEqual(t, lazyWant, lazyOutput.Resident)
	})

	attentionReq := hipAttentionRequest{
		Query:  []float32{1, 0},
		Keys:   []float32{1, 0, 0, 1},
		Values: []float32{2, 0, 0, 4},
	}
	attentionBuffers, err := attentionReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer attentionBuffers.Close()
	attentionLaunch, err := attentionReq.launchArgs(attentionBuffers)
	core.RequireNoError(t, err)
	attentionLaunchBytes, err := attentionLaunch.Binary()
	core.RequireNoError(t, err)
	attentionConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, attentionLaunchBytes, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, attentionConfig))
	attentionOutput, err := attentionBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210}, attentionOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, attentionOutput.Weights, 0.0001)

	attentionCache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, attentionCache.AppendVectors(0, 2, 2, attentionReq.Keys, attentionReq.Values))
	attentionDeviceKV, err := attentionCache.MirrorToDevice(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer attentionDeviceKV.Close()
	attentionTable, err := attentionDeviceKV.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer attentionTable.Close()
	attentionDeviceOutput, err := hipRunAttentionKernel(context.Background(), hipRuntime.driver, hipAttentionRequest{
		Query:           attentionReq.Query,
		DeviceKV:        attentionDeviceKV,
		DescriptorTable: attentionTable,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, attentionOutput.Output, attentionDeviceOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, attentionOutput.Weights, attentionDeviceOutput.Weights, 0.0001)

	for _, mode := range []string{rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4} {
		modeAttentionCache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
		core.RequireNoError(t, err)
		core.RequireNoError(t, modeAttentionCache.AppendVectors(0, 2, 2, attentionReq.Keys, attentionReq.Values))
		modeAttentionDeviceKV, err := modeAttentionCache.MirrorToDevice(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer modeAttentionDeviceKV.Close()
		modeAttentionTable, err := modeAttentionDeviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer modeAttentionTable.Close()
		modeAttentionOutput, err := hipRunAttentionKernel(context.Background(), hipRuntime.driver, hipAttentionRequest{
			Query:           attentionReq.Query,
			DeviceKV:        modeAttentionDeviceKV,
			DescriptorTable: modeAttentionTable,
		})
		core.RequireNoError(t, err)
		restoredKeys, restoredValues, err := modeAttentionCache.Restore(0, modeAttentionCache.TokenCount())
		core.RequireNoError(t, err)
		wantKeys, err := splitHIPReferenceVectors(restoredKeys, 2)
		core.RequireNoError(t, err)
		wantValues, err := splitHIPReferenceVectors(restoredValues, 2)
		core.RequireNoError(t, err)
		wantOutput, wantWeights, err := hipReferenceSingleHeadAttention(attentionReq.Query, wantKeys, wantValues)
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, wantOutput, modeAttentionOutput.Output, 0.0001)
		assertFloat32SlicesNear(t, wantWeights, modeAttentionOutput.Weights, 0.0001)
	}

	t.Run("attention-heads-chunked-direct-token-kv", func(t *testing.T) {
		for _, dim := range []int{256, 512} {
			t.Run(core.Sprintf("dim%d", dim), func(t *testing.T) {
				const tokenCount = 320
				headCount := 2
				if dim == 512 {
					headCount = 4
				}
				queryValues := make([]float32, headCount*dim)
				keyValues := make([]float32, tokenCount*dim)
				valueValues := make([]float32, tokenCount*dim)
				for index := range queryValues {
					queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.75)
				}
				for index := range keyValues {
					keyValues[index] = float32(math.Sin(float64(index)*0.017) * 0.5)
				}
				for index := range valueValues {
					valueValues[index] = float32(math.Cos(float64(index)*0.011) * 0.5)
				}
				cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 1)
				core.RequireNoError(t, err)
				core.RequireNoError(t, cache.AppendVectors(0, dim, dim, keyValues, valueValues))
				deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
				core.RequireNoError(t, err)
				defer deviceKV.Close()
				table, err := deviceKV.KernelDescriptorTable()
				core.RequireNoError(t, err)
				defer table.Close()
				queryPayload, err := hipFloat32Payload(queryValues)
				core.RequireNoError(t, err)
				queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware chunked attention query", queryPayload, len(queryValues))
				core.RequireNoError(t, err)
				defer queryBuffer.Close()
				normalOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware normal attention output", uint64(len(queryValues)*4), len(queryValues))
				core.RequireNoError(t, err)
				defer normalOutput.Close()
				chunkedOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware chunked attention output", uint64(len(queryValues)*4), len(queryValues))
				core.RequireNoError(t, err)
				defer chunkedOutput.Close()
				req := hipAttentionRequest{
					QueryDim:        dim,
					DeviceKV:        deviceKV,
					DescriptorTable: table,
					Scale:           1,
				}
				core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, normalOutput))
				workspace := &hipAttentionHeadsChunkedWorkspace{}
				defer workspace.Close()
				core.RequireNoError(t, hipRunAttentionHeadsChunked(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, dim, tokenCount, chunkedOutput, workspace))
				normalGot, err := hipReadFloat32DeviceOutput(normalOutput, "rocm.hip.AttentionHeadsLaunch", "hardware normal attention output", len(queryValues))
				core.RequireNoError(t, err)
				chunkedGot, err := hipReadFloat32DeviceOutput(chunkedOutput, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware chunked attention output", len(queryValues))
				core.RequireNoError(t, err)
				assertFloat32SlicesNear(t, normalGot, chunkedGot, 0.001)
			})
		}
	})

	t.Run("attention-heads-chunked-block-row-kv", func(t *testing.T) {
		const (
			dim        = 256
			tokenCount = 320
			headCount  = 2
		)
		queryValues := make([]float32, headCount*dim)
		keyValues := make([]float32, tokenCount*dim)
		valueValues := make([]float32, tokenCount*dim)
		for index := range queryValues {
			queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.75)
		}
		for index := range keyValues {
			keyValues[index] = float32(math.Sin(float64(index)*0.017) * 0.5)
		}
		for index := range valueValues {
			valueValues[index] = float32(math.Cos(float64(index)*0.011) * 0.5)
		}
		keyPayload, err := hipFloat32Payload(keyValues)
		core.RequireNoError(t, err)
		keyBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware block row key values", keyPayload, len(keyValues))
		core.RequireNoError(t, err)
		defer keyBuffer.Close()
		valuePayload, err := hipFloat32Payload(valueValues)
		core.RequireNoError(t, err)
		valueBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware block row value values", valuePayload, len(valueValues))
		core.RequireNoError(t, err)
		defer valueBuffer.Close()
		cache := &rocmDeviceKVCache{driver: hipRuntime.driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 16}
		deviceKV, err := cache.withAppendedDeviceRowsWindow(context.Background(), keyBuffer, valueBuffer, dim, dim, tokenCount, 0)
		core.RequireNoError(t, err)
		defer deviceKV.Close()
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer table.Close()
		queryPayload, err := hipFloat32Payload(queryValues)
		core.RequireNoError(t, err)
		queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware block row attention query", queryPayload, len(queryValues))
		core.RequireNoError(t, err)
		defer queryBuffer.Close()
		normalOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware block row normal attention output", uint64(len(queryValues)*4), len(queryValues))
		core.RequireNoError(t, err)
		defer normalOutput.Close()
		chunkedOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware block row chunked attention output", uint64(len(queryValues)*4), len(queryValues))
		core.RequireNoError(t, err)
		defer chunkedOutput.Close()
		req := hipAttentionRequest{
			QueryDim:        dim,
			DeviceKV:        deviceKV,
			DescriptorTable: table,
			Scale:           1,
		}
		core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, normalOutput))
		workspace := &hipAttentionHeadsChunkedWorkspace{}
		defer workspace.Close()
		core.RequireNoError(t, hipRunAttentionHeadsChunked(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, dim, tokenCount, chunkedOutput, workspace))
		normalGot, err := hipReadFloat32DeviceOutput(normalOutput, "rocm.hip.AttentionHeadsLaunch", "hardware block row normal attention output", len(queryValues))
		core.RequireNoError(t, err)
		chunkedGot, err := hipReadFloat32DeviceOutput(chunkedOutput, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware block row chunked attention output", len(queryValues))
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, normalGot, chunkedGot, 0.001)
	})

	t.Run("attention-heads-sliced-interleaved-window-kv-reference", func(t *testing.T) {
		const (
			dim         = 256
			inputTokens = 529
			window      = 512
			headCount   = 2
		)
		queryValues := make([]float32, headCount*dim)
		keyValues := make([]float32, inputTokens*dim)
		valueValues := make([]float32, inputTokens*dim)
		for index := range queryValues {
			queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.75)
		}
		for index := range keyValues {
			keyValues[index] = float32(math.Sin(float64(index)*0.017) * 0.5)
		}
		for index := range valueValues {
			valueValues[index] = float32(math.Cos(float64(index)*0.011) * 0.5)
		}
		keyPayload, err := hipFloat32Payload(keyValues)
		core.RequireNoError(t, err)
		keyBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware sliced interleaved key values", keyPayload, len(keyValues))
		core.RequireNoError(t, err)
		defer keyBuffer.Close()
		valuePayload, err := hipFloat32Payload(valueValues)
		core.RequireNoError(t, err)
		valueBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware sliced interleaved value values", valuePayload, len(valueValues))
		core.RequireNoError(t, err)
		defer valueBuffer.Close()
		cache := &rocmDeviceKVCache{driver: hipRuntime.driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 16}
		deviceKV, err := cache.withAppendedDeviceRowsWindow(context.Background(), keyBuffer, valueBuffer, dim, dim, inputTokens, window)
		core.RequireNoError(t, err)
		defer deviceKV.Close()
		if deviceKV.TokenCount() != window || deviceKV.PageCount() == 0 || deviceKV.pages[0].tokenCount != 15 {
			t.Fatalf("sliced window shape = tokens:%d pages:%d first:%d, want 512 tokens and sliced first page", deviceKV.TokenCount(), deviceKV.PageCount(), deviceKV.pages[0].tokenCount)
		}
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer table.Close()
		queryPayload, err := hipFloat32Payload(queryValues)
		core.RequireNoError(t, err)
		queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware sliced interleaved attention query", queryPayload, len(queryValues))
		core.RequireNoError(t, err)
		defer queryBuffer.Close()
		output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware sliced interleaved attention output", uint64(len(queryValues)*4), len(queryValues))
		core.RequireNoError(t, err)
		defer output.Close()
		req := hipAttentionRequest{
			QueryDim:        dim,
			DeviceKV:        deviceKV,
			DescriptorTable: table,
			Scale:           1,
		}
		core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, output))
		got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsLaunch", "hardware sliced interleaved attention output", len(queryValues))
		core.RequireNoError(t, err)
		host, err := deviceKV.hostCache()
		core.RequireNoError(t, err)
		restoredKeys, restoredValues, err := host.Restore(0, window)
		core.RequireNoError(t, err)
		keys, err := splitHIPReferenceVectors(restoredKeys, dim)
		core.RequireNoError(t, err)
		values, err := splitHIPReferenceVectors(restoredValues, dim)
		core.RequireNoError(t, err)
		want := make([]float32, 0, len(queryValues))
		for head := 0; head < headCount; head++ {
			headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[head*dim:(head+1)*dim], keys, values, 1)
			core.RequireNoError(t, err)
			want = append(want, headOutput...)
		}
		assertFloat32SlicesNear(t, want, got, 0.001)
	})

	t.Run("attention-heads-batch-causal-sliced-interleaved-window-kv-reference", func(t *testing.T) {
		const (
			dim          = 256
			priorTokens  = 529
			window       = 512
			queryCount   = 8
			headCount    = 2
			localKVBlock = 4
		)
		priorKeyValues := make([]float32, priorTokens*dim)
		priorValueValues := make([]float32, priorTokens*dim)
		for index := range priorKeyValues {
			priorKeyValues[index] = float32(math.Sin(float64(index)*0.017) * 0.5)
		}
		for index := range priorValueValues {
			priorValueValues[index] = float32(math.Cos(float64(index)*0.011) * 0.5)
		}
		priorKeyPayload, err := hipFloat32Payload(priorKeyValues)
		core.RequireNoError(t, err)
		priorKeyBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced prior key values", priorKeyPayload, len(priorKeyValues))
		core.RequireNoError(t, err)
		defer priorKeyBuffer.Close()
		priorValuePayload, err := hipFloat32Payload(priorValueValues)
		core.RequireNoError(t, err)
		priorValueBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced prior value values", priorValuePayload, len(priorValueValues))
		core.RequireNoError(t, err)
		defer priorValueBuffer.Close()
		cache := &rocmDeviceKVCache{driver: hipRuntime.driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: localKVBlock}
		priorKV, err := cache.withAppendedDeviceRowsWindow(context.Background(), priorKeyBuffer, priorValueBuffer, dim, dim, priorTokens, window)
		core.RequireNoError(t, err)
		defer priorKV.Close()
		if priorKV.TokenCount() != window || priorKV.PageCount() == 0 || priorKV.pages[0].tokenCount != 3 {
			t.Fatalf("prior sliced window shape = tokens:%d pages:%d first:%d, want 512 tokens and a sliced first block page", priorKV.TokenCount(), priorKV.PageCount(), priorKV.pages[0].tokenCount)
		}

		appendKeyValues := make([]float32, queryCount*dim)
		appendValueValues := make([]float32, queryCount*dim)
		for index := range appendKeyValues {
			appendKeyValues[index] = float32(math.Sin(float64(index+priorTokens*dim)*0.017) * 0.5)
		}
		for index := range appendValueValues {
			appendValueValues[index] = float32(math.Cos(float64(index+priorTokens*dim)*0.011) * 0.5)
		}
		appendKeyPayload, err := hipFloat32Payload(appendKeyValues)
		core.RequireNoError(t, err)
		appendKeyBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced appended key values", appendKeyPayload, len(appendKeyValues))
		core.RequireNoError(t, err)
		defer appendKeyBuffer.Close()
		appendValuePayload, err := hipFloat32Payload(appendValueValues)
		core.RequireNoError(t, err)
		appendValueBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced appended value values", appendValuePayload, len(appendValueValues))
		core.RequireNoError(t, err)
		defer appendValueBuffer.Close()
		deviceKV, err := priorKV.withAppendedDeviceRowsWindow(context.Background(), appendKeyBuffer, appendValueBuffer, dim, dim, queryCount, window+queryCount)
		core.RequireNoError(t, err)
		defer deviceKV.Close()
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer table.Close()

		queryValues := make([]float32, queryCount*headCount*dim)
		for index := range queryValues {
			queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.75)
		}
		queryPayload, err := hipFloat32Payload(queryValues)
		core.RequireNoError(t, err)
		queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced attention query", queryPayload, len(queryValues))
		core.RequireNoError(t, err)
		defer queryBuffer.Close()
		output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced attention output", uint64(len(queryValues)*4), len(queryValues))
		core.RequireNoError(t, err)
		defer output.Close()
		req := hipAttentionHeadsBatchCausalDeviceRequest{
			Dim:             dim,
			DeviceKV:        deviceKV,
			DescriptorTable: table,
			TokenCount:      deviceKV.TokenCount(),
			HeadCount:       headCount,
			QueryCount:      queryCount,
			QueryStartToken: window,
			WindowSize:      window,
			Scale:           1,
		}
		core.RequireNoError(t, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, req, queryBuffer, output))
		got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware batch sliced attention output", len(queryValues))
		core.RequireNoError(t, err)

		host, err := deviceKV.hostCache()
		core.RequireNoError(t, err)
		restoredKeys, restoredValues, err := host.Restore(0, deviceKV.TokenCount())
		core.RequireNoError(t, err)
		keys, err := splitHIPReferenceVectors(restoredKeys, dim)
		core.RequireNoError(t, err)
		values, err := splitHIPReferenceVectors(restoredValues, dim)
		core.RequireNoError(t, err)
		want := make([]float32, 0, len(queryValues))
		for row := 0; row < queryCount; row++ {
			visible := window + row + 1
			windowStart := 0
			if visible > window {
				windowStart = visible - window
			}
			for head := 0; head < headCount; head++ {
				queryOffset := (row*headCount + head) * dim
				headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryOffset:queryOffset+dim], keys[windowStart:visible], values[windowStart:visible], 1)
				core.RequireNoError(t, err)
				want = append(want, headOutput...)
			}
		}
		assertFloat32SlicesNear(t, want, got, 0.001)
	})

	t.Run("attention-heads-batch-chunked-block-kv", func(t *testing.T) {
		const (
			dim             = 4
			tokenCount      = hipAttentionHeadsSharedMaxTokens + 3
			headCount       = 1
			queryCount      = 2
			queryStartToken = tokenCount - queryCount
		)
		queryValues := []float32{
			0.75, -0.25, 0.5, -0.125,
			-0.5, 0.5, -0.375, 0.25,
		}
		keyValues := make([]float32, tokenCount*dim)
		valueValues := make([]float32, tokenCount*dim)
		for index := 0; index < tokenCount; index++ {
			for dimIndex := 0; dimIndex < dim; dimIndex++ {
				keyValues[index*dim+dimIndex] = float32(math.Sin(float64(index+dimIndex)*0.017) * 0.5)
				valueValues[index*dim+dimIndex] = float32(math.Cos(float64(index+dimIndex*2)*0.019) * 0.5)
			}
		}
		cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, hipGemma4Q4DeviceKVBlockSize())
		core.RequireNoError(t, err)
		core.RequireNoError(t, cache.AppendVectors(0, dim, dim, keyValues, valueValues))
		deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer deviceKV.Close()
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer table.Close()
		queryPayload, err := hipFloat32Payload(queryValues)
		core.RequireNoError(t, err)
		queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchChunkedLaunch", "hardware batch chunked attention query", queryPayload, len(queryValues))
		core.RequireNoError(t, err)
		defer queryBuffer.Close()
		output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchChunkedLaunch", "hardware batch chunked attention output", uint64(len(queryValues)*4), len(queryValues))
		core.RequireNoError(t, err)
		defer output.Close()
		workspace := &hipAttentionHeadsChunkedWorkspace{}
		defer workspace.Close()
		core.RequireNoError(t, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(context.Background(), hipRuntime.driver, hipAttentionHeadsBatchCausalDeviceRequest{
			DeviceKV:        deviceKV,
			DescriptorTable: table,
			Dim:             dim,
			TokenCount:      tokenCount,
			HeadCount:       headCount,
			QueryCount:      queryCount,
			QueryStartToken: queryStartToken,
			Scale:           1,
		}, queryBuffer, output, workspace))
		got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchChunkedLaunch", "hardware batch chunked attention output", len(queryValues))
		core.RequireNoError(t, err)
		restoredKeys, restoredValues, err := cache.Restore(0, cache.TokenCount())
		core.RequireNoError(t, err)
		keys, err := splitHIPReferenceVectors(restoredKeys, dim)
		core.RequireNoError(t, err)
		values, err := splitHIPReferenceVectors(restoredValues, dim)
		core.RequireNoError(t, err)
		want := make([]float32, 0, len(queryValues))
		for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
			visibleTokens := queryStartToken + queryIndex + 1
			headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryIndex*dim:(queryIndex+1)*dim], keys[:visibleTokens], values[:visibleTokens], 1)
			core.RequireNoError(t, err)
			want = append(want, headOutput...)
		}
		assertFloat32SlicesNear(t, want, got, 0.005)
	})

	attentionBatchQueryValues := []float32{
		1, 0,
		0, 1,
		0, 1,
		1, 1,
	}
	attentionBatchKeyValues := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	attentionBatchValueValues := []float32{
		2, 0,
		0, 4,
		4, 4,
	}
	attentionBatchQueryPayload, err := hipFloat32Payload(attentionBatchQueryValues)
	core.RequireNoError(t, err)
	attentionBatchQuery, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware attention batch query", attentionBatchQueryPayload, len(attentionBatchQueryValues))
	core.RequireNoError(t, err)
	defer attentionBatchQuery.Close()
	attentionBatchKeyPayload, err := hipFloat32Payload(attentionBatchKeyValues)
	core.RequireNoError(t, err)
	attentionBatchKeys, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware attention batch keys", attentionBatchKeyPayload, len(attentionBatchKeyValues))
	core.RequireNoError(t, err)
	defer attentionBatchKeys.Close()
	attentionBatchValuePayload, err := hipFloat32Payload(attentionBatchValueValues)
	core.RequireNoError(t, err)
	attentionBatchValues, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware attention batch values", attentionBatchValuePayload, len(attentionBatchValueValues))
	core.RequireNoError(t, err)
	defer attentionBatchValues.Close()
	attentionBatchOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware attention batch output", uint64(len(attentionBatchQueryValues)*4), len(attentionBatchQueryValues))
	core.RequireNoError(t, err)
	defer attentionBatchOutput.Close()
	core.RequireNoError(t, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, hipAttentionHeadsBatchCausalDeviceRequest{
		Key:             attentionBatchKeys,
		Value:           attentionBatchValues,
		Dim:             2,
		TokenCount:      3,
		HeadCount:       2,
		QueryCount:      2,
		QueryStartToken: 1,
		Scale:           1,
	}, attentionBatchQuery, attentionBatchOutput))
	attentionBatchGot, err := hipReadFloat32DeviceOutput(attentionBatchOutput, "rocm.hip.AttentionHeadsBatchCausalLaunch", "hardware attention batch output", len(attentionBatchQueryValues))
	core.RequireNoError(t, err)
	attentionBatchKeysSplit, err := splitHIPReferenceVectors(attentionBatchKeyValues, 2)
	core.RequireNoError(t, err)
	attentionBatchValuesSplit, err := splitHIPReferenceVectors(attentionBatchValueValues, 2)
	core.RequireNoError(t, err)
	attentionBatchWant := make([]float32, 0, len(attentionBatchQueryValues))
	for queryIndex := 0; queryIndex < 2; queryIndex++ {
		visibleTokens := 1 + queryIndex + 1
		for head := 0; head < 2; head++ {
			queryBase := (queryIndex*2 + head) * 2
			headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(attentionBatchQueryValues[queryBase:queryBase+2], attentionBatchKeysSplit[:visibleTokens], attentionBatchValuesSplit[:visibleTokens], 1)
			core.RequireNoError(t, err)
			attentionBatchWant = append(attentionBatchWant, headOutput...)
		}
	}
	assertFloat32SlicesNear(t, attentionBatchWant, attentionBatchGot, 0.0001)

	vectorReq := hipVectorAddRequest{Left: []float32{1, -2, 0.5}, Right: []float32{4, 3, -0.25}}
	vectorBuffers, err := vectorReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer vectorBuffers.Close()
	vectorLaunch, err := vectorReq.launchArgs(vectorBuffers)
	core.RequireNoError(t, err)
	vectorLaunchBytes, err := vectorLaunch.Binary()
	core.RequireNoError(t, err)
	vectorConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAdd, vectorLaunchBytes, vectorBuffers.Count)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, vectorConfig))
	vectorOutput, err := vectorBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5, 1, 0.25}, vectorOutput, 0.0001)

	scaleReq := hipVectorScaleRequest{Input: []float32{1, -2, 0.5}, Scale: 4}
	scaleBuffers, err := scaleReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer scaleBuffers.Close()
	scaleLaunch, err := scaleReq.launchArgs(scaleBuffers)
	core.RequireNoError(t, err)
	scaleLaunchBytes, err := scaleLaunch.Binary()
	core.RequireNoError(t, err)
	scaleConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorScale, scaleLaunchBytes, scaleBuffers.Count)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, scaleConfig))
	scaleOutput, err := scaleBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{4, -8, 2}, scaleOutput, 0.0001)

	swigluReq := hipSwiGLURequest{Gate: []float32{0, 1, -1}, Up: []float32{2, 4, 8}}
	swigluBuffers, err := swigluReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer swigluBuffers.Close()
	swigluLaunch, err := swigluReq.launchArgs(swigluBuffers)
	core.RequireNoError(t, err)
	swigluLaunchBytes, err := swigluLaunch.Binary()
	core.RequireNoError(t, err)
	swigluConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameSwiGLU, swigluLaunchBytes, swigluBuffers.Count)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, swigluConfig))
	swigluOutput, err := swigluBuffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0, 2.9242, -2.1515}, swigluOutput, 0.0001)

	smallDecodeReq := hipSmallDecodeFixture("qwen3")
	smallDecodeWant, err := hipReferenceSmallDecode(smallDecodeReq)
	core.RequireNoError(t, err)
	smallDecodeOutput, err := hipRunSmallDecode(context.Background(), hipRuntime.driver, smallDecodeReq)
	core.RequireNoError(t, err)
	core.AssertEqual(t, smallDecodeWant.TokenID, smallDecodeOutput.TokenID)
	assertFloat32Near(t, smallDecodeWant.Score, smallDecodeOutput.Score)
	assertFloat32SlicesNear(t, smallDecodeWant.Logits, smallDecodeOutput.Logits, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.Attention, smallDecodeOutput.Attention, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.UpdatedKeys, smallDecodeOutput.UpdatedKeys, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.UpdatedValues, smallDecodeOutput.UpdatedValues, 0.0001)

	smallPayload, smallTensors := hipSmallDecodeModelPayload(t, "qwen3")
	smallModelPath := core.PathJoin(t.TempDir(), "small-loaded.bin")
	writeSmall := core.WriteFile(smallModelPath, smallPayload, 0o644)
	core.RequireTrue(t, writeSmall.OK)
	smallLoadedModel, err := hipRuntime.LoadModel(smallModelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 3, HiddenSize: 2, NumLayers: 1, QuantBits: 16},
		Tensors:   smallTensors,
	})
	core.RequireNoError(t, err)
	defer smallLoadedModel.Close()
	smallLoaded, ok := smallLoadedModel.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	smallLoadedCfg, err := smallLoaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)
	smallLoadedOutput, err := hipRunLoadedSmallDecode(context.Background(), hipRuntime.driver, smallLoadedCfg, hipLoadedSmallDecodeRequest{
		Input:       smallDecodeReq.Input,
		PriorKeys:   smallDecodeReq.PriorKeys,
		PriorValues: smallDecodeReq.PriorValues,
		Position:    smallDecodeReq.Position,
		RoPEBase:    smallDecodeReq.RoPEBase,
		Epsilon:     smallDecodeReq.Epsilon,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, smallDecodeWant.TokenID, smallLoadedOutput.TokenID)
	assertFloat32Near(t, smallDecodeWant.Score, smallLoadedOutput.Score)
	assertFloat32SlicesNear(t, smallDecodeWant.Logits, smallLoadedOutput.Logits, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.Attention, smallLoadedOutput.Attention, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.UpdatedKeys, smallLoadedOutput.UpdatedKeys, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.UpdatedValues, smallLoadedOutput.UpdatedValues, 0.0001)

	smallCache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, smallCache.AppendVectors(0, smallDecodeReq.HiddenSize, smallDecodeReq.HiddenSize, smallDecodeReq.PriorKeys, smallDecodeReq.PriorValues))
	smallDecoded, err := smallLoaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: smallCache})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(smallDecodeWant.TokenID), smallDecoded.Token.ID)
	core.AssertEqual(t, 3, smallDecoded.KV.TokenCount())
	smallDecodedKeys, smallDecodedValues, err := smallDecoded.KV.Restore(0, smallDecoded.KV.TokenCount())
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, smallDecodeWant.Logits, smallDecoded.Logits, 0.0001)
	assertFloat32SlicesNear(t, smallDecodeWant.UpdatedKeys, smallDecodedKeys, 0.0005)
	assertFloat32SlicesNear(t, smallDecodeWant.UpdatedValues, smallDecodedValues, 0.0005)
	core.AssertEqual(t, "loaded_device", smallDecoded.Labels["decode_tensor_backing"])
	core.AssertEqual(t, "2", smallDecoded.Labels["decode_launch_token"])

	for _, tt := range []struct {
		mode           string
		keyTolerance   float32
		valueTolerance float32
	}{
		{mode: rocmKVCacheModeQ8, keyTolerance: 0.01, valueTolerance: 0.03},
		{mode: rocmKVCacheModeKQ8VQ4, keyTolerance: 0.01, valueTolerance: 0.15},
	} {
		t.Run("loaded-small-"+tt.mode, func(t *testing.T) {
			modeCache, err := newROCmKVCache(tt.mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, modeCache.AppendVectors(0, smallDecodeReq.HiddenSize, smallDecodeReq.HiddenSize, smallDecodeReq.PriorKeys, smallDecodeReq.PriorValues))
			modeDecoded, err := smallLoaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: modeCache})
			core.RequireNoError(t, err)
			core.AssertEqual(t, int32(smallDecodeWant.TokenID), modeDecoded.Token.ID)
			core.AssertEqual(t, 3, modeDecoded.KV.TokenCount())
			core.AssertEqual(t, tt.mode, modeDecoded.KV.Stats().CacheMode)
			modeKeys, modeValues, err := modeDecoded.KV.Restore(0, modeDecoded.KV.TokenCount())
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, smallDecodeWant.Logits, modeDecoded.Logits, 0.0001)
			assertFloat32SlicesNear(t, smallDecodeWant.UpdatedKeys, modeKeys, tt.keyTolerance)
			assertFloat32SlicesNear(t, smallDecodeWant.UpdatedValues, modeValues, tt.valueTolerance)
			core.AssertEqual(t, "loaded_device", modeDecoded.Labels["decode_tensor_backing"])
			core.AssertEqual(t, "2", modeDecoded.Labels["decode_launch_token"])

			modeDeviceCache, err := newROCmKVCache(tt.mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, modeDeviceCache.AppendVectors(0, smallDecodeReq.HiddenSize, smallDecodeReq.HiddenSize, smallDecodeReq.PriorKeys, smallDecodeReq.PriorValues))
			modeDeviceKV, modeTable, err := hipMirrorTinyKV(hipRuntime.driver, modeDeviceCache, map[string]string{})
			core.RequireNoError(t, err)
			defer modeDeviceKV.Close()
			defer modeTable.Close()
			modeDecodedWithDevice, err := smallLoaded.DecodeToken(context.Background(), hipDecodeRequest{
				TokenID:         2,
				KV:              modeDeviceCache,
				DeviceKV:        modeDeviceKV,
				DescriptorTable: modeTable,
			})
			core.RequireNoError(t, err)
			defer modeDecodedWithDevice.DeviceKV.Close()
			defer modeDecodedWithDevice.DescriptorTable.Close()
			core.AssertEqual(t, int32(smallDecodeWant.TokenID), modeDecodedWithDevice.Token.ID)
			core.AssertEqual(t, 2, modeDeviceCache.TokenCount())
			core.AssertEqual(t, 3, modeDecodedWithDevice.KV.TokenCount())
			core.AssertEqual(t, 3, modeDecodedWithDevice.DeviceKV.TokenCount())
			core.AssertEqual(t, tt.mode, modeDecodedWithDevice.KV.Stats().CacheMode)
			core.AssertEqual(t, tt.mode, modeDecodedWithDevice.DeviceKV.Stats().CacheMode)
			if !modeDeviceKV.closed || !modeTable.closed {
				t.Fatalf("original %s device resources should be closed after successful small decode remirror", tt.mode)
			}
			modeDeviceKeys, modeDeviceValues, err := modeDecodedWithDevice.KV.Restore(0, modeDecodedWithDevice.KV.TokenCount())
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, smallDecodeWant.Logits, modeDecodedWithDevice.Logits, 0.0001)
			assertFloat32SlicesNear(t, smallDecodeWant.UpdatedKeys, modeDeviceKeys, tt.keyTolerance)
			assertFloat32SlicesNear(t, smallDecodeWant.UpdatedValues, modeDeviceValues, tt.valueTolerance)
			core.AssertEqual(t, "loaded_device", modeDecodedWithDevice.Labels["decode_tensor_backing"])
			core.AssertEqual(t, "hip_device", modeDecodedWithDevice.Labels["kv_descriptor_table"])
			core.AssertEqual(t, "3", modeDecodedWithDevice.Labels["kv_tokens"])
		})
	}

	fixture := hipReferenceTinyLMFixture()
	tinyReq := hipTinyPrefillRequest{
		TokenIDs:       []int32{0, 1},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	tinyBuffers, err := tinyReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer tinyBuffers.Close()
	tinyLaunch, err := tinyReq.launchArgs(tinyBuffers)
	core.RequireNoError(t, err)
	tinyLaunchBytes, err := tinyLaunch.Binary()
	core.RequireNoError(t, err)
	tinyConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyPrefill, tinyLaunchBytes, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, tinyConfig))
	tinyOutput, err := tinyBuffers.ReadOutput()
	core.RequireNoError(t, err)
	if tinyOutput.NextTokenID != 2 || math.Abs(float64(tinyOutput.NextScore-1)) > 0.0001 {
		t.Fatalf("tiny prefill result = %+v, want token 2 score 1", tinyOutput)
	}
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, tinyOutput.Logits, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, tinyOutput.Attention, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, tinyOutput.StateKeys, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, tinyOutput.StateValues, 0.0001)

	tinyDecodeReq := hipTinyDecodeRequest{
		TokenID:        2,
		PriorKeys:      tinyOutput.StateKeys,
		PriorValues:    tinyOutput.StateValues,
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	tinyDecodeBuffers, err := tinyDecodeReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer tinyDecodeBuffers.Close()
	tinyDecodeLaunch, err := tinyDecodeReq.launchArgs(tinyDecodeBuffers)
	core.RequireNoError(t, err)
	tinyDecodeLaunchBytes, err := tinyDecodeLaunch.Binary()
	core.RequireNoError(t, err)
	tinyDecodeConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyDecode, tinyDecodeLaunchBytes, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, tinyDecodeConfig))
	tinyDecodeOutput, err := tinyDecodeBuffers.ReadOutput()
	core.RequireNoError(t, err)
	if tinyDecodeOutput.NextTokenID != 2 || math.Abs(float64(tinyDecodeOutput.NextScore-1.5035)) > 0.0001 {
		t.Fatalf("tiny decode result = %+v, want token 2 score 1.5035", tinyDecodeOutput)
	}
	assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, tinyDecodeOutput.Logits, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.2483, 0.2483, 0.5035}, tinyDecodeOutput.Attention, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 1, 1}, tinyDecodeOutput.UpdatedKeys, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 1, 1}, tinyDecodeOutput.UpdatedValues, 0.0001)

	for _, tt := range []struct {
		name    string
		fp16    []uint16
		q8      []int8
		q8Scale float32
	}{{
		name: "fp16-output",
		fp16: hipTinyOutputWeightsFP16Fixture(),
	}, {
		name:    "q8-output",
		q8:      hipTinyOutputWeightsQ8Fixture(),
		q8Scale: 0.5,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			prefillReq := hipTinyPrefillRequest{
				TokenIDs:       []int32{0, 1},
				EmbeddingTable: fixture.EmbeddingTable,
				OutputFP16:     tt.fp16,
				OutputQ8:       tt.q8,
				Q8Scale:        tt.q8Scale,
				VocabSize:      fixture.VocabSize,
				HiddenSize:     fixture.HiddenSize,
			}
			prefillBuffers, err := prefillReq.deviceBuffers(hipRuntime.driver)
			core.RequireNoError(t, err)
			defer prefillBuffers.Close()
			prefillLaunch, err := prefillReq.launchArgs(prefillBuffers)
			core.RequireNoError(t, err)
			prefillLaunchBytes, err := prefillLaunch.Binary()
			core.RequireNoError(t, err)
			prefillConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyPrefill, prefillLaunchBytes, 1)
			core.RequireNoError(t, err)
			core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, prefillConfig))
			prefillOutput, err := prefillBuffers.ReadOutput()
			core.RequireNoError(t, err)
			core.AssertEqual(t, 2, prefillOutput.NextTokenID)
			assertFloat32Near(t, 1, prefillOutput.NextScore)
			assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, prefillOutput.Logits, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, prefillOutput.StateKeys, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, prefillOutput.StateValues, 0.0001)

			decodeReq := hipTinyDecodeRequest{
				TokenID:        2,
				PriorKeys:      prefillOutput.StateKeys,
				PriorValues:    prefillOutput.StateValues,
				EmbeddingTable: fixture.EmbeddingTable,
				OutputFP16:     tt.fp16,
				OutputQ8:       tt.q8,
				Q8Scale:        tt.q8Scale,
				VocabSize:      fixture.VocabSize,
				HiddenSize:     fixture.HiddenSize,
			}
			decodeBuffers, err := decodeReq.deviceBuffers(hipRuntime.driver)
			core.RequireNoError(t, err)
			defer decodeBuffers.Close()
			decodeLaunch, err := decodeReq.launchArgs(decodeBuffers)
			core.RequireNoError(t, err)
			decodeLaunchBytes, err := decodeLaunch.Binary()
			core.RequireNoError(t, err)
			decodeConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyDecode, decodeLaunchBytes, 1)
			core.RequireNoError(t, err)
			core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, decodeConfig))
			decodeOutput, err := decodeBuffers.ReadOutput()
			core.RequireNoError(t, err)
			core.AssertEqual(t, 2, decodeOutput.NextTokenID)
			assertFloat32Near(t, 1.5035, decodeOutput.NextScore)
			assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, decodeOutput.Logits, 0.0001)
		})
	}

	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	for _, tt := range []struct {
		name           string
		outputType     uint32
		outputTypeName string
		outputPayload  []byte
	}{{
		name:       "loaded-tiny-f32",
		outputType: 0,
	}, {
		name:           "loaded-tiny-q8",
		outputType:     24,
		outputTypeName: "q8:0.5",
		outputPayload:  hipInt8Payload(hipTinyOutputWeightsQ8Fixture()),
	}} {
		t.Run(tt.name, func(t *testing.T) {
			outputPayload := tt.outputPayload
			if len(outputPayload) == 0 {
				var err error
				outputPayload, err = hipFloat32Payload(fixture.OutputWeights)
				core.RequireNoError(t, err)
			}
			tinyModelPath := core.PathJoin(t.TempDir(), "tiny-loaded.bin")
			write := core.WriteFile(tinyModelPath, append(append([]byte(nil), embeddingPayload...), outputPayload...), 0o644)
			core.RequireTrue(t, write.OK)
			loadedModel, err := hipRuntime.LoadModel(tinyModelPath, nativeLoadConfig{
				ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
				Tensors: []nativeTensorInfo{{
					Name:       "tok_embeddings.weight",
					Type:       0,
					Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
					Offset:     0,
					ByteSize:   uint64(len(embeddingPayload)),
				}, {
					Name:       "output.weight",
					Type:       tt.outputType,
					TypeName:   tt.outputTypeName,
					Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
					Offset:     uint64(len(embeddingPayload)),
					ByteSize:   uint64(len(outputPayload)),
				}},
			})
			core.RequireNoError(t, err)
			defer loadedModel.Close()
			loadedTiny, ok := loadedModel.(*hipLoadedModel)
			core.RequireTrue(t, ok)
			if loadedTiny.KernelStatus().Decode != hipKernelStatusLinked || loadedTiny.KernelStatus().Prefill != hipKernelStatusLinked {
				t.Fatalf("loaded tiny kernel status = %+v, want linked prefill/decode", loadedTiny.KernelStatus())
			}
			loadedStream, loadedErr := loadedTiny.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 2})
			var loadedIDs []int32
			for token := range loadedStream {
				loadedIDs = append(loadedIDs, token.ID)
			}
			core.RequireNoError(t, loadedErr())
			core.AssertEqual(t, []int32{1, 1}, loadedIDs)
		})
	}
}

func TestHIPHardwarePrefillDecodeKernelSource_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	cases := []struct {
		name       string
		mode       string
		keyWidth   int
		valueWidth int
		keys       []float32
		values     []float32
	}{{
		name:       "fp16",
		mode:       rocmKVCacheModeFP16,
		keyWidth:   2,
		valueWidth: 2,
		keys:       []float32{1, 0, 0, 1},
		values:     []float32{0.5, 0, 0, 0.5},
	}, {
		name:       "q8",
		mode:       rocmKVCacheModeQ8,
		keyWidth:   2,
		valueWidth: 2,
		keys:       []float32{1, 0, 0, 1},
		values:     []float32{2, 0, 0, 2},
	}, {
		name:       "k-q8-v-q4",
		mode:       rocmKVCacheModeKQ8VQ4,
		keyWidth:   2,
		valueWidth: 3,
		keys:       []float32{1, 0.5, -1, 0},
		values:     []float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	}}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tokenBuffer, err := hipUploadTokenIDs(hipRuntime.driver, []int32{11, 12})
			core.RequireNoError(t, err)
			defer tokenBuffer.Close()
			prefillStatus, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.Test", "prefill status", make([]byte, 4), 1)
			core.RequireNoError(t, err)
			defer prefillStatus.Close()
			prefillLaunch, err := (hipPrefillRequest{
				TokenIDs:   []int32{11, 12},
				CacheMode:  tt.mode,
				KeyWidth:   tt.keyWidth,
				ValueWidth: tt.valueWidth,
			}).prefillLaunchArgs(tokenBuffer)
			core.RequireNoError(t, err)
			prefillLaunch.StatusPointer = prefillStatus.Pointer()
			prefillLaunchBytes, err := prefillLaunch.Binary()
			core.RequireNoError(t, err)
			prefillConfig, err := hipOneDimensionalLaunchConfig(hipKernelNamePrefill, prefillLaunchBytes, tokenBuffer.Count())
			core.RequireNoError(t, err)
			core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, prefillConfig))
			if got := readHIPDeviceUint32(t, hipRuntime.driver, prefillStatus.Pointer()); got != hipPrefillLaunchStatusOK {
				t.Fatalf("prefill status = %#x, want %#x", got, hipPrefillLaunchStatusOK)
			}

			cache, err := newROCmKVCache(tt.mode, 2)
			core.RequireNoError(t, err)
			core.RequireNoError(t, cache.AppendVectors(0, tt.keyWidth, tt.valueWidth, tt.keys, tt.values))
			device, err := cache.MirrorToDevice(hipRuntime.driver)
			core.RequireNoError(t, err)
			defer device.Close()
			table, err := device.KernelDescriptorTable()
			core.RequireNoError(t, err)
			defer table.Close()
			decodeStatus, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.Test", "decode status", make([]byte, 4), 1)
			core.RequireNoError(t, err)
			defer decodeStatus.Close()
			decodeLaunch, err := (hipDecodeRequest{
				TokenID:         13,
				KV:              cache,
				DeviceKV:        device,
				DescriptorTable: table,
				KeyWidth:        tt.keyWidth,
				ValueWidth:      tt.valueWidth,
			}).decodeLaunchArgs()
			core.RequireNoError(t, err)
			decodeLaunch.KV.StatusPointer = decodeStatus.Pointer()
			decodeLaunchBytes, err := decodeLaunch.Binary()
			core.RequireNoError(t, err)
			decodeConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameDecode, decodeLaunchBytes, 1)
			core.RequireNoError(t, err)
			core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, decodeConfig))
			if got := readHIPDeviceUint32(t, hipRuntime.driver, decodeStatus.Pointer()); got != hipDecodeLaunchStatusOK {
				t.Fatalf("decode status = %#x, want %#x", got, hipDecodeLaunchStatusOK)
			}
		})
	}
}

func readHIPDeviceUint32(t *testing.T, driver nativeHIPDriver, pointer nativeDevicePointer) uint32 {
	t.Helper()
	payload := make([]byte, 4)
	core.RequireNoError(t, driver.CopyDeviceToHost(pointer, payload))
	return binary.LittleEndian.Uint32(payload)
}
