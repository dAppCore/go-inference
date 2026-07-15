// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
	"dappco.re/go/inference/engine/hip/internal/gguf"
	"dappco.re/go/inference/model"
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

func TestHIPHardwareDiffusionExpectedEmbedding_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || !runtime.Available() || hipRuntime.driver == nil {
		t.Fatal("native ROCm runtime is not available")
	}

	const vocab, hidden, rows = 3, 4, 2
	embeddingValues := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	embeddingPayload := make([]byte, len(embeddingValues)*2)
	for index, value := range embeddingValues {
		binary.LittleEndian.PutUint16(embeddingPayload[index*2:], hipFloat32ToBFloat16(value))
	}
	embedding, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.DiffusionExpectedEmbeddingHardware", "embedding", embeddingPayload, len(embeddingValues))
	core.RequireNoError(t, err)
	defer embedding.Close()
	probabilities := []float32{0.5, 0.25, 0.25, 0.2, 0.3, 0.5}
	got, err := hipRunDiffusionExpectedEmbeddingKernel(context.Background(), hipRuntime.driver, probabilities, rows, hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embedding.Pointer(),
		EmbeddingBytes:   embedding.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingBF16,
		VocabSize:        vocab,
		HiddenSize:       hidden,
	}, 2)
	core.RequireNoError(t, err)
	want := make([]float32, rows*hidden)
	for row := 0; row < rows; row++ {
		for dim := 0; dim < hidden; dim++ {
			for token := 0; token < vocab; token++ {
				want[row*hidden+dim] += probabilities[row*vocab+token] * embeddingValues[token*hidden+dim] * 2
			}
		}
	}
	assertFloat32SlicesNearRelativeNamedForHardwareTest(t, "diffusion expected embedding", want, got, 0.00001, 0.00001)

	const affineVocab, affineHidden, affineGroupSize = 3, 2816, 64
	for _, bits := range []int{4, 8} {
		values := make([]uint32, affineVocab*affineHidden)
		dense := make([]float32, len(values))
		scaleValues := []float32{1, 0.5, 0.25}
		biasValues := []float32{0, -1, 2}
		groupsPerRow := affineHidden / affineGroupSize
		scales := make([]uint16, affineVocab*groupsPerRow)
		biases := make([]uint16, affineVocab*groupsPerRow)
		for token := 0; token < affineVocab; token++ {
			for group := 0; group < groupsPerRow; group++ {
				scales[token*groupsPerRow+group] = hipFloat32ToBFloat16(scaleValues[token])
				biases[token*groupsPerRow+group] = hipFloat32ToBFloat16(biasValues[token])
			}
			for dim := 0; dim < affineHidden; dim++ {
				index := token*affineHidden + dim
				values[index] = uint32((token*29 + dim*3) % (1 << bits))
				dense[index] = float32(values[index])*scaleValues[token] + biasValues[token]
			}
		}
		packed := hipPackMLXAffineValuesForTest(values, affineHidden, bits)
		payload, err := hipUint32Payload(packed)
		core.RequireNoError(t, err)
		label := core.Sprintf("q%d", bits)
		embedding, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.DiffusionExpectedEmbeddingHardware", label+" embedding", payload, len(packed))
		core.RequireNoError(t, err)
		defer embedding.Close()
		scalePayload, err := hipUint16Payload(scales)
		core.RequireNoError(t, err)
		scaleBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.DiffusionExpectedEmbeddingHardware", label+" scales", scalePayload, len(scales))
		core.RequireNoError(t, err)
		defer scaleBuffer.Close()
		biasPayload, err := hipUint16Payload(biases)
		core.RequireNoError(t, err)
		biasBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.DiffusionExpectedEmbeddingHardware", label+" biases", biasPayload, len(biases))
		core.RequireNoError(t, err)
		defer biasBuffer.Close()
		rowCounts := []int{33}
		for _, affineRows := range rowCounts {
			probabilities := make([]float32, affineRows*affineVocab)
			for row := 0; row < affineRows; row++ {
				for token := 0; token < affineVocab; token++ {
					probabilities[row*affineVocab+token] = float32((row+1)*(token+2)) / 37
				}
			}
			got, err := hipRunDiffusionExpectedEmbeddingKernel(context.Background(), hipRuntime.driver, probabilities, affineRows, hipDeviceEmbeddingLookupConfig{
				EmbeddingPointer: embedding.Pointer(),
				EmbeddingBytes:   embedding.SizeBytes(),
				ScalePointer:     scaleBuffer.Pointer(),
				ScaleBytes:       scaleBuffer.SizeBytes(),
				BiasPointer:      biasBuffer.Pointer(),
				BiasBytes:        biasBuffer.SizeBytes(),
				TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
				GroupSize:        affineGroupSize,
				QuantBits:        bits,
				VocabSize:        affineVocab,
				HiddenSize:       affineHidden,
			}, 2)
			core.RequireNoError(t, err)
			want := make([]float32, affineRows*affineHidden)
			for row := 0; row < affineRows; row++ {
				for dim := 0; dim < affineHidden; dim++ {
					for token := 0; token < affineVocab; token++ {
						want[row*affineHidden+dim] += probabilities[row*affineVocab+token] * dense[token*affineHidden+dim] * 2
					}
				}
			}
			assertFloat32SlicesNearRelativeNamedForHardwareTest(t, core.Sprintf("diffusion expected embedding %s group64 rows%d", label, affineRows), want, got, 0.00001, 0.00001)
		}
	}
}

func TestHIPHardwareDiffusionSampleProbabilities_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || !runtime.Available() || hipRuntime.driver == nil {
		t.Fatal("native ROCm runtime is not available")
	}

	const rows, vocab = 3, 67
	const temperature, softcap = float32(0.73), float32(30)
	raw := make([]float32, rows*vocab)
	for index := range raw {
		raw[index] = float32(math.Sin(float64(index)*0.37))*19 + float32(index%11-5)*0.625
	}
	raw[4] = 58
	raw[vocab+17] = 42
	raw[2*vocab+66] = 37

	hostShaped, err := hipGemma4Q4SoftcapLogits(append([]float32(nil), raw...), softcap)
	core.RequireNoError(t, err)
	shapedBF16 := make([]byte, len(hostShaped)*2)
	for index, value := range hostShaped {
		bf16 := hipFloat32ToBFloat16(value / temperature)
		binary.LittleEndian.PutUint16(shapedBF16[index*2:], bf16)
		hostShaped[index] = hipBFloat16ToFloat32(bf16)
	}

	const seed = uint64(0x4d595df4d0f33173)
	drawSampler := model.NewSampler(seed)
	hostSampler := model.NewSampler(seed)
	draws := make([]float32, rows)
	wantSampled := make([]int32, rows)
	wantGreedy := make([]int32, rows)
	wantEntropy := make([]float32, rows)
	wantProbabilities := make([]float32, len(raw))
	for row := range rows {
		draws[row] = drawSampler.Draw()
		rowBytes := shapedBF16[row*vocab*2 : (row+1)*vocab*2]
		wantSampled[row], err = hostSampler.Sample(rowBytes, vocab, model.SampleParams{Temperature: 1})
		core.RequireNoError(t, err)
		wantGreedy[row], err = model.Greedy(rowBytes, vocab)
		core.RequireNoError(t, err)
		wantEntropy[row], err = rocmDiffusionSoftmaxEntropy(hostShaped[row*vocab:(row+1)*vocab], wantProbabilities[row*vocab:(row+1)*vocab])
		core.RequireNoError(t, err)
	}

	logits, err := hipUploadGemma4Q4Float32Input(hipRuntime.driver, "diffusion sample hardware logits", raw)
	core.RequireNoError(t, err)
	defer logits.Close()
	got, err := hipRunDiffusionSampleKernel(context.Background(), hipRuntime.driver, logits, rows, vocab, temperature, softcap, draws)
	core.RequireNoError(t, err)
	gotProbabilities, err := hipReadFloat32DeviceOutput(logits, "rocm.hip.DiffusionSampleHardware", "probabilities", len(raw))
	core.RequireNoError(t, err)
	assertFloat32SlicesNearRelativeNamedForHardwareTest(t, "diffusion sample probabilities", wantProbabilities, gotProbabilities, 0.00001, 0.00001)
	for row := range rows {
		core.AssertEqual(t, wantSampled[row], got[row].Sampled)
		core.AssertEqual(t, wantGreedy[row], got[row].Greedy)
		assertFloat32Near(t, wantEntropy[row], got[row].Entropy)
	}
}

func TestHIPHardwareDiffusionGemmaMLXMoE_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_DIFFUSION_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_DIFFUSION_MODEL_PATH to a DiffusionGemma MLX affine checkpoint")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatal("native ROCm runtime is not available")
	}
	textModel, err := newROCmBackendWithRuntime(runtime).loadModelWithROCmConfig(modelPath, inference.LoadConfig{ContextLen: 64}, ROCmLoadConfig{})
	if err != nil {
		t.Fatalf("load DiffusionGemma MLX MoE model: %v", err)
	}
	defer textModel.Close()
	rocmLoaded, ok := textModel.(*rocmModel)
	core.RequireTrue(t, ok)
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "diffusion_gemma", core.Lower(loaded.modelInfo.Architecture))
	core.AssertEqual(t, true, loaded.gemma4TextConfig.EnableMoEBlock)
	core.AssertEqual(t, 128, loaded.gemma4TextConfig.NumExperts)
	core.AssertEqual(t, 8, loaded.gemma4TextConfig.TopKExperts)
	core.AssertEqual(t, 704, loaded.gemma4TextConfig.MoEIntermediateSize)
	core.AssertEqual(t, loaded.modelInfo.NumLayers*6, len(loaded.hostTensors))
	moe, err := loaded.loadedGemma4MoELayerConfig(0, loaded.modelInfo.HiddenSize)
	core.RequireNoError(t, err)
	core.RequireTrue(t, moe != nil)
	core.AssertEqual(t, hipGemma4MoEExpertStorageMLXAffine, moe.ExpertStorage)
	core.AssertEqual(t, 8, moe.RouterProjectionMLX.Bits)
	core.AssertEqual(t, 4, moe.MLXPreferredBits)
	core.AssertEqual(t, 64, moe.MLXGroupSize)

	session, err := loaded.OpenROCmDiffusionSession(context.Background())
	core.RequireNoError(t, err)
	closer, ok := session.(interface{ Close() error })
	core.RequireTrue(t, ok)
	defer closer.Close()
	position, err := session.PrefillTokens([]int32{1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, position)
	core.AssertEqual(t, 1, session.CacheOffset())
	core.RequireTrue(t, loaded.expertCache != nil)
	core.AssertGreater(t, len(loaded.expertCache.entries), 0)
	t.Logf("DiffusionGemma MLX MoE resident prefill: device=%s host_expert_tensors=%d cached_experts=%d cache_bytes=%d", loaded.driver.DeviceInfo().Name, len(loaded.hostTensors), len(loaded.expertCache.entries), loaded.expertCache.bytes)
	core.RequireNoError(t, closer.Close())

	rocmLoaded.modelLabels["diffusion_canvas_length"] = "1"
	rocmLoaded.modelLabels["diffusion_default_max_steps"] = "1"
	generated := make([]int32, 0, 1)
	metrics, err := rocmLoaded.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{
		MaxTokens: 1,
		Seed:      1,
		SeedSet:   true,
	}, func(token int32) bool {
		generated = append(generated, token)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	core.AssertEqual(t, 1, len(generated))
	core.AssertEqual(t, 1, metrics.EmittedTokens)
	core.AssertEqual(t, 1, metrics.TotalSteps)
	t.Logf("DiffusionGemma MLX MoE one-step generation: token=%d prefill=%s denoise=%s commit=%s total=%s", generated[0], metrics.PrefillDur, metrics.DenoiseDur, metrics.CommitDur, metrics.TotalDur)
}

func TestHIPHardwareGemma4AudioChat_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_AUDIO_CHAT_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_AUDIO_CHAT_TESTS=1 to run the Gemma 4 audio chat smoke")
	}
	textPath := strings.TrimSpace(os.Getenv("GO_ROCM_AUDIO_TEXT_MODEL_PATH"))
	audioPath := strings.TrimSpace(os.Getenv("GO_ROCM_AUDIO_MODEL_PATH"))
	if textPath == "" || audioPath == "" {
		t.Fatal("GO_ROCM_AUDIO_TEXT_MODEL_PATH and GO_ROCM_AUDIO_MODEL_PATH are required")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Fatal("GO_ROCM_KERNEL_HSACO is required")
	}

	textModel, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(
		textPath,
		ROCmLoadConfig{AudioModelPath: audioPath},
		inference.WithContextLen(4096),
	)
	if err != nil {
		t.Fatalf("LoadModelWithConfig audio: %v", err)
	}
	defer textModel.Close()
	audioModel, ok := textModel.(inference.AudioModel)
	if !ok || !audioModel.AcceptsAudio() {
		t.Fatalf("loaded model audio capability = %T/%v, want true", textModel, ok)
	}

	waveform := syntheticWaveform(8000)
	samples := make([]int16, len(waveform))
	for index, value := range waveform {
		samples[index] = int16(value * 32767)
	}
	wav := hipTestPCM16WAV(16000, 1, samples)
	generated := 0
	for range textModel.Chat(context.Background(), []inference.Message{{
		Role: "user", Content: "Describe this sound briefly.", Audios: [][]byte{wav},
	}}, inference.WithMaxTokens(1), inference.WithTemperature(0)) {
		generated++
	}
	if err := resultError(textModel.Err()); err != nil {
		t.Fatalf("audio Chat: %v", err)
	}
	if generated != 1 {
		t.Fatalf("audio Chat generated %d tokens, want 1", generated)
	}
}

func TestHIPHardwareGemma4VisionChat_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_VISION_CHAT_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_VISION_CHAT_TESTS=1 to run the Gemma 4 vision chat smoke")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_ENCODER_VISION_MODEL_PATH"))
	if modelPath == "" {
		t.Fatal("GO_ROCM_ENCODER_VISION_MODEL_PATH is required")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Fatal("GO_ROCM_KERNEL_HSACO is required")
	}

	textModel, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(
		modelPath,
		ROCmLoadConfig{VisionModelPath: modelPath},
		inference.WithContextLen(4096),
	)
	if err != nil {
		t.Fatalf("LoadModelWithConfig vision: %v", err)
	}
	defer textModel.Close()
	visionModel, ok := textModel.(inference.VisionModel)
	if !ok || !visionModel.AcceptsImages() {
		t.Fatalf("loaded model vision capability = %T/%v, want true", textModel, ok)
	}

	img := image.NewRGBA(image.Rect(0, 0, 96, 96))
	for y := range 96 {
		for x := range 96 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x * 2), G: uint8(y * 2), B: 80, A: 255})
		}
	}
	var encoded bytes.Buffer
	core.RequireNoError(t, png.Encode(&encoded, img))
	generated := 0
	for range textModel.Chat(context.Background(), []inference.Message{{
		Role: "user", Content: "Describe this image briefly.", Images: [][]byte{encoded.Bytes()},
	}}, inference.WithMaxTokens(1), inference.WithTemperature(0)) {
		generated++
	}
	if err := resultError(textModel.Err()); err != nil {
		t.Fatalf("vision Chat: %v", err)
	}
	if generated != 1 {
		t.Fatalf("vision Chat generated %d tokens, want 1", generated)
	}
}

func TestHIPHardwareMoECombineNormsMatchesRMSNormAndVectorAdd_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatal("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	localValues := []float32{3, -4, 2, -1, 5, -6, 7, -8}
	expertValues := []float32{-2, 1, 4, -3, 6, -5, 8, -7}
	localWeights := []uint16{
		hipFloat32ToBFloat16(0.5), hipFloat32ToBFloat16(-0.25), hipFloat32ToBFloat16(0.75), hipFloat32ToBFloat16(0.125),
		hipFloat32ToBFloat16(-0.5), hipFloat32ToBFloat16(0.25), hipFloat32ToBFloat16(1.0), hipFloat32ToBFloat16(-0.125),
	}
	expertWeights := []float32{1, 0.5, -0.5, 2, 0.25, -1, 1.5, 0.75}
	local, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MoECombineNormsHardware", "local input", mustHIPFloat32Payload(t, localValues), len(localValues))
	core.RequireNoError(t, err)
	defer local.Close()
	expert, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MoECombineNormsHardware", "expert input", mustHIPFloat32Payload(t, expertValues), len(expertValues))
	core.RequireNoError(t, err)
	defer expert.Close()
	localWeightPayload, err := hipUint16Payload(localWeights)
	core.RequireNoError(t, err)
	localWeight, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MoECombineNormsHardware", "local weight", localWeightPayload, len(localWeights))
	core.RequireNoError(t, err)
	defer localWeight.Close()
	expertWeight, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MoECombineNormsHardware", "expert weight", mustHIPFloat32Payload(t, expertWeights), len(expertWeights))
	core.RequireNoError(t, err)
	defer expertWeight.Close()

	localCfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  localWeight.Pointer(),
		WeightBytes:    localWeight.SizeBytes(),
		Count:          len(localValues),
		Epsilon:        1e-6,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
		Flags:          hipRMSNormLaunchFlagAddUnitWeight,
	}
	expertCfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  expertWeight.Pointer(),
		WeightBytes:    expertWeight.SizeBytes(),
		Count:          len(expertValues),
		Epsilon:        1e-5,
		WeightEncoding: hipRMSNormWeightEncodingF32,
	}
	baselineLocal, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(context.Background(), hipRuntime.driver, local, localCfg)
	core.RequireNoError(t, err)
	defer baselineLocal.Close()
	baselineExpert, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(context.Background(), hipRuntime.driver, expert, expertCfg)
	core.RequireNoError(t, err)
	defer baselineExpert.Close()
	baseline, err := hipRunVectorAddDeviceKernel(context.Background(), hipRuntime.driver, baselineLocal, baselineExpert)
	core.RequireNoError(t, err)
	defer baseline.Close()
	fused, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.MoECombineNormsHardware", "fused output", uint64(len(localValues)*4), len(localValues))
	core.RequireNoError(t, err)
	defer fused.Close()
	core.RequireNoError(t, hipRunMoECombineNormsDeviceKernelOutput(context.Background(), hipRuntime.driver, local, expert, localCfg, expertCfg, fused))

	want, err := hipReadFloat32DeviceOutput(baseline, "rocm.hip.MoECombineNormsHardware", "baseline output", len(localValues))
	core.RequireNoError(t, err)
	got, err := hipReadFloat32DeviceOutput(fused, "rocm.hip.MoECombineNormsHardware", "fused output", len(localValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNearRelativeNamedForHardwareTest(t, "MoE combine norms", want, got, 0.00001, 0.00001)
}

func TestHIPHardwareAttentionHeadsBatchCapped_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || !runtime.Available() || hipRuntime.driver == nil {
		t.Fatal("native ROCm runtime is not available")
	}

	queryValues := []float32{1, 0, 0, 1}
	keyValues := []float32{1, 0, 0, 1, 1, 1}
	valueValues := []float32{2, 0, 0, 4, 8, 8}
	upload := func(label string, values []float32) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipFloat32Payload(values)
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCappedLaunch", label, payload, len(values))
		core.RequireNoError(t, err)
		t.Cleanup(func() { _ = buffer.Close() })
		return buffer
	}
	query := upload("capped attention query", queryValues)
	keys := upload("capped attention keys", keyValues)
	values := upload("capped attention values", valueValues)
	caps, err := hipUploadTokenIDs(hipRuntime.driver, []int32{2, 2})
	core.RequireNoError(t, err)
	defer caps.Close()
	output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCappedLaunch", "capped attention output", uint64(len(queryValues)*4), len(queryValues))
	core.RequireNoError(t, err)
	defer output.Close()

	core.RequireNoError(t, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, hipAttentionHeadsBatchCausalDeviceRequest{
		Key: keys, Value: values, VisibleTokenCaps: caps,
		Dim: 2, TokenCount: 3, HeadCount: 1, QueryCount: 2, Scale: 1,
	}, query, output))
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchCappedLaunch", "capped attention output", len(queryValues))
	core.RequireNoError(t, err)

	keyRows, err := splitHIPReferenceVectors(keyValues, 2)
	core.RequireNoError(t, err)
	valueRows, err := splitHIPReferenceVectors(valueValues, 2)
	core.RequireNoError(t, err)
	want := make([]float32, 0, len(queryValues))
	for row := range 2 {
		rowOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[row*2:(row+1)*2], keyRows[:2], valueRows[:2], 1)
		core.RequireNoError(t, err)
		want = append(want, rowOutput...)
	}
	assertFloat32SlicesNear(t, want, got, 0.0001)
}

func TestHIPHardwarePackedTopKSamplerMatchesHostReference_Good(t *testing.T) {
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

	const vocabSize = 262144
	logits := make([]float32, vocabSize)
	packed := make([]byte, vocabSize*hipMLXQ4ProjectionBestBytes)
	for token := range logits {
		logits[token] = -100
	}
	for rank := 0; rank < 64; rank++ {
		token := 200000 + rank*17
		logits[token] = 60 - float32(rank)/2
	}
	for token, score := range logits {
		binary.LittleEndian.PutUint64(packed[token*hipMLXQ4ProjectionBestBytes:], hipPackGreedyBest(score, token))
	}
	input, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.PackedTopKSampleOracle", "12B sampler oracle logits", uint64(len(packed)), vocabSize)
	core.RequireNoError(t, err)
	defer input.Close()
	core.RequireNoError(t, hipRuntime.driver.CopyHostToDevice(input.Pointer(), packed))
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	for _, topK := range []int{40, 64} {
		for _, topP := range []float32{1, 0.95} {
			for _, draw := range []float64{0, 0.1, 0.5, 0.9, 0.999999} {
				name := core.Sprintf("topk%d-topp%.2f-draw%.6f", topK, topP, draw)
				t.Run(name, func(t *testing.T) {
					generate := inference.GenerateConfig{Temperature: 1, TopK: topK, TopP: topP, RepeatPenalty: 1}
					hostLogits, err := hipGemma4Q4SoftcapLogits(append([]float32(nil), logits...), 30)
					core.RequireNoError(t, err)
					want, err := hipGemma4Q4HostSampleResult(hostLogits, generate, nil, nil, draw)
					core.RequireNoError(t, err)
					candidates, candidateCount, err := hipRunPackedTopKReduceKernelWithWorkspace(context.Background(), hipRuntime.driver, input, vocabSize, topK, workspace)
					core.RequireNoError(t, err)
					got, _, err := hipRunPackedTopKSampleKernel(context.Background(), hipRuntime.driver, candidates, candidateCount, topK, generate.Temperature, generate.TopP, 30, draw, nil, workspace)
					core.RequireNoError(t, err)
					if got.TokenID != want.TokenID {
						t.Fatalf("device token = %d (score %.6f), host softcap reference = %d (score %.6f)", got.TokenID, got.Score, want.TokenID, want.Score)
					}
				})
			}
		}
	}

	off := inference.GenerateConfig{Temperature: 1, TopK: 0, TopP: 0.95, RepeatPenalty: 1}
	if hipGemma4Q4DeviceTopKSamplingRequested(off) {
		t.Fatal("top-k off unexpectedly routes through the bounded device top-k sampler")
	}
}

func TestHIPHardwareGemma4MoEGGUFHostResidency_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_PRODUCTION_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_PRODUCTION_MODEL_PATH to a local Gemma4 MoE GGUF")
	}
	info, err := gguf.ReadInfo(modelPath)
	core.RequireNoError(t, err)
	if info.Metadata.ExpertCount == 0 {
		t.Skip("GO_ROCM_PRODUCTION_MODEL_PATH is not a sparse Gemma4 GGUF")
	}
	textConfig := nativeGemma4TextConfigFromGGUFMetadata(info.Metadata)
	core.AssertEqual(t, true, textConfig.EnableMoEBlock)
	core.AssertEqual(t, 128, textConfig.NumExperts)
	core.AssertEqual(t, 8, textConfig.TopKExperts)
	core.AssertEqual(t, 704, textConfig.MoEIntermediateSize)
	if os.Getenv("GO_ROCM_LOG_MOE_TENSORS") == "1" {
		for _, tensor := range info.Tensors {
			if strings.Contains(tensor.Name, "_exps.weight") {
				t.Logf("%s type=%s dims=%v bytes=%d", tensor.Name, tensor.TypeName, tensor.Dimensions, tensor.ByteSize)
			}
		}
	}

	textModel, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).loadModelWithROCmConfig(modelPath, inference.LoadConfig{ContextLen: 64}, ROCmLoadConfig{})
	core.RequireNoError(t, err)
	defer textModel.Close()
	rocmLoaded, ok := textModel.(*rocmModel)
	core.RequireTrue(t, ok)
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	loadConfig := nativeLoadConfig{ModelInfo: loaded.modelInfo}

	var deviceBytes uint64
	for _, tensor := range loaded.tensors {
		deviceBytes += tensor.info.ByteSize
		if hipGemma4HostResidentExpertTensor(loadConfig, tensor.info) {
			t.Fatalf("expert tensor %q was allocated in device memory", tensor.info.Name)
		}
	}
	var hostBytes uint64
	for _, tensor := range loaded.hostTensors {
		hostBytes += tensor.ByteSize
	}
	core.AssertEqual(t, int(info.Metadata.BlockCount)*2, len(loaded.hostTensors))
	core.AssertGreater(t, hostBytes, uint64(11*memoryGiB))
	moe, err := loaded.loadedGemma4MoELayerConfig(0, loaded.modelInfo.HiddenSize)
	core.RequireNoError(t, err)
	core.RequireTrue(t, moe != nil)
	core.AssertEqual(t, 128, moe.NumExperts)
	core.AssertEqual(t, 8, moe.TopKExperts)
	core.AssertEqual(t, 704, moe.ExpertIntermediateSize)
	core.AssertEqual(t, 128, moe.RouterProjection.Rows)
	core.AssertEqual(t, 2816, moe.RouterProjection.Cols)
	core.AssertEqual(t, 128, len(moe.PerExpertScale))
	core.RequireTrue(t, moe.ExpertCache != nil)
	forward, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("load complete Gemma4 MoE forward config: %v", err)
	}
	core.AssertEqual(t, loaded.modelInfo.NumLayers, len(forward.Layers))
	for index, layer := range forward.Layers {
		if layer.MoE == nil {
			t.Fatalf("forward layer %d has no MoE config", index)
		}
	}
	inputValues := make([]float32, loaded.modelInfo.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(math.Sin(float64(index+1) * 0.013))
	}
	input, err := hipUploadGemma4Q4Float32Input(loaded.driver, "real Gemma4 MoE layer input", inputValues)
	core.RequireNoError(t, err)
	defer input.Close()
	preCfg := forward.Layers[0].PreFeedForwardNorm
	preCfg.Epsilon = 1e-6
	localInput, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(context.Background(), loaded.driver, input, preCfg)
	core.RequireNoError(t, err)
	defer localInput.Close()
	moeOutput, err := hipRunGemma4MoEDeviceMLP(context.Background(), loaded.driver, input, localInput, forward.Layers[0], 1e-6)
	core.RequireNoError(t, err)
	defer moeOutput.Close()
	outputValues, err := hipReadFloat32DeviceOutput(moeOutput, "rocm.hip.Gemma4MoE", "real MoE layer output", loaded.modelInfo.HiddenSize)
	core.RequireNoError(t, err)
	core.RequireTrue(t, rocmFloat32SliceFinite(outputValues))
	var outputMagnitude float64
	for _, value := range outputValues {
		outputMagnitude += math.Abs(float64(value))
	}
	core.AssertGreater(t, outputMagnitude, 0.0)
	core.AssertEqual(t, 8, len(moe.ExpertCache.entries))
	device := loaded.driver.DeviceInfo()
	t.Logf("Gemma4 MoE residency: device_tensors=%d device_bytes=%d host_tensors=%d host_bytes=%d device_free_bytes=%d expert_cache_bytes=%d", len(loaded.tensors), deviceBytes, len(loaded.hostTensors), hostBytes, device.FreeBytes, moe.ExpertCache.maxBytes)
}

func TestHIPHardwareGGUFQ4_0ProjectionAndGateUp_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatal("native ROCm runtime is not available")
	}
	const cols = 32
	input := make([]float32, cols)
	for index := range input {
		input[index] = float32(index%7-3) / 8
	}
	weightRows := [][]byte{
		hipHardwareGGUFQ4_0Row(0.25, 1),
		hipHardwareGGUFQ4_0Row(-0.125, 3),
		hipHardwareGGUFQ4_0Row(0.0625, 5),
		hipHardwareGGUFQ4_0Row(-0.03125, 7),
	}
	weights := make([]byte, 0, len(weightRows)*hipGGUFQ4_0BlockBytes)
	for _, row := range weightRows {
		weights = append(weights, row...)
	}
	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0ProjectionLaunch", "hardware input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	weightBuffer, err := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0ProjectionLaunch", "hardware weights", weights, len(weights))
	core.RequireNoError(t, err)
	defer weightBuffer.Close()

	projectionOutput, err := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0ProjectionLaunch", "hardware projection output", 2*4, 2)
	core.RequireNoError(t, err)
	defer projectionOutput.Close()
	core.RequireNoError(t, hipRunGGUFQ4_0ProjectionKernelWithDeviceInputOutput(context.Background(), runtime.(*hipRuntime).driver, inputBuffer, weightBuffer, 2, cols, 1, len(weightRows), projectionOutput))
	projection, err := hipReadFloat32DeviceOutput(projectionOutput, "rocm.hip.GGUFQ4_0ProjectionLaunch", "hardware projection output", 2)
	core.RequireNoError(t, err)
	for row := range projection {
		assertFloat32Near(t, hipHardwareGGUFQ4_0Dot(input, weightRows[row+1]), projection[row])
	}

	gateUpOutput, err := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0ProjectionLaunch", "hardware gate-up output", 2*4, 2)
	core.RequireNoError(t, err)
	defer gateUpOutput.Close()
	core.RequireNoError(t, hipRunGGUFQ4_0GELUTanhGateUpKernelWithDeviceInputOutput(context.Background(), runtime.(*hipRuntime).driver, inputBuffer, weightBuffer, 2, cols, 0, len(weightRows), gateUpOutput))
	gateUp, err := hipReadFloat32DeviceOutput(gateUpOutput, "rocm.hip.GGUFQ4_0ProjectionLaunch", "hardware gate-up output", 2)
	core.RequireNoError(t, err)
	wantGate := []float32{hipHardwareGGUFQ4_0Dot(input, weightRows[0]), hipHardwareGGUFQ4_0Dot(input, weightRows[1])}
	wantUp := []float32{hipHardwareGGUFQ4_0Dot(input, weightRows[2]), hipHardwareGGUFQ4_0Dot(input, weightRows[3])}
	wantGateUp := expectedGELUTanhMultiply(wantGate, wantUp)
	assertFloat32SlicesNear(t, wantGateUp, gateUp, 1e-5)

	const selectedExperts = 2
	expertGateUpRows := make([][][]byte, selectedExperts)
	expertDownRows := make([][][]byte, selectedExperts)
	entries := make([]*hipGemma4ExpertCacheEntry, selectedExperts)
	for expert := 0; expert < selectedExperts; expert++ {
		expertGateUpRows[expert] = make([][]byte, 2*cols)
		gateUpPayload := make([]byte, 0, 2*cols*hipGGUFQ4_0BlockBytes)
		for row := range expertGateUpRows[expert] {
			expertGateUpRows[expert][row] = hipHardwareGGUFQ4_0Row(0.015625*float32(expert+1), expert*7+row+1)
			gateUpPayload = append(gateUpPayload, expertGateUpRows[expert][row]...)
		}
		expertDownRows[expert] = make([][]byte, cols)
		downPayload := make([]byte, 0, cols*hipGGUFQ4_0BlockBytes)
		for row := range expertDownRows[expert] {
			expertDownRows[expert][row] = hipHardwareGGUFQ4_0Row(0.03125*float32(expert+1), expert*11+row+3)
			downPayload = append(downPayload, expertDownRows[expert][row]...)
		}
		gateUpBuffer, err := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "hardware selected gate/up", gateUpPayload, len(gateUpPayload))
		core.RequireNoError(t, err)
		defer gateUpBuffer.Close()
		downBuffer, err := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "hardware selected down", downPayload, len(downPayload))
		core.RequireNoError(t, err)
		defer downBuffer.Close()
		entries[expert] = &hipGemma4ExpertCacheEntry{
			GateUp: gateUpBuffer, Down: downBuffer,
			GateUpRows: 2 * cols, GateUpCols: cols, DownRows: cols, DownCols: cols,
		}
	}
	selectedActivation, err := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "hardware selected activation", selectedExperts*cols*4, selectedExperts*cols)
	core.RequireNoError(t, err)
	defer selectedActivation.Close()
	selectedOutput, err := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "hardware selected output", cols*4, cols)
	core.RequireNoError(t, err)
	defer selectedOutput.Close()
	routeWeights := []float32{0.75, 0.25}
	runSelected := func(pair16 string) []float32 {
		t.Helper()
		t.Setenv(hipGemma4SelectedExpertPair16Env, pair16)
		core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), runtime.(*hipRuntime).driver, inputBuffer, entries, routeWeights, cols, cols, selectedActivation, selectedOutput))
		selected, readErr := hipReadFloat32DeviceOutput(selectedOutput, "rocm.hip.GGUFQ4_0SelectedExpertsLaunch", "hardware selected output", cols)
		core.RequireNoError(t, readErr)
		return selected
	}
	selectedBaseline := runSelected("0")
	selectedPair16 := runSelected("1")
	for index := range selectedBaseline {
		if math.Float32bits(selectedBaseline[index]) != math.Float32bits(selectedPair16[index]) {
			t.Fatalf("pair16 selected output[%d] = %08x, baseline = %08x", index, math.Float32bits(selectedPair16[index]), math.Float32bits(selectedBaseline[index]))
		}
	}
	wantSelected := make([]float32, cols)
	for expert := 0; expert < selectedExperts; expert++ {
		activationValues := make([]float32, cols)
		for row := 0; row < cols; row++ {
			gate := hipHardwareGGUFQ4_0Dot(input, expertGateUpRows[expert][row])
			up := hipHardwareGGUFQ4_0Dot(input, expertGateUpRows[expert][cols+row])
			activationValues[row] = expectedGELUTanhMultiply([]float32{gate}, []float32{up})[0]
		}
		for row := 0; row < cols; row++ {
			wantSelected[row] += routeWeights[expert] * hipHardwareGGUFQ4_0Dot(activationValues, expertDownRows[expert][row])
		}
	}
	assertFloat32SlicesNear(t, wantSelected, selectedPair16, 1e-4)
}

func TestHIPHardwareGGUFMixedSelectedExperts_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatal("native ROCm runtime is not available")
	}
	const (
		hidden          = 256
		expertFF        = 32
		selectedExperts = 8
	)
	input := make([]float32, hidden)
	for index := range input {
		input[index] = float32(index%17-8) / 32
	}
	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()

	expertGateUpRows := make([][][]byte, selectedExperts)
	expertDownRows := make([][][]byte, selectedExperts)
	entries := make([]*hipGemma4ExpertCacheEntry, selectedExperts)
	rawGateUps := make([]*hipDeviceByteBuffer, selectedExperts)
	expandedGateUps := make([]*hipDeviceByteBuffer, selectedExperts)
	for expert := 0; expert < selectedExperts; expert++ {
		expertGateUpRows[expert] = make([][]byte, 2*expertFF)
		gateUpPayload := make([]byte, 0, 2*expertFF*hipGGUFQ4KBlockBytes)
		for row := range expertGateUpRows[expert] {
			expertGateUpRows[expert][row] = hipHardwareGGUFQ4KRow(0.00390625*float32(expert+1), 0.001953125, expert*19+row+1)
			gateUpPayload = append(gateUpPayload, expertGateUpRows[expert][row]...)
		}
		expertDownRows[expert] = make([][]byte, hidden)
		downPayload := make([]byte, 0, hidden*hipGGUFQ5_1BlockBytes)
		for row := range expertDownRows[expert] {
			expertDownRows[expert][row] = hipHardwareGGUFQ5_1Row(0.0078125*float32(expert+1), -0.0625, expert*23+row+3)
			downPayload = append(downPayload, expertDownRows[expert][row]...)
		}
		gateUpBuffer, uploadErr := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed gate/up", gateUpPayload, len(gateUpPayload))
		core.RequireNoError(t, uploadErr)
		defer gateUpBuffer.Close()
		expandedGateUpPayload := hipHardwareExpandGGUFQ4KMetadata(t, gateUpPayload)
		expandedGateUpBuffer, allocateErr := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware expanded mixed gate/up", uint64(len(expandedGateUpPayload)), len(expandedGateUpPayload))
		core.RequireNoError(t, allocateErr)
		defer expandedGateUpBuffer.Close()
		core.RequireNoError(t, hipRunGGUFQ4KExpandMetadataKernel(context.Background(), runtime.(*hipRuntime).driver, gateUpBuffer, uint64(len(gateUpPayload)), expandedGateUpBuffer, len(gateUpPayload)/hipGGUFQ4KBlockBytes))
		expandedGateUpGot := make([]byte, len(expandedGateUpPayload))
		core.RequireNoError(t, runtime.(*hipRuntime).driver.CopyDeviceToHost(expandedGateUpBuffer.Pointer(), expandedGateUpGot))
		core.AssertEqual(t, expandedGateUpPayload, expandedGateUpGot)
		downBuffer, uploadErr := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed down", downPayload, len(downPayload))
		core.RequireNoError(t, uploadErr)
		defer downBuffer.Close()
		rawGateUps[expert] = gateUpBuffer
		expandedGateUps[expert] = expandedGateUpBuffer
		entries[expert] = &hipGemma4ExpertCacheEntry{
			GateUp: gateUpBuffer, Down: downBuffer,
			GateUpRows: 2 * expertFF, GateUpCols: hidden, DownRows: hidden, DownCols: expertFF,
			GateUpFormat: hipGGUFExpertFormatQ4K, DownFormat: hipGGUFExpertFormatQ5_1,
		}
	}
	activation, err := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed activation", selectedExperts*expertFF*4, selectedExperts*expertFF)
	core.RequireNoError(t, err)
	defer activation.Close()
	output, err := hipAllocateByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed output", hidden*4, hidden)
	core.RequireNoError(t, err)
	defer output.Close()
	routeWeights := []float32{0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04}
	runMixed := func(pair16, expert8, gateUpSplit string) []float32 {
		t.Helper()
		t.Setenv(hipGemma4SelectedExpertPair16Env, pair16)
		t.Setenv(hipGemma4SelectedExpertDownExpert8Env, expert8)
		t.Setenv(hipGemma4SelectedExpertGateUpSplitEnv, gateUpSplit)
		core.RequireNoError(t, hipRunGGUFQ4_0SelectedExpertsKernelWithDeviceInputOutput(context.Background(), runtime.(*hipRuntime).driver, inputBuffer, entries, routeWeights, hidden, expertFF, activation, output))
		got, readErr := hipReadFloat32DeviceOutput(output, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed output", hidden)
		core.RequireNoError(t, readErr)
		return got
	}
	useGateUps := func(buffers []*hipDeviceByteBuffer, format uint32) {
		t.Helper()
		for expert := range entries {
			entries[expert].GateUp = buffers[expert]
			entries[expert].GateUpFormat = format
		}
	}
	gotBaseline := runMixed("0", "0", "0")
	gotPair16 := runMixed("1", "0", "0")
	gotExpert8 := runMixed("1", "1", "0")
	got := runMixed("1", "1", "1")
	useGateUps(expandedGateUps, hipGGUFExpertFormatQ4KExpanded)
	gotExpanded := runMixed("1", "1", "1")
	for index := range gotBaseline {
		if math.Float32bits(gotBaseline[index]) != math.Float32bits(gotPair16[index]) {
			t.Fatalf("mixed Q5_1 pair16 output[%d] = %08x, baseline = %08x", index, math.Float32bits(gotPair16[index]), math.Float32bits(gotBaseline[index]))
		}
		if math.Float32bits(gotBaseline[index]) != math.Float32bits(gotExpert8[index]) {
			t.Fatalf("mixed Q5_1 expert8 output[%d] = %08x, baseline = %08x", index, math.Float32bits(gotExpert8[index]), math.Float32bits(gotBaseline[index]))
		}
		if math.Float32bits(gotBaseline[index]) != math.Float32bits(got[index]) {
			t.Fatalf("mixed Q4_K gate/up split output[%d] = %08x, baseline = %08x", index, math.Float32bits(got[index]), math.Float32bits(gotBaseline[index]))
		}
		if math.Float32bits(gotBaseline[index]) != math.Float32bits(gotExpanded[index]) {
			t.Fatalf("expanded mixed Q4_K gate/up output[%d] = %08x, raw baseline = %08x", index, math.Float32bits(gotExpanded[index]), math.Float32bits(gotBaseline[index]))
		}
	}

	want := make([]float32, hidden)
	for expert := 0; expert < selectedExperts; expert++ {
		activationValues := make([]float32, expertFF)
		for row := 0; row < expertFF; row++ {
			gate := hipHardwareGGUFQ4KDot(input, expertGateUpRows[expert][row])
			up := hipHardwareGGUFQ4KDot(input, expertGateUpRows[expert][expertFF+row])
			activationValues[row] = expectedGELUTanhMultiply([]float32{gate}, []float32{up})[0]
		}
		for row := 0; row < hidden; row++ {
			want[row] += routeWeights[expert] * hipHardwareGGUFQ5_1Dot(activationValues, expertDownRows[expert][row])
		}
	}
	assertFloat32SlicesNear(t, want, gotExpanded, 2e-3)

	q8DownRows := make([][][]byte, selectedExperts)
	for expert := 0; expert < selectedExperts; expert++ {
		q8DownRows[expert] = make([][]byte, hidden)
		downPayload := make([]byte, 0, hidden*hipGGUFQ8_0BlockBytes)
		for row := range q8DownRows[expert] {
			q8DownRows[expert][row] = hipHardwareGGUFQ8_0Row(0.00390625*float32(expert+1), expert*29+row+5)
			downPayload = append(downPayload, q8DownRows[expert][row]...)
		}
		downBuffer, uploadErr := hipUploadByteBuffer(runtime.(*hipRuntime).driver, "rocm.hip.GGUFMixedSelectedExpertsLaunch", "hardware mixed Q8_0 down", downPayload, len(downPayload))
		core.RequireNoError(t, uploadErr)
		defer downBuffer.Close()
		entries[expert].Down = downBuffer
		entries[expert].DownFormat = hipGGUFExpertFormatQ8_0
	}
	useGateUps(rawGateUps, hipGGUFExpertFormatQ4K)
	gotQ8Baseline := runMixed("0", "0", "0")
	gotQ8Pair16 := runMixed("1", "0", "0")
	gotQ8 := runMixed("1", "0", "1")
	useGateUps(expandedGateUps, hipGGUFExpertFormatQ4KExpanded)
	gotQ8Expanded := runMixed("1", "0", "1")
	for index := range gotQ8Baseline {
		if math.Float32bits(gotQ8Baseline[index]) != math.Float32bits(gotQ8Pair16[index]) {
			t.Fatalf("mixed Q8_0 pair16 output[%d] = %08x, baseline = %08x", index, math.Float32bits(gotQ8Pair16[index]), math.Float32bits(gotQ8Baseline[index]))
		}
		if math.Float32bits(gotQ8Baseline[index]) != math.Float32bits(gotQ8[index]) {
			t.Fatalf("mixed Q8_0 gate/up split output[%d] = %08x, baseline = %08x", index, math.Float32bits(gotQ8[index]), math.Float32bits(gotQ8Baseline[index]))
		}
		if math.Float32bits(gotQ8Baseline[index]) != math.Float32bits(gotQ8Expanded[index]) {
			t.Fatalf("expanded mixed Q4_K/Q8_0 output[%d] = %08x, raw baseline = %08x", index, math.Float32bits(gotQ8Expanded[index]), math.Float32bits(gotQ8Baseline[index]))
		}
	}
	wantQ8 := make([]float32, hidden)
	for expert := 0; expert < selectedExperts; expert++ {
		activationValues := make([]float32, expertFF)
		for row := 0; row < expertFF; row++ {
			gate := hipHardwareGGUFQ4KDot(input, expertGateUpRows[expert][row])
			up := hipHardwareGGUFQ4KDot(input, expertGateUpRows[expert][expertFF+row])
			activationValues[row] = expectedGELUTanhMultiply([]float32{gate}, []float32{up})[0]
		}
		for row := 0; row < hidden; row++ {
			wantQ8[row] += routeWeights[expert] * hipHardwareGGUFQ8_0Dot(activationValues, q8DownRows[expert][row])
		}
	}
	assertFloat32SlicesNear(t, wantQ8, gotQ8Expanded, 2e-3)
}

func TestHIPHardwareMoERouterParallel_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatal("native ROCm runtime is not available")
	}
	logits := make([]float32, 128)
	for expert := range logits {
		logits[expert] = float32((expert*37)%23) / 8
	}
	req := hipMoERouterRequest{Logits: logits, TopK: 8, Layer: 11}
	want, err := rocmReferenceRouteExperts(req.Logits, req.TopK, req.Layer, nil)
	core.RequireNoError(t, err)
	got, err := hipRunMoERouterKernel(context.Background(), runtime.(*hipRuntime).driver, req)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipMoERouterLaunchStatusOK, got.Status)
	core.AssertEqual(t, len(want), len(got.Routes))
	for index := range want {
		core.AssertEqual(t, want[index].ID, got.Routes[index].ID)
		assertFloat32Near(t, want[index].Prob, got.Routes[index].Prob)
	}
}

func hipHardwareGGUFQ4_0Row(scale float32, salt int) []byte {
	row := make([]byte, hipGGUFQ4_0BlockBytes)
	binary.LittleEndian.PutUint16(row, rocmFloat32ToFloat16(scale))
	for lane := 0; lane < 16; lane++ {
		low := byte((lane + salt) & 0x0f)
		high := byte((31 - lane + salt) & 0x0f)
		row[2+lane] = low | high<<4
	}
	return row
}

func hipHardwareGGUFQ4_0Dot(input []float32, row []byte) float32 {
	scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(row))
	var sum float32
	for lane, value := range input {
		packed := row[2+lane%16]
		quant := packed & 0x0f
		if lane >= 16 {
			quant = packed >> 4
		}
		sum += value * scale * (float32(quant) - 8)
	}
	return sum
}

func hipHardwareGGUFQ4KRow(scale, minimumScale float32, salt int) []byte {
	row := make([]byte, hipGGUFQ4KBlockBytes)
	binary.LittleEndian.PutUint16(row[0:], rocmFloat32ToFloat16(scale))
	binary.LittleEndian.PutUint16(row[2:], rocmFloat32ToFloat16(minimumScale))
	for group := 0; group < hipGGUFQ4KGroupsPerBlock; group++ {
		groupScale := byte(1 + (salt+group)%7)
		groupMinimum := byte((salt + group*2) % 4)
		if group < 4 {
			row[4+group] = groupScale
			row[8+group] = groupMinimum
			continue
		}
		row[8+group] = groupScale | groupMinimum<<4
	}
	for group := 0; group < hipGGUFQ4KGroupsPerBlock; group++ {
		for lane := 0; lane < 32; lane++ {
			quant := byte((salt + group*5 + lane*3) & 0x0f)
			index := 16 + (group/2)*32 + lane
			if group&1 == 0 {
				row[index] = quant
			} else {
				row[index] |= quant << 4
			}
		}
	}
	return row
}

func hipHardwareExpandGGUFQ4KMetadata(t *testing.T, payload []byte) []byte {
	t.Helper()
	if len(payload) == 0 || len(payload)%hipGGUFQ4KBlockBytes != 0 {
		t.Fatalf("Q4_K payload bytes = %d, want complete non-empty blocks", len(payload))
	}
	blockCount := len(payload) / hipGGUFQ4KBlockBytes
	expanded := make([]byte, blockCount*hipGGUFQ4KExpandedBlockBytes)
	for blockIndex := 0; blockIndex < blockCount; blockIndex++ {
		rawBlock := payload[blockIndex*hipGGUFQ4KBlockBytes : (blockIndex+1)*hipGGUFQ4KBlockBytes]
		expandedBlock := expanded[blockIndex*hipGGUFQ4KExpandedBlockBytes : (blockIndex+1)*hipGGUFQ4KExpandedBlockBytes]
		binary.LittleEndian.PutUint32(expandedBlock[0:], math.Float32bits(hipFloat16ToFloat32(binary.LittleEndian.Uint16(rawBlock[0:]))))
		binary.LittleEndian.PutUint32(expandedBlock[4:], math.Float32bits(hipFloat16ToFloat32(binary.LittleEndian.Uint16(rawBlock[2:]))))
		for group := 0; group < hipGGUFQ4KGroupsPerBlock; group++ {
			scale, minimum := hipGGUFQ4KScaleMin(rawBlock[4:16], group)
			expandedBlock[8+group] = scale
			expandedBlock[16+group] = minimum
		}
		copy(expandedBlock[24:], rawBlock[16:])
	}
	return expanded
}

func hipHardwareGGUFQ4KDot(input []float32, row []byte) float32 {
	scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(row[0:]))
	minimumScale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(row[2:]))
	var sum float32
	for lane, inputValue := range input {
		group := lane / 32
		groupScale, groupMinimum := hipGGUFQ4KScaleMin(row[4:16], group)
		packed := row[16+(group/2)*32+lane%32]
		quant := packed & 0x0f
		if group&1 != 0 {
			quant = packed >> 4
		}
		value := scale*float32(groupScale)*float32(quant) - minimumScale*float32(groupMinimum)
		sum += inputValue * value
	}
	return sum
}

func hipHardwareGGUFQ5_1Row(scale, minimum float32, salt int) []byte {
	row := make([]byte, hipGGUFQ5_1BlockBytes)
	binary.LittleEndian.PutUint16(row[0:], rocmFloat32ToFloat16(scale))
	binary.LittleEndian.PutUint16(row[2:], rocmFloat32ToFloat16(minimum))
	var highBits uint32
	for lane := 0; lane < hipGGUFQ5_1BlockSize; lane++ {
		quant := byte((salt + lane*7) & 0x1f)
		if quant&0x10 != 0 {
			highBits |= 1 << lane
		}
		index := 8 + lane%16
		if lane < 16 {
			row[index] = quant & 0x0f
		} else {
			row[index] |= (quant & 0x0f) << 4
		}
	}
	binary.LittleEndian.PutUint32(row[4:], highBits)
	return row
}

func hipHardwareGGUFQ5_1Dot(input []float32, row []byte) float32 {
	scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(row[0:]))
	minimum := hipFloat16ToFloat32(binary.LittleEndian.Uint16(row[2:]))
	highBits := binary.LittleEndian.Uint32(row[4:])
	var sum float32
	for lane, inputValue := range input {
		packed := row[8+lane%16]
		quant := packed & 0x0f
		if lane >= 16 {
			quant = packed >> 4
		}
		quant |= byte((highBits>>lane)&1) << 4
		sum += inputValue * (scale*float32(quant) + minimum)
	}
	return sum
}

func hipHardwareGGUFQ8_0Row(scale float32, salt int) []byte {
	row := make([]byte, hipGGUFQ8_0BlockBytes)
	binary.LittleEndian.PutUint16(row[0:], rocmFloat32ToFloat16(scale))
	for lane := 0; lane < hipGGUFQ8_0BlockSize; lane++ {
		row[2+lane] = byte(int8((salt+lane*11)%255 - 127))
	}
	return row
}

func hipHardwareGGUFQ8_0Dot(input []float32, row []byte) float32 {
	scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(row[0:]))
	var sum float32
	for lane, inputValue := range input {
		sum += inputValue * scale * float32(int8(row[2+lane]))
	}
	return sum
}

// TestHIPHardwareAudioQ4ProjectorGolden_Good runs embed_audio's packed q4 weights through the real HIP
// projection kernel and compares both soft-token rows with the engine-neutral host golden.
func TestHIPHardwareAudioQ4ProjectorGolden_Good(t *testing.T) {
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
	projector := audioQ4GoldenProjector()
	weights := []uint32{
		binary.LittleEndian.Uint32(projector.Weight[0:4]),
		binary.LittleEndian.Uint32(projector.Weight[4:8]),
		binary.LittleEndian.Uint32(projector.Weight[8:12]),
	}
	scales := []uint16{
		binary.LittleEndian.Uint16(projector.Scales[0:2]),
		binary.LittleEndian.Uint16(projector.Scales[2:4]),
		binary.LittleEndian.Uint16(projector.Scales[4:6]),
	}
	biases := []uint16{
		binary.LittleEndian.Uint16(projector.Biases[0:2]),
		binary.LittleEndian.Uint16(projector.Biases[2:4]),
		binary.LittleEndian.Uint16(projector.Biases[4:6]),
	}
	features := []float32{1, -2, 3, -4, 5, -6, 7, -8, -1, 2, -3, 4, -5, 6, -7, 8}
	want := []float32{-1.584236, 3.5645308, -1.5594823, 1.584236, -3.5645308, 1.5594823}
	for row := range 2 {
		input := append([]float32(nil), features[row*projector.InDim:(row+1)*projector.InDim]...)
		var squares float32
		for _, value := range input {
			squares += value * value
		}
		invRMS := float32(1 / math.Sqrt(float64(squares/float32(projector.InDim)+1e-6)))
		for col := range input {
			input[col] *= invRMS
		}
		got, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, hipMLXQ4ProjectionRequest{
			Input: input, Weight: weights, Scales: scales, Biases: biases,
			Rows: projector.OutDim, Cols: projector.InDim, GroupSize: projector.GroupSize, Bits: projector.Bits,
		})
		if err != nil {
			t.Fatalf("embed_audio row %d: %v", row, err)
		}
		for output := range projector.OutDim {
			expected := want[row*projector.OutDim+output]
			if delta := math.Abs(float64(got[output] - expected)); delta > 1e-5 {
				t.Fatalf("row=%d output=%d GPU=%.9f golden=%.9f delta=%.9g", row, output, got[output], expected, delta)
			}
		}
		t.Logf("embed_audio row=%d GPU=%v golden=%v", row, got, want[row*projector.OutDim:(row+1)*projector.OutDim])
	}
}

func TestHIPHardwareAttentionHeadsDeviceKVGQA_Good(t *testing.T) {
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

	const (
		dim        = 2
		tokenCount = 3
		headCount  = 4
		keyHeads   = 2
	)
	queryValues := []float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 0.5,
	}
	keyValues := []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
		1, 1, -1, 1,
	}
	valueValues := []float32{
		2, 0, 0, 4,
		0, 6, 8, 0,
		4, 4, -2, 2,
	}
	for _, mode := range []string{rocmKVCacheModeFP16, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4} {
		t.Run(mode, func(t *testing.T) {
			queryPayload, err := hipFloat32Payload(queryValues)
			core.RequireNoError(t, err)
			query, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware GQA attention query", queryPayload, len(queryValues))
			core.RequireNoError(t, err)
			defer query.Close()
			output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware GQA attention output", uint64(len(queryValues)*4), len(queryValues))
			core.RequireNoError(t, err)
			defer output.Close()

			cache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, cache.AppendVectors(0, keyHeads*dim, keyHeads*dim, keyValues, valueValues))
			deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
			core.RequireNoError(t, err)
			defer deviceKV.Close()
			table, err := deviceKV.KernelDescriptorTable()
			core.RequireNoError(t, err)
			defer table.Close()

			core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, hipAttentionRequest{
				QueryDim:        dim,
				KeyHeads:        keyHeads,
				DeviceKV:        deviceKV,
				DescriptorTable: table,
				Scale:           1,
			}, query, headCount, output))
			got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsLaunch", "hardware GQA attention output", len(queryValues))
			core.RequireNoError(t, err)
			restoredKeys, restoredValues, err := cache.Restore(0, cache.TokenCount())
			core.RequireNoError(t, err)
			want := make([]float32, 0, len(queryValues))
			for head := 0; head < headCount; head++ {
				keys, err := fakeROCmAttentionHeadVectors(restoredKeys, tokenCount, keyHeads, dim, headCount, head)
				core.RequireNoError(t, err)
				values, err := fakeROCmAttentionHeadVectors(restoredValues, tokenCount, keyHeads, dim, headCount, head)
				core.RequireNoError(t, err)
				queryBase := head * dim
				headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryBase:queryBase+dim], keys, values, 1)
				core.RequireNoError(t, err)
				want = append(want, headOutput...)
			}
			assertFloat32SlicesNear(t, want, got, 0.005)
		})
	}
}

func TestHIPHardwareAttentionHeadsLaneBatchIndependentKV_Good(t *testing.T) {
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

	const (
		dim       = 2
		headCount = 2
	)
	laneKeys := [][]float32{
		{1, 0, 0, 1},
		{1, 1, 1, 0, 0, 1},
	}
	laneValues := [][]float32{
		{2, 0, 0, 4},
		{8, 8, 6, 0, 0, 10},
	}
	queryValues := []float32{1, 0, 0, 1, 1, 1, -1, 1}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaneBatchLaunch", "hardware attention lane batch query", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaneBatchLaunch", "hardware attention lane batch output", uint64(len(queryValues)*4), len(queryValues))
	core.RequireNoError(t, err)
	defer output.Close()

	lanes := make([]hipAttentionHeadsLaneBatchLane, 0, len(laneKeys))
	deviceCaches := make([]*rocmDeviceKVCache, 0, len(laneKeys))
	tables := make([]*rocmDeviceKVDescriptorTable, 0, len(laneKeys))
	defer func() {
		for _, table := range tables {
			_ = table.Close()
		}
		for _, cache := range deviceCaches {
			_ = cache.Close()
		}
	}()
	for lane := range laneKeys {
		cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
		core.RequireNoError(t, err)
		core.RequireNoError(t, cache.AppendVectors(0, dim, dim, laneKeys[lane], laneValues[lane]))
		deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
		core.RequireNoError(t, err)
		deviceCaches = append(deviceCaches, deviceKV)
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		tables = append(tables, table)
		window := 0
		if lane == 0 {
			window = 1
		}
		lanes = append(lanes, hipAttentionHeadsLaneBatchLane{DeviceKV: deviceKV, DescriptorTable: table, WindowSize: window})
	}

	core.RequireNoError(t, hipRunAttentionHeadsLaneBatchOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, hipAttentionHeadsLaneBatchDeviceRequest{
		Lanes:     lanes,
		Dim:       dim,
		HeadCount: headCount,
		Scale:     1,
	}, query, output))
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsLaneBatchLaunch", "hardware attention lane batch output", len(queryValues))
	core.RequireNoError(t, err)

	want := make([]float32, 0, len(queryValues))
	for lane := range laneKeys {
		keys, err := splitHIPReferenceVectors(laneKeys[lane], dim)
		core.RequireNoError(t, err)
		values, err := splitHIPReferenceVectors(laneValues[lane], dim)
		core.RequireNoError(t, err)
		if lanes[lane].WindowSize > 0 && len(keys) > lanes[lane].WindowSize {
			keys = keys[len(keys)-lanes[lane].WindowSize:]
			values = values[len(values)-lanes[lane].WindowSize:]
		}
		for head := 0; head < headCount; head++ {
			queryBase := (lane*headCount + head) * dim
			headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryBase:queryBase+dim], keys, values, 1)
			core.RequireNoError(t, err)
			want = append(want, headOutput...)
		}
	}
	assertFloat32SlicesNear(t, want, got, 0.005)
}

func TestHIPHardwareAttentionHeadsLaneBatchMultiKVHead_Good(t *testing.T) {
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

	const (
		dim        = 2
		headCount  = 4
		keyHeads   = 2
		keyWidth   = dim * keyHeads
		tokenCount = 3
	)
	laneKeys := [][]float32{
		{1, 0, 0, 1, 0, 1, 1, 0, 1, 1, -1, 1},
		{1, 1, 1, -1, 2, 0, 0, 2, 0, 1, 1, 0},
	}
	laneValues := [][]float32{
		{2, 0, 0, 4, 4, 0, 0, 8, 6, 2, 3, 9},
		{8, 8, 6, 0, 2, 6, 4, 4, 0, 10, 12, 2},
	}
	queryValues := []float32{
		1, 0, 0, 1, 1, 1, -1, 1,
		1, 1, -1, 1, 0, 1, 1, 0,
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaneBatchLaunch", "hardware multi-head lane batch query", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaneBatchLaunch", "hardware multi-head lane batch output", uint64(len(queryValues)*4), len(queryValues))
	core.RequireNoError(t, err)
	defer output.Close()

	lanes := make([]hipAttentionHeadsLaneBatchLane, 0, len(laneKeys))
	deviceCaches := make([]*rocmDeviceKVCache, 0, len(laneKeys))
	tables := make([]*rocmDeviceKVDescriptorTable, 0, len(laneKeys))
	defer func() {
		for _, table := range tables {
			_ = table.Close()
		}
		for _, cache := range deviceCaches {
			_ = cache.Close()
		}
	}()
	for lane := range laneKeys {
		cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
		core.RequireNoError(t, err)
		core.RequireNoError(t, cache.AppendVectors(0, keyWidth, keyWidth, laneKeys[lane], laneValues[lane]))
		deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
		core.RequireNoError(t, err)
		deviceCaches = append(deviceCaches, deviceKV)
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		tables = append(tables, table)
		lanes = append(lanes, hipAttentionHeadsLaneBatchLane{DeviceKV: deviceKV, DescriptorTable: table})
	}

	core.RequireNoError(t, hipRunAttentionHeadsLaneBatchOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, hipAttentionHeadsLaneBatchDeviceRequest{
		Lanes: lanes, Dim: dim, HeadCount: headCount, KeyHeads: keyHeads, Scale: 1,
	}, query, output))
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsLaneBatchLaunch", "hardware multi-head lane batch output", len(queryValues))
	core.RequireNoError(t, err)

	want := make([]float32, 0, len(queryValues))
	for lane := range laneKeys {
		for head := 0; head < headCount; head++ {
			keys, err := fakeROCmAttentionHeadVectors(laneKeys[lane], tokenCount, keyHeads, dim, headCount, head)
			core.RequireNoError(t, err)
			values, err := fakeROCmAttentionHeadVectors(laneValues[lane], tokenCount, keyHeads, dim, headCount, head)
			core.RequireNoError(t, err)
			queryBase := (lane*headCount + head) * dim
			headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryBase:queryBase+dim], keys, values, 1)
			core.RequireNoError(t, err)
			want = append(want, headOutput...)
		}
	}
	assertFloat32SlicesNear(t, want, got, 0.005)
}

func TestHIPHardwareRMSNormRoPEHeadsPairMatchesSeparateKernels_Good(t *testing.T) {
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

	queryValues := []float32{1, 2, 3, 4, 4, 3, 2, 1}
	keyValues := []float32{2, -1, 0.5, 3}
	queryRawWeights := []float32{0.25, -0.5, 1, -0.25}
	queryBF16Weights := make([]uint16, len(queryRawWeights))
	for index, weight := range queryRawWeights {
		queryBF16Weights[index] = hipFloat32ToBFloat16(weight)
	}
	keyWeights := []float32{0.5, 1.5, 0.75, 1.25}

	query, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "query input", mustHIPFloat32Payload(t, queryValues), len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	key, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "key input", mustHIPFloat32Payload(t, keyValues), len(keyValues))
	core.RequireNoError(t, err)
	defer key.Close()
	queryWeightPayload, err := hipUint16Payload(queryBF16Weights)
	core.RequireNoError(t, err)
	queryWeight, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "query BF16 weight", queryWeightPayload, len(queryBF16Weights))
	core.RequireNoError(t, err)
	defer queryWeight.Close()
	keyWeight, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "key F32 weight", mustHIPFloat32Payload(t, keyWeights), len(keyWeights))
	core.RequireNoError(t, err)
	defer keyWeight.Close()

	queryCfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  queryWeight.Pointer(),
		WeightBytes:    queryWeight.SizeBytes(),
		Count:          4,
		Epsilon:        1e-6,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
		Flags:          hipRMSNormLaunchFlagAddUnitWeight | hipRMSNormLaunchFlagRoPENeoX,
	}
	keyCfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  keyWeight.Pointer(),
		WeightBytes:    keyWeight.SizeBytes(),
		Count:          4,
		Epsilon:        1e-5,
		WeightEncoding: hipRMSNormWeightEncodingF32,
		Flags:          hipRMSNormLaunchFlagRoPENeoX,
	}
	pairedQuery, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "paired query output", query.SizeBytes(), query.Count())
	core.RequireNoError(t, err)
	defer pairedQuery.Close()
	pairedKey, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "paired key output", key.SizeBytes(), key.Count())
	core.RequireNoError(t, err)
	defer pairedKey.Close()
	separateQuery, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "separate query output", query.SizeBytes(), query.Count())
	core.RequireNoError(t, err)
	defer separateQuery.Close()
	separateKey, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairHardware", "separate key output", key.SizeBytes(), key.Count())
	core.RequireNoError(t, err)
	defer separateKey.Close()

	core.RequireNoError(t, hipRunRMSNormRoPEHeadsPairKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(context.Background(), hipRuntime.driver, query, key, queryCfg, keyCfg, 2, 1, 7, 10000, 4, 2, 0.5, pairedQuery, pairedKey, nil))
	core.RequireNoError(t, hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScale(context.Background(), hipRuntime.driver, query, queryCfg, 2, 7, 10000, 4, 2, 0.5, separateQuery))
	core.RequireNoError(t, hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScale(context.Background(), hipRuntime.driver, key, keyCfg, 1, 7, 10000, 4, 2, 0.5, separateKey))

	gotQuery, err := hipReadFloat32DeviceOutput(pairedQuery, "rocm.hip.RMSNormRoPEHeadsPairHardware", "paired query output", len(queryValues))
	core.RequireNoError(t, err)
	gotKey, err := hipReadFloat32DeviceOutput(pairedKey, "rocm.hip.RMSNormRoPEHeadsPairHardware", "paired key output", len(keyValues))
	core.RequireNoError(t, err)
	wantSeparateQuery, err := hipReadFloat32DeviceOutput(separateQuery, "rocm.hip.RMSNormRoPEHeadsPairHardware", "separate query output", len(queryValues))
	core.RequireNoError(t, err)
	wantSeparateKey, err := hipReadFloat32DeviceOutput(separateKey, "rocm.hip.RMSNormRoPEHeadsPairHardware", "separate key output", len(keyValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, wantSeparateQuery, gotQuery, 0.0001)
	assertFloat32SlicesNear(t, wantSeparateKey, gotKey, 0.0001)

	queryWeights := []float32{1.25, 0.5, 2, 0.75}
	var wantQuery []float32
	for head := 0; head < 2; head++ {
		start := head * 4
		normalized, err := hipReferenceRMSNorm(queryValues[start:start+4], queryWeights, queryCfg.Epsilon)
		core.RequireNoError(t, err)
		rotated, err := hipReferenceRoPENeoXWithFrequencyDimScale(normalized, 7, 10000, 4, 2, 0.5)
		core.RequireNoError(t, err)
		wantQuery = append(wantQuery, rotated...)
	}
	normalizedKey, err := hipReferenceRMSNorm(keyValues, keyWeights, keyCfg.Epsilon)
	core.RequireNoError(t, err)
	wantKey, err := hipReferenceRoPENeoXWithFrequencyDimScale(normalizedKey, 7, 10000, 4, 2, 0.5)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, wantQuery, gotQuery, 0.0005)
	assertFloat32SlicesNear(t, wantKey, gotKey, 0.0001)
}

func TestHIPHardwareRMSNormRoPEHeadsPairLaneBatch_Good(t *testing.T) {
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

	queryValues := []float32{
		1, 0, 3, 4,
		0, 2, 5, 12,
		2, 0, 1, 1,
		0, 3, 4, 3,
	}
	keyValues := []float32{
		1, 2, 3, 4,
		4, 3, 2, 1,
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch", "hardware rms norm rope pair lane batch query input", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch", "hardware rms norm rope pair lane batch key input", keyPayload, len(keyValues))
	core.RequireNoError(t, err)
	defer key.Close()
	positions, err := hipUploadTokenIDs(hipRuntime.driver, []int32{2, 11})
	core.RequireNoError(t, err)
	defer positions.Close()
	queryOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch", "hardware rms norm rope pair lane batch query output", query.SizeBytes(), query.Count())
	core.RequireNoError(t, err)
	defer queryOutput.Close()
	keyOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch", "hardware rms norm rope pair lane batch key output", key.SizeBytes(), key.Count())
	core.RequireNoError(t, err)
	defer keyOutput.Close()

	queryCfg := hipRMSNormDeviceWeightConfig{
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingNone,
		Flags:          hipRMSNormLaunchFlagRoPENeoX,
	}
	keyCfg := hipRMSNormDeviceWeightConfig{
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	core.RequireNoError(t, hipRunRMSNormRoPEHeadsPairLaneBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(context.Background(), hipRuntime.driver, query, key, queryCfg, keyCfg, 2, 1, positions, 8, 8, 2, 0.5, queryOutput, keyOutput))

	gotQuery, err := hipReadFloat32DeviceOutput(queryOutput, "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch", "hardware rms norm rope pair lane batch query output", len(queryValues))
	core.RequireNoError(t, err)
	gotKey, err := hipReadFloat32DeviceOutput(keyOutput, "rocm.hip.RMSNormRoPEHeadsPairLaneBatchLaunch", "hardware rms norm rope pair lane batch key output", len(keyValues))
	core.RequireNoError(t, err)

	unitWeight := []float32{1, 1, 1, 1}
	var wantQuery []float32
	var wantKey []float32
	for batch, position := range []int{2, 11} {
		for head := 0; head < 2; head++ {
			start := (batch*2 + head) * 4
			normalized, err := hipReferenceRMSNorm(queryValues[start:start+4], unitWeight, 0)
			core.RequireNoError(t, err)
			rotated, err := hipReferenceRoPENeoXWithFrequencyDimScale(normalized, position, 8, 8, 2, 0.5)
			core.RequireNoError(t, err)
			wantQuery = append(wantQuery, rotated...)
		}
		start := batch * 4
		normalized, err := hipReferenceRMSNorm(keyValues[start:start+4], unitWeight, 0)
		core.RequireNoError(t, err)
		rotated, err := hipReferenceRoPEWithFrequencyDimScale(normalized[:2], position, 8, 8, 0.5)
		core.RequireNoError(t, err)
		normalized[0] = rotated[0]
		normalized[1] = rotated[1]
		wantKey = append(wantKey, normalized...)
	}
	assertFloat32SlicesNear(t, wantQuery, gotQuery, 0.0001)
	assertFloat32SlicesNear(t, wantKey, gotKey, 0.0001)
}

func TestHIPHardwareMLXAffineQ8ProjectionCols256Group32_Good(t *testing.T) {
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

	const (
		rows      = 3
		cols      = 256
		groupSize = 32
		bits      = 8
		groups    = cols / groupSize
	)
	input := make([]float32, cols)
	values := make([]uint32, rows*cols)
	scales := make([]uint16, rows*groups)
	biases := make([]uint16, rows*groups)
	for col := range input {
		input[col] = float32((col%17)-8) / 8
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			values[row*cols+col] = uint32((row*19 + col*7) & 0xff)
		}
		for group := 0; group < groups; group++ {
			scales[row*groups+group] = hipFloat32ToBFloat16(0.015625 * float32(row+group+1))
			biases[row*groups+group] = hipFloat32ToBFloat16(-0.25 * float32(row+1))
		}
	}
	req := hipMLXQ4ProjectionRequest{
		Input:     input,
		Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
		Scales:    scales,
		Biases:    biases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	want, err := hipReferenceMLXAffineProjection(req.Input, req.Weight, req.Scales, req.Biases, req.Rows, req.Cols, req.GroupSize, req.Bits)
	core.RequireNoError(t, err)
	got, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.01)
}

func TestHIPHardwareMLXAffineQ8ProjectionCols3072Group32_Good(t *testing.T) {
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

	const (
		rows      = 5
		cols      = 3072
		groupSize = 32
		bits      = 8
		groups    = cols / groupSize
	)
	input := make([]float32, cols)
	values := make([]uint32, rows*cols)
	scales := make([]uint16, rows*groups)
	biases := make([]uint16, rows*groups)
	for col := range input {
		input[col] = float32((col%23)-11) / 64
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			values[row*cols+col] = uint32((row*29 + col*11) & 0xff)
		}
		for group := 0; group < groups; group++ {
			scales[row*groups+group] = hipFloat32ToBFloat16(0.001953125 * float32((row+group)%7+1))
			biases[row*groups+group] = hipFloat32ToBFloat16(-0.03125 * float32(row+1))
		}
	}
	req := hipMLXQ4ProjectionRequest{
		Input:     input,
		Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
		Scales:    scales,
		Biases:    biases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	want, err := hipReferenceMLXAffineProjection(req.Input, req.Weight, req.Scales, req.Biases, req.Rows, req.Cols, req.GroupSize, req.Bits)
	core.RequireNoError(t, err)
	got, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.03)
}

func TestHIPHardwareMLXAffineQ8ProjectionCols2560Group64_Good(t *testing.T) {
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

	const (
		rows      = 5
		cols      = 2560
		groupSize = 64
		bits      = 8
		groups    = cols / groupSize
	)
	input := make([]float32, cols)
	values := make([]uint32, rows*cols)
	scales := make([]uint16, rows*groups)
	biases := make([]uint16, rows*groups)
	for col := range input {
		input[col] = float32((col%29)-14) / 128
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			values[row*cols+col] = uint32((row*31 + col*13) & 0xff)
		}
		for group := 0; group < groups; group++ {
			scales[row*groups+group] = hipFloat32ToBFloat16(float32((row+group)%7+1) / 4096)
			biases[row*groups+group] = hipFloat32ToBFloat16(-float32(row+1) / 512)
		}
	}
	req := hipMLXQ4ProjectionRequest{
		Input:     input,
		Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
		Scales:    scales,
		Biases:    biases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	want, err := hipReferenceMLXAffineProjection(req.Input, req.Weight, req.Scales, req.Biases, req.Rows, req.Cols, req.GroupSize, req.Bits)
	core.RequireNoError(t, err)
	got, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.05)
}

func TestHIPHardwareMLXAffineQ8GELUTanhCols2560Group64_Good(t *testing.T) {
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

	const (
		rows      = 9
		cols      = 2560
		groupSize = 64
		bits      = 8
	)
	input := make([]float32, cols)
	for col := range input {
		input[col] = float32((col%23)-11) / 128
	}
	makeReq := func(valueStride int, scaleBase float32) hipMLXQ4ProjectionRequest {
		groups := cols / groupSize
		values := make([]uint32, rows*cols)
		scales := make([]uint16, rows*groups)
		biases := make([]uint16, rows*groups)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				values[row*cols+col] = uint32((row*valueStride + col*(valueStride+2)) & 0xff)
			}
			for group := 0; group < groups; group++ {
				scales[row*groups+group] = hipFloat32ToBFloat16(scaleBase * float32((row+group)%7+1))
				biases[row*groups+group] = hipFloat32ToBFloat16(-scaleBase * float32((row+group)%3+1))
			}
		}
		return hipMLXQ4ProjectionRequest{
			Input:     input,
			Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
			Scales:    scales,
			Biases:    biases,
			Rows:      rows,
			Cols:      cols,
			GroupSize: groupSize,
			Bits:      bits,
		}
	}
	gateReq := makeReq(5, 0.0001220703125)
	upReq := makeReq(7, 0.000091552734375)
	gate, err := hipReferenceMLXAffineProjection(gateReq.Input, gateReq.Weight, gateReq.Scales, gateReq.Biases, gateReq.Rows, gateReq.Cols, gateReq.GroupSize, gateReq.Bits)
	core.RequireNoError(t, err)
	up, err := hipReferenceMLXAffineProjection(upReq.Input, upReq.Weight, upReq.Scales, upReq.Biases, upReq.Rows, upReq.Cols, upReq.GroupSize, upReq.Bits)
	core.RequireNoError(t, err)
	want := expectedGELUTanhMultiply(gate, up)
	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, hipGemma4Q4Layer0Operation, "hardware q8 group64 GELU input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	gateBuffers, err := gateReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer upBuffers.Close()
	deviceConfig := func(req hipMLXQ4ProjectionRequest, buffers *hipMLXQ4ProjectionDeviceBuffers) hipMLXQ4DeviceWeightConfig {
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: buffers.Weight.Pointer(),
			ScalePointer:  buffers.Scales.Pointer(),
			BiasPointer:   buffers.Biases.Pointer(),
			WeightBytes:   buffers.Weight.SizeBytes(),
			ScaleBytes:    buffers.Scales.SizeBytes(),
			BiasBytes:     buffers.Biases.SizeBytes(),
			Rows:          req.Rows,
			Cols:          req.Cols,
			GroupSize:     req.GroupSize,
			Bits:          req.Bits,
		}
	}
	output, err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), hipRuntime.driver, inputBuffer, deviceConfig(gateReq, gateBuffers), deviceConfig(upReq, upBuffers))
	core.RequireNoError(t, err)
	defer output.Close()
	got, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "hardware q8 group64 GELU output", rows)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.05)
}

func TestHIPHardwareMLXAffineQ4ProjectionCols2560Group32_Good(t *testing.T) {
	testHIPHardwareMLXAffineQ4ProjectionCols2560(t, 32, 9)
}

func TestHIPHardwareMLXAffineQ4ProjectionCols2560Group64_Good(t *testing.T) {
	testHIPHardwareMLXAffineQ4ProjectionCols2560(t, 64, 2048)
}

func testHIPHardwareMLXAffineQ4ProjectionCols2560(t *testing.T, groupSize, rows int) {
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

	const (
		cols = 2560
		bits = 4
	)
	groups := cols / groupSize
	input := make([]float32, cols)
	values := make([]uint32, rows*cols)
	scales := make([]uint16, rows*groups)
	biases := make([]uint16, rows*groups)
	for col := range input {
		input[col] = float32((col%29)-14) / 128
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			values[row*cols+col] = uint32((row*11 + col*5) & 0x0f)
		}
		for group := 0; group < groups; group++ {
			scales[row*groups+group] = hipFloat32ToBFloat16(float32((row+group)%7+1) / 256)
			biases[row*groups+group] = hipFloat32ToBFloat16(float32((row-group)%5) / 128)
		}
	}
	req := hipMLXQ4ProjectionRequest{
		Input:     input,
		Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
		Scales:    scales,
		Biases:    biases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	want, err := hipReferenceMLXAffineProjection(req.Input, req.Weight, req.Scales, req.Biases, req.Rows, req.Cols, req.GroupSize, req.Bits)
	core.RequireNoError(t, err)
	got, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.03)
}

func TestHIPHardwareMLXAffineQ4GELUTanhCols2560Group32_Good(t *testing.T) {
	testHIPHardwareMLXAffineQ4GELUTanhCols2560(t, 32)
}

func TestHIPHardwareMLXAffineQ4GELUTanhCols2560Group64_Good(t *testing.T) {
	testHIPHardwareMLXAffineQ4GELUTanhCols2560(t, 64)
}

func testHIPHardwareMLXAffineQ4GELUTanhCols2560(t *testing.T, groupSize int) {
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

	const (
		rows = 32
		cols = 2560
		bits = 4
	)
	input := make([]float32, cols)
	for col := range input {
		input[col] = float32((col%23)-11) / 128
	}
	makeReq := func(valueStride int, scaleBase float32) hipMLXQ4ProjectionRequest {
		groups := cols / groupSize
		values := make([]uint32, rows*cols)
		scales := make([]uint16, rows*groups)
		biases := make([]uint16, rows*groups)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				values[row*cols+col] = uint32((row*valueStride + col*(valueStride+2)) & 0x0f)
			}
			for group := 0; group < groups; group++ {
				scales[row*groups+group] = hipFloat32ToBFloat16(scaleBase * float32((row+group)%7+1))
				biases[row*groups+group] = hipFloat32ToBFloat16(-scaleBase * float32((row+group)%3+1))
			}
		}
		return hipMLXQ4ProjectionRequest{
			Input:     input,
			Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
			Scales:    scales,
			Biases:    biases,
			Rows:      rows,
			Cols:      cols,
			GroupSize: groupSize,
			Bits:      bits,
		}
	}
	gateReq := makeReq(5, 0.001953125)
	upReq := makeReq(7, 0.00146484375)
	gate, err := hipReferenceMLXAffineProjection(gateReq.Input, gateReq.Weight, gateReq.Scales, gateReq.Biases, gateReq.Rows, gateReq.Cols, gateReq.GroupSize, gateReq.Bits)
	core.RequireNoError(t, err)
	up, err := hipReferenceMLXAffineProjection(upReq.Input, upReq.Weight, upReq.Scales, upReq.Biases, upReq.Rows, upReq.Cols, upReq.GroupSize, upReq.Bits)
	core.RequireNoError(t, err)
	want := expectedGELUTanhMultiply(gate, up)
	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, hipGemma4Q4Layer0Operation, core.Sprintf("hardware q4 group%d GELU input", groupSize), inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	gateBuffers, err := gateReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer upBuffers.Close()
	deviceConfig := func(req hipMLXQ4ProjectionRequest, buffers *hipMLXQ4ProjectionDeviceBuffers) hipMLXQ4DeviceWeightConfig {
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: buffers.Weight.Pointer(),
			ScalePointer:  buffers.Scales.Pointer(),
			BiasPointer:   buffers.Biases.Pointer(),
			WeightBytes:   buffers.Weight.SizeBytes(),
			ScaleBytes:    buffers.Scales.SizeBytes(),
			BiasBytes:     buffers.Biases.SizeBytes(),
			Rows:          req.Rows,
			Cols:          req.Cols,
			GroupSize:     req.GroupSize,
			Bits:          req.Bits,
		}
	}
	output, err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), hipRuntime.driver, inputBuffer, deviceConfig(gateReq, gateBuffers), deviceConfig(upReq, upBuffers))
	core.RequireNoError(t, err)
	defer output.Close()
	got, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, core.Sprintf("hardware q4 group%d GELU output", groupSize), rows)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.03)
}

func TestHIPHardwareMLXAffineQ4GELUTanhE4BRow16_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatal("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	const (
		rows         = 10240
		cols         = 2560
		groupSize    = 64
		bits         = 4
		groups       = cols / groupSize
		packedPerRow = cols / 8
	)
	input := make([]float32, cols)
	for col := range input {
		input[col] = float32((col%31)-15) / 128
	}
	makeRequest := func(seed uint32, scaleBase float32) hipMLXQ4ProjectionRequest {
		weights := make([]uint32, rows*packedPerRow)
		scales := make([]uint16, rows*groups)
		biases := make([]uint16, rows*groups)
		for row := range rows {
			for packed := range packedPerRow {
				quant := uint32((row*3+packed*5+int(seed))&0x0f) * 0x11111111
				weights[row*packedPerRow+packed] = quant
			}
			for group := range groups {
				scale := scaleBase * float32((row+group)%5+1)
				scales[row*groups+group] = hipFloat32ToBFloat16(scale)
				biases[row*groups+group] = hipFloat32ToBFloat16(-7.5 * scale)
			}
		}
		return hipMLXQ4ProjectionRequest{
			Input: input, Weight: weights, Scales: scales, Biases: biases,
			Rows: rows, Cols: cols, GroupSize: groupSize, Bits: bits,
		}
	}
	gateRequest := makeRequest(3, 0.000244140625)
	upRequest := makeRequest(11, 0.00018310546875)
	gate, err := hipReferenceMLXAffineProjection(gateRequest.Input, gateRequest.Weight, gateRequest.Scales, gateRequest.Biases, rows, cols, groupSize, bits)
	core.RequireNoError(t, err)
	up, err := hipReferenceMLXAffineProjection(upRequest.Input, upRequest.Weight, upRequest.Scales, upRequest.Biases, rows, cols, groupSize, bits)
	core.RequireNoError(t, err)
	want := expectedGELUTanhMultiply(gate, up)

	gateBuffers, err := gateRequest.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upRequest.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer upBuffers.Close()
	deviceConfig := func(request hipMLXQ4ProjectionRequest, buffers *hipMLXQ4ProjectionDeviceBuffers) hipMLXQ4DeviceWeightConfig {
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: buffers.Weight.Pointer(), ScalePointer: buffers.Scales.Pointer(), BiasPointer: buffers.Biases.Pointer(),
			WeightBytes: buffers.Weight.SizeBytes(), ScaleBytes: buffers.Scales.SizeBytes(), BiasBytes: buffers.Biases.SizeBytes(),
			Rows: request.Rows, Cols: request.Cols, GroupSize: request.GroupSize, Bits: request.Bits,
		}
	}
	output, err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), hipRuntime.driver, gateBuffers.Input, deviceConfig(gateRequest, gateBuffers), deviceConfig(upRequest, upBuffers))
	core.RequireNoError(t, err)
	defer output.Close()
	got, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "hardware E4B q4 group64 row16 GELU output", rows)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.03)
}

func TestHIPHardwareMLXAffineQ4GELUTanh12BRow8MatchesRow16_Good(t *testing.T) {
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

	const (
		rows         = 15360
		cols         = 3840
		groupSize    = 32
		bits         = 4
		groups       = cols / groupSize
		packedPerRow = cols / 8
	)
	input := make([]float32, cols)
	for col := range input {
		input[col] = float32((col%31)-15) / 128
	}
	makeReq := func(seed int, scale float32) hipMLXQ4ProjectionRequest {
		weights := make([]uint32, rows*packedPerRow)
		scales := make([]uint16, rows*groups)
		biases := make([]uint16, rows*groups)
		for row := 0; row < rows; row++ {
			for packed := 0; packed < packedPerRow; packed++ {
				var word uint32
				for lane := 0; lane < 8; lane++ {
					quant := uint32((row*3 + packed*5 + lane*7 + seed) & 0x0f)
					word |= quant << (uint32(lane) * 4)
				}
				weights[row*packedPerRow+packed] = word
			}
			for group := 0; group < groups; group++ {
				groupScale := scale * float32((row+group)%5+1)
				scales[row*groups+group] = hipFloat32ToBFloat16(groupScale)
				biases[row*groups+group] = hipFloat32ToBFloat16(-7.5 * groupScale)
			}
		}
		return hipMLXQ4ProjectionRequest{
			Input:     input,
			Weight:    weights,
			Scales:    scales,
			Biases:    biases,
			Rows:      rows,
			Cols:      cols,
			GroupSize: groupSize,
			Bits:      bits,
		}
	}
	gateReq := makeReq(3, 0.000244140625)
	upReq := makeReq(11, 0.00018310546875)
	gateBuffers, err := gateReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer upBuffers.Close()
	deviceConfig := func(req hipMLXQ4ProjectionRequest, buffers *hipMLXQ4ProjectionDeviceBuffers) hipMLXQ4DeviceWeightConfig {
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: buffers.Weight.Pointer(),
			ScalePointer:  buffers.Scales.Pointer(),
			BiasPointer:   buffers.Biases.Pointer(),
			WeightBytes:   buffers.Weight.SizeBytes(),
			ScaleBytes:    buffers.Scales.SizeBytes(),
			BiasBytes:     buffers.Biases.SizeBytes(),
			Rows:          req.Rows,
			Cols:          req.Cols,
			GroupSize:     req.GroupSize,
			Bits:          req.Bits,
		}
	}

	previousRoute := hipMLXQ4GELUTanh12BGateUpRouteEnabled
	previousGeometry := hipMLXQ4GELUTanh12BGateUpGeometry
	hipMLXQ4GELUTanh12BGateUpRouteEnabled = true
	t.Cleanup(func() {
		hipMLXQ4GELUTanh12BGateUpRouteEnabled = previousRoute
		hipMLXQ4GELUTanh12BGateUpGeometry = previousGeometry
	})
	hipMLXQ4GELUTanh12BGateUpGeometry = ""
	row16Output, err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), hipRuntime.driver, gateBuffers.Input, deviceConfig(gateReq, gateBuffers), deviceConfig(upReq, upBuffers))
	core.RequireNoError(t, err)
	row16, err := hipReadFloat32DeviceOutput(row16Output, hipGemma4Q4Layer0Operation, "hardware 12B row16 GELU output", rows)
	row16Output.Close()
	core.RequireNoError(t, err)

	hipMLXQ4GELUTanh12BGateUpGeometry = "row8"
	row8Output, err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), hipRuntime.driver, gateBuffers.Input, deviceConfig(gateReq, gateBuffers), deviceConfig(upReq, upBuffers))
	core.RequireNoError(t, err)
	row8, err := hipReadFloat32DeviceOutput(row8Output, hipGemma4Q4Layer0Operation, "hardware 12B row8 GELU output", rows)
	row8Output.Close()
	core.RequireNoError(t, err)

	assertFloat32SlicesNear(t, row16, row8, 0.001)
}

func TestHIPHardwareMLXAffineQ6ProjectionCols3072Group16_Good(t *testing.T) {
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

	const (
		rows      = 5
		cols      = 3072
		groupSize = 16
		bits      = 6
		groups    = cols / groupSize
	)
	input := make([]float32, cols)
	values := make([]uint32, rows*cols)
	scales := make([]uint16, rows*groups)
	biases := make([]uint16, rows*groups)
	for col := range input {
		input[col] = float32((col%23)-11) / 64
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			values[row*cols+col] = uint32((row*17 + col*11) & 0x3f)
		}
		for group := 0; group < groups; group++ {
			scales[row*groups+group] = hipFloat32ToBFloat16(0.00390625 * float32((row+group)%7+1))
			biases[row*groups+group] = hipFloat32ToBFloat16(-0.015625 * float32(row+1))
		}
	}
	req := hipMLXQ4ProjectionRequest{
		Input:     input,
		Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
		Scales:    scales,
		Biases:    biases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	want, err := hipReferenceMLXAffineProjection(req.Input, req.Weight, req.Scales, req.Biases, req.Rows, req.Cols, req.GroupSize, req.Bits)
	core.RequireNoError(t, err)
	got, err := hipRunMLXQ4ProjectionKernel(context.Background(), hipRuntime.driver, req)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.03)
}

func TestHIPHardwareMLXAffineQ8AssistantMLPGroup32_Good(t *testing.T) {
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

	const (
		hidden       = 256
		intermediate = 2048
		groupSize    = 32
		bits         = 8
	)
	input := make([]float32, hidden)
	for index := range input {
		input[index] = float32((index%29)-14) / 256
	}
	makeReq := func(rows, cols, valueStride int, scaleBase float32) hipMLXQ4ProjectionRequest {
		groups := cols / groupSize
		values := make([]uint32, rows*cols)
		scales := make([]uint16, rows*groups)
		biases := make([]uint16, rows*groups)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				values[row*cols+col] = uint32((row*valueStride + col*(valueStride+4)) & 0xff)
			}
			for group := 0; group < groups; group++ {
				scales[row*groups+group] = hipFloat32ToBFloat16(scaleBase * float32((row+group)%5+1))
				biases[row*groups+group] = hipFloat32ToBFloat16(scaleBase * float32((row-group)%7) / 8)
			}
		}
		return hipMLXQ4ProjectionRequest{
			Input:     input,
			Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
			Scales:    scales,
			Biases:    biases,
			Rows:      rows,
			Cols:      cols,
			GroupSize: groupSize,
			Bits:      bits,
		}
	}
	gateReq := makeReq(intermediate, hidden, 7, 0.00048828125)
	upReq := makeReq(intermediate, hidden, 11, 0.00036621094)
	downReq := makeReq(hidden, intermediate, 13, 0.00012207031)
	gate, err := hipReferenceMLXAffineProjection(gateReq.Input, gateReq.Weight, gateReq.Scales, gateReq.Biases, gateReq.Rows, gateReq.Cols, gateReq.GroupSize, gateReq.Bits)
	core.RequireNoError(t, err)
	up, err := hipReferenceMLXAffineProjection(upReq.Input, upReq.Weight, upReq.Scales, upReq.Biases, upReq.Rows, upReq.Cols, upReq.GroupSize, upReq.Bits)
	core.RequireNoError(t, err)
	activated := expectedGELUTanhMultiply(gate, up)
	downReq.Input = activated
	want, err := hipReferenceMLXAffineProjection(activated, downReq.Weight, downReq.Scales, downReq.Biases, downReq.Rows, downReq.Cols, downReq.GroupSize, downReq.Bits)
	core.RequireNoError(t, err)

	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, hipGemma4Q4Layer0Operation, "hardware assistant MLP input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	gateBuffers, err := gateReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer upBuffers.Close()
	downBuffers, err := downReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer downBuffers.Close()
	deviceConfig := func(req hipMLXQ4ProjectionRequest, buffers *hipMLXQ4ProjectionDeviceBuffers) hipMLXQ4DeviceWeightConfig {
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: buffers.Weight.Pointer(),
			ScalePointer:  buffers.Scales.Pointer(),
			BiasPointer:   buffers.Biases.Pointer(),
			WeightBytes:   buffers.Weight.SizeBytes(),
			ScaleBytes:    buffers.Scales.SizeBytes(),
			BiasBytes:     buffers.Biases.SizeBytes(),
			Rows:          req.Rows,
			Cols:          req.Cols,
			GroupSize:     req.GroupSize,
			Bits:          req.Bits,
		}
	}

	output, err := hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInput(context.Background(), hipRuntime.driver, inputBuffer, deviceConfig(gateReq, gateBuffers), deviceConfig(upReq, upBuffers), deviceConfig(downReq, downBuffers))
	core.RequireNoError(t, err)
	defer output.Close()
	got, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "hardware assistant MLP output", hidden)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0.03)
}

func TestHIPHardwareMLXAffineQ8ProjectionBatchRow16MatchesRow8_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || !runtime.Available() || hipRuntime.driver == nil {
		t.Fatal("native ROCm runtime is not available")
	}

	const rows, cols, batch, groupSize, bits = 17, 2816, 32, 64, 8
	input := make([]float32, batch*cols)
	values := make([]uint32, rows*cols)
	scales := make([]uint16, rows*(cols/groupSize))
	biases := make([]uint16, len(scales))
	for index := range input {
		input[index] = float32((index*17)%251+1) / 251
	}
	for index := range values {
		values[index] = uint32((index*29 + 11) % 256)
	}
	for index := range scales {
		scales[index] = hipFloat32ToBFloat16(float32(index%7+1) / 997)
		biases[index] = hipFloat32ToBFloat16(float32(index%9-4) / 509)
	}
	req := hipMLXQ4ProjectionRequest{
		Input:     input[:cols],
		Weight:    hipPackMLXAffineValuesForTest(values, cols, bits),
		Scales:    scales,
		Biases:    biases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	buffers, err := req.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ8BatchRow16Hardware", "production-shaped input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
		Bits:          bits,
	}
	row16Output, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, inputBuffer, cfg, batch)
	core.RequireNoError(t, err)
	defer row16Output.Close()

	row8Output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ8BatchRow16Hardware", "row8 baseline output", uint64(rows*batch*4), rows*batch)
	core.RequireNoError(t, err)
	defer row8Output.Close()
	launchBytes, err := (hipMLXQ4ProjectionBatchLaunchArgs{
		InputPointer:  inputBuffer.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: row8Output.Pointer(),
		Rows:          rows,
		Cols:          cols,
		Batch:         batch,
		GroupSize:     groupSize,
		Bits:          bits,
		InputBytes:    inputBuffer.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   row8Output.SizeBytes(),
	}).Binary()
	core.RequireNoError(t, err)
	row8Config, err := hipMLXQ4ProjectionBatchLaunchConfig(launchBytes, rows, batch)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelNameMLXQ4ProjBatch, row8Config.Name)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, row8Config))

	row16Values, err := hipReadFloat32DeviceOutput(row16Output, "rocm.hip.MLXQ8BatchRow16Hardware", "row16 output", rows*batch)
	core.RequireNoError(t, err)
	row8Values, err := hipReadFloat32DeviceOutput(row8Output, "rocm.hip.MLXQ8BatchRow16Hardware", "row8 output", rows*batch)
	core.RequireNoError(t, err)
	maxAbs := 0.0
	maxRel := 0.0
	for index := range row8Values {
		diff := math.Abs(float64(row16Values[index] - row8Values[index]))
		scale := math.Max(math.Abs(float64(row8Values[index])), 1)
		maxAbs = math.Max(maxAbs, diff)
		maxRel = math.Max(maxRel, diff/scale)
	}
	t.Logf("q8 row16 production-shape delta: max_abs=%g max_rel=%g", maxAbs, maxRel)
	assertFloat32SlicesNearRelativeNamedForHardwareTest(t, "q8 row16 production-shaped projection", row8Values, row16Values, 0.001, 0.000001)
}

func TestHIPHardwareMLXAffineQ8GELUTanhBatchPackedProductionShape_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || !runtime.Available() || hipRuntime.driver == nil {
		t.Fatal("native ROCm runtime is not available")
	}

	const rows, cols, batch, groupSize, bits = 17, 2816, 32, 64, 8
	input := make([]float32, batch*cols)
	gateValues := make([]uint32, rows*cols)
	upValues := make([]uint32, rows*cols)
	gateScales := make([]uint16, rows*(cols/groupSize))
	gateBiases := make([]uint16, len(gateScales))
	upScales := make([]uint16, len(gateScales))
	upBiases := make([]uint16, len(gateScales))
	for index := range input {
		input[index] = float32((index*17)%251-125) / 251
	}
	for index := range gateValues {
		gateValues[index] = uint32((index*29 + 11) % 256)
		upValues[index] = uint32((index*43 + 7) % 256)
	}
	for index := range gateScales {
		gateScales[index] = hipFloat32ToBFloat16(float32(index%7+1) / 8192)
		gateBiases[index] = hipFloat32ToBFloat16(float32(index%7-3) / 1024)
		upScales[index] = hipFloat32ToBFloat16(float32(index%5+1) / 8192)
		upBiases[index] = hipFloat32ToBFloat16(float32(index%5-2) / 1024)
	}
	gateReq := hipMLXQ4ProjectionRequest{
		Input:     input[:cols],
		Weight:    hipPackMLXAffineValuesForTest(gateValues, cols, bits),
		Scales:    gateScales,
		Biases:    gateBiases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	upReq := hipMLXQ4ProjectionRequest{
		Input:     input[:cols],
		Weight:    hipPackMLXAffineValuesForTest(upValues, cols, bits),
		Scales:    upScales,
		Biases:    upBiases,
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Bits:      bits,
	}
	gateBuffers, err := gateReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upReq.deviceBuffers(hipRuntime.driver)
	core.RequireNoError(t, err)
	defer upBuffers.Close()
	inputPayload, err := hipFloat32Payload(input)
	core.RequireNoError(t, err)
	inputBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ8GELUPackedHardware", "production-width input", inputPayload, len(input))
	core.RequireNoError(t, err)
	defer inputBuffer.Close()
	deviceConfig := func(req hipMLXQ4ProjectionRequest, buffers *hipMLXQ4ProjectionDeviceBuffers) hipMLXQ4DeviceWeightConfig {
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: buffers.Weight.Pointer(),
			ScalePointer:  buffers.Scales.Pointer(),
			BiasPointer:   buffers.Biases.Pointer(),
			WeightBytes:   buffers.Weight.SizeBytes(),
			ScaleBytes:    buffers.Scales.SizeBytes(),
			BiasBytes:     buffers.Biases.SizeBytes(),
			Rows:          req.Rows,
			Cols:          req.Cols,
			GroupSize:     req.GroupSize,
			Bits:          req.Bits,
		}
	}
	output, err := hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(
		context.Background(),
		hipRuntime.driver,
		inputBuffer,
		deviceConfig(gateReq, gateBuffers),
		deviceConfig(upReq, upBuffers),
		batch,
	)
	core.RequireNoError(t, err)
	defer output.Close()
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.MLXQ8GELUPackedHardware", "packed q8 GELU output", rows*batch)
	core.RequireNoError(t, err)
	want := make([]float32, 0, rows*batch)
	for token := 0; token < batch; token++ {
		gateToken := gateReq
		gateToken.Input = input[token*cols : (token+1)*cols]
		upToken := upReq
		upToken.Input = gateToken.Input
		want = append(want, expectedGELUTanhMultiplyFromMLXAffine(t, gateToken, upToken, bits)...)
	}
	maxAbs := 0.0
	maxRel := 0.0
	for index := range want {
		diff := math.Abs(float64(got[index] - want[index]))
		scale := math.Max(math.Abs(float64(want[index])), 1)
		maxAbs = math.Max(maxAbs, diff)
		maxRel = math.Max(maxRel, diff/scale)
	}
	t.Logf("q8 packed GELU production-width delta: max_abs=%g max_rel=%g", maxAbs, maxRel)
	assertFloat32SlicesNearRelativeNamedForHardwareTest(t, "q8 packed GELU production-width", want, got, 0.000001, 0.000001)
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
		return
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

func TestHIPHardwareGemma4BF16Generate_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_GEMMA4_BF16_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_GEMMA4_BF16_TESTS=1 to run Gemma4 BF16 generation tests")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_GEMMA4_BF16_MODEL_PATH"))
	if modelPath == "" {
		t.Fatal("GO_ROCM_GEMMA4_BF16_MODEL_PATH is required")
	}

	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModel(modelPath, inference.WithContextLen(128)))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	rocmLoaded, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("loaded model = %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		t.Fatalf("native model = %T, want *hipLoadedModel", rocmLoaded.native)
	}
	if !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("Gemma4 BF16 model is not linked for generation: labels=%v", loaded.modelLabels)
	}
	forward, err := loaded.loadedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	core.RequireNoError(t, err)
	if len(forward.Layers) != loaded.modelInfo.NumLayers {
		t.Fatalf("BF16 forward layers = %d, want %d", len(forward.Layers), loaded.modelInfo.NumLayers)
	}
	for index, layer := range forward.Layers {
		for label, projection := range map[string]hipMLXQ4DeviceWeightConfig{
			"q_proj":        layer.QueryProjection,
			"k_proj":        layer.KeyProjection,
			"v_proj":        layer.ValueProjection,
			"o_proj":        layer.OutputProjection,
			"mlp.gate_proj": layer.GateProjection,
			"mlp.up_proj":   layer.UpProjection,
			"mlp.down_proj": layer.DownProjection,
			"lm_head":       layer.LMHeadProjection,
		} {
			if projection.WeightEncoding != hipProjectionWeightEncodingBF16 {
				t.Fatalf("layer %d %s encoding = %d, want BF16", index, label, projection.WeightEncoding)
			}
		}
	}

	generated := 0
	for token := range model.Generate(context.Background(), "hello", inference.WithMaxTokens(2)) {
		if token.ID < 0 {
			t.Fatalf("generated token = %+v, want non-negative ID", token)
		}
		generated++
	}
	core.RequireNoError(t, resultError(model.Err()))
	if generated != 2 {
		t.Fatalf("generated tokens = %d, want 2", generated)
	}
}

func TestHIPHardwareAttentionHeadsBatchCausalKQ8VQ4FutureRowsInvariant_Good(t *testing.T) {
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

	const (
		dim         = 128
		headCount   = 8
		keyHeads    = 4
		priorTokens = 64
		queryRows   = 5
	)
	keyWidth := keyHeads * dim
	valueWidth := keyHeads * dim
	priorKeyValues := make([]float32, priorTokens*keyWidth)
	priorValueValues := make([]float32, priorTokens*valueWidth)
	for index := range priorKeyValues {
		priorKeyValues[index] = float32(math.Sin(float64(index+3)*0.013) * 0.75)
	}
	for index := range priorValueValues {
		priorValueValues[index] = float32(math.Cos(float64(index+7)*0.017) * 0.5)
	}
	appendKeyValues := make([]float32, queryRows*keyWidth)
	appendValueValues := make([]float32, queryRows*valueWidth)
	for index := range appendKeyValues {
		appendKeyValues[index] = float32(math.Sin(float64(index+priorTokens*keyWidth)*0.019) * 0.75)
	}
	for index := range appendValueValues {
		appendValueValues[index] = float32(math.Cos(float64(index+priorTokens*valueWidth)*0.023) * 0.5)
	}
	queryValues := make([]float32, queryRows*headCount*dim)
	for index := range queryValues {
		queryValues[index] = float32(math.Sin(float64(index+11)*0.011) * 0.65)
	}

	priorKeyInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant prior keys", mustHIPFloat32Payload(t, priorKeyValues), len(priorKeyValues))
	core.RequireNoError(t, err)
	defer priorKeyInput.Close()
	priorValueInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant prior values", mustHIPFloat32Payload(t, priorValueValues), len(priorValueValues))
	core.RequireNoError(t, err)
	defer priorValueInput.Close()
	appendKeyInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant appended keys", mustHIPFloat32Payload(t, appendKeyValues), len(appendKeyValues))
	core.RequireNoError(t, err)
	defer appendKeyInput.Close()
	appendValueInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant appended values", mustHIPFloat32Payload(t, appendValueValues), len(appendValueValues))
	core.RequireNoError(t, err)
	defer appendValueInput.Close()
	queryInput, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant queries", mustHIPFloat32Payload(t, queryValues), len(queryValues))
	core.RequireNoError(t, err)
	defer queryInput.Close()

	engineConfig := defaultHIPGemma4Q4EngineConfig()
	prior, err := newROCmDeviceKVCacheFromDeviceRowsWithEngineConfig(context.Background(), hipRuntime.driver, rocmKVCacheModeKQ8VQ4, engineConfig.globalDeviceKVBlockSize(), priorKeyInput, priorValueInput, keyWidth, valueWidth, priorTokens, 0, engineConfig)
	core.RequireNoError(t, err)
	defer prior.Close()

	runBatch := func(t *testing.T, rows int) []float32 {
		t.Helper()
		keyBytes := uint64(rows * keyWidth * 4)
		valueBytes := uint64(rows * valueWidth * 4)
		keyRows := hipBorrowDeviceByteBufferValue(hipRuntime.driver, "future invariant appended key rows", appendKeyInput.Pointer(), keyBytes, rows*keyWidth)
		valueRows := hipBorrowDeviceByteBufferValue(hipRuntime.driver, "future invariant appended value rows", appendValueInput.Pointer(), valueBytes, rows*valueWidth)
		deviceKV, err := prior.withAppendedDeviceRowsWindowWithEngineConfig(context.Background(), &keyRows, &valueRows, keyWidth, valueWidth, rows, 0, engineConfig)
		core.RequireNoError(t, err)
		defer deviceKV.Close()
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer table.Close()
		outputCount := rows * headCount * dim
		output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant attention output", uint64(outputCount*4), outputCount)
		core.RequireNoError(t, err)
		defer output.Close()
		queryBytes := uint64(outputCount * 4)
		queryRows := hipBorrowDeviceByteBufferValue(hipRuntime.driver, "future invariant query rows", queryInput.Pointer(), queryBytes, outputCount)
		core.RequireNoError(t, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, hipAttentionHeadsBatchCausalDeviceRequest{
			DeviceKV:        deviceKV,
			DescriptorTable: table,
			Dim:             dim,
			TokenCount:      deviceKV.TokenCount(),
			HeadCount:       headCount,
			KeyHeads:        keyHeads,
			QueryCount:      rows,
			QueryStartToken: priorTokens,
			Scale:           1,
		}, &queryRows, output))
		got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchCausalLaunch", "future invariant attention output", outputCount)
		core.RequireNoError(t, err)
		return got
	}

	twoRows := runBatch(t, 2)
	fiveRows := runBatch(t, 5)
	firstTwoCount := 2 * headCount * dim
	assertFloat32SlicesNear(t, twoRows[:firstTwoCount], fiveRows[:firstTwoCount], 0.0001)
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

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	draftROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_KV_MODE")); raw != "" {
		draftROCmConfig.DeviceKVMode = raw
		if _, err := draftROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_DRAFT_KV_MODE=%q: %v", raw, err)
		}
	}

	backend := newROCmBackendWithRuntime(newSystemNativeRuntime())
	pair, err := backend.LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetOptions:    []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		DraftOptions:     []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		TargetROCmConfig: targetROCmConfig,
		DraftROCmConfig:  draftROCmConfig,
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
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_DRAFT_TOKENS=%q, want positive integer", raw)
		}
		draftTokens = value
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
	if result.Metrics.DraftCalls == 0 {
		t.Fatalf("GenerateNative metrics = %+v, want assistant draft calls", result.Metrics)
	}
	if result.Metrics.TargetCalls == 0 {
		t.Fatalf("GenerateNative metrics = %+v, want target verification calls", result.Metrics)
	}
	if result.Metrics.AcceptedTokens+result.Metrics.RejectedTokens != result.Metrics.DraftTokens {
		t.Fatalf("GenerateNative metrics = %+v, want accepted+rejected to match draft tokens", result.Metrics)
	}
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("native attached tokens=%v text=%q metrics=%+v", decodeTokenIDs(result.Tokens), result.Text, result.Metrics)
	}
	if err := pair.Close(); err != nil {
		t.Fatalf("close attached drafter pair before reference target load: %v", err)
	}
	pairClosed = true

	referenceModel, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel reference target %q: %v", targetPath, err)
	}
	defer referenceModel.Close()
	referenceTokens := collectInferenceTokens(referenceModel.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens), inference.WithTemperature(0)))
	targetText := strings.Join(inferenceTokenText(referenceTokens), "")
	if err := resultError(referenceModel.Err()); err != nil {
		t.Fatalf("reference target Generate(%q): %v", prompt, err)
	}
	if targetText == "" {
		t.Fatalf("reference target Generate(%q) returned empty text", prompt)
	}
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("reference target tokens=%v text=%q", inferenceTokenIDs(referenceTokens), targetText)
	}
	if result.Text == targetText {
		t.Logf("native attached smoke exact-match: text=%q metrics=%+v", result.Text, result.Metrics)
	} else {
		assertNativeAttachedDrafterTargetARMatchStable(t, targetPath, draftPath, targetROCmConfig, draftROCmConfig, prompt, maxTokens, draftTokens, result.Text, targetText)
	}
}

func TestNativeGemma4Q4GenerateMatchesBatchStepwise_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}
	maxTokens := 8
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		maxTokens = value
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel target %q: %v", targetPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok || rocmLoaded == nil {
		t.Fatalf("LoadModel target returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", rocmLoaded.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	stream, streamErr := hipGemma4Q4GenerateTokenSeqWithEngineConfig(context.Background(), loaded, cfg, promptTokens, inference.GenerateConfig{MaxTokens: maxTokens}, engineConfig)
	unrolled := inferenceTokenIDs(collectInferenceTokens(stream))
	if err := streamErr(); err != nil {
		t.Fatalf("unrolled Generate(%q): %v", prompt, err)
	}
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig)
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("target generate tokens=%v batch_stepwise tokens=%v", unrolled, stepwise)
	}
	core.AssertEqual(t, stepwise, unrolled)
}

func TestNativeGemma4Q4VerifierAcceptedBlocksMatchStepwise_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
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
	blockTokens := 4
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_VERIFY_BLOCK_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_VERIFY_BLOCK_TOKENS=%q, want positive integer", raw)
		}
		blockTokens = value
	}
	compactPrefixTokens := 0
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_VERIFY_COMPACT_PREFIX_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value < 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_VERIFY_COMPACT_PREFIX_TOKENS=%q, want non-negative integer", raw)
		}
		compactPrefixTokens = value
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel target %q: %v", targetPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok || rocmLoaded == nil {
		t.Fatalf("LoadModel target returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", rocmLoaded.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DISABLE_INTERLEAVED_ROWS") == "1" {
		engineConfig.DisableInterleavedRowPages = true
	}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_KV_BLOCK_SIZE")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_KV_BLOCK_SIZE=%q, want positive integer", raw)
		}
		engineConfig.DeviceKVBlockSize = value
		engineConfig.GlobalDeviceKVBlockSize = value
	}
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig)
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_STEPWISE_COMPACT") == "1" {
		stepwise = runNativeGemma4Q4StepwiseAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig, true)
	}
	blocked := runNativeGemma4Q4VerifierAcceptedBlocksAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, stepwise, blockTokens, compactPrefixTokens, engineConfig)
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("target stepwise tokens=%v verifier_accepted_block_%d tokens=%v", stepwise, blockTokens, blocked)
	}
	core.AssertEqual(t, stepwise, blocked)
}

func TestNativeGemma4Q4VerifierCarryWrongDraftMatchesStepwise_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}
	maxTokens := 80
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		maxTokens = value
	}
	if maxTokens < 3 {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS to at least 3 for verifier carry testing")
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel target %q: %v", targetPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok || rocmLoaded == nil {
		t.Fatalf("LoadModel target returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", rocmLoaded.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig)
	wrongDraftToken := stepwise[0]
	if len(stepwise) > 1 {
		wrongDraftToken = stepwise[1]
	}
	hasMismatchPoint := false
	for _, tokenID := range stepwise {
		if tokenID != wrongDraftToken {
			hasMismatchPoint = true
			break
		}
	}
	if !hasMismatchPoint {
		t.Skipf("target sequence does not diverge from repeated token %d within %d tokens", wrongDraftToken, maxTokens)
	}
	wrongDraft := []int32{wrongDraftToken, wrongDraftToken, wrongDraftToken, wrongDraftToken, wrongDraftToken}
	carried := runNativeGemma4Q4VerifierCarryAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig, wrongDraft)
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("target stepwise tokens=%v verifier_wrong_draft_carry tokens=%v wrong_draft=%v", stepwise, carried, wrongDraft)
	}
	core.AssertEqual(t, stepwise, carried)
}

func TestNativeAttachedDrafterAssistantDraftDoesNotShiftTargetVerify_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	draftPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_PATH"))
	if draftPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_DRAFT_PATH to a local Gemma4 MTP-QAT assistant pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}
	maxTokens := 80
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		maxTokens = value
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	draftROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_KV_MODE")); raw != "" {
		draftROCmConfig.DeviceKVMode = raw
		if _, err := draftROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_DRAFT_KV_MODE=%q: %v", raw, err)
		}
	}

	pair, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetOptions:    []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		DraftOptions:     []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		TargetROCmConfig: targetROCmConfig,
		DraftROCmConfig:  draftROCmConfig,
	})
	if err != nil {
		t.Fatalf("LoadAttachedDrafterPair(%q, %q): %v", targetPath, draftPath, err)
	}
	defer pair.Close()
	if !pair.NativeReady() {
		t.Fatalf("attached drafter native ready = false labels=%+v error=%q", pair.Attachment.Labels, pair.NativeError)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil {
		t.Fatalf("pair target = %T, want *rocmModel", pair.Target)
	}
	loaded, ok := target.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", target.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	runtime, err := loaded.attachedDrafterRuntimeSnapshot()
	if err != nil {
		t.Fatalf("attached drafter runtime snapshot: %v", err)
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig)
	repeated := stepwise[1]
	changeIndex := -1
	for index := 2; index < len(stepwise); index++ {
		if stepwise[index] != repeated {
			changeIndex = index
			break
		}
	}
	if changeIndex < 3 {
		t.Skipf("target sequence %v does not contain a late repeated-token change", stepwise)
	}
	prefix := changeIndex - 2
	if prefix+6 > len(stepwise) {
		t.Skipf("target sequence length %d cannot cover verifier window at prefix %d", len(stepwise), prefix)
	}
	draftTokens := make([]int32, 6)
	for index := range draftTokens {
		draftTokens[index] = stepwise[prefix]
	}
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(loaded, nil)
	withoutWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	withWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		if err := hipRecycleAttentionHeadsChunkedWorkspace(withoutWorkspace); err != nil {
			t.Fatalf("recycle verifier workspace: %v", err)
		}
		if err := hipRecycleAttentionHeadsChunkedWorkspace(withWorkspace); err != nil {
			t.Fatalf("recycle assistant workspace: %v", err)
		}
	}()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, withoutWorkspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		t.Fatalf("verifier decode workspace: %v", err)
	}
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, withWorkspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		t.Fatalf("assistant decode workspace: %v", err)
	}
	withoutWorkspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)
	withWorkspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)

	withoutAssistant := runNativeGemma4Q4AttachedTargetStateAfterTokensForHardwareTest(t, loaded, cfg, promptTokens, stepwise[:prefix], engineConfig, suppressTokens, withoutWorkspace)
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &withoutAssistant)
	withAssistant := runNativeGemma4Q4AttachedTargetStateAfterTokensForHardwareTest(t, loaded, cfg, promptTokens, stepwise[:prefix], engineConfig, suppressTokens, withWorkspace)
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &withAssistant)

	assistantBlock, err := hipRunAttachedDrafterAssistantDraftBlock(context.Background(), loaded.driver, hipAttachedDrafterAssistantDraftBlockRequest{
		LastToken:         stepwise[prefix-1],
		TargetHidden:      withAssistant.Current.DeviceFinalHidden,
		TargetForward:     cfg,
		TargetDeviceState: withAssistant.DeviceState,
		Plan:              runtime.assistantPlan,
		InputPlan:         runtime.inputPlan,
		Position:          withAssistant.Position,
		Epsilon:           1e-6,
		Softcap:           runtime.softcap,
		SuppressTokens:    suppressTokens,
		MaxDraftTokens:    6,
		Workspace:         withWorkspace,
	})
	if err != nil {
		t.Fatalf("assistant draft block at prefix %d: %v", prefix, err)
	}
	assistantTokens := append([]int32(nil), assistantBlock.Tokens...)
	if err := assistantBlock.Close(); err != nil {
		t.Fatalf("close assistant draft block: %v", err)
	}

	withoutVerify := runNativeGemma4Q4VerifierDecisionForHardwareTest(t, loaded, cfg, withoutAssistant, draftTokens, engineConfig, suppressTokens, withoutWorkspace)
	withVerify := runNativeGemma4Q4VerifierDecisionForHardwareTest(t, loaded, cfg, withAssistant, draftTokens, engineConfig, suppressTokens, withWorkspace)
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("assistant mutation probe prefix=%d stepwise=%v draft=%v assistant=%v without=%+v with=%+v", prefix, stepwise, draftTokens, assistantTokens, withoutVerify, withVerify)
	}
	core.AssertEqual(t, withoutVerify, withVerify)
}

func TestNativeGemma4Q4VerifierMixedDraftBlocksMatchStepwise_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}
	maxTokens := 80
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		maxTokens = value
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel target %q: %v", targetPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok || rocmLoaded == nil {
		t.Fatalf("LoadModel target returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", rocmLoaded.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig)
	if len(stepwise) < 6 {
		t.Skip("target sequence is too short for mixed verifier draft blocks")
	}
	repeated := stepwise[1]
	mixedBlocks := [][]int32{
		{180613, 36955, 62945, 144558, 236779, 36955},
		{243179, 83834, 230657, 205097, stepwise[0], 3292},
		{repeated, repeated, 236775, stepwise[0], 237221, stepwise[0]},
	}
	for len(mixedBlocks) < maxTokens {
		mixedBlocks = append(mixedBlocks, []int32{repeated, repeated, repeated, repeated, repeated, repeated})
	}
	mixed := runNativeGemma4Q4VerifierDraftBlocksAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, maxTokens, engineConfig, mixedBlocks)
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_TOKENS") == "1" {
		t.Logf("target stepwise tokens=%v verifier_mixed_draft tokens=%v", stepwise, mixed)
	}
	core.AssertEqual(t, stepwise, mixed)
}

func TestNativeGemma4Q4VerifierPartialChunkStateMatchesStepwise_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel target %q: %v", targetPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok || rocmLoaded == nil {
		t.Fatalf("LoadModel target returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", rocmLoaded.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	probeTokens := 65
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_STATE_PROBE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_STATE_PROBE_TOKENS=%q, want positive integer", raw)
		}
		probeTokens = value
	}
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, probeTokens+8, engineConfig)
	if len(stepwise) < probeTokens {
		t.Skipf("target sequence %v is too short for partial verifier state probe", stepwise)
	}
	repeated := stepwise[1]
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(loaded, nil)
	stepWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	verifyWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		if err := hipRecycleAttentionHeadsChunkedWorkspace(verifyWorkspace); err != nil {
			t.Fatalf("recycle verifier workspace: %v", err)
		}
		if err := hipRecycleAttentionHeadsChunkedWorkspace(stepWorkspace); err != nil {
			t.Fatalf("recycle stepwise workspace: %v", err)
		}
	}()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, stepWorkspace, cfg, len(promptTokens)+probeTokens+8); err != nil {
		t.Fatalf("stepwise decode workspace: %v", err)
	}
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, verifyWorkspace, cfg, len(promptTokens)+probeTokens+8); err != nil {
		t.Fatalf("verifier decode workspace: %v", err)
	}
	stepWorkspace.EnsureProjectionGreedyBestCapacity(probeTokens + 8)
	verifyWorkspace.EnsureProjectionGreedyBestCapacity(probeTokens + 8)

	want := runNativeGemma4Q4AttachedTargetStateAfterTokensForHardwareTest(t, loaded, cfg, promptTokens, stepwise[:probeTokens], engineConfig, suppressTokens, stepWorkspace)
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &want)

	mixedBlocks := [][]int32{
		{180613, 36955, 62945, 144558, 236779, 36955},
		{243179, 83834, 230657, 205097, stepwise[0], 3292},
		{repeated, repeated, 236775, stepwise[0], 237221, stepwise[0]},
	}
	for len(mixedBlocks)*6 < probeTokens+12 {
		mixedBlocks = append(mixedBlocks, []int32{repeated, repeated, repeated, repeated, repeated, repeated})
	}
	got := runNativeGemma4Q4AttachedTargetStateAfterMixedDraftPrefixForHardwareTest(t, loaded, cfg, promptTokens, engineConfig, suppressTokens, verifyWorkspace, mixedBlocks, probeTokens)
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &got)

	core.AssertEqual(t, want.Position, got.Position)
	core.AssertEqual(t, int32(want.Current.Greedy.TokenID), int32(got.Current.Greedy.TokenID))
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_LOG_STATE_PAGES") == "1" {
		t.Logf("probe=%d want pages %s", probeTokens, nativeGemma4Q4DeviceStatePageSummaryForHardwareTest(want.DeviceState, []int{2, 14}))
		t.Logf("probe=%d got pages %s", probeTokens, nativeGemma4Q4DeviceStatePageSummaryForHardwareTest(got.DeviceState, []int{2, 14}))
	}
	assertNativeGemma4Q4DeviceStatesMatchForHardwareTest(t, cfg, want.DeviceState, got.DeviceState)
}

func TestNativeGemma4Q4VerifierBatchedChunkStateMatchesStepwise_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}

	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
	if err != nil {
		t.Fatalf("LoadModel target %q: %v", targetPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok || rocmLoaded == nil {
		t.Fatalf("LoadModel target returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", rocmLoaded.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	prefixTokens := 5
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_BATCH_STATE_PREFIX_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value < 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_BATCH_STATE_PREFIX_TOKENS=%q, want non-negative integer", raw)
		}
		prefixTokens = value
	}
	chunkTokens := 6
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_BATCH_STATE_CHUNK_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 1 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_BATCH_STATE_CHUNK_TOKENS=%q, want integer greater than one", raw)
		}
		chunkTokens = value
	}
	stepwise := runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t, loaded, cfg, promptTokens, prefixTokens+chunkTokens+4, engineConfig)
	if len(stepwise) < prefixTokens+chunkTokens {
		t.Skipf("target sequence %v is too short for batched verifier state probe", stepwise)
	}
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(loaded, nil)
	stepWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	verifyWorkspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		if err := hipRecycleAttentionHeadsChunkedWorkspace(verifyWorkspace); err != nil {
			t.Fatalf("recycle verifier workspace: %v", err)
		}
		if err := hipRecycleAttentionHeadsChunkedWorkspace(stepWorkspace); err != nil {
			t.Fatalf("recycle stepwise workspace: %v", err)
		}
	}()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, stepWorkspace, cfg, len(promptTokens)+prefixTokens+chunkTokens+4); err != nil {
		t.Fatalf("stepwise decode workspace: %v", err)
	}
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, verifyWorkspace, cfg, len(promptTokens)+prefixTokens+chunkTokens+4); err != nil {
		t.Fatalf("verifier decode workspace: %v", err)
	}
	stepWorkspace.EnsureProjectionGreedyBestCapacity(prefixTokens + chunkTokens + 4)
	verifyWorkspace.EnsureProjectionGreedyBestCapacity(prefixTokens + chunkTokens + 4)

	want := runNativeGemma4Q4AttachedTargetStateAfterTokensForHardwareTest(t, loaded, cfg, promptTokens, stepwise[:prefixTokens+chunkTokens], engineConfig, suppressTokens, stepWorkspace)
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &want)
	var prefix hipAttachedDrafterTargetPrefillResult
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_BATCH_STATE_MIXED_PREFIX") == "1" {
		repeated := stepwise[1]
		mixedBlocks := [][]int32{
			{180613, 36955, 62945, 144558, 236779, 36955},
			{243179, 83834, 230657, 205097, stepwise[0], 3292},
			{repeated, repeated, 236775, stepwise[0], 237221, stepwise[0]},
		}
		prefix = runNativeGemma4Q4AttachedTargetStateAfterMixedDraftPrefixForHardwareTest(t, loaded, cfg, promptTokens, engineConfig, suppressTokens, verifyWorkspace, mixedBlocks, prefixTokens)
	} else {
		prefix = runNativeGemma4Q4AttachedTargetStateAfterTokensForHardwareTest(t, loaded, cfg, promptTokens, stepwise[:prefixTokens], engineConfig, suppressTokens, verifyWorkspace)
	}
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &prefix)
	batchWorkspace := verifyWorkspace
	var freshBatchWorkspace *hipAttentionHeadsChunkedWorkspace
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_BATCH_STATE_FRESH_WORKSPACE") == "1" {
		freshBatchWorkspace = hipBorrowAttentionHeadsChunkedWorkspace()
		if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, freshBatchWorkspace, cfg, len(promptTokens)+prefixTokens+chunkTokens+4); err != nil {
			t.Fatalf("fresh batch decode workspace: %v", err)
		}
		freshBatchWorkspace.EnsureProjectionGreedyBestCapacity(prefixTokens + chunkTokens + 4)
		batchWorkspace = freshBatchWorkspace
	}
	defer func() {
		if freshBatchWorkspace != nil {
			if err := hipRecycleAttentionHeadsChunkedWorkspace(freshBatchWorkspace); err != nil {
				t.Fatalf("recycle fresh batch workspace: %v", err)
			}
		}
	}()
	got := runNativeGemma4Q4AttachedTargetStateAfterBatchedVerifyForHardwareTest(t, loaded, cfg, &prefix, stepwise[prefixTokens:prefixTokens+chunkTokens], engineConfig, suppressTokens, batchWorkspace)
	defer closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &got)

	core.AssertEqual(t, want.Position, got.Position)
	core.AssertEqual(t, int32(want.Current.Greedy.TokenID), int32(got.Current.Greedy.TokenID))
	assertNativeGemma4Q4DeviceStatesMatchForHardwareTest(t, cfg, want.DeviceState, got.DeviceState)
}

func TestNativeAttachedDrafterAssistantDraftSeedProbe_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run ROCm model smoke tests")
	}
	targetPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH"))
	if targetPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_TARGET_PATH to a local Gemma4 QAT target pack")
	}
	draftPath := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_PATH"))
	if draftPath == "" {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_DRAFT_PATH to a local Gemma4 MTP-QAT assistant pack")
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_PROMPT"))
	if prompt == "" {
		prompt = "text:Write one concise sentence about ROCm inference."
	}
	maxTokens := 4
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS=%q, want positive integer", raw)
		}
		maxTokens = value
	}
	if maxTokens < 2 {
		t.Skip("set GO_ROCM_ATTACHED_DRAFTER_GENERATE_TOKENS to at least 2 for the assistant seed probe")
	}
	maxDraftTokens := 2
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DRAFT_TOKENS")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_DRAFT_TOKENS=%q, want positive integer", raw)
		}
		maxDraftTokens = value
	}
	targetROCmConfig := ROCmLoadConfig{}
	if raw := strings.TrimSpace(os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE")); raw != "" {
		targetROCmConfig.DeviceKVMode = raw
		if _, err := targetROCmConfig.deviceKVMode(); err != nil {
			t.Fatalf("GO_ROCM_ATTACHED_DRAFTER_TARGET_KV_MODE=%q: %v", raw, err)
		}
	}

	pair, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetROCmConfig: targetROCmConfig,
		TargetOptions:    []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		DraftOptions:     []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
	})
	if err != nil {
		t.Fatalf("LoadAttachedDrafterPair(%q, %q): %v", targetPath, draftPath, err)
	}
	defer pair.Close()
	if !pair.NativeReady() {
		t.Fatalf("attached drafter native ready = false labels=%+v error=%q", pair.Attachment.Labels, pair.NativeError)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil {
		t.Fatalf("pair target = %T, want *rocmModel", pair.Target)
	}
	loaded, ok := target.native.(*hipLoadedModel)
	if !ok || loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Fatalf("native target = %T linked=%v, want linked *hipLoadedModel", target.native, ok && loaded != nil && hipLoadedGemma4Q4GenerateLinked(loaded))
	}
	runtime, err := loaded.attachedDrafterRuntimeSnapshot()
	if err != nil {
		t.Fatalf("attached drafter runtime snapshot: %v", err)
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		t.Fatalf("loaded target forward config: %v", err)
	}
	promptTokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded)
	if err != nil {
		t.Fatalf("Gemma4 prompt %q: %v", prompt, err)
	}
	if !matched {
		t.Fatalf("Gemma4 prompt %q did not resolve to native target token IDs", prompt)
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	targetStream, targetStreamErr := hipGemma4Q4GenerateTokenSeqWithEngineConfig(context.Background(), loaded, cfg, promptTokens, inference.GenerateConfig{MaxTokens: maxTokens}, engineConfig)
	targetTokens := inferenceTokenIDs(collectInferenceTokens(targetStream))
	if err := targetStreamErr(); err != nil {
		t.Fatalf("target Generate(%q): %v", prompt, err)
	}
	if len(targetTokens) < 2 {
		t.Fatalf("target Generate(%q) tokens=%v, want at least 2 tokens", prompt, targetTokens)
	}
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(loaded.driver, workspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("decode workspace: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(loaded.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("greedy buffer: %v", err)
	}
	defer func() {
		_ = greedyBuffer
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil {
			t.Fatalf("recycle workspace: %v", err)
		}
	}()
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(loaded, nil)
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), loaded.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&prefill.Current)
		if prefill.DeviceState != nil {
			_ = prefill.DeviceState.Close()
		}
	}()
	lastPromptSeed, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, runtime, cfg, prefill, prefill.LastToken, maxDraftTokens, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant draft from last prompt token: %v", err)
	}
	targetGreedySeed := int32(prefill.Current.Greedy.TokenID)
	nextTargetSeed, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, runtime, cfg, prefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant draft from target greedy token: %v", err)
	}
	fullHeadRuntime := *runtime
	fullHeadRuntime.assistantPlan.OrderedEmbeddings = false
	fullHeadRuntime.assistantPlan.MaskedCentroids = hipAttachedDrafterAssistantProjectionPlan{}
	fullHeadRuntime.assistantPlan.TokenOrdering = nil
	fullHeadRuntime.assistantPlan.TokenOrderingPointer = 0
	fullHeadRuntime.assistantPlan.TokenOrderingBytes = 0
	fullHeadRuntime.assistantPlan.TokenOrderingElementBytes = 0
	fullHeadRuntime.assistantPlan.TokenOrderingDeviceReady = false
	lastPromptSeedFullHead, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, &fullHeadRuntime, cfg, prefill, prefill.LastToken, maxDraftTokens, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant full-head draft from last prompt token: %v", err)
	}
	targetGreedySeedFullHead, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, &fullHeadRuntime, cfg, prefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant full-head draft from target greedy token: %v", err)
	}
	lastPromptRaw, err := runNativeAttachedDrafterAssistantDraftSeedProbeStepForHardwareTest(t, loaded, runtime, cfg, prefill, prefill.LastToken, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant raw-hidden draft from last prompt token: %v", err)
	}
	targetGreedyRaw, err := runNativeAttachedDrafterAssistantDraftSeedProbeStepForHardwareTest(t, loaded, runtime, cfg, prefill, targetGreedySeed, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant raw-hidden draft from target greedy token: %v", err)
	}
	lastPromptAligned, err := runNativeAttachedDrafterAssistantDraftSeedProbeAlignedStepForHardwareTest(t, loaded, runtime, cfg, prefill, prefill.LastToken, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant aligned draft from last prompt token: %v", err)
	}
	targetGreedyAligned, err := runNativeAttachedDrafterAssistantDraftSeedProbeAlignedStepForHardwareTest(t, loaded, runtime, cfg, prefill, targetGreedySeed, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant aligned draft from target greedy token: %v", err)
	}
	noSoftcapRuntime := *runtime
	noSoftcapRuntime.softcap = 0
	targetGreedyNoSoftcap, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, &noSoftcapRuntime, cfg, prefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant no-softcap draft from target greedy token: %v", err)
	}
	forcedSourceProbes := make([]string, 0, 3)
	for _, sourceProbe := range []struct {
		name    string
		sliding int
		full    int
	}{
		{name: "first", sliding: 0, full: 4},
		{name: "shared", sliding: 13, full: 14},
		{name: "last", sliding: 33, full: 34},
	} {
		forcedCfg := hipForceAttachedDrafterTargetSourcesForHardwareTest(cfg, sourceProbe.sliding, sourceProbe.full)
		proposals, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, runtime, forcedCfg, prefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
		if err != nil {
			forcedSourceProbes = append(forcedSourceProbes, core.Sprintf("%s=%s", sourceProbe.name, err.Error()))
			continue
		}
		forcedSourceProbes = append(forcedSourceProbes, core.Sprintf("%s=%v", sourceProbe.name, proposals))
	}
	positionProbes := make([]string, 0, 3)
	seenPositions := []int{0, 1, prefill.Position - 2, prefill.Position - 1, prefill.Position, prefill.Position + 1}
	seenPositionLogged := map[int]bool{}
	for _, seenPosition := range seenPositions {
		if seenPosition < 0 || seenPositionLogged[seenPosition] {
			continue
		}
		seenPositionLogged[seenPosition] = true
		positionPrefill := prefill
		positionPrefill.Position = seenPosition + 1
		proposals, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, runtime, cfg, positionPrefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
		if err != nil {
			positionProbes = append(positionProbes, core.Sprintf("seen%d=%s", seenPosition, err.Error()))
			continue
		}
		positionProbes = append(positionProbes, core.Sprintf("seen%d=%v", seenPosition, proposals))
	}
	scaleProbes := make([]string, 0, 3)
	for _, scaleProbe := range []struct {
		name  string
		scale float32
	}{
		{name: "unit", scale: 1},
		{name: "default", scale: runtime.inputPlan.TargetEmbeddingScale},
		{name: "inverse", scale: 1 / runtime.inputPlan.TargetEmbeddingScale},
	} {
		scaleRuntime := *runtime
		scaleRuntime.inputPlan.TargetEmbeddingScale = scaleProbe.scale
		proposals, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, &scaleRuntime, cfg, prefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
		if err != nil {
			scaleProbes = append(scaleProbes, core.Sprintf("%s=%s", scaleProbe.name, err.Error()))
			continue
		}
		scaleProbes = append(scaleProbes, core.Sprintf("%s=%v", scaleProbe.name, proposals))
	}
	rawScaleProbes := make([]string, 0, 3)
	for _, scaleProbe := range []struct {
		name  string
		scale float32
	}{
		{name: "unit", scale: 1},
		{name: "default", scale: runtime.inputPlan.TargetEmbeddingScale},
		{name: "inverse", scale: 1 / runtime.inputPlan.TargetEmbeddingScale},
	} {
		scaleRuntime := *runtime
		scaleRuntime.inputPlan.TargetEmbeddingScale = scaleProbe.scale
		proposal, err := runNativeAttachedDrafterAssistantDraftSeedProbeStepForHardwareTest(t, loaded, &scaleRuntime, cfg, prefill, targetGreedySeed, suppressTokens, workspace)
		if err != nil {
			rawScaleProbes = append(rawScaleProbes, core.Sprintf("%s=%s", scaleProbe.name, err.Error()))
			continue
		}
		rawScaleProbes = append(rawScaleProbes, core.Sprintf("%s=%d", scaleProbe.name, proposal))
	}
	layerPrefixProbes := make([]string, 0, len(runtime.assistantPlan.Layers))
	for count := 1; count <= len(runtime.assistantPlan.Layers); count++ {
		prefixRuntime := *runtime
		prefixRuntime.assistantPlan = runtime.assistantPlan
		prefixRuntime.assistantPlan.Layers = append([]hipAttachedDrafterAssistantVerifierLayerPlan(nil), runtime.assistantPlan.Layers[:count]...)
		prefixRuntime.assistantPlan.LayerCount = count
		proposal, err := runNativeAttachedDrafterAssistantDraftSeedProbeStepForHardwareTest(t, loaded, &prefixRuntime, cfg, prefill, targetGreedySeed, suppressTokens, workspace)
		if err != nil {
			layerPrefixProbes = append(layerPrefixProbes, core.Sprintf("%d=%s", count, err.Error()))
			continue
		}
		layerPrefixProbes = append(layerPrefixProbes, core.Sprintf("%d=%d", count, proposal))
	}
	layerSources := make([]string, 0, len(runtime.assistantPlan.Layers))
	layerPlans := make([]string, 0, len(runtime.assistantPlan.Layers))
	for _, layer := range runtime.assistantPlan.Layers {
		_, _, source, err := hipAttachedDrafterAssistantTargetLayerFor(layer.LayerType, cfg, prefill.DeviceState)
		if err != nil {
			t.Fatalf("assistant target layer source for %s: %v", layer.LayerType, err)
		}
		layerSources = append(layerSources, core.Sprintf("%d:%s->%d", layer.Layer, layer.LayerType, source))
		layerPlans = append(layerPlans, core.Sprintf("%d:%s q=%d h=%d heads=%d rope=%d/%g win=%d scalar=%g",
			layer.Layer,
			layer.LayerType,
			layer.QueryProjection.Rows,
			layer.HeadDim,
			layer.QueryHeads,
			layer.RoPERotaryDim,
			layer.RoPEBase,
			layer.SlidingWindow,
			layer.LayerScalar,
		))
	}
	postFirstTarget, err := hipAdvanceAttachedDrafterCarryLead(context.Background(), loaded.driver, hipAttachedDrafterCarryAdvanceRequest{
		TargetForward:    cfg,
		DeviceKVMode:     deviceKVMode,
		EngineConfig:     engineConfig,
		State:            prefill.State,
		PriorDeviceState: prefill.DeviceState,
		TokenID:          targetGreedySeed,
		Position:         prefill.Position,
		Epsilon:          1e-6,
		SuppressTokens:   suppressTokens,
		GreedyBuffer:     greedyBuffer,
		Workspace:        workspace,
	})
	if err != nil {
		t.Fatalf("advance first target token: %v", err)
	}
	postFirstPrefill := hipAttachedDrafterTargetPrefillResult{
		Current:     postFirstTarget.Current,
		State:       postFirstTarget.State,
		DeviceState: postFirstTarget.DeviceState,
		Position:    postFirstTarget.Position,
		LastToken:   targetGreedySeed,
	}
	postFirstTarget.Current = hipGemma4Q4ForwardResult{}
	postFirstTarget.DeviceState = nil
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&postFirstPrefill.Current)
		if postFirstPrefill.DeviceState != nil {
			_ = postFirstPrefill.DeviceState.Close()
		}
		_ = postFirstTarget.Close()
	}()
	postFirstSeed, err := runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t, loaded, runtime, cfg, postFirstPrefill, targetGreedySeed, maxDraftTokens, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant draft after first target token: %v", err)
	}
	postFirstRaw, err := runNativeAttachedDrafterAssistantDraftSeedProbeStepForHardwareTest(t, loaded, runtime, cfg, postFirstPrefill, targetGreedySeed, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant raw-hidden draft after first target token: %v", err)
	}
	postFirstAligned, err := runNativeAttachedDrafterAssistantDraftSeedProbeAlignedStepForHardwareTest(t, loaded, runtime, cfg, postFirstPrefill, targetGreedySeed, suppressTokens, workspace)
	if err != nil {
		t.Fatalf("assistant aligned draft after first target token: %v", err)
	}
	draftCfg := nativeGemma4TextConfig{}
	if runtime.draft != nil {
		draftCfg = runtime.draft.gemma4TextConfig
	}
	t.Logf("assistant seed probe target=%v target_kv_mode=%s last_prompt_seed_token=%d ordered=%v full_head=%v raw_step=%d aligned_step=%d target_greedy_seed_token=%d ordered=%v full_head=%v raw_step=%d aligned_step=%d no_softcap=%v runtime_softcap=%g post_first_ordered=%v post_first_raw_step=%d post_first_aligned_step=%d forced_sources=%s position_delta=%s embedding_scales=%s raw_embedding_scales=%s layer_prefix_raw=%s plan_hidden=%d vocab=%d quant=%s/%d group=%d ordered=%t input=%dx%d post=%dx%d layers=%d sources=%s layer_plans=%s target_cfg_layers=%d target_kv_shared=%d/%t target_pattern=%d target_layer_types=%v draft_cfg_layers=%d draft_kv_shared=%d/%t draft_pattern=%d draft_head=%d/%d draft_hidden_per_input=%d draft_layer_types=%v",
		targetTokens,
		deviceKVMode,
		prefill.LastToken,
		lastPromptSeed,
		lastPromptSeedFullHead,
		lastPromptRaw,
		lastPromptAligned,
		targetGreedySeed,
		nextTargetSeed,
		targetGreedySeedFullHead,
		targetGreedyRaw,
		targetGreedyAligned,
		targetGreedyNoSoftcap,
		runtime.softcap,
		postFirstSeed,
		postFirstRaw,
		postFirstAligned,
		strings.Join(forcedSourceProbes, ","),
		strings.Join(positionProbes, ","),
		strings.Join(scaleProbes, ","),
		strings.Join(rawScaleProbes, ","),
		strings.Join(layerPrefixProbes, ","),
		runtime.assistantPlan.HiddenSize,
		runtime.assistantPlan.VocabSize,
		runtime.assistantPlan.QuantMode,
		runtime.assistantPlan.QuantBits,
		runtime.assistantPlan.QuantGroup,
		runtime.assistantPlan.OrderedEmbeddings,
		runtime.inputPlan.PreProjection.Rows,
		runtime.inputPlan.PreProjection.Cols,
		runtime.assistantPlan.PostProjection.Rows,
		runtime.assistantPlan.PostProjection.Cols,
		len(runtime.assistantPlan.Layers),
		strings.Join(layerSources, ","),
		strings.Join(layerPlans, ";"),
		loaded.gemma4TextConfig.NumLayers,
		loaded.gemma4TextConfig.KVSharedLayers,
		loaded.gemma4TextConfig.KVSharedLayersSet,
		loaded.gemma4TextConfig.SlidingWindowPattern,
		loaded.gemma4TextConfig.LayerTypes,
		draftCfg.NumLayers,
		draftCfg.KVSharedLayers,
		draftCfg.KVSharedLayersSet,
		draftCfg.SlidingWindowPattern,
		draftCfg.HeadDim,
		draftCfg.GlobalHeadDim,
		draftCfg.HiddenSizePerLayerInput,
		draftCfg.LayerTypes,
	)
}

func hipForceAttachedDrafterTargetSourcesForHardwareTest(cfg hipGemma4Q4ForwardConfig, slidingSource, fullSource int) hipGemma4Q4ForwardConfig {
	cfg.SharedKVSources = make([]int, len(cfg.Layers))
	for index, layer := range cfg.Layers {
		source := index
		switch layer.LayerType {
		case "sliding_attention":
			source = slidingSource
		case "full_attention":
			source = fullSource
		}
		if source < 0 || source >= len(cfg.Layers) {
			source = index
		}
		cfg.SharedKVSources[index] = source
	}
	return cfg
}

func runNativeAttachedDrafterAssistantDraftSeedProbeBlockForHardwareTest(t *testing.T, model *hipLoadedModel, runtime *hipAttachedDrafterRuntime, cfg hipGemma4Q4ForwardConfig, prefill hipAttachedDrafterTargetPrefillResult, seedToken int32, maxDraftTokens int, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) ([]int32, error) {
	t.Helper()
	block, err := hipRunAttachedDrafterAssistantDraftBlock(context.Background(), model.driver, hipAttachedDrafterAssistantDraftBlockRequest{
		LastToken:         seedToken,
		TargetHidden:      prefill.Current.DeviceFinalHidden,
		TargetForward:     cfg,
		TargetDeviceState: prefill.DeviceState,
		Plan:              runtime.assistantPlan,
		InputPlan:         runtime.inputPlan,
		Position:          prefill.Position,
		Epsilon:           1e-6,
		Softcap:           runtime.softcap,
		SuppressTokens:    suppressTokens,
		MaxDraftTokens:    maxDraftTokens,
		Workspace:         workspace,
	})
	if err != nil {
		return nil, err
	}
	tokens := append([]int32(nil), block.Tokens...)
	if err := block.Close(); err != nil {
		return nil, err
	}
	return tokens, nil
}

func runNativeAttachedDrafterAssistantDraftSeedProbeStepForHardwareTest(t *testing.T, model *hipLoadedModel, runtime *hipAttachedDrafterRuntime, cfg hipGemma4Q4ForwardConfig, prefill hipAttachedDrafterTargetPrefillResult, seedToken int32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (int32, error) {
	t.Helper()
	return runNativeAttachedDrafterAssistantDraftSeedProbeStepWithHiddenForHardwareTest(t, model, runtime, cfg, prefill, prefill.Current.DeviceFinalHidden, prefill.Position, seedToken, suppressTokens, workspace)
}

func runNativeAttachedDrafterAssistantDraftSeedProbeAlignedStepForHardwareTest(t *testing.T, model *hipLoadedModel, runtime *hipAttachedDrafterRuntime, cfg hipGemma4Q4ForwardConfig, prefill hipAttachedDrafterTargetPrefillResult, seedToken int32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (int32, error) {
	t.Helper()
	if len(cfg.Layers) == 0 {
		return 0, core.E("rocm.hip.AttachedDrafterAssistantDraftSeedProbe", "target forward config has no layers", nil)
	}
	targetNormCfg := cfg.Layers[len(cfg.Layers)-1].FinalNorm
	targetNormCfg.Epsilon = 1e-6
	if err := hipValidateRMSNormDeviceWeightConfig("AttachedDrafterAssistantDraftSeedProbe.target_final_norm", targetNormCfg); err != nil {
		return 0, err
	}
	normedHidden, err := hipAllocateByteBuffer(model.driver, "rocm.hip.AttachedDrafterAssistantDraftSeedProbe", "target final-norm aligned seed", uint64(targetNormCfg.Count*4), targetNormCfg.Count)
	if err != nil {
		return 0, err
	}
	defer normedHidden.Close()
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(context.Background(), model.driver, prefill.Current.DeviceFinalHidden.Pointer(), uint64(targetNormCfg.Count)*4, normedHidden.Pointer(), normedHidden.SizeBytes(), targetNormCfg, workspace); err != nil {
		return 0, err
	}
	return runNativeAttachedDrafterAssistantDraftSeedProbeStepWithHiddenForHardwareTest(t, model, runtime, cfg, prefill, normedHidden, hipAttachedDrafterAssistantSeenPosition(prefill.Position), seedToken, suppressTokens, workspace)
}

func runNativeAttachedDrafterAssistantDraftSeedProbeStepWithHiddenForHardwareTest(t *testing.T, model *hipLoadedModel, runtime *hipAttachedDrafterRuntime, cfg hipGemma4Q4ForwardConfig, prefill hipAttachedDrafterTargetPrefillResult, targetHidden *hipDeviceByteBuffer, position int, seedToken int32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (int32, error) {
	t.Helper()
	step, err := hipRunAttachedDrafterAssistantDraftStepProposal(context.Background(), model.driver, hipAttachedDrafterAssistantDraftStepProposalRequest{
		LastToken:         seedToken,
		TargetHidden:      targetHidden,
		TargetForward:     cfg,
		TargetDeviceState: prefill.DeviceState,
		Plan:              runtime.assistantPlan,
		InputPlan:         runtime.inputPlan,
		Position:          position,
		Epsilon:           1e-6,
		Softcap:           runtime.softcap,
		SuppressTokens:    suppressTokens,
		Workspace:         workspace,
	})
	if err != nil {
		return 0, err
	}
	token := int32(step.Token.TokenID)
	if err := step.Close(); err != nil {
		return 0, err
	}
	return token, nil
}

type nativeGemma4Q4VerifierDecisionForHardwareTest struct {
	Accepted    int
	AllAccepted bool
	Replacement int32
	Next        int32
	Verified    []int32
}

func runNativeGemma4Q4AttachedTargetStateAfterTokensForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens, generatedTokens []int32, engineConfig hipGemma4Q4EngineConfig, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) hipAttachedDrafterTargetPrefillResult {
	t.Helper()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("greedy buffer: %v", err)
	}
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	current := prefill.Current
	state := prefill.State
	deviceState := prefill.DeviceState
	position := prefill.Position
	success := false
	defer func() {
		if success {
			return
		}
		hipReleaseForwardDeviceFinalHidden(&current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	for index, tokenID := range generatedTokens {
		if int32(current.Greedy.TokenID) != tokenID {
			t.Fatalf("advance prefix token %d current greedy=%d want %d", index, current.Greedy.TokenID, tokenID)
		}
		advanced, err := hipAdvanceAttachedDrafterCarryLead(context.Background(), model.driver, hipAttachedDrafterCarryAdvanceRequest{
			TargetForward:    cfg,
			DeviceKVMode:     deviceKVMode,
			EngineConfig:     engineConfig,
			State:            state,
			PriorDeviceState: deviceState,
			TokenID:          tokenID,
			Position:         position,
			Epsilon:          1e-6,
			SuppressTokens:   suppressTokens,
			GreedyBuffer:     greedyBuffer,
			Workspace:        workspace,
		})
		if err != nil {
			t.Fatalf("advance prefix token %d: %v", index, err)
		}
		hipReleaseForwardDeviceFinalHidden(&current)
		previousDeviceState := deviceState
		current = advanced.Current
		advanced.Current = hipGemma4Q4ForwardResult{}
		state = advanced.State
		deviceState = advanced.DeviceState
		advanced.DeviceState = nil
		position = advanced.Position
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		if err := advanced.Close(); err != nil {
			t.Fatalf("close advanced prefix token %d: %v", index, err)
		}
	}
	success = true
	return hipAttachedDrafterTargetPrefillResult{
		Current:     current,
		State:       state,
		DeviceState: deviceState,
		Position:    position,
		LastToken:   promptTokens[len(promptTokens)-1],
	}
}

func closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t *testing.T, state *hipAttachedDrafterTargetPrefillResult) {
	t.Helper()
	if state == nil {
		return
	}
	hipReleaseForwardDeviceFinalHidden(&state.Current)
	if state.DeviceState != nil {
		if err := state.DeviceState.Close(); err != nil {
			t.Fatalf("close target device state: %v", err)
		}
		state.DeviceState = nil
	}
}

func assertNativeGemma4Q4DeviceStatesMatchForHardwareTest(t *testing.T, cfg hipGemma4Q4ForwardConfig, want, got *hipGemma4Q4DeviceDecodeState) {
	t.Helper()
	if want == nil || got == nil {
		t.Fatalf("device states want=%v got=%v, want both non-nil", want, got)
	}
	core.AssertEqual(t, want.LayerTokenCounts(), got.LayerTokenCounts())
	assertNativeGemma4Q4DeviceDescriptorTablesMatchCachesForHardwareTest(t, "want", want)
	assertNativeGemma4Q4DeviceDescriptorTablesMatchCachesForHardwareTest(t, "got", got)
	wantHost, err := want.HostState()
	if err != nil {
		t.Fatalf("restore want device state: %v", err)
	}
	gotHost, err := got.HostState()
	if err != nil {
		t.Fatalf("restore got device state: %v", err)
	}
	core.AssertEqual(t, len(wantHost.Layers), len(gotHost.Layers))
	if len(wantHost.Layers) != len(cfg.Layers) {
		t.Fatalf("restored layer count = %d, want %d", len(wantHost.Layers), len(cfg.Layers))
	}
	for index := range wantHost.Layers {
		assertFloat32SlicesNearRelativeNamedForHardwareTest(t, core.Sprintf("layer %d keys", index), wantHost.Layers[index].Keys, gotHost.Layers[index].Keys, 0.0001, 0.0001)
		assertFloat32SlicesNearRelativeNamedForHardwareTest(t, core.Sprintf("layer %d values", index), wantHost.Layers[index].Values, gotHost.Layers[index].Values, 0.0001, 0.0001)
	}
}

func assertNativeGemma4Q4DeviceDescriptorTablesMatchCachesForHardwareTest(t *testing.T, name string, state *hipGemma4Q4DeviceDecodeState) {
	t.Helper()
	if state == nil {
		t.Fatalf("%s device state is nil", name)
	}
	for index := range state.layers {
		layer := &state.layers[index]
		if layer.cache == nil || layer.descriptorTable == nil {
			t.Fatalf("%s layer %d cache=%v descriptor=%v, want both non-nil", name, index, layer.cache, layer.descriptorTable)
		}
		want, err := layer.cache.KernelDescriptorBytes()
		if err != nil {
			t.Fatalf("%s layer %d descriptor bytes from cache: %v", name, index, err)
		}
		if layer.descriptorTable.SizeBytes() < uint64(len(want)) {
			t.Fatalf("%s layer %d descriptor table bytes = %d, want at least %d", name, index, layer.descriptorTable.SizeBytes(), len(want))
		}
		got := make([]byte, len(want))
		if err := layer.descriptorTable.driver.CopyDeviceToHost(layer.descriptorTable.Pointer(), got); err != nil {
			t.Fatalf("%s layer %d copy descriptor table: %v", name, index, err)
		}
		for offset := range want {
			if got[offset] != want[offset] {
				t.Fatalf("%s layer %d descriptor byte[%d] = %d, want %d", name, index, offset, got[offset], want[offset])
			}
		}
	}
}

func assertFloat32SlicesNearRelativeNamedForHardwareTest(t *testing.T, name string, want, got []float32, absTol, relTol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for index := range want {
		diff := math.Abs(float64(got[index] - want[index]))
		scale := math.Max(math.Abs(float64(want[index])), 1)
		if diff > absTol && diff/scale > relTol {
			t.Fatalf("%s[%d] = %f, want %f within abs=%f rel=%f", name, index, got[index], want[index], absTol, relTol)
		}
	}
}

func nativeGemma4Q4DeviceStatePageSummaryForHardwareTest(state *hipGemma4Q4DeviceDecodeState, layers []int) string {
	if state == nil {
		return "<nil>"
	}
	parts := make([]string, 0, len(layers))
	for _, layerIndex := range layers {
		if layerIndex < 0 || layerIndex >= len(state.layers) || state.layers[layerIndex].cache == nil {
			parts = append(parts, core.Sprintf("L%d:<nil>", layerIndex))
			continue
		}
		layer := state.layers[layerIndex]
		pageParts := make([]string, 0, len(layer.cache.pages))
		for _, page := range layer.cache.pages {
			pageParts = append(pageParts, core.Sprintf("%d:%d:%t", page.tokenStart, page.tokenCount, page.owned))
		}
		parts = append(parts, core.Sprintf("L%d borrowed=%t tokens=%d pages=[%s]", layerIndex, layer.borrowedCache, layer.cache.TokenCount(), strings.Join(pageParts, ",")))
	}
	return strings.Join(parts, " ")
}

func runNativeGemma4Q4AttachedTargetStateAfterBatchedVerifyForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, prefix *hipAttachedDrafterTargetPrefillResult, draftTokens []int32, engineConfig hipGemma4Q4EngineConfig, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) hipAttachedDrafterTargetPrefillResult {
	t.Helper()
	if prefix == nil || prefix.DeviceState == nil {
		t.Fatalf("batched verifier prefix state is required")
	}
	if len(draftTokens) <= 1 {
		t.Fatalf("batched verifier draft tokens length = %d, want more than one", len(draftTokens))
	}
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("verifier greedy buffer: %v", err)
	}
	verify, err := hipRunAttachedDrafterTargetVerifyBlockBatched(context.Background(), model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
		TargetForward:     cfg,
		DeviceKVMode:      deviceKVMode,
		EngineConfig:      engineConfig,
		TargetDeviceState: prefix.DeviceState,
		CurrentGreedy:     prefix.Current.Greedy,
		DraftTokens:       draftTokens,
		Position:          prefix.Position,
		Epsilon:           1e-6,
		SuppressTokens:    suppressTokens,
		GreedyBuffer:      greedyBuffer,
		Workspace:         workspace,
	})
	if err != nil {
		t.Fatalf("batched verifier target block: %v", err)
	}
	success := false
	defer func() {
		if !success {
			_ = verify.Close()
		}
	}()
	if !verify.AllAccepted || verify.AcceptedCount != len(draftTokens) {
		t.Fatalf("batched verifier accepted=%d all=%v replacement=%d draft=%v, want all accepted", verify.AcceptedCount, verify.AllAccepted, verify.Replacement.TokenID, draftTokens)
	}
	if verify.DeviceState == nil {
		t.Fatalf("batched verifier did not return device state")
	}
	previousDeviceState := prefix.DeviceState
	if !verify.PriorDeviceStateFinalized {
		if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, verify.DeviceState); err != nil {
			t.Fatalf("finalize batched verifier state: %v", err)
		}
	}
	hipReleaseForwardDeviceFinalHidden(&prefix.Current)
	prefix.DeviceState = nil
	hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
	current := hipGemma4Q4ForwardResult{
		Greedy:                    verify.NextGreedy,
		DeviceFinalHidden:         verify.DeviceHidden,
		DeviceFinalHiddenBorrowed: false,
	}
	verify.DeviceHidden = nil
	result := hipAttachedDrafterTargetPrefillResult{
		Current:     current,
		DeviceState: verify.DeviceState,
		Position:    prefix.Position + len(draftTokens),
		LastToken:   int32(draftTokens[len(draftTokens)-1]),
	}
	verify.DeviceState = nil
	if err := verify.Close(); err != nil {
		closeNativeGemma4Q4AttachedTargetStateForHardwareTest(t, &result)
		t.Fatalf("close batched verifier result: %v", err)
	}
	success = true
	return result
}

func runNativeGemma4Q4AttachedTargetStateAfterMixedDraftPrefixForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, engineConfig hipGemma4Q4EngineConfig, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace, draftBlocks [][]int32, wantGeneratedTokens int) hipAttachedDrafterTargetPrefillResult {
	t.Helper()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("greedy buffer: %v", err)
	}
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	current := prefill.Current
	state := prefill.State
	deviceState := prefill.DeviceState
	position := prefill.Position
	tokens := make([]int32, 0, wantGeneratedTokens+1)
	success := false
	defer func() {
		if success {
			return
		}
		hipReleaseForwardDeviceFinalHidden(&current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	for blockIndex, draftTokens := range draftBlocks {
		remaining := wantGeneratedTokens - len(tokens)
		if remaining <= 0 {
			success = true
			return hipAttachedDrafterTargetPrefillResult{
				Current:     current,
				State:       state,
				DeviceState: deviceState,
				Position:    position,
				LastToken:   promptTokens[len(promptTokens)-1],
			}
		}
		if len(draftTokens) > remaining {
			draftTokens = draftTokens[:remaining]
		}
		verifyGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
		if err != nil {
			t.Fatalf("verifier greedy buffer: %v", err)
		}
		verify, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
			TargetForward:     cfg,
			DeviceKVMode:      deviceKVMode,
			EngineConfig:      engineConfig,
			TargetDeviceState: deviceState,
			CurrentGreedy:     current.Greedy,
			DraftTokens:       draftTokens,
			Position:          position,
			Epsilon:           1e-6,
			SuppressTokens:    suppressTokens,
			GreedyBuffer:      verifyGreedyBuffer,
			Workspace:         workspace,
		})
		if err != nil {
			t.Fatalf("verifier mixed block %d: %v", blockIndex, err)
		}
		if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_TRACE_BLOCKS") == "1" {
			t.Logf("mixed_state block=%d output=%d position=%d current=%d draft=%v verified=%v accepted=%d all=%t replacement=%d next=%d",
				blockIndex,
				len(tokens),
				position,
				current.Greedy.TokenID,
				draftTokens,
				hipAttachedDrafterGreedyTokenIDs(verify.VerifiedGreedies),
				verify.AcceptedCount,
				verify.AllAccepted,
				verify.Replacement.TokenID,
				verify.NextGreedy.TokenID,
			)
		}
		for index := 0; index < verify.AcceptedCount && len(tokens) < wantGeneratedTokens; index++ {
			tokens = append(tokens, draftTokens[index])
		}
		if verify.DeviceState != nil {
			previousDeviceState := deviceState
			if !verify.PriorDeviceStateFinalized {
				if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, verify.DeviceState); err != nil {
					_ = verify.Close()
					t.Fatalf("verifier mixed block %d finalize state: %v", blockIndex, err)
				}
			}
			deviceState = verify.DeviceState
			verify.DeviceState = nil
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			position = deviceState.maxLayerTokenCount()
		}
		if verify.DeviceHidden != nil {
			hipReleaseForwardDeviceFinalHidden(&current)
			current.DeviceFinalHidden = verify.DeviceHidden
			current.DeviceFinalHiddenBorrowed = false
			verify.DeviceHidden = nil
		}
		current.Greedy = verify.NextGreedy
		current.GreedyDevice = nil
		if verify.AllAccepted {
			if err := verify.Close(); err != nil {
				t.Fatalf("close verifier mixed block %d: %v", blockIndex, err)
			}
			if len(tokens) >= wantGeneratedTokens {
				success = true
				return hipAttachedDrafterTargetPrefillResult{
					Current:     current,
					State:       state,
					DeviceState: deviceState,
					Position:    position,
					LastToken:   promptTokens[len(promptTokens)-1],
				}
			}
			continue
		}
		replacement := int32(verify.Replacement.TokenID)
		if len(tokens) >= wantGeneratedTokens {
			if err := verify.Close(); err != nil {
				t.Fatalf("close verifier mixed block %d: %v", blockIndex, err)
			}
			success = true
			return hipAttachedDrafterTargetPrefillResult{
				Current:     current,
				State:       state,
				DeviceState: deviceState,
				Position:    position,
				LastToken:   promptTokens[len(promptTokens)-1],
			}
		}
		tokens = append(tokens, replacement)
		advanceGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
		if err != nil {
			_ = verify.Close()
			t.Fatalf("advance greedy buffer mixed block %d: %v", blockIndex, err)
		}
		advanced, err := hipAdvanceAttachedDrafterCarryLead(context.Background(), model.driver, hipAttachedDrafterCarryAdvanceRequest{
			TargetForward:    cfg,
			DeviceKVMode:     deviceKVMode,
			EngineConfig:     engineConfig,
			State:            state,
			PriorDeviceState: deviceState,
			TokenID:          replacement,
			Position:         position,
			Epsilon:          1e-6,
			SuppressTokens:   suppressTokens,
			GreedyBuffer:     advanceGreedyBuffer,
			Workspace:        workspace,
		})
		if err != nil {
			_ = verify.Close()
			t.Fatalf("advance mixed block %d replacement %d: %v", blockIndex, replacement, err)
		}
		hipReleaseForwardDeviceFinalHidden(&current)
		previousDeviceState := deviceState
		current = advanced.Current
		advanced.Current = hipGemma4Q4ForwardResult{}
		state = advanced.State
		deviceState = advanced.DeviceState
		advanced.DeviceState = nil
		position = advanced.Position
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		if err := advanced.Close(); err != nil {
			_ = verify.Close()
			t.Fatalf("close advanced mixed block %d: %v", blockIndex, err)
		}
		if err := verify.Close(); err != nil {
			t.Fatalf("close verifier mixed block %d: %v", blockIndex, err)
		}
		if len(tokens) >= wantGeneratedTokens {
			success = true
			return hipAttachedDrafterTargetPrefillResult{
				Current:     current,
				State:       state,
				DeviceState: deviceState,
				Position:    position,
				LastToken:   promptTokens[len(promptTokens)-1],
			}
		}
	}
	t.Fatalf("mixed draft prefix produced %d tokens, want %d", len(tokens), wantGeneratedTokens)
	return hipAttachedDrafterTargetPrefillResult{}
}

func runNativeGemma4Q4VerifierDecisionForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, targetState hipAttachedDrafterTargetPrefillResult, draftTokens []int32, engineConfig hipGemma4Q4EngineConfig, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) nativeGemma4Q4VerifierDecisionForHardwareTest {
	t.Helper()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("verifier greedy buffer: %v", err)
	}
	verify, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
		TargetForward:     cfg,
		DeviceKVMode:      deviceKVMode,
		EngineConfig:      engineConfig,
		TargetDeviceState: targetState.DeviceState,
		CurrentGreedy:     targetState.Current.Greedy,
		DraftTokens:       draftTokens,
		Position:          targetState.Position,
		Epsilon:           1e-6,
		SuppressTokens:    suppressTokens,
		GreedyBuffer:      greedyBuffer,
		Workspace:         workspace,
	})
	if err != nil {
		t.Fatalf("verifier decision: %v", err)
	}
	decision := nativeGemma4Q4VerifierDecisionForHardwareTest{
		Accepted:    verify.AcceptedCount,
		AllAccepted: verify.AllAccepted,
		Replacement: int32(verify.Replacement.TokenID),
		Next:        int32(verify.NextGreedy.TokenID),
		Verified:    hipAttachedDrafterGreedyTokenIDs(verify.VerifiedGreedies),
	}
	if err := verify.Close(); err != nil {
		t.Fatalf("close verifier decision: %v", err)
	}
	return decision
}

func runNativeGemma4Q4StepwiseAfterAttachedPrefillForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, maxTokens int, engineConfig hipGemma4Q4EngineConfig, useDeviceToken bool) []int32 {
	t.Helper()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("decode workspace: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("greedy buffer: %v", err)
	}
	defer func() {
		_ = greedyBuffer
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil {
			t.Fatalf("recycle workspace: %v", err)
		}
	}()
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	current := prefill.Current
	state := prefill.State
	deviceState := prefill.DeviceState
	position := prefill.Position
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	tokens := make([]int32, 0, maxTokens)
	for len(tokens) < maxTokens {
		tokenID := int32(current.Greedy.TokenID)
		tokens = append(tokens, tokenID)
		if len(tokens) == maxTokens {
			break
		}
		var tokenIDDeviceBuffer *hipDeviceByteBuffer
		if useDeviceToken {
			tokenIDDeviceBuffer = current.GreedyDevice
		}
		forward, nextState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), model.driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID:             tokenID,
			Position:            position,
			Epsilon:             1e-6,
			DeviceKVAttention:   true,
			DeviceKVMode:        deviceKVMode,
			EngineConfig:        engineConfig,
			PriorDeviceState:    deviceState,
			ReturnDeviceState:   true,
			DeviceFinalSample:   true,
			FinalGreedyBuffer:   greedyBuffer,
			TokenIDDeviceBuffer: tokenIDDeviceBuffer,
			SuppressTokens:      suppressTokens,
			AttentionWorkspace:  workspace,
			OmitDebugTensors:    true,
			OmitLabels:          true,
			OmitHostState:       true,
		}, false)
		if err != nil {
			t.Fatalf("stepwise token %d forward: %v", len(tokens), err)
		}
		if forward.DeviceState == nil {
			t.Fatalf("stepwise token %d forward did not return device state", len(tokens))
		}
		hipReleaseForwardDeviceFinalHidden(&current)
		previousDeviceState := deviceState
		deviceState = forward.DeviceState
		forward.DeviceState = nil
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		current = forward
		state = nextState
		position++
	}
	return tokens
}

func runNativeGemma4Q4StepwiseBatchPrefillAfterAttachedPrefillForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, maxTokens int, engineConfig hipGemma4Q4EngineConfig) []int32 {
	t.Helper()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("decode workspace: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("greedy buffer: %v", err)
	}
	defer func() {
		_ = greedyBuffer
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil {
			t.Fatalf("recycle workspace: %v", err)
		}
	}()
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	currentGreedy := prefill.Current.Greedy
	deviceState := prefill.DeviceState
	position := prefill.Position
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&prefill.Current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	tokens := make([]int32, 0, maxTokens)
	for len(tokens) < maxTokens {
		tokenID := int32(currentGreedy.TokenID)
		tokens = append(tokens, tokenID)
		if len(tokens) == maxTokens {
			break
		}
		priorLayerKV := hipGemma4Q4DeviceLayerCaches(deviceState, nil, len(cfg.Layers))
		priorLayerDescriptors, err := hipGemma4Q4DeviceLayerDescriptorTableAliases(deviceState, nil, len(cfg.Layers))
		if err != nil {
			t.Fatalf("batch1 token %d descriptor aliases: %v", len(tokens), err)
		}
		forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(context.Background(), model.driver, cfg, []int32{tokenID}, position, 1e-6, deviceKVMode, priorLayerKV, priorLayerDescriptors, nil, nil, 0, greedyBuffer, workspace, engineConfig)
		hipCloseGemma4Q4DeviceLayerDescriptorTables(priorLayerDescriptors)
		if err != nil {
			t.Fatalf("batch1 token %d forward: %v", len(tokens), err)
		}
		if len(forward.Greedy) != 1 {
			_ = forward.Close()
			t.Fatalf("batch1 token %d greedy rows = %d, want 1", len(tokens), len(forward.Greedy))
		}
		nextGreedy := forward.Greedy[0].Greedy
		nextDeviceState, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, deviceKVMode)
		closeErr := forward.Close()
		if stateErr != nil {
			t.Fatalf("batch1 token %d state: %v", len(tokens), stateErr)
		}
		if closeErr != nil {
			_ = nextDeviceState.Close()
			t.Fatalf("batch1 token %d close forward: %v", len(tokens), closeErr)
		}
		previousDeviceState := deviceState
		if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, nextDeviceState); err != nil {
			_ = nextDeviceState.Close()
			t.Fatalf("batch1 token %d finalize state: %v", len(tokens), err)
		}
		deviceState = nextDeviceState
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		currentGreedy = nextGreedy
		position = deviceState.maxLayerTokenCount()
	}
	return tokens
}

func runNativeGemma4Q4VerifierCarryAfterAttachedPrefillForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, maxTokens int, engineConfig hipGemma4Q4EngineConfig, wrongDraft []int32) []int32 {
	t.Helper()
	if len(wrongDraft) == 0 {
		t.Fatalf("wrong draft suffix is required")
	}
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("decode workspace: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("greedy buffer: %v", err)
	}
	defer func() {
		_ = greedyBuffer
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil {
			t.Fatalf("recycle workspace: %v", err)
		}
	}()
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	current := prefill.Current
	state := prefill.State
	deviceState := prefill.DeviceState
	position := prefill.Position
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	tokens := make([]int32, 0, maxTokens)
	carryLead := int32(-1)
	for len(tokens) < maxTokens {
		if carryLead < 0 {
			carryLead = int32(current.Greedy.TokenID)
			tokens = append(tokens, carryLead)
			continue
		}
		verifyTokens := make([]int32, 0, len(wrongDraft)+1)
		verifyTokens = append(verifyTokens, carryLead)
		verifyTokens = append(verifyTokens, wrongDraft...)
		verify, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
			TargetForward:     cfg,
			DeviceKVMode:      deviceKVMode,
			EngineConfig:      engineConfig,
			TargetDeviceState: deviceState,
			CurrentGreedy:     current.Greedy,
			DraftTokens:       verifyTokens,
			Position:          position,
			Epsilon:           1e-6,
			SuppressTokens:    suppressTokens,
			GreedyBuffer:      greedyBuffer,
			Workspace:         workspace,
		})
		if err != nil {
			t.Fatalf("verifier token %d: %v", len(tokens), err)
		}
		if verify.AcceptedCount == 0 {
			_ = verify.Close()
			t.Fatalf("verifier token %d rejected carried token %d", len(tokens), carryLead)
		}
		for index := 1; index < verify.AcceptedCount && len(tokens) < maxTokens; index++ {
			tokens = append(tokens, verifyTokens[index])
		}
		if verify.DeviceState != nil {
			previousDeviceState := deviceState
			if !verify.PriorDeviceStateFinalized {
				if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, verify.DeviceState); err != nil {
					_ = verify.Close()
					t.Fatalf("verifier token %d finalize state: %v", len(tokens), err)
				}
			}
			deviceState = verify.DeviceState
			verify.DeviceState = nil
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			position = deviceState.maxLayerTokenCount()
		}
		if verify.DeviceHidden != nil {
			hipReleaseForwardDeviceFinalHidden(&current)
			current.DeviceFinalHidden = verify.DeviceHidden
			current.DeviceFinalHiddenBorrowed = false
			verify.DeviceHidden = nil
		}
		current.Greedy = verify.NextGreedy
		current.GreedyDevice = nil
		if verify.AllAccepted {
			carryLead = -1
			_ = verify.Close()
			continue
		}
		carryLead = int32(verify.Replacement.TokenID)
		if len(tokens) < maxTokens {
			tokens = append(tokens, carryLead)
		}
		_ = verify.Close()
		state = hipGemma4Q4DecodeState{}
	}
	_ = state
	return tokens
}

func runNativeGemma4Q4VerifierDraftBlocksAfterAttachedPrefillForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, maxTokens int, engineConfig hipGemma4Q4EngineConfig, draftBlocks [][]int32) []int32 {
	t.Helper()
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(promptTokens)+maxTokens+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("decode workspace: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(maxTokens + 2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("greedy buffer: %v", err)
	}
	defer func() {
		_ = greedyBuffer
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil {
			t.Fatalf("recycle workspace: %v", err)
		}
	}()
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      workspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	current := prefill.Current
	state := prefill.State
	deviceState := prefill.DeviceState
	position := prefill.Position
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	tokens := make([]int32, 0, maxTokens)
	carryLead := int32(-1)
	blockIndex := 0
	for len(tokens) < maxTokens {
		if carryLead >= 0 {
			carryGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
			if err != nil {
				t.Fatalf("carry greedy buffer: %v", err)
			}
			advanced, err := hipAdvanceAttachedDrafterCarryLead(context.Background(), model.driver, hipAttachedDrafterCarryAdvanceRequest{
				TargetForward:    cfg,
				DeviceKVMode:     deviceKVMode,
				EngineConfig:     engineConfig,
				State:            state,
				PriorDeviceState: deviceState,
				TokenID:          carryLead,
				Position:         position,
				Epsilon:          1e-6,
				SuppressTokens:   suppressTokens,
				GreedyBuffer:     carryGreedyBuffer,
				Workspace:        workspace,
			})
			if err != nil {
				t.Fatalf("advance carried token %d: %v", carryLead, err)
			}
			hipReleaseForwardDeviceFinalHidden(&current)
			previousDeviceState := deviceState
			current = advanced.Current
			advanced.Current = hipGemma4Q4ForwardResult{}
			state = advanced.State
			deviceState = advanced.DeviceState
			advanced.DeviceState = nil
			position = advanced.Position
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			if err := advanced.Close(); err != nil {
				t.Fatalf("close carried token advance: %v", err)
			}
			carryLead = -1
		}
		if blockIndex >= len(draftBlocks) {
			t.Fatalf("ran out of draft blocks after %d tokens", len(tokens))
		}
		draftTokens := draftBlocks[blockIndex]
		blockIndex++
		if len(draftTokens) == 0 {
			t.Fatalf("draft block %d is empty", blockIndex-1)
		}
		remaining := maxTokens - len(tokens)
		if len(draftTokens) > remaining {
			draftTokens = draftTokens[:remaining]
		}
		verifyGreedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
		if err != nil {
			t.Fatalf("verifier greedy buffer: %v", err)
		}
		verify, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
			TargetForward:     cfg,
			DeviceKVMode:      deviceKVMode,
			EngineConfig:      engineConfig,
			TargetDeviceState: deviceState,
			CurrentGreedy:     current.Greedy,
			DraftTokens:       draftTokens,
			Position:          position,
			Epsilon:           1e-6,
			SuppressTokens:    suppressTokens,
			GreedyBuffer:      verifyGreedyBuffer,
			Workspace:         workspace,
		})
		if err != nil {
			t.Fatalf("verifier mixed block %d token %d: %v", blockIndex-1, len(tokens), err)
		}
		for index := 0; index < verify.AcceptedCount && len(tokens) < maxTokens; index++ {
			tokens = append(tokens, draftTokens[index])
		}
		if verify.DeviceState != nil {
			previousDeviceState := deviceState
			if !verify.PriorDeviceStateFinalized {
				if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, verify.DeviceState); err != nil {
					_ = verify.Close()
					t.Fatalf("verifier mixed block %d finalize state: %v", blockIndex-1, err)
				}
			}
			deviceState = verify.DeviceState
			verify.DeviceState = nil
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			position = deviceState.maxLayerTokenCount()
		}
		if verify.DeviceHidden != nil {
			hipReleaseForwardDeviceFinalHidden(&current)
			current.DeviceFinalHidden = verify.DeviceHidden
			current.DeviceFinalHiddenBorrowed = false
			verify.DeviceHidden = nil
		}
		current.Greedy = verify.NextGreedy
		current.GreedyDevice = nil
		if verify.AllAccepted || len(tokens) == maxTokens {
			carryLead = -1
			_ = verify.Close()
			continue
		}
		carryLead = int32(verify.Replacement.TokenID)
		if len(tokens) < maxTokens {
			tokens = append(tokens, carryLead)
		}
		_ = verify.Close()
	}
	_ = state
	return tokens
}

func runNativeGemma4Q4VerifierAcceptedBlocksAfterAttachedPrefillForHardwareTest(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, targetTokens []int32, blockTokens, compactPrefixTokens int, engineConfig hipGemma4Q4EngineConfig) []int32 {
	t.Helper()
	if len(targetTokens) == 0 {
		t.Fatalf("target tokens are required")
	}
	if blockTokens <= 0 {
		t.Fatalf("block tokens must be positive")
	}
	deviceKVMode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(promptTokens)+len(targetTokens)+1); err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("decode workspace: %v", err)
	}
	requestWorkspace := workspace
	if os.Getenv("GO_ROCM_ATTACHED_DRAFTER_DISABLE_WORKSPACE") == "1" {
		requestWorkspace = nil
	}
	workspace.EnsureProjectionGreedyBestCapacity(len(targetTokens) + 2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		_ = hipRecycleAttentionHeadsChunkedWorkspace(workspace)
		t.Fatalf("greedy buffer: %v", err)
	}
	defer func() {
		_ = greedyBuffer
		if err := hipRecycleAttentionHeadsChunkedWorkspace(workspace); err != nil {
			t.Fatalf("recycle workspace: %v", err)
		}
	}()
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	prefill, err := hipRunAttachedDrafterTargetPrefill(context.Background(), model.driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward:  cfg,
		DeviceKVMode:   deviceKVMode,
		EngineConfig:   engineConfig,
		InputTokenIDs:  promptTokens,
		Epsilon:        1e-6,
		SuppressTokens: suppressTokens,
		GreedyBuffer:   greedyBuffer,
		Workspace:      requestWorkspace,
	})
	if err != nil {
		t.Fatalf("attached target prefill: %v", err)
	}
	current := prefill.Current
	state := prefill.State
	deviceState := prefill.DeviceState
	position := prefill.Position
	defer func() {
		hipReleaseForwardDeviceFinalHidden(&current)
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	tokens := make([]int32, 0, len(targetTokens))
	for cursor := 0; cursor < len(targetTokens); {
		nextBlockTokens := blockTokens
		if cursor < compactPrefixTokens {
			nextBlockTokens = 1
		}
		end := cursor + nextBlockTokens
		if end > len(targetTokens) {
			end = len(targetTokens)
		}
		verifyTokens := targetTokens[cursor:end]
		if int32(current.Greedy.TokenID) != verifyTokens[0] {
			t.Fatalf("verifier block cursor %d current greedy=%d want %d", cursor, current.Greedy.TokenID, verifyTokens[0])
		}
		verifyGreedyBuffer := (*hipDeviceByteBuffer)(nil)
		if requestWorkspace != nil {
			verifyGreedyBuffer, err = workspace.BorrowProjectionGreedyBest(model.driver)
			if err != nil {
				t.Fatalf("verifier block cursor %d greedy buffer: %v", cursor, err)
			}
		}
		verify, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), model.driver, hipAttachedDrafterTargetVerifyBlockRequest{
			TargetForward:     cfg,
			DeviceKVMode:      deviceKVMode,
			EngineConfig:      engineConfig,
			TargetDeviceState: deviceState,
			CurrentGreedy:     current.Greedy,
			DraftTokens:       verifyTokens,
			Position:          position,
			Epsilon:           1e-6,
			SuppressTokens:    suppressTokens,
			GreedyBuffer:      verifyGreedyBuffer,
			Workspace:         requestWorkspace,
		})
		if err != nil {
			t.Fatalf("verifier block cursor %d: %v", cursor, err)
		}
		if !verify.AllAccepted || verify.AcceptedCount != len(verifyTokens) {
			verified := make([]int32, 0, len(verify.VerifiedGreedies))
			for _, greedy := range verify.VerifiedGreedies {
				verified = append(verified, int32(greedy.TokenID))
			}
			_ = verify.Close()
			t.Fatalf("verifier block cursor %d tokens=%v verified=%v replacement=%d accepted=%d all=%v want %d", cursor, verifyTokens, verified, verify.Replacement.TokenID, verify.AcceptedCount, verify.AllAccepted, len(verifyTokens))
		}
		tokens = append(tokens, verifyTokens...)
		if verify.DeviceState != nil {
			previousDeviceState := deviceState
			if !verify.PriorDeviceStateFinalized {
				if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, verify.DeviceState); err != nil {
					_ = verify.Close()
					t.Fatalf("verifier block cursor %d finalize state: %v", cursor, err)
				}
			}
			deviceState = verify.DeviceState
			verify.DeviceState = nil
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			position = deviceState.maxLayerTokenCount()
		}
		if verify.DeviceHidden != nil {
			hipReleaseForwardDeviceFinalHidden(&current)
			current.DeviceFinalHidden = verify.DeviceHidden
			current.DeviceFinalHiddenBorrowed = false
			verify.DeviceHidden = nil
		}
		current.Greedy = verify.NextGreedy
		current.GreedyDevice = nil
		_ = verify.Close()
		cursor = end
	}
	_ = state
	return tokens
}

func assertNativeAttachedDrafterTargetARMatchStable(t *testing.T, targetPath, draftPath string, targetROCmConfig, draftROCmConfig ROCmLoadConfig, prompt string, maxTokens, draftTokens int, nativeText, targetText string) {
	t.Helper()
	targetAgain := loadNativeAttachedDrafterReferenceText(t, targetPath, targetROCmConfig, prompt, maxTokens)
	if targetAgain != targetText {
		t.Skipf("reference target Generate(%q) shifted between runs (%q -> %q); attached-drafter equivalence comparison is not stable", prompt, targetText, targetAgain)
	}
	nativeAgain := loadNativeAttachedDrafterText(t, targetPath, draftPath, targetROCmConfig, draftROCmConfig, prompt, maxTokens, draftTokens)
	if nativeAgain != nativeText {
		t.Skipf("native attached Generate(%q) shifted between runs (%q -> %q); attached-drafter equivalence comparison is not stable", prompt, nativeText, nativeAgain)
	}
	t.Fatalf("native attached drafter text differs from stable target AR route: native=%q target=%q", nativeText, targetText)
}

func loadNativeAttachedDrafterReferenceText(t *testing.T, targetPath string, targetROCmConfig ROCmLoadConfig, prompt string, maxTokens int) string {
	t.Helper()
	referenceModel, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModelWithConfig(targetPath, targetROCmConfig, inference.WithContextLen(defaultContextLengthCap))
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

func loadNativeAttachedDrafterText(t *testing.T, targetPath, draftPath string, targetROCmConfig, draftROCmConfig ROCmLoadConfig, prompt string, maxTokens, draftTokens int) string {
	t.Helper()
	pair, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetOptions:    []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		DraftOptions:     []inference.LoadOption{inference.WithContextLen(defaultContextLengthCap)},
		TargetROCmConfig: targetROCmConfig,
		DraftROCmConfig:  draftROCmConfig,
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
	want := readLoadedBF16EmbeddingRows(t, model.driver, tensor, tokenIDs, hidden)
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
	if vocab <= 0 || hidden <= 0 || groupSize <= 0 || hidden%groupSize != 0 {
		t.Fatalf("loaded Gemma4 q%d dimensions vocab=%d hidden=%d group=%d are not a valid affine layout", bits, vocab, hidden, groupSize)
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
	wantWeights := readLoadedUint32EmbeddingRows(t, model.driver, weight, tokenIDs, packedPerRow)
	wantScales := readLoadedBF16TensorRowsByID(t, model.driver, scales, tokenIDs, groups)
	wantBiases := readLoadedBF16TensorRowsByID(t, model.driver, biases, tokenIDs, groups)
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

	bf16Weights := readLoadedBF16TensorRows(t, model.driver, tensor, 1, len(input))
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
	config, err := hipProjectionLaunchConfig(launch, spec.rows)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(model.driver, config))
	output, err := (&hipProjectionDeviceBuffers{Output: outputBuffer, Rows: spec.rows}).ReadOutput()
	core.RequireNoError(t, err)

	compareRows := 8
	if spec.rows < compareRows {
		compareRows = spec.rows
	}
	expectedWeights := readLoadedBF16TensorRows(t, model.driver, tensor, compareRows, spec.cols)
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
			wantRoPEBase, wantRoPERotaryDim, _ := model.loadedGemma4Q4LayerRoPE(layerCfg.LayerType, check.headDim)
			wantSlidingWindow := model.loadedGemma4Q4EffectiveSlidingWindow(layerCfg.LayerType, check.headDim)
			if layerCfg.HeadDim != check.headDim ||
				layerCfg.QueryHeads != 8 ||
				layerCfg.IntermediateSize != check.intermediate ||
				layerCfg.RoPEBase != wantRoPEBase ||
				layerCfg.RoPERotaryDim != wantRoPERotaryDim ||
				layerCfg.SlidingWindow != wantSlidingWindow {
				t.Fatalf("Gemma4 q4 layer %d config head=%d qheads=%d intermediate=%d rope=%f rotary=%d sliding=%d, want head=%d qheads=8 intermediate=%d rope=%f rotary=%d sliding=%d",
					check.layer, layerCfg.HeadDim, layerCfg.QueryHeads, layerCfg.IntermediateSize, layerCfg.RoPEBase, layerCfg.RoPERotaryDim, layerCfg.SlidingWindow, check.headDim, check.intermediate, wantRoPEBase, wantRoPERotaryDim, wantSlidingWindow)
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
	if resettable, ok := any(textModel).(interface{ ResetState() error }); ok {
		core.RequireNoError(t, resettable.ResetState())
	}
	chatMessages := []inference.Message{{Role: "user", Content: "Hi"}}
	chatPrompt := formatGemma4ChatTemplateWithConfig(chatMessages, loaded.gemma4ChatTemplateConfig(inference.GenerateConfig{}, false))
	if rocmChatModel, ok := any(textModel).(*rocmModel); ok {
		chatPrompt = formatGemma4ChatTemplateWithConfig(chatMessages, rocmChatModel.gemma4ChatTemplateConfig(inference.GenerateConfig{}, false))
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
	if resettable, ok := any(textModel).(interface{ ResetState() error }); ok {
		core.RequireNoError(t, resettable.ResetState())
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
		speculative.Metrics.AcceptedTokens+speculative.Metrics.RejectedTokens != 1 ||
		len(speculative.Tokens) != 1 ||
		speculative.Tokens[0].ID < 0 ||
		int(speculative.Tokens[0].ID) >= loaded.modelInfo.VocabSize {
		t.Fatalf("Gemma4 q4 public SpeculativeDecode = %+v, want one proposed/emitted in-vocab token", speculative)
	}
	if resettable, ok := any(textModel).(interface{ ResetState() error }); ok {
		core.RequireNoError(t, resettable.ResetState())
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
	wantWeights := readLoadedUint32TensorRows(t, model.driver, weight, compareRows, packedPerRow)
	wantScales := readLoadedBF16TensorRows(t, model.driver, scales, compareRows, groups)
	wantBiases := readLoadedBF16TensorRows(t, model.driver, biases, compareRows, groups)
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

func readLoadedBF16TensorRows(t *testing.T, driver nativeHIPDriver, tensor hipTensor, rows, cols int) []uint16 {
	t.Helper()
	payload := readLoadedTensorBytes(t, driver, tensor, 0, rows*cols*2)
	values := make([]uint16, rows*cols)
	for index := range values {
		values[index] = binary.LittleEndian.Uint16(payload[index*2:])
	}
	return values
}

func readLoadedUint32TensorRows(t *testing.T, driver nativeHIPDriver, tensor hipTensor, rows, cols int) []uint32 {
	t.Helper()
	payload := readLoadedTensorBytes(t, driver, tensor, 0, rows*cols*4)
	values := make([]uint32, rows*cols)
	for index := range values {
		values[index] = binary.LittleEndian.Uint32(payload[index*4:])
	}
	return values
}

func readLoadedUint32EmbeddingRows(t *testing.T, driver nativeHIPDriver, tensor hipTensor, tokenIDs []int32, cols int) []uint32 {
	t.Helper()
	rowBytes := cols * 4
	values := make([]uint32, 0, len(tokenIDs)*cols)
	for _, id := range tokenIDs {
		if id < 0 {
			t.Fatalf("token ID %d is negative", id)
		}
		payload := readLoadedTensorBytes(t, driver, tensor, int64(id)*int64(rowBytes), rowBytes)
		for index := 0; index < cols; index++ {
			values = append(values, binary.LittleEndian.Uint32(payload[index*4:]))
		}
	}
	return values
}

func readLoadedBF16TensorRowsByID(t *testing.T, driver nativeHIPDriver, tensor hipTensor, tokenIDs []int32, cols int) []uint16 {
	t.Helper()
	rowBytes := cols * 2
	values := make([]uint16, 0, len(tokenIDs)*cols)
	for _, id := range tokenIDs {
		if id < 0 {
			t.Fatalf("token ID %d is negative", id)
		}
		payload := readLoadedTensorBytes(t, driver, tensor, int64(id)*int64(rowBytes), rowBytes)
		for index := 0; index < cols; index++ {
			values = append(values, binary.LittleEndian.Uint16(payload[index*2:]))
		}
	}
	return values
}

func readLoadedBF16EmbeddingRows(t *testing.T, driver nativeHIPDriver, tensor hipTensor, tokenIDs []int32, hidden int) []float32 {
	t.Helper()
	rowBytes := hidden * 2
	values := make([]float32, 0, len(tokenIDs)*hidden)
	for _, id := range tokenIDs {
		if id < 0 {
			t.Fatalf("token ID %d is negative", id)
		}
		payload := readLoadedTensorBytes(t, driver, tensor, int64(id)*int64(rowBytes), rowBytes)
		for index := 0; index < hidden; index++ {
			values = append(values, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:])))
		}
	}
	return values
}

func readLoadedTensorBytes(t *testing.T, driver nativeHIPDriver, tensor hipTensor, relativeOffset int64, size int) []byte {
	t.Helper()
	if relativeOffset < 0 || size < 0 || uint64(relativeOffset)+uint64(size) > tensor.info.ByteSize {
		t.Fatalf("read tensor %s offset=%d size=%d exceeds %d bytes", tensor.info.Name, relativeOffset, size, tensor.info.ByteSize)
	}
	payload := make([]byte, size)
	if sourcePath := tensor.info.SourcePath; sourcePath != "" {
		file, err := os.Open(sourcePath)
		core.RequireNoError(t, err)
		defer file.Close()
		start := tensor.info.DataOffset + int64(tensor.info.Offset) + relativeOffset
		n, err := file.ReadAt(payload, start)
		if err != nil || n != len(payload) {
			t.Fatalf("read tensor %s from %s at %d: n=%d err=%v", tensor.info.Name, sourcePath, start, n, err)
		}
		return payload
	}
	if driver == nil || tensor.pointer == 0 {
		t.Fatalf("loaded tensor %s has neither source path nor device allocation", tensor.info.Name)
	}
	if err := driver.CopyDeviceToHost(tensor.pointer+nativeDevicePointer(relativeOffset), payload); err != nil {
		t.Fatalf("read tensor %s from device at offset %d: %v", tensor.info.Name, relativeOffset, err)
	}
	return payload
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
	config, err := hipProjectionLaunchConfig(launchBytes, req.Rows)
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
	q8Config, err := hipProjectionLaunchConfig(q8LaunchBytes, q8Req.Rows)
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
	bf16Config, err := hipProjectionLaunchConfig(bf16LaunchBytes, bf16Req.Rows)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(hipRuntime.driver, bf16Config))
	bf16Output, err := bf16Buffers.ReadOutput()
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.75, 1.5}, bf16Output, 0.0001)

	t.Run("f32-router-shape", func(t *testing.T) {
		const rows = 128
		const cols = 2816
		input := make([]float32, cols)
		weights := make([]float32, rows*cols)
		for col := range input {
			input[col] = float32(col%29-14) / 17
		}
		for index := range weights {
			weights[index] = float32(index%31-15) / 19
		}
		want, err := hipReferenceF32Projection(input, weights, rows, cols, nil)
		core.RequireNoError(t, err)
		got, err := hipRunProjectionKernel(context.Background(), hipRuntime.driver, hipProjectionRequest{
			Input: input,
			F32:   weights,
			Rows:  rows,
			Cols:  cols,
		})
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, want, got, 0.001)
	})

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

		const q4G64Rows, q4G64Cols, q4G64Batch = 2, 64, 16
		q4G64Input := make([]float32, q4G64Cols)
		q4G64Values := make([]uint32, q4G64Rows*q4G64Cols)
		for index := range q4G64Input {
			q4G64Input[index] = float32(index%11-5) / 8
		}
		for index := range q4G64Values {
			q4G64Values[index] = uint32((index*7 + 3) % 16)
		}
		q4G64Req := hipMLXQ4ProjectionRequest{
			Input:     q4G64Input,
			Weight:    hipPackMLXAffineValuesForTest(q4G64Values, q4G64Cols, 4),
			Scales:    []uint16{hipFloat32ToBFloat16(0.25), hipFloat32ToBFloat16(0.5)},
			Biases:    []uint16{hipFloat32ToBFloat16(-0.5), hipFloat32ToBFloat16(0.25)},
			Rows:      q4G64Rows,
			Cols:      q4G64Cols,
			GroupSize: q4G64Cols,
		}
		q4G64Buffers, err := q4G64Req.deviceBuffers(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer q4G64Buffers.Close()
		q4G64BatchInput := make([]float32, 0, q4G64Batch*q4G64Cols)
		q4G64Want := make([]float32, 0, q4G64Batch*q4G64Rows)
		for batch := 0; batch < q4G64Batch; batch++ {
			factor := float32(batch+1) / q4G64Batch
			tokenInput := make([]float32, q4G64Cols)
			for index, value := range q4G64Input {
				tokenInput[index] = value * factor
			}
			q4G64BatchInput = append(q4G64BatchInput, tokenInput...)
			tokenWant, err := hipReferenceMLXQ4Projection(tokenInput, q4G64Req.Weight, q4G64Req.Scales, q4G64Req.Biases, q4G64Rows, q4G64Cols, q4G64Cols)
			core.RequireNoError(t, err)
			q4G64Want = append(q4G64Want, tokenWant...)
		}
		q4G64BatchPayload, err := hipFloat32Payload(q4G64BatchInput)
		core.RequireNoError(t, err)
		q4G64BatchBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 group64 tokens16 input", q4G64BatchPayload, len(q4G64BatchInput))
		core.RequireNoError(t, err)
		defer q4G64BatchBuffer.Close()
		q4G64OutputBuffer, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, q4G64BatchBuffer, hipMLXQ4DeviceWeightConfig{
			WeightPointer: q4G64Buffers.Weight.Pointer(),
			ScalePointer:  q4G64Buffers.Scales.Pointer(),
			BiasPointer:   q4G64Buffers.Biases.Pointer(),
			WeightBytes:   q4G64Buffers.Weight.SizeBytes(),
			ScaleBytes:    q4G64Buffers.Scales.SizeBytes(),
			BiasBytes:     q4G64Buffers.Biases.SizeBytes(),
			Rows:          q4G64Rows,
			Cols:          q4G64Cols,
			GroupSize:     q4G64Cols,
		}, q4G64Batch)
		core.RequireNoError(t, err)
		defer q4G64OutputBuffer.Close()
		q4G64Output, err := hipReadFloat32DeviceOutput(q4G64OutputBuffer, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 group64 tokens16 output", len(q4G64Want))
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, q4G64Want, q4G64Output, 0.001)

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

		const q8G64Rows, q8G64Cols, q8G64Batch = 2, 64, 32
		q8G64Input := make([]float32, q8G64Cols)
		q8G64Values := make([]uint32, q8G64Rows*q8G64Cols)
		for index := range q8G64Input {
			q8G64Input[index] = float32(index%13-6) / 16
		}
		for index := range q8G64Values {
			q8G64Values[index] = uint32((index*29 + 11) % 256)
		}
		q8G64Req := hipMLXQ4ProjectionRequest{
			Input:     q8G64Input,
			Weight:    hipPackMLXAffineValuesForTest(q8G64Values, q8G64Cols, 8),
			Scales:    []uint16{hipFloat32ToBFloat16(0.03125), hipFloat32ToBFloat16(0.0625)},
			Biases:    []uint16{hipFloat32ToBFloat16(-1), hipFloat32ToBFloat16(0.5)},
			Rows:      q8G64Rows,
			Cols:      q8G64Cols,
			GroupSize: q8G64Cols,
			Bits:      8,
		}
		q8G64Buffers, err := q8G64Req.deviceBuffers(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer q8G64Buffers.Close()
		q8G64BatchInput := make([]float32, 0, q8G64Batch*q8G64Cols)
		q8G64Want := make([]float32, 0, q8G64Batch*q8G64Rows)
		for batch := 0; batch < q8G64Batch; batch++ {
			factor := float32(batch+1) / q8G64Batch
			tokenInput := make([]float32, q8G64Cols)
			for index, value := range q8G64Input {
				tokenInput[index] = value * factor
			}
			q8G64BatchInput = append(q8G64BatchInput, tokenInput...)
			tokenWant, err := hipReferenceMLXAffineProjection(tokenInput, q8G64Req.Weight, q8G64Req.Scales, q8G64Req.Biases, q8G64Rows, q8G64Cols, q8G64Cols, 8)
			core.RequireNoError(t, err)
			q8G64Want = append(q8G64Want, tokenWant...)
		}
		q8G64BatchPayload, err := hipFloat32Payload(q8G64BatchInput)
		core.RequireNoError(t, err)
		q8G64BatchBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q8 group64 tokens16 input", q8G64BatchPayload, len(q8G64BatchInput))
		core.RequireNoError(t, err)
		defer q8G64BatchBuffer.Close()
		q8G64OutputBuffer, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), hipRuntime.driver, q8G64BatchBuffer, hipMLXQ4DeviceWeightConfig{
			WeightPointer: q8G64Buffers.Weight.Pointer(),
			ScalePointer:  q8G64Buffers.Scales.Pointer(),
			BiasPointer:   q8G64Buffers.Biases.Pointer(),
			WeightBytes:   q8G64Buffers.Weight.SizeBytes(),
			ScaleBytes:    q8G64Buffers.Scales.SizeBytes(),
			BiasBytes:     q8G64Buffers.Biases.SizeBytes(),
			Rows:          q8G64Rows,
			Cols:          q8G64Cols,
			GroupSize:     q8G64Cols,
			Bits:          8,
		}, q8G64Batch)
		core.RequireNoError(t, err)
		defer q8G64OutputBuffer.Close()
		q8G64Output, err := hipReadFloat32DeviceOutput(q8G64OutputBuffer, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q8 group64 tokens16 output", len(q8G64Want))
		core.RequireNoError(t, err)
		assertFloat32SlicesNear(t, q8G64Want, q8G64Output, 0.001)

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

func TestHIPHardwareAttentionHeadsBatchChunkedGQASharedMatchesV2_Good(t *testing.T) {
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
	previousGQA2 := hipAttentionHeadsBatchChunkedGQA2Enabled
	previousGQA4 := hipAttentionHeadsBatchChunkedGQA4Enabled
	t.Cleanup(func() {
		hipAttentionHeadsBatchChunkedGQA2Enabled = previousGQA2
		hipAttentionHeadsBatchChunkedGQA4Enabled = previousGQA4
	})

	cases := []struct {
		name       string
		dim        int
		tokenCount int
		headCount  int
		keyHeads   int
		queryCount int
		blockSize  int
		windowSize int
	}{
		{
			name:       "descriptor-pages-e2b-global-dim512-gqa4",
			dim:        512,
			tokenCount: hipAttentionHeadsBatchChunkedGQA4MinChunks*hipAttentionHeadsDefaultChunkSize + 5,
			headCount:  8,
			keyHeads:   1,
			queryCount: 2,
			blockSize:  hipGemma4Q4DeviceKVBlockSize(),
		},
		{
			name:       "descriptor-pages-12b-global-dim512-gqa2",
			dim:        512,
			tokenCount: hipAttentionHeadsChunkedBlockSize + 5,
			headCount:  16,
			keyHeads:   8,
			queryCount: 2,
			blockSize:  hipGemma4Q4DeviceKVBlockSize(),
		},
		{
			name:       "direct-pages-windowed-multi-kv",
			dim:        64,
			tokenCount: hipAttentionHeadsChunkedBlockSize + 5,
			headCount:  4,
			keyHeads:   2,
			queryCount: 2,
			blockSize:  1,
			windowSize: hipAttentionHeadsChunkedBlockSize,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			queryStartToken := tc.tokenCount - tc.queryCount
			kvWidth := tc.keyHeads * tc.dim
			keyValues := make([]float32, tc.tokenCount*kvWidth)
			valueValues := make([]float32, tc.tokenCount*kvWidth)
			for token := 0; token < tc.tokenCount; token++ {
				for kvHead := 0; kvHead < tc.keyHeads; kvHead++ {
					for dimIndex := 0; dimIndex < tc.dim; dimIndex++ {
						index := (token*tc.keyHeads+kvHead)*tc.dim + dimIndex
						keyValues[index] = float32(math.Sin(float64(token*3+kvHead*17+dimIndex)*0.007) * 0.25)
						valueValues[index] = float32(math.Cos(float64(token*5+kvHead*23+dimIndex*2)*0.009) * 0.5)
					}
				}
			}
			queryValues := make([]float32, tc.queryCount*tc.headCount*tc.dim)
			for index := range queryValues {
				queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.125)
			}

			cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, tc.blockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, cache.AppendVectors(0, kvWidth, kvWidth, keyValues, valueValues))
			deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
			core.RequireNoError(t, err)
			defer deviceKV.Close()
			table, err := deviceKV.KernelDescriptorTable()
			core.RequireNoError(t, err)
			defer table.Close()
			queryPayload, err := hipFloat32Payload(queryValues)
			core.RequireNoError(t, err)
			const operation = "rocm.hip.AttentionHeadsBatchChunkedGQAShared"
			query, err := hipUploadByteBuffer(hipRuntime.driver, operation, "GQA shared hardware query", queryPayload, len(queryValues))
			core.RequireNoError(t, err)
			defer query.Close()

			req := hipAttentionHeadsBatchCausalDeviceRequest{
				DeviceKV:        deviceKV,
				DescriptorTable: table,
				Dim:             tc.dim,
				TokenCount:      tc.tokenCount,
				HeadCount:       tc.headCount,
				KeyHeads:        tc.keyHeads,
				QueryCount:      tc.queryCount,
				QueryStartToken: queryStartToken,
				WindowSize:      tc.windowSize,
				Scale:           1,
			}
			run := func(enableGQA2, enableGQA4 bool, label string) []float32 {
				hipAttentionHeadsBatchChunkedGQA2Enabled = enableGQA2
				hipAttentionHeadsBatchChunkedGQA4Enabled = enableGQA4
				output, err := hipAllocateByteBuffer(hipRuntime.driver, operation, label, uint64(len(queryValues)*4), len(queryValues))
				core.RequireNoError(t, err)
				defer output.Close()
				workspace := &hipAttentionHeadsChunkedWorkspace{}
				defer workspace.Close()
				core.RequireNoError(t, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(context.Background(), hipRuntime.driver, req, query, output, workspace))
				got, err := hipReadFloat32DeviceOutput(output, operation, label, len(queryValues))
				core.RequireNoError(t, err)
				return got
			}
			v2 := run(false, false, "v2 hardware output")
			gqa2 := run(true, false, "GQA2 hardware output")
			assertFloat32SlicesNear(t, v2, gqa2, 0.0001)
			candidate := gqa2
			if tc.headCount%tc.keyHeads == 0 && (tc.headCount/tc.keyHeads)%4 == 0 {
				gqa4 := run(false, true, "GQA4 hardware output")
				assertFloat32SlicesNear(t, v2, gqa4, 0.0001)
				candidate = gqa4
			}

			restoredKeys, restoredValues, err := cache.Restore(0, tc.tokenCount)
			core.RequireNoError(t, err)
			want := make([]float32, 0, len(queryValues))
			for queryIndex := 0; queryIndex < tc.queryCount; queryIndex++ {
				visibleTokens := queryStartToken + queryIndex + 1
				windowStart := 0
				if tc.windowSize > 0 && visibleTokens > tc.windowSize {
					windowStart = visibleTokens - tc.windowSize
				}
				for head := 0; head < tc.headCount; head++ {
					keys, err := fakeROCmAttentionHeadVectors(restoredKeys, tc.tokenCount, tc.keyHeads, tc.dim, tc.headCount, head)
					core.RequireNoError(t, err)
					values, err := fakeROCmAttentionHeadVectors(restoredValues, tc.tokenCount, tc.keyHeads, tc.dim, tc.headCount, head)
					core.RequireNoError(t, err)
					queryOffset := (queryIndex*tc.headCount + head) * tc.dim
					headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryOffset:queryOffset+tc.dim], keys[windowStart:visibleTokens], values[windowStart:visibleTokens], 1)
					core.RequireNoError(t, err)
					want = append(want, headOutput...)
				}
			}
			assertFloat32SlicesNear(t, want, candidate, 0.005)
		})
	}
}

func TestHIPHardwareAttentionHeadsIncrementalGQA2MatchesPerHead_Good(t *testing.T) {
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
	previousIncremental := hipAttentionHeadsIncrementalGQA2Enabled
	previousGQA2 := hipAttentionHeadsBatchChunkedGQA2Enabled
	hipAttentionHeadsIncrementalGQA2Enabled = true
	hipAttentionHeadsBatchChunkedGQA2Enabled = true
	t.Cleanup(func() {
		hipAttentionHeadsIncrementalGQA2Enabled = previousIncremental
		hipAttentionHeadsBatchChunkedGQA2Enabled = previousGQA2
	})

	const (
		dim       = 512
		headCount = 8
		keyHeads  = 4
	)
	for _, tc := range []struct {
		name       string
		tokenCount int
		windowSize int
	}{
		{name: "sliding-1k", tokenCount: 1024, windowSize: 1024},
		{name: "global-2k", tokenCount: 2050},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kvWidth := keyHeads * dim
			queryValues := make([]float32, headCount*dim)
			keyValues := make([]float32, tc.tokenCount*kvWidth)
			valueValues := make([]float32, tc.tokenCount*kvWidth)
			for index := range queryValues {
				queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.125)
			}
			for index := range keyValues {
				keyValues[index] = float32(math.Sin(float64(index)*0.007) * 0.25)
				valueValues[index] = float32(math.Cos(float64(index)*0.009) * 0.5)
			}

			cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 1)
			core.RequireNoError(t, err)
			core.RequireNoError(t, cache.AppendVectors(0, kvWidth, kvWidth, keyValues, valueValues))
			deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
			core.RequireNoError(t, err)
			defer deviceKV.Close()
			table, err := deviceKV.KernelDescriptorTable()
			core.RequireNoError(t, err)
			defer table.Close()
			queryPayload, err := hipFloat32Payload(queryValues)
			core.RequireNoError(t, err)
			const operation = "rocm.hip.AttentionHeadsIncrementalGQA2"
			query, err := hipUploadByteBuffer(hipRuntime.driver, operation, "incremental GQA2 query", queryPayload, len(queryValues))
			core.RequireNoError(t, err)
			defer query.Close()
			perHeadOutput, err := hipAllocateByteBuffer(hipRuntime.driver, operation, "per-head output", uint64(len(queryValues)*4), len(queryValues))
			core.RequireNoError(t, err)
			defer perHeadOutput.Close()
			sharedOutput, err := hipAllocateByteBuffer(hipRuntime.driver, operation, "shared output", uint64(len(queryValues)*4), len(queryValues))
			core.RequireNoError(t, err)
			defer sharedOutput.Close()

			req := hipAttentionRequest{
				QueryDim:        dim,
				KeyHeads:        keyHeads,
				DeviceKV:        deviceKV,
				DescriptorTable: table,
				Scale:           1,
				WindowSize:      tc.windowSize,
			}
			core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, req, query, headCount, perHeadOutput))
			workspace := &hipAttentionHeadsChunkedWorkspace{}
			defer workspace.Close()
			core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernelWithWorkspace(context.Background(), hipRuntime.driver, req, query, headCount, sharedOutput, workspace))
			perHead, err := hipReadFloat32DeviceOutput(perHeadOutput, operation, "per-head output", len(queryValues))
			core.RequireNoError(t, err)
			shared, err := hipReadFloat32DeviceOutput(sharedOutput, operation, "shared output", len(queryValues))
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, perHead, shared, 0.0001)
		})
	}
}

func BenchmarkHIPHardwareAttentionHeadsBatchChunked_E2B32K(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_HIP_ATTENTION_BENCHMARK") != "1" {
		b.Skip("set GO_ROCM_RUN_HIP_ATTENTION_BENCHMARK=1 to run the deep attention benchmark")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		b.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernels/rocm_kernels.hip HSACO")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		b.Fatal("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		b.Fatalf("runtime = %T, want HIP runtime with driver", runtime)
	}

	const (
		dim        = 512
		tokenCount = 32 * 1024
		headCount  = 8
		keyHeads   = 1
	)
	kvWidth := keyHeads * dim
	keyValues := make([]float32, tokenCount*kvWidth)
	valueValues := make([]float32, tokenCount*kvWidth)
	for token := 0; token < tokenCount; token++ {
		for dimIndex := 0; dimIndex < dim; dimIndex++ {
			index := token*dim + dimIndex
			keyValues[index] = float32(math.Sin(float64(token*3+dimIndex)*0.007) * 0.25)
			valueValues[index] = float32(math.Cos(float64(token*5+dimIndex*2)*0.009) * 0.5)
		}
	}
	queryValues := make([]float32, headCount*dim)
	for index := range queryValues {
		queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.125)
	}

	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, hipGemma4Q4DeviceKVBlockSize())
	core.RequireNoError(b, err)
	core.RequireNoError(b, cache.AppendVectors(0, kvWidth, kvWidth, keyValues, valueValues))
	deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
	core.RequireNoError(b, err)
	defer deviceKV.Close()
	table, err := deviceKV.KernelDescriptorTable()
	core.RequireNoError(b, err)
	defer table.Close()
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(b, err)
	query, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchChunkedBenchmark", "deep attention query", queryPayload, len(queryValues))
	core.RequireNoError(b, err)
	defer query.Close()
	output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchChunkedBenchmark", "deep attention output", uint64(len(queryValues)*4), len(queryValues))
	core.RequireNoError(b, err)
	defer output.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	req := hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        deviceKV,
		DescriptorTable: table,
		Dim:             dim,
		TokenCount:      tokenCount,
		HeadCount:       headCount,
		KeyHeads:        keyHeads,
		QueryCount:      1,
		QueryStartToken: tokenCount - 1,
		Scale:           1,
	}
	core.RequireNoError(b, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(context.Background(), hipRuntime.driver, req, query, output, workspace))
	_, err = hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchChunkedBenchmark", "deep attention warmup output", len(queryValues))
	core.RequireNoError(b, err)

	b.ReportAllocs()
	b.ReportMetric(tokenCount, "context_tokens/op")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		core.RequireNoError(b, hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(context.Background(), hipRuntime.driver, req, query, output, workspace))
	}
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchChunkedBenchmark", "deep attention timed output", len(queryValues))
	core.RequireNoError(b, err)
	if len(got) != len(queryValues) || math.IsNaN(float64(got[0])) {
		b.Fatalf("deep attention output is invalid: len=%d first=%v", len(got), got[0])
	}
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

	t.Run("attention-heads-chunked-multi-kv", func(t *testing.T) {
		const (
			dim        = 64
			tokenCount = 320
			headCount  = 4
			keyHeads   = 2
		)
		for _, blockSize := range []int{1, 16} {
			t.Run(core.Sprintf("block%d", blockSize), func(t *testing.T) {
				queryValues := make([]float32, headCount*dim)
				keyValues := make([]float32, tokenCount*keyHeads*dim)
				valueValues := make([]float32, tokenCount*keyHeads*dim)
				for index := range queryValues {
					queryValues[index] = float32(math.Sin(float64(index)*0.013) * 0.75)
				}
				for token := 0; token < tokenCount; token++ {
					for kvHead := 0; kvHead < keyHeads; kvHead++ {
						for dimIndex := 0; dimIndex < dim; dimIndex++ {
							index := (token*keyHeads+kvHead)*dim + dimIndex
							keyValues[index] = float32(math.Sin(float64(token+kvHead*7+dimIndex)*0.017) * 0.5)
							valueValues[index] = float32(math.Cos(float64(token+kvHead*11+dimIndex*2)*0.019) * 0.5)
						}
					}
				}
				cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, blockSize)
				core.RequireNoError(t, err)
				core.RequireNoError(t, cache.AppendVectors(0, keyHeads*dim, keyHeads*dim, keyValues, valueValues))
				deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
				core.RequireNoError(t, err)
				defer deviceKV.Close()
				table, err := deviceKV.KernelDescriptorTable()
				core.RequireNoError(t, err)
				defer table.Close()
				queryPayload, err := hipFloat32Payload(queryValues)
				core.RequireNoError(t, err)
				queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware multi-KV chunked attention query", queryPayload, len(queryValues))
				core.RequireNoError(t, err)
				defer queryBuffer.Close()
				normalOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsLaunch", "hardware multi-KV normal attention output", uint64(len(queryValues)*4), len(queryValues))
				core.RequireNoError(t, err)
				defer normalOutput.Close()
				chunkedOutput, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware multi-KV chunked attention output", uint64(len(queryValues)*4), len(queryValues))
				core.RequireNoError(t, err)
				defer chunkedOutput.Close()
				req := hipAttentionRequest{
					QueryDim:        dim,
					KeyHeads:        keyHeads,
					DeviceKV:        deviceKV,
					DescriptorTable: table,
					Scale:           1,
				}
				core.RequireNoError(t, hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, normalOutput))
				workspace := &hipAttentionHeadsChunkedWorkspace{}
				defer workspace.Close()
				core.RequireNoError(t, hipRunAttentionHeadsChunked(context.Background(), hipRuntime.driver, req, queryBuffer, headCount, dim, tokenCount, chunkedOutput, workspace))
				normalGot, err := hipReadFloat32DeviceOutput(normalOutput, "rocm.hip.AttentionHeadsLaunch", "hardware multi-KV normal attention output", len(queryValues))
				core.RequireNoError(t, err)
				chunkedGot, err := hipReadFloat32DeviceOutput(chunkedOutput, "rocm.hip.AttentionHeadsChunkedLaunch", "hardware multi-KV chunked attention output", len(queryValues))
				core.RequireNoError(t, err)

				restoredKeys, restoredValues, err := cache.Restore(0, cache.TokenCount())
				core.RequireNoError(t, err)
				want := make([]float32, 0, len(queryValues))
				for head := 0; head < headCount; head++ {
					keys, err := fakeROCmAttentionHeadVectors(restoredKeys, tokenCount, keyHeads, dim, headCount, head)
					core.RequireNoError(t, err)
					values, err := fakeROCmAttentionHeadVectors(restoredValues, tokenCount, keyHeads, dim, headCount, head)
					core.RequireNoError(t, err)
					headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[head*dim:(head+1)*dim], keys, values, 1)
					core.RequireNoError(t, err)
					want = append(want, headOutput...)
				}
				assertFloat32SlicesNear(t, want, normalGot, 0.005)
				assertFloat32SlicesNear(t, want, chunkedGot, 0.005)
				assertFloat32SlicesNear(t, normalGot, chunkedGot, 0.005)
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

	t.Run("attention-heads-batch-chunked-multi-kv", func(t *testing.T) {
		const (
			dim             = 4
			tokenCount      = hipAttentionHeadsSharedMaxTokens + 3
			headCount       = 4
			keyHeads        = 2
			queryCount      = 1
			queryStartToken = tokenCount - 1
		)
		queryValues := make([]float32, headCount*dim)
		for index := range queryValues {
			queryValues[index] = float32(math.Sin(float64(index)*0.13) * 0.75)
		}
		keyValues := make([]float32, tokenCount*keyHeads*dim)
		valueValues := make([]float32, tokenCount*keyHeads*dim)
		for token := 0; token < tokenCount; token++ {
			for kvHead := 0; kvHead < keyHeads; kvHead++ {
				for dimIndex := 0; dimIndex < dim; dimIndex++ {
					index := (token*keyHeads+kvHead)*dim + dimIndex
					keyValues[index] = float32(math.Sin(float64(token+kvHead*7+dimIndex)*0.017) * 0.5)
					valueValues[index] = float32(math.Cos(float64(token+kvHead*11+dimIndex*2)*0.019) * 0.5)
				}
			}
		}
		cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, hipGemma4Q4DeviceKVBlockSize())
		core.RequireNoError(t, err)
		core.RequireNoError(t, cache.AppendVectors(0, keyHeads*dim, keyHeads*dim, keyValues, valueValues))
		deviceKV, err := cache.MirrorToDevice(hipRuntime.driver)
		core.RequireNoError(t, err)
		defer deviceKV.Close()
		table, err := deviceKV.KernelDescriptorTable()
		core.RequireNoError(t, err)
		defer table.Close()
		queryPayload, err := hipFloat32Payload(queryValues)
		core.RequireNoError(t, err)
		queryBuffer, err := hipUploadByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchChunkedLaunch", "hardware multi-KV batch chunked attention query", queryPayload, len(queryValues))
		core.RequireNoError(t, err)
		defer queryBuffer.Close()
		output, err := hipAllocateByteBuffer(hipRuntime.driver, "rocm.hip.AttentionHeadsBatchChunkedLaunch", "hardware multi-KV batch chunked attention output", uint64(len(queryValues)*4), len(queryValues))
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
			KeyHeads:        keyHeads,
			QueryCount:      queryCount,
			QueryStartToken: queryStartToken,
			Scale:           1,
		}, queryBuffer, output, workspace))
		got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchChunkedLaunch", "hardware multi-KV batch chunked attention output", len(queryValues))
		core.RequireNoError(t, err)
		restoredKeys, restoredValues, err := cache.Restore(0, cache.TokenCount())
		core.RequireNoError(t, err)
		want := make([]float32, 0, len(queryValues))
		for head := 0; head < headCount; head++ {
			keys, err := fakeROCmAttentionHeadVectors(restoredKeys, tokenCount, keyHeads, dim, headCount, head)
			core.RequireNoError(t, err)
			values, err := fakeROCmAttentionHeadVectors(restoredValues, tokenCount, keyHeads, dim, headCount, head)
			core.RequireNoError(t, err)
			headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[head*dim:(head+1)*dim], keys, values, 1)
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

func decodeTokenIDs(tokens []inferdecode.Token) []int32 {
	ids := make([]int32, 0, len(tokens))
	for _, token := range tokens {
		ids = append(ids, token.ID)
	}
	return ids
}

func collectInferenceTokens(stream func(func(inference.Token) bool)) []inference.Token {
	tokens := []inference.Token{}
	for token := range stream {
		tokens = append(tokens, token)
	}
	return tokens
}

func inferenceTokenIDs(tokens []inference.Token) []int32 {
	ids := make([]int32, 0, len(tokens))
	for _, token := range tokens {
		ids = append(ids, token.ID)
	}
	return ids
}

func inferenceTokenText(tokens []inference.Token) []string {
	text := make([]string, 0, len(tokens))
	for _, token := range tokens {
		text = append(text, token.Text)
	}
	return text
}
