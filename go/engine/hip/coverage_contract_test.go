// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/provider/anthropic"
	"dappco.re/go/inference/provider/ollama"
	openaicompat "dappco.re/go/inference/provider/openai"
)

func TestCoverage_NativeFallbackHelpers_Good(t *testing.T) {
	core.AssertEqual(t, 2, approximatePromptTokens(" alpha beta "))
	core.AssertEqual(t, 3, approximatePromptsTokens([]string{"one two", "three"}))
	core.AssertEqual(t, 3, approximateMessageTokens([]inference.Message{
		{Role: "user", Content: "hello world"},
		{Role: "assistant", Content: "ok"},
	}))
	core.AssertEqual(t, []int32(nil), approximateTokenIDs("   "))
	core.AssertEqual(t, []int32{1, 2}, approximateTokenIDs("hello core"))
	core.AssertEqual(t, "user: hello\nassistant: ok\n", formatFallbackChatTemplate([]inference.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "ok"},
	}))
	core.AssertEqual(t, "<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", formatGemma4ChatTemplate([]inference.Message{
		{Role: "user", Content: " hello "},
	}))
	core.AssertEqual(t, "<bos><|turn>system\nbe concise<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", formatGemma4ChatTemplate([]inference.Message{
		{Role: "developer", Content: " be concise "},
		{Role: "user", Content: "hello"},
	}))
	core.AssertEqual(t, "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n", formatGemma4ChatTemplateWithConfig([]inference.Message{
		{Role: "user", Content: "hello"},
	}, gemma4ChatTemplateConfig{EnableThinking: true}))
	core.AssertEqual(t, "<turn|>\n<|turn>user\nnext<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", formatGemma4ChatTemplateWithConfig([]inference.Message{
		{Role: "user", Content: "next"},
	}, gemma4ChatTemplateConfig{Continuation: true, LargeVariant: true}))
	core.AssertEqual(t, "<bos><|turn>user\nhi<turn|>\n<|turn>model\nvisible<turn|>\n", formatGemma4ChatTemplateWithConfig([]inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "<|channel>thought\nprivate<channel|>visible"},
	}, gemma4ChatTemplateConfig{NoGenerationPrompt: true}))
	core.AssertEqual(t, "<bos><|turn>user\nhi<turn|>\n<|turn>model\none<turn|>\ntwo<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", formatGemma4ChatTemplateWithConfig([]inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "one"},
		{Role: "assistant", Content: "two"},
	}, gemma4ChatTemplateConfig{}))
	templateCfg := gemma4ChatTemplateConfigForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels:       map[string]string{"gemma4_size": "31B"},
	}, inference.GenerateConfig{}, true)
	core.AssertTrue(t, templateCfg.EnableThinking)
	core.AssertTrue(t, templateCfg.LargeVariant)
	core.AssertTrue(t, templateCfg.Continuation)
	templateCfg = gemma4ChatTemplateConfigForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"attention_heads": "16",
		},
	}, inference.GenerateConfig{}, false)
	core.AssertTrue(t, templateCfg.LargeVariant)
	templateCfg = gemma4ChatTemplateConfigForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"attention_heads": "8",
			"gemma4_size":     "31B",
		},
	}, inference.GenerateConfig{}, false)
	core.AssertFalse(t, templateCfg.LargeVariant)
	disableThinking := false
	templateCfg = gemma4ChatTemplateConfigForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels:       map[string]string{"gemma4_size": "E2B"},
	}, inference.GenerateConfig{EnableThinking: &disableThinking}, false)
	core.AssertFalse(t, templateCfg.EnableThinking)
	core.AssertFalse(t, templateCfg.LargeVariant)
	core.AssertEqual(t, "visible", stripGemma4ThinkingChannels("visible<|channel>hidden<channel|>"))

	core.AssertEqual(t, "text", sampleText(inference.DatasetSample{Text: "text", Reasoning: "reason"}))
	core.AssertEqual(t, "prompt response", sampleText(inference.DatasetSample{Prompt: "prompt", Response: "response"}))
	core.AssertEqual(t, "user: hello\n", sampleText(inference.DatasetSample{Messages: []inference.Message{{Role: "user", Content: "hello"}}}))
	core.AssertEqual(t, "reason", sampleText(inference.DatasetSample{Reasoning: "reason"}))

	start := time.Unix(100, 0)
	first := start.Add(10 * time.Millisecond)
	end := first.Add(20 * time.Millisecond)
	prefill, decode := splitDurations(start, first, end)
	core.AssertEqual(t, 10*time.Millisecond, prefill)
	core.AssertEqual(t, 20*time.Millisecond, decode)
	prefill, decode = splitDurations(start, time.Time{}, end)
	core.AssertEqual(t, 30*time.Millisecond, prefill)
	core.AssertEqual(t, time.Duration(0), decode)
	core.AssertEqual(t, float64(20), tokensPerSecond(2, 100*time.Millisecond))
	core.AssertEqual(t, float64(0), tokensPerSecond(0, time.Second))

	called := false
	emptyTokenSeq(func(inference.Token) bool {
		called = true
		return true
	})
	core.AssertFalse(t, called)
}

func TestCoverage_NativePlanningHelpers_GoodBad(t *testing.T) {
	core.AssertEqual(t, 3*time.Millisecond, metricsTotalDuration(inference.GenerateMetrics{
		TotalDuration:   3 * time.Millisecond,
		PrefillDuration: 10 * time.Millisecond,
		DecodeDuration:  20 * time.Millisecond,
	}))
	core.AssertEqual(t, 30*time.Millisecond, metricsTotalDuration(inference.GenerateMetrics{
		PrefillDuration: 10 * time.Millisecond,
		DecodeDuration:  20 * time.Millisecond,
	}))
	core.AssertEqual(t, 0, benchmarkOperationCount(nil, 3))
	core.AssertEqual(t, 0, benchmarkOperationCount([]string{"a"}, 0))
	core.AssertEqual(t, 6, benchmarkOperationCount([]string{"a", "b"}, 3))
	core.AssertEqual(t, "0.000", averageDurationMillisecondsLabel(time.Second, 0))
	core.AssertEqual(t, "500.000", averageDurationMillisecondsLabel(time.Second, 2))

	for _, tc := range []struct {
		fileType int
		bits     int
		group    int
	}{
		{fileType: 0, bits: 32, group: 0},
		{fileType: 1, bits: 16, group: 0},
		{fileType: 2, bits: 4, group: 32},
		{fileType: 8, bits: 5, group: 32},
		{fileType: 7, bits: 8, group: 32},
		{fileType: 10, bits: 2, group: 16},
		{fileType: 11, bits: 3, group: 32},
		{fileType: 18, bits: 6, group: 64},
		{fileType: 999, bits: 0, group: 0},
	} {
		bits, group := quantisationFromFileType(uint32(tc.fileType))
		core.AssertEqual(t, tc.bits, bits)
		core.AssertEqual(t, tc.group, group)
	}

	core.AssertEqual(t, uint64(0), estimateKVCacheElementSpan(0, 4, 8, inference.ModelIdentity{}))
	core.AssertEqual(t, uint64(2*4*8), estimateKVCacheElementSpan(2, 4, 8, inference.ModelIdentity{}))
	core.AssertEqual(t, uint64(48), estimateKVCacheElementSpan(4, 8, 2, inference.ModelIdentity{Labels: map[string]string{
		"attention_full_layers":    "1",
		"attention_sliding_layers": "2",
		"sliding_window":           "4",
	}}))
	core.AssertEqual(t, ^uint64(0), rocmEstimatedRuntimeBytes(^uint64(0), 1))
	core.AssertEqual(t, uint64(30), rocmEstimatedRuntimeBytes(10, 20))
}

func TestCoverage_NativeBranchHelpers_GoodBad(t *testing.T) {
	core.AssertEqual(t, hipKernelStatusNotLinked, nativeRuntimeKernelStatus(nil).Decode)
	core.AssertEqual(t, hipKernelStatusNotLinked, nativeRuntimeKernelStatus(&fakeNativeRuntime{}).Decode)
	core.AssertEqual(t, hipKernelStatusLinked, nativeRuntimeKernelStatus(&coverageKernelRuntime{status: hipKernelStatus{Decode: hipKernelStatusLinked}}).Decode)
	core.AssertEqual(t, 0, rocmLogitProbeTopK(0))
	core.AssertEqual(t, 3, rocmLogitProbeTopK(3))
	core.AssertEqual(t, 5, rocmLogitProbeTopK(128))

	var nilModel *rocmModel
	nilModel.SetProbeSink(inference.ProbeSinkFunc(func(inference.ProbeEvent) {}))
	core.AssertNil(t, nilModel.probeSinkSnapshot())
	events := 0
	model := &rocmModel{native: &fakeNativeModel{metrics: inference.GenerateMetrics{PeakMemoryBytes: ^uint64(0), ActiveMemoryBytes: 5}}}
	model.SetProbeSink(inference.ProbeSinkFunc(func(inference.ProbeEvent) { events++ }))
	core.AssertNotNil(t, model.probeSinkSnapshot())
	model.emitProbe(inference.ProbeEvent{})
	core.AssertEqual(t, 1, events)
	restoreSuspended := model.suspendProbeSink()
	model.emitProbe(inference.ProbeEvent{})
	core.AssertEqual(t, 1, events)
	restoreSuspended()
	model.emitProbe(inference.ProbeEvent{})
	core.AssertEqual(t, 2, events)

	counter, restoreCounter := model.beginBenchmarkProbeCounter()
	model.emitProbe(inference.ProbeEvent{})
	core.AssertEqual(t, 1, counter.Count())
	core.AssertEqual(t, 3, events)
	restoreCounter()
	model.emitProbe(inference.ProbeEvent{})
	core.AssertEqual(t, 4, events)
	var nilCounter *rocmBenchmarkProbeCounter
	nilCounter.EmitProbe(inference.ProbeEvent{})
	core.AssertEqual(t, 0, nilCounter.Count())

	model.recordMetricsDurations(2, 3, -time.Second, -time.Millisecond)
	metrics := model.Metrics()
	core.AssertEqual(t, 2, metrics.PromptTokens)
	core.AssertEqual(t, 3, metrics.GeneratedTokens)
	core.AssertEqual(t, time.Duration(0), metrics.PrefillDuration)
	core.AssertEqual(t, ^uint64(0), metrics.PeakMemoryBytes)
	core.AssertEqual(t, uint64(5), metrics.ActiveMemoryBytes)
	nilModel.recordMetricsDurations(1, 1, time.Second, time.Second)
	nilModel.setLastMetrics(inference.GenerateMetrics{GeneratedTokens: 99})
	nilModel.setLastFailure(core.NewError("ignored"))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertNil(t, rocmContextErr(nil))
	core.AssertError(t, rocmContextErr(ctx))
	start := time.Unix(10, 0)
	prefill, decode := splitDurations(time.Time{}, start, start)
	core.AssertEqual(t, time.Duration(0), prefill)
	core.AssertEqual(t, time.Duration(0), decode)
	prefill, decode = splitDurations(start, start, start.Add(-time.Second))
	core.AssertEqual(t, time.Duration(0), prefill)
	core.AssertEqual(t, time.Duration(0), decode)

	core.AssertEqual(t, 7, rocmModelLabelInt(map[string]string{"value": "7"}, "value"))
	core.AssertEqual(t, 0, rocmModelLabelInt(nil, "missing"))
	core.AssertTrue(t, rocmAtLeastMemoryClass(16*memoryGiB-memoryClassToleranceBytes, 16*memoryGiB))
	core.AssertFalse(t, rocmAtLeastMemoryClass(8*memoryGiB, 16*memoryGiB))
	core.AssertEqual(t, uint64(60), estimateKVCacheElementSpan(3, 10, 2, inference.ModelIdentity{Labels: map[string]string{
		"attention_full_layers":    "3",
		"attention_sliding_layers": "3",
		"sliding_window":           "2",
	}}))

	for input, want := range map[string]string{
		"MiniMax M2":                            "minimax_m2",
		"Qwen3.5ForCausalLM":                    "qwen3_6",
		"Qwen3_5MoeForConditionalGeneration":    "qwen3_6_moe",
		"qwen3-next":                            "qwen3_next",
		"qwen3 moe":                             "qwen3_moe",
		"deepseek-r1":                           "deepseek_r1",
		"gptoss":                                "gpt-oss",
		"Gemma4AssistantForCausalLM":            "gemma4_assistant",
		"Gemma3TextForCausalLM":                 "gemma3_text",
		"Gemma4ForCausalLM":                     "gemma4_text",
		"Gemma4UnifiedForConditionalGeneration": "gemma4_unified",
		"gemma4_unified_text":                   "gemma4_unified_text",
		"BertForSequenceClassification":         "bert_rerank",
		"Phi4ForCausalLM":                       "phi",
		"glm4":                                  "glm4",
		"unknown model":                         "unknown_model",
	} {
		core.AssertEqual(t, want, normalizeROCmArchitecture(input))
	}

	labels := rocmMemoryPlanLabels(24*memoryGiB, 16384, 2, 4, inference.ModelIdentity{
		Architecture: "mixtral",
		QuantType:    "jang",
		Labels: map[string]string{
			"sliding_window": "128",
		},
	}, 100, 200, 300, "q8")
	core.AssertEqual(t, "4", labels["moe_max_resident_experts"])
	core.AssertEqual(t, "jang", labels["metadata_quantization"])
	core.AssertEqual(t, "128", labels["sliding_window"])

	lossLabels := map[string]string{}
	lossMetrics := inference.EvalMetrics{}
	rocmEvalLossAccumulator{candidates: 1, batches: 1, batchSize: 2, source: "classification", skipped: 3, status: "logits_unavailable", err: "missing logits"}.apply(context.Background(), model, &lossMetrics, lossLabels)
	core.AssertEqual(t, "1", lossLabels["eval.loss_candidates"])
	core.AssertEqual(t, "logits_unavailable", lossLabels["loss_status"])
	core.AssertEqual(t, "missing logits", lossLabels["loss_error"])

	lossLabels = map[string]string{}
	rocmEvalLossAccumulator{logits: [][]float32{{0, 1}}, targets: []int{1}}.apply(context.Background(), &rocmModel{}, &lossMetrics, lossLabels)
	core.AssertEqual(t, "reference", lossLabels["loss_backend"])
	core.AssertEqual(t, "experimental", lossLabels["loss_status"])
}

func TestCoverage_RocmModelAccessorsAndTokenCounts_GoodBad(t *testing.T) {
	var nilModel *rocmModel
	core.AssertEqual(t, "", nilModel.ModelType())
	core.AssertEqual(t, inference.ModelInfo{}, nilModel.Info())
	core.AssertFalse(t, nilModel.Capabilities().Available)
	core.AssertEqual(t, inference.GenerateMetrics{}, nilModel.Metrics())
	core.AssertNil(t, resultError(nilModel.Err()))
	core.AssertEqual(t, []int32{1, 2}, nilModel.Encode("alpha beta"))
	core.AssertEqual(t, "", nilModel.Decode([]int32{1, 2}))
	core.AssertEqual(t, 2, nilModel.promptTokenCount("alpha beta"))
	core.AssertEqual(t, 3, nilModel.promptsTokenCount([]string{"alpha beta", "gamma"}))
	core.AssertEqual(t, 2, nilModel.chatPromptTokenCount([]inference.Message{{Role: "user", Content: "alpha beta"}}))
	core.AssertEqual(t, 2, nilModel.evalSampleTokenCount(inference.DatasetSample{Text: "alpha beta"}))
	core.AssertNoError(t, resultError(nilModel.Close()))

	native := &fakeNativeModel{
		encodeByText: map[string][]int32{
			"prompt":          {7, 8, 9},
			"first":           {1},
			"second prompt":   {2, 3},
			"templated":       {4, 5, 6, 7},
			"question answer": {8, 9},
			"reason":          {10},
		},
		chatTemplateResult:       "templated",
		chatTemplateMutatesInput: true,
		decodeMutatesInput:       true,
	}
	model := &rocmModel{
		native:      native,
		modelType:   "gemma4-q4",
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 11, NumLayers: 2, HiddenSize: 3, QuantBits: 4},
		lastMetrics: inference.GenerateMetrics{GeneratedTokens: 5, DecodeDuration: time.Millisecond},
		lastError:   core.NewError("last failed"),
	}

	core.AssertEqual(t, "gemma4-q4", model.ModelType())
	core.AssertEqual(t, "gemma4", model.Info().Architecture)
	core.AssertTrue(t, model.Capabilities().Available)
	core.AssertEqual(t, 5, model.Metrics().GeneratedTokens)
	core.AssertContains(t, model.Err().Error(), "last failed")
	core.AssertFalse(t, model.classifyLinked())
	core.AssertFalse(t, model.gemma4Q4GenerateLinked())

	encoded := model.Encode("prompt")
	encoded[0] = 99
	core.AssertEqual(t, []int32{7, 8, 9}, native.Encode("prompt"))

	ids := []int32{1, 2, 3}
	core.AssertEqual(t, "3 tokens", model.Decode(ids))
	core.AssertEqual(t, []int32{1, 2, 3}, ids)

	messages := []inference.Message{{Role: "user", Content: "alpha beta"}}
	prompt, err := model.applyChatTemplate(messages)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "templated", prompt)
	core.AssertEqual(t, "user", messages[0].Role)
	core.AssertEqual(t, 3, model.promptTokenCount("prompt"))
	core.AssertEqual(t, 3, model.promptsTokenCount([]string{"first", "second prompt"}))
	core.AssertEqual(t, 2, model.chatPromptTokenCount(messages))
	core.AssertEqual(t, 2, model.evalSampleTokenCount(inference.DatasetSample{Prompt: "question", Response: "answer"}))
	core.AssertEqual(t, 2, model.evalSampleTokenCount(inference.DatasetSample{Messages: messages}))
	core.AssertEqual(t, 1, model.evalSampleTokenCount(inference.DatasetSample{Reasoning: "reason"}))

	native.chatTemplateErr = core.NewError("template failed")
	core.AssertEqual(t, 2, model.chatPromptTokenCount(messages))
	_, err = model.ApplyChatTemplate(messages)
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "template failed")
}

func TestCoverage_ScheduledModelAccessors_Good(t *testing.T) {
	var nilModel *ScheduledModel
	core.AssertEqual(t, "", nilModel.ModelType())
	core.AssertEqual(t, inference.ModelInfo{}, nilModel.Info())
	core.AssertEqual(t, inference.GenerateMetrics{}, nilModel.Metrics())
	core.AssertNil(t, resultError(nilModel.Err()))

	fake := &schedulerFakeTextModel{}
	fake.lastMetrics = inference.GenerateMetrics{GeneratedTokens: 7, DecodeDuration: time.Millisecond}
	fake.setErr(core.NewError("base failed"))
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	core.AssertEqual(t, "fake", model.ModelType())
	core.AssertEqual(t, "fake", model.Info().Architecture)
	core.AssertEqual(t, 7, model.Metrics().GeneratedTokens)
	core.AssertContains(t, model.Err().Error(), "base failed")
}

func TestCoverage_OpenAIWrappersAndErrorBranches_Bad(t *testing.T) {
	core.AssertNotNil(t, NewOpenAIResponsesHandlerForModel("model.gguf"))
	core.AssertNotNil(t, NewOpenAIServiceMuxForModel("model.gguf"))

	handler := NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{}))
	missingModel := httptest.NewRecorder()
	handler.ServeHTTP(missingModel, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"input":[{"role":"user","content":"hello"}]}`)))
	core.AssertEqual(t, http.StatusBadRequest, missingModel.Code)
	core.AssertContains(t, missingModel.Body.String(), "model is required")

	failing := &coverageFailingTextModel{tokens: []inference.Token{{Text: "partial"}}, err: core.NewError("chat failed")}
	handler = NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": failing}))
	modelErr := httptest.NewRecorder()
	handler.ServeHTTP(modelErr, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hello"}]}`)))
	core.AssertEqual(t, http.StatusInternalServerError, modelErr.Code)
	core.AssertContains(t, modelErr.Body.String(), "chat failed")
}

func TestCoverage_CompatWireErrorBranches_Bad(t *testing.T) {
	anthropicHandler := NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{}))
	malformedAnthropic := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(malformedAnthropic, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":`)))
	core.AssertEqual(t, http.StatusBadRequest, malformedAnthropic.Code)
	core.AssertContains(t, malformedAnthropic.Body.String(), "invalid request body")

	blankAnthropicModel := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(blankAnthropicModel, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)))
	core.AssertEqual(t, http.StatusBadRequest, blankAnthropicModel.Code)
	core.AssertContains(t, blankAnthropicModel.Body.String(), "model is required")

	failing := &coverageFailingTextModel{tokens: []inference.Token{{Text: "partial"}}, err: core.NewError("compat chat failed")}
	anthropicHandler = NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": failing}))
	modelErr := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(modelErr, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)))
	core.AssertEqual(t, http.StatusInternalServerError, modelErr.Code)
	core.AssertContains(t, modelErr.Body.String(), "compat chat failed")

	ollamaMux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{}))
	malformedOllama := httptest.NewRecorder()
	ollamaMux.ServeHTTP(malformedOllama, httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":`)))
	core.AssertEqual(t, http.StatusBadRequest, malformedOllama.Code)
	core.AssertContains(t, malformedOllama.Body.String(), "invalid request body")

	blankOllamaModel := httptest.NewRecorder()
	ollamaMux.ServeHTTP(blankOllamaModel, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"prompt":"hello"}`)))
	core.AssertEqual(t, http.StatusBadRequest, blankOllamaModel.Code)
	core.AssertContains(t, blankOllamaModel.Body.String(), "model is required")
}

func TestCoverage_CompatWireSuccessAndGuardBranches_GoodBad(t *testing.T) {
	tokens := []inference.Token{{Text: "hi"}, {Text: "!"}}
	model := &coverageFailingTextModel{tokens: tokens}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})

	var nilAnthropic *anthropicMessagesHandler
	unconfiguredAnthropic := httptest.NewRecorder()
	nilAnthropic.ServeHTTP(unconfiguredAnthropic, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{}`)))
	core.AssertEqual(t, http.StatusServiceUnavailable, unconfiguredAnthropic.Code)

	anthropicHandler := NewAnthropicMessagesHandler(resolver)
	wrongAnthropicMethod := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(wrongAnthropicMethod, httptest.NewRequest(http.MethodGet, anthropic.DefaultMessagesPath, nil))
	core.AssertEqual(t, http.StatusMethodNotAllowed, wrongAnthropicMethod.Code)
	core.AssertEqual(t, http.MethodPost, wrongAnthropicMethod.Header().Get("Allow"))

	streamAnthropic := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(streamAnthropic, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)))
	core.AssertEqual(t, http.StatusOK, streamAnthropic.Code)
	core.AssertEqual(t, "text/event-stream", streamAnthropic.Header().Get("Content-Type"))
	core.AssertContains(t, streamAnthropic.Body.String(), "event: content_block_delta")

	emptyAnthropic := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(emptyAnthropic, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"   "}]}]}`)))
	core.AssertEqual(t, http.StatusBadRequest, emptyAnthropic.Code)
	core.AssertContains(t, emptyAnthropic.Body.String(), "messages or system are required")

	successAnthropic := httptest.NewRecorder()
	anthropicHandler.ServeHTTP(successAnthropic, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","max_tokens":2,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)))
	core.AssertEqual(t, http.StatusOK, successAnthropic.Code)
	core.AssertContains(t, successAnthropic.Body.String(), "hi!")

	var nilOllama *ollamaCompatHandler
	unconfiguredOllamaChat := httptest.NewRecorder()
	nilOllama.chat(unconfiguredOllamaChat, httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{}`)))
	core.AssertEqual(t, http.StatusServiceUnavailable, unconfiguredOllamaChat.Code)
	unconfiguredOllamaGenerate := httptest.NewRecorder()
	nilOllama.generate(unconfiguredOllamaGenerate, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{}`)))
	core.AssertEqual(t, http.StatusServiceUnavailable, unconfiguredOllamaGenerate.Code)

	ollamaMux := NewOllamaHandler(resolver)
	wrongOllamaMethod := httptest.NewRecorder()
	ollamaMux.ServeHTTP(wrongOllamaMethod, httptest.NewRequest(http.MethodGet, ollama.DefaultChatPath, nil))
	core.AssertEqual(t, http.StatusMethodNotAllowed, wrongOllamaMethod.Code)

	streamOllamaChat := httptest.NewRecorder()
	ollamaMux.ServeHTTP(streamOllamaChat, httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hello"}]}`)))
	core.AssertEqual(t, http.StatusOK, streamOllamaChat.Code)
	core.AssertContains(t, streamOllamaChat.Body.String(), `"message"`)
	core.AssertContains(t, streamOllamaChat.Body.String(), `"done":true`)

	emptyOllamaChat := httptest.NewRecorder()
	ollamaMux.ServeHTTP(emptyOllamaChat, httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"   "}]}`)))
	core.AssertEqual(t, http.StatusBadRequest, emptyOllamaChat.Code)
	core.AssertContains(t, emptyOllamaChat.Body.String(), "messages are required")

	successOllamaChat := httptest.NewRecorder()
	ollamaMux.ServeHTTP(successOllamaChat, httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hello"}],"options":{"num_predict":2}}`)))
	core.AssertEqual(t, http.StatusOK, successOllamaChat.Code)
	core.AssertContains(t, successOllamaChat.Body.String(), "hi!")

	streamOllamaGenerate := httptest.NewRecorder()
	ollamaMux.ServeHTTP(streamOllamaGenerate, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"hello","stream":true}`)))
	core.AssertEqual(t, http.StatusOK, streamOllamaGenerate.Code)
	core.AssertContains(t, streamOllamaGenerate.Body.String(), `"response":"hi"`)
	core.AssertContains(t, streamOllamaGenerate.Body.String(), `"done":true`)

	emptyOllamaGenerate := httptest.NewRecorder()
	ollamaMux.ServeHTTP(emptyOllamaGenerate, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"   "}`)))
	core.AssertEqual(t, http.StatusBadRequest, emptyOllamaGenerate.Code)
	core.AssertContains(t, emptyOllamaGenerate.Body.String(), "prompt is required")

	successOllamaGenerate := httptest.NewRecorder()
	ollamaMux.ServeHTTP(successOllamaGenerate, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"hello","options":{"num_predict":2}}`)))
	core.AssertEqual(t, http.StatusOK, successOllamaGenerate.Code)
	core.AssertContains(t, successOllamaGenerate.Body.String(), "hi!")

	nilBody := httptest.NewRecorder()
	core.AssertFalse(t, decodeROCmWireRequest(nilBody, &http.Request{Method: http.MethodPost}, &struct{}{}))
	core.AssertEqual(t, http.StatusBadRequest, nilBody.Code)

	decoded := struct {
		Model string `json:"model"`
	}{}
	validBody := httptest.NewRecorder()
	core.AssertTrue(t, decodeROCmWireRequest(validBody, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"model":"qwen"}`)), &decoded))
	core.AssertEqual(t, "qwen", decoded.Model)
}

func TestCoverage_DiscoverModelsEmptyAndBadGlob_GoodBad(t *testing.T) {
	_, err := DiscoverModels("[")
	core.AssertError(t, err)

	dir := t.TempDir()
	core.RequireNoError(t, os.WriteFile(core.PathJoin(dir, "broken.gguf"), []byte("not a gguf"), 0o600))
	models, err := DiscoverModels(dir)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(models))
}

func TestCoverage_EmbeddingClassifierHelpers_GoodBad(t *testing.T) {
	core.AssertTrue(t, isHIPSequenceClassifierBiasTensor("classifier.bias"))
	core.AssertTrue(t, isHIPSequenceClassifierBiasTensor("encoder.score.bias"))
	core.AssertFalse(t, isHIPSequenceClassifierBiasTensor("classifier.weight"))

	priority, biasName, ok := hipSequenceClassifierWeightCandidate("encoder.classifier.weight")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 2, priority)
	core.AssertEqual(t, "encoder.classifier.bias", biasName)
	_, _, ok = hipSequenceClassifierWeightCandidate("pooler.weight")
	core.AssertFalse(t, ok)

	labels, hidden, err := hipSequenceClassifierWeightShape(inference.ModelInfo{HiddenSize: 3}, nativeTensorInfo{Dimensions: []uint64{2, 3}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, labels)
	core.AssertEqual(t, 3, hidden)
	_, _, err = hipSequenceClassifierWeightShape(inference.ModelInfo{}, nativeTensorInfo{Dimensions: []uint64{2}})
	core.AssertError(t, err)
	_, _, err = hipSequenceClassifierWeightShape(inference.ModelInfo{HiddenSize: 4}, nativeTensorInfo{Dimensions: []uint64{2, 3}})
	core.AssertError(t, err)

	encoding, err := hipSequenceClassifierWeightEncoding(nativeTensorInfo{TypeName: "F32"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(hipProjectionWeightEncodingF32), encoding)
	encoding, err = hipSequenceClassifierBiasEncoding(nativeTensorInfo{Type: 1, TypeName: "F16"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(hipProjectionWeightEncodingFP16), encoding)
	_, err = hipSequenceClassifierWeightEncoding(nativeTensorInfo{Type: 999, TypeName: "Q8"})
	core.AssertError(t, err)
	_, err = hipSequenceClassifierBiasEncoding(nativeTensorInfo{Type: 999, TypeName: "Q8"})
	core.AssertError(t, err)

	core.AssertEqual(t, "f32", hipProjectionWeightEncodingLabel(hipProjectionWeightEncodingF32))
	core.AssertEqual(t, "fp16", hipProjectionWeightEncodingLabel(hipProjectionWeightEncodingFP16))
	core.AssertEqual(t, "q8", hipProjectionWeightEncodingLabel(hipProjectionWeightEncodingQ8))
	core.AssertEqual(t, "99", hipProjectionWeightEncodingLabel(99))

	core.AssertNoError(t, hipSequenceClassifierBiasShape(2, nativeTensorInfo{Dimensions: []uint64{2}}))
	core.AssertError(t, hipSequenceClassifierBiasShape(2, nativeTensorInfo{Dimensions: []uint64{2, 1}}))
	core.AssertError(t, hipSequenceClassifierBiasShape(3, nativeTensorInfo{Dimensions: []uint64{2}}))
	core.AssertEqual(t, 0, hipSequenceClassifierPositiveLabelIndex(1))
	core.AssertEqual(t, 1, hipSequenceClassifierPositiveLabelIndex(2))
	core.AssertEqual(t, "query [SEP] document", hipSequenceClassifierPairText(" query ", " document "))

	score, err := hipSequenceClassifierRerankScore([]float32{-1, 2}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, float32(2), score)
	_, err = hipSequenceClassifierRerankScore(nil, 0)
	core.AssertError(t, err)
	_, err = hipSequenceClassifierRerankScore([]float32{1}, 2)
	core.AssertError(t, err)

	vectors, err := hipTinyTokenEmbeddingVectors([]float32{1, 2, 3, 4}, hipLoadedTinyLMConfig{VocabSize: 2, HiddenSize: 2}, []int32{1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{3, 4}, vectors)
	_, err = hipTinyTokenEmbeddingVectors([]float32{1, 2}, hipLoadedTinyLMConfig{VocabSize: 1, HiddenSize: 2}, []int32{2})
	core.AssertError(t, err)

	flat, err := flattenEqualFloat32Vectors([][]float32{{1, 2}, {3, 4}}, 2)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, 3, 4}, flat)
	_, err = flattenEqualFloat32Vectors(nil, 2)
	core.AssertError(t, err)
	_, err = flattenEqualFloat32Vectors([][]float32{{1}}, 0)
	core.AssertError(t, err)
}

func TestCoverage_RocmEmbeddingAndRerankWrappers_GoodBad(t *testing.T) {
	native := &coverageEmbeddingNative{
		embedding: &inference.EmbeddingResult{
			Vectors: [][]float32{{1, 2}},
			Labels:  map[string]string{"source": "native"},
			Model:   inference.ModelIdentity{Labels: map[string]string{"native": "label"}},
		},
		rerank: &inference.RerankResult{
			Results: []inference.RerankScore{{Index: 0, Score: 0.9, Text: "doc", Labels: map[string]string{"rank": "1"}}},
			Labels:  map[string]string{"source": "native"},
			Model:   inference.ModelIdentity{Labels: map[string]string{"native": "label"}},
		},
	}
	model := &rocmModel{
		native:    native,
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2, VocabSize: 8},
	}

	embedding, err := model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}, Labels: map[string]string{"tenant": "a"}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "bert", embedding.Model.Architecture)
	embedding.Vectors[0][0] = 99
	embedding.Labels["source"] = "mutated"
	core.AssertEqual(t, float32(1), native.embedding.Vectors[0][0])
	core.AssertEqual(t, "native", native.embedding.Labels["source"])

	rerank, err := model.Rerank(context.Background(), inference.RerankRequest{Query: "q", Documents: []string{"doc"}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "bert", rerank.Model.Architecture)
	rerank.Results[0].Labels["rank"] = "mutated"
	rerank.Labels["source"] = "mutated"
	core.AssertEqual(t, "1", native.rerank.Results[0].Labels["rank"])
	core.AssertEqual(t, "native", native.rerank.Labels["source"])

	_, err = model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{" "}})
	core.AssertError(t, err)
	_, err = model.Rerank(context.Background(), inference.RerankRequest{Query: " ", Documents: []string{"doc"}})
	core.AssertError(t, err)
	_, err = (&rocmModel{native: &fakeNativeModel{}}).Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})
	core.AssertError(t, err)
	_, err = (&rocmModel{native: &fakeNativeModel{}}).Rerank(context.Background(), inference.RerankRequest{Query: "q", Documents: []string{"doc"}})
	core.AssertError(t, err)

	native.embeddingErr = core.NewError("embedding failed")
	_, err = model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})
	core.AssertError(t, err)
	native.embeddingErr = nil
	native.embedding = nil
	_, err = model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})
	core.AssertError(t, err)
	native.embedding = &inference.EmbeddingResult{Vectors: [][]float32{{1}}}

	native.rerankErr = core.NewError("rerank failed")
	_, err = model.Rerank(context.Background(), inference.RerankRequest{Query: "q", Documents: []string{"doc"}})
	core.AssertError(t, err)
	native.rerankErr = nil
	native.rerank = nil
	_, err = model.Rerank(context.Background(), inference.RerankRequest{Query: "q", Documents: []string{"doc"}})
	core.AssertError(t, err)
}

type coverageFailingTextModel struct {
	tokens []inference.Token
	err    error
}

func (model *coverageFailingTextModel) Generate(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.stream(ctx)
}

func (model *coverageFailingTextModel) Chat(ctx context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.stream(ctx)
}

func (model *coverageFailingTextModel) stream(ctx context.Context) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range model.tokens {
			if ctx != nil {
				select {
				case <-ctx.Done():
					return
				default:
				}
			}
			if !yield(token) {
				return
			}
		}
	}
}

func (model *coverageFailingTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.ResultOf(nil, model.err)
}

func (model *coverageFailingTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.ResultOf(nil, model.err)
}

func (model *coverageFailingTextModel) ModelType() string { return "coverage" }
func (model *coverageFailingTextModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "coverage"}
}
func (model *coverageFailingTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{GeneratedTokens: len(model.tokens)}
}
func (model *coverageFailingTextModel) Err() core.Result   { return core.ResultOf(nil, model.err) }
func (model *coverageFailingTextModel) Close() core.Result { return core.Ok(nil) }

type coverageEmbeddingNative struct {
	fakeNativeModel
	embedding    *inference.EmbeddingResult
	embeddingErr error
	rerank       *inference.RerankResult
	rerankErr    error
}

func (model *coverageEmbeddingNative) Embed(context.Context, inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	if model.embeddingErr != nil {
		return nil, model.embeddingErr
	}
	return model.embedding, nil
}

func (model *coverageEmbeddingNative) Rerank(context.Context, inference.RerankRequest) (*inference.RerankResult, error) {
	if model.rerankErr != nil {
		return nil, model.rerankErr
	}
	return model.rerank, nil
}

type coverageKernelRuntime struct {
	fakeNativeRuntime
	status hipKernelStatus
}

func (runtime *coverageKernelRuntime) KernelStatus() hipKernelStatus {
	return runtime.status
}
