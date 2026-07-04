// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const rocmTinyLoRAFormat = "rocm-tiny-lora"
const rocmSmallLoRAFormat = "rocm-small-lm-head-lora"
const rocmClassifierLoRAFormat = "rocm-classifier-lora"

type hipTinyLoRAAdapterFile struct {
	Format     string    `json:"format,omitempty"`
	Name       string    `json:"name,omitempty"`
	Target     string    `json:"target,omitempty"`
	Rank       int       `json:"rank,omitempty"`
	Alpha      float32   `json:"alpha,omitempty"`
	HiddenSize int       `json:"hidden_size,omitempty"`
	VocabSize  int       `json:"vocab_size,omitempty"`
	LoRAA      []float32 `json:"lora_a,omitempty"`
	LoRAB      []float32 `json:"lora_b,omitempty"`
	Bias       []float32 `json:"bias,omitempty"`
}

type hipLoadedTinyLoRAAdapter struct {
	identity inference.AdapterIdentity
	a        []float32
	b        []float32
	bias     []float32
	rank     int
	alpha    float32
}

type hipLoadedSmallLoRAAdapter struct {
	identity inference.AdapterIdentity
	a        []float32
	b        []float32
	bias     []float32
	rank     int
	alpha    float32
}

type hipClassifierLoRAAdapterFile struct {
	Format     string    `json:"format,omitempty"`
	Name       string    `json:"name,omitempty"`
	Target     string    `json:"target,omitempty"`
	Rank       int       `json:"rank,omitempty"`
	Alpha      float32   `json:"alpha,omitempty"`
	HiddenSize int       `json:"hidden_size,omitempty"`
	NumLabels  int       `json:"num_labels,omitempty"`
	LoRAA      []float32 `json:"lora_a,omitempty"`
	LoRAB      []float32 `json:"lora_b,omitempty"`
	Bias       []float32 `json:"bias,omitempty"`
}

type hipLoadedClassifierLoRAAdapter struct {
	identity inference.AdapterIdentity
	a        []float32
	b        []float32
	bias     []float32
	rank     int
	alpha    float32
}

func (model *hipLoadedModel) loadTinyLoRAAdapter(path string) (*hipLoadedTinyLoRAAdapter, inference.AdapterIdentity, error) {
	cfg, err := model.loadedTinyLMConfig()
	if err != nil {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "load tiny model config", err)
	}
	adapterPath := resolveTinyLoRAAdapterPath(path)
	read := core.ReadFile(adapterPath)
	if !read.OK {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "read adapter", read.Value.(error))
	}
	payload := read.Value.([]byte)
	var file hipTinyLoRAAdapterFile
	if result := core.JSONUnmarshal(payload, &file); !result.OK {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "parse adapter", result.Value.(error))
	}
	adapter, err := validateTinyLoRAAdapterFile(file, cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	base, err := model.loadedTinyOutputWeights(cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "read base output weights", err)
	}
	if _, err := rocmReferenceLoRAProjection(make([]float32, cfg.HiddenSize), base, adapter.a, adapter.b, cfg.VocabSize, cfg.HiddenSize, adapter.rank, adapter.alpha, adapter.bias); err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	sum := sha256.Sum256(payload)
	identity := inference.AdapterIdentity{
		Path:       path,
		Hash:       hex.EncodeToString(sum[:]),
		Format:     rocmTinyLoRAFormat,
		Rank:       adapter.rank,
		Alpha:      adapter.alpha,
		TargetKeys: []string{"output.weight"},
		Labels: map[string]string{
			"adapter_file":       adapterPath,
			"adapter_name":       firstNonEmptyString(file.Name, rocmTinyLoRAFormat),
			"adapter_runtime":    "hip_tiny_loaded",
			"lora_kernel":        hipKernelStatusLinked,
			"lora_kernel_name":   hipKernelNameLoRA,
			"lora_model_status":  "experimental_tiny_loaded",
			"target":             firstNonEmptyString(file.Target, "output.weight"),
			"target_hidden_size": core.Sprintf("%d", cfg.HiddenSize),
			"target_vocab_size":  core.Sprintf("%d", cfg.VocabSize),
		},
	}
	adapter.identity = identity
	return adapter, identity, nil
}

func resolveTinyLoRAAdapterPath(path string) string {
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		return filepath.Join(path, "rocm_tiny_lora.json")
	}
	return path
}

func resolveSmallLoRAAdapterPath(path string) string {
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		candidate := filepath.Join(path, "rocm_lm_head_lora.json")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		return filepath.Join(path, "rocm_tiny_lora.json")
	}
	return path
}

func resolveClassifierLoRAAdapterPath(path string) string {
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		return filepath.Join(path, "rocm_classifier_lora.json")
	}
	return path
}

func validateTinyLoRAAdapterFile(file hipTinyLoRAAdapterFile, cfg hipLoadedTinyLMConfig) (*hipLoadedTinyLoRAAdapter, error) {
	format := core.Trim(file.Format)
	if format != "" && format != rocmTinyLoRAFormat && format != "lora" {
		return nil, core.E("rocm.hip.LoadAdapter", "unsupported adapter format", nil)
	}
	target := core.Trim(file.Target)
	if target != "" && target != "output" && target != "output.weight" && target != "lm_head" && target != "lm_head.weight" {
		return nil, core.E("rocm.hip.LoadAdapter", "unsupported adapter target", nil)
	}
	if file.HiddenSize > 0 && file.HiddenSize != cfg.HiddenSize {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter hidden size mismatch", nil)
	}
	if file.VocabSize > 0 && file.VocabSize != cfg.VocabSize {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter vocab size mismatch", nil)
	}
	if file.Rank <= 0 {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter rank must be positive", nil)
	}
	if !hipQ8ScaleIsPositiveFinite(file.Alpha) {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter alpha must be positive and finite", nil)
	}
	if len(file.LoRAA) != file.Rank*cfg.HiddenSize {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter LoRA A length must match rank*hidden", nil)
	}
	if len(file.LoRAB) != cfg.VocabSize*file.Rank {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter LoRA B length must match vocab*rank", nil)
	}
	if len(file.Bias) != 0 && len(file.Bias) != cfg.VocabSize {
		return nil, core.E("rocm.hip.LoadAdapter", "adapter bias length must match vocab", nil)
	}
	return &hipLoadedTinyLoRAAdapter{
		a:     append([]float32(nil), file.LoRAA...),
		b:     append([]float32(nil), file.LoRAB...),
		bias:  append([]float32(nil), file.Bias...),
		rank:  file.Rank,
		alpha: file.Alpha,
	}, nil
}

func (model *hipLoadedModel) loadSmallLoRAAdapter(path string, cfg hipLoadedSmallDecodeConfig) (*hipLoadedSmallLoRAAdapter, inference.AdapterIdentity, error) {
	adapterPath := resolveSmallLoRAAdapterPath(path)
	read := core.ReadFile(adapterPath)
	if !read.OK {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "read adapter", read.Value.(error))
	}
	payload := read.Value.([]byte)
	var file hipTinyLoRAAdapterFile
	if result := core.JSONUnmarshal(payload, &file); !result.OK {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "parse adapter", result.Value.(error))
	}
	adapter, err := validateSmallLoRAAdapterFile(file, cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	base, err := model.loadedSmallLMHeadWeightsF32(cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "read small LM head weights", err)
	}
	if _, err := rocmReferenceLoRAProjection(make([]float32, cfg.HiddenSize), base, adapter.a, adapter.b, cfg.VocabSize, cfg.HiddenSize, adapter.rank, adapter.alpha, adapter.bias); err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	sum := sha256.Sum256(payload)
	identity := inference.AdapterIdentity{
		Path:       path,
		Hash:       hex.EncodeToString(sum[:]),
		Format:     rocmSmallLoRAFormat,
		Rank:       adapter.rank,
		Alpha:      adapter.alpha,
		TargetKeys: []string{"output.weight"},
		Labels: map[string]string{
			"adapter_file":        adapterPath,
			"adapter_name":        firstNonEmptyString(file.Name, rocmSmallLoRAFormat),
			"adapter_runtime":     "hip_small_lm_head",
			"decode_architecture": cfg.Architecture,
			"lora_kernel":         hipKernelStatusLinked,
			"lora_kernel_name":    hipKernelNameLoRA,
			"lora_model_status":   hipSmallDecodeLoRAModelStatus(cfg.Architecture),
			"target":              firstNonEmptyString(file.Target, "output.weight"),
			"target_hidden_size":  core.Sprintf("%d", cfg.HiddenSize),
			"target_vocab_size":   core.Sprintf("%d", cfg.VocabSize),
		},
	}
	adapter.identity = identity
	return adapter, identity, nil
}

func validateSmallLoRAAdapterFile(file hipTinyLoRAAdapterFile, cfg hipLoadedSmallDecodeConfig) (*hipLoadedSmallLoRAAdapter, error) {
	format := core.Trim(file.Format)
	if format != "" && format != rocmSmallLoRAFormat && format != rocmTinyLoRAFormat && format != "lora" {
		return nil, core.E("rocm.hip.LoadAdapter", "unsupported small LM-head adapter format", nil)
	}
	file.Format = rocmTinyLoRAFormat
	adapter, err := validateTinyLoRAAdapterFile(file, hipLoadedTinyLMConfig{HiddenSize: cfg.HiddenSize, VocabSize: cfg.VocabSize})
	if err != nil {
		return nil, err
	}
	return &hipLoadedSmallLoRAAdapter{
		a:     adapter.a,
		b:     adapter.b,
		bias:  adapter.bias,
		rank:  adapter.rank,
		alpha: adapter.alpha,
	}, nil
}

func (model *hipLoadedModel) loadClassifierLoRAAdapter(path string, cfg hipLoadedSequenceClassifierConfig) (*hipLoadedClassifierLoRAAdapter, inference.AdapterIdentity, error) {
	adapterPath := resolveClassifierLoRAAdapterPath(path)
	read := core.ReadFile(adapterPath)
	if !read.OK {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "read adapter", read.Value.(error))
	}
	payload := read.Value.([]byte)
	var file hipClassifierLoRAAdapterFile
	if result := core.JSONUnmarshal(payload, &file); !result.OK {
		return nil, inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "parse adapter", result.Value.(error))
	}
	adapter, err := validateClassifierLoRAAdapterFile(file, cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	base, err := model.loadedSequenceClassifierWeightsF32(cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	baseBias, err := model.loadedClassifierBias(cfg)
	if err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	bias, err := mergeClassifierLoRABias(baseBias, adapter.bias, cfg.NumLabels)
	if err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	if _, err := rocmReferenceLoRAProjection(make([]float32, cfg.HiddenSize), base, adapter.a, adapter.b, cfg.NumLabels, cfg.HiddenSize, adapter.rank, adapter.alpha, bias); err != nil {
		return nil, inference.AdapterIdentity{}, err
	}
	sum := sha256.Sum256(payload)
	target := firstNonEmptyString(file.Target, cfg.WeightTensor)
	identity := inference.AdapterIdentity{
		Path:       path,
		Hash:       hex.EncodeToString(sum[:]),
		Format:     rocmClassifierLoRAFormat,
		Rank:       adapter.rank,
		Alpha:      adapter.alpha,
		TargetKeys: []string{cfg.WeightTensor},
		Labels: map[string]string{
			"adapter_file":          adapterPath,
			"adapter_name":          firstNonEmptyString(file.Name, rocmClassifierLoRAFormat),
			"adapter_runtime":       "hip_bert_classifier",
			"classifier_labels":     core.Sprintf("%d", cfg.NumLabels),
			"classifier_tensor":     cfg.WeightTensor,
			"lora_kernel":           hipKernelStatusLinked,
			"lora_kernel_name":      hipKernelNameLoRA,
			"lora_model_status":     "experimental_bert_sequence_classifier",
			"target":                target,
			"target_hidden_size":    core.Sprintf("%d", cfg.HiddenSize),
			"target_positive_label": core.Sprintf("%d", cfg.PositiveLabelIndex),
		},
	}
	adapter.identity = identity
	return adapter, identity, nil
}

func validateClassifierLoRAAdapterFile(file hipClassifierLoRAAdapterFile, cfg hipLoadedSequenceClassifierConfig) (*hipLoadedClassifierLoRAAdapter, error) {
	format := core.Trim(file.Format)
	if format != "" && format != rocmClassifierLoRAFormat && format != "lora" {
		return nil, core.E("rocm.hip.LoadAdapter", "unsupported classifier adapter format", nil)
	}
	target := core.Trim(file.Target)
	if target != "" && target != "classifier" && target != "score" && !isHIPSequenceClassifierWeightTensor(target) {
		return nil, core.E("rocm.hip.LoadAdapter", "unsupported classifier adapter target", nil)
	}
	if file.HiddenSize > 0 && file.HiddenSize != cfg.HiddenSize {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter hidden size mismatch", nil)
	}
	if file.NumLabels > 0 && file.NumLabels != cfg.NumLabels {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter label count mismatch", nil)
	}
	if file.Rank <= 0 {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter rank must be positive", nil)
	}
	if !hipQ8ScaleIsPositiveFinite(file.Alpha) {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter alpha must be positive and finite", nil)
	}
	if len(file.LoRAA) != file.Rank*cfg.HiddenSize {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter LoRA A length must match rank*hidden", nil)
	}
	if len(file.LoRAB) != cfg.NumLabels*file.Rank {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter LoRA B length must match labels*rank", nil)
	}
	if len(file.Bias) != 0 && len(file.Bias) != cfg.NumLabels {
		return nil, core.E("rocm.hip.LoadAdapter", "classifier adapter bias length must match label count", nil)
	}
	return &hipLoadedClassifierLoRAAdapter{
		a:     append([]float32(nil), file.LoRAA...),
		b:     append([]float32(nil), file.LoRAB...),
		bias:  append([]float32(nil), file.Bias...),
		rank:  file.Rank,
		alpha: file.Alpha,
	}, nil
}

func (model *hipLoadedModel) loadedTinyOutputWeights(cfg hipLoadedTinyLMConfig) ([]float32, error) {
	if model == nil || model.driver == nil {
		return nil, core.E("rocm.hip.TinyLoRA", "HIP driver is nil", nil)
	}
	payload := make([]byte, cfg.OutputWeightBytes)
	if err := model.driver.CopyDeviceToHost(cfg.OutputWeightPointer, payload); err != nil {
		return nil, core.E("rocm.hip.TinyLoRA", "copy output weights", err)
	}
	if hipTinyUsesJANGTQOutput(cfg) {
		weights, err := hipTinyJANGTQOutputWeightValues(payload, cfg)
		if err != nil {
			return nil, err
		}
		if len(weights) != cfg.VocabSize*cfg.HiddenSize {
			return nil, core.E("rocm.hip.TinyLoRA", "output weight length must match vocab*hidden", nil)
		}
		return weights, nil
	}
	if hipTinyUsesCodebookOutput(cfg) {
		weights, err := model.loadedTinyCodebookOutputWeights(cfg, payload)
		if err != nil {
			return nil, err
		}
		if len(weights) != cfg.VocabSize*cfg.HiddenSize {
			return nil, core.E("rocm.hip.TinyLoRA", "output weight length must match vocab*hidden", nil)
		}
		return weights, nil
	}
	weights, err := hipTinyOutputWeightValues(payload, cfg.OutputWeightEncoding, cfg.Q8Scale)
	if err != nil {
		return nil, err
	}
	if len(weights) != cfg.VocabSize*cfg.HiddenSize {
		return nil, core.E("rocm.hip.TinyLoRA", "output weight length must match vocab*hidden", nil)
	}
	return weights, nil
}

func (model *hipLoadedModel) loadedTinyCodebookOutputWeights(cfg hipLoadedTinyLMConfig, codes []byte) ([]float32, error) {
	codebook, err := model.loadedF32TensorPayload("rocm.hip.TinyCodebook", "codebook output table", cfg.OutputCodebookPointer, cfg.OutputCodebookBytes, cfg.OutputCodebookCount*cfg.OutputCodebookDim)
	if err != nil {
		return nil, err
	}
	return rocmReferenceCodebookLookup(codes, codebook, cfg.OutputCodebookDim)
}

func (model *hipLoadedModel) loadedSmallLMHeadWeightsF32(cfg hipLoadedSmallDecodeConfig) ([]float32, error) {
	payload, err := model.loadedTensorBytes("rocm.hip.SmallLoRA", "LM head weights", cfg.LMHeadPointer, cfg.LMHeadBytes)
	if err != nil {
		return nil, err
	}
	weights, err := hipTinyOutputWeightValues(payload, hipTinyOutputWeightEncodingFP16, 0)
	if err != nil {
		return nil, err
	}
	if len(weights) != cfg.VocabSize*cfg.HiddenSize {
		return nil, core.E("rocm.hip.SmallLoRA", "LM head weight length must match vocab*hidden", nil)
	}
	return weights, nil
}

func hipTinyJANGTQOutputWeightValues(payload []byte, cfg hipLoadedTinyLMConfig) ([]float32, error) {
	count := cfg.VocabSize * cfg.HiddenSize
	quantized, err := unpackROCmSignedBits(payload, cfg.OutputJANGTQDescriptor.Bits, count)
	if err != nil {
		return nil, err
	}
	if !hipQ8ScaleIsPositiveFinite(cfg.OutputJANGTQScale) {
		return nil, core.E("rocm.hip.TinyJANGTQ", "JANGTQ scale must be positive and finite", nil)
	}
	out := make([]float32, len(quantized))
	for index, value := range quantized {
		out[index] = float32(value) * cfg.OutputJANGTQScale
	}
	return out, nil
}

func (model *hipLoadedModel) loadedSequenceClassifierWeightsF32(cfg hipLoadedSequenceClassifierConfig) ([]float32, error) {
	weights, err := model.loadedClassifierWeights(cfg)
	if err != nil {
		return nil, err
	}
	return hipSequenceClassifierWeightsF32(weights)
}

func hipSequenceClassifierWeightsF32(weights hipLoadedSequenceClassifierWeights) ([]float32, error) {
	var values []float32
	switch {
	case len(weights.F32) > 0:
		values = append([]float32(nil), weights.F32...)
	case len(weights.FP16) > 0:
		values = make([]float32, len(weights.FP16))
		for index, value := range weights.FP16 {
			values[index] = hipFloat16ToFloat32(value)
		}
	default:
		return nil, core.E("rocm.hip.SequenceClassifierLoRA", "classifier base weights are required", nil)
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.SequenceClassifierLoRA", "classifier base weight values must be finite", nil)
	}
	return values, nil
}

func mergeClassifierLoRABias(baseBias, adapterBias []float32, rows int) ([]float32, error) {
	if rows <= 0 {
		return nil, core.E("rocm.hip.SequenceClassifierLoRA", "classifier row count must be positive", nil)
	}
	if len(baseBias) == 0 && len(adapterBias) == 0 {
		return nil, nil
	}
	if len(baseBias) != 0 && len(baseBias) != rows {
		return nil, core.E("rocm.hip.SequenceClassifierLoRA", "classifier base bias length must match label count", nil)
	}
	if len(adapterBias) != 0 && len(adapterBias) != rows {
		return nil, core.E("rocm.hip.SequenceClassifierLoRA", "classifier adapter bias length must match label count", nil)
	}
	out := make([]float32, rows)
	for index := range out {
		if len(baseBias) > 0 {
			out[index] += baseBias[index]
		}
		if len(adapterBias) > 0 {
			out[index] += adapterBias[index]
		}
	}
	return out, nil
}

func (model *hipLoadedModel) applyTinyLoRAToPrefill(ctx context.Context, cfg hipLoadedTinyLMConfig, output hipTinyPrefillResult) (hipTinyPrefillResult, error) {
	if model == nil || model.tinyLoRA == nil {
		return output, nil
	}
	hidden, err := hipTinyAttentionWeightedOutput(output.StateValues, output.Attention, cfg.HiddenSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	logits, next, score, err := model.runTinyLoRAProjection(ctx, cfg, hidden)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	output.Logits = logits
	output.NextTokenID = next
	output.NextScore = score
	return output, nil
}

func (model *hipLoadedModel) applyTinyLoRAToDecode(ctx context.Context, cfg hipLoadedTinyLMConfig, output hipTinyDecodeResult) (hipTinyDecodeResult, error) {
	if model == nil || model.tinyLoRA == nil {
		return output, nil
	}
	hidden, err := hipTinyAttentionWeightedOutput(output.UpdatedValues, output.Attention, cfg.HiddenSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	logits, next, score, err := model.runTinyLoRAProjection(ctx, cfg, hidden)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	output.Logits = logits
	output.NextTokenID = next
	output.NextScore = score
	return output, nil
}

func (model *hipLoadedModel) runSequenceClassifierLoRAProjection(ctx context.Context, cfg hipLoadedSequenceClassifierConfig, input, baseBias []float32) ([]float32, error) {
	if model == nil || model.classLoRA == nil {
		return nil, core.E("rocm.hip.SequenceClassifierLoRA", "active classifier LoRA adapter is required", nil)
	}
	base, err := model.loadedSequenceClassifierWeightsF32(cfg)
	if err != nil {
		return nil, err
	}
	bias, err := mergeClassifierLoRABias(baseBias, model.classLoRA.bias, cfg.NumLabels)
	if err != nil {
		return nil, err
	}
	return hipRunLoRAProjectionKernel(ctx, model.driver, hipLoRAProjectionRequest{
		Input:      input,
		BaseWeight: base,
		LoRAA:      model.classLoRA.a,
		LoRAB:      model.classLoRA.b,
		Rows:       cfg.NumLabels,
		Cols:       cfg.HiddenSize,
		Rank:       model.classLoRA.rank,
		Alpha:      model.classLoRA.alpha,
		Bias:       bias,
	})
}

func (model *hipLoadedModel) runTinyLoRAProjection(ctx context.Context, cfg hipLoadedTinyLMConfig, hidden []float32) ([]float32, int, float32, error) {
	if model == nil || model.tinyLoRA == nil {
		return nil, 0, 0, core.E("rocm.hip.TinyLoRA", "active LoRA adapter is required", nil)
	}
	base, err := model.loadedTinyOutputWeights(cfg)
	if err != nil {
		return nil, 0, 0, err
	}
	logits, err := hipRunLoRAProjectionKernel(ctx, model.driver, hipLoRAProjectionRequest{
		Input:      hidden,
		BaseWeight: base,
		LoRAA:      model.tinyLoRA.a,
		LoRAB:      model.tinyLoRA.b,
		Rows:       cfg.VocabSize,
		Cols:       cfg.HiddenSize,
		Rank:       model.tinyLoRA.rank,
		Alpha:      model.tinyLoRA.alpha,
		Bias:       model.tinyLoRA.bias,
	})
	if err != nil {
		return nil, 0, 0, err
	}
	next, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return nil, 0, 0, err
	}
	return logits, next, score, nil
}

func (model *hipLoadedModel) runSmallLoRAProjection(ctx context.Context, cfg hipLoadedSmallDecodeConfig, hidden []float32) ([]float32, int, float32, error) {
	if model == nil || model.smallLoRA == nil {
		return nil, 0, 0, core.E("rocm.hip.SmallLoRA", "active small LM-head LoRA adapter is required", nil)
	}
	base, err := model.loadedSmallLMHeadWeightsF32(cfg)
	if err != nil {
		return nil, 0, 0, err
	}
	logits, err := hipRunLoRAProjectionKernel(ctx, model.driver, hipLoRAProjectionRequest{
		Input:      hidden,
		BaseWeight: base,
		LoRAA:      model.smallLoRA.a,
		LoRAB:      model.smallLoRA.b,
		Rows:       cfg.VocabSize,
		Cols:       cfg.HiddenSize,
		Rank:       model.smallLoRA.rank,
		Alpha:      model.smallLoRA.alpha,
		Bias:       model.smallLoRA.bias,
	})
	if err != nil {
		return nil, 0, 0, err
	}
	next, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return nil, 0, 0, err
	}
	return logits, next, score, nil
}

func hipTinyAttentionWeightedOutput(values, weights []float32, hiddenSize int) ([]float32, error) {
	if hiddenSize <= 0 {
		return nil, core.E("rocm.hip.TinyLoRA", "hidden size must be positive", nil)
	}
	if len(weights) == 0 || len(values) != len(weights)*hiddenSize {
		return nil, core.E("rocm.hip.TinyLoRA", "attention values must align with weights and hidden size", nil)
	}
	out := make([]float32, hiddenSize)
	for token := range weights {
		for dim := 0; dim < hiddenSize; dim++ {
			out[dim] += weights[token] * values[token*hiddenSize+dim]
		}
	}
	return out, nil
}

func (model *hipLoadedModel) addClassifierLoRALabels(labels map[string]string) {
	if model == nil || model.classLoRA == nil || labels == nil {
		return
	}
	labels["adapter_hash"] = model.classLoRA.identity.Hash
	labels["adapter_runtime"] = "hip_bert_classifier"
	labels["lora_kernel"] = hipKernelStatusLinked
	labels["lora_kernel_name"] = hipKernelNameLoRA
	labels["lora_model_status"] = "experimental_bert_sequence_classifier"
}

func (model *hipLoadedModel) addTinyLoRALabels(labels map[string]string) {
	if model == nil || model.tinyLoRA == nil || labels == nil {
		return
	}
	labels["adapter_hash"] = model.tinyLoRA.identity.Hash
	labels["adapter_runtime"] = "hip_tiny_loaded"
	labels["lora_kernel"] = hipKernelStatusLinked
	labels["lora_kernel_name"] = hipKernelNameLoRA
	labels["lora_model_status"] = "experimental_tiny_loaded"
}

func (model *hipLoadedModel) addSmallLoRALabels(labels map[string]string) {
	if model == nil || model.smallLoRA == nil || labels == nil {
		return
	}
	labels["adapter_hash"] = model.smallLoRA.identity.Hash
	labels["adapter_runtime"] = "hip_small_lm_head"
	labels["lora_kernel"] = hipKernelStatusLinked
	labels["lora_kernel_name"] = hipKernelNameLoRA
	labels["lora_model_status"] = hipSmallDecodeLoRAModelStatus(model.modelInfo.Architecture)
}
