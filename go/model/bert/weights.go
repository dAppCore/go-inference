// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// bindWeights pulls the named BertModel tensors out of a decoded safetensors map
// and validates every span against the config shape, so a truncated or mis-keyed
// checkpoint fails at load rather than producing silent garbage at inference.
func bindWeights(cfg Config, tensors map[string]safetensors.Tensor) (*Weights, error) {
	hiddenSize := cfg.HiddenSize
	weights := &Weights{}
	var err error

	if weights.wordEmbeddings, err = tensorFloats(tensors, "embeddings.word_embeddings.weight", cfg.VocabSize*hiddenSize); err != nil {
		return nil, err
	}
	if weights.posEmbeddings, err = tensorFloats(tensors, "embeddings.position_embeddings.weight", cfg.MaxPositionEmbeddings*hiddenSize); err != nil {
		return nil, err
	}
	if weights.typeEmbeddings, err = tensorFloats(tensors, "embeddings.token_type_embeddings.weight", cfg.TypeVocabSize*hiddenSize); err != nil {
		return nil, err
	}
	if weights.embLNW, err = tensorFloats(tensors, "embeddings.LayerNorm.weight", hiddenSize); err != nil {
		return nil, err
	}
	if weights.embLNB, err = tensorFloats(tensors, "embeddings.LayerNorm.bias", hiddenSize); err != nil {
		return nil, err
	}

	weights.layers = make([]layerWeights, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("encoder.layer.%d.", i)
		layer := &weights.layers[i]
		binds := []struct {
			dst   *[]float32
			name  string
			count int
		}{
			{&layer.queryW, prefix + "attention.self.query.weight", hiddenSize * hiddenSize},
			{&layer.queryB, prefix + "attention.self.query.bias", hiddenSize},
			{&layer.keyW, prefix + "attention.self.key.weight", hiddenSize * hiddenSize},
			{&layer.keyB, prefix + "attention.self.key.bias", hiddenSize},
			{&layer.valueW, prefix + "attention.self.value.weight", hiddenSize * hiddenSize},
			{&layer.valueB, prefix + "attention.self.value.bias", hiddenSize},
			{&layer.attnDenseW, prefix + "attention.output.dense.weight", hiddenSize * hiddenSize},
			{&layer.attnDenseB, prefix + "attention.output.dense.bias", hiddenSize},
			{&layer.attnLNW, prefix + "attention.output.LayerNorm.weight", hiddenSize},
			{&layer.attnLNB, prefix + "attention.output.LayerNorm.bias", hiddenSize},
			{&layer.interW, prefix + "intermediate.dense.weight", cfg.IntermediateSize * hiddenSize},
			{&layer.interB, prefix + "intermediate.dense.bias", cfg.IntermediateSize},
			{&layer.outW, prefix + "output.dense.weight", hiddenSize * cfg.IntermediateSize},
			{&layer.outB, prefix + "output.dense.bias", hiddenSize},
			{&layer.outLNW, prefix + "output.LayerNorm.weight", hiddenSize},
			{&layer.outLNB, prefix + "output.LayerNorm.bias", hiddenSize},
		}
		for _, bind := range binds {
			values, bindErr := tensorFloats(tensors, bind.name, bind.count)
			if bindErr != nil {
				return nil, bindErr
			}
			*bind.dst = values
		}
	}
	return weights, nil
}

// tensorFloats decodes one named tensor to float32 and checks its element count.
// It accepts any float dtype the safetensors decoder understands (F32/F16/BF16),
// so a half-precision encoder checkpoint still loads onto the host path.
func tensorFloats(tensors map[string]safetensors.Tensor, name string, want int) ([]float32, error) {
	tensor, ok := tensors[name]
	if !ok {
		return nil, core.E("bert.bindWeights", "missing tensor "+name, nil)
	}
	elements := 1
	for _, dim := range tensor.Shape {
		elements *= dim
	}
	values, err := safetensors.DecodeFloatData(tensor.Dtype, tensor.Data, elements)
	if err != nil {
		return nil, core.E("bert.bindWeights", "decode tensor "+name, err)
	}
	if len(values) != want {
		return nil, core.E("bert.bindWeights", core.Sprintf("tensor %s has %d elements, want %d", name, len(values), want), nil)
	}
	return values, nil
}

// lowerCaseFromSnapshot reads do_lower_case from sentence_bert_config.json (then
// tokenizer_config.json), defaulting to true — bge/MiniLM/E5 are all uncased.
func lowerCaseFromSnapshot(dir string) bool {
	for _, name := range []string{"sentence_bert_config.json", "tokenizer_config.json"} {
		result := core.ReadFile(core.PathJoin(dir, name))
		if !result.OK {
			continue
		}
		var probe struct {
			DoLowerCase *bool `json:"do_lower_case"`
		}
		if r := core.JSONUnmarshal(result.Value.([]byte), &probe); r.OK && probe.DoLowerCase != nil {
			return *probe.DoLowerCase
		}
	}
	return true
}

// poolingFromSnapshot reads the sentence-transformers 1_Pooling/config.json to
// choose CLS vs mean, and detects a 2_Normalize module for the normalise flag.
// A snapshot without those files defaults to mean pooling with normalisation —
// the common bare-BertModel-as-embedder convention.
func poolingFromSnapshot(dir string) (Pooling, bool) {
	pooling := PoolingMean
	result := core.ReadFile(core.PathJoin(dir, "1_Pooling", "config.json"))
	if result.OK {
		var probe struct {
			CLS  bool `json:"pooling_mode_cls_token"`
			Mean bool `json:"pooling_mode_mean_tokens"`
		}
		if r := core.JSONUnmarshal(result.Value.([]byte), &probe); r.OK {
			switch {
			case probe.CLS:
				pooling = PoolingCLS
			case probe.Mean:
				pooling = PoolingMean
			}
		}
	}
	return pooling, normaliseFromModules(dir)
}

// normaliseFromModules reports whether modules.json declares a Normalize module.
// When modules.json is absent it defaults to true — an embedder snapshot without
// module metadata is conventionally used with unit vectors.
func normaliseFromModules(dir string) bool {
	result := core.ReadFile(core.PathJoin(dir, "modules.json"))
	if !result.OK {
		return true
	}
	var modules []struct {
		Type string `json:"type"`
	}
	if r := core.JSONUnmarshal(result.Value.([]byte), &modules); !r.OK {
		return true
	}
	for _, module := range modules {
		if core.Contains(core.Lower(module.Type), "normalize") {
			return true
		}
	}
	return false
}
