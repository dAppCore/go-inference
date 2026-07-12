// SPDX-Licence-Identifier: EUPL-1.2

package bloom

import (
	"testing"

	"dappco.re/go/inference/model"
)

// Tensor names are from https://huggingface.co/bigscience/bloom-7b1/blob/main/model.safetensors.index.json
// (bloom-560m is unsharded and publishes no index).
func TestRegisterWeightMapping_Good(t *testing.T) {
	spec, ok := model.LookupArch("bloom")
	if !ok {
		t.Fatal("bloom not registered")
	}
	if spec.Weights.Embed != "word_embeddings" || spec.Weights.EmbedNorm != "word_embeddings_layernorm.weight" || spec.Weights.Q != ".self_attention.query" {
		t.Fatalf("BLOOM weight mapping = %+v", spec.Weights)
	}
}

// Fixture source: https://huggingface.co/bigscience/bloom-560m/blob/main/tokenizer_config.json
func TestMultilingualTokenizerFixture_Good(t *testing.T) {
	if bloomTokenizerFixture != `{"add_prefix_space":false,"model_max_length":1000000000000000019884624838656,"special_tokens_map_file":null,"tokenizer_class":null}` {
		t.Fatal("BLOOM multilingual tokenizer fixture changed")
	}
}

const bloomTokenizerFixture = `{"add_prefix_space":false,"model_max_length":1000000000000000019884624838656,"special_tokens_map_file":null,"tokenizer_class":null}`
