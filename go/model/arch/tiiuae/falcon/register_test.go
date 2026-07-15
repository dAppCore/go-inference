// SPDX-Licence-Identifier: EUPL-1.2

package falcon

import (
	"testing"

	"dappco.re/go/inference/model"
)

// Tensor names are from https://huggingface.co/tiiuae/falcon-7b/blob/main/model.safetensors.index.json
// (falcon-rw-1b is unsharded and publishes no index).
func TestRegisterWeightMapping_Good(t *testing.T) {
	spec, ok := model.LookupArch("falcon")
	if !ok {
		t.Fatal("falcon not registered")
	}
	if spec.Weights.Embed != "transformer.word_embeddings" || spec.Weights.Q != ".self_attention.query" || spec.Weights.K != ".self_attention.key" || spec.Weights.V != ".self_attention.value" {
		t.Fatalf("Falcon weight mapping = %+v", spec.Weights)
	}
}
