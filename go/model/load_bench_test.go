// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The ProbeModelTypes bench baselines the reactive loader's front-door probe (AX-11): the
// once-per-load JSON peek at config.json for the top-level model_type and the nested
// text_config.model_type (multimodal wrappers carry both), which the registry keys on to
// route the checkpoint. It is JSON-unmarshal bound; its allocation is the probe-struct
// decode. Synthetic config bytes — no checkpoint read (the disk path is Load, not benched
// here because it needs a real model directory).

func benchConfigJSON() []byte {
	return []byte(`{"model_type":"gemma4","architectures":["Gemma4ForConditionalGeneration"],` +
		`"text_config":{"model_type":"gemma4_text","hidden_size":2048,"num_hidden_layers":34,` +
		`"num_attention_heads":8,"num_key_value_heads":2,"head_dim":256,"vocab_size":262144,` +
		`"max_position_embeddings":131072,"rms_norm_eps":1e-06},"vision_config":{"hidden_size":1152}}`)
}

// BenchmarkProbeModelTypes — the loader's model_type probe: a JSON unmarshal into the
// two-field probe struct, discarding the rest of the config. The decode is the whole cost.
func BenchmarkProbeModelTypes(b *testing.B) {
	data := benchConfigJSON()
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mt, textMT := ProbeModelTypes(data)
		if mt != "gemma4" || textMT != "gemma4_text" {
			b.Fatal("probe mis-parsed the model_type ids")
		}
	}
}
