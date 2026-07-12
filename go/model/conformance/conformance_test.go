// SPDX-Licence-Identifier: EUPL-1.2

// Package conformance is the cross-family CI contract for built-in model arches.
package conformance_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/builtin"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

type familyCase struct {
	modelType string
	config    string
	skip      string
}

const denseConfig = `"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":32,"rms_norm_eps":0.00001,"layer_norm_eps":0.00001,"rope_theta":10000,"max_position_embeddings":32,"tie_word_embeddings":false`

func registeredFamilies() []familyCase {
	dense := func(mt string) string { return `{"model_type":"` + mt + `",` + denseConfig + `}` }
	return []familyCase{
		{modelType: "bloom", config: `{"model_type":"bloom","n_embed":8,"n_inner":16,"n_layer":1,"num_attention_heads":2,"vocab_size":32}`},
		{modelType: "cohere", config: dense("cohere")},
		{modelType: "cohere2", config: `{"model_type":"cohere2",` + denseConfig + `,"sliding_window":16,"sliding_window_pattern":1}`},
		{modelType: "deepseek_v2", skip: "registered loader deliberately rejects MLA: deepseek.Load reports that its attention core is not implemented"},
		{modelType: "deepseek_v3", skip: "registered loader deliberately rejects MLA: deepseek.Load reports that its attention core is not implemented"},
		{modelType: "falcon", config: dense("falcon")},
		{modelType: "gemma3", skip: "BRIEF Gemma4 guard makes model/gemma3 and its shared Gemma engine paths untouchable on this lane"},
		{modelType: "gemma3_text", skip: "BRIEF Gemma4 guard makes model/gemma3 and its shared Gemma engine paths untouchable on this lane"},
		{modelType: "gemma4", skip: "BRIEF Gemma4 guard makes this architecture and its engine paths untouchable on this lane"},
		{modelType: "gemma4_text", skip: "BRIEF Gemma4 guard makes this architecture and its engine paths untouchable on this lane"},
		{modelType: "gemma4_unified", skip: "BRIEF Gemma4 guard makes this architecture and its engine paths untouchable on this lane"},
		{modelType: "diffusion_gemma", skip: "BRIEF Gemma4 guard makes this architecture and its engine paths untouchable on this lane"},
		{modelType: "gpt2", config: `{"model_type":"gpt2","n_embd":8,"n_head":2,"n_layer":1,"n_inner":16,"n_positions":32,"vocab_size":32}`},
		{modelType: "gpt_bigcode", config: `{"model_type":"gpt_bigcode","n_embd":8,"n_head":2,"n_layer":1,"n_inner":16,"n_positions":32,"vocab_size":32}`},
		{modelType: "starcoder", config: `{"model_type":"starcoder","n_embd":8,"n_head":2,"n_layer":1,"n_inner":16,"n_positions":32,"vocab_size":32}`},
		{modelType: "gpt_neox", config: dense("gpt_neox")},
		{modelType: "gptj", config: `{"model_type":"gptj","n_embd":8,"n_head":2,"n_layer":1,"n_inner":16,"vocab_size":32,"layer_norm_epsilon":0.00001}`},
		{modelType: "gpt_neo", config: `{"model_type":"gpt_neo","n_embd":8,"n_head":2,"n_layer":1,"n_inner":16,"vocab_size":32,"layer_norm_epsilon":0.00001}`},
		{modelType: "granite", config: `{"model_type":"granite",` + denseConfig + `,"logits_scaling":1,"residual_multiplier":1,"embedding_multiplier":1,"attention_multiplier":0.5}`},
		{modelType: "llama", config: dense("llama")},
		{modelType: "mistral3", config: dense("mistral3")}, {modelType: "ministral3", config: dense("ministral3")},
		{modelType: "mistral", config: dense("mistral")}, {modelType: "ministral", config: dense("ministral")},
		{modelType: "mixtral", skip: "sparse MoE synthetic construction is covered by model/mixtral integration; its registered Composed hook is not a dense Assemble path"},
		{modelType: "olmo", config: dense("olmo")}, {modelType: "olmo2", config: dense("olmo2")},
		{modelType: "opt", config: `{"model_type":"opt","hidden_size":8,"word_embed_proj_dim":8,"num_attention_heads":2,"num_hidden_layers":1,"ffn_dim":16,"max_position_embeddings":32,"vocab_size":32,"do_layer_norm_before":true,"tie_word_embeddings":true}`},
		{modelType: "phi", config: dense("phi")}, {modelType: "phi3", config: `{"model_type":"phi3",` + denseConfig + `,"num_key_value_heads":2}`},
		{modelType: "qwen2", config: dense("qwen2")}, {modelType: "qwen3", config: dense("qwen3")},
		{modelType: "starcoder2", config: dense("starcoder2")},
		{modelType: "qwen3_5", skip: "hybrid gated-delta construction needs family-specific recurrent tensors"},
		{modelType: "qwen3_5_text", skip: "hybrid gated-delta construction needs family-specific recurrent tensors"},
		{modelType: "qwen3_5_moe", skip: "hybrid gated-delta plus sparse-MoE construction needs family-specific tensors"},
		{modelType: "qwen3_5_moe_text", skip: "hybrid gated-delta plus sparse-MoE construction needs family-specific tensors"},
		{modelType: "qwen3_next", skip: "hybrid gated-delta construction needs family-specific recurrent tensors"},
	}
}

// TestBuiltinFamiliesConformance keeps registration honesty and the portable
// load/forward/generate surface in one table: adding a builtin without adding a
// row fails the registry round-trip exercised by the new package's own CI review.
func TestBuiltinFamiliesConformance(t *testing.T) {
	for _, tc := range registeredFamilies() {
		t.Run(tc.modelType, func(t *testing.T) {
			spec, ok := model.LookupArch(tc.modelType)
			if !ok || !contains(spec.ModelTypes, tc.modelType) {
				t.Fatalf("LookupArch(%q) did not round-trip its registered ModelTypes", tc.modelType)
			}
			if tc.skip != "" {
				t.Skip(tc.skip)
			}
			archConfig, err := spec.Parse([]byte(tc.config))
			if err != nil {
				t.Fatalf("parse cited family geometry: %v", err)
			}
			arch, err := archConfig.Arch()
			if err != nil {
				t.Fatalf("derive arch: %v", err)
			}
			weights := seededWeights(spec.Weights, arch)
			dir := t.TempDir()
			writeModel(t, dir, tc.config, weights)
			loaded, mapping, err := model.Load(dir)
			if err != nil {
				t.Fatalf("reactive load: %v", err)
			}
			_ = mapping.Close()
			if loaded.Arch.Hidden != arch.Hidden || len(loaded.Layers) != len(arch.Layer) {
				t.Fatalf("loaded shape = hidden %d layers %d, want %d/%d", loaded.Arch.Hidden, len(loaded.Layers), arch.Hidden, len(arch.Layer))
			}

			// The portable host executor consumes canonical tensor roles. Keeping
			// this copy separate from the on-disk family map makes the reactive
			// name mapping above part of the contract rather than bypassing it.
			hostConfig := core.Sprintf(`{"model_type":"%s","hidden_size":%d,"intermediate_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,"vocab_size":%d,"rms_norm_eps":0.00001}`, tc.modelType, arch.Hidden, arch.FF, len(arch.Layer), arch.Heads, arch.KVHeads, arch.HeadDim, arch.Vocab)
			host, err := composed.LoadComposed(canonicalWeights(arch), []byte(hostConfig))
			if err != nil {
				t.Fatalf("host load: %v", err)
			}
			tm := composed.NewTokenModel(host)
			inputs := make([][]byte, 3)
			for i, id := range []int32{1, 5, 9} {
				inputs[i], err = tm.Embed(id)
				if err != nil {
					t.Fatalf("embed: %v", err)
				}
			}
			hidden, err := tm.DecodeForward(inputs)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			if len(hidden) != 3 || len(hidden[0]) != arch.Hidden*2 {
				t.Fatalf("forward shape = [%d,%d]", len(hidden), len(hidden[0]))
			}
			for row := range hidden {
				for i := 0; i < len(hidden[row]); i += 2 {
					v := math.Float32frombits(uint32(hidden[row][i])<<16 | uint32(hidden[row][i+1])<<24)
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("hidden is not finite")
					}
				}
			}
			got, err := model.Generate(tm, []int32{1, 5, 9}, 4, -1)
			if err != nil || len(got) != 4 {
				t.Fatalf("greedy generate = %v, %v; want four tokens", got, err)
			}
		})
	}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

type prng struct{ state uint32 }

func (p *prng) tensor(rows, cols int) safetensors.Tensor {
	data := make([]byte, rows*cols*4)
	for i := 0; i < rows*cols; i++ {
		p.state = 1664525*p.state + 1013904223
		value := (float32(p.state>>8)/float32(1<<24) - 0.5) * 0.1
		bits := math.Float32bits(value)
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: []int{rows, cols}, Data: data}
}
func (p *prng) norm(n int) safetensors.Tensor {
	values := p.tensor(1, n)
	values.Shape = []int{n}
	for i := 0; i < n; i++ {
		bits := math.Float32bits(1 + float32(i+1)/100)
		values.Data[4*i], values.Data[4*i+1], values.Data[4*i+2], values.Data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return values
}

func seededWeights(names model.WeightNames, arch model.Arch) map[string]safetensors.Tensor {
	return weightsFor(names, arch, 0x5eed1234)
}
func canonicalWeights(arch model.Arch) map[string]safetensors.Tensor {
	names := model.StandardWeightNames()
	names.MLPNorm = ".post_attention_layernorm.weight"
	names.PostAttnNorm = ""
	names.PostFFNorm = ""
	names.QNorm = ""
	names.KNorm = ""
	return weightsFor(names, arch, 0x5eed1234)
}
func weightsFor(n model.WeightNames, a model.Arch, seed uint32) map[string]safetensors.Tensor {
	p := prng{state: seed}
	out := map[string]safetensors.Tensor{}
	put := func(name string, t safetensors.Tensor) {
		if name != "" {
			if !core.HasSuffix(name, ".weight") && name != n.FinalNorm && name != n.AttnNorm && name != n.MLPNorm {
				name += ".weight"
			}
			out[name] = t
		}
	}
	put(n.Embed, p.tensor(a.Vocab, a.Hidden))
	put(n.LMHead, p.tensor(a.Vocab, a.Hidden))
	put(n.FinalNorm, p.norm(a.Hidden))
	if n.PositionEmbed != "" {
		put(n.PositionEmbed, p.tensor(34, a.Hidden))
	}
	for i, l := range a.Layer {
		prefix := core.Sprintf(n.LayerPrefix, i)
		put(prefix+n.AttnNorm, p.norm(a.Hidden))
		put(prefix+n.MLPNorm, p.norm(a.Hidden))
		put(prefix+n.PostAttnNorm, p.norm(a.Hidden))
		put(prefix+n.PostFFNorm, p.norm(a.Hidden))
		put(prefix+n.Q, p.tensor(a.Heads*l.HeadDim, a.Hidden))
		put(prefix+n.K, p.tensor(l.KVHeads*l.HeadDim, a.Hidden))
		put(prefix+n.V, p.tensor(l.KVHeads*l.HeadDim, a.Hidden))
		put(prefix+n.O, p.tensor(a.Hidden, a.Heads*l.HeadDim))
		put(prefix+n.Gate, p.tensor(a.FF, a.Hidden))
		if n.Up != n.Gate {
			put(prefix+n.Up, p.tensor(a.FF, a.Hidden))
		}
		put(prefix+n.Down, p.tensor(a.Hidden, a.FF))
	}
	return out
}

func writeModel(t *testing.T, dir, config string, weights map[string]safetensors.Tensor) {
	t.Helper()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
		t.Fatal(err)
	}
	blob, err := safetensors.Encode(weights)
	if err != nil {
		t.Fatal(err)
	}
	if err = coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatal(err)
	}
}
