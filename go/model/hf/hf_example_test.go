// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"fmt"

	core "dappco.re/go"
)

// ExampleNewRemoteSource constructs a Hugging Face Hub metadata source. The
// constructor trims a trailing slash from the base URL and defaults the
// user-agent when none is supplied — no network is touched here.
func ExampleNewRemoteSource() {
	source := NewRemoteSource(RemoteConfig{
		BaseURL: "https://huggingface.co/",
	})
	fmt.Println(source.baseURL, source.userAgent)
	// Output: https://huggingface.co go-inference
}

// ExampleRemoteSource_SearchModels queries the Hub model-search endpoint. The
// example points the source at a loopback test server (no real network) that
// returns one model, so the result is deterministic.
func ExampleRemoteSource_SearchModels() {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `[{"id": "Qwen/Qwen3-0.6B", "config": {"model_type": "qwen3"}}]`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	models, err := source.SearchModels(context.Background(), "qwen", 5)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(models), models[0].ID, models[0].Config.ModelType)
	// Output: 1 Qwen/Qwen3-0.6B qwen3
}

// ExampleRemoteSource_ModelMetadata fetches metadata for a single model id.
// The example points the source at a loopback test server (no real
// network); when the body carries no `id`/`modelId`, the requested id is
// filled in.
func ExampleRemoteSource_ModelMetadata() {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{"modelId": "Qwen/Qwen3-0.6B", "config": {"model_type": "qwen3", "num_hidden_layers": 28}}`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	meta, err := source.ModelMetadata(context.Background(), "Qwen/Qwen3-0.6B")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(meta.ModelID, meta.Config.ModelType, meta.Config.NumHiddenLayers)
	// Output: Qwen/Qwen3-0.6B qwen3 28
}

// ExampleInspectLocalMetadata resolves a synthetic `models--org--name` cache
// directory (built on a temp dir here, in place of a real
// ~/.cache/huggingface/hub entry) into its metadata + snapshot root.
func ExampleInspectLocalMetadata() {
	baseResult := core.MkdirTemp("", "hf-example-*")
	if !baseResult.OK {
		fmt.Println("tempdir failed")
		return
	}
	base := baseResult.Value.(string)
	defer core.RemoveAll(base)

	cacheRoot := core.PathJoin(base, "models--org--name")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "abc123")
	core.MkdirAll(snapshot, 0o755)
	core.WriteFile(core.PathJoin(snapshot, "config.json"), []byte(`{"model_type":"qwen3"}`), 0o644)
	core.WriteFile(core.PathJoin(snapshot, "model.safetensors"), []byte("weights"), 0o644)

	meta, root, err := InspectLocalMetadata(cacheRoot)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(meta.ID, meta.Config.ModelType, len(meta.Files), root == snapshot)
	// Output: org/name qwen3 1 true
}

// ExampleLocalModelID decodes the HuggingFace `models--org--name` cache
// directory convention back to an "org/name" model id.
func ExampleLocalModelID() {
	root := "/cache/models--mlx-community--gemma-4-e2b-it-4bit"
	snapshot := core.PathJoin(root, "snapshots", "abc123")
	fmt.Println(LocalModelID(snapshot, root))
	// Output: mlx-community/gemma-4-e2b-it-4bit
}

// ExampleWeightFormatAndBytes inspects a resolved file list and reports the
// predominant weight format plus the summed weight byte size.
func ExampleWeightFormatAndBytes() {
	files := []ModelFile{
		{Name: "model-00001-of-00002.safetensors", Size: 100},
		{Name: "model-00002-of-00002.safetensors", Size: 200},
		{Name: "tokenizer.json", Size: 5}, // not a weight file, excluded
	}
	format, total := WeightFormatAndBytes(files)
	fmt.Println(format, total)
	// Output: safetensors 300
}

// ExampleInferJANG shows JANG metadata inference from a model's id, tags and
// filenames. A "jangtq" token (here in the tag list) selects the fixed
// JANGTQ profile; the group size falls back to 64 when no quantization
// block declares one. The filename is only a needle — quant width comes
// from the profile, not the file name.
func ExampleInferJANG() {
	info := InferJANG(ModelMetadata{
		ID:   "dealignai/MiniMax-M2-JANGTQ",
		Tags: []string{"mlx", "jang", "jangtq"},
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
	})
	fmt.Println(info.Profile, info.WeightFormat, info.BitsDefault, info.GroupSize)
	// Output: JANGTQ mxtq 2 64
}

// ExampleInferJANG_filenameNeedle shows that the JANGTQ profile is selected
// from a weight *filename* alone — neither the id nor the tags carry a
// needle here. A "jangtq" filename is the strongest signal and pins the
// JANGTQ profile (2-bit MXTQ, group size 64) just as a tag would.
func ExampleInferJANG_filenameNeedle() {
	info := InferJANG(ModelMetadata{
		ID: "acme/MiniMax-M2",
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
	})
	fmt.Println(info.Profile, info.WeightFormat, info.BitsDefault, info.GroupSize)
	// Output: JANGTQ mxtq 2 64
}

// ExampleInferJANG_noNeedle shows the negative result: a model with no JANG
// needle in its id, tags or filenames is not a JANG model, so InferJANG
// returns nil. Callers treat nil as "ordinary (non-JANG) weights".
func ExampleInferJANG_noNeedle() {
	info := InferJANG(ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}},
	})
	fmt.Println(info == nil)
	// Output: true
}
