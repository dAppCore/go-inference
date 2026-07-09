// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/inference/model/safetensors"
)

// exampleFs is the filesystem handle these examples use for temp dirs (mirroring
// safetensors' example fixtures).
var exampleFs = (&core.Fs{}).New("/")

// ExampleLoad shows the whole reactive checkpoint load in one call: read config.json,
// probe model_type, dispatch to the registered ArchSpec, mmap the safetensors, assemble
// the neutral LoadedModel. dm.Close() must be called once the weight byte-views are done
// (or handed to a device buffer).
func ExampleLoad() {
	RegisterArch(ArchSpec{
		ModelTypes: []string{"exampleload"},
		Parse:      func([]byte) (ArchConfig, error) { return fakeLoadArchConfig{hidden: 4}, nil },
		Weights:    WeightNames{Embed: "embed", FinalNorm: "norm.weight"},
	})
	r := exampleFs.TempDir("exampleload")
	if !r.OK {
		return
	}
	dir := r.Value.(string)
	defer exampleFs.DeleteAll(dir)
	_ = coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"exampleload"}`)
	_ = safetensors.WriteSafetensors(core.PathJoin(dir, "model.safetensors"),
		map[string]safetensors.SafetensorsTensorInfo{
			"embed.weight": {Dtype: "F32", Shape: []int{8, 4}},
			"norm.weight":  {Dtype: "F32", Shape: []int{4}},
		},
		map[string][]byte{
			"embed.weight": make([]byte, 8*4*4),
			"norm.weight":  make([]byte, 4*4),
		})

	m, dm, err := Load(dir)
	if err != nil {
		return
	}
	defer func() { _ = dm.Close() }()
	core.Println(m.Embed.OutDim) // the vocab, read from the embed tensor's shape
	// Output: 8
}

// ExampleProbeDirArch shows the front-door check a backend uses to route a checkpoint
// whose loader is NOT the reactive Assemble path: it reads config.json's model_type
// without dispatching anywhere.
func ExampleProbeDirArch() {
	r := exampleFs.TempDir("exampleprobe")
	if !r.OK {
		return
	}
	dir := r.Value.(string)
	defer exampleFs.DeleteAll(dir)
	_ = coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"mamba2"}`)
	mt, _, err := ProbeDirArch(dir)
	if err != nil {
		return
	}
	core.Println(mt)
	// Output: mamba2
}

// ExampleProbeModelTypes shows resolving a config's architecture id(s) without
// re-parsing the JSON: the top-level model_type, plus a multimodal wrapper's nested
// text_config.model_type when present.
func ExampleProbeModelTypes() {
	mt, text := ProbeModelTypes([]byte(`{"model_type":"wrap_unified","text_config":{"model_type":"wrap_text"}}`))
	core.Println(mt)
	core.Println(text)
	// Output:
	// wrap_unified
	// wrap_text
}
