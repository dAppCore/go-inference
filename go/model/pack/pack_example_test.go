// SPDX-Licence-Identifier: EUPL-1.2

package pack_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/pack"
)

// ExamplePack shows how to wrap an unpacked safetensors pack into a
// .model Trix container.
func ExamplePack() {
	r := pack.Pack(
		"/tmp/gemma-3-4b-it",
		"/tmp/gemma-3-4b-it.model",
		pack.PackOptions{
			Manifest: pack.Manifest{
				Model: inference.ModelIdentity{
					ID:           "google/gemma-3-4b-it",
					Architecture: "gemma",
					QuantBits:    4,
				},
				Tokenizer: inference.TokenizerIdentity{
					Kind: "sentencepiece",
				},
				SourceFormat: "safetensors",
				Producer:     pack.Producer{Name: "go-mlx"},
			},
		},
	)
	if !r.OK {
		_ = r.Value
	}
}

// ExampleUnpack shows how to extract a .model back into a directory.
func ExampleUnpack() {
	r := pack.Unpack(
		"/tmp/gemma-3-4b-it.model",
		"/tmp/extracted",
		pack.UnpackOptions{},
	)
	if !r.OK {
		_ = r.Value
	}
}

// ExampleInspect shows how to read only the .model header and synthesise
// an inference.ModelPackInspection without extracting the payload.
func ExampleInspect() {
	manifest, inspection, r := pack.Inspect("/tmp/gemma-3-4b-it.model")
	if !r.OK {
		return
	}
	_ = manifest.Producer.Name
	_ = inspection.Model.Architecture
	_ = core.Sprintf("inspected %s", inspection.Path)
}

// ExampleExtractVindex shows how to read the embedded vindex blob back out
// of a .model that was packed with PackOptions.VindexBlob set. A nil blob
// with an OK Result means this pack carries no vindex — not an error.
func ExampleExtractVindex() {
	blob, r := pack.ExtractVindex("/tmp/gemma-3-4b-it.model")
	if !r.OK {
		return
	}
	if blob == nil {
		return // this pack carries no vindex
	}
	_ = core.Sprintf("vindex blob: %d bytes", len(blob))
}
