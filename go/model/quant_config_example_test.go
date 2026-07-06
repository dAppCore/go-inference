// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleQuantConfig_UnmarshalJSON shows the mlx quantization block parse: the scalar
// group_size/bits/mode become the default, and any object-valued key becomes a
// per-module override (its "language_model." wrapper prefix stripped to match the
// normalised tensor names the assembler builds).
func ExampleQuantConfig_UnmarshalJSON() {
	var q QuantConfig
	data := []byte(`{"group_size":64,"bits":4,"mode":"affine","language_model.model.layers.0.mlp.experts":{"bits":8}}`)
	if err := q.UnmarshalJSON(data); err != nil {
		return
	}
	core.Println(q.Bits)                                  // the block-level default
	core.Println(q.Overrides["model.layers.0.mlp.experts"].Bits) // the stripped per-module override
	// Output:
	// 4
	// 8
}

// ExampleQuantConfig_For shows the per-tensor resolve: an overridden module gets its own
// (groupSize,bits); anything else falls back to the block's default.
func ExampleQuantConfig_For() {
	q := &QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]ModuleQuant{
		"model.layers.0.mlp.experts": {GroupSize: 32, Bits: 8},
	}}
	gs, bits := q.For("model.layers.0.mlp.experts")
	core.Println(gs, bits)
	gs, bits = q.For("model.layers.0.self_attn.q_proj")
	core.Println(gs, bits)
	// Output:
	// 32 8
	// 64 4
}

// ExampleNormalizeQuantizationMode shows the mode normaliser: mixed case and surrounding
// whitespace lowercase-trim to the declared mode, and an absent mode defaults to affine.
func ExampleNormalizeQuantizationMode() {
	core.Println(NormalizeQuantizationMode("  MXFP4  "))
	core.Println(NormalizeQuantizationMode(""))
	// Output:
	// mxfp4
	// affine
}

// ExampleQuantConfig_Validate shows the quant-config validator: a supported
// (mode,bits,group_size) combination passes; an unsupported one is rejected.
func ExampleQuantConfig_Validate() {
	good := &QuantConfig{Mode: "affine", GroupSize: 64, Bits: 4}
	bad := &QuantConfig{Mode: "affine", Bits: 7}
	core.Println(good.Validate())
	core.Println(bad.Validate() != nil)
	// Output:
	// <nil>
	// true
}
