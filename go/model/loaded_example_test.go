// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleLoadedModel_Tied shows the tied-vs-separate lm_head check: a nil LMHead means the
// LM head reuses the token embedding (the common weight-tying convention).
func ExampleLoadedModel_Tied() {
	tied := &LoadedModel{Embed: &Linear{OutDim: 8}}
	separate := &LoadedModel{Embed: &Linear{OutDim: 8}, LMHead: &Linear{OutDim: 8}}
	core.Println(tied.Tied())
	core.Println(separate.Tied())
	// Output:
	// true
	// false
}

// ExampleLoadedModel_ValidateRequired shows the always-present-weights check: a
// well-formed single dense layer passes, while a missing final norm is rejected before
// the decode ever runs.
func ExampleLoadedModel_ValidateRequired() {
	arch := Arch{Layer: []LayerSpec{{CacheIndex: 0}}}
	m := &LoadedModel{
		Embed: &Linear{OutDim: 4}, FinalNorm: []byte{1, 2},
		Layers: []LoadedLayer{{
			AttnNorm: []byte{1, 2}, Q: &Linear{}, K: &Linear{}, O: &Linear{},
			MLPNorm: []byte{1, 2}, Gate: &Linear{}, Up: &Linear{}, Down: &Linear{},
		}},
	}
	core.Println(m.ValidateRequired(arch))
	m.FinalNorm = nil
	core.Println(m.ValidateRequired(arch) != nil)
	// Output:
	// <nil>
	// true
}
