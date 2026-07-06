// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// ExampleLoadLinear shows the per-weight quant decision: prefix+".scales" present makes
// a quantised Linear (geometry derived from the tensor shapes), and it is absent when
// the checkpoint holds no weight at all under that prefix.
func ExampleLoadLinear() {
	tensors := map[string]safetensors.Tensor{
		"w.weight": {Shape: []int{4, 8}, Data: make([]byte, 4*8/8)},  // 4-bit packed
		"w.scales": {Shape: []int{4, 2}, Data: make([]byte, 4*2)},
	}
	l := LoadLinear(tensors, "w", 64, "affine")
	core.Println(l.Quantised())
	core.Println(l.GroupSize)
	// Output:
	// true
	// 32
}

// ExampleLinear_Quantised shows the per-weight format check a backend's matvec dispatch
// reads: both a scales tensor AND a declared Kind are required for "quantised".
func ExampleLinear_Quantised() {
	dense := &Linear{Weight: []byte{1, 2, 3, 4}}
	quantised := &Linear{Weight: []byte{1, 2}, Scales: []byte{1}, Kind: "affine"}
	core.Println(dense.Quantised())
	core.Println(quantised.Quantised())
	// Output:
	// false
	// true
}
