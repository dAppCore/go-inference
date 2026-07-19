// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

func ExampleLoadWeights() {
	tensors, cfg := syntheticCheckpoint()
	w, err := LoadWeights(tensors, cfg)
	core.Println(err == nil, len(w.Vision.Blocks), len(w.Text.Layers))
	// Output: true 1 1
}
