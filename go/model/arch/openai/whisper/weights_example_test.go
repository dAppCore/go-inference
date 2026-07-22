// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleLoadWeights() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	core.Println(err == nil, len(w.EncoderLayers), len(w.DecoderLayers))
	// Output: true 1 1
}
