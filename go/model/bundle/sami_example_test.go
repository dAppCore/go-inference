// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	core "dappco.re/go"
	"dappco.re/go/inference/kv"
)

func ExampleSAMIFromKV() {
	snapshot := exampleBundleSnapshot()
	sami := SAMIFromKV(snapshot, kv.Analyze(snapshot), SAMIOptions{
		Model:  "gemma4-e2b",
		Prompt: "draft the next section",
	})

	core.Println(sami.Model, sami.Architecture, sami.NumLayers, len(sami.LayerCoherence))
	// Output: gemma4-e2b gemma4_text 1 1
}
