// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// ExampleContentHash computes the canonicalised content hash used for
// within-dataset dedupe.
func ExampleContentHash() {
	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	r := dataset.ContentHash(dataset.KindPair, content)
	hash := r.Value.(string)
	core.Println(r.OK, len(hash))
	// Output:
	// true 64
}

// ExampleManifestHash computes an export manifest's receipt hash over
// the ordered content hashes a training run saw.
func ExampleManifestHash() {
	hash := dataset.ManifestHash([]string{"contenthash-a", "contenthash-b"})
	core.Println(len(hash))
	// Output:
	// 64
}
