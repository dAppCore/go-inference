// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

func ExampleHostility() {
	// Stacked directed insults + an exclamation run → strong, person-
	// directed hostility — the AngerScore the welfare layer gates on.
	h := lek.Hostility("you useless idiot, you absolute moron!!!")
	fmt.Println("lexicon hits:", h.LexiconHits)
	fmt.Println("directed:", h.Directed)
	fmt.Println("exclaim run:", h.ExclaimRun)
	fmt.Println("score:", h.Score)
	// Output:
	// lexicon hits: 3
	// directed: true
	// exclaim run: 3
	// score: 0.75
}
