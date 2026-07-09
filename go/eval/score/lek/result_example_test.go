// SPDX-Licence-Identifier: EUPL-1.2

package lek_test

import (
	"encoding/json"
	"fmt"

	"dappco.re/go/inference/eval/score/lek"
)

// Example shows the unified result wire shape. result.go declares only
// the JSON-tagged result structs (ScoreResult, DiffResult, …) that the
// scorer serialises; this example demonstrates the documented contract —
// optional slots are omitted when empty, so a result carrying only a
// sycophancy read marshals to a compact object.
func Example() {
	r := lek.ScoreResult{
		Sycophancy: &lek.SycophancyInfo{
			Tier:      lek.TierSoftAgreement,
			Label:     lek.TierLabel(lek.TierSoftAgreement),
			Composite: 10,
		},
	}
	blob, _ := json.Marshal(r)
	fmt.Println(string(blob))

	// A pair result always carries both sides, even when empty.
	blob, _ = json.Marshal(lek.DiffResult{})
	fmt.Println(string(blob))
	// Output:
	// {"sycophancy":{"tier":1,"label":"soft_agreement","composite":10}}
	// {"prompt":{},"response":{}}
}
