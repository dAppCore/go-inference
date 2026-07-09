// SPDX-Licence-Identifier: EUPL-1.2

package inference_test

import (
	"fmt"

	"dappco.re/go/inference"
)

type exampleSpeculativeModel struct{}

func (exampleSpeculativeModel) SpeculativeMetrics() inference.SpeculativeMetrics {
	return inference.SpeculativeMetrics{ProposedTokens: 12, AcceptedTokens: 9, AcceptanceRate: 0.75}
}

// SpeculativeMetricsProvider is probed off a model the same way as
// inference.AttentionInspector — models without a speculative lane simply
// don't implement it.
func ExampleSpeculativeMetricsProvider() {
	var model any = exampleSpeculativeModel{}
	if p, ok := model.(inference.SpeculativeMetricsProvider); ok {
		sm := p.SpeculativeMetrics()
		fmt.Printf("accepted %d of %d (%.0f%%)\n", sm.AcceptedTokens, sm.ProposedTokens, sm.AcceptanceRate*100)
	}
	// Output: accepted 9 of 12 (75%)
}
