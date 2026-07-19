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

// exampleSpeculativePairBackend implements ONLY SpeculativePairBackend — a
// registered inference.Backend (the metal engine's metalBackend) satisfies it
// alongside Backend itself, but the probe below only needs this one method.
type exampleSpeculativePairBackend struct{}

func (exampleSpeculativePairBackend) LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	return nil, nil // a real engine returns a loaded speculative TextModel
}

// SpeculativePairBackend is probed off a REGISTERED BACKEND, one step earlier
// than inference.SpeculativeMetricsProvider (a capability of an already-LOADED
// model): loading the pair is the operation being asked for, so there is no
// model yet to probe.
func ExampleSpeculativePairBackend() {
	var b any = exampleSpeculativePairBackend{}
	if spl, ok := b.(inference.SpeculativePairBackend); ok {
		_, err := spl.LoadSpeculativePair("/models/target", "/models/draft", 5)
		fmt.Println(err == nil)
	}
	// Output: true
}
