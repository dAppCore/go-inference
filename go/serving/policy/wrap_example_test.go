// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ExampleWrapResolverMediated shows the grade-G2 wrapper decorating a resolver so
// every resolved model's output routes rewrite hits through the mediator. Tokens
// arrive split across the span; the mediator still sees it whole, and its
// (sanitised) output is re-enforced once before it reaches the client.
func ExampleWrapResolverMediated() {
	pol, _ := Compile([]byte(`{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`))
	mediate := func(_ context.Context, _ int, _ string) (string, error) {
		return "our flagship", nil
	}
	fake := &policyFakeModel{tokens: []string{"the PROJ", "ECT-X ships"}}
	resolver := WrapResolverMediated(resolverOf(fake), pol, nil, mediate)

	model, _ := resolver.ResolveModel(context.Background(), "demo")
	var b core.Builder
	for tok := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		b.WriteString(tok.Text)
	}
	core.Println(b.String())
	// Output:
	// the our flagship ships
}
