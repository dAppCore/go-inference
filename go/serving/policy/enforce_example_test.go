// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"

	core "dappco.re/go"
)

// ExamplePolicy_NewMediatingEnforcer shows a grade-G2 rewrite: the matched span
// is handed to the mediator, whose transform is emitted in its place, and the
// rest of the reply streams through unchanged.
func ExamplePolicy_NewMediatingEnforcer() {
	pol, _ := Compile([]byte(`{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`))
	// A real deployment reroutes the span through its own transform (a model, a
	// lookup table, …); here it simply brackets it.
	mediate := func(_ context.Context, _ int, span string) (string, error) {
		return "«" + span + "»", nil
	}
	enf := pol.NewMediatingEnforcer(context.Background(), mediate)
	out, _, _ := enf.Feed("ship PROJECT-X now")
	rest, _, _ := enf.Close()
	core.Println(out + rest)
	// Output:
	// ship «PROJECT-X» now
}
