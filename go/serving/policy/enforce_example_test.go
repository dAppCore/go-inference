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
	// lookup table, …). The mediator output is re-enforced once before emission, so
	// the transform must genuinely sanitise the span — here it rewords it.
	mediate := func(_ context.Context, _ int, _ string) (string, error) {
		return "our flagship", nil
	}
	enf := pol.NewMediatingEnforcer(context.Background(), mediate)
	out, _, _ := enf.Feed("ship PROJECT-X now")
	rest, _, _ := enf.Close()
	core.Println(out + rest)
	// Output:
	// ship our flagship now
}

// ExamplePolicy_NewMediatingEnforcer_reEnforced shows the untrusted-mediator
// guard: a mediator (perhaps the model itself) that fails to sanitise — echoing
// the banned term straight back — never gets to emit it. The single
// re-enforcement pass over the mediator output redacts the residual hit.
func ExamplePolicy_NewMediatingEnforcer_reEnforced() {
	pol, _ := Compile([]byte(`{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`))
	echo := func(_ context.Context, _ int, span string) (string, error) {
		return "our " + span + " team", nil
	}
	enf := pol.NewMediatingEnforcer(context.Background(), echo)
	out, _, _ := enf.Feed("ship PROJECT-X now")
	rest, _, _ := enf.Close()
	core.Println(out + rest)
	// Output:
	// ship our [redacted] team now
}
