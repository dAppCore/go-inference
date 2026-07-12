// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ParallelResidual adds attention and MLP results, computed from the same
// pre-normalised input, to the residual stream.
func ParallelResidual(residual, attention, mlp []float32) core.Result {
	if len(residual) != len(attention) || len(residual) != len(mlp) {
		return core.Fail(core.NewError("model.ParallelResidual: vector lengths differ"))
	}
	out := make([]float32, len(residual))
	for i := range out {
		out[i] = residual[i] + attention[i] + mlp[i]
	}
	return core.Ok(out)
}
