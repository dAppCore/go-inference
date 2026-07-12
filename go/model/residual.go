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

// ApplyResidualOrder executes one attention-plus-feed-forward block using the
// architecture's declared norm placement. Pre-norm families normalise each
// residual stream before its sublayer; post-norm families normalise each
// sublayer output before adding it back.
func ApplyResidualOrder(placement NormPlacement, hidden []float32,
	attentionNorm, feedForwardNorm func([]float32) []float32,
	attention, feedForward func([]float32) []float32,
) core.Result {
	if attentionNorm == nil || feedForwardNorm == nil || attention == nil || feedForward == nil {
		return core.Fail(core.NewError("model.ApplyResidualOrder: nil operation"))
	}
	if placement != NormPlacementPre && placement != NormPlacementPost {
		return core.Fail(core.NewError("model.ApplyResidualOrder: unsupported norm placement"))
	}
	attnInput := hidden
	if placement == NormPlacementPre {
		attnInput = attentionNorm(hidden)
	}
	attnOutput := attention(attnInput)
	if placement == NormPlacementPost {
		attnOutput = attentionNorm(attnOutput)
	}
	afterAttention, ok := addResidual(hidden, attnOutput)
	if !ok {
		return core.Fail(core.NewError("model.ApplyResidualOrder: attention output length differs"))
	}
	mlpInput := afterAttention
	if placement == NormPlacementPre {
		mlpInput = feedForwardNorm(afterAttention)
	}
	mlpOutput := feedForward(mlpInput)
	if placement == NormPlacementPost {
		mlpOutput = feedForwardNorm(mlpOutput)
	}
	out, ok := addResidual(afterAttention, mlpOutput)
	if !ok {
		return core.Fail(core.NewError("model.ApplyResidualOrder: feed-forward output length differs"))
	}
	return core.Ok(out)
}

func addResidual(residual, update []float32) ([]float32, bool) {
	if len(residual) != len(update) {
		return nil, false
	}
	out := make([]float32, len(residual))
	for i := range out {
		out[i] = residual[i] + update[i]
	}
	return out, true
}
