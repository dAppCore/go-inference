// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"

	core "dappco.re/go"
)

// RopeParams declares the backend-neutral geometry of a rotary-position op.
// PartialRotaryFactor is a checkpoint parameter, not family-side execution
// logic: zero and one both mean full-head rotary for backwards compatibility.
type RopeParams struct {
	HeadDim             int
	PartialRotaryFactor float32
}

// RotaryDim resolves the even prefix width rotated by the RoPE op.
func (p RopeParams) RotaryDim() (int, error) {
	if p.HeadDim <= 0 {
		return 0, core.NewError("model.RopeParams.RotaryDim: head dimension must be > 0")
	}
	factor := p.PartialRotaryFactor
	if factor == 0 {
		factor = 1
	}
	if factor <= 0 || factor > 1 {
		return 0, core.NewError("model.RopeParams.RotaryDim: partial rotary factor must be in (0,1]")
	}
	resolved := float64(p.HeadDim) * float64(factor)
	dim := int(math.Round(resolved))
	if math.Abs(resolved-float64(dim)) > 1e-5 {
		return 0, core.NewError("model.RopeParams.RotaryDim: factor must resolve to a whole dimension")
	}
	if dim <= 0 || dim > p.HeadDim || dim%2 != 0 {
		return 0, core.NewError("model.RopeParams.RotaryDim: resolved rotary dimension must be positive and even")
	}
	return dim, nil
}
