// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// AddLearnedPositions adds rows from a learned absolute-position table to token embeddings.
func AddLearnedPositions(hidden, table []float32, tokens, width, position int) error {
	if tokens < 0 || width <= 0 || position < 0 || len(hidden) != tokens*width || len(table) < (position+tokens)*width {
		return core.NewError("model.AddLearnedPositions: invalid shape or position")
	}
	for tok := range tokens {
		for col := range width {
			hidden[tok*width+col] += table[(position+tok)*width+col]
		}
	}
	return nil
}

// ExpandMultiQueryKV repeats one key/value head for each query head.
func ExpandMultiQueryKV(kv []float32, tokens, heads, headDim int) ([]float32, error) {
	if tokens < 0 || heads <= 0 || headDim <= 0 || len(kv) != tokens*headDim {
		return nil, core.NewError("model.ExpandMultiQueryKV: invalid shape")
	}
	out := make([]float32, tokens*heads*headDim)
	for tok := range tokens {
		for head := range heads {
			copy(out[(tok*heads+head)*headDim:], kv[tok*headDim:(tok+1)*headDim])
		}
	}
	return out, nil
}
