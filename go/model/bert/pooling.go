// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"math"

	core "dappco.re/go"
)

// Pooling selects how a sequence of token hidden states collapses to one
// sentence vector. It mirrors sentence-transformers' Pooling module: bge-class
// checkpoints use CLS (the [CLS] token's hidden state), all-MiniLM-class use
// mean over the tokens.
type Pooling string

const (
	// PoolingCLS takes the first token ([CLS]) hidden state — bge-small's mode.
	PoolingCLS Pooling = "cls"
	// PoolingMean averages every token's hidden state — the MiniLM/E5 mode.
	PoolingMean Pooling = "mean"
)

// pool reduces a last-hidden-state (seqLen × hidden) to one vector under the
// selected mode. An unknown mode is an error rather than a silent default, so a
// mis-parsed pooling config surfaces loudly.
func pool(mode Pooling, hidden [][]float32) ([]float32, error) {
	if len(hidden) == 0 {
		return nil, core.E("bert.pool", "no hidden states to pool", nil)
	}
	switch mode {
	case PoolingCLS:
		out := make([]float32, len(hidden[0]))
		copy(out, hidden[0])
		return out, nil
	case PoolingMean:
		dim := len(hidden[0])
		out := make([]float32, dim)
		for _, row := range hidden {
			for j := 0; j < dim; j++ {
				out[j] += row[j]
			}
		}
		inv := float32(1.0 / float64(len(hidden)))
		for j := range out {
			out[j] *= inv
		}
		return out, nil
	default:
		return nil, core.E("bert.pool", core.Sprintf("unknown pooling mode %q", mode), nil)
	}
}

// l2Normalise scales vec to unit length in place. A zero-magnitude vector is
// left untouched rather than producing NaNs — the same guard cosine callers
// rely on downstream.
func l2Normalise(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range vec {
		vec[i] *= inv
	}
}
