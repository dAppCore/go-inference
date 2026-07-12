// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"math"

	core "dappco.re/go"
)

// layerWeights holds one transformer encoder block's parameters. Linear weights
// keep the PyTorch [out, in] row-major layout straight from the safetensors, so
// linear() reads each output row as a contiguous span.
type layerWeights struct {
	queryW, queryB []float32
	keyW, keyB     []float32
	valueW, valueB []float32
	attnDenseW     []float32
	attnDenseB     []float32
	attnLNW        []float32
	attnLNB        []float32
	interW, interB []float32
	outW, outB     []float32
	outLNW, outLNB []float32
}

// Weights is the full set of BertModel encoder parameters (the pooler head is
// intentionally omitted — CLS pooling reads the raw last hidden state, and mean
// pooling never touches it). Populated by bindWeights from a safetensors map.
type Weights struct {
	wordEmbeddings []float32
	posEmbeddings  []float32
	typeEmbeddings []float32
	embLNW         []float32
	embLNB         []float32
	layers         []layerWeights
}

// forward runs the encoder over a single unpadded token sequence and returns the
// last hidden state — one hidden-size vector per input token. attention_mask is
// implicitly all-ones (no padding on the single-sequence path) and
// token_type_ids are all zero, matching a plain sentence-embedding forward.
//
// The maths is naive host float32 with float64 accumulation in the hot dot
// products, which keeps it within cosine 0.999 of PyTorch's float32 result.
func (w *Weights) forward(cfg Config, ids []int32) ([][]float32, error) {
	hiddenSize := cfg.HiddenSize
	seqLen := len(ids)
	if seqLen == 0 {
		return nil, core.E("bert.forward", "token sequence is empty", nil)
	}
	if seqLen > cfg.MaxPositionEmbeddings {
		return nil, core.E("bert.forward", core.Sprintf("sequence length %d exceeds max_position_embeddings %d", seqLen, cfg.MaxPositionEmbeddings), nil)
	}

	hidden := make([][]float32, seqLen)
	for pos, id := range ids {
		if int(id) < 0 || int(id) >= cfg.VocabSize {
			return nil, core.E("bert.forward", core.Sprintf("token id %d is outside vocab size %d", id, cfg.VocabSize), nil)
		}
		wordRow := w.wordEmbeddings[int(id)*hiddenSize : (int(id)+1)*hiddenSize]
		posRow := w.posEmbeddings[pos*hiddenSize : (pos+1)*hiddenSize]
		typeRow := w.typeEmbeddings[0:hiddenSize]
		summed := make([]float32, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			summed[j] = wordRow[j] + posRow[j] + typeRow[j]
		}
		hidden[pos] = layerNorm(summed, w.embLNW, w.embLNB, cfg.LayerNormEps)
	}

	for i := range w.layers {
		hidden = encoderLayer(cfg, &w.layers[i], hidden)
	}
	return hidden, nil
}

// encoderLayer applies one block: multi-head self-attention with a residual +
// LayerNorm, then a GELU feed-forward with its own residual + LayerNorm.
func encoderLayer(cfg Config, layer *layerWeights, hidden [][]float32) [][]float32 {
	hiddenSize := cfg.HiddenSize
	heads := cfg.NumAttentionHeads
	headDim := cfg.HeadDim()
	seqLen := len(hidden)
	scale := 1.0 / math.Sqrt(float64(headDim))

	query := make([][]float32, seqLen)
	key := make([][]float32, seqLen)
	value := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		query[i] = linear(hidden[i], layer.queryW, layer.queryB, hiddenSize, hiddenSize)
		key[i] = linear(hidden[i], layer.keyW, layer.keyB, hiddenSize, hiddenSize)
		value[i] = linear(hidden[i], layer.valueW, layer.valueB, hiddenSize, hiddenSize)
	}

	context := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		context[i] = make([]float32, hiddenSize)
	}
	scores := make([]float64, seqLen)
	for h := 0; h < heads; h++ {
		off := h * headDim
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(query[i][off+d]) * float64(key[j][off+d])
				}
				scores[j] = dot * scale
			}
			softmax(scores)
			for d := 0; d < headDim; d++ {
				var acc float64
				for j := 0; j < seqLen; j++ {
					acc += scores[j] * float64(value[j][off+d])
				}
				context[i][off+d] = float32(acc)
			}
		}
	}

	out := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		attnDense := linear(context[i], layer.attnDenseW, layer.attnDenseB, hiddenSize, hiddenSize)
		addInPlace(attnDense, hidden[i])
		attnNorm := layerNorm(attnDense, layer.attnLNW, layer.attnLNB, cfg.LayerNormEps)

		intermediate := linear(attnNorm, layer.interW, layer.interB, hiddenSize, cfg.IntermediateSize)
		for k := range intermediate {
			intermediate[k] = gelu(intermediate[k])
		}
		ffn := linear(intermediate, layer.outW, layer.outB, cfg.IntermediateSize, hiddenSize)
		addInPlace(ffn, attnNorm)
		out[i] = layerNorm(ffn, layer.outLNW, layer.outLNB, cfg.LayerNormEps)
	}
	return out
}

// linear computes y = x·Wᵀ + b for a single vector. weight is the PyTorch
// [outDim, inDim] row-major layout, so output row o is weight[o*inDim:(o+1)*inDim].
// The dot product accumulates in float64 to stay close to the reference.
func linear(x, weight, bias []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		row := weight[o*inDim : (o+1)*inDim]
		var acc float64
		for i := 0; i < inDim; i++ {
			acc += float64(x[i]) * float64(row[i])
		}
		if bias != nil {
			acc += float64(bias[o])
		}
		out[o] = float32(acc)
	}
	return out
}

// layerNorm normalises x over its full length with biased variance and applies
// the affine weight/bias — torch.nn.LayerNorm with eps inside the sqrt.
func layerNorm(x, weight, bias []float32, eps float64) []float32 {
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(n)
	var variance float64
	for _, v := range x {
		diff := float64(v) - mean
		variance += diff * diff
	}
	variance /= float64(n)
	denom := math.Sqrt(variance + eps)
	out := make([]float32, n)
	for i, v := range x {
		normalised := (float64(v) - mean) / denom
		out[i] = float32(normalised*float64(weight[i]) + float64(bias[i]))
	}
	return out
}

// gelu is the exact error-function GELU (config hidden_act "gelu"): the
// tanh approximation is deliberately not used, so the activation matches
// transformers' default BertIntermediate.
func gelu(x float32) float32 {
	v := float64(x)
	return float32(0.5 * v * (1.0 + math.Erf(v/math.Sqrt2)))
}

// softmax normalises scores in place with the standard max-shift for numerical
// stability, matching the attention probability computation.
func softmax(scores []float64) {
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	var sum float64
	for i, s := range scores {
		e := math.Exp(s - maxScore)
		scores[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	for i := range scores {
		scores[i] /= sum
	}
}

// addInPlace adds the residual into dst element-wise (dst += residual).
func addInPlace(dst, residual []float32) {
	for i := range dst {
		dst[i] += residual[i]
	}
}
