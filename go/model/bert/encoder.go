// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"math"

	core "dappco.re/go"
)

// Device-hook design note (item C of #50 — measured baseline, no kernel built
// this slice).
//
// Baseline (Apple M3 Ultra, this package's BenchmarkModel_Embed_TokensPerSecond,
// -benchmem, bge-small-en-v1.5, 8 realistic sentences / 114 tokens/batch):
//
//	2534785211 ns/op   44.97 tok/s   25795105 B/op   13949 allocs/op
//
// Today's forward (bindWeights, forward, encoderLayer below) is naive host
// float32/float64-accumulate, single-threaded, zero SIMD/BLAS, zero device
// hook — every number above is pure Go scalar loops. 45 tok/s means a
// realistic 100-doc reranking batch (~1.5K tokens) costs on the order of 30s;
// the composed AX-8 device-hook pattern (model/composed/composed.go's
// ProjMatMulInto/MLPDevice/ResidualNormMLPDevice — a nil package-level func
// var the lib calls conditionally, bound by engine/metal at runtime, never
// imported the other way per AX-8) is the house precedent for closing that
// gap without coupling this package to any engine.
//
// The blocking prerequisite is NOT the missing hook — it's that encoderLayer
// calls linear() ONE TOKEN AT A TIME (`for i := range seqLen { query[i] =
// linear(hidden[i], ...) }`, likewise key/value/attnDense/intermediate/
// output), i.e. seqLen separate M=1 GEMVs per projection per layer instead of
// ONE M=seqLen GEMM. BERT's encoder is not autoregressive — every token's
// projection at a given layer is independent of its neighbours — so batching
// the row loop into a single [seqLen,hidden]×[hidden,hidden] matmul is a pure
// host refactor (same maths, bit-identical accumulation order achievable),
// and it is what makes a device hook profitable at all: composed.go gates its
// hooks behind deviceMinWork = 1<<20 (M·K·N), and bge-small's shapes (hidden
// 384, intermediate 1536) never reach that floor at M=1 — Q/K/V/attnDense is
// 1×384×384 = 147,456 and the FFN intermediate is 1×384×1536 = 589,824, both
// sub-floor — but at M=seqLen (14 tokens, this benchmark's average) they
// become 2,064,384 and 8,257,536, both comfortably above it. So the shape is:
//
//  1. Batch encoderLayer's per-token loops into per-layer [seqLen,·] matmuls
//     (matNTInto in composed.go is the exact reusable primitive — it already
//     shards output columns across cores for large shapes and is the natural
//     home for a shared host GEMM if one doesn't already fit here).
//  2. THEN add the hook seam: a package-level `var LinearDevice func(x, w,
//     bias []float32, M, K, N int) ([]float32, error)` that linear()-family
//     call sites try first when bound and M·K·N is above the same floor,
//     falling back to the host path on a nil hook or a device error — byte-
//     for-byte the composed.go pattern, so engine/metal binds it exactly as
//     it binds ProjMatMulInto today.
//
// The allocation side of the baseline (13,949 allocs for one 8-sentence
// batch) is the same root cause: every linear()/layerNorm() call allocates a
// fresh output slice per token per layer. Batching to per-layer buffers cuts
// both the compute shape AND the allocation count in the same change — no
// device dependency required to bank that part of the win.

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
	poolerW        []float32
	poolerB        []float32
	classifierW    []float32
	classifierB    []float32
}

// forward runs the encoder over a single unpadded token sequence and returns the
// last hidden state — one hidden-size vector per input token. attention_mask is
// implicitly all-ones (no padding on the single-sequence path) and
// token_type_ids are all zero, matching a plain sentence-embedding forward.
//
// The maths is naive host float32 with float64 accumulation in the hot dot
// products, which keeps it within cosine 0.999 of PyTorch's float32 result.
func (w *Weights) forward(cfg Config, ids []int32) ([][]float32, error) {
	return w.forwardSegments(cfg, ids, nil)
}

func (w *Weights) forwardSegments(cfg Config, ids, tokenTypes []int32) ([][]float32, error) {
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
		typeID := int32(0)
		if tokenTypes != nil {
			if len(tokenTypes) != seqLen {
				return nil, core.E("bert.forward", "token_type_ids length does not match input_ids", nil)
			}
			typeID = tokenTypes[pos]
		}
		if typeID < 0 || int(typeID) >= cfg.TypeVocabSize {
			return nil, core.E("bert.forward", core.Sprintf("token type id %d is outside type vocab size %d", typeID, cfg.TypeVocabSize), nil)
		}
		typeRow := w.typeEmbeddings[int(typeID)*hiddenSize : (int(typeID)+1)*hiddenSize]
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
