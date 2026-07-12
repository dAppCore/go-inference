// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"

	core "dappco.re/go"
)

type hipReferenceTinyLMConfig struct {
	EmbeddingTable []float32
	OutputWeights  []float32
	VocabSize      int
	HiddenSize     int
}

type hipReferenceTinyLMState struct {
	Keys   [][]float32
	Values [][]float32
}

type hipReferenceTinyLMResult struct {
	Logits       []float32
	NextTokenID  int
	NextScore    float32
	Attention    []float32
	PrefillHeads [][]float32
	State        hipReferenceTinyLMState
}

type hipReferenceCandidate struct {
	index int
	value float32
}

func hipReferenceEmbeddingLookup(table []float32, vocabSize, hiddenSize int, tokenIDs []int32) ([]float32, error) {
	if vocabSize <= 0 || hiddenSize <= 0 {
		return nil, core.E("rocm.hip.ReferenceEmbeddingLookup", "vocab and hidden size must be positive", nil)
	}
	if len(table) != vocabSize*hiddenSize {
		return nil, core.E("rocm.hip.ReferenceEmbeddingLookup", core.Sprintf("embedding table length %d does not match vocab*hidden %d", len(table), vocabSize*hiddenSize), nil)
	}
	if len(tokenIDs) == 0 {
		return nil, core.E("rocm.hip.ReferenceEmbeddingLookup", "token ids are required", nil)
	}
	out := make([]float32, 0, len(tokenIDs)*hiddenSize)
	for _, id := range tokenIDs {
		if id < 0 || int(id) >= vocabSize {
			return nil, core.E("rocm.hip.ReferenceEmbeddingLookup", core.Sprintf("token id %d outside vocab size %d", id, vocabSize), nil)
		}
		start := int(id) * hiddenSize
		out = append(out, table[start:start+hiddenSize]...)
	}
	return out, nil
}

func hipReferenceMLXQ4EmbeddingLookup(weights []uint32, scales []uint16, biases []uint16, vocabSize, hiddenSize, groupSize int, tokenIDs []int32) ([]float32, error) {
	return hipReferenceMLXAffineEmbeddingLookup(weights, scales, biases, vocabSize, hiddenSize, groupSize, tokenIDs, hipMLXQ4ProjectionBits)
}

func hipReferenceMLXAffineEmbeddingLookup(weights []uint32, scales []uint16, biases []uint16, vocabSize, hiddenSize, groupSize int, tokenIDs []int32, bits int) ([]float32, error) {
	if err := validateHIPMLXAffineProjectionShape(hiddenSize, len(weights), len(scales), len(biases), vocabSize, hiddenSize, groupSize, bits); err != nil {
		return nil, err
	}
	if len(tokenIDs) == 0 {
		return nil, core.E("rocm.hip.ReferenceMLXQ4EmbeddingLookup", "token ids are required", nil)
	}
	packedPerRow, err := hipMLXAffinePackedCols(hiddenSize, bits)
	if err != nil {
		return nil, err
	}
	groupsPerRow := hiddenSize / groupSize
	out := make([]float32, 0, len(tokenIDs)*hiddenSize)
	for _, id := range tokenIDs {
		if id < 0 || int(id) >= vocabSize {
			return nil, core.E("rocm.hip.ReferenceMLXQ4EmbeddingLookup", core.Sprintf("token id %d outside vocab size %d", id, vocabSize), nil)
		}
		row := int(id)
		for dim := 0; dim < hiddenSize; dim++ {
			quantized, err := hipMLXAffineUnpackValue(weights[row*packedPerRow:], dim, bits)
			if err != nil {
				return nil, err
			}
			group := row*groupsPerRow + dim/groupSize
			out = append(out, float32(quantized)*hipBFloat16ToFloat32(scales[group])+hipBFloat16ToFloat32(biases[group]))
		}
	}
	return out, nil
}

func hipReferenceTinyPrefill(cfg hipReferenceTinyLMConfig, tokenIDs []int32) (hipReferenceTinyLMResult, error) {
	if err := validateHIPReferenceTinyLMConfig(cfg); err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	flat, err := hipReferenceEmbeddingLookup(cfg.EmbeddingTable, cfg.VocabSize, cfg.HiddenSize, tokenIDs)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	embeddings, err := splitHIPReferenceVectors(flat, cfg.HiddenSize)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	outputs, weights, err := hipReferenceCausalPrefillAttention(embeddings, embeddings, embeddings)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	last := outputs[len(outputs)-1]
	logits, err := hipReferenceFP32Projection(last, cfg.OutputWeights, cfg.VocabSize, cfg.HiddenSize, nil)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	next, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	return hipReferenceTinyLMResult{
		Logits:       logits,
		NextTokenID:  next,
		NextScore:    score,
		Attention:    weights[len(weights)-1],
		PrefillHeads: outputs,
		State:        hipReferenceTinyLMState{Keys: copyFloat32Matrix(embeddings), Values: copyFloat32Matrix(embeddings)},
	}, nil
}

func hipReferenceTinyDecode(cfg hipReferenceTinyLMConfig, state hipReferenceTinyLMState, tokenID int32) (hipReferenceTinyLMResult, error) {
	if err := validateHIPReferenceTinyLMConfig(cfg); err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	flat, err := hipReferenceEmbeddingLookup(cfg.EmbeddingTable, cfg.VocabSize, cfg.HiddenSize, []int32{tokenID})
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	embedding := append([]float32(nil), flat...)
	output, attention, keys, values, err := hipReferenceDecodeWithKV(embedding, embedding, embedding, state.Keys, state.Values)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	logits, err := hipReferenceFP32Projection(output, cfg.OutputWeights, cfg.VocabSize, cfg.HiddenSize, nil)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	next, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return hipReferenceTinyLMResult{}, err
	}
	return hipReferenceTinyLMResult{
		Logits:      logits,
		NextTokenID: next,
		NextScore:   score,
		Attention:   attention,
		State:       hipReferenceTinyLMState{Keys: keys, Values: values},
	}, nil
}

func flattenHIPReferenceMatrix(values [][]float32) []float32 {
	total := 0
	for _, row := range values {
		total += len(row)
	}
	out := make([]float32, 0, total)
	for _, row := range values {
		out = append(out, row...)
	}
	return out
}

func hipReferenceRMSNorm(input, weight []float32, epsilon float32) ([]float32, error) {
	if len(input) == 0 {
		return nil, core.E("rocm.hip.ReferenceRMSNorm", "input is required", nil)
	}
	if len(weight) != len(input) {
		return nil, core.E("rocm.hip.ReferenceRMSNorm", "weight length must match input length", nil)
	}
	if epsilon < 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return nil, core.E("rocm.hip.ReferenceRMSNorm", "epsilon must be non-negative and finite", nil)
	}
	sumSquares := float64(0)
	for _, value := range input {
		sumSquares += float64(value * value)
	}
	rms := float32(math.Sqrt(sumSquares/float64(len(input)) + float64(epsilon)))
	if rms == 0 {
		return nil, core.E("rocm.hip.ReferenceRMSNorm", "rms is zero", nil)
	}
	out := make([]float32, len(input))
	for i, value := range input {
		out[i] = value / rms * weight[i]
	}
	return out, nil
}

func hipReferenceRoPE(input []float32, position int, base float64) ([]float32, error) {
	return hipReferenceRoPEWithFrequencyDim(input, position, base, 0)
}

func hipReferenceRoPEWithFrequencyDim(input []float32, position int, base float64, frequencyDim int) ([]float32, error) {
	return hipReferenceRoPEWithFrequencyDimScale(input, position, base, frequencyDim, 1)
}

func hipReferenceRoPEWithFrequencyDimScale(input []float32, position int, base float64, frequencyDim int, frequencyScale float64) ([]float32, error) {
	if len(input) == 0 || len(input)%2 != 0 {
		return nil, core.E("rocm.hip.ReferenceRoPE", "input length must be positive and even", nil)
	}
	if position < 0 {
		return nil, core.E("rocm.hip.ReferenceRoPE", "position must be non-negative", nil)
	}
	if base <= 0 || math.IsNaN(base) || math.IsInf(base, 0) {
		return nil, core.E("rocm.hip.ReferenceRoPE", "base must be positive and finite", nil)
	}
	if frequencyScale <= 0 || math.IsNaN(frequencyScale) || math.IsInf(frequencyScale, 0) {
		return nil, core.E("rocm.hip.ReferenceRoPE", "frequency scale must be positive and finite", nil)
	}
	if frequencyDim < 0 || (frequencyDim > 0 && frequencyDim < len(input)) {
		return nil, core.E("rocm.hip.ReferenceRoPE", "frequency dimension must be zero or at least input length", nil)
	}
	if frequencyDim == 0 {
		frequencyDim = len(input)
	}
	out := append([]float32(nil), input...)
	dim := float64(frequencyDim)
	for i := 0; i < len(input); i += 2 {
		frequency := 1 / math.Pow(base, float64(i)/dim)
		angle := float64(position) * frequency * frequencyScale
		cosine := float32(math.Cos(angle))
		sine := float32(math.Sin(angle))
		x := input[i]
		y := input[i+1]
		out[i] = x*cosine - y*sine
		out[i+1] = x*sine + y*cosine
	}
	return out, nil
}

func hipReferenceRoPENeoXWithFrequencyDim(input []float32, position int, base float64, frequencyDim, rotaryCount int) ([]float32, error) {
	return hipReferenceRoPENeoXWithFrequencyDimScale(input, position, base, frequencyDim, rotaryCount, 1)
}

func hipReferenceRoPENeoXWithFrequencyDimScale(input []float32, position int, base float64, frequencyDim, rotaryCount int, frequencyScale float64) ([]float32, error) {
	if len(input) == 0 || len(input)%2 != 0 {
		return nil, core.E("rocm.hip.ReferenceRoPENeoX", "input length must be positive and even", nil)
	}
	if position < 0 {
		return nil, core.E("rocm.hip.ReferenceRoPENeoX", "position must be non-negative", nil)
	}
	if base <= 0 || math.IsNaN(base) || math.IsInf(base, 0) {
		return nil, core.E("rocm.hip.ReferenceRoPENeoX", "base must be positive and finite", nil)
	}
	if frequencyScale <= 0 || math.IsNaN(frequencyScale) || math.IsInf(frequencyScale, 0) {
		return nil, core.E("rocm.hip.ReferenceRoPENeoX", "frequency scale must be positive and finite", nil)
	}
	if frequencyDim < 0 || (frequencyDim > 0 && frequencyDim < len(input)) {
		return nil, core.E("rocm.hip.ReferenceRoPENeoX", "frequency dimension must be zero or at least input length", nil)
	}
	if rotaryCount < 0 || rotaryCount > len(input) || rotaryCount%2 != 0 {
		return nil, core.E("rocm.hip.ReferenceRoPENeoX", "rotary count must be zero or an even count no larger than input length", nil)
	}
	if frequencyDim == 0 {
		frequencyDim = len(input)
	}
	if rotaryCount == 0 {
		rotaryCount = len(input)
	}
	out := append([]float32(nil), input...)
	half := len(input) / 2
	activePairs := rotaryCount / 2
	dim := float64(frequencyDim)
	for pair := 0; pair < half; pair++ {
		first := pair
		second := pair + half
		if pair >= activePairs {
			out[first] = input[first]
			out[second] = input[second]
			continue
		}
		frequency := 1 / math.Pow(base, float64(pair*2)/dim)
		angle := float64(position) * frequency * frequencyScale
		cosine := float32(math.Cos(angle))
		sine := float32(math.Sin(angle))
		x := input[first]
		y := input[second]
		out[first] = x*cosine - y*sine
		out[second] = x*sine + y*cosine
	}
	return out, nil
}

func hipReferenceSingleHeadAttention(query []float32, keys, values [][]float32) ([]float32, []float32, error) {
	return hipReferenceSingleHeadAttentionWithScale(query, keys, values, 0)
}

func hipReferenceSingleHeadAttentionWithScale(query []float32, keys, values [][]float32, scale float32) ([]float32, []float32, error) {
	if len(query) == 0 {
		return nil, nil, core.E("rocm.hip.ReferenceAttention", "query is required", nil)
	}
	if len(keys) == 0 || len(keys) != len(values) {
		return nil, nil, core.E("rocm.hip.ReferenceAttention", "keys and values must be non-empty and equal length", nil)
	}
	dim := len(query)
	for i := range keys {
		if len(keys[i]) != dim || len(values[i]) != dim {
			return nil, nil, core.E("rocm.hip.ReferenceAttention", core.Sprintf("key/value %d dimension must match query dimension %d", i, dim), nil)
		}
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return nil, nil, core.E("rocm.hip.ReferenceAttention", "scale must be non-negative and finite", nil)
	}
	scores := make([]float32, len(keys))
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(dim)))
	}
	for i, key := range keys {
		score := float32(0)
		for j, value := range key {
			score += query[j] * value
		}
		scores[i] = score * scale
	}
	weights := softmaxFloat32(scores)
	out := make([]float32, dim)
	for i, value := range values {
		for j := range value {
			out[j] += weights[i] * value[j]
		}
	}
	return out, weights, nil
}

func hipReferenceMultiHeadAttention(query []float32, keys, values [][]float32, heads int) ([]float32, [][]float32, error) {
	if heads <= 0 {
		return nil, nil, core.E("rocm.hip.ReferenceMultiHeadAttention", "head count must be positive", nil)
	}
	if len(query) == 0 || len(query)%heads != 0 {
		return nil, nil, core.E("rocm.hip.ReferenceMultiHeadAttention", "query length must be a positive multiple of head count", nil)
	}
	if len(keys) == 0 || len(keys) != len(values) {
		return nil, nil, core.E("rocm.hip.ReferenceMultiHeadAttention", "keys and values must be non-empty and equal length", nil)
	}
	for i := range keys {
		if len(keys[i]) != len(query) || len(values[i]) != len(query) {
			return nil, nil, core.E("rocm.hip.ReferenceMultiHeadAttention", core.Sprintf("key/value %d dimension must match query dimension %d", i, len(query)), nil)
		}
	}
	headDim := len(query) / heads
	output := make([]float32, len(query))
	weights := make([][]float32, heads)
	for head := 0; head < heads; head++ {
		start := head * headDim
		end := start + headDim
		headKeys := make([][]float32, len(keys))
		headValues := make([][]float32, len(values))
		for i := range keys {
			headKeys[i] = keys[i][start:end]
			headValues[i] = values[i][start:end]
		}
		headOutput, headWeights, err := hipReferenceSingleHeadAttention(query[start:end], headKeys, headValues)
		if err != nil {
			return nil, nil, err
		}
		copy(output[start:end], headOutput)
		weights[head] = headWeights
	}
	return output, weights, nil
}

func hipReferenceCausalPrefillAttention(queries, keys, values [][]float32) ([][]float32, [][]float32, error) {
	if len(queries) == 0 || len(queries) != len(keys) || len(keys) != len(values) {
		return nil, nil, core.E("rocm.hip.ReferenceCausalPrefill", "queries, keys, and values must be non-empty and equal length", nil)
	}
	outputs := make([][]float32, len(queries))
	weights := make([][]float32, len(queries))
	for i := range queries {
		out, attention, err := hipReferenceSingleHeadAttention(queries[i], keys[:i+1], values[:i+1])
		if err != nil {
			return nil, nil, err
		}
		outputs[i] = out
		weights[i] = attention
	}
	return outputs, weights, nil
}

func hipReferenceDecodeWithKV(query, newKey, newValue []float32, keys, values [][]float32) ([]float32, []float32, [][]float32, [][]float32, error) {
	if len(newKey) == 0 || len(newKey) != len(query) || len(newValue) != len(query) {
		return nil, nil, nil, nil, core.E("rocm.hip.ReferenceDecodeKV", "new key/value dimensions must match query", nil)
	}
	updatedKeys := copyFloat32Matrix(keys)
	updatedValues := copyFloat32Matrix(values)
	updatedKeys = append(updatedKeys, append([]float32(nil), newKey...))
	updatedValues = append(updatedValues, append([]float32(nil), newValue...))
	out, attention, err := hipReferenceSingleHeadAttention(query, updatedKeys, updatedValues)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return out, attention, updatedKeys, updatedValues, nil
}

func hipReferenceGreedySample(logits []float32) (int, float32, error) {
	if len(logits) == 0 {
		return 0, 0, core.E("rocm.hip.ReferenceGreedySample", "logits are required", nil)
	}
	index := 0
	value := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > value {
			index = i
			value = logits[i]
		}
	}
	return index, value, nil
}

func hipReferenceGreedySampleSuppress(logits []float32, suppressTokens []int32) (int, float32, error) {
	if len(suppressTokens) == 0 {
		return hipReferenceGreedySample(logits)
	}
	if len(logits) == 0 {
		return 0, 0, core.E("rocm.hip.ReferenceGreedySample", "logits are required", nil)
	}
	index := -1
	value := float32(0)
	for i, logit := range logits {
		if hipTokenIsSuppressed(int32(i), suppressTokens) {
			continue
		}
		if index < 0 || logit > value {
			index = i
			value = logit
		}
	}
	if index < 0 {
		return 0, 0, core.E("rocm.hip.ReferenceGreedySample", "all logits are suppressed", nil)
	}
	return index, value, nil
}

func hipReferenceTopKProbabilities(logits []float32, topK int, temperature float32) ([]float32, error) {
	if len(logits) == 0 {
		return nil, core.E("rocm.hip.ReferenceTopKSampler", "logits are required", nil)
	}
	if topK <= 0 || topK > len(logits) {
		return nil, core.E("rocm.hip.ReferenceTopKSampler", "top-k must be within vocabulary size", nil)
	}
	if temperature <= 0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) {
		return nil, core.E("rocm.hip.ReferenceTopKSampler", "temperature must be positive and finite", nil)
	}
	candidates := make([]hipReferenceCandidate, len(logits))
	for i, value := range logits {
		candidates[i] = hipReferenceCandidate{index: i, value: value}
	}
	sortHIPReferenceCandidates(candidates)
	filtered := make([]float32, len(logits))
	for i := range filtered {
		filtered[i] = float32(math.Inf(-1))
	}
	scaled := make([]float32, topK)
	for i := 0; i < topK; i++ {
		scaled[i] = candidates[i].value / temperature
	}
	probs := softmaxFloat32(scaled)
	for i := 0; i < topK; i++ {
		filtered[candidates[i].index] = probs[i]
	}
	return filtered, nil
}

func copyFloat32Matrix(values [][]float32) [][]float32 {
	out := make([][]float32, len(values))
	for i := range values {
		out[i] = append([]float32(nil), values[i]...)
	}
	return out
}

func validateHIPReferenceTinyLMConfig(cfg hipReferenceTinyLMConfig) error {
	if cfg.VocabSize <= 0 || cfg.HiddenSize <= 0 {
		return core.E("rocm.hip.ReferenceTinyLM", "vocab and hidden size must be positive", nil)
	}
	if len(cfg.EmbeddingTable) != cfg.VocabSize*cfg.HiddenSize {
		return core.E("rocm.hip.ReferenceTinyLM", core.Sprintf("embedding table length %d does not match vocab*hidden %d", len(cfg.EmbeddingTable), cfg.VocabSize*cfg.HiddenSize), nil)
	}
	if len(cfg.OutputWeights) != cfg.VocabSize*cfg.HiddenSize {
		return core.E("rocm.hip.ReferenceTinyLM", core.Sprintf("output weight length %d does not match vocab*hidden %d", len(cfg.OutputWeights), cfg.VocabSize*cfg.HiddenSize), nil)
	}
	return nil
}

func splitHIPReferenceVectors(flat []float32, width int) ([][]float32, error) {
	if width <= 0 || len(flat) == 0 || len(flat)%width != 0 {
		return nil, core.E("rocm.hip.ReferenceVectors", "flat tensor length must be a positive multiple of width", nil)
	}
	vectors := make([][]float32, 0, len(flat)/width)
	for offset := 0; offset < len(flat); offset += width {
		vectors = append(vectors, append([]float32(nil), flat[offset:offset+width]...))
	}
	return vectors, nil
}

func sortHIPReferenceCandidates(candidates []hipReferenceCandidate) {
	// Full-vocabulary host sampling depends on this exact total order. Keep the
	// softcapped score descending and token ID ascending. A four-pass radix sort
	// makes the 262144-candidate Gemma 4 TopK=0 path linear without changing its
	// draw mapping.
	if len(candidates) < 2 {
		return
	}
	scratch := make([]hipReferenceCandidate, len(candidates))
	counts := make([]int, 1<<16)
	input, output := candidates, scratch
	for shift := uint(0); shift < 64; shift += 16 {
		clear(counts)
		for _, candidate := range input {
			counts[uint16(hipReferenceCandidateSortKey(candidate)>>shift)]++
		}
		offset := 0
		for bucket, count := range counts {
			counts[bucket] = offset
			offset += count
		}
		for _, candidate := range input {
			bucket := uint16(hipReferenceCandidateSortKey(candidate) >> shift)
			output[counts[bucket]] = candidate
			counts[bucket]++
		}
		input, output = output, input
	}
}

func hipReferenceCandidateSortKey(candidate hipReferenceCandidate) uint64 {
	bits := math.Float32bits(candidate.value)
	if bits&(1<<31) != 0 {
		bits = ^bits
	} else {
		bits ^= 1 << 31
	}
	return uint64(^bits)<<32 | uint64(uint32(candidate.index))
}
