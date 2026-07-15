// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

var (
	benchmarkHIPTopPackedScoresSink       []uint64
	benchmarkHIPTopPackedScoreSink        uint64
	benchmarkHIPCandidateSampleResultSink hipGreedySampleResult
)

func TestHIPTransformerReferenceEmbeddingLookup_Good(t *testing.T) {
	output, err := hipReferenceEmbeddingLookup(
		[]float32{1, 2, 3, 4, 5, 6},
		3,
		2,
		[]int32{2, 0},
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5, 6, 1, 2}, output, 0)

	q4Output, err := hipReferenceMLXQ4EmbeddingLookup(
		[]uint32{0x76543210, 0x11111111, 0xfedcba98},
		[]uint16{0x3f80, 0x3f80, 0x3f00},
		[]uint16{0x0000, 0x0000, 0xbf80},
		3,
		8,
		8,
		[]int32{2, 0},
	)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 0, 1, 2, 3, 4, 5, 6, 7}, q4Output, 0)

	q6Output, err := hipReferenceMLXAffineEmbeddingLookup(
		hipPackMLXAffineValuesForTest([]uint32{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
		}, 16, 6),
		[]uint16{0x3f80, 0x3f80},
		[]uint16{0x0000, 0x0000},
		2,
		16,
		16,
		[]int32{1},
		6,
	)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, q6Output, 0)
}

func TestHIPTransformerReferenceTinyPrefill_Good(t *testing.T) {
	result, err := hipReferenceTinyPrefill(hipReferenceTinyLMFixture(), []int32{0, 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, result.NextTokenID)
	assertFloat32Near(t, 1, result.NextScore)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, result.Logits, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, result.Attention, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0}, result.PrefillHeads[0], 0.0001)
	core.AssertEqual(t, 2, len(result.State.Keys))
}

func TestHIPTransformerReferenceTinyDecode_Good(t *testing.T) {
	prefill, err := hipReferenceTinyPrefill(hipReferenceTinyLMFixture(), []int32{0, 1})
	core.RequireNoError(t, err)

	result, err := hipReferenceTinyDecode(hipReferenceTinyLMFixture(), prefill.State, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, result.NextTokenID)
	assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, result.Logits, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.2483, 0.2483, 0.5035}, result.Attention, 0.0001)
	core.AssertEqual(t, 3, len(result.State.Keys))
	core.AssertEqual(t, 3, len(result.State.Values))
}

func TestHIPTransformerReferenceRMSNorm_Good(t *testing.T) {
	output, err := hipReferenceRMSNorm([]float32{3, 4}, []float32{1, 0.5}, 0)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, output, 0.0001)
}

func TestHIPTransformerReferenceRoPE_Good(t *testing.T) {
	output, err := hipReferenceRoPE([]float32{1, 0}, 1, 1)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(1)), float32(math.Sin(1))}, output, 0.0001)

	output, err = hipReferenceRoPEWithFrequencyDim([]float32{1, 0, 1, 0}, 1, 10000, 8)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{
		float32(math.Cos(1)),
		float32(math.Sin(1)),
		float32(math.Cos(0.1)),
		float32(math.Sin(0.1)),
	}, output, 0.0001)

	output, err = hipReferenceRoPEWithFrequencyDimScale([]float32{1, 0}, 1, 1, 2, 0.5)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(0.5)), float32(math.Sin(0.5))}, output, 0.0001)

	output, err = hipReferenceRoPENeoXWithFrequencyDim([]float32{1, 2, 3, 4}, 1, 1, 4, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{
		1*float32(math.Cos(1)) - 3*float32(math.Sin(1)),
		2,
		1*float32(math.Sin(1)) + 3*float32(math.Cos(1)),
		4,
	}, output, 0.0001)
}

func TestHIPTransformerReferenceSingleHeadAttention_Good(t *testing.T) {
	output, weights, err := hipReferenceSingleHeadAttention(
		[]float32{1, 0},
		[][]float32{{1, 0}, {0, 1}},
		[][]float32{{2, 0}, {0, 4}},
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210}, output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, weights, 0.0001)
}

func TestHIPTransformerReferenceMultiHeadAttention_Good(t *testing.T) {
	output, weights, err := hipReferenceMultiHeadAttention(
		[]float32{1, 0, 0, 1},
		[][]float32{{1, 0, 1, 0}, {0, 1, 0, 1}},
		[][]float32{{2, 0, 10, 0}, {0, 4, 0, 20}},
		2,
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210, 3.3024, 13.3952}, output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, weights[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, weights[1], 0.0001)
}

func TestHIPTransformerReferenceCausalPrefillAttention_Good(t *testing.T) {
	outputs, weights, err := hipReferenceCausalPrefillAttention(
		[][]float32{{1, 0}, {0, 1}},
		[][]float32{{1, 0}, {0, 1}},
		[][]float32{{2, 0}, {0, 4}},
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{2, 0}, outputs[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6605, 2.6790}, outputs[1], 0.0001)
	assertFloat32SlicesNear(t, []float32{1}, weights[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, weights[1], 0.0001)
}

func TestHIPTransformerReferenceDecodeWithKV_Good(t *testing.T) {
	output, weights, updatedKeys, updatedValues, err := hipReferenceDecodeWithKV(
		[]float32{0, 1},
		[]float32{0, 1},
		[]float32{0, 4},
		[][]float32{{1, 0}},
		[][]float32{{2, 0}},
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.6605, 2.6790}, output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, weights, 0.0001)
	core.AssertEqual(t, 2, len(updatedKeys))
	core.AssertEqual(t, 2, len(updatedValues))
	assertFloat32SlicesNear(t, []float32{0, 1}, updatedKeys[1], 0)
	assertFloat32SlicesNear(t, []float32{0, 4}, updatedValues[1], 0)
}

func TestHIPTransformerReferenceGreedySample_Good(t *testing.T) {
	index, value, err := hipReferenceGreedySample([]float32{-1, 0.25, 0.2})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, index)
	assertFloat32Near(t, 0.25, value)
}

func TestHIPTransformerReferenceTopKProbabilities_Good(t *testing.T) {
	probs, err := hipReferenceTopKProbabilities([]float32{1, 3, 2}, 2, 1)

	core.RequireNoError(t, err)
	if !math.IsInf(float64(probs[0]), -1) {
		t.Fatalf("probs = %+v, want filtered token probability to be -Inf", probs)
	}
	assertFloat32Near(t, 0.7311, probs[1])
	assertFloat32Near(t, 0.2689, probs[2])
}

func TestHIPTransformerReferenceEmbeddingLookupBadInputs_Bad(t *testing.T) {
	_, err := hipReferenceEmbeddingLookup([]float32{1}, 0, 1, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vocab and hidden")

	_, err = hipReferenceEmbeddingLookup([]float32{1}, 1, 0, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vocab and hidden")

	_, err = hipReferenceEmbeddingLookup([]float32{1}, 1, 2, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding table length")

	_, err = hipReferenceEmbeddingLookup([]float32{1}, 1, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ids")

	_, err = hipReferenceMLXQ4EmbeddingLookup([]uint32{0}, []uint16{0}, []uint16{0}, 1, 8, 8, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ids")

	_, err = hipReferenceMLXQ4EmbeddingLookup([]uint32{0}, []uint16{0}, []uint16{0}, 1, 8, 8, []int32{2})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocab")
}

func TestHIPTransformerReferenceTinyLMBadInputs_Bad(t *testing.T) {
	_, err := hipReferenceTinyPrefill(hipReferenceTinyLMConfig{VocabSize: 0, HiddenSize: 1}, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vocab and hidden")

	cfg := hipReferenceTinyLMFixture()
	cfg.OutputWeights = cfg.OutputWeights[:len(cfg.OutputWeights)-1]
	_, err = hipReferenceTinyPrefill(cfg, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output weight length")

	_, err = hipReferenceTinyDecode(hipReferenceTinyLMFixture(), hipReferenceTinyLMState{
		Keys:   [][]float32{{1, 2, 3}},
		Values: [][]float32{{1, 2}},
	}, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")
}

func TestHIPTransformerReferenceRMSNormBadInputs_Bad(t *testing.T) {
	_, err := hipReferenceRMSNorm(nil, nil, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input is required")

	_, err = hipReferenceRMSNorm([]float32{1}, []float32{1}, -1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "epsilon must be non-negative")

	_, err = hipReferenceRMSNorm([]float32{1}, []float32{1}, float32(math.NaN()))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipReferenceRMSNorm([]float32{1}, []float32{1}, float32(math.Inf(1)))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipReferenceRMSNorm([]float32{0, 0}, []float32{1, 1}, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rms is zero")
}

func TestHIPTransformerReferenceRoPEBadInputs_Bad(t *testing.T) {
	_, err := hipReferenceRoPE(nil, 0, 10000)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive and even")

	_, err = hipReferenceRoPE([]float32{1, 0}, -1, 10000)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "position")

	_, err = hipReferenceRoPE([]float32{1, 0}, 0, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "base")

	_, err = hipReferenceRoPE([]float32{1, 0}, 0, math.NaN())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipReferenceRoPE([]float32{1, 0}, 0, math.Inf(1))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipReferenceRoPEWithFrequencyDim([]float32{1, 0, 0, 1}, 0, 10000, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "frequency dimension")

	_, err = hipReferenceRoPENeoXWithFrequencyDim([]float32{1, 0, 0, 1}, 0, 10000, 4, 3)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rotary count")
}

func TestHIPTransformerReferenceAttentionBadInputs_Bad(t *testing.T) {
	_, _, err := hipReferenceSingleHeadAttention(nil, [][]float32{{1}}, [][]float32{{1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query is required")

	_, _, err = hipReferenceSingleHeadAttention([]float32{1, 2}, [][]float32{{1}}, [][]float32{{1, 2}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")

	_, _, err = hipReferenceMultiHeadAttention([]float32{1, 2}, [][]float32{{1, 2}}, [][]float32{{1, 2}}, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "head count")

	_, _, err = hipReferenceMultiHeadAttention(nil, [][]float32{{1}}, [][]float32{{1}}, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive multiple")

	_, _, err = hipReferenceMultiHeadAttention([]float32{1, 2}, [][]float32{{1, 2}}, nil, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "keys and values")

	_, _, err = hipReferenceMultiHeadAttention([]float32{1, 2}, [][]float32{{1}}, [][]float32{{1}}, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")

	_, _, err = hipReferenceCausalPrefillAttention([][]float32{{1, 2}, {1, 2}}, [][]float32{{1, 2}}, [][]float32{{1, 2}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "equal length")

	_, _, err = hipReferenceCausalPrefillAttention([][]float32{{1, 2}}, [][]float32{{1}}, [][]float32{{1, 2}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")
}

func TestHIPTransformerReferenceDecodeWithKVBadInputs_Bad(t *testing.T) {
	_, _, _, _, err := hipReferenceDecodeWithKV([]float32{1, 2}, []float32{1}, []float32{1, 2}, nil, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "new key/value")

	_, _, _, _, err = hipReferenceDecodeWithKV(
		[]float32{1, 2},
		[]float32{1, 2},
		[]float32{1, 2},
		[][]float32{{1}},
		[][]float32{{1, 2}},
	)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")
}

func TestHIPTransformerReferenceSamplerBadInputsAndTies_Bad(t *testing.T) {
	index, value, err := hipReferenceGreedySample([]float32{1, 2, 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, index)
	assertFloat32Near(t, 2, value)

	index, value, err = hipReferenceGreedySampleSuppress([]float32{1, 4, 3}, []int32{1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, index)
	assertFloat32Near(t, 3, value)

	_, _, err = hipReferenceGreedySampleSuppress([]float32{1}, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "all logits are suppressed")

	sampled, err := hipGemma4Q4HostSampleResult(
		[]float32{1, 5, 4},
		inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1},
		[]int32{1},
		nil,
		0,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, sampled.TokenID)

	minPSampled, err := hipGemma4Q4HostSampleResult(
		[]float32{1, 5, 4},
		inference.GenerateConfig{Temperature: 1, TopP: 1, MinP: 0.5, RepeatPenalty: 1},
		nil,
		nil,
		0.99,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, minPSampled.TokenID)

	_, err = hipGemma4Q4HostSampleResult(
		[]float32{1, 5, 4},
		inference.GenerateConfig{Temperature: 1, MinP: -0.1},
		nil,
		nil,
		0,
	)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "min-p")

	penalized, err := hipGemma4Q4HostSampleResult(
		[]float32{1, 5, 4},
		inference.GenerateConfig{RepeatPenalty: 2},
		nil,
		[]int32{1},
		0,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, penalized.TokenID)

	candidateSampled, err := hipGemma4Q4HostSampleCandidateResult(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1},
		nil,
		0,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, candidateSampled.TokenID)
	scratchSampled, scratchCandidates, scratchWeights, err := hipGemma4Q4HostSampleCandidateResultScratch(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1},
		nil,
		0,
		make([]hipReferenceCandidate, 0, 2),
		make([]float64, 0, 2),
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, candidateSampled, scratchSampled)
	core.AssertEqual(t, 2, cap(scratchCandidates))
	core.AssertEqual(t, 2, cap(scratchWeights))
	sortedSampled, err := hipGemma4Q4HostSampleSortedCandidateResultWorkspace(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1},
		nil,
		0,
		nil,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, candidateSampled, sortedSampled)

	candidateMinPSampled, err := hipGemma4Q4HostSampleCandidateResult(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{Temperature: 1, TopP: 1, MinP: 0.5, RepeatPenalty: 1},
		nil,
		0.99,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, candidateMinPSampled.TokenID)
	sortedMinPSampled, err := hipGemma4Q4HostSampleSortedCandidateResultWorkspace(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{Temperature: 1, TopP: 1, MinP: 0.5, RepeatPenalty: 1},
		nil,
		0.99,
		nil,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, candidateMinPSampled, sortedMinPSampled)

	candidatePenalized, err := hipGemma4Q4HostSampleCandidateResult(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{RepeatPenalty: 2},
		[]int32{1},
		0,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, candidatePenalized.TokenID)
	sortedPenalized, err := hipGemma4Q4HostSampleSortedCandidateResultWorkspace(
		[]hipGreedySampleResult{{TokenID: 1, Score: 5}, {TokenID: 2, Score: 4}},
		inference.GenerateConfig{RepeatPenalty: 2},
		[]int32{1},
		0,
		nil,
	)
	core.RequireNoError(t, err)
	core.AssertEqual(t, candidatePenalized, sortedPenalized)
	core.AssertTrue(t, hipGemma4Q4DeviceTopKSamplingRequested(inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1}), "top-k sampling can stay on device without repeat penalty")
	core.AssertTrue(t, !hipGemma4Q4DeviceTopKSamplingRequested(inference.GenerateConfig{Temperature: 1, RepeatPenalty: 1}), "temperature-only sampling has top-k disabled and takes the full-logit host path")
	core.AssertTrue(t, !hipGemma4Q4DeviceTopKSamplingRequested(inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, MinP: 0.1, RepeatPenalty: 1}), "min-p is a host sampler contract until device sampling supports it")
	core.AssertTrue(t, !hipGemma4Q4DeviceCandidateSamplingRequested(inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1}), "host candidate copy path is not the default neutral top-k route")
	core.AssertTrue(t, !hipGemma4Q4DeviceCandidateSamplingRequested(inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 2}), "repeat penalty changes the top-k set and must use full logits")
	core.AssertTrue(t, !hipGemma4Q4RepeatHistoryRequired(inference.GenerateConfig{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1}), "repeat history is unused when repeat penalty is neutral")
	core.AssertTrue(t, hipGemma4Q4RepeatHistoryRequired(inference.GenerateConfig{RepeatPenalty: 2}), "repeat history is required when repeat penalty is active")
	packed := []uint64{
		hipPackGreedyBest(1, 0),
		hipPackGreedyBest(3, 1),
		hipPackGreedyBest(2, 2),
	}
	payload := make([]byte, len(packed)*hipMLXQ4ProjectionBestBytes)
	for index, value := range packed {
		binary.LittleEndian.PutUint64(payload[index*hipMLXQ4ProjectionBestBytes:], value)
	}
	expectedPackedTop := hipTopPackedScores(packed, 2)
	core.AssertEqual(t, expectedPackedTop, hipTopPackedScoresBytes(payload, 2))
	scratchPackedTop := make([]uint64, 0, 2)
	scratchBacking := scratchPackedTop[:cap(scratchPackedTop)]
	intoPackedTop := hipTopPackedScoresBytesInto(payload, 2, scratchPackedTop)
	core.AssertEqual(t, expectedPackedTop, intoPackedTop)
	if len(intoPackedTop) > 0 && &intoPackedTop[0] != &scratchBacking[0] {
		t.Fatalf("hipTopPackedScoresBytesInto did not reuse caller-provided capacity")
	}
	sortedPayload := make([]byte, len(expectedPackedTop)*hipMLXQ4ProjectionBestBytes)
	for index, value := range expectedPackedTop {
		binary.LittleEndian.PutUint64(sortedPayload[index*hipMLXQ4ProjectionBestBytes:], value)
	}
	sortedPackedTop := hipSortedPackedScoresBytesInto(sortedPayload, 2, scratchPackedTop)
	core.AssertEqual(t, expectedPackedTop, sortedPackedTop)

	probs, err := hipReferenceTopKProbabilities([]float32{1, 2, 2}, 1, 1)
	core.RequireNoError(t, err)
	if !math.IsInf(float64(probs[0]), -1) || !math.IsInf(float64(probs[2]), -1) {
		t.Fatalf("probs = %+v, want only lower-index tied token kept", probs)
	}
	assertFloat32Near(t, 1, probs[1])

	_, err = hipReferenceTopKProbabilities(nil, 1, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits")

	_, err = hipReferenceTopKProbabilities([]float32{1}, 0, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")

	_, err = hipReferenceTopKProbabilities([]float32{1}, 1, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "temperature")

	_, err = hipReferenceTopKProbabilities([]float32{1}, 1, float32(math.NaN()))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipReferenceTopKProbabilities([]float32{1}, 1, float32(math.Inf(1)))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestHIPTransformerReferenceSplitVectorsBadInputs_Bad(t *testing.T) {
	_, err := splitHIPReferenceVectors([]float32{1}, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive multiple")

	_, err = splitHIPReferenceVectors(nil, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive multiple")

	_, err = splitHIPReferenceVectors([]float32{1, 2, 3}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive multiple")
}

func TestHIPTransformerReferenceBadInputs_Bad(t *testing.T) {
	_, err := hipReferenceEmbeddingLookup([]float32{1}, 1, 1, []int32{2})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocab")

	_, err = hipReferenceTinyPrefill(hipReferenceTinyLMConfig{VocabSize: 2, HiddenSize: 2}, []int32{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding table length")

	_, err = hipReferenceRMSNorm([]float32{1}, nil, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "weight length")

	_, err = hipReferenceRoPE([]float32{1}, 0, 10000)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive and even")

	_, _, err = hipReferenceSingleHeadAttention([]float32{1}, [][]float32{{1}}, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "keys and values")

	_, _, err = hipReferenceMultiHeadAttention([]float32{1, 2, 3}, [][]float32{{1, 2, 3}}, [][]float32{{1, 2, 3}}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "multiple of head count")

	_, _, err = hipReferenceCausalPrefillAttention([][]float32{{1}}, nil, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "queries, keys, and values")

	_, _, _, _, err = hipReferenceDecodeWithKV([]float32{1}, nil, nil, nil, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "new key/value")

	_, _, err = hipReferenceGreedySample(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits are required")

	_, err = hipReferenceTopKProbabilities([]float32{1}, 2, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")
}

func BenchmarkHIPTopPackedScores_VocabTopK64(b *testing.B) {
	const vocabSize = 256000
	values := make([]uint64, vocabSize)
	for index := range values {
		score := float32((index*1103515245+12345)&0xffff) / 4096
		if index%257 == 0 {
			score += 100
		}
		values[index] = hipPackGreedyBest(score, index)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkHIPTopPackedScoresSink = hipTopPackedScores(values, 64)
	}
}

func BenchmarkHIPTopPackedScoresBytes_VocabTopK64(b *testing.B) {
	const vocabSize = 256000
	payload := make([]byte, vocabSize*hipMLXQ4ProjectionBestBytes)
	for index := 0; index < vocabSize; index++ {
		score := float32((index*1103515245+12345)&0xffff) / 4096
		if index%257 == 0 {
			score += 100
		}
		binary.LittleEndian.PutUint64(payload[index*hipMLXQ4ProjectionBestBytes:], hipPackGreedyBest(score, index))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkHIPTopPackedScoresSink = hipTopPackedScoresBytes(payload, 64)
	}
}

func BenchmarkHIPTopPackedScoresBytesInto_VocabTopK64(b *testing.B) {
	const vocabSize = 256000
	payload := make([]byte, vocabSize*hipMLXQ4ProjectionBestBytes)
	for index := 0; index < vocabSize; index++ {
		score := float32((index*1103515245+12345)&0xffff) / 4096
		if index%257 == 0 {
			score += 100
		}
		binary.LittleEndian.PutUint64(payload[index*hipMLXQ4ProjectionBestBytes:], hipPackGreedyBest(score, index))
	}
	top := make([]uint64, 0, 64)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		top = hipTopPackedScoresBytesInto(payload, 64, top)
		benchmarkHIPTopPackedScoreSink ^= top[0]
	}
}

func BenchmarkHIPSortedPackedScoresBytesInto_TopK64(b *testing.B) {
	const topK = 64
	payload := make([]byte, topK*hipMLXQ4ProjectionBestBytes)
	for index := 0; index < topK; index++ {
		binary.LittleEndian.PutUint64(payload[index*hipMLXQ4ProjectionBestBytes:], hipPackGreedyBest(float32(topK-index), index))
	}
	top := make([]uint64, 0, topK)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		top = hipSortedPackedScoresBytesInto(payload, topK, top)
		benchmarkHIPTopPackedScoreSink ^= top[0]
	}
}

func BenchmarkHIPPackedTopKPartialPayload_VocabTopK64(b *testing.B) {
	const (
		vocabSize = 256000
		topK      = 64
	)
	chunks := (vocabSize + hipPackedTopKChunkSize - 1) / hipPackedTopKChunkSize
	partialCount := chunks * topK
	b.ReportAllocs()
	for b.Loop() {
		benchmarkHIPTopPackedScoreSink ^= uint64(partialCount)
	}
	b.ReportMetric(float64(chunks), "chunks/op")
	b.ReportMetric(float64(partialCount*hipMLXQ4ProjectionBestBytes), "partial_payload_bytes/op")
}

func BenchmarkHIPGemma4Q4HostSampleCandidateResult_TopK64(b *testing.B) {
	candidates := make([]hipGreedySampleResult, 64)
	for index := range candidates {
		candidates[index] = hipGreedySampleResult{TokenID: index, Score: float32(64 - index)}
	}
	generate := inference.GenerateConfig{Temperature: 1, TopK: 64, TopP: 0.95, RepeatPenalty: 1}

	b.ReportAllocs()
	for b.Loop() {
		result, err := hipGemma4Q4HostSampleCandidateResult(candidates, generate, nil, 0.42)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkHIPCandidateSampleResultSink = result
	}
}

func BenchmarkHIPGemma4Q4HostSampleCandidateResultScratch_TopK64(b *testing.B) {
	candidates := make([]hipGreedySampleResult, 64)
	for index := range candidates {
		candidates[index] = hipGreedySampleResult{TokenID: index, Score: float32(64 - index)}
	}
	generate := inference.GenerateConfig{Temperature: 1, TopK: 64, TopP: 0.95, RepeatPenalty: 1}
	scratchCandidates := make([]hipReferenceCandidate, 0, 64)
	scratchWeights := make([]float64, 0, 64)

	b.ReportAllocs()
	for b.Loop() {
		result, nextCandidates, nextWeights, err := hipGemma4Q4HostSampleCandidateResultScratch(candidates, generate, nil, 0.42, scratchCandidates, scratchWeights)
		if err != nil {
			b.Fatal(err)
		}
		scratchCandidates = nextCandidates
		scratchWeights = nextWeights
		benchmarkHIPCandidateSampleResultSink = result
	}
}

func BenchmarkHIPGemma4Q4HostSampleResult_FullVocabulary(b *testing.B) {
	logits := make([]float32, 262144)
	for index := range logits {
		logits[index] = float32((index*7919)%65537) / 1024
	}
	generate := inference.GenerateConfig{Temperature: 1, TopP: 1, RepeatPenalty: 1}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		result, err := hipGemma4Q4HostSampleResult(logits, generate, nil, nil, 0.42)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkHIPCandidateSampleResultSink = result
	}
}

func BenchmarkHIPGemma4Q4HostSampleSortedCandidateResultScratch_TopK64(b *testing.B) {
	candidates := make([]hipGreedySampleResult, 64)
	for index := range candidates {
		candidates[index] = hipGreedySampleResult{TokenID: index, Score: float32(64 - index)}
	}
	generate := inference.GenerateConfig{Temperature: 1, TopK: 64, TopP: 0.95, RepeatPenalty: 1}
	scratchCandidates := make([]hipReferenceCandidate, 0, 64)
	scratchWeights := make([]float64, 0, 64)

	b.ReportAllocs()
	for b.Loop() {
		result, nextCandidates, nextWeights, err := hipGemma4Q4HostSampleCandidateResultScratchOrder(candidates, generate, nil, 0.42, scratchCandidates, scratchWeights, true)
		if err != nil {
			b.Fatal(err)
		}
		scratchCandidates = nextCandidates
		scratchWeights = nextWeights
		benchmarkHIPCandidateSampleResultSink = result
	}
}

func BenchmarkHIPGemma4Q4RepeatHistoryRequired_Hot(b *testing.B) {
	generate := inference.GenerateConfig{Temperature: 1, TopK: 64, TopP: 0.95, RepeatPenalty: 1}
	b.ReportAllocs()
	for b.Loop() {
		if hipGemma4Q4RepeatHistoryRequired(generate) {
			b.Fatal("neutral repeat penalty should not require history")
		}
	}
}

func hipReferenceTinyLMFixture() hipReferenceTinyLMConfig {
	return hipReferenceTinyLMConfig{
		EmbeddingTable: []float32{
			1, 0,
			0, 1,
			1, 1,
		},
		OutputWeights: []float32{
			1, 0,
			0, 1,
			1, 1,
		},
		VocabSize:  3,
		HiddenSize: 2,
	}
}
