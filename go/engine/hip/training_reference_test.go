// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestTrainingReferenceCrossEntropy_Good(t *testing.T) {
	loss, perplexity, err := rocmReferenceCrossEntropyLoss([][]float32{{2, 0}, {0, 2}}, []int{0, 1})

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.1269, loss, 0.0001)
	assertFloat64Near(t, 1.1353, perplexity, 0.0001)
}

func TestTrainingReferenceCrossEntropy_Good_StableLargeLogits(t *testing.T) {
	loss, perplexity, err := rocmReferenceCrossEntropyLoss([][]float32{{1000, 999, 998}}, []int{0})

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.4076, loss, 0.0001)
	assertFloat64Near(t, 1.5032, perplexity, 0.0001)
}

func TestTrainingReferenceCrossEntropy_Bad_RejectsEmptyInputs(t *testing.T) {
	_, _, err := rocmReferenceCrossEntropyLoss(nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-empty")
}

func TestTrainingReferenceCrossEntropy_Bad_RejectsMismatchedTargets(t *testing.T) {
	_, _, err := rocmReferenceCrossEntropyLoss([][]float32{{1, 2}}, []int{0, 1})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "equal length")
}

func TestTrainingReferenceCrossEntropy_Bad_RejectsEmptyLogitRow(t *testing.T) {
	_, _, err := rocmReferenceCrossEntropyLoss([][]float32{{}}, []int{0})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "row")
}

func TestTrainingReferenceCrossEntropy_Bad_RejectsNegativeTarget(t *testing.T) {
	_, _, err := rocmReferenceCrossEntropyLoss([][]float32{{1, 2}}, []int{-1})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")
}

func TestTrainingReferenceCrossEntropy_Bad_RejectsTargetOutOfRange(t *testing.T) {
	_, _, err := rocmReferenceCrossEntropyLoss([][]float32{{1, 2}}, []int{3})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")
}

func TestTrainingReferenceCrossEntropy_Bad_RejectsNonFiniteLogits(t *testing.T) {
	_, _, err := rocmReferenceCrossEntropyLoss([][]float32{{1, float32(math.NaN())}}, []int{0})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestTrainingReferenceDistillationKL_Good(t *testing.T) {
	kl, err := rocmReferenceDistillationKL(
		[][]float32{{1, 0}},
		[][]float32{{2, 0}},
		1,
	)

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.0671, kl, 0.0001)
}

func TestTrainingReferenceDistillationKL_Bad_RejectsEmptyInputs(t *testing.T) {
	_, err := rocmReferenceDistillationKL(nil, nil, 1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-empty")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsMismatchedBatches(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{1}}, [][]float32{{1}, {2}}, 1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "equal length")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsTemperature(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{1}}, [][]float32{{1}}, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "temperature")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsNegativeTemperature(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{1}}, [][]float32{{1}}, -1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "temperature")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsNonFiniteTemperature(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{1}}, [][]float32{{1}}, math.Inf(1))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsEmptyVocabulary(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{}}, [][]float32{{}}, 1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vocabulary")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsMismatchedVocabulary(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{1, 2}}, [][]float32{{1}}, 1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vocabulary")
}

func TestTrainingReferenceDistillationKL_Bad_RejectsNonFiniteLogits(t *testing.T) {
	_, err := rocmReferenceDistillationKL([][]float32{{1, 2}}, [][]float32{{1, float32(math.Inf(-1))}}, 1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestTrainingReferenceNormalizeAdvantages_Good(t *testing.T) {
	advantages, err := rocmReferenceNormalizeAdvantages([]float64{1, 2, 3})

	core.RequireNoError(t, err)
	assertFloat64Near(t, -1.2247, advantages[0], 0.0001)
	assertFloat64Near(t, 0, advantages[1], 0.0001)
	assertFloat64Near(t, 1.2247, advantages[2], 0.0001)
}

func TestTrainingReferenceNormalizeAdvantages_Good_ZeroVariance(t *testing.T) {
	advantages, err := rocmReferenceNormalizeAdvantages([]float64{5, 5})

	core.RequireNoError(t, err)
	core.AssertEqual(t, []float64{0, 0}, advantages)
}

func TestTrainingReferenceNormalizeAdvantages_Bad_RejectsEmptyRewards(t *testing.T) {
	_, err := rocmReferenceNormalizeAdvantages(nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "required")
}

func TestTrainingReferenceNormalizeAdvantages_Bad_RejectsNonFiniteRewards(t *testing.T) {
	_, err := rocmReferenceNormalizeAdvantages([]float64{1, math.NaN()})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func assertFloat64Near(t *testing.T, want, got, tolerance float64) {
	t.Helper()
	if got < want-tolerance || got > want+tolerance {
		t.Fatalf("value = %f, want %f within %f", got, want, tolerance)
	}
}
