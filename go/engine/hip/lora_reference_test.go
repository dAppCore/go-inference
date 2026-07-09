// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestLoRAReferenceProjection_Good_AppliesLowRankDelta(t *testing.T) {
	output, err := rocmReferenceLoRAProjection(
		[]float32{2, 3},
		[]float32{1, 0, 0, 1},
		[]float32{1, 1},
		[]float32{2, -1},
		2,
		2,
		1,
		0.5,
		nil,
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{7, 0.5}, output, 0)
}

func TestLoRAReferenceProjection_Good_PreservesBiasAndRankScaling(t *testing.T) {
	output, err := rocmReferenceLoRAProjection(
		[]float32{1, 2},
		[]float32{1, 1},
		[]float32{1, 0, 0, 1},
		[]float32{1, 1},
		1,
		2,
		2,
		4,
		[]float32{0.5},
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{9.5}, output, 0)
}

func TestLoRAReferenceProjection_Bad_RejectsShapeMismatch(t *testing.T) {
	_, err := rocmReferenceLoRAProjection([]float32{1}, []float32{1}, []float32{1}, nil, 1, 1, 1, 1, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA B length")
}

func TestLoRAReferenceProjection_Bad_RejectsBaseInputShape(t *testing.T) {
	_, err := rocmReferenceLoRAProjection(
		[]float32{1},
		[]float32{1, 1},
		[]float32{1, 1},
		[]float32{1},
		1,
		2,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input length")
}

func TestLoRAReferenceProjection_Bad_RejectsBaseWeightShape(t *testing.T) {
	_, err := rocmReferenceLoRAProjection(
		[]float32{1, 2},
		[]float32{1},
		[]float32{1, 1},
		[]float32{1},
		1,
		2,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "weight length")
}

func TestLoRAReferenceProjection_Bad_RejectsBiasShape(t *testing.T) {
	_, err := rocmReferenceLoRAProjection(
		[]float32{1, 2},
		[]float32{1, 1},
		[]float32{1, 1},
		[]float32{1},
		1,
		2,
		1,
		1,
		[]float32{0, 1},
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "bias length")
}

func TestLoRAReferenceProjection_Bad_RejectsLoRAALength(t *testing.T) {
	_, err := rocmReferenceLoRAProjection(
		[]float32{1, 2},
		[]float32{1, 1},
		[]float32{1, 1, 1},
		[]float32{1, 1},
		1,
		2,
		2,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA A length")
}

func TestLoRAReferenceProjection_Bad_RejectsLoRABLength(t *testing.T) {
	_, err := rocmReferenceLoRAProjection(
		[]float32{1, 2},
		[]float32{1, 1},
		[]float32{1, 1},
		nil,
		1,
		2,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA B length")
}

func TestLoRAReferenceProjection_Bad_RejectsInvalidRankAndAlpha(t *testing.T) {
	_, err := rocmReferenceLoRAProjection([]float32{1}, []float32{1}, nil, nil, 1, 1, 0, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rank must be positive")

	_, err = rocmReferenceLoRAProjection([]float32{1}, []float32{1}, []float32{1}, []float32{1}, 1, 1, 1, 0, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "alpha must be positive")
}

func TestLoRAReferenceProjection_Bad_RejectsNonFiniteAlpha(t *testing.T) {
	for _, alpha := range []float32{float32(math.Inf(1)), float32(math.NaN())} {
		_, err := rocmReferenceLoRAProjection([]float32{1}, []float32{1}, []float32{1}, []float32{1}, 1, 1, 1, alpha, nil)

		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "alpha must be positive")
	}
}
