// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestMoEReferenceRouter_Good_SelectsTopKAndEmitsProbe(t *testing.T) {
	var events []inference.ProbeEvent
	routes, err := rocmReferenceRouteExperts(
		[]float32{0.1, 2, 1, -1},
		2,
		7,
		inference.ProbeSinkFunc(func(event inference.ProbeEvent) { events = append(events, event) }),
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(routes))
	core.AssertEqual(t, 1, routes[0].ID)
	core.AssertEqual(t, 2, routes[1].ID)
	if routes[0].Prob <= routes[1].Prob {
		t.Fatalf("routes = %+v, want first route probability higher than second", routes)
	}
	core.AssertEqual(t, 1, len(events))
	core.AssertEqual(t, inference.ProbeEventRouterDecision, events[0].Kind)
	core.AssertEqual(t, 7, events[0].RouterDecision.Layer)
	core.AssertEqual(t, []int{1, 2}, events[0].RouterDecision.ExpertIDs)
}

func TestMoEReferenceRouter_Good_TieBreaksByExpertID(t *testing.T) {
	routes, err := rocmReferenceRouteExperts([]float32{1, 2, 2}, 2, 0, nil)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, routes[0].ID)
	core.AssertEqual(t, 2, routes[1].ID)
}

func TestMoEReferenceRouter_Bad_RejectsEmptyLogits(t *testing.T) {
	_, err := rocmReferenceRouteExperts(nil, 1, 0, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits")
}

func TestMoEReferenceRouter_Bad_RejectsInvalidTopK(t *testing.T) {
	_, err := rocmReferenceRouteExperts([]float32{1}, 0, 0, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")

	_, err = rocmReferenceRouteExperts([]float32{1}, 2, 0, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")
}

func TestMoEReferenceRouter_Bad_RejectsNonFiniteLogits(t *testing.T) {
	_, err := rocmReferenceRouteExperts([]float32{1, float32(math.Inf(1))}, 1, 0, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestMoEReferenceLazyExperts_Good_LoadsSelectedOnly(t *testing.T) {
	resident, err := rocmReferenceLazyExpertResidency([]rocmExpertRoute{{ID: 3}, {ID: 1}}, 5)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []bool{false, true, false, true, false}, resident)
}

func TestMoEReferenceLazyExperts_Bad_RejectsInvalidExpertCount(t *testing.T) {
	_, err := rocmReferenceLazyExpertResidency(nil, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "expert count")
}

func TestMoEReferenceLazyExperts_Bad_RejectsOutOfRangeRoute(t *testing.T) {
	_, err := rocmReferenceLazyExpertResidency([]rocmExpertRoute{{ID: -1}}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside expert count")

	_, err = rocmReferenceLazyExpertResidency([]rocmExpertRoute{{ID: 2}}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside expert count")
}

func TestJANGTQReference_Good_PackedProjection(t *testing.T) {
	output, err := rocmReferenceJANGTQProjection(
		[]float32{2, 4},
		[]byte{0x8d}, // signed 2-bit weights: [1, -1, 0, -2]
		rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
		2,
		2,
		0.5,
		[]float32{0, 1},
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, -3}, output, 0)
}

func TestJANGTQReference_Bad_RejectsInvalidBitLayout(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{1},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 3, GroupSize: 64},
		1,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported bit layout")
}

func TestJANGTQReference_Bad_RejectsInvalidFormat(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{1},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "plain", Bits: 2, GroupSize: 64},
		1,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "weight format")
}

func TestJANGTQReference_Bad_RejectsInvalidGroupSize(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{1},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "jangtq", Bits: 2, GroupSize: 3},
		1,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "group size")
}

func TestJANGTQReference_Bad_RejectsInvalidScale(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{1},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "jangtq", Bits: 2, GroupSize: 64},
		1,
		1,
		0,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scale")
}

func TestJANGTQReference_Bad_RejectsNonFiniteValues(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{float32(math.NaN())},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "jangtq", Bits: 2, GroupSize: 64},
		1,
		1,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = rocmReferenceJANGTQProjection(
		[]float32{1},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "jangtq", Bits: 2, GroupSize: 64},
		1,
		1,
		float32(math.Inf(1)),
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestJANGTQReference_Bad_RejectsProjectionShape(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{1},
		[]byte{0},
		rocmJANGTQDescriptor{WeightFormat: "jangtq", Bits: 2, GroupSize: 64},
		1,
		2,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input length")
}

func TestJANGTQReference_Bad_RejectsShortPackedWeights(t *testing.T) {
	_, err := rocmReferenceJANGTQProjection(
		[]float32{1, 2},
		nil,
		rocmJANGTQDescriptor{WeightFormat: "jangtq", Bits: 4, GroupSize: 64},
		1,
		2,
		1,
		nil,
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "packed weights")
}

func TestJANGTQReferenceUnpack_Good_Signed4BitValues(t *testing.T) {
	values, err := unpackROCmSignedBits([]byte{0x8f}, 4, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int8{-1, -8}, values)
}

func TestJANGTQReferenceUnpack_Bad_RejectsUnsupportedBits(t *testing.T) {
	_, err := unpackROCmSignedBits([]byte{0}, 3, 1)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported bit width")
}

func TestCodebookReference_Good_Lookup(t *testing.T) {
	output, err := rocmReferenceCodebookLookup(
		[]uint8{2, 0},
		[]float32{1, 2, 3, 4, 5, 6},
		2,
	)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5, 6, 1, 2}, output, 0)
}

func TestCodebookReference_Bad_RejectsInvalidCode(t *testing.T) {
	_, err := rocmReferenceCodebookLookup([]uint8{3}, []float32{1, 2, 3, 4, 5, 6}, 2)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside codebook size")
}

func TestCodebookReference_Good_EmptyCodesReturnEmpty(t *testing.T) {
	output, err := rocmReferenceCodebookLookup(nil, []float32{1, 2, 3, 4}, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(output))
}

func TestCodebookReference_Bad_RejectsInvalidCodeDimension(t *testing.T) {
	_, err := rocmReferenceCodebookLookup([]uint8{0}, []float32{1}, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")
}

func TestCodebookReference_Bad_RejectsInvalidCodebookShape(t *testing.T) {
	_, err := rocmReferenceCodebookLookup([]uint8{0}, nil, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape")

	_, err = rocmReferenceCodebookLookup([]uint8{0}, []float32{1, 2, 3}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape")
}

func TestCodebookReference_Bad_RejectsNonFiniteCodebook(t *testing.T) {
	_, err := rocmReferenceCodebookLookup([]uint8{0}, []float32{1, float32(math.NaN())}, 2)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestResidualReference_Good_SummarisesAndEmitsProbe(t *testing.T) {
	var events []inference.ProbeEvent
	summary, err := rocmReferenceResidualSummary(4, []float32{1, -1, 2, -2}, inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))

	core.RequireNoError(t, err)
	assertFloat32Near(t, 0, float32(summary.Mean))
	assertFloat32Near(t, 1.5811, float32(summary.RMS))
	assertFloat32Near(t, 3.1622, float32(summary.Norm))
	core.AssertEqual(t, 1, len(events))
	core.AssertEqual(t, inference.ProbeEventResidual, events[0].Kind)
	core.AssertEqual(t, 4, events[0].Residual.Layer)
}

func TestResidualReference_Bad_RejectsEmptyValues(t *testing.T) {
	_, err := rocmReferenceResidualSummary(0, nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "required")
}

func TestResidualReference_Bad_RejectsNonFiniteValues(t *testing.T) {
	_, err := rocmReferenceResidualSummary(0, []float32{1, float32(math.Inf(-1))}, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}
