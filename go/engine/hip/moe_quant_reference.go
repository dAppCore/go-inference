// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type rocmExpertRoute struct {
	ID    int
	Score float32
	Prob  float32
}

type rocmJANGTQDescriptor struct {
	WeightFormat string
	Bits         int
	GroupSize    int
}

func rocmReferenceRouteExperts(logits []float32, topK, layer int, sink inference.ProbeSink) ([]rocmExpertRoute, error) {
	if len(logits) == 0 {
		return nil, core.E("rocm.MoE.Router", "router logits are required", nil)
	}
	if topK <= 0 || topK > len(logits) {
		return nil, core.E("rocm.MoE.Router", "top-k must be within the expert count", nil)
	}
	if !rocmFloat32SliceFinite(logits) {
		return nil, core.E("rocm.MoE.Router", "router logits must be finite", nil)
	}
	routes := make([]rocmExpertRoute, len(logits))
	for i, logit := range logits {
		routes[i] = rocmExpertRoute{ID: i, Score: logit}
	}
	sort.SliceStable(routes, func(i, j int) bool {
		if routes[i].Score == routes[j].Score {
			return routes[i].ID < routes[j].ID
		}
		return routes[i].Score > routes[j].Score
	})
	routes = append([]rocmExpertRoute(nil), routes[:topK]...)
	selected := make([]float32, len(routes))
	for index, route := range routes {
		selected[index] = route.Score
	}
	probs := softmaxFloat32(selected)
	for index := range routes {
		routes[index].Prob = probs[index]
	}
	if sink != nil {
		ids := make([]int, len(routes))
		routeProbs := make([]float32, len(routes))
		for i, route := range routes {
			ids[i] = route.ID
			routeProbs[i] = route.Prob
		}
		sink.EmitProbe(inference.ProbeEvent{
			Kind:  inference.ProbeEventRouterDecision,
			Phase: inference.ProbePhasePrefill,
			Labels: map[string]string{
				"backend": "rocm",
				"source":  "cpu_reference",
			},
			RouterDecision: &inference.ProbeRouterDecision{
				Layer:       layer,
				ExpertIDs:   ids,
				ExpertProbs: routeProbs,
			},
		})
	}
	return routes, nil
}

func rocmReferenceLazyExpertResidency(routes []rocmExpertRoute, totalExperts int) ([]bool, error) {
	if totalExperts <= 0 {
		return nil, core.E("rocm.MoE.LazyExperts", "expert count must be positive", nil)
	}
	resident := make([]bool, totalExperts)
	for _, route := range routes {
		if route.ID < 0 || route.ID >= totalExperts {
			return nil, core.E("rocm.MoE.LazyExperts", core.Sprintf("expert id %d outside expert count %d", route.ID, totalExperts), nil)
		}
		resident[route.ID] = true
	}
	return resident, nil
}

func rocmReferenceJANGTQProjection(input []float32, packedWeights []byte, desc rocmJANGTQDescriptor, rows, cols int, scale float32, bias []float32) ([]float32, error) {
	if rows <= 0 {
		return nil, core.E("rocm.JANGTQ.ReferenceProjection", "row count must be positive", nil)
	}
	output := make([]float32, rows)
	if err := rocmReferenceJANGTQProjectionInto(output, input, packedWeights, nil, desc, rows, cols, scale, bias); err != nil {
		return nil, err
	}
	return output, nil
}

func rocmReferenceJANGTQProjectionInto(output []float32, input []float32, packedWeights []byte, quantized []int8, desc rocmJANGTQDescriptor, rows, cols int, scale float32, bias []float32) error {
	if err := validateROCmJANGTQDescriptor(desc); err != nil {
		return err
	}
	if scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.JANGTQ.ReferenceProjection", "scale must be positive and finite", nil)
	}
	if err := validateHIPProjectionShape(len(input), rows*cols, len(bias), rows, cols); err != nil {
		return err
	}
	if !rocmFloat32SliceFinite(input) || !rocmFloat32SliceFinite(bias) {
		return core.E("rocm.JANGTQ.ReferenceProjection", "input and bias values must be finite", nil)
	}
	if len(output) != rows {
		return core.E("rocm.JANGTQ.ReferenceProjection", "output row count mismatch", nil)
	}
	quantizedCount := rows * cols
	if cap(quantized) < quantizedCount {
		var err error
		quantized, err = unpackROCmSignedBits(packedWeights, desc.Bits, quantizedCount)
		if err != nil {
			return err
		}
	} else {
		quantized = quantized[:quantizedCount]
		if err := unpackROCmSignedBitsInto(quantized, packedWeights, desc.Bits); err != nil {
			return err
		}
	}
	for row := 0; row < rows; row++ {
		sum := float32(0)
		if len(bias) > 0 {
			sum = bias[row]
		}
		for col := 0; col < cols; col++ {
			sum += input[col] * float32(quantized[row*cols+col]) * scale
		}
		output[row] = sum
	}
	return nil
}

func validateROCmJANGTQDescriptor(desc rocmJANGTQDescriptor) error {
	format := core.Lower(desc.WeightFormat)
	if !core.Contains(format, "mxtq") && !core.Contains(format, "jangtq") {
		return core.E("rocm.JANGTQ.Descriptor", "weight format must be MXTQ/JANGTQ", nil)
	}
	switch desc.Bits {
	case 2, 4, 8:
	default:
		return core.E("rocm.JANGTQ.Descriptor", core.Sprintf("unsupported bit layout %d", desc.Bits), nil)
	}
	if desc.GroupSize <= 0 || desc.GroupSize&(desc.GroupSize-1) != 0 {
		return core.E("rocm.JANGTQ.Descriptor", "group size must be a positive power of two", nil)
	}
	return nil
}

func rocmReferenceCodebookLookup(codes []uint8, codebook []float32, codeDim int) ([]float32, error) {
	if codeDim <= 0 {
		return nil, core.E("rocm.Codebook.Lookup", "code dimension must be positive", nil)
	}
	out := make([]float32, len(codes)*codeDim)
	if err := rocmReferenceCodebookLookupInto(out, codes, codebook, codeDim); err != nil {
		return nil, err
	}
	return out, nil
}

func rocmReferenceCodebookLookupInto(out []float32, codes []uint8, codebook []float32, codeDim int) error {
	if codeDim <= 0 {
		return core.E("rocm.Codebook.Lookup", "code dimension must be positive", nil)
	}
	if len(codebook) == 0 || len(codebook)%codeDim != 0 {
		return core.E("rocm.Codebook.Lookup", "codebook shape does not match code dimension", nil)
	}
	if !rocmFloat32SliceFinite(codebook) {
		return core.E("rocm.Codebook.Lookup", "codebook values must be finite", nil)
	}
	if len(out) != len(codes)*codeDim {
		return core.E("rocm.Codebook.Lookup", "output shape does not match codes and code dimension", nil)
	}
	codeCount := len(codebook) / codeDim
	for codeIndex, code := range codes {
		index := int(code)
		if index >= codeCount {
			return core.E("rocm.Codebook.Lookup", core.Sprintf("code %d outside codebook size %d", index, codeCount), nil)
		}
		start := index * codeDim
		copy(out[codeIndex*codeDim:(codeIndex+1)*codeDim], codebook[start:start+codeDim])
	}
	return nil
}

func rocmReferenceResidualSummary(layer int, values []float32, sink inference.ProbeSink) (inference.ProbeResidualSummary, error) {
	if len(values) == 0 {
		return inference.ProbeResidualSummary{}, core.E("rocm.Residual.Reference", "residual values are required", nil)
	}
	if !rocmFloat32SliceFinite(values) {
		return inference.ProbeResidualSummary{}, core.E("rocm.Residual.Reference", "residual values must be finite", nil)
	}
	sum := float64(0)
	sumSquares := float64(0)
	for _, value := range values {
		v := float64(value)
		sum += v
		sumSquares += v * v
	}
	summary := inference.ProbeResidualSummary{
		Layer: layer,
		Mean:  sum / float64(len(values)),
		RMS:   math.Sqrt(sumSquares / float64(len(values))),
		Norm:  math.Sqrt(sumSquares),
	}
	if sink != nil {
		sink.EmitProbe(inference.ProbeEvent{
			Kind:  inference.ProbeEventResidual,
			Phase: inference.ProbePhasePrefill,
			Labels: map[string]string{
				"backend": "rocm",
				"source":  "cpu_reference",
			},
			Residual: &summary,
		})
	}
	return summary, nil
}

func unpackROCmSignedBits(packed []byte, bits, count int) ([]int8, error) {
	out := make([]int8, count)
	if err := unpackROCmSignedBitsInto(out, packed, bits); err != nil {
		return nil, err
	}
	return out, nil
}

func unpackROCmSignedBitsInto(out []int8, packed []byte, bits int) error {
	if bits != 2 && bits != 4 && bits != 8 {
		return core.E("rocm.JANGTQ.Unpack", core.Sprintf("unsupported bit width %d", bits), nil)
	}
	requiredBytes := (bits*len(out) + 7) / 8
	if len(packed) < requiredBytes {
		return core.E("rocm.JANGTQ.Unpack", core.Sprintf("packed weights need %d bytes, got %d", requiredBytes, len(packed)), nil)
	}
	mask := (1 << bits) - 1
	signBit := 1 << (bits - 1)
	for i := range out {
		bitOffset := i * bits
		byteIndex := bitOffset / 8
		shift := bitOffset % 8
		raw := int(packed[byteIndex] >> shift)
		if shift+bits > 8 {
			raw |= int(packed[byteIndex+1]) << (8 - shift)
		}
		raw &= mask
		if raw&signBit != 0 {
			raw -= 1 << bits
		}
		out[i] = int8(raw)
	}
	return nil
}

func softmaxFloat32(values []float32) []float32 {
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	out := make([]float32, len(values))
	sum := float64(0)
	for i, value := range values {
		exp := math.Exp(float64(value - maxValue))
		out[i] = float32(exp)
		sum += exp
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out
}
