// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestNativeAdamWState_PacksParametersAndMoments_Good(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "a", Shape: []int{2, 3}, Values: []float32{1, 2, 3, 4, 5, 6}},
		{Name: "b", Shape: []int{2}, Values: []float32{7, 8}},
	}, NativeAdamWConfig{LearningRate: 0.1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 24, len(state.Slab))
	core.AssertEqual(t, 8, len(state.Parameters()))
	core.AssertEqual(t, 8, len(state.FirstMoment()))
	core.AssertEqual(t, 8, len(state.SecondMoment()))
	core.AssertEqual(t, "a", state.Layout[0].Name)
	core.AssertEqual(t, 0, state.Layout[0].Offset)
	core.AssertEqual(t, 6, state.Layout[0].Length)
	core.AssertEqual(t, "b", state.Layout[1].Name)
	core.AssertEqual(t, 6, state.Layout[1].Offset)
	core.AssertEqual(t, []float32{1, 2, 3, 4, 5, 6, 7, 8}, append([]float32(nil), state.Parameters()...))
	view, ok := state.ParamView(1)
	core.AssertTrue(t, ok)
	view[0] = 70
	core.AssertEqual(t, float32(70), state.Parameters()[6])
}

func TestNativeAdamWState_StepInPlace_Good(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	err = state.StepInPlace([][]float32{{0.5, -0.25}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, state.Step)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
	assertAdamWFloat32Near(t, 0.05, state.FirstMoment()[0], 0.0001)
	assertAdamWFloat32Near(t, -0.025, state.FirstMoment()[1], 0.0001)
	assertAdamWFloat32Near(t, 0.00025, state.SecondMoment()[0], 0.00001)
	assertAdamWFloat32Near(t, 0.0000625, state.SecondMoment()[1], 0.00001)
}

func TestNativeAdamWState_WeightDecay_Good(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{10}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0.1})
	core.RequireNoError(t, err)

	err = state.StepInPlace([][]float32{{0}})

	core.RequireNoError(t, err)
	assertAdamWFloat32Near(t, 9.9, state.Parameters()[0], 0.0001)
}

func TestNativeLoRAAdamWState_Good(t *testing.T) {
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 2, 3, 4},
		[]float32{5, 6, 7, 8, 9, 10},
		3,
		2,
		2,
		NativeAdamWConfig{},
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "lora_a", state.Layout[0].Name)
	core.AssertEqual(t, []int{2, 2}, state.Layout[0].Shape)
	core.AssertEqual(t, "lora_b", state.Layout[1].Name)
	core.AssertEqual(t, []int{3, 2}, state.Layout[1].Shape)
	core.AssertEqual(t, 10, len(state.Parameters()))
	core.AssertTrue(t, state.Config.Packed)
}

func TestNativeAdamWState_Bad(t *testing.T) {
	_, err := NewNativeAdamWState(nil, NativeAdamWConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "parameters are required")

	_, err = NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1}}}, NativeAdamWConfig{Beta1: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "beta1")

	_, err = NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1}}}, NativeAdamWConfig{Eps: math.NaN()})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "epsilon")

	_, err = NewNativeAdamWState([]NativeAdamWParam{{Shape: []int{2}, Values: []float32{1}}}, NativeAdamWConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape")

	_, err = NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{float32(math.Inf(1))}}}, NativeAdamWConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)
	err = state.StepInPlace(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "gradients length")
	err = state.StepInPlace([][]float32{{1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "does not match")
	err = state.StepInPlace([][]float32{{1, float32(math.NaN())}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = NewNativeLoRAAdamWState([]float32{1}, []float32{1}, 1, 2, 1, NativeAdamWConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "A length")
}

func assertAdamWFloat32Near(t *testing.T, want, got, tolerance float32) {
	t.Helper()
	if got < want-tolerance || got > want+tolerance {
		t.Fatalf("value = %f, want %f within %f", got, want, tolerance)
	}
}
