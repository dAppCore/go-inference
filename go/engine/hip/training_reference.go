// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"

	core "dappco.re/go"
)

func rocmReferenceCrossEntropyLoss(logits [][]float32, targets []int) (float64, float64, error) {
	if len(logits) == 0 || len(logits) != len(targets) {
		return 0, 0, core.E("rocm.Training.ReferenceCrossEntropy", "logits and targets must be non-empty and equal length", nil)
	}
	total := float64(0)
	for i, row := range logits {
		if len(row) == 0 {
			return 0, 0, core.E("rocm.Training.ReferenceCrossEntropy", "logit row must be non-empty", nil)
		}
		if !rocmFloat32SliceFinite(row) {
			return 0, 0, core.E("rocm.Training.ReferenceCrossEntropy", "logit values must be finite", nil)
		}
		target := targets[i]
		if target < 0 || target >= len(row) {
			return 0, 0, core.E("rocm.Training.ReferenceCrossEntropy", core.Sprintf("target %d outside vocabulary size %d", target, len(row)), nil)
		}
		total += logSumExpFloat32(row) - float64(row[target])
	}
	loss := total / float64(len(logits))
	return loss, math.Exp(loss), nil
}

func rocmReferenceDistillationKL(studentLogits, teacherLogits [][]float32, temperature float64) (float64, error) {
	if len(studentLogits) == 0 || len(studentLogits) != len(teacherLogits) {
		return 0, core.E("rocm.Training.ReferenceDistillationKL", "student and teacher logits must be non-empty and equal length", nil)
	}
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return 0, core.E("rocm.Training.ReferenceDistillationKL", "temperature must be positive and finite", nil)
	}
	total := float64(0)
	for i := range studentLogits {
		if len(studentLogits[i]) == 0 || len(studentLogits[i]) != len(teacherLogits[i]) {
			return 0, core.E("rocm.Training.ReferenceDistillationKL", "student and teacher vocabulary sizes must match", nil)
		}
		if !rocmFloat32SliceFinite(studentLogits[i]) || !rocmFloat32SliceFinite(teacherLogits[i]) {
			return 0, core.E("rocm.Training.ReferenceDistillationKL", "student and teacher logits must be finite", nil)
		}
		studentLogProbs := logSoftmaxWithTemperature(studentLogits[i], temperature)
		teacherLogProbs := logSoftmaxWithTemperature(teacherLogits[i], temperature)
		for j := range studentLogProbs {
			teacherProb := math.Exp(teacherLogProbs[j])
			total += teacherProb * (teacherLogProbs[j] - studentLogProbs[j])
		}
	}
	return total * temperature * temperature / float64(len(studentLogits)), nil
}

func rocmReferenceNormalizeAdvantages(rewards []float64) ([]float64, error) {
	if len(rewards) == 0 {
		return nil, core.E("rocm.Training.ReferenceGRPO", "rewards are required", nil)
	}
	mean := float64(0)
	for _, reward := range rewards {
		if math.IsNaN(reward) || math.IsInf(reward, 0) {
			return nil, core.E("rocm.Training.ReferenceGRPO", "rewards must be finite", nil)
		}
		mean += reward
	}
	mean /= float64(len(rewards))
	variance := float64(0)
	for _, reward := range rewards {
		diff := reward - mean
		variance += diff * diff
	}
	variance /= float64(len(rewards))
	if variance == 0 {
		return make([]float64, len(rewards)), nil
	}
	stddev := math.Sqrt(variance)
	out := make([]float64, len(rewards))
	for i, reward := range rewards {
		out[i] = (reward - mean) / stddev
	}
	return out, nil
}

func rocmFloat32SliceFinite(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func logSumExpFloat32(values []float32) float64 {
	maxValue := float64(values[0])
	for _, value := range values[1:] {
		if float64(value) > maxValue {
			maxValue = float64(value)
		}
	}
	sum := float64(0)
	for _, value := range values {
		sum += math.Exp(float64(value) - maxValue)
	}
	return maxValue + math.Log(sum)
}

func logSoftmaxWithTemperature(values []float32, temperature float64) []float64 {
	scaled := make([]float32, len(values))
	for i, value := range values {
		scaled[i] = float32(float64(value) / temperature)
	}
	normalizer := logSumExpFloat32(scaled)
	out := make([]float64, len(values))
	for i, value := range scaled {
		out[i] = float64(value) - normalizer
	}
	return out
}
