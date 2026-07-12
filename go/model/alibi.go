// SPDX-Licence-Identifier: EUPL-1.2

package model

import "math"

// ALiBiSlopes returns the per-head slopes from the published ALiBi reference.
func ALiBiSlopes(heads int) []float32 {
	if heads <= 0 {
		return nil
	}
	power := 1
	for power*2 <= heads {
		power *= 2
	}
	start := math.Pow(2, -math.Pow(2, -(math.Log2(float64(power))-3)))
	ratio := start
	out := make([]float32, 0, heads)
	for i := 0; i < power; i++ {
		out = append(out, float32(start*math.Pow(ratio, float64(i))))
	}
	if power != heads {
		extra := ALiBiSlopes(2 * power)
		for i := 0; len(out) < heads; i += 2 {
			out = append(out, extra[i])
		}
	}
	return out
}

// ApplyALiBi adds slope*(keyPosition-queryPosition) to one head's scores.
func ApplyALiBi(scores []float64, slope float32, queryPosition, firstKeyPosition int) {
	for i := range scores {
		scores[i] += float64(slope) * float64(firstKeyPosition+i-queryPosition)
	}
}
