// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import "testing"

// BenchmarkRewardStats measures the mean + population-stddev reduction over a
// GRPO group's rollouts — the per-step normalisation input that feeds every
// rollout's advantage. Index iteration keeps the two passes copy-free over the
// (string + slice heavy) Rollout struct; this instrument holds that at zero
// allocations. A 16-rollout group is a representative GRPO group size.
func BenchmarkRewardStats(b *testing.B) {
	rollouts := make([]Rollout, 16)
	for i := range rollouts {
		rollouts[i].Reward = float64((i*37)%11) - 5.0
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = RewardStats(rollouts)
	}
}
