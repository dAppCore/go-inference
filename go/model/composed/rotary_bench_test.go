// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "testing"

// BenchmarkMRoPEInterleavedFreqs baselines the interleaved mRoPE angle construction over the real Qwen 3.6
// geometry (rotary_dim 64 ⇒ 32 pairs, section [11,11,10]) — the per-position work a 3D-position (vision)
// decode would pay. The single make([]float64, 32) is the allocation story.
func BenchmarkMRoPEInterleavedFreqs(b *testing.B) {
	invFreq := rotaryInvFreq(64, 1e7)
	section := [3]int{11, 11, 10}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mRoPEInterleavedFreqs(float64(i), float64(i), float64(i), invFreq, section)
	}
}
