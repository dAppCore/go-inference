// SPDX-Licence-Identifier: EUPL-1.2

package merge

import "testing"

// The merge benches baseline the pure string/scalar helpers of the merge orchestration
// (AX-11): equalFold / containsFold / hasSuffixFold are the allocation-free ASCII case-fold
// comparisons the pack-compatibility + weight-file validation run per source (avoiding a
// core.Lower allocation), and clampFloat64 is the scalar clamp the SLERP + cosine paths lean
// on. All expected zero-alloc. (Packs / prepare / indexSources need real safetensors files —
// benched via the merge path, not here.)

// BenchmarkEqualFold — the allocation-free ASCII case-insensitive equality the format checks use.
func BenchmarkEqualFold(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !equalFold("Model.SafeTensors", "model.safetensors") {
			b.Fatal("fold mismatch")
		}
	}
}

// BenchmarkContainsFold — the case-insensitive substring scan the pack-path checks run.
func BenchmarkContainsFold(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !containsFold("/models/Gemma-4-Base/model-00001.SAFETENSORS", "safetensors") {
			b.Fatal("substring not found")
		}
	}
}

// BenchmarkHasSuffixFold — the case-insensitive suffix check the weight-file validation runs
// per path.
func BenchmarkHasSuffixFold(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !hasSuffixFold("model-00004-of-00004.SafeTensors", ".safetensors") {
			b.Fatal("suffix not matched")
		}
	}
}

// BenchmarkClampFloat64 — the scalar clamp on the SLERP/cosine numerical-stability paths, no
// allocation.
func BenchmarkClampFloat64(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = clampFloat64(1.0001, -1, 1)
	}
}
