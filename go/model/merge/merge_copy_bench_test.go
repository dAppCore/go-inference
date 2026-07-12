// SPDX-Licence-Identifier: EUPL-1.2

package merge

import "testing"

// The merge-copy benches baseline the path-identity checks (AX-11) the merge runs to guard a
// source==destination overwrite: SamePath resolves both paths to absolute and compares;
// SamePathResolved compares a path against an already-absolute one (skipping one resolve).
// Both are pure path-string work, run once per merge. (HashFile / copyModelPackMetadata are
// file I/O — not benched here.)

// BenchmarkSamePath — resolve both paths to absolute and compare: the guard the merge runs
// before writing, so it never overwrites a source pack.
func BenchmarkSamePath(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SamePath("/models/base-pack", "/models/merged-out")
	}
}

// BenchmarkSamePathResolved — compare a path against an already-absolute one: one resolve
// instead of two, the cheaper guard when the destination is pre-resolved.
func BenchmarkSamePathResolved(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SamePathResolved("/models/base-pack", "/models/merged-out")
	}
}
