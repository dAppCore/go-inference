// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// The compare benches baseline the per-tensor delta computation (AX-11), run once per tensor
// when comparing a base vs a fine-tuned pack: compareTensorEntries now streams each tensor's
// payload from its shard file on demand (via a ShardCache, ReadAt-backed — see merge_tensors.go)
// before running the stats scan (mean-abs / RMS / max-abs / L2 / cosine) in one f64 pass;
// compareCosine is the cosine close from the accumulated dot + norms. The per-tensor read +
// decode + the single scan are the cost. Sized to a realistic weight tensor. The shard file is
// written once in TestMain-adjacent setup (b.TempDir, outside the timed loop) and reopened via
// a cache shared across all b.N iterations — the same handle-reuse shape a real pack comparison
// gets from ComparePacks' shared cache, so the benchmark measures the real per-tensor cost
// (syscall included), not a synthetic in-memory-only number.

func benchCompareEntry(b *testing.B, path, name string, elems, seed int) tensorEntry {
	b.Helper()
	vals := make([]float32, elems)
	for i := range vals {
		vals[i] = float32((i*seed)%4096-2048) * 0.001
	}
	info := map[string]safetensors.SafetensorsTensorInfo{name: {Dtype: "F32", Shape: []int{1024, elems / 1024}}}
	data := map[string][]byte{name: safetensors.EncodeFloat32(vals)}
	if result := safetensors.WriteSafetensors(path, info, data); !result.OK {
		b.Fatalf("write bench shard: %v", result.Value)
	}
	idx, err := indexWeightFiles([]string{path})
	if err != nil {
		b.Fatalf("index bench shard: %v", err)
	}
	return idx.Tensors[name]
}

// BenchmarkCompareTensorEntries — read + decode base + tuned (1M elements each, from their own
// shard file) then the single-pass delta scan: the two ReadAt+decode calls + the O(elements)
// f64 stats loop. The per-tensor cost of a pack comparison.
func BenchmarkCompareTensorEntries(b *testing.B) {
	const elems = 1 << 20
	dir := b.TempDir()
	base := benchCompareEntry(b, core.PathJoin(dir, "base.safetensors"), "base", elems, 131)
	tuned := benchCompareEntry(b, core.PathJoin(dir, "tuned.safetensors"), "tuned", elems, 137)
	cache := safetensors.NewShardCache()
	defer cache.Close()
	b.SetBytes(int64(elems * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := compareTensorEntries(cache, "model.layer.0.weight", base, tuned); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCompareCosine — the cosine close from accumulated dot + norms: a couple of sqrts
// and a clamp, no allocation. The per-tensor scalar finish.
func BenchmarkCompareCosine(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = compareCosine(0.87, 1.2, 1.3)
	}
}
