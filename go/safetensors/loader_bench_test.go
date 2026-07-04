// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"strconv"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// benchTensors builds n small tensors named like a real gemma4 shard's weight keys
// (model.layers.<i>.<part>.weight). Small Data per tensor keeps the focus on the
// PER-TENSOR header-decode cost (the map-of-any boxing) rather than byte movement.
func benchTensors(n int) map[string]Tensor {
	m := make(map[string]Tensor, n)
	parts := []string{
		"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
		"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
		"input_layernorm", "post_attention_layernorm",
	}
	for i := 0; i < n; i++ {
		name := "model.layers." + strconv.Itoa(i/len(parts)) + "." + parts[i%len(parts)] + ".weight"
		// 64 bf16 elements = 128 bytes; shape [8,8] so ∏shape×dtype validates.
		data := make([]byte, 128)
		for j := range data {
			data[j] = byte((i + j) & 0xff)
		}
		m[name] = Tensor{Dtype: "BF16", Shape: []int{8, 8}, Data: data}
	}
	return m
}

// benchBlob encodes n synthetic tensors to a single safetensors blob.
func benchBlob(tb testing.TB, n int) []byte {
	tb.Helper()
	blob, err := Encode(benchTensors(n))
	if err != nil {
		tb.Fatalf("Encode: %v", err)
	}
	return blob
}

// BenchmarkParse isolates the header decode + per-tensor validation from any file I/O —
// the allocator under test is the JSON header unmarshal (the map-of-any boxing) plus the
// per-tensor shape/offset slices. 256 tensors ≈ a real gemma4 shard's tensor count, so the
// any-boxing of every dtype/shape/offset shows up in B/op. AX-11 synthetic — no model load.
func BenchmarkParse(b *testing.B) {
	blob := benchBlob(b, 256)
	b.ReportAllocs()
	b.SetBytes(int64(len(blob)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ts, err := Parse(blob)
		if err != nil {
			b.Fatal(err)
		}
		if len(ts) != 256 {
			b.Fatalf("got %d tensors, want 256", len(ts))
		}
	}
}

// BenchmarkParseSmall is the few-tensor case (a norm-only or tiny shard) — the per-call
// fixed cost (header map allocation, sort-free path) without the boxing dominating.
func BenchmarkParseSmall(b *testing.B) {
	blob := benchBlob(b, 8)
	b.ReportAllocs()
	b.SetBytes(int64(len(blob)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Parse(blob); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncode is the inverse — header marshal + the per-tensor data concat. Surfaces the
// append-grown data buffer (whole-checkpoint byte movement) and the header map build.
func BenchmarkEncode(b *testing.B) {
	tensors := benchTensors(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Encode(tensors); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkLoad exercises the on-disk single-file path: coreio read + Parse. Includes the
// whole-file read alloc (the string→[]byte conversion the task flags). Temp file built once,
// outside the timed loop. AX-11 synthetic — a small fixture, not a real multi-GB model.
func BenchmarkLoad(b *testing.B) {
	blob := benchBlob(b, 256)
	dir := b.TempDir()
	path := core.PathJoin(dir, "m.safetensors")
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(blob)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ts, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		if len(ts) != 256 {
			b.Fatalf("got %d tensors, want 256", len(ts))
		}
	}
}

// BenchmarkLoadMmap is the zero-copy path: mmap + Parse + Close per iteration. The weights are
// never copied (Tensor.Data views the mmap), so allocs/op here should be the header decode
// only — the loader's whole reason to exist. AX-11 synthetic.
func BenchmarkLoadMmap(b *testing.B) {
	blob := benchBlob(b, 256)
	dir := b.TempDir()
	path := core.PathJoin(dir, "m.safetensors")
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(blob)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := LoadMmap(path)
		if err != nil {
			b.Fatal(err)
		}
		if len(m.Tensors) != 256 {
			b.Fatalf("got %d tensors, want 256", len(m.Tensors))
		}
		if err := m.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkLoadDir exercises the sharded merge (heap path): index parse + per-shard Load +
// merge. Two shards, each parsed once and cached. Surfaces the index-JSON copy and the
// per-shard whole-file read. AX-11 synthetic.
func BenchmarkLoadDir(b *testing.B) {
	dir := benchShardDir(b, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ts, err := LoadDir(dir)
		if err != nil {
			b.Fatal(err)
		}
		if len(ts) != 256 {
			b.Fatalf("got %d tensors, want 256", len(ts))
		}
	}
}

// BenchmarkLoadDirMmap is the sharded zero-copy path: index parse + per-shard mmap + merge +
// Close. The shard-merge map and the index decode are the allocators; the weights are views.
// AX-11 synthetic.
func BenchmarkLoadDirMmap(b *testing.B) {
	dir := benchShardDir(b, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dm, err := LoadDirMmap(dir)
		if err != nil {
			b.Fatal(err)
		}
		if len(dm.Tensors) != 256 {
			b.Fatalf("got %d tensors, want 256", len(dm.Tensors))
		}
		if err := dm.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

// benchShardDir writes a two-shard checkpoint (split in half) + index.json into a temp dir.
func benchShardDir(tb testing.TB, n int) string {
	tb.Helper()
	tensors := benchTensors(n)
	names := make([]string, 0, n)
	for name := range tensors {
		names = append(names, name)
	}
	half := len(names) / 2
	s1 := make(map[string]Tensor, half)
	s2 := make(map[string]Tensor, len(names)-half)
	weightMap := make(map[string]string, n)
	const f1, f2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	for i, name := range names {
		if i < half {
			s1[name] = tensors[name]
			weightMap[name] = f1
		} else {
			s2[name] = tensors[name]
			weightMap[name] = f2
		}
	}
	dir := tb.TempDir()
	for f, m := range map[string]map[string]Tensor{f1: s1, f2: s2} {
		blob, err := Encode(m)
		if err != nil {
			tb.Fatalf("Encode shard %s: %v", f, err)
		}
		if err := coreio.Local.Write(core.PathJoin(dir, f), string(blob)); err != nil {
			tb.Fatalf("write shard %s: %v", f, err)
		}
	}
	idx := core.JSONMarshalString(shardIndex{WeightMap: weightMap})
	if err := coreio.Local.Write(core.PathJoin(dir, indexName), idx); err != nil {
		tb.Fatalf("write index: %v", err)
	}
	return dir
}
