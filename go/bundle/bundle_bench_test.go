// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for bundle assembly + save/load + SAMI conversion.
// Per AX-11 — bundle.New runs once per "save session state" call;
// Save/Load happen per host-to-host migration. SAMIFromKV fires on
// every New (the visualisation-friendly summary) and is the inner
// loop dashboards land on. Normalisation helpers fire per Save.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/bundle

package bundle

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/train/lora"
	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	bundleSinkBundle    *Bundle
	bundleSinkErr       error
	bundleSinkString    string
	bundleSinkTokenizer Tokenizer
	bundleSinkAdapter   Adapter
	bundleSinkSAMI      SAMIResult
	bundleSinkAInfo     lora.AdapterInfo
)

// benchBundleSnapshot builds a representative kv.Snapshot — token
// count and layer/head shape sized to the qwen3-class range.
func benchBundleSnapshot(tokenCount, numLayers int) *kv.Snapshot {
	tokens := make([]int32, tokenCount)
	headKey := make([]float32, tokenCount)
	headValue := make([]float32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		headKey[i] = float32(i)
		headValue[i] = float32(i + 1000)
	}
	layers := make([]kv.LayerSnapshot, numLayers)
	for i := range layers {
		layers[i] = kv.LayerSnapshot{
			Layer:      i,
			CacheIndex: i,
			Heads:      []kv.HeadSnapshot{{Key: headKey, Value: headValue}},
		}
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     numLayers,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers:        layers,
	}
}

// --- New — bundle assembly hot path ---

func BenchmarkBundle_New_Small(b *testing.B) {
	snap := benchBundleSnapshot(64, 2)
	opts := Options{
		Model:     "qwen3-0.6b",
		ModelPath: "/models/qwen3",
		Source: ModelInfo{
			Architecture: "qwen3", NumLayers: 2,
			VocabSize: 100, QuantBits: 4,
		},
		Prompt:  "hello",
		Sampler: Sampler{MaxTokens: 32, Temperature: 0.2},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkBundle, bundleSinkErr = New(snap, opts)
	}
}

func BenchmarkBundle_New_Typical(b *testing.B) {
	snap := benchBundleSnapshot(2048, 28)
	opts := Options{
		Model:     "qwen3-0.6b",
		ModelPath: "/models/qwen3",
		Source: ModelInfo{
			Architecture: "qwen3", NumLayers: 28,
			VocabSize: 1000, QuantBits: 4, ContextLength: 40960,
		},
		Prompt:  "trace me",
		Sampler: Sampler{MaxTokens: 64, Temperature: 0.7},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkBundle, bundleSinkErr = New(snap, opts)
	}
}

// --- Save / Load roundtrip ---

func BenchmarkBundle_Save_Typical(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 8}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.Save(core.JoinPath(dir, "state.bundle.json"))
	}
}

// SaveCompact — newlineless variant for cold storage. Time delta vs Save
// is small (one fewer per-element whitespace write); the win is on-disk
// size (~75% smaller on typical bundles). See parity test for the live
// disk-size assertion.
func BenchmarkBundle_SaveCompact_Typical(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 8}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.SaveCompact(core.JoinPath(dir, "state.bundle.json"))
	}
}

// SaveCompact_Small — under 256 bytes of metadata. Whitespace ratio is
// lower here, so the disk-size delta narrows; useful as a floor.
func BenchmarkBundle_SaveCompact_Small(b *testing.B) {
	snap := benchBundleSnapshot(64, 2)
	bundle, err := New(snap, Options{Model: "qwen3-0.6b", Source: ModelInfo{Architecture: "qwen3", NumLayers: 2}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.SaveCompact(core.JoinPath(dir, "state.bundle.json"))
	}
}

// SaveCompact_Large — qwen3-class shape (2048 tokens × 28 layers).
// Largest whitespace surface; expect the strongest size reduction.
func BenchmarkBundle_SaveCompact_Large(b *testing.B) {
	snap := benchBundleSnapshot(2048, 28)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 28}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.SaveCompact(core.JoinPath(dir, "state.bundle.json"))
	}
}

// Save_Small / Save_Large — sibling Save coverage so the bench output
// shows the indented-vs-compact delta at each shape (Small / Typical
// already lives above / Large).
func BenchmarkBundle_Save_Small(b *testing.B) {
	snap := benchBundleSnapshot(64, 2)
	bundle, err := New(snap, Options{Model: "qwen3-0.6b", Source: ModelInfo{Architecture: "qwen3", NumLayers: 2}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.Save(core.JoinPath(dir, "state.bundle.json"))
	}
}

func BenchmarkBundle_Save_Large(b *testing.B) {
	snap := benchBundleSnapshot(2048, 28)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 28}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.Save(core.JoinPath(dir, "state.bundle.json"))
	}
}

func BenchmarkBundle_Load_Typical(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 8}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	path := core.JoinPath(b.TempDir(), "state.bundle.json")
	if err := bundle.Save(path); err != nil {
		b.Fatalf("Save: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkBundle, bundleSinkErr = Load(path)
	}
}

// --- Validate ---

func BenchmarkBundle_Validate(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 8}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkErr = bundle.Validate()
	}
}

// --- HashString — fires per bundle field that needs a hash ---

func BenchmarkBundle_HashString_Short(b *testing.B) {
	value := "qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString = HashString(value)
	}
}

func BenchmarkBundle_HashString_Long(b *testing.B) {
	value := "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\nhello<end_of_turn>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString = HashString(value)
	}
}

func BenchmarkBundle_HashString_Empty(b *testing.B) {
	value := ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString = HashString(value)
	}
}

// --- NormaliseTokenizer / AdapterFromInfo / AdapterToInfo ---

func BenchmarkBundle_NormaliseTokenizer(b *testing.B) {
	tokenizer := Tokenizer{
		Kind:         "hf-tokenizer-json",
		Path:         "/models/qwen3/tokenizer.json",
		ChatTemplate: "<start_of_turn>model\n",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkTokenizer = NormaliseTokenizer(tokenizer)
	}
}

func BenchmarkBundle_AdapterFromInfo(b *testing.B) {
	info := lora.AdapterInfo{
		Name: "domain-lora", Path: "/adapters/domain", Hash: "abc",
		Rank: 8, Alpha: 16, Scale: 2,
		TargetKeys: []string{"q_proj", "v_proj", "k_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkAdapter = AdapterFromInfo(info)
	}
}

func BenchmarkBundle_AdapterToInfo(b *testing.B) {
	adapter := Adapter{
		Name: "domain-lora", Path: "/adapters/domain", Hash: "abc",
		Rank: 8, Alpha: 16, Scale: 2,
		TargetKeys: []string{"q_proj", "v_proj", "k_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkAInfo = AdapterToInfo(adapter)
	}
}

func BenchmarkBundle_AdapterEmpty(b *testing.B) {
	adapter := Adapter{
		Name: "domain-lora", Path: "/adapters/domain",
		Rank: 8, Alpha: 16,
	}
	var sink bool
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = AdapterEmpty(adapter)
	}
	_ = sink
}

// --- FileHash — content-hash of an on-disk file (e.g. tokenizer.json) ---

func BenchmarkBundle_FileHash_1KB(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "file.bin")
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i)
	}
	if r := core.WriteFile(path, data, 0o644); !r.OK {
		b.Fatalf("WriteFile: %v", r.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString, bundleSinkErr = FileHash(path)
	}
}

func BenchmarkBundle_FileHash_64KB(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "file.bin")
	data := make([]byte, 64*1024)
	for i := range data {
		data[i] = byte(i)
	}
	if r := core.WriteFile(path, data, 0o644); !r.OK {
		b.Fatalf("WriteFile: %v", r.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString, bundleSinkErr = FileHash(path)
	}
}

// 1MB — representative tokenizer.json (tokenizer + chat-template + merges).
func BenchmarkBundle_FileHash_1MB(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "file.bin")
	data := make([]byte, 1024*1024)
	for i := range data {
		data[i] = byte(i)
	}
	if r := core.WriteFile(path, data, 0o644); !r.OK {
		b.Fatalf("WriteFile: %v", r.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString, bundleSinkErr = FileHash(path)
	}
}

// 10MB — representative LoRA adapter shard / large vocab tokenizer.
// (100MB scale gated behind the 1MB bench because hash bandwidth is
// linear past this point — alloc-side win flattens by 1MB.)
func BenchmarkBundle_FileHash_10MB(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "file.bin")
	data := make([]byte, 10*1024*1024)
	for i := range data {
		data[i] = byte(i)
	}
	if r := core.WriteFile(path, data, 0o644); !r.OK {
		b.Fatalf("WriteFile: %v", r.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString, bundleSinkErr = FileHash(path)
	}
}

// --- SAMIFromKV — visualisation summary, runs per New + per dashboard tick ---

func BenchmarkBundle_SAMIFromKV_512Tokens(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	opts := SAMIOptions{Model: "qwen3", Prompt: "trace"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSAMI = SAMIFromKV(snap, nil, opts)
	}
}

func BenchmarkBundle_SAMIFromKV_2048Tokens(b *testing.B) {
	snap := benchBundleSnapshot(2048, 28)
	opts := SAMIOptions{Model: "qwen3", Prompt: "trace"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSAMI = SAMIFromKV(snap, nil, opts)
	}
}

func BenchmarkBundle_SAMIFromKV_PrecomputedAnalysis_2048(b *testing.B) {
	snap := benchBundleSnapshot(2048, 28)
	analysis := kv.Analyze(snap)
	opts := SAMIOptions{Model: "qwen3", Prompt: "trace"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSAMI = SAMIFromKV(snap, analysis, opts)
	}
}

// --- StateURI / MemvidURI — fires per ref on bundle build ---

func BenchmarkBundle_StateURI_WithSegment(b *testing.B) {
	ref := state.ChunkRef{Segment: "/tmp/trace.mp4", ChunkID: 42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString = StateURI(ref)
	}
}

func BenchmarkBundle_StateURI_NoSegment(b *testing.B) {
	ref := state.ChunkRef{ChunkID: 42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString = StateURI(ref)
	}
}

func BenchmarkBundle_MemvidURI_WithSegment(b *testing.B) {
	ref := state.ChunkRef{Segment: "/tmp/trace.mp4", ChunkID: 42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkString = MemvidURI(ref)
	}
}
