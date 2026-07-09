// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for LoRA adapter inspection + identity helpers.
// Per AX-11 — InspectAdapter fires per model load when a LoRA is
// attached (config parse + safetensors hashing), and IsEmpty fires
// per session state check. hashAdapter is the inner SHA-256 path
// that scales with adapter weight size + shard count.
//
// Run:    go test -bench='BenchmarkAdapter' -benchmem -run='^$' ./lora

package lora

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	loraAdapterBenchSinkInfo   AdapterInfo
	loraAdapterBenchSinkConfig AdapterConfig
	loraAdapterBenchSinkErr    error
	loraAdapterBenchSinkBool   bool
	loraAdapterBenchSinkString string
)

// writeBenchAdapter materialises a synthetic adapter directory with a
// config + a stub weight blob. Hash-side bench cost scales with the
// weight length — feeding small payloads keeps timing dominated by
// the parser, larger payloads exercise the SHA path.
//
//	dir := writeBenchAdapter(b, `{"rank":8,...}`, weightBytes)
func writeBenchAdapter(b *testing.B, config string, weightSize int) string {
	b.Helper()
	dir := b.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(config), 0o600); !result.OK {
		b.Fatalf("WriteFile adapter_config: %v", result.Value)
	}
	weights := make([]byte, weightSize)
	for i := range weights {
		weights[i] = byte(i)
	}
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), weights, 0o600); !result.OK {
		b.Fatalf("WriteFile adapter.safetensors: %v", result.Value)
	}
	return dir
}

// --- InspectAdapter — full path: read config + hash weights ---

func BenchmarkAdapter_InspectAdapter_SmallWeights(b *testing.B) {
	dir := writeBenchAdapter(b, `{"rank":8,"alpha":16,"lora_layers":["self_attn.q_proj","self_attn.v_proj"]}`, 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkInfo, loraAdapterBenchSinkErr = InspectAdapter(dir)
	}
}

func BenchmarkAdapter_InspectAdapter_TypicalWeights(b *testing.B) {
	// 256KiB weight stub — proxy for a small rank-8 adapter file. The
	// SHA-256 over the weight blob dominates timing once rank gets real.
	dir := writeBenchAdapter(b, `{"rank":8,"alpha":16,"lora_layers":["self_attn.q_proj","self_attn.v_proj","self_attn.k_proj","self_attn.o_proj"]}`, 256*1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkInfo, loraAdapterBenchSinkErr = InspectAdapter(dir)
	}
}

func BenchmarkAdapter_InspectAdapter_PEFTAliasesConfig(b *testing.B) {
	// PEFT-style config — exercises the firstNonZero* fallback chains
	// that pick between rank/r, alpha/lora_alpha, target_keys/target_modules.
	dir := writeBenchAdapter(b, `{"r":16,"lora_alpha":32,"target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}`, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkInfo, loraAdapterBenchSinkErr = InspectAdapter(dir)
	}
}

// --- Inspect — explicit identity path (used by staged adapters) ---

func BenchmarkAdapter_Inspect_StagedIdentity(b *testing.B) {
	dir := writeBenchAdapter(b, `{"rank":32,"alpha":64,"lora_layers":["q_proj","v_proj"]}`, 8192)
	stagedIdentity := "/agents/active/adapter"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkInfo, loraAdapterBenchSinkErr = Inspect(dir, stagedIdentity)
	}
}

// --- InspectAdapter (.safetensors file path) — exercises the
// adapterConfigPath branch where path points at a single safetensors
// file rather than a directory. ---

func BenchmarkAdapter_InspectAdapter_SafetensorsPath(b *testing.B) {
	dir := writeBenchAdapter(b, `{"rank":4,"alpha":8,"lora_layers":["q_proj"]}`, 4096)
	path := core.PathJoin(dir, "adapter.safetensors")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkInfo, loraAdapterBenchSinkErr = InspectAdapter(path)
	}
}

// --- AdapterInfo.IsEmpty — predicate hit on every session bootstrap ---

func BenchmarkAdapter_IsEmpty_Empty(b *testing.B) {
	info := AdapterInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkBool = info.IsEmpty()
	}
}

func BenchmarkAdapter_IsEmpty_Populated(b *testing.B) {
	info := AdapterInfo{
		Name:       "q-domain",
		Path:       "/adapters/q-domain",
		Hash:       "sha256:abcdef",
		Rank:       16,
		Alpha:      32,
		Scale:      2,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkBool = info.IsEmpty()
	}
}

// --- adapterConfigPath — branch on .safetensors suffix ---

func BenchmarkAdapter_AdapterConfigPath_Dir(b *testing.B) {
	path := "/adapters/q-domain"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkString = adapterConfigPath(path)
	}
}

func BenchmarkAdapter_AdapterConfigPath_Safetensors(b *testing.B) {
	path := "/adapters/q-domain/adapter.safetensors"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkString = adapterConfigPath(path)
	}
}

// --- shared adapter_config normalisation — alias/default hot path ---

func BenchmarkAdapter_NormalizeAdapterConfig_PEFTAliases(b *testing.B) {
	cfg := AdapterConfig{
		R:             16,
		LoRAAlpha:     32,
		TargetModules: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkConfig = NormalizeAdapterConfig(cfg)
	}
}

func BenchmarkAdapter_ParseAdapterConfig_TargetPrecedence(b *testing.B) {
	config := []byte(`{"rank":4,"scale":2,"target_keys":["explicit"],"target_modules":["peft"],"lora_layers":["mlx-lm"]}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkConfig, loraAdapterBenchSinkErr = ParseAdapterConfig(config)
	}
}

// --- hashAdapter — SHA-256 over config + sorted weight files.
// Cost scales with weight blob size; vary the payload to see the
// constant-factor vs payload-bytes split. ---

func BenchmarkAdapter_HashAdapter_SmallWeights(b *testing.B) {
	dir := writeBenchAdapter(b, `{"rank":8,"alpha":16}`, 1024)
	read := core.ReadFile(core.PathJoin(dir, "adapter_config.json"))
	if !read.OK {
		b.Fatalf("read config: %v", read.Value)
	}
	config := read.Value.([]byte)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkString = hashAdapter(dir, config)
	}
}

func BenchmarkAdapter_HashAdapter_TypicalWeights(b *testing.B) {
	dir := writeBenchAdapter(b, `{"rank":8,"alpha":16}`, 256*1024)
	read := core.ReadFile(core.PathJoin(dir, "adapter_config.json"))
	if !read.OK {
		b.Fatalf("read config: %v", read.Value)
	}
	config := read.Value.([]byte)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraAdapterBenchSinkString = hashAdapter(dir, config)
	}
}
