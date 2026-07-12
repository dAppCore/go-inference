// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

func TestGGUF_ReadGGUFInfo_Good(t *testing.T) {
	path := writeMinimalGGUF(t, map[string]any{
		"general.architecture": "qwen3",
		"general.file_type":    uint32(15),
		"qwen3.block_count":    uint32(28),
		"qwen3.context_length": uint32(40960),
	})

	info, err := ReadGGUFInfo(path)

	checkNoError(t, err)
	checkEqual(t, "qwen3", info.Architecture)
	checkEqual(t, 4, info.QuantBits)
	checkEqual(t, 28, info.NumLayers)
	checkEqual(t, 40960, info.ContextLength)
}

func TestGGUF_ReadGGUFInfo_Bad(t *testing.T) {
	info, err := ReadGGUFInfo(core.JoinPath(t.TempDir(), "missing.gguf"))

	checkError(t, err)
	checkEqual(t, GGUFInfo{}, info)
}

func TestGGUF_DiscoverModels_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := writeMinimalGGUFAt(t, core.JoinPath(dir, "model.gguf"), map[string]any{
		"general.architecture": "gemma4_text",
		"general.file_type":    uint32(7),
	})

	models := DiscoverModels(dir)

	checkLen(t, models, 1)
	checkEqual(t, path, models[0].Path)
	checkEqual(t, "gemma4_text", models[0].ModelType)
	checkEqual(t, "gguf", models[0].Format)
}

func writeMinimalGGUF(t *testing.T, metadata map[string]any) string {
	t.Helper()
	return writeMinimalGGUFAt(t, core.JoinPath(t.TempDir(), "model.gguf"), metadata)
}

func writeMinimalGGUFAt(t *testing.T, path string, metadata map[string]any) string {
	t.Helper()
	buf := core.NewBuffer()
	mustWrite := func(value any) {
		checkNoError(t, binary.Write(buf, binary.LittleEndian, value))
	}
	writeString := func(value string) {
		mustWrite(uint64(len(value)))
		_, err := buf.Write([]byte(value))
		checkNoError(t, err)
	}

	mustWrite(uint32(0x46554747))
	mustWrite(uint32(3))
	mustWrite(uint64(0))
	mustWrite(uint64(len(metadata)))
	for key, value := range metadata {
		writeString(key)
		switch typed := value.(type) {
		case string:
			mustWrite(uint32(8))
			writeString(typed)
		case uint32:
			mustWrite(uint32(4))
			mustWrite(typed)
		default:
			t.Fatalf("unsupported metadata test value %T", value)
		}
	}
	result := core.WriteFile(path, buf.Bytes(), 0o644)
	checkResultOK(t, result)
	return path
}

// AX-11: alloc + behavioural lock for ReadGGUFInfo on a vocab-heavy
// header. Mirrors BenchmarkGGUF_ReadInfo_VocabHeavy's fixture shape
// (5 real fields + 200 synthetic noise entries) so this gate catches
// the same regressions the bench would surface, except mechanically
// in `go test`.
//
// Baselines (Apple M3 Ultra, -benchmem):
//
//	pre-bufio  (per-entry syscalls): 22 allocs / ~437µs
//	post-bufio (one buffer fill):    23 allocs / ~23µs   ← current
//
// Alloc +1 is from bufio.Reader's internal buffer allocation; time
// drops 18.7x because skipGGUFValue serves from buffered bytes
// instead of one syscall per entry skipped. Net trade is clear: model
// load is one-shot, not per-token.
//
// Twin assertions:
//  1. ALLOCS — stays below ceiling (regression gate)
//  2. OUTPUT — the parsed GGUFInfo matches expected values (behaviour gate)
//
// The output assertion is the TDD anchor — any refactor that produces
// a different GGUFInfo for the same fixture fails loud BEFORE the
// downstream backends (go-mlx, go-rocm) try to load the model and
// see "context_length=0".
func TestGGUF_AllocBudget_ReadInfo_VocabHeavy(t *testing.T) {
	metadata := map[string]any{
		"general.architecture":   "qwen3",
		"general.file_type":      uint32(15),
		"qwen3.block_count":      uint32(28),
		"qwen3.context_length":   uint32(40960),
		"qwen3.embedding_length": uint32(2048),
	}
	for i := range 200 {
		metadata[core.Sprintf("synthetic.meta.%d", i)] = core.Sprintf("value-payload-%d", i)
	}
	path := writeMinimalGGUF(t, metadata)

	// Behavioural lock — output for this fixture is the contract every
	// optimisation must preserve.
	info, err := ReadGGUFInfo(path)
	checkNoError(t, err)
	checkEqual(t, "qwen3", info.Architecture)
	checkEqual(t, 28, info.NumLayers)
	checkEqual(t, 40960, info.ContextLength)
	checkEqual(t, 2048, info.HiddenSize)
	checkEqual(t, 4, info.QuantBits)
	checkEqual(t, "q4_k_m", info.QuantType)

	// Alloc-budget lock — set with deliberate headroom for stdlib drift.
	// Ratchet DOWN when wins land; bumping UP needs a documented reason.
	avg := testing.AllocsPerRun(5, func() {
		_, _ = ReadGGUFInfo(path)
	})
	const budget = 25.0 // current measured: 22
	if avg > budget {
		t.Fatalf("ReadGGUFInfo alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Vocab-heavy headers are model-load hot path — every backend pays this per Load.\n"+
			"Profile: go test -bench=BenchmarkGGUF_ReadInfo_VocabHeavy -benchmem -memprofile=/tmp/g.mem",
			avg, budget)
	}
}

// TestGGUF_Valid_Good pins the metadata-validity read across all three shapes:
// no issues is valid, warnings alone stay valid, and any error issue makes the
// info invalid — so a caller gates a load on Valid() without re-walking the list.
func TestGGUF_Valid_Good(t *testing.T) {
	if !(GGUFInfo{}).Valid() {
		t.Fatal("GGUFInfo with no issues should be valid")
	}
	warnOnly := GGUFInfo{ValidationIssues: []GGUFValidationIssue{{Severity: GGUFValidationWarning}}}
	if !warnOnly.Valid() {
		t.Fatal("warnings alone should not invalidate GGUF metadata")
	}
	withError := GGUFInfo{ValidationIssues: []GGUFValidationIssue{
		{Severity: GGUFValidationWarning},
		{Severity: GGUFValidationError},
	}}
	if withError.Valid() {
		t.Fatal("an error issue must make GGUF metadata invalid")
	}
}
