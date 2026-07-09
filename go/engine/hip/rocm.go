// Package hip provides the AMD ROCm backend for the Core Go inference stack
// (quarantined into go-mlx as the Tier-4 pkg/hip engine; upstream source is
// dappco.re/go/rocm).
//
// The default linux/amd64 build is native-first: it registers the ROCm backend
// through go-inference, exposes model-fit planning, probing, benchmarking,
// evaluation, tokenizer, and adapter contracts, and avoids the previous
// OpenAI-compatible llama-server subprocess path.
//
// The native HIP loader is intentionally explicit. Until it is linked in,
// Available reports false instead of hiding behind a server fallback. The old
// llama-server subprocess bridge (rocm_legacy_server build tag upstream) was
// not carried into this quarantine — it depended on internal/llamacpp, which
// this landing pass deliberately left behind (see the landing commit body).
//
// # Quick Start
//
//	import (
//	    "dappco.re/go/inference"
//	    _ "dappco.re/go/inference/engine/hip" // auto-registers ROCm backend
//	)
//
//	m, err := inference.LoadModel("/path/to/model.gguf")
//	defer m.Close()
//	for tok := range m.Generate(ctx, "Hello", inference.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
//
// # Requirements
//
//   - Linux (amd64) for the ROCm runtime build
//   - AMD GPU with ROCm support (RDNA 2+ / gfx10xx+ target class)
//   - ROCm/HIP runtime for the forthcoming native loader
package hip

import (
	core "dappco.re/go"
)

// VRAMInfo reports GPU video memory usage in bytes.
type VRAMInfo struct {
	Total uint64
	Used  uint64
	Free  uint64
}

// ModelInfo describes a GGUF model file discovered on disk.
type ModelInfo struct {
	Path         string // full path to .gguf file
	Architecture string // GGUF architecture (e.g. "gemma3", "llama", "qwen2")
	Name         string // human-readable model name from GGUF metadata
	Quantisation string // quantisation level (e.g. "Q4_K_M", "Q8_0")
	Parameters   string // parameter size label (e.g. "1B", "8B")
	FileSize     int64  // file size in bytes
	ContextLen   uint32 // native context window length
}

type rocmFailure interface {
	Error() string
}

// errHIPResultFailed is the fallback resultError returns when a failed
// core.Result carries no error value. Mirrors the per-package resultError
// helpers used across go-mlx (see native_speculative_textmodel.go's
// errCoreResultFailed in the mlx package) after inference.Backend.LoadModel's
// migration to core.Result — a go-inference contract change discovered
// while landing this quarantine (go-rocm's source at 308c4d6 predates it;
// see the landing commit body). Lives in this untagged file rather than
// native.go because both the native (linux&&amd64) and stub (everywhere
// else) rocmBackend variants — and their respective tests — need it, and
// the two are mutually exclusive by build tag.
var errHIPResultFailed = core.NewError("rocm: core.Result reported failure without an error value")

// resultError unwraps a core.Result into a plain error — nil when OK, the
// unwrapped underlying error (identity preserved for core.Is / errors.Is)
// when failed, falling back to errHIPResultFailed when a failed Result
// carries no error value.
func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errHIPResultFailed
}
