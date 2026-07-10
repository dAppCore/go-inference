// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// inference_register.go re-expresses go-mlx's register_native.go registration
// glue (which stays in go-mlx and dies with pkg/metal) against engine/metal's
// own loader. Importing this package self-registers the no-cgo Apple-GPU engine
// as inference backend "metal" — so serving.NewMLXBackend (WithBackend("metal"))
// and state/session.Session resolve a real model from go-inference alone, no
// go-mlx composition root. The registration is a plain init(): the concrete
// runtime package registers "metal", exactly as serving/backend_mlx.go documents.
package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
)

func init() { inference.Register(metalBackend{}) }

// metalBackend is the inference.Backend for the no-cgo Metal engine. Name is the
// stable "metal" selector; Available reports whether the Metal device + kernels
// initialise (ensureInit); LoadModel loads a checkpoint directory as an
// inference.TextModel through the reactive native loader + tokenizer.
type metalBackend struct{}

var _ inference.Backend = metalBackend{}

// Name is the registration/selection identifier.
func (metalBackend) Name() string { return "metal" }

// Available reports whether the Metal device and the compiled kernel library
// initialise on this host — the same gate the engine's own runtime tests use.
// Returns false (rather than panicking) on non-Apple hardware or a missing
// metallib, so inference.LoadModel fails cleanly with "not available".
func (metalBackend) Available() bool { return ensureInit() == nil }

// LoadModel reads the checkpoint directory at path and returns a ready
// inference.TextModel: the reactive native token model (dense / MoE / PLE, bf16
// or 4-bit) with the directory's tokenizer attached. WithContextLen sizes the
// KV cache (default 4096).
func (metalBackend) LoadModel(path string, opts ...inference.LoadOption) core.Result {
	cfg := inference.ApplyLoadOpts(opts)
	// maxLen <= 0 defers to the loader's checkpoint-window default
	// (resolveDefaultContext — the trained window capped at 32768).
	maxLen := cfg.ContextLen
	tm, err := LoadTokenModelDirWithConfig(path, maxLen, TokenModelLoadConfig{AdapterPath: cfg.AdapterPath})
	if err != nil {
		return core.Fail(core.E("native.metalBackend.LoadModel", "load token model", err))
	}
	ntm, ok := tm.(*NativeTokenModel)
	if !ok {
		if closer, closeOK := tm.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return core.Fail(core.E("native.metalBackend.LoadModel", "loader did not return a NativeTokenModel", nil))
	}
	tok, terr := tokenizer.LoadTokenizer(core.PathJoin(path, "tokenizer.json"))
	if terr != nil {
		_ = ntm.Close()
		return core.Fail(core.E("native.metalBackend.LoadModel", "load tokenizer", terr))
	}
	ntm.AttachTokenizer(tok)
	ntm.declaredStops = loadGenerationConfigStops(path)
	return core.Ok(newNativeTextModel(ntm, "gemma4"))
}
