// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gguf"
	"dappco.re/go/inference/model/safetensors"
	"dappco.re/go/inference/decode/tokenizer"
	coreio "dappco.re/go/io"
)

// ResolveAssistantGGUFDrafterFile reports whether path is a GGUF
// assistant drafter source: either a .gguf file directly or a directory with
// exactly one .gguf file. Ambiguous directories stand down.
func ResolveAssistantGGUFDrafterFile(path string) (string, bool) {
	if path == "" {
		return "", false
	}
	if nativeHasGGUFSuffix(path) {
		if _, err := coreio.Local.Stat(path); err != nil {
			return "", false
		}
		return path, true
	}
	entries, err := coreio.Local.List(path)
	if err != nil {
		return "", false
	}
	var matches []string
	for _, entry := range entries {
		if nativeHasGGUFSuffix(entry.Name()) {
			matches = append(matches, core.PathJoin(path, entry.Name()))
		}
	}
	if len(matches) != 1 {
		return "", false
	}
	return matches[0], true
}

func nativeHasGGUFSuffix(path string) bool {
	return core.HasSuffix(core.Lower(path), ".gguf")
}

// loadNativeAssistantFromGGUF loads a single-file GGUF drafter export through the
// reactive assistant registry: general.architecture picks the registered model package's
// spec (model.RegisterAssistant), whose weight-name map and metadata parser turn the GGUF
// into the same neutral config + canonical tensor names the safetensors path produces —
// the engine itself knows nothing about any drafter's format.
func loadNativeAssistantFromGGUF(file string, tok *tokenizer.Tokenizer) (*AssistantModel, error) {
	if tok == nil {
		return nil, core.E("native.assistant.gguf", "target tokenizer required", nil)
	}
	meta, err := gguf.Metadata(file)
	if err != nil {
		return nil, core.E("native.assistant.gguf", "read gguf metadata", err)
	}
	arch, _ := meta["general.architecture"].(string)
	spec, ok := model.LookupAssistantGGUF(arch)
	if !ok {
		return nil, core.E("native.assistant.gguf", "no registered assistant spec for gguf architecture "+arch, nil)
	}
	raw, err := gguf.LoadTensors(file)
	if err != nil {
		return nil, core.E("native.assistant.gguf", "load gguf tensors", err)
	}
	m, err := buildNativeAssistantFromGGUFTensors(spec, meta, raw, tok)
	if err != nil {
		_ = raw.Close()
		return nil, err
	}
	return m, nil
}

func buildNativeAssistantFromGGUFTensors(spec model.AssistantSpec, meta map[string]any, raw *gguf.TensorMapping, tok *tokenizer.Tokenizer) (*AssistantModel, error) {
	if raw == nil {
		return nil, core.NewError("native.assistant.gguf tensor map is nil")
	}
	weights := make(map[string]safetensors.Tensor, len(raw.Tensors))
	for name, tensor := range raw.Tensors {
		mapped := spec.GGUFWeightName(name)
		if mapped == "" {
			continue
		}
		weights[mapped] = tensor
	}
	// exports may omit vocab_size — the embed tensor's leading dim is the hint.
	vocabHint := 0
	if embed, ok := weights["model.embed_tokens.weight"]; ok && len(embed.Shape) > 0 {
		vocabHint = embed.Shape[0]
	}
	cfg, err := spec.ParseGGUF(meta, vocabHint)
	if err != nil {
		return nil, err
	}
	m := &AssistantModel{
		Config:                   cfg,
		Arch:                     cfg.Arch,
		Tensors:                  weights,
		BackboneHiddenSize:       cfg.BackboneHidden,
		NumCentroids:             cfg.NumCentroids,
		CentroidIntermediateTopK: cfg.CentroidTopK,
		UseOrderedEmbeddings:     cfg.OrderedEmbeddings,
		Tok:                      tok,
		gguf:                     raw,
	}
	if err := validateNativeAssistantModel(m); err != nil {
		_ = m.Close()
		return nil, core.E("native.assistant.gguf", "validate tensors", err)
	}
	return m, nil
}
