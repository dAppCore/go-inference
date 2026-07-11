// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// load.go is the engine's single REACTIVE loader: read a checkpoint dir, probe model_type, and react to
// the registered ArchSpec — parse, resolve dims from the weight shapes, derive the Arch, assemble. It
// replaces every per-architecture loader and lives in the backend-agnostic root, so native + go-rocm
// share ONE loader; a backend's LoadDir delegates here.

// Load reads dir's config.json + safetensors and returns the neutral LoadedModel plus the DirMapping
// whose mmap the weight byte-views reference (Close it once the device buffers are bound). It dispatches
// on model_type through the ArchSpec registry, so adding an architecture needs no edit here.
func Load(dir string) (*LoadedModel, *safetensors.DirMapping, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, nil, core.E("model.Load", "read config.json", err)
	}
	cfg := []byte(cfgStr)
	mt, textMT := probeModelTypes(cfg)
	spec, ok := LookupArch(mt)
	if !ok && textMT != "" { // multimodal wrapper: fall back to the nested text arch's model_type
		spec, ok = LookupArch(textMT)
	}
	if !ok {
		return nil, nil, core.NewError("model.Load: no architecture registered for model_type " + mt)
	}
	ac, err := spec.Parse(cfg)
	if err != nil {
		return nil, nil, err
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, nil, err
	}
	tensors := dm.Tensors
	if spec.Normalize != nil {
		tensors = spec.Normalize(tensors)
		dm.Tensors = tensors
	}
	ac.InferFromWeights(NormalizeWrapperNames(tensors)) // resolve omitted dims from the shapes (don't-guess)
	arch, err := ac.Arch()
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	m, err := Assemble(tensors, arch, spec.Weights)
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	if spec.Vision != nil {
		m.Vision, err = spec.Vision(tensors, ac)
		if err != nil {
			_ = dm.Close()
			return nil, nil, err
		}
	}
	if spec.UnifiedVision != nil {
		m.UnifiedVision, err = spec.UnifiedVision(tensors, ac)
		if err != nil {
			_ = dm.Close()
			return nil, nil, err
		}
	}
	if spec.Audio != nil {
		m.Audio, err = spec.Audio(tensors, ac)
		if err != nil {
			_ = dm.Close()
			return nil, nil, err
		}
	}
	if spec.Diffusion != nil {
		m.Diffusion, err = spec.Diffusion(tensors, ac)
		if err != nil {
			_ = dm.Close()
			return nil, nil, err
		}
	}
	return m, dm, nil
}

// ProbeDirArch reads dir/config.json and returns its top-level model_type plus the raw config bytes —
// the front-door check a backend uses to route a checkpoint whose loader is NOT the reactive Assemble
// path (a recurrent SSM like mamba2 carries its own loader; its weights have no attention to assemble).
// A registered transformer arch ignores this and goes straight through Load.
func ProbeDirArch(dir string) (modelType string, configJSON []byte, err error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return "", nil, core.E("model.ProbeDirArch", "read config.json", err)
	}
	mt, _ := probeModelTypes([]byte(cfgStr))
	return mt, []byte(cfgStr), nil
}

// ProbeDirContextWindow reads dir's config.json and returns the checkpoint's
// trained context window — the LARGER of the top-level and nested
// text_config max_position_embeddings (multimodal wrappers put the real
// window on the text config; the gemma4 parser applies the same
// prefer-the-larger rule). 0 when the config is unreadable or carries
// neither, so callers keep their own floor.
func ProbeDirContextWindow(dir string) int {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return 0
	}
	var probe struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
		TextConfig            struct {
			MaxPositionEmbeddings int `json:"max_position_embeddings"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal([]byte(cfgStr), &probe); !r.OK {
		return 0
	}
	return max(probe.MaxPositionEmbeddings, probe.TextConfig.MaxPositionEmbeddings)
}

// ProbeModelTypes returns config.json's top-level model_type and the nested text_config.model_type ids
// (a multimodal wrapper carries both). It is the exported front door onto probeModelTypes, so a backend
// or test can resolve a config's architecture through LookupArch without re-parsing the JSON itself.
func ProbeModelTypes(data []byte) (modelType, textModelType string) {
	return probeModelTypes(data)
}

// probeModelTypes peeks config.json for the architecture id: the top-level model_type and the nested
// text_config.model_type (multimodal wrappers). The registry keys on every alias an arch declares
// (the bare id plus any text/unified wrapper aliases), so LookupArch resolves these directly — no
// separate architecture-name resolver, and no dependency on a backend's probe.
func probeModelTypes(data []byte) (modelType, textModelType string) {
	var probe struct {
		ModelType  string `json:"model_type"`
		TextConfig struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	_ = core.JSONUnmarshal(data, &probe)
	return probe.ModelType, probe.TextConfig.ModelType
}
