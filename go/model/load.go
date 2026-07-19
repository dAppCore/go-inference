// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// mmapRetainer is a TokenModel whose weights may ALIAS the checkpoint mmap (a zero-copy load).
// LoadComposedDir offers the mapping to RetainMmap: the model takes ownership (returns true) and
// unmaps it on its own Close/finalize when its weights view it, or declines (returns false) so the
// loader unmaps it immediately when every weight was copied/widened out. This keeps the mapping's
// lifetime with whoever actually references it, without LoadComposedDir needing to know which.
type mmapRetainer interface {
	RetainMmap(io.Closer) bool
}

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
	if spec.Composed != nil && spec.Parse == nil {
		// A composed-ONLY arch is not the reactive transformer Assemble path (its linear_attention layers
		// have no q/k/v to assemble) — route it through LoadComposedDir, not here. A DUAL-route arch
		// (Composed AND Parse+Weights, e.g. qwen35) carries both, so it reaches Assemble here when a caller
		// deliberately bypasses LoadComposedDir (engine/metal's default factory route for qwen3_5*) — #18.
		return nil, nil, core.NewError("model.Load: " + mt + " is a composed/hybrid arch — load via LoadComposedDir")
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
	if spec.NormalizeConfig != nil {
		tensors = spec.NormalizeConfig(tensors, ac)
		dm.Tensors = tensors
	}
	// Widen any F16 float tensors to BF16 before the assembler binds them: many MLX community packs
	// ship the un-quantised floats (norms, affine scales/biases, q/k/v biases, dense weights) as IEEE
	// half, which the byte-native engine's bf16 kernels would misread bit-for-bit. Byte length and
	// shape are unchanged, so InferFromWeights/Assemble below read identical geometry; a bf16 pack has
	// nothing to widen and stays byte-identical + zero-copy. tensors aliases dm.Tensors, so the
	// in-place widen is visible to Assemble.
	dm.WidenF16TensorsToBF16()
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

// LoadComposedDir reads dir and builds the hybrid (non-Assemble) TokenModel through the model_type's
// registered Composed hook — the neutral, backend-agnostic routing for a config-composed stack (Qwen 3.6
// gated-delta + full attention) whose weights the reactive transformer Assemble does not describe. It is
// the registry-driven replacement for a backend's hardcoded model_type switch: probe model_type (with the
// text_config fallback for the multimodal wrapper), look up the ArchSpec, and when it carries a Composed
// hook, map the safetensors and call it. ok=false means the model_type is a standard transformer (or is
// unregistered) — load it via Load instead.
//
// Mmap lifetime: the composed hook builds ZERO-COPY, so a quant checkpoint's packed weights VIEW the mmap
// rather than being copied to the heap (the RSS win). The model takes ownership of the mapping through
// RetainMmap and unmaps it on its own Close/finalize; when the model aliases nothing (a dense pack widened
// to f32, or an all-1-bit pack repacked to owned heap) RetainMmap declines and the mapping is unmapped here.
func LoadComposedDir(dir string) (TokenModel, bool, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, false, core.E("model.LoadComposedDir", "read config.json", err)
	}
	cfg := []byte(cfgStr)
	mt, textMT := probeModelTypes(cfg)
	spec, ok := LookupArch(mt)
	if !ok && textMT != "" {
		spec, ok = LookupArch(textMT)
	}
	if !ok || spec.Composed == nil {
		return nil, false, nil
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, false, err
	}
	tm, err := spec.Composed(dm.Tensors, cfg)
	if err != nil {
		_ = dm.Close()
		return nil, false, err
	}
	// Hand the mapping to the model if its weights alias it; otherwise unmap now (nothing views it).
	if r, ok := tm.(mmapRetainer); !ok || !r.RetainMmap(dm) {
		_ = dm.Close()
	}
	return tm, true, nil
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
