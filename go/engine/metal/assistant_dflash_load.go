// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/dflash"
	coreio "dappco.re/go/io"
)

// assistant_dflash_load.go builds a DFlashDrafter from a checkpoint on disk. It
// hand-rolls NO loader: the drafter's decoder stack loads through the ordinary
// reactive pack loader (LoadAssistantDir → model.ParseAssistantConfig →
// safetensors.LoadDirMmap), exactly as an MTP -assistant does; the reactive spec
// (model/gemma4/assistant_dflash.go) recognises the DFlash config and stamps
// model.MTPDFlash onto the neutral config. This file only reads the DFlash-specific
// parameters the block forward needs — the block size and fused verifier layers via
// the model-free decode/dflash.ParseConfig contract, and the reduced-vocab d2t map —
// and validates the DFlash injection tensors the ordinary assistant validation does
// not know about (per-layer k_proj / v_proj, aux_projection, lm_head).

// LoadDFlashDrafter loads a DFlash block-diffusion drafter from dir: its config.json
// must carry the DFlash marker (speculators_model_type "dflash") and a decoder arch a
// registered assistant spec parses. It returns a ready DFlashDrafter or an error
// naming the first missing / mis-shaped DFlash tensor — the honest load, never a
// silent degrade.
//
//	d, err := native.LoadDFlashDrafter("/models/deepseek-v4-flash-speculator.dflash")
func LoadDFlashDrafter(dir string) (*DFlashDrafter, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("native.dflash.Load", "read config.json", err)
	}
	cfg, ok := dflash.ParseConfig([]byte(cfgStr))
	if !ok {
		return nil, core.NewError("native.dflash.Load: config.json is not a DFlash speculator (speculators_model_type != dflash)")
	}
	m, err := LoadAssistantDir(dir)
	if err != nil {
		return nil, core.E("native.dflash.Load", "load drafter pack", err)
	}
	drafter, err := newDFlashDrafter(m, cfg)
	if err != nil {
		_ = m.Close()
		return nil, err
	}
	return drafter, nil
}

// newDFlashDrafter validates a loaded AssistantModel against the DFlash contract and
// resolves the block forward's parameters. It fails unless the reactive spec stamped
// model.MTPDFlash (so an ordinary MTP drafter can never be mistaken for a DFlash one)
// and every DFlash injection tensor is present and correctly shaped.
func newDFlashDrafter(m *AssistantModel, cfg dflash.Config) (*DFlashDrafter, error) {
	if m == nil {
		return nil, core.NewError("native.dflash: assistant model is nil")
	}
	if !resolveDFlashMethod(m) {
		return nil, core.NewError("native.dflash: drafter config method is " + string(m.Config.Method) + ", want dflash (the reactive spec did not stamp MTPDFlash)")
	}
	numAux := len(cfg.AuxHiddenLayerIDs)
	if numAux <= 0 {
		return nil, core.NewError("native.dflash: config declares no aux_hidden_state_layer_ids (no verifier layers to fuse)")
	}
	backbone := m.BackboneHiddenSize
	hidden := m.Arch.Hidden
	headDim := m.Arch.HeadDim
	kvHeads := m.Arch.KVHeads
	if kvHeads <= 0 {
		kvHeads = m.Arch.Heads
	}
	if backbone <= 0 || hidden <= 0 || headDim <= 0 || kvHeads <= 0 || len(m.Arch.Layer) == 0 {
		return nil, core.NewError("native.dflash: drafter arch has incomplete dimensions")
	}

	// aux_projection: [backbone, numAux*backbone] — fuses the concatenated verifier
	// hiddens into one context feature.
	if err := requireDFlashTensorShape(m, dflashAuxProjectionWeight, backbone, numAux*backbone); err != nil {
		return nil, err
	}
	// per-layer k_proj / v_proj: [kvHeads*headDim, backbone] — the injection weights a
	// plain MTP drafter lacks (it reads the target's live K/V instead).
	rowElems := kvHeads * headDim
	for li := range m.Arch.Layer {
		prefix := core.Sprintf("model.layers.%d.self_attn.", li)
		if err := requireDFlashTensorShape(m, prefix+"k_proj.weight", rowElems, backbone); err != nil {
			return nil, err
		}
		if err := requireDFlashTensorShape(m, prefix+"v_proj.weight", rowElems, backbone); err != nil {
			return nil, err
		}
	}
	// lm_head: [draftVocab, hidden] — the reduced-vocab draft head.
	headTensor, ok := m.Tensors[dflashLMHeadWeight]
	if !ok || len(headTensor.Shape) != 2 || headTensor.Shape[1] != hidden || headTensor.Shape[0] <= 0 {
		return nil, core.NewError("native.dflash: missing or mis-shaped " + dflashLMHeadWeight + " (want [draftVocab, hidden])")
	}
	draftVocab := headTensor.Shape[0]

	d2t, err := loadDFlashD2T(m, draftVocab)
	if err != nil {
		return nil, err
	}

	return &DFlashDrafter{
		m:          m,
		blockSize:  max(cfg.BlockSize, 1),
		numAux:     numAux,
		auxLayers:  append([]int(nil), cfg.AuxHiddenLayerIDs...),
		draftVocab: draftVocab,
		d2t:        d2t,
	}, nil
}

// loadDFlashD2T reads the draft→target vocab map (dflash.d2t, bf16, [draftVocab]).
// The table is optional: a pack that omits it drafts directly in the target vocab
// (identity), still lossless. Present, it must have exactly draftVocab entries.
func loadDFlashD2T(m *AssistantModel, draftVocab int) ([]int32, error) {
	t, ok := m.Tensors[dflashD2TTensor]
	if !ok {
		return nil, nil // identity mapping
	}
	f := bf16ToF32Slice(t.Data)
	if len(f) != draftVocab {
		return nil, core.NewError(core.Sprintf("native.dflash: %s has %d entries, want draftVocab %d", dflashD2TTensor, len(f), draftVocab))
	}
	d2t := make([]int32, draftVocab)
	for i, v := range f {
		d2t[i] = int32(v + 0.5) // bf16 stores exact small integer target ids
	}
	return d2t, nil
}

// requireDFlashTensorShape asserts a bf16 tensor is present with exactly [rows,
// cols], naming it in the DFlash vocabulary so a mis-exported pack fails loudly.
func requireDFlashTensorShape(m *AssistantModel, name string, rows, cols int) error {
	t, ok := m.Tensors[name]
	if !ok || t.Dtype == "" || len(t.Data) == 0 {
		return core.NewError("native.dflash: missing required tensor " + name)
	}
	if len(t.Shape) != 2 || t.Shape[0] != rows || t.Shape[1] != cols {
		return core.NewError(core.Sprintf("native.dflash: tensor %s shape %v, want [%d, %d]", name, t.Shape, rows, cols))
	}
	return nil
}
