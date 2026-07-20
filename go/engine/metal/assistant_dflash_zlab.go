// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	zlabdflash "dappco.re/go/inference/model/arch/z-lab/dflash"
)

// assistant_dflash_zlab.go is the z-lab-convention twin of assistant_dflash.go
// + assistant_dflash_load.go: those files load and run the SPECULATORS
// (RedHatAI) convention drafter — model.-prefixed tensors, its own
// reduced-vocab dflash.lm_head + d2t, hand-injected per-layer K/V
// (docs/design-dflash-forward.md §1 names it the pre-checkpoint forward the
// #37 survey falsified). Every PUBLISHED checkpoint (z-lab/Qwen3-4B-DFlash-b16
// and siblings) uses a different convention instead — unprefixed layers.N.*
// tensors, no embedding/head/d2t of its own (borrowed from the TARGET) — and
// runs through the real forward, dflash_zlab.go's DFlashZLabForward,
// oracle-gated against decode/dflash.ZLabForward AND the real checkpoint
// (dflash_zlab_test.go's TestDFlashZLabForward_RealCheckpoint). This file is
// the glue that arms that forward for live serving:
//
//   - recognition + load (loadZLabSpeculativePair, called from
//     LoadSpeculativePair BEFORE it would attempt LoadAssistantPairDirs — the
//     z-lab convention cannot load through that loader at all: it demands
//     model.embed_tokens.weight, which this convention architecturally lacks);
//   - the BlockProposer adapter (zLabDFlashProposer) that seeds each block
//     from the TARGET's own embedding table (the real anchor token at
//     position 0, MaskTokenID's embedding at every other position — never a
//     single shared anchor broadcast) and reads each proposed position off
//     the TARGET's own borrowed lm_head (never a reduced draft vocab / d2t
//     remap — the real checkpoint has neither).
//
// Nothing here touches assistant_dflash*.go's speculators-convention maths;
// docs/design-dflash-forward.md §7 keeps that as a separate arm.

// zLabDFlashDrafter is a loaded z-lab-convention DFlash drafter: the typed
// decoder payload (model/arch/z-lab/dflash.DraftModel) plus the block
// parameters the live proposer reads. It carries no embedding, head, or KV
// state of its own — every generate call borrows the paired TARGET
// ArchSession for both (speculativeModel.generateDFlashZLab).
type zLabDFlashDrafter struct {
	model *zlabdflash.DraftModel
}

// loadZLabDFlashDrafter loads a z-lab DFlash drafter checkpoint directly
// through the arch package (config + safetensors → payload) — no reactive
// pack loader, no AssistantModel: the real checkpoint carries none of the
// tensors that loader demands.
func loadZLabDFlashDrafter(dir string) (*zLabDFlashDrafter, error) {
	m, err := zlabdflash.Load(dir)
	if err != nil {
		return nil, err
	}
	return &zLabDFlashDrafter{model: m}, nil
}

// BlockSize is the number of positions ProposeBlock's forward evaluates per
// readout (config block_size — 16 on every published checkpoint); the
// PROPOSED block itself is BlockSize-1 tokens (position 0 is the seed anchor,
// never re-proposed — spec_generate's own convention, docs/design-dflash-
// forward.md §7a item 2).
func (d *zLabDFlashDrafter) BlockSize() int { return max(d.model.Cfg.Block.BlockSize, 1) }

// AuxLayers returns the target_layer_ids whose hidden states fuse into the
// drafter's context — the taps a caller extracts from the target forward, in
// fusion order (fc.weight's column order).
func (d *zLabDFlashDrafter) AuxLayers() []int {
	return append([]int(nil), d.model.Cfg.Block.AuxHiddenLayerIDs...)
}

// MaskTokenID is the token id every not-yet-drafted block position embeds as
// (the real checkpoint's dflash_config.mask_token_id).
func (d *zLabDFlashDrafter) MaskTokenID() int32 { return int32(d.model.Cfg.Block.MaskTokenID) }

// zLabDFlashSource supplies the z-lab block forward's inputs for the current
// context: the growing target hidden states at target_layer_ids — EVERY
// context token, not just the last (docs/design-dflash-forward.md §2's
// "context length is the number of target TOKENS fused in" — the
// numAux-as-context-length conflation assistant_dflash.go's speculators
// forward made, §1's falsified-forward table) — as ONE f32 array [ctxLen,
// numAux*hidden] (ExtractAuxHiddensAllRaw's shape), plus the TARGET's own
// bf16 embedding of the anchor (last committed) token.
type zLabDFlashSource func(context []int) (targetHiddenRaw []float32, ctxLen int, anchorEmbedding []byte, ok bool)

// zLabDFlashProposer adapts a zLabDFlashDrafter + a zLabDFlashSource + the
// paired target's borrowed head to decode/dflash.BlockProposer — the z-lab
// twin of dflashProposer (assistant_dflash_proposer.go), which adapts the
// speculators-convention drafter instead.
type zLabDFlashProposer struct {
	drafter   *zLabDFlashDrafter
	source    zLabDFlashSource
	maskEmbed []byte                              // TARGET bf16 embedding of MaskTokenID — constant per generate call
	head      func(hidden []byte) (int32, error) // TARGET's own borrowed lm_head + argmax, bf16 hidden -> target-vocab id
}

var _ dflash.BlockProposer = (*zLabDFlashProposer)(nil)

// ProposeBlock runs the real z-lab forward (DFlashZLabForward,
// dflash_zlab.go) for the current context and reads each proposed position
// off the TARGET's own borrowed head — never the drafter's own (it has
// none). An unavailable source, a forward error, or a head-readout error
// returns an empty block, which the driver (decode/dflash.AcceptBlock) treats
// exactly as a drafter miss: the target decodes one token itself, still
// lossless.
func (p *zLabDFlashProposer) ProposeBlock(context []int) []int {
	if p == nil || p.drafter == nil || p.source == nil || p.head == nil || len(p.maskEmbed) == 0 {
		return nil
	}
	targetHiddenRaw, ctxLen, anchorEmbedding, ok := p.source(context)
	if !ok {
		return nil
	}
	hidden := p.drafter.model.Cfg.Hidden
	if len(anchorEmbedding) != hidden*bf16Size || len(p.maskEmbed) != hidden*bf16Size {
		return nil
	}
	blockLen := p.drafter.BlockSize()
	// noiseEmbedding: position 0 is the real anchor (the target's own
	// embedding of the last committed/verified token); every other position
	// is still literally MaskTokenID's embedding — output_ids' global
	// mask_token_id pre-fill in the reference (docs/design-dflash-forward.md
	// §7a item 2), never a broadcast of the SAME anchor.
	noise := make([]float32, blockLen*hidden)
	copy(noise[:hidden], bf16ToF32Slice(anchorEmbedding))
	maskF32 := bf16ToF32Slice(p.maskEmbed)
	for j := 1; j < blockLen; j++ {
		copy(noise[j*hidden:(j+1)*hidden], maskF32)
	}
	out, err := DFlashZLabForward(p.drafter.model, noise, targetHiddenRaw, ctxLen, blockLen)
	if err != nil {
		return nil
	}
	// Position 0's output is the seed's own infilled prediction — the
	// reference never samples it (spec_generate keeps only
	// self(...)[:, -block_size+1:, :]); positions [1, blockLen) are the
	// proposed block, read off the BORROWED target head, never a reduced
	// draft vocab.
	block := make([]int, 0, blockLen-1)
	for j := 1; j < blockLen; j++ {
		row := out[j*hidden : (j+1)*hidden]
		id, herr := p.head(f32ToBf16Slice(row))
		if herr != nil {
			return nil
		}
		block = append(block, int(id))
	}
	return block
}

// loadZLabSpeculativePair builds the speculativeModel for a z-lab-convention
// DFlash pairing: target loads exactly as the plain/speculators paths do
// (already open by the caller — LoadSpeculativePair); the drafter loads
// directly through the arch package. Called from LoadSpeculativePair BEFORE
// it attempts LoadAssistantPairDirs — see this file's header for why the
// z-lab convention cannot load through that loader.
func loadZLabSpeculativePair(target *ArchSession, targetPath, draftPath string, draftBlock int) (inference.TextModel, error) {
	zlab, err := loadZLabDFlashDrafter(draftPath)
	if err != nil {
		_ = target.Close()
		return nil, core.E("native.LoadSpeculativePair", "attach z-lab DFlash drafter", err)
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetPath, "tokenizer.json"))
	if err != nil {
		_ = target.Close()
		return nil, core.E("native.LoadSpeculativePair", "load tokenizer", err)
	}
	modelType := probeModelType(targetPath)
	return &speculativeModel{
		target:           target,
		zlab:             zlab,
		tok:              tok,
		modelType:        modelType,
		draftBlock:       draftBlock,
		turns:            engine.DetectTurnTokens(tok),
		declaredStops:    loadGenerationConfigStops(targetPath),
		declaredSampling: loadGenerationConfigSamplingDefaults(targetPath),
		info: inference.ModelInfo{
			Architecture: modelType,
			VocabSize:    target.arch.Vocab,
			NumLayers:    len(target.arch.Layer),
			HiddenSize:   target.arch.Hidden,
		},
	}, nil
}
