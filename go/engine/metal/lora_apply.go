// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// lora_apply.go is the load-time half of the train→save→serve round trip: it HONOURS AdapterPath so a
// saved adapter reapplies at inference, mirroring engine/hip's model.LoadAdapter. It reads the go-mlx
// on-disk adapter package (adapter.safetensors + adapter_config.json) and folds the trained head LoRA
// into a CLONED copy of the model's output head before the head encoder is built — so `serve --adapter
// <path>` generates through the adapted head with no per-token cost and no mutation of the frozen (and,
// for a directory-loaded model, memory-mapped) base weights. The head is the target LoRATrainer.Save
// writes; a per-LAYER adapter (go-mlx's layers.N.proj format, folded by FuseLoRAIntoModel) applies to an
// in-memory BF16Model but needs the decode weight buffers rebuilt for a zero-copy directory model — the
// separate follow-up this file deliberately does not attempt.

// headLoRA holds a parsed head adapter: the A [rank,dModel] and B [vocab,rank] factors plus the derived
// scaling (alpha/rank) and dimensions.
type headLoRA struct {
	a, b    []float32
	rank    int
	dModel  int
	vocab   int
	scaling float32
}

// loadHeadAdapter reads the adapter package at dir and extracts the head LoRA (lm_head.lora_a /
// lm_head.lora_b). Returns nil (no error) when the adapter is not a head adapter (e.g. a layers.N.proj
// adapter), so the caller can route it elsewhere.
func loadHeadAdapter(dir string) (*headLoRA, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "adapter_config.json"))
	if err != nil {
		return nil, core.E("native.loadHeadAdapter", "read adapter_config.json", err)
	}
	var cfg adapterConfigJSON
	if !core.JSONUnmarshal([]byte(cfgStr), &cfg).OK {
		return nil, core.NewError("native.loadHeadAdapter: parse adapter_config.json")
	}
	tensors, err := safetensors.Load(core.PathJoin(dir, "adapter.safetensors"))
	if err != nil {
		return nil, core.E("native.loadHeadAdapter", "load adapter.safetensors", err)
	}
	ta, okA := tensors["lm_head.lora_a"]
	tb, okB := tensors["lm_head.lora_b"]
	if !okA || !okB {
		return nil, nil // not a head adapter
	}
	if len(ta.Shape) != 2 || len(tb.Shape) != 2 {
		return nil, core.NewError("native.loadHeadAdapter: lm_head factors must be 2-D")
	}
	rank, dModel, vocab := ta.Shape[0], ta.Shape[1], tb.Shape[0]
	if tb.Shape[1] != rank {
		return nil, core.NewError("native.loadHeadAdapter: lm_head A/B rank mismatch")
	}
	a, err := safetensors.DecodeFloat32(ta.Dtype, ta.Data, rank*dModel)
	if err != nil {
		return nil, core.E("native.loadHeadAdapter", "decode lm_head.lora_a", err)
	}
	b, err := safetensors.DecodeFloat32(tb.Dtype, tb.Data, vocab*rank)
	if err != nil {
		return nil, core.E("native.loadHeadAdapter", "decode lm_head.lora_b", err)
	}
	rankF := cfg.Rank
	if rankF <= 0 {
		rankF = rank
	}
	scaling := cfg.Alpha / float32(rankF)
	if cfg.Alpha == 0 {
		scaling = 1
	}
	return &headLoRA{a: a, b: b, rank: rank, dModel: dModel, vocab: vocab, scaling: scaling}, nil
}

// applyHeadAdapterToModel folds a trained head adapter from dir into g's output head: g.LMHead becomes a
// freshly-owned bf16 copy of the base head with base + scaling·(B·A) written in. The clone is essential —
// a directory-loaded model's LMHead is a read-only mmap view (and, when tied, aliases the input
// embedding), so folding in place would corrupt the base; the clone leaves g.Embed and the on-disk
// weights untouched. Returns false (no error) when dir is not a head adapter, so the caller can route it.
func applyHeadAdapterToModel(g *BF16Model, dir string) (bool, error) {
	if g == nil {
		return false, core.NewError("native.applyHeadAdapterToModel: nil model")
	}
	head, err := loadHeadAdapter(dir)
	if err != nil {
		return false, err
	}
	if head == nil {
		return false, nil
	}
	if err := ensureInit(); err != nil {
		return false, err
	}
	if head.vocab*head.dModel*bf16Size != len(g.LMHead) {
		return false, core.NewError("native.applyHeadAdapterToModel: adapter head shape does not match the model head")
	}
	ba, err := MatMulF32(head.b, head.a, head.vocab, head.rank, head.dModel) // [vocab,dModel]
	if err != nil {
		return false, err
	}
	fused := make([]byte, len(g.LMHead)) // owned clone — never mutate the base mmap / tied embedding
	for i := 0; i < head.vocab*head.dModel; i++ {
		base := bf16ToF32(g.LMHead[2*i], g.LMHead[2*i+1])
		nv := f32ToBF16(base + head.scaling*ba[i])
		fused[2*i], fused[2*i+1] = byte(nv), byte(nv>>8)
	}
	g.LMHead = fused
	g.Tied = false // the fused head is now distinct from the input embedding
	return true, nil
}

// applyAdapterToBF16Model honours a load-time AdapterPath against an in-memory bf16 model: it applies the
// head adapter (the LoRATrainer.Save target) when present. A layers.N.proj adapter is not applied here
// (it needs the decode weight buffers rebuilt on a zero-copy directory model — the follow-up); an
// adapter that is neither is a loud error rather than a silent no-op.
func applyAdapterToBF16Model(g *BF16Model, adapterDir string) error {
	applied, err := applyHeadAdapterToModel(g, adapterDir)
	if err != nil {
		return err
	}
	if !applied {
		return core.NewError("native.applyAdapterToBF16Model: adapter at " + adapterDir + " is not a head adapter; per-layer adapter apply on the native load path is not wired yet")
	}
	return nil
}
