// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// lora_fuse.go bridges native training to native serving: a LoRA adapter trained by the train_*.go
// stack (or loaded from disk) is folded directly into a BF16Model's projection weights in memory, so
// NewArchSession serves the adapted model without a disk round-trip. Each targeted weight W becomes
// W + scaling·(B·A), rounded back to bf16 — the same fuse lora.FuseIntoPack does on disk, here against
// the engine's own in-memory model so a freshly trained adapter goes straight to serving.

// LoRADelta is a trained LoRA adapter for one projection: A is [rank, in_features], B is [out_features,
// rank], and the fused weight is W + Scaling·(B·A). Proj names the DecodeLayerWeights field to fold into.
type LoRADelta struct {
	Layer   int
	Proj    string // "wq" "wk" "wv" "wo" "wgate" "wup" "wdown"
	A, B    []float32
	Rank    int
	Scaling float32
}

// selectProj returns a pointer to the named projection's bf16 bytes in a layer (nil for an unknown name).
func selectProj(lw *DecodeLayerWeights, name string) *[]byte {
	switch name {
	case "wq":
		return &lw.WQ
	case "wk":
		return &lw.WK
	case "wv":
		return &lw.WV
	case "wo":
		return &lw.WO
	case "wgate":
		return &lw.WGate
	case "wup":
		return &lw.WUp
	case "wdown":
		return &lw.WDown
	}
	return nil
}

// FuseLoRAIntoModel folds the given LoRA deltas into g's projection weights in place (W += scaling·B·A,
// re-rounded to bf16), so a session built from g serves the adapted model. The weight bytes are mutated,
// so pass a model you own (a shared/tied weight would propagate the fold). A delta whose B·A shape does
// not match the target weight is a loud error.
func FuseLoRAIntoModel(g *BF16Model, deltas []LoRADelta) error {
	if err := ensureInit(); err != nil {
		return err
	}
	for _, d := range deltas {
		if d.Layer < 0 || d.Layer >= len(g.Layers) {
			return core.NewError("native.FuseLoRAIntoModel: layer index out of range")
		}
		if d.Rank <= 0 || len(d.A)%d.Rank != 0 || len(d.B)%d.Rank != 0 {
			return core.NewError("native.FuseLoRAIntoModel: A/B not divisible by rank")
		}
		w := selectProj(&g.Layers[d.Layer], d.Proj)
		if w == nil {
			return core.NewError("native.FuseLoRAIntoModel: unknown projection " + d.Proj)
		}
		in := len(d.A) / d.Rank  // A is [rank, in]
		out := len(d.B) / d.Rank // B is [out, rank]
		if out*in*bf16Size != len(*w) {
			return core.NewError("native.FuseLoRAIntoModel: B·A shape != target weight shape")
		}
		ba, err := MatMulF32(d.B, d.A, out, d.Rank, in) // [out, in]
		if err != nil {
			return err
		}
		bytes := *w
		for i := 0; i < out*in; i++ {
			base := bf16ToF32(bytes[2*i], bytes[2*i+1])
			nv := f32ToBF16(base + d.Scaling*ba[i])
			bytes[2*i], bytes[2*i+1] = byte(nv), byte(nv>>8)
		}
	}
	return nil
}
