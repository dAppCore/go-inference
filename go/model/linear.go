// SPDX-Licence-Identifier: EUPL-1.2

package model

import "dappco.re/go/inference/safetensors"

// Linear is a backend-agnostic linear weight: bf16-dense OR affine-quantised, the choice
// made PER WEIGHT by the presence of a ".scales" tensor — mirroring the metal model
// package's per-weight Linear (weights.go). A model declares no quant; the backend's
// registered QuantMatVec (Quantised) or a plain bf16 matvec (dense) does the rest, so model
// architecture stays independent of quant format. The byte slices VIEW the source
// safetensors mmap (zero-copy); a backend uploads them to its device.
//
// This is why a checkpoint in bf16 / 4 / 5 / 6 / 8-bit — or one that quantises a weight
// another leaves bf16 (e4b's per_layer_model_projection) — loads through ONE path: the
// loader never assumes a weight's format, and the affine geometry is read from the tensor
// SHAPES (not a config scalar), so each weight carries its own width.
type Linear struct {
	Weight         []byte // packed quant codes, or dense bf16 — raw little-endian, viewing the mmap
	Scales, Biases []byte // affine group scales / biases (nil ⇒ dense bf16)
	Bias           []byte // optional additive bias (nil ⇒ none)
	OutDim, InDim  int    // logical shape of the dequantised weight (rows × cols)
	GroupSize      int    // affine group size (0 ⇒ dense bf16)
	Bits           int    // affine bit-width (0 ⇒ dense bf16)
	Kind           string // quant kind for the (backend,kind) registry ("affine"…); "" ⇒ dense bf16
}

// Quantised reports whether this weight carries affine quant metadata (a ".scales" tensor)
// — i.e. its MatVec must go through the registered QuantMatVec, not a dense bf16 matvec.
func (l *Linear) Quantised() bool { return l != nil && l.Scales != nil && l.Kind != "" }

// LoadLinear builds the Linear at prefix from a safetensors tensor set, making the per-weight
// quant decision: prefix+".scales" present ⇒ quantised (Kind set, GroupSize/Bits derived from
// the scales + packed-weight shapes), else dense bf16. OutDim is read from the weight's first
// dimension (rows are never packed) — so a per-layer-varying FFN width (MatFormer) is taken from
// the shape, not assumed; inDim is the LOGICAL input width (from the arch — a packed weight's
// columns differ). Returns nil when prefix+".weight" is absent (an optional weight). Mirrors
// the metal package's per-weight Linear loader.
func LoadLinear(t map[string]safetensors.Tensor, prefix string, inDim int, kind string) *Linear {
	w, ok := t[prefix+".weight"]
	if !ok {
		return nil
	}
	lin := &Linear{Weight: w.Data, OutDim: firstDim(w.Shape), InDim: inDim}
	if b, ok := t[prefix+".bias"]; ok {
		lin.Bias = b.Data
	}
	if s, ok := t[prefix+".scales"]; ok && len(s.Data) > 0 {
		lin.Scales = s.Data
		if b, ok := t[prefix+".biases"]; ok {
			lin.Biases = b.Data
		}
		lin.Kind = kind
		lin.GroupSize, lin.Bits = affineGeometry(inDim, s.Shape, w.Shape)
	}
	return lin
}

// affineGeometry derives the affine group size + bit-width from the tensor shapes alone —
// the per-weight fact that makes loading quant-agnostic. The scales tensor is
// [outDim, nGroups] (one scale per group per row), so groupSize = inDim / nGroups; the packed
// weight is uint32 [outDim, inDim·bits/32], so bits = packedCols·32 / inDim. Holds for every
// MLX affine width (4/5/6/8). Returns 0,0 when the shapes don't encode a quantised weight.
func affineGeometry(inDim int, scalesShape, weightShape []int) (groupSize, bits int) {
	if inDim <= 0 {
		return 0, 0
	}
	if n := lastDim(scalesShape); n > 0 {
		groupSize = inDim / n
	}
	if packedCols := lastDim(weightShape); packedCols > 0 {
		bits = packedCols * 32 / inDim
	}
	return groupSize, bits
}

// lastDim returns the final dimension of a shape, or 0 for a rank-0/empty shape.
func lastDim(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	return shape[len(shape)-1]
}

// firstDim returns the first dimension (the output rows of a weight — never packed), or 0.
func firstDim(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	return shape[0]
}
