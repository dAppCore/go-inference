// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// loader.go builds a MambaModel from a checkpoint's safetensors. Mamba-2 weights do not fit the
// transformer model.Assemble (there is no attention; the layer is a recurrent mixer), so the family
// carries its own loader. It reads the standard HF Mamba2ForCausalLM names — backbone.embeddings,
// backbone.layers.N.norm, backbone.layers.N.mixer.{in_proj,conv1d,A_log,D,dt_bias,norm,out_proj},
// backbone.norm_f, lm_head (tied to embeddings when absent) — widens each to f32, and derives the
// per-layer SSD geometry from the weight shapes exactly like metal's configFromShapes (nGroups=1).

// tensorF32 widens a bf16/f32 safetensors tensor to a flat f32 slice (the precision the host scan runs in).
func tensorF32(t safetensors.Tensor) ([]float32, error) {
	switch t.Dtype {
	case "BF16", "bfloat16":
		if len(t.Data)%2 != 0 {
			return nil, core.NewError("mamba2.tensorF32: bf16 byte length odd")
		}
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			b := uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8
			out[i] = math.Float32frombits(uint32(b) << 16)
		}
		return out, nil
	case "F32", "float32":
		if len(t.Data)%4 != 0 {
			return nil, core.NewError("mamba2.tensorF32: f32 byte length not /4")
		}
		out := make([]float32, len(t.Data)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(t.Data[4*i]) | uint32(t.Data[4*i+1])<<8 | uint32(t.Data[4*i+2])<<16 | uint32(t.Data[4*i+3])<<24)
		}
		return out, nil
	}
	return nil, core.NewError("mamba2.tensorF32: unsupported dtype " + t.Dtype)
}

// LoadMambaModel assembles a MambaModel from the checkpoint tensors. eps is the RMSNorm epsilon from the
// config. The geometry (H, head_dim, d_state, conv_kernel) is read from the layer-0 weight shapes and
// assumed uniform; a model with nGroups>1 is rejected (it would mis-split B/C — declared out of band).
func LoadMambaModel(tensors map[string]safetensors.Tensor, eps float32) (*MambaModel, error) {
	get := func(name string) (safetensors.Tensor, bool) {
		t, ok := tensors[name]
		return t, ok
	}
	f32req := func(name string) ([]float32, error) {
		t, ok := get(name)
		if !ok {
			return nil, core.NewError("mamba2.LoadMambaModel: missing " + name)
		}
		return tensorF32(t)
	}
	f32opt := func(name string) []float32 {
		if t, ok := get(name); ok {
			if v, err := tensorF32(t); err == nil {
				return v
			}
		}
		return nil
	}

	embedT, ok := get("backbone.embeddings.weight")
	if !ok {
		embedT, ok = get("backbone.embed_tokens.weight")
	}
	if !ok || len(embedT.Shape) != 2 {
		return nil, core.NewError("mamba2.LoadMambaModel: missing/!2D backbone.embeddings.weight")
	}
	vocab, d := embedT.Shape[0], embedT.Shape[1]
	embed, err := tensorF32(embedT)
	if err != nil {
		return nil, err
	}
	normF, err := f32req("backbone.norm_f.weight")
	if err != nil {
		return nil, err
	}
	lmHead := f32opt("lm_head.weight") // nil ⇒ tied to embed

	m := &MambaModel{Embed: embed, NormF: normF, LMHead: lmHead, D: d, Vocab: vocab}
	for li := 0; ; li++ {
		mp := core.Sprintf("backbone.layers.%d.mixer.", li)
		inT, ok := get(mp + "in_proj.weight")
		if !ok {
			break // no more layers
		}
		convT, ok := get(mp + "conv1d.weight")
		if !ok {
			return nil, core.NewError("mamba2.LoadMambaModel: layer missing conv1d.weight")
		}
		aLogT, ok := get(mp + "A_log")
		if !ok {
			return nil, core.NewError("mamba2.LoadMambaModel: layer missing A_log")
		}
		// geometry from shapes (nGroups=1): H=len(A_log), convDim=conv[0], K=conv[last],
		// projOut=in_proj[0], dInner=projOut-convDim-H, N=(convDim-dInner)/2, headDim=dInner/H.
		if len(inT.Shape) != 2 || len(aLogT.Shape) != 1 || len(convT.Shape) == 0 {
			return nil, core.NewError("mamba2.LoadMambaModel: unexpected mixer weight ranks")
		}
		H := aLogT.Shape[0]
		convDim := convT.Shape[0]
		K := convT.Shape[len(convT.Shape)-1]
		projOut := inT.Shape[0]
		dInner := projOut - convDim - H
		if H <= 0 || dInner <= 0 || dInner%H != 0 || (convDim-dInner)%2 != 0 {
			return nil, core.NewError(core.Sprintf("mamba2.LoadMambaModel: layer %d geometry not nGroups=1 (H=%d projOut=%d convDim=%d dInner=%d)", li, H, projOut, convDim, dInner))
		}
		cfg := BlockConfig{NumHeads: H, HeadDim: dInner / H, StateDim: (convDim - dInner) / 2, NumGroups: 1, ConvKernel: K, Eps: eps}
		if li == 0 {
			m.Cfg = cfg
		} else if m.Cfg != cfg {
			return nil, core.NewError(core.Sprintf("mamba2.LoadMambaModel: layer %d geometry differs from layer 0", li))
		}
		inProj, err := tensorF32(inT)
		if err != nil {
			return nil, err
		}
		convW, err := tensorF32(convT) // [convDim,1,K] is contiguous [convDim,K] in memory
		if err != nil {
			return nil, err
		}
		aLog, err := tensorF32(aLogT)
		if err != nil {
			return nil, err
		}
		outProj, err := f32req(mp + "out_proj.weight")
		if err != nil {
			return nil, err
		}
		layerNorm, err := f32req(core.Sprintf("backbone.layers.%d.norm.weight", li))
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, MambaLayer{
			Norm: layerNorm,
			W: &BlockWeights{
				InProj: inProj, OutProj: outProj, ConvWeight: convW,
				ConvBias: f32opt(mp + "conv1d.bias"),
				ALog:     aLog,
				D:        f32opt(mp + "D"),
				DtBias:   f32opt(mp + "dt_bias"),
				Norm:     f32opt(mp + "norm.weight"),
			},
		})
	}
	if len(m.Layers) == 0 {
		return nil, core.NewError("mamba2.LoadMambaModel: no backbone.layers.N.mixer found")
	}
	return m, nil
}
