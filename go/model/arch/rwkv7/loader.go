// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// loader.go builds an RWKV7Model from a real checkpoint's safetensors — RWKV-7 weights do not fit the
// transformer model.Assemble (no q/k/v attention; the layer is a token-shift + WKV7 recurrence + channel-
// mix stack), so the family carries its own loader, on the same scaffold as mamba2/loader.go. It reads the
// standard HF fla RWKV7ForCausalLM names — model.embeddings.weight, model.layers.N.{pre_norm (layer 0
// only),attn_norm,attn.*,ffn_norm,ffn.*}, model.norm, lm_head — verified against the real
// RWKV/RWKV7-Goose-World2.8-0.1B-HF checkpoint's safetensors header (399 tensors, 12 layers). Every
// geometry fact (H, head_dim, value_dim, the LoRA ranks, intermediate_size) is read from the weight SHAPES,
// not config.json (config.go's epsFromConfig/checkUnsupportedConfig cover the two facts that genuinely
// cannot be) — a checkpoint whose config drifted from its own weights still loads correctly.

// tensorF32 widens a bf16/f32 safetensors tensor to a flat f32 slice (the precision this host chain runs
// in) — identical algorithm to mamba2/composed's tensorF32 (round-trip verified, not shared across
// packages since none of the three arch packages import each other, AX-8's lib-never-imports-consumer
// applied sideways: no arch package is another arch package's consumer).
func tensorF32(t safetensors.Tensor) ([]float32, error) {
	switch t.Dtype {
	case "BF16", "bfloat16":
		if len(t.Data)%2 != 0 {
			return nil, core.NewError("rwkv7.tensorF32: bf16 byte length odd")
		}
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			b := uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8
			out[i] = math.Float32frombits(uint32(b) << 16)
		}
		return out, nil
	case "F32", "float32":
		if len(t.Data)%4 != 0 {
			return nil, core.NewError("rwkv7.tensorF32: f32 byte length not /4")
		}
		out := make([]float32, len(t.Data)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(t.Data[4*i]) | uint32(t.Data[4*i+1])<<8 | uint32(t.Data[4*i+2])<<16 | uint32(t.Data[4*i+3])<<24)
		}
		return out, nil
	}
	return nil, core.NewError("rwkv7.tensorF32: unsupported dtype " + t.Dtype)
}

// loadLora reads one LoRA-MLP's tensors at prefix+"lora.{0,2}.{weight,bias}" — down A [Low,In], up
// B [Out,Low], and (when wantBias) the up projection's bias [Out]. wantBias is false only for g_lora
// (the checkpoint's one bias=False LoRA).
func loadLora(tensors map[string]safetensors.Tensor, prefix string, wantBias bool) (lora, error) {
	aT, ok := tensors[prefix+"lora.0.weight"]
	if !ok || len(aT.Shape) != 2 {
		return lora{}, core.NewError("rwkv7.loadLora: missing/!2D " + prefix + "lora.0.weight")
	}
	bT, ok := tensors[prefix+"lora.2.weight"]
	if !ok || len(bT.Shape) != 2 {
		return lora{}, core.NewError("rwkv7.loadLora: missing/!2D " + prefix + "lora.2.weight")
	}
	a, err := tensorF32(aT)
	if err != nil {
		return lora{}, err
	}
	b, err := tensorF32(bT)
	if err != nil {
		return lora{}, err
	}
	var bias []float32
	if wantBias {
		biasT, ok := tensors[prefix+"lora.2.bias"]
		if !ok {
			return lora{}, core.NewError("rwkv7.loadLora: missing " + prefix + "lora.2.bias")
		}
		bias, err = tensorF32(biasT)
		if err != nil {
			return lora{}, err
		}
	}
	return lora{A: a, B: b, Bias: bias, In: aT.Shape[1], Low: aT.Shape[0], Out: bT.Shape[0]}, nil
}

// LoadRWKV7Model assembles an RWKV7Model from the checkpoint tensors + config.json bytes. Geometry (H,
// head_dim, value_dim, intermediate_size) is derived per layer from r_k/v_proj/ffn.key.weight's own
// shapes and asserted uniform across layers (mirroring mamba2.LoadMambaModel's nGroups=1 shape-derivation
// discipline); norm_eps and the hybrid-attn/hidden_act sanity checks come from configJSON (config.go).
func LoadRWKV7Model(tensors map[string]safetensors.Tensor, configJSON []byte) (*RWKV7Model, error) {
	if err := checkUnsupportedConfig(configJSON); err != nil {
		return nil, err
	}
	eps := epsFromConfig(configJSON)

	get := func(name string) (safetensors.Tensor, bool) { t, ok := tensors[name]; return t, ok }
	f32req := func(name string) ([]float32, error) {
		t, ok := get(name)
		if !ok {
			return nil, core.NewError("rwkv7.LoadRWKV7Model: missing " + name)
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

	embedT, ok := get("model.embeddings.weight")
	if !ok || len(embedT.Shape) != 2 {
		return nil, core.NewError("rwkv7.LoadRWKV7Model: missing/!2D model.embeddings.weight")
	}
	vocab, D := embedT.Shape[0], embedT.Shape[1]
	embed, err := tensorF32(embedT)
	if err != nil {
		return nil, err
	}
	normW, err := f32req("model.norm.weight")
	if err != nil {
		return nil, err
	}
	normB := f32opt("model.norm.bias")
	lmHead := f32opt("lm_head.weight") // nil ⇒ tied to embed

	m := &RWKV7Model{Embed: embed, NormW: normW, NormB: normB, LMHead: lmHead, D: D, Vocab: vocab, Eps: eps}

	for li := 0; ; li++ {
		prefix := core.Sprintf("model.layers.%d.", li)
		rProjT, ok := get(prefix + "attn.r_proj.weight")
		if !ok {
			break // no more layers
		}
		if len(rProjT.Shape) != 2 || rProjT.Shape[1] != D {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d bad attn.r_proj.weight shape", li))
		}
		rProj, err := tensorF32(rProjT)
		if err != nil {
			return nil, err
		}

		rkT, ok := get(prefix + "attn.r_k")
		if !ok || len(rkT.Shape) != 2 {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d missing/!2D attn.r_k", li))
		}
		H, K := rkT.Shape[0], rkT.Shape[1]
		rk, err := tensorF32(rkT)
		if err != nil {
			return nil, err
		}

		vProjT, ok := get(prefix + "attn.v_proj.weight")
		if !ok || len(vProjT.Shape) != 2 {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d missing/!2D attn.v_proj.weight", li))
		}
		Dv := vProjT.Shape[0]
		if H <= 0 || K <= 0 || Dv <= 0 || Dv%H != 0 {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d value_dim %d not divisible by H %d", li, Dv, H))
		}
		V := Dv / H
		vProj, err := tensorF32(vProjT)
		if err != nil {
			return nil, err
		}

		cfg := BlockConfig{NumHeads: H, KeyDim: K, ValueDim: V}
		if li == 0 {
			m.Cfg = cfg
		} else if m.Cfg != cfg {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d geometry %+v differs from layer 0 %+v", li, cfg, m.Cfg))
		}

		kProj, err := f32req(prefix + "attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		oProj, err := f32req(prefix + "attn.o_proj.weight")
		if err != nil {
			return nil, err
		}
		xr, err := f32req(prefix + "attn.x_r")
		if err != nil {
			return nil, err
		}
		xw, err := f32req(prefix + "attn.x_w")
		if err != nil {
			return nil, err
		}
		xk, err := f32req(prefix + "attn.x_k")
		if err != nil {
			return nil, err
		}
		xv, err := f32req(prefix + "attn.x_v")
		if err != nil {
			return nil, err
		}
		xa, err := f32req(prefix + "attn.x_a")
		if err != nil {
			return nil, err
		}
		xg, err := f32req(prefix + "attn.x_g")
		if err != nil {
			return nil, err
		}
		kk, err := f32req(prefix + "attn.k_k")
		if err != nil {
			return nil, err
		}
		ka, err := f32req(prefix + "attn.k_a")
		if err != nil {
			return nil, err
		}
		gnW, err := f32req(prefix + "attn.g_norm.weight")
		if err != nil {
			return nil, err
		}
		gnB := f32opt(prefix + "attn.g_norm.bias")

		wLora, err := loadLora(tensors, prefix+"attn.w_lora.", true)
		if err != nil {
			return nil, err
		}
		aLora, err := loadLora(tensors, prefix+"attn.a_lora.", true)
		if err != nil {
			return nil, err
		}
		gLora, err := loadLora(tensors, prefix+"attn.g_lora.", false)
		if err != nil {
			return nil, err
		}
		var vLora *lora
		if li > 0 {
			vl, verr := loadLora(tensors, prefix+"attn.v_lora.", true)
			if verr != nil {
				return nil, verr
			}
			vLora = &vl
		}

		attn := &timeMixWeights{
			XR: xr, XW: xw, XK: xk, XV: xv, XA: xa, XG: xg,
			RProj: rProj, KProj: kProj, VProj: vProj, OProj: oProj,
			WLora: wLora, ALora: aLora, GLora: gLora, VLora: vLora,
			KK: kk, KA: ka, RK: rk,
			GroupNormW: gnW, GroupNormB: gnB,
		}

		var preW, preB []float32
		if li == 0 {
			preW = f32opt(prefix + "pre_norm.weight")
			preB = f32opt(prefix + "pre_norm.bias")
		}
		attnNormW, err := f32req(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		attnNormB := f32opt(prefix + "attn_norm.bias")
		ffnNormW, err := f32req(prefix + "ffn_norm.weight")
		if err != nil {
			return nil, err
		}
		ffnNormB := f32opt(prefix + "ffn_norm.bias")

		ffnKeyT, ok := get(prefix + "ffn.key.weight")
		if !ok || len(ffnKeyT.Shape) != 2 || ffnKeyT.Shape[1] != D {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d missing/bad ffn.key.weight", li))
		}
		FF := ffnKeyT.Shape[0]
		if li == 0 {
			m.FF = FF
		} else if m.FF != FF {
			return nil, core.NewError(core.Sprintf("rwkv7.LoadRWKV7Model: layer %d intermediate_size %d differs from layer 0 %d", li, FF, m.FF))
		}
		ffnKey, err := tensorF32(ffnKeyT)
		if err != nil {
			return nil, err
		}
		ffnValue, err := f32req(prefix + "ffn.value.weight")
		if err != nil {
			return nil, err
		}
		ffnXK, err := f32req(prefix + "ffn.x_k")
		if err != nil {
			return nil, err
		}

		m.Layers = append(m.Layers, RWKV7Layer{
			PreNormW: preW, PreNormB: preB,
			AttnNormW: attnNormW, AttnNormB: attnNormB,
			Attn:     attn,
			FfnNormW: ffnNormW, FfnNormB: ffnNormB,
			FFN: &channelMixWeights{XK: ffnXK, KeyProj: ffnKey, ValueProj: ffnValue},
		})
	}
	if len(m.Layers) == 0 {
		return nil, core.NewError("rwkv7.LoadRWKV7Model: no model.layers.N.attn.r_proj.weight found")
	}
	return m, nil
}
