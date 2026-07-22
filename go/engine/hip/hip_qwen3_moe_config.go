// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// hipQwen3MoEConfig is the resolved qwen3_moe decode geometry: SiLU-gated sparse-expert
// MoE, per-head QK-norm, uniform full attention (no sliding/global split, no PLE), rope
// theta from the checkpoint. The v1 sibling of hipGemma4Q4Layer0Config — plain (F32
// device-resident, host-orchestrated) rather than gemma4's Q4-packed pipeline, since
// performance tuning is out of scope for a first correctness pass.
type hipQwen3MoEConfig struct {
	HiddenSize int
	NumLayers  int
	VocabSize  int
	Heads      int
	KVHeads    int
	HeadDim    int

	NumExperts int
	TopK       int
	ExpertFF   int

	Epsilon       float32
	RopeTheta     float32
	NormaliseTopK bool // config's norm_topk_prob — see hipQwen3MoERouterSelect
}

// resolveHIPQwen3MoEConfig derives the decode geometry from a parsed DenseConfig — the
// shared config surface dense_config.go already parses qwen3_moe's fields into
// (NumExperts, NumExpertsPerTok, MoEIntermediateSize, HeadDim, RopeTheta, RMSNormEps, …).
func resolveHIPQwen3MoEConfig(cfg *DenseConfig) (hipQwen3MoEConfig, error) {
	if cfg == nil {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "config is required", nil)
	}
	if normalizeROCmArchitecture(cfg.ModelType) != "qwen3_moe" {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "config model_type is not qwen3_moe", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "hidden size, layer count, and vocab size must be positive", nil)
	}
	if cfg.NumAttentionHeads <= 0 {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "attention head count must be positive", nil)
	}
	headDim := cfg.HeadDim
	if headDim <= 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "hidden size must divide by attention head count when head_dim is absent", nil)
		}
		headDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	kvHeads := cfg.NumKeyValueHeads
	if kvHeads <= 0 {
		kvHeads = cfg.NumAttentionHeads
	}
	if cfg.NumAttentionHeads%kvHeads != 0 {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "attention head count must be a multiple of key/value head count", nil)
	}
	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsPerTok
	expertFF := cfg.MoEIntermediateSize
	if numExperts <= 0 || topK <= 0 || expertFF <= 0 {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "expert count, top-k, and expert intermediate size must be positive", nil)
	}
	if topK > numExperts {
		return hipQwen3MoEConfig{}, core.E("rocm.hip.Qwen3MoE", "top-k exceeds expert count", nil)
	}
	eps := float32(cfg.RMSNormEps)
	if eps <= 0 {
		eps = 1e-6
	}
	ropeTheta := float32(cfg.RopeTheta)
	if ropeTheta <= 0 {
		ropeTheta = 1_000_000
	}
	return hipQwen3MoEConfig{
		HiddenSize: cfg.HiddenSize, NumLayers: cfg.NumHiddenLayers, VocabSize: cfg.VocabSize,
		Heads: cfg.NumAttentionHeads, KVHeads: kvHeads, HeadDim: headDim,
		NumExperts: numExperts, TopK: topK, ExpertFF: expertFF,
		Epsilon: eps, RopeTheta: ropeTheta, NormaliseTopK: cfg.NormTopKProb,
	}, nil
}

// hipQwen3MoELayerWeights holds one decode layer's device-resident weights — every
// tensor uploaded once at load time (F32-encoded) and referenced by pointer thereafter.
// Q/K carry a per-head norm (QNorm/KNorm); there is no value-norm and no shared expert
// in the qwen3_moe family (qwenmoe/weights.go: Qwen3-MoE dropped the shared expert
// Qwen1.5/Qwen2-MoE carried — out of this v1's scope, see qwen3_moe_runtime.go).
type hipQwen3MoELayerWeights struct {
	InputNorm    *hipDeviceByteBuffer // [D]
	QProj        *hipDeviceByteBuffer // [Heads*HeadDim, D]
	KProj        *hipDeviceByteBuffer // [KVHeads*HeadDim, D]
	VProj        *hipDeviceByteBuffer // [KVHeads*HeadDim, D]
	OProj        *hipDeviceByteBuffer // [D, Heads*HeadDim]
	QNorm        *hipDeviceByteBuffer // [HeadDim]
	KNorm        *hipDeviceByteBuffer // [HeadDim]
	PostAttnNorm *hipDeviceByteBuffer // [D] — the pre-MoE ("pre-FFN") norm
	Router       *hipDeviceByteBuffer // [NumExperts, D]
	ExpertGate   []*hipDeviceByteBuffer // [NumExperts] of [ExpertFF, D]
	ExpertUp     []*hipDeviceByteBuffer // [NumExperts] of [ExpertFF, D]
	ExpertDown   []*hipDeviceByteBuffer // [NumExperts] of [D, ExpertFF]
}

// Close releases every device buffer this layer owns. Nil-safe on a partially built
// layer so a failed load can unwind cleanly.
func (w *hipQwen3MoELayerWeights) Close() {
	if w == nil {
		return
	}
	for _, buf := range []*hipDeviceByteBuffer{w.InputNorm, w.QProj, w.KProj, w.VProj, w.OProj, w.QNorm, w.KNorm, w.PostAttnNorm, w.Router} {
		if buf != nil {
			_ = buf.Close()
		}
	}
	for _, group := range [][]*hipDeviceByteBuffer{w.ExpertGate, w.ExpertUp, w.ExpertDown} {
		for _, buf := range group {
			if buf != nil {
				_ = buf.Close()
			}
		}
	}
}

// hipQwen3MoEWeights is the whole model's device-resident weight set: the token
// embedding table stays host-resident (Embed is a plain row lookup, never a kernel
// launch — see hip_qwen3_moe_model.go), everything else uploads once at load time.
type hipQwen3MoEWeights struct {
	Embed     []float32 // host [VocabSize, HiddenSize] row-major — table lookup only
	Layers    []hipQwen3MoELayerWeights
	FinalNorm *hipDeviceByteBuffer // [D]
	LMHead    *hipDeviceByteBuffer // [VocabSize, D]
}

// Close releases every device buffer this weight set owns.
func (w *hipQwen3MoEWeights) Close() {
	if w == nil {
		return
	}
	for i := range w.Layers {
		w.Layers[i].Close()
	}
	if w.FinalNorm != nil {
		_ = w.FinalNorm.Close()
	}
	if w.LMHead != nil {
		_ = w.LMHead.Close()
	}
}

// hipQwen3MoETensorFloat32 decodes a named checkpoint tensor to a host float32 slice,
// dtype-agnostic (F32/F16/BF16 — safetensors.DecodeFloat32 resolves the checkpoint's
// own dtype).
func hipQwen3MoETensorFloat32(weights map[string]safetensors.Tensor, name string) ([]float32, error) {
	t, ok := weights[name]
	if !ok {
		return nil, core.NewError("rocm.hip.Qwen3MoE: tensor " + name + " absent")
	}
	elements := 1
	for _, d := range t.Shape {
		elements *= d
	}
	values, err := safetensors.DecodeFloat32(t.Dtype, t.Data, elements)
	if err != nil {
		return nil, core.E("rocm.hip.Qwen3MoE", "decode tensor "+name, err)
	}
	return values, nil
}

// hipQwen3MoEUploadFloat32 uploads a host float32 slice as an F32 device buffer that
// outlives the call — the caller owns the returned buffer and must Close it (via the
// owning hipQwen3MoEWeights/hipQwen3MoELayerWeights Close).
func hipQwen3MoEUploadFloat32(driver nativeHIPDriver, label string, values []float32) (*hipDeviceByteBuffer, error) {
	payload, err := hipFloat32Payload(values)
	if err != nil {
		return nil, core.E("rocm.hip.Qwen3MoE", "encode "+label, err)
	}
	buf, err := hipUploadByteBuffer(driver, "rocm.hip.Qwen3MoE", label, payload, len(values))
	if err != nil {
		return nil, core.E("rocm.hip.Qwen3MoE", "upload "+label, err)
	}
	return buf, nil
}

// hipQwen3MoEUploadNamedTensor decodes and uploads a named checkpoint tensor in one step.
func hipQwen3MoEUploadNamedTensor(driver nativeHIPDriver, weights map[string]safetensors.Tensor, name string) (*hipDeviceByteBuffer, error) {
	values, err := hipQwen3MoETensorFloat32(weights, name)
	if err != nil {
		return nil, err
	}
	return hipQwen3MoEUploadFloat32(driver, name, values)
}

// hipQwen3MoELayerTensorName builds the standard HF qwen3_moe per-layer tensor name.
func hipQwen3MoELayerTensorName(layer int, suffix string) string {
	return core.Sprintf("model.layers.%d.%s", layer, suffix)
}

// hipQwen3MoEExpertTensorName builds one routed expert's per-layer tensor name — the
// real (unpacked) per-expert HF layout: model.layers.{i}.mlp.experts.{e}.{role}_proj.weight
// (qwenmoe/weights.go's packExperts documents the same source layout; this v1 loader
// addresses experts individually instead of synthesising a packed tensor, matching
// hip_small_decode.go's plain-projection-per-call style rather than gemma4's batched
// byte-offset addressing — a deliberate v1 simplification, performance out of scope).
func hipQwen3MoEExpertTensorName(layer, expert int, role string) string {
	return core.Sprintf("model.layers.%d.mlp.experts.%d.%s_proj.weight", layer, expert, role)
}

// loadHIPQwen3MoEWeights resolves and uploads every tensor cfg's geometry names,
// standard HF qwen3_moe tensor layout (llama-shaped attention + q_norm/k_norm +
// post_attention_layernorm as the pre-MoE norm + per-expert gate/up/down + lm_head,
// tied to the embedding table when the checkpoint omits its own).
func loadHIPQwen3MoEWeights(driver nativeHIPDriver, weights map[string]safetensors.Tensor, cfg hipQwen3MoEConfig) (*hipQwen3MoEWeights, error) {
	embed, err := hipQwen3MoETensorFloat32(weights, "model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if len(embed) != cfg.VocabSize*cfg.HiddenSize {
		return nil, core.NewError("rocm.hip.Qwen3MoE: embedding tensor shape must be vocab*hidden")
	}
	out := &hipQwen3MoEWeights{Embed: embed, Layers: make([]hipQwen3MoELayerWeights, cfg.NumLayers)}
	success := false
	defer func() {
		if !success {
			out.Close()
		}
	}()

	qDim := cfg.Heads * cfg.HeadDim
	kvDim := cfg.KVHeads * cfg.HeadDim
	for layer := 0; layer < cfg.NumLayers; layer++ {
		lw := &out.Layers[layer]
		named := func(suffix string) (*hipDeviceByteBuffer, error) {
			return hipQwen3MoEUploadNamedTensor(driver, weights, hipQwen3MoELayerTensorName(layer, suffix))
		}
		if lw.InputNorm, err = named("input_layernorm.weight"); err != nil {
			return nil, err
		}
		if lw.QProj, err = named("self_attn.q_proj.weight"); err != nil {
			return nil, err
		}
		if lw.KProj, err = named("self_attn.k_proj.weight"); err != nil {
			return nil, err
		}
		if lw.VProj, err = named("self_attn.v_proj.weight"); err != nil {
			return nil, err
		}
		if lw.OProj, err = named("self_attn.o_proj.weight"); err != nil {
			return nil, err
		}
		if lw.QNorm, err = named("self_attn.q_norm.weight"); err != nil {
			return nil, err
		}
		if lw.KNorm, err = named("self_attn.k_norm.weight"); err != nil {
			return nil, err
		}
		if lw.PostAttnNorm, err = named("post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		if lw.Router, err = named("mlp.gate.weight"); err != nil {
			return nil, err
		}
		if lw.QProj.Count() != qDim*cfg.HiddenSize || lw.KProj.Count() != kvDim*cfg.HiddenSize || lw.VProj.Count() != kvDim*cfg.HiddenSize {
			return nil, core.NewError("rocm.hip.Qwen3MoE: attention projection shape mismatch at layer " + core.Sprintf("%d", layer))
		}
		lw.ExpertGate = make([]*hipDeviceByteBuffer, cfg.NumExperts)
		lw.ExpertUp = make([]*hipDeviceByteBuffer, cfg.NumExperts)
		lw.ExpertDown = make([]*hipDeviceByteBuffer, cfg.NumExperts)
		for e := 0; e < cfg.NumExperts; e++ {
			if lw.ExpertGate[e], err = hipQwen3MoEUploadNamedTensor(driver, weights, hipQwen3MoEExpertTensorName(layer, e, "gate")); err != nil {
				return nil, err
			}
			if lw.ExpertUp[e], err = hipQwen3MoEUploadNamedTensor(driver, weights, hipQwen3MoEExpertTensorName(layer, e, "up")); err != nil {
				return nil, err
			}
			if lw.ExpertDown[e], err = hipQwen3MoEUploadNamedTensor(driver, weights, hipQwen3MoEExpertTensorName(layer, e, "down")); err != nil {
				return nil, err
			}
		}
	}
	if out.FinalNorm, err = hipQwen3MoEUploadNamedTensor(driver, weights, "model.norm.weight"); err != nil {
		return nil, err
	}
	lmHeadName := "lm_head.weight"
	if _, ok := weights["lm_head.weight"]; !ok {
		out.LMHead, err = hipQwen3MoEUploadFloat32(driver, lmHeadName, embed)
	} else {
		out.LMHead, err = hipQwen3MoEUploadNamedTensor(driver, weights, lmHeadName)
	}
	if err != nil {
		return nil, err
	}
	if out.LMHead.Count() != cfg.VocabSize*cfg.HiddenSize {
		return nil, core.NewError("rocm.hip.Qwen3MoE: lm_head tensor shape must be vocab*hidden")
	}
	success = true
	return out, nil
}
