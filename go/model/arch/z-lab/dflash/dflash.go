// SPDX-Licence-Identifier: EUPL-1.2

// Package dflash owns the z-lab DFlash drafter checkpoint contract — config
// parsing and weight mapping for the block-diffusion speculative DRAFTER family
// z-lab publishes (z-lab/Qwen3-4B-DFlash-b16, z-lab/Qwen3-8B/27B/35B-A3B-DFlash,
// z-lab/gemma-4-26B-A4B-it-DFlash, ...; arXiv 2602.06036). The vendor directory
// matches the Hugging Face org, as arch/zai-org and arch/rednote-hilab do.
//
// A DFlash drafter is NOT a standalone text model: its checkpoint carries a
// 5-layer qwen3-style decoder plus the fused-context projection, and NO
// embedding, NO lm_head, NO tokenizer of its own — it borrows the TARGET
// model's tied embedding and head at serve time (the real z-lab convention;
// see docs/design-dflash-survey.md §4 and docs/design-dflash-forward.md §3).
// Accordingly this package deliberately registers NO model.ArchSpec: the
// checkpoint's model_type is literally "qwen3", which must keep resolving to
// the real qwen3 text arch, and loading a drafter as a primary model would be
// a misload. Marker recognition stays with the model-free contract
// (decode/dflash.ParseConfig — SPOR); this package adds what that contract
// does not carry: the drafter's own decoder geometry and the typed weight
// payload the engine forward consumes.
//
//	cfg, ok := dflash.ParseConfig(configBytes)   // z-lab convention recognised?
//	m, err := dflash.Load(dir)                   // config + safetensors → payload
//	// engine: native.DFlashZLabForward(m, noise, targetHidden, ctxLen, blockLen)
package dflash

import (
	core "dappco.re/go"
	decodedflash "dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

const (
	defaultRopeTheta  float32 = 1_000_000
	defaultRMSNormEps float32 = 1e-6
)

// Config is a z-lab DFlash drafter's full contract: the drafter-facing block
// parameters (delegated to decode/dflash.ParseConfig — block size, fused
// target layers, mask token, verifier) plus the drafter's OWN decoder geometry,
// which the z-lab convention declares flat qwen3-style at the top level
// (hidden_size, num_attention_heads, num_key_value_heads, head_dim,
// intermediate_size, num_hidden_layers, rms_norm_eps, rope_theta). NumAux is
// derived — len(Block.AuxHiddenLayerIDs) — because fc.weight's input width is
// NumAux*Hidden.
type Config struct {
	Block decodedflash.Config // block size, target_layer_ids, mask token, verifier

	Hidden       int     // hidden_size
	Heads        int     // num_attention_heads
	KVHeads      int     // num_key_value_heads
	HeadDim      int     // head_dim
	Intermediate int     // intermediate_size
	NumLayers    int     // num_hidden_layers (every published checkpoint: 5)
	Eps          float32 // rms_norm_eps (default 1e-6)
	RopeTheta    float32 // rope_theta (default 1e6)
}

// NumAux is the count of fused target layers — the factor by which fc.weight's
// input width exceeds Hidden. It is NOT a context length (docs/design-dflash-
// forward.md §2: context length is the number of target TOKENS fused in).
func (c Config) NumAux() int { return len(c.Block.AuxHiddenLayerIDs) }

// zlabConfigJSON is the decoder-geometry parse shape — the flat qwen3-style
// keys every published z-lab DFlash config.json carries beside its
// dflash_config block (evidenced from the real z-lab/Qwen3-4B-DFlash-b16
// config; see the survey). The speculators convention (RedHatAI) nests its
// decoder under transformer_layer_config instead and carries a reduced head —
// a DIFFERENT payload this package does not claim; its configs parse ok=false
// here and stay with the model-free contract's recognition only.
type zlabConfigJSON struct {
	ModelType         string  `json:"model_type"`
	HiddenSize        int     `json:"hidden_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
}

// ParseConfig recognises a z-lab-convention DFlash drafter from its config.json
// bytes: the DFlash marker must be present (decode/dflash.ParseConfig — the
// one recognition authority) AND the flat qwen3-style decoder geometry must be
// complete. A speculators-convention checkpoint (decoder nested under
// transformer_layer_config, no flat dims) returns ok=false — recognised as
// DFlash by the model-free contract, but not loadable by THIS package. Eps and
// rope_theta default (1e-6, 1e6) when absent; HeadDim falls back to
// Hidden/Heads when divisible.
//
//	if cfg, ok := dflash.ParseConfig(data); ok { /* a loadable z-lab drafter */ }
func ParseConfig(data []byte) (Config, bool) {
	block, isDFlash := decodedflash.ParseConfig(data)
	if !isDFlash {
		return Config{}, false
	}
	var raw zlabConfigJSON
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return Config{}, false
	}
	if raw.HiddenSize <= 0 || raw.NumAttentionHeads <= 0 || raw.NumHiddenLayers <= 0 {
		return Config{}, false // speculators nesting, or a broken export — not this package's shape
	}
	kv := raw.NumKeyValueHeads
	if kv <= 0 {
		kv = raw.NumAttentionHeads
	}
	headDim := raw.HeadDim
	if headDim <= 0 {
		if raw.HiddenSize%raw.NumAttentionHeads != 0 {
			return Config{}, false
		}
		headDim = raw.HiddenSize / raw.NumAttentionHeads
	}
	eps := raw.RMSNormEps
	if eps <= 0 {
		eps = defaultRMSNormEps
	}
	theta := raw.RopeTheta
	if theta <= 0 {
		theta = defaultRopeTheta
	}
	return Config{
		Block:  block,
		Hidden: raw.HiddenSize, Heads: raw.NumAttentionHeads, KVHeads: kv,
		HeadDim: headDim, Intermediate: raw.IntermediateSize,
		NumLayers: raw.NumHiddenLayers, Eps: eps, RopeTheta: theta,
	}, true
}

// DraftLayer is one of the drafter's qwen3-style decoder layers, widened to
// owned f32 — plain pre-norm (TWO norms), per-head q/k RMSNorm, GQA
// projections, SiLU-gated MLP. Weight matrices keep the checkpoint's PyTorch
// row-major [out, in] layout.
type DraftLayer struct {
	InputNorm    []float32 // layers.N.input_layernorm.weight            [hidden]
	PostAttnNorm []float32 // layers.N.post_attention_layernorm.weight   [hidden]
	Q            []float32 // layers.N.self_attn.q_proj.weight           [heads*headDim, hidden]
	K            []float32 // layers.N.self_attn.k_proj.weight           [kvHeads*headDim, hidden]
	V            []float32 // layers.N.self_attn.v_proj.weight           [kvHeads*headDim, hidden]
	O            []float32 // layers.N.self_attn.o_proj.weight           [hidden, heads*headDim]
	QNorm        []float32 // layers.N.self_attn.q_norm.weight           [headDim]
	KNorm        []float32 // layers.N.self_attn.k_norm.weight           [headDim]
	Gate         []float32 // layers.N.mlp.gate_proj.weight              [intermediate, hidden]
	Up           []float32 // layers.N.mlp.up_proj.weight                [intermediate, hidden]
	Down         []float32 // layers.N.mlp.down_proj.weight              [hidden, intermediate]
}

// DraftModel is the assembled drafter payload the engine forward consumes:
// geometry + every tensor the real checkpoint carries, owned host f32 (the
// mamba2/rwkv7/vision-tower posture — the source mmap is released after
// Assemble copies out of it). There is deliberately no Embed and no LMHead
// field: the real z-lab checkpoint has neither (borrowed from the target).
type DraftModel struct {
	Cfg        Config
	FC         []float32 // fc.weight           [hidden, numAux*hidden]
	HiddenNorm []float32 // hidden_norm.weight  [hidden]
	FinalNorm  []float32 // norm.weight         [hidden]
	Layers     []DraftLayer
}

// tensorF32 widens one safetensors tensor to owned f32 after validating its
// shape exactly — a mis-exported checkpoint fails loudly, naming the tensor,
// never silently zero-filling or truncating.
func tensorF32(tensors map[string]safetensors.Tensor, name string, want ...int) ([]float32, error) {
	t, ok := tensors[name]
	if !ok {
		return nil, core.NewError("dflash.Assemble: missing required tensor " + name)
	}
	if len(t.Shape) != len(want) {
		return nil, core.NewError(core.Sprintf("dflash.Assemble: tensor %s has %d dims, want %d", name, len(t.Shape), len(want)))
	}
	elements := 1
	for i, w := range want {
		if t.Shape[i] != w {
			return nil, core.NewError(core.Sprintf("dflash.Assemble: tensor %s shape %v, want %v", name, t.Shape, want))
		}
		elements *= w
	}
	out, err := safetensors.DecodeFloat32(t.Dtype, t.Data, elements)
	if err != nil {
		return nil, core.E("dflash.Assemble", "widen "+name, err)
	}
	return out, nil
}

// Assemble builds the typed DraftModel from a checkpoint's tensor map,
// validating every shape against cfg's geometry. Tensor names are EXACTLY the
// checkpoint's own (fc.weight, hidden_norm.weight, layers.N.*, norm.weight —
// no "model." or "dflash." prefix; those prefixes belong to other families).
// The presence of embed_tokens or lm_head is NOT required and NOT read.
func Assemble(tensors map[string]safetensors.Tensor, cfg Config) (*DraftModel, error) {
	if cfg.Hidden <= 0 || cfg.Heads <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 || cfg.Intermediate <= 0 || cfg.NumLayers <= 0 {
		return nil, core.NewError("dflash.Assemble: incomplete decoder geometry")
	}
	if cfg.Heads%cfg.KVHeads != 0 {
		return nil, core.NewError(core.Sprintf("dflash.Assemble: heads %d not a multiple of kv_heads %d (GQA)", cfg.Heads, cfg.KVHeads))
	}
	numAux := cfg.NumAux()
	if numAux <= 0 {
		return nil, core.NewError("dflash.Assemble: config declares no target_layer_ids (nothing to fuse)")
	}
	m := &DraftModel{Cfg: cfg}
	var err error
	if m.FC, err = tensorF32(tensors, "fc.weight", cfg.Hidden, numAux*cfg.Hidden); err != nil {
		return nil, err
	}
	if m.HiddenNorm, err = tensorF32(tensors, "hidden_norm.weight", cfg.Hidden); err != nil {
		return nil, err
	}
	if m.FinalNorm, err = tensorF32(tensors, "norm.weight", cfg.Hidden); err != nil {
		return nil, err
	}
	qDim, kvDim := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim
	m.Layers = make([]DraftLayer, cfg.NumLayers)
	for li := range m.Layers {
		p := core.Sprintf("layers.%d.", li)
		l := &m.Layers[li]
		if l.InputNorm, err = tensorF32(tensors, p+"input_layernorm.weight", cfg.Hidden); err != nil {
			return nil, err
		}
		if l.PostAttnNorm, err = tensorF32(tensors, p+"post_attention_layernorm.weight", cfg.Hidden); err != nil {
			return nil, err
		}
		if l.Q, err = tensorF32(tensors, p+"self_attn.q_proj.weight", qDim, cfg.Hidden); err != nil {
			return nil, err
		}
		if l.K, err = tensorF32(tensors, p+"self_attn.k_proj.weight", kvDim, cfg.Hidden); err != nil {
			return nil, err
		}
		if l.V, err = tensorF32(tensors, p+"self_attn.v_proj.weight", kvDim, cfg.Hidden); err != nil {
			return nil, err
		}
		if l.O, err = tensorF32(tensors, p+"self_attn.o_proj.weight", cfg.Hidden, qDim); err != nil {
			return nil, err
		}
		if l.QNorm, err = tensorF32(tensors, p+"self_attn.q_norm.weight", cfg.HeadDim); err != nil {
			return nil, err
		}
		if l.KNorm, err = tensorF32(tensors, p+"self_attn.k_norm.weight", cfg.HeadDim); err != nil {
			return nil, err
		}
		if l.Gate, err = tensorF32(tensors, p+"mlp.gate_proj.weight", cfg.Intermediate, cfg.Hidden); err != nil {
			return nil, err
		}
		if l.Up, err = tensorF32(tensors, p+"mlp.up_proj.weight", cfg.Intermediate, cfg.Hidden); err != nil {
			return nil, err
		}
		if l.Down, err = tensorF32(tensors, p+"mlp.down_proj.weight", cfg.Hidden, cfg.Intermediate); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// Load reads a z-lab DFlash drafter checkpoint directory (config.json +
// safetensors shards) into an assembled DraftModel. The shard mmap is released
// before return — Assemble widens every tensor to owned f32, so nothing views
// the mapping afterwards (the mamba2/rwkv7 load posture).
//
//	m, err := dflash.Load("/models/z-lab/Qwen3-4B-DFlash-b16")
func Load(dir string) (*DraftModel, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("dflash.Load", "read config.json", err)
	}
	cfg, ok := ParseConfig([]byte(cfgStr))
	if !ok {
		return nil, core.NewError("dflash.Load: config.json is not a z-lab-convention DFlash drafter (marker absent, or decoder geometry not flat qwen3-style)")
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("dflash.Load", "load weights", err)
	}
	defer func() { _ = dm.Close() }()
	m, err := Assemble(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return m, nil
}
