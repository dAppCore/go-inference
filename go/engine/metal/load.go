// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/mamba2"
	"dappco.re/go/inference/model/safetensors"
)

// load.go is the native backend's directory loader: it delegates to the engine's reactive loader
// (model.Load) — read config.json, probe model_type, react to the registered ArchSpec (parse → infer →
// derive → assemble) — then turns the neutral model.LoadedModel into the native decode structs. The
// backend holds no per-model knowledge: a model package (pkg/model/gemma4, /mistral, …) owns its config
// + weight-name declaration and registers it from init(); adding an arch is a new package + an init(),
// no edit here. The generic loadedToBF16/loadedToQuant build the native decode structs (quant vs bf16
// read from the loaded weights, not a re-parse of the config).

// LoadDir loads any registered architecture's checkpoint directory into a persistent decode session
// with maxLen cache rows — the one-call path from an on-disk checkpoint to a ready-to-Generate
// ArchSession. Zero-copy: the weights view the shard mmap, held on the session via shardBuffers for
// its life (Close unmaps).
func LoadDir(dir string, maxLen int) (*ArchSession, error) {
	usedDefaultContext := maxLen <= 0
	if usedDefaultContext {
		// The loader owns the context default (checkpoint window, capped) — the
		// speculative pair serve passes an unset -context straight through here
		// and NewArch*Session rejects 0 (the post-601ac4e pair-serve break).
		maxLen = resolveDefaultContext(model.ProbeDirContextWindow(dir))
	}
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		return nil, err
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		_ = dm.Close()
		return nil, err
	}
	if usedDefaultContext { // exact geometry + weight bytes in hand: fit the default to the box
		maxLen = clampDefaultContextToRAM(maxLen, lm.Arch, sb)
	}
	var sess *ArchSession
	if quantised(lm) {
		qm, qerr := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
		if qerr != nil {
			_ = sb.Close()
			return nil, qerr
		}
		sess, err = newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	} else {
		sess, err = newArchSessionShards(loadedToBF16(lm), lm.Arch, maxLen, sb)
	}
	if err != nil {
		_ = sb.Close()
		return nil, err
	}
	sess.shards = sb
	return sess, nil
}

// probeModelType reads the checkpoint's declared architecture — config.json's
// model_type — so a loaded model self-reports its real arch (gemma4, gemma3,
// llama, …) instead of a hardcoded stamp. It falls back to "gemma4" only when
// the probe fails or the config declares nothing, keeping a broken-config load
// no worse than the historical default. This is the truthful source
// inference.ModelInfo.Architecture / ModelType report for ANY architecture the
// native engine loads.
func probeModelType(dir string) string {
	mt, _, err := model.ProbeDirArch(dir)
	if err != nil || mt == "" {
		return "gemma4"
	}
	return mt
}

type TokenModelLoadConfig struct {
	PagedKVPageSize int
	PagedKVPrealloc bool
	// AdapterPath, when set, is a saved LoRA adapter directory (adapter.safetensors +
	// adapter_config.json) applied at load so `serve --adapter <path>` generates through the adapted
	// model. Honoured for the bf16 head adapter (LoRATrainer.Save); see lora_apply.go.
	AdapterPath string
}

// LoadTokenModelDir loads any registered architecture's checkpoint directory as a model.TokenModel —
// the backend-agnostic token-loop contract the serve adapter drives (model.Generate over the returned
// SessionModel). The quant/bf16 path is read from the loaded weights; the per-token serve head is
// built once (buildHeadEncoder) to kill the per-token re-upload.
func LoadTokenModelDir(dir string, maxLen int) (model.TokenModel, error) {
	return LoadTokenModelDirWithConfig(dir, maxLen, TokenModelLoadConfig{})
}

// defaultContextCap bounds the checkpoint-window context default at 256K —
// the #367 target stance (MTP + q8 + 256K as THE default). The first 256K
// attempt swapped a 96GB box (31B all-defaults, 64.9GB footprint) and was
// pulled back to 128K the same day; the three gremlins behind it are fixed —
// the duplicate lb KV set (13ffe18), the batch-scratch slab leaks (14fa6aa)
// and the missing RAM budget (fad5212: clampContextToRAM now fits an unset
// -context to the box, so smaller machines land below this cap naturally).
// An explicit -context overrides in both directions.
const defaultContextCap = 262144

// resolveDefaultContext maps an unset context length to the checkpoint
// window capped at defaultContextCap, keeping the old 4096 floor when the
// config declares no window.
func resolveDefaultContext(window int) int {
	if window <= 0 {
		return 4096
	}
	return min(window, defaultContextCap)
}

func LoadTokenModelDirWithConfig(dir string, maxLen int, loadCfg TokenModelLoadConfig) (model.TokenModel, error) {
	if loadCfg.PagedKVPageSize < 0 {
		return nil, core.NewError("native.LoadTokenModelDir: paged KV page size must be >= 0")
	}
	usedDefaultContext := maxLen <= 0
	if usedDefaultContext {
		// Serve/generate without -context used to cap every session at 4096 —
		// the resident-conversation killer (#370's book bench died mid-book).
		// Default to the checkpoint's trained window instead, capped.
		maxLen = resolveDefaultContext(model.ProbeDirContextWindow(dir))
	}
	// SSM / hybrid families don't fit the reactive transformer Assemble — route them to their own loader
	// before the registered path. mamba2 is a standalone recurrent SSM; qwen3_5/3.6 is a config-composed
	// hybrid (linear_attention gated-delta + full attention) built by the composed loader.
	// (They keep the plain window-capped default: their state geometry isn't the
	// transformer KV plan the RAM-aware clamp below budgets.)
	if mt, cfg, perr := model.ProbeDirArch(dir); perr == nil {
		if mt == "mamba2" {
			// mamba2 is a standalone recurrent SSM with its own loader — it registers no
			// ArchSpec, so it is reached by name here rather than through the composed registry.
			return loadMamba2TokenModel(dir, cfg)
		}
		// Composed/hybrid archs route through the neutral registry: each registers an
		// ArchSpec.Composed hook, and model.LoadComposedDir looks it up and builds the
		// serve-ready TokenModel. A future qwenX checkpoint therefore loads with ZERO edits
		// here — it is a new model-package init(), not an engine change. ok=false means the
		// model_type is a plain transformer (or a composed type not yet registered): fall through.
		if tm, ok, cerr := model.LoadComposedDir(dir); cerr != nil {
			return nil, cerr
		} else if ok {
			return tm, nil
		}
		// Fallback for the historical composed model_types the registry does not (yet)
		// cover, so every type the old hardcoded switch loaded still loads identically.
		switch mt {
		case "qwen3_6", "qwen3_6_moe", "composed", "hybrid":
			return loadComposedTokenModel(dir, cfg)
		}
	}
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		return nil, err
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		_ = dm.Close()
		return nil, err
	}
	if usedDefaultContext { // exact geometry + weight bytes in hand: fit the default to the box
		maxLen = clampDefaultContextToRAM(maxLen, lm.Arch, sb)
	}
	arch := lm.Arch
	backendOpts := nativeTokenModelBackendOptions(loadCfg)
	if quantised(lm) {
		if loadCfg.AdapterPath != "" {
			_ = sb.Close()
			return nil, core.NewError("native.LoadTokenModelDir: load-time adapter apply is only wired for bf16 models, not the quantised head")
		}
		gs, bits := lm.Embed.GroupSize, lm.Embed.Bits
		g, qerr := loadedToQuant(lm, gs, bits)
		if qerr != nil {
			_ = sb.Close()
			return nil, qerr
		}
		tm, terr := NewQuantTokenModel(g, arch, maxLen, backendOpts...)
		if terr != nil {
			_ = sb.Close()
			return nil, terr
		}
		tm.shards = sb
		he, herr := buildHeadEncoder(sb, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
		if herr != nil {
			_ = sb.Close()
			return nil, herr
		}
		tm.headEnc = he
		tm.vision = lm.Vision
		tm.unifiedVision = lm.UnifiedVision
		if lm.Vision != nil || lm.UnifiedVision != nil {
			// Best-effort: absent/malformed processor config leaves the cfg nil and
			// ProjectImage falls back to HF defaults, so it never fails the load.
			tm.visionFeatureCfg, _ = LoadVisionImageFeatureConfig(dir)
		}
		tm.audio = lm.Audio
		tm.diffusion = lm.Diffusion
		return tm, nil
	}
	g := loadedToBF16(lm)
	if loadCfg.AdapterPath != "" {
		// Apply the saved adapter before the head encoder binds the head weight: the fold clones the
		// head (leaving the base mmap / tied embedding intact) so generation runs through the adapted head.
		if aerr := applyAdapterToBF16Model(g, loadCfg.AdapterPath); aerr != nil {
			_ = sb.Close()
			return nil, core.E("native.LoadTokenModelDir", "apply adapter", aerr)
		}
	}
	tm, terr := NewBF16TokenModel(g, arch, maxLen, backendOpts...)
	if terr != nil {
		_ = sb.Close()
		return nil, terr
	}
	tm.shards = sb
	he, herr := buildHeadEncoder(sb, g.FinalNorm, g.LMHead, nil, nil, arch.Hidden, arch.Vocab, 0, 0, arch.Eps, arch.SoftCap, false)
	if herr != nil {
		_ = sb.Close()
		return nil, herr
	}
	tm.headEnc = he
	tm.vision = lm.Vision
	tm.unifiedVision = lm.UnifiedVision
	if lm.Vision != nil || lm.UnifiedVision != nil {
		// Best-effort: absent/malformed processor config leaves the cfg nil and
		// ProjectImage falls back to HF defaults, so it never fails the load.
		tm.visionFeatureCfg, _ = LoadVisionImageFeatureConfig(dir)
	}
	tm.audio = lm.Audio
	tm.diffusion = lm.Diffusion
	return tm, nil
}

func nativeTokenModelBackendOptions(cfg TokenModelLoadConfig) []BackendOption {
	var opts []BackendOption
	if cfg.PagedKVPageSize != 0 {
		opts = append(opts, withPagedKVPageSize(cfg.PagedKVPageSize))
	}
	if cfg.PagedKVPrealloc {
		opts = append(opts, withPagedKVPrealloc(true))
	}
	return opts
}

// loadMamba2TokenModel loads a mamba2 checkpoint into the host-f32 recurrent MambaModel and wraps it as a
// model.SessionModel. LoadMambaModel widens every weight to its own f32 slices, so the shard mmap is only
// needed during the load and is unmapped before return (no shardBuffers held). Host f32 today — correct
// and the SSM scaffold; a device path (GPU GEMM for the projections, the bench-flagged hot spot) is the
// perf follow-up.
func loadMamba2TokenModel(dir string, cfg []byte) (model.TokenModel, error) {
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, err
	}
	defer func() { _ = dm.Close() }()
	mm, err := mamba2.LoadMambaModel(dm.Tensors, mamba2EpsFromConfig(cfg))
	if err != nil {
		return nil, err
	}
	return mamba2.NewTokenModel(mm), nil
}

// mamba2EpsFromConfig reads rms_norm_eps from the checkpoint config (top-level or nested text_config),
// defaulting to 1e-5 (the mamba2 default) when absent.
func mamba2EpsFromConfig(cfg []byte) float32 {
	var probe struct {
		Eps        float32 `json:"rms_norm_eps"`
		TextConfig struct {
			Eps float32 `json:"rms_norm_eps"`
		} `json:"text_config"`
	}
	_ = core.JSONUnmarshal(cfg, &probe)
	switch {
	case probe.Eps > 0:
		return probe.Eps
	case probe.TextConfig.Eps > 0:
		return probe.TextConfig.Eps
	default:
		return 1e-5
	}
}

// loadComposedTokenModel loads a config-composed hybrid checkpoint (Qwen 3.6) into the host-f32
// ComposedModel and wraps it as a model.SessionModel. LoadComposed widens every weight to f32, so the
// shard mmap is unmapped before return. Host f32 today (correct, the orchestration scaffold); a device
// path (the projections already have a GEMM seam; attention is a later device kernel) is the perf follow-up.
func loadComposedTokenModel(dir string, cfg []byte) (model.TokenModel, error) {
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, err
	}
	defer func() { _ = dm.Close() }()
	cm, err := composed.LoadComposed(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return composed.NewTokenModel(cm), nil
}

// loadRegistered delegates to the reactive engine loader (model.Load): probe model_type → the registered
// ArchSpec → parse / infer-from-weights / derive Arch / assemble. The shared front half of every directory
// load; returns the neutral LoadedModel + the DirMapping its byte views reference (the caller binds device
// buffers from it, then it lives on the session/token-model via shardBuffers). The backend holds no
// per-arch knowledge — the model package owns its config + weight-name declaration, model.Load reacts.
func loadRegistered(dir string) (*model.LoadedModel, *safetensors.DirMapping, error) {
	return model.Load(dir)
}

// quantised reports whether the loaded model's weights are quantised — read from the assembled
// embedding (it carries scales) rather than re-parsing the config's quant block. The model-wide
// group-size/bits the native quant structs use come from the same weight (uniform across the pack).
func quantised(m *model.LoadedModel) bool {
	return m != nil && m.Embed != nil && m.Embed.Quantised()
}

// buildHeadEncoder wraps newHeadEncoder in an autorelease pool — the 4-bit head uploads its weight
// once into retained owned buffers, which must be created inside a pool (they survive it, retained).
// The shared constructor for the directory token-model loaders.
func buildHeadEncoder(sb *shardBuffers, finalNormW, weight, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32, quant bool) (*headEncoder, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	var he *headEncoder
	var err error
	withAutoreleasePool(func() {
		he, err = newHeadEncoder(sb, finalNormW, weight, scales, biases, dModel, vocab, groupSize, bits, eps, softCap, quant)
	})
	return he, err
}

// buildShardBuffers wraps each shard's page-aligned mmap in a no-copy Metal buffer inside an
// autorelease pool (the buffers are objc-retained, so they survive the pool — the Go reference on
// the returned shardBuffers keeps them alive). The shared constructor for both directory holders.
func buildShardBuffers(dm *safetensors.DirMapping) (*shardBuffers, error) {
	var sb *shardBuffers
	var err error
	withAutoreleasePool(func() {
		sb, err = newShardBuffers(dm)
	})
	return sb, err
}
