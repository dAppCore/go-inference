// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/mamba2"
	"dappco.re/go/inference/model/arch/rwkv7"
	"dappco.re/go/inference/model/safetensors"
)

// subTwoBitQuantDecline returns a typed refusal when the checkpoint's declared quantisation is a
// width no factory device path serves: the affine qmv kernels have no sub-2-bit instantiation
// (#24's open half), so a 1-bit pack (Bonsai) cannot decode. The retired composed engine used to
// serve these host-side (#50); refusing with the gap named beats a missing-kernel error deep in the
// head build. An absent/unparseable quantization block (bf16) is servable — nil.
func subTwoBitQuantDecline(cfg []byte) error {
	var q struct {
		Quantization struct {
			Bits int `json:"bits"`
		} `json:"quantization"`
	}
	if r := core.JSONUnmarshal(cfg, &q); !r.OK {
		return nil
	}
	if q.Quantization.Bits > 0 && q.Quantization.Bits < 2 {
		return core.NewError("native.LoadTokenModelDir: this checkpoint declares a sub-2-bit quant pack — the factory qmv kernels have no sub-2-bit width (#24) and the composed host lane that served it is retired (#50)")
	}
	return nil
}

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
	// KVCacheMode is the requested LIVE cache mode ("" / "native" = the built-in
	// bf16 cache; "turboquant[:N]" = TurboQuant codes on qualifying GLOBAL
	// attention owners — #41 S3). Unknown modes and unservable combinations
	// (paged KV, MoE/sinks archs, hybrid/recurrent archs) refuse at load rather
	// than running silently native.
	KVCacheMode string
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
	// TurboQuant live KV (#41 S3): parse the requested mode up front — an
	// unknown string refuses here — and refuse the combinations the v1 lane
	// declares out of scope, loudly, before any weights load.
	tqMode, tqErr := parseTurboQuantCacheMode(loadCfg.KVCacheMode)
	if tqErr != nil {
		return nil, tqErr
	}
	if tqMode != nil && loadCfg.PagedKVPageSize != 0 {
		return nil, core.NewError("native.LoadTokenModelDir: -kv-cache turboquant declines paged KV (v1) — drop one of the two")
	}
	usedDefaultContext := maxLen <= 0
	if usedDefaultContext {
		// Serve/generate without -context used to cap every session at 4096 —
		// the resident-conversation killer (#370's book bench died mid-book).
		// Default to the checkpoint's trained window instead, capped.
		maxLen = resolveDefaultContext(model.ProbeDirContextWindow(dir))
	}
	// Standalone recurrent SSMs don't fit the reactive transformer Assemble — route them to their
	// own loaders before the registered path. (They keep the plain window-capped default: their
	// state geometry isn't the transformer KV plan the RAM-aware clamp below budgets.)
	if mt, cfg, perr := model.ProbeDirArch(dir); perr == nil {
		if mt == "mamba2" {
			if tqMode != nil {
				return nil, core.NewError("native.LoadTokenModelDir: -kv-cache turboquant declines recurrent archs (mamba2 holds SSM state, not a KV cache) — reload without -kv-cache")
			}
			// mamba2 is a standalone recurrent SSM with its own loader — it registers no
			// ArchSpec, so it is reached by name here.
			return loadMamba2TokenModel(dir, cfg)
		}
		if mt == "rwkv7" {
			if tqMode != nil {
				return nil, core.NewError("native.LoadTokenModelDir: -kv-cache turboquant declines recurrent archs (rwkv7 carries per-layer state, not a KV cache) — reload without -kv-cache")
			}
			// rwkv7 is a standalone recurrent SSM (token-shift + WKV7 time-mix +
			// channel-mix, no attention) with its own loader — see model/arch/rwkv7.
			// It registers no ArchSpec, so it is reached by name here.
			return loadRWKV7TokenModel(dir, cfg)
		}
		// Every registered arch loads through the ONE factory route below (model.Load: parse →
		// infer → derive → assemble). The retired composed engine's registry detour
		// (model.LoadComposedDir) is gone (#50): qwen3_5/3.6 hybrids ride the factory's fused
		// whole-token chain (#18 — faster than the composed lane was on both released hybrids),
		// and a checkpoint the factory cannot serve fails HERE with a named error instead of
		// falling back silently. Known capability gaps are declined by name: sub-2-bit packs
		// (below) and the qwen vision tower (the factory serves the TEXT stack; an image turn
		// on a vision-towered qwen checkpoint gets the engine's clean not-a-vision-model refusal).
		if derr := subTwoBitQuantDecline(cfg); derr != nil {
			return nil, derr
		}
	}
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		return nil, err
	}
	if tqMode != nil {
		if terr := tqKVArchServable(lm.Arch, lm.Layers, tqMode); terr != nil {
			_ = dm.Close()
			return nil, terr
		}
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
		if lm.Audio != nil {
			tm.audioExtractor = buildAudioExtractor(dir)
		}
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
	if lm.Audio != nil {
		tm.audioExtractor = buildAudioExtractor(dir)
	}
	tm.diffusion = lm.Diffusion
	return tm, nil
}

// buildAudioExtractor builds the host mel front-end for a Conformer audio tower from the model
// directory's processor_config.json. Best-effort, mirroring visionFeatureCfg: a missing or malformed
// config returns nil (audio disabled, ProjectAudio then errors clearly) and never blocks the load. The
// disable is traced so a broken front-end is diagnosable.
func buildAudioExtractor(dir string) *AudioFeatureExtractor {
	cfg, err := LoadAudioFeatureConfig(dir)
	if err != nil || cfg == nil {
		nativeTraceLog(core.Sprintf("audio: feature extractor disabled (config: %v)\n", err))
		return nil
	}
	extractor, err := NewAudioFeatureExtractor(cfg)
	if err != nil {
		nativeTraceLog(core.Sprintf("audio: feature extractor disabled (build: %v)\n", err))
		return nil
	}
	return extractor
}

func nativeTokenModelBackendOptions(cfg TokenModelLoadConfig) []BackendOption {
	var opts []BackendOption
	if cfg.PagedKVPageSize != 0 {
		opts = append(opts, withPagedKVPageSize(cfg.PagedKVPageSize))
	}
	if cfg.PagedKVPrealloc {
		opts = append(opts, withPagedKVPrealloc(true))
	}
	if core.Trim(cfg.KVCacheMode) != "" {
		opts = append(opts, withKVCacheMode(cfg.KVCacheMode))
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

// loadRWKV7TokenModel loads an rwkv7 checkpoint into the host-f32 recurrent RWKV7Model and wraps it as a
// model.SessionModel — the mamba2 shape exactly: weights widen to owned f32 at load, the shard mmap is
// unmapped before return. Dense/LoRA projections ride rwkv7.ProjMatMul, already wired to the steel GEMM
// (rwkv7_backend.go), so the host chain gets device matmuls with no further engine wiring.
func loadRWKV7TokenModel(dir string, cfg []byte) (model.TokenModel, error) {
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, err
	}
	defer func() { _ = dm.Close() }()
	rm, err := rwkv7.LoadRWKV7Model(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return rwkv7.NewTokenModel(rm), nil
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
