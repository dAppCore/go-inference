// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/filestore"
	"dappco.re/go/inference/model/state/ramspill"
	adminpkg "dappco.re/go/inference/serving/admin"
	"dappco.re/go/inference/serving/compat"
	"dappco.re/go/inference/serving/policy"
	openai "dappco.re/go/inference/serving/provider/openai"
)

// Resolver resolves an OpenAI-API `model` field to a loaded inference.TextModel
// for the compatibility mux. It is the provider/openai.Resolver, re-exported so
// serve callers depend on one package.
type Resolver = openai.Resolver

// ContinuityEnabler attaches the no-prompt-replay conversation loop to a freshly
// loaded model, backed by store. It is injected by the composition root so the
// serving library carries no engine coupling: an engine whose TextModel supports
// continuity supplies an enabler; an engine that doesn't leaves it nil and serve
// degrades to stateless with an honest notice. Errors degrade too — continuity
// never blocks the serve from coming up.
type ContinuityEnabler func(model inference.TextModel, store state.Store) error

// ServeConfig is the declarative serve request: the model + listen surface plus
// the reactive-drafter, tuned-profile, conversation-continuity and admin knobs
// that lthn-mlx's serve command carried. RunServe turns it into the assembled
// HTTP server. cmd/lem builds this from flags and calls RunServe — the serve
// business logic (draft resolution, profile resolution, hot-swap, continuity)
// lives here in the library, not in cmd/.
type ServeConfig struct {
	Addr        string                 // listen address (e.g. ":36911")
	ModelPath   string                 // model to load; "" starts model-less (reload later)
	ContextLen  int                    // context-length override (0 = model default); reported by /v1/admin/serve/status
	LoadOptions []inference.LoadOption // adapter, slots, … forwarded to the loader (context is applied from ContextLen)

	// Reactive MTP drafter (Gemma 4 targets).
	DraftPath     string // "auto" runs the ladder, "" disables, a path forces the drafter
	DraftDetect   bool   // reactive detection for Gemma 4 targets
	DraftBlock    int    // explicit MTP draft block; 0 = tuned profile or engine default
	NoAutoProfile bool   // ignore tuned profiles
	ProfileDir    string // tuned-profile directory (default ~/Lethean/lem/tuning)
	MachineHash   string // machine identity for profile matching; "" accepts hash-less profiles

	KVCacheMode string // requested KV cache mode; wired when the engine exposes it
	Native      bool   // no-cgo native path (the default go-inference metal engine already is)

	// Scheduler routes requests through a mode-selected request scheduler
	// (serving/scheduler) between the HTTP handlers and the model. Empty (the
	// default) leaves the request path byte-for-byte unchanged — NO scheduler
	// is built and nothing new runs on the hot path. "serial" / "batch" /
	// "interleave" select the discipline; an unknown value fails closed at boot.
	// Single-model serve builds ONE scheduler shared by every resolve
	// (schedulerResolver). Multi-model serve (Models non-empty) builds one
	// scheduler instance PER RESIDENT model instead — created when that model
	// loads, closed when it evicts or unloads (see multiModelResolver's
	// schedCfg/ensureResident/evict) — since each resident model is a
	// different underlying inference.TextModel with its own lifecycle. See
	// docs/design-continuous-batching.md for the design rationale.
	Scheduler string

	// Multi-model serving lifecycle. Models empty = today's single-model
	// behaviour, unchanged. When Models is non-empty, RunServe builds the
	// multiModelResolver instead of the single-model hot-swap: ModelPath (the
	// -model flag) becomes the pinned default, the extra Models load on demand,
	// and residency is bounded by MemoryCeiling with LRU + IdleTTL eviction.
	Models        []ModelSpec   // additional models beyond ModelPath (id, path, aliases, profiles, pin)
	MemoryCeiling uint64        // max resident bytes across all models; 0 = unbounded
	IdleTTL       time.Duration // evict a model idle longer than this (unpinned only); 0 = never
	SweepInterval time.Duration // background idle-sweep tick; 0 = auto (min(IdleTTL, 30s))

	// Conversation continuity.
	StateConversations bool   // wake each chat from its slept state, no prompt replay
	StateStorePath     string // durable per-project store file; empty = ephemeral default, wiped each serve run
	// StateRAMBudget caps resident bytes in the RAM conversation store (only
	// meaningful when StateStorePath is unset — an explicit durable store is
	// already disk-backed). <= 0 is unlimited, byte-identical to pre-#48
	// behaviour. Over budget, the coldest chunks spill to a scratch .kv file
	// (model/state/ramspill) and wake back transparently on the next turn.
	StateRAMBudget int64

	// Welfare guard (go/welfare): per-turn detect + engine-model mediation on
	// every chat route; lem_end is additionally offered to Lemma checkpoints.
	// cmd/lem defaults this ON (-welfare=false disables); the zero value keeps
	// library constructions unguarded.
	Welfare bool

	// Outbound policy (serving/policy): a deployment-owned filter on model
	// OUTPUT — term/pattern rules with redact/refuse actions, loaded from a JSON
	// file. Empty disables the layer (zero overhead). It is composed OUTERMOST
	// on output, after the welfare guard, so it enforces on the final tokens the
	// deployment would otherwise emit. A load failure at boot is FATAL.
	PolicyPath string

	// PolicyMediator is the grade-G2 rewrite hook (serving/policy): a rewrite rule
	// routes its matched span through this to transform rather than redact it. nil
	// suits any redact/refuse-only policy; a policy that NeedsMediator loaded with
	// a nil mediator is FATAL at boot (a rewrite would otherwise silently degrade
	// to redact — the deployer who asked for mediation must get it or none).
	PolicyMediator policy.Mediator

	// EmbedModelPath is an optional bert/BGE-class host encoder snapshot
	// directory (model/bert) served ALONGSIDE — or, with ModelPath empty,
	// INSTEAD OF — the chat model. Empty (the default) changes nothing:
	// /v1/embeddings and /v1/rerank keep serving only whatever the resolved
	// chat model itself implements (a clean 400 on every shipped engine today
	// — see serving/provider/openai/services.go). A load failure here is
	// FATAL at boot, matching the outbound-policy and admin-token pattern: a
	// deployer who asked for an embeddings model gets one or an honest
	// refusal to serve, never a silent fallback to "no embeddings model".
	EmbedModelPath string
	// EmbedModelID is the request-facing `model` name that routes to
	// EmbedModelPath; empty derives the pack's basename (matching how the
	// chat model's own default id is derived — see serveHost.listModels).
	EmbedModelID string

	// HTTP + admin.
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	ShutdownTimeout time.Duration
	AdminToken      string    // Bearer token for /v1/admin/*; "" leaves the subtree open
	Log             io.Writer // boot notices + admin auth-deny audit (cmd passes os.Stderr)

	// Injected engine seams (nil = use the registered-metal defaults / degrade).
	Loader            ModelLoader       // overrides the default metal loader
	SpeculativeLoader SpeculativeLoader // target+draft pair loader; nil = no speculative path
	EnableContinuity  ContinuityEnabler // conversation-continuity attach; nil = stateless
}

// RunServe assembles and runs the serve command's server from cfg: it resolves
// the reactive drafter, the tuned draft block, and conversation continuity, then
// stages a hot-swap resolver and hosts the OpenAI/Anthropic/Ollama-compatible
// HTTP API. It blocks until ctx is cancelled or the listener fails. This is the
// serve business logic ported out of lthn-mlx's cmd/mlx so cmd/lem stays thin.
func RunServe(ctx context.Context, cfg ServeConfig) error {
	log := cfg.Log

	// Outbound policy — loaded and compiled up front so a misconfigured
	// deployment fails at BOOT, never mid-serve: a deployer who set -policy must
	// never silently serve without the filter they asked for.
	outboundPolicy, err := loadOutboundPolicy(cfg)
	if err != nil {
		return err
	}

	// Multi-model serving is a separate host path built on the multiModelResolver;
	// the single-model hot-swap below is untouched and stays the default.
	if len(cfg.Models) > 0 {
		return runMultiModelServe(ctx, cfg, outboundPolicy, log)
	}

	// Reactive MTP pair resolution (the model declares, the serve reacts): an
	// explicit --draft path wins; --draft="" disables; "auto" runs the ladder.
	detection := ResolveServeDraft(cfg.ModelPath, cfg.DraftPath, cfg.DraftDetect)
	// A detected drafter is only armed when the registered engine exposes a
	// speculative loader; otherwise degrade to plain autoregressive with an
	// honest notice, faithful to lthn-mlx (a drafter that can't load never
	// blocks the serve — it surfaces the failure rather than refusing to boot).
	// A DFlash block-diffusion drafter is recognised but never armed on the
	// autoregressive MTP lane — the engine has no block-diffusion draft forward
	// yet, so it degrades to plain autoregressive with a specific honest notice
	// rather than misloading the checkpoint (docs/design-dflash.md).
	armDrafter := detection.Active() && cfg.SpeculativeLoader != nil && !detection.IsDFlash()
	draftPathForResolver := ""
	if armDrafter {
		draftPathForResolver = detection.DraftPath
	} else if detection.IsDFlash() {
		printServe(log, "serve: %s", DFlashDraftNotice(detection))
	} else if detection.Active() {
		printServe(log, "serve: drafter %s detected but this engine exposes no speculative path — serving plain autoregressive", detection.DraftPath)
	}

	resolvedBlock, blockNote := resolveServeDraftBlock(detection, cfg.ModelPath, cfg.DraftBlock, cfg.NoAutoProfile, cfg.ProfileDir, cfg.MachineHash)

	loadOpts := append([]inference.LoadOption(nil), cfg.LoadOptions...)
	if cfg.ContextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(cfg.ContextLen))
	}
	hotSwap := newHotSwapResolver(cfg.ModelPath, draftPathForResolver, resolvedBlock, loadOpts)
	hotSwap.setLoader(cfg.Loader) // nil-safe: keeps the registered-metal default
	hotSwap.setSpeculativeLoader(cfg.SpeculativeLoader)
	// Reload symmetry: each /v1/admin/serve/reload re-runs the reactive ladder
	// over the swapped-in target, honouring the boot --draft-detect choice.
	hotSwap.setDraftDetect(DraftDetectOptions{Disabled: !cfg.DraftDetect})

	if cfg.Native {
		printServe(log, "serve: native no-cgo engine (the default go-inference metal path)")
	}
	if core.Trim(cfg.KVCacheMode) != "" {
		printServe(log, "serve: -kv-cache %s requested; the registered engine loads its default cache mode (per-engine cache-mode selection lands with the engine load-config surface)", cfg.KVCacheMode)
	}
	if cfg.ModelPath == "" {
		printServe(log, "serve: starting model-less — POST /v1/admin/serve/reload to load a model")
	}

	// Conversation continuity — on by default. Any failure here degrades to
	// stateless serving with an honest notice; it never blocks the serve. The
	// cleanup closes the store at shutdown and clears the DEFAULTED store's
	// file (per-run scratch); an explicit -state-store survives as the durable
	// per-project state.
	if cfg.StateConversations {
		defer wireContinuity(ctx, cfg, hotSwap.setOnLoad, log)()
	}

	if armDrafter {
		if notice := speculativeServeNotice(detection, resolvedBlock); notice != "" {
			if blockNote != "" {
				notice += " [" + blockNote + "]"
			}
			printServe(log, "serve: %s", notice)
		}
	}
	printServe(log, "serve: listening on %s (model=%s)", cfg.Addr, cfg.ModelPath)

	// Single-model host: /v1/models advertises the loaded model's basename,
	// /v1/health reports its full path, and the hot-swap resolver drives
	// serve/status + serve/reload. No multi-model control plane is mounted.
	host := serveHost{
		resolver:    hotSwap.openaiResolver(),
		currentPath: hotSwap.CurrentPath,
		setOnLoad:   hotSwap.setOnLoad,
		reloader:    hotSwap,
		listModels: func() []string {
			if p := hotSwap.CurrentPath(); p != "" {
				return []string{core.PathBase(p)}
			}
			return nil
		},
		healthModels: func() []string {
			// Report the currently-loaded model (post-reload), or no models when
			// the driver started model-less and none has been loaded yet.
			models := []string{}
			if p := hotSwap.CurrentPath(); p != "" {
				models = append(models, p)
			}
			return models
		},
		status: adminpkg.ServeStatus{
			ModelPath:    cfg.ModelPath,
			Runtime:      "metal",
			LoadedAtUnix: time.Now().Unix(),
			Config: adminpkg.ServeStatusConfig{
				ContextLength: cfg.ContextLen,
				CacheMode:     cfg.KVCacheMode,
			},
		},
	}

	// Optional request scheduler — wraps the resolver so every request routes
	// through the mode's scheduler. Unset leaves host.resolver exactly as built
	// above (the request path is byte-for-byte unchanged; nothing is built).
	if core.Trim(cfg.Scheduler) != "" {
		mode, err := parseSchedulerMode(cfg.Scheduler)
		if err != nil {
			return err
		}
		sched := newSchedulerResolver(host.resolver, schedulerServeConfig(mode))
		host.resolver = sched
		defer sched.close()
		printServe(log, "serve: scheduler %s — requests route through the %s scheduler between the HTTP handlers and the model", mode, mode)
	}

	return hostServe(ctx, cfg, host, outboundPolicy, log)
}

// serveHost bundles the resolver and per-mode wiring hostServe needs to stand up
// the HTTP surface, so the single-model hot-swap and the multi-model registry
// converge on one host path. controller is nil in single-model mode (no
// /v1/admin/models routes); listModels/healthModels differ per mode (single-model
// advertises the one model, multi-model the registry).
type serveHost struct {
	resolver     Resolver                        // base openai.Resolver, pre welfare/policy
	currentPath  func() string                   // active/default model path (status + welfare gate)
	setOnLoad    func(func(inference.TextModel)) // continuity hook registrar (wired by the caller)
	reloader     adminpkg.Reloader               // serve/status + serve/reload
	controller   adminpkg.ModelController        // multi-model control plane; nil = single-model
	listModels   func() []string                 // /v1/models ids
	healthModels func() []string                 // /v1/health resident models
	status       adminpkg.ServeStatus            // boot status snapshot
}

// loadOutboundPolicy compiles the -policy file up front so a misconfigured
// deployment fails at BOOT, never mid-serve. Empty path = no policy (nil).
func loadOutboundPolicy(cfg ServeConfig) (*policy.Policy, error) {
	if core.Trim(cfg.PolicyPath) == "" {
		return nil, nil
	}
	pol, err := policy.Load(cfg.PolicyPath)
	if err != nil {
		return nil, core.E("serving.RunServe", core.Sprintf("outbound policy %q — refusing to serve unguarded", cfg.PolicyPath), err)
	}
	if pol.NeedsMediator() && cfg.PolicyMediator == nil {
		return nil, core.E("serving.RunServe", core.Sprintf("outbound policy %q declares rewrite rules but no mediator is wired — refusing to serve (a rewrite would silently degrade to redact)", cfg.PolicyPath), nil)
	}
	return pol, nil
}

// hostServe composes the compatibility mux, the /v1/admin control plane, the
// welfare + outbound-policy wraps, and runs the server. It is the shared tail
// both serve modes reach after wiring their resolver + continuity.
func hostServe(ctx context.Context, cfg ServeConfig, host serveHost, outboundPolicy *policy.Policy, log io.Writer) error {
	// Embeddings/rerank model — loaded once, up front, so a bad -embed-model
	// fails the boot before any listener binds (fail-closed, matching the
	// outbound-policy and admin-token pattern below). Folded into /v1/models
	// and /v1/health here so both surfaces are honest about what the serve
	// actually answers; the resolver wrap itself is applied further down,
	// OUTERMOST on the welfare/policy stack (see serve_embed.go).
	var embedModel inference.TextModel
	var embedID string
	if core.Trim(cfg.EmbedModelPath) != "" {
		var err error
		embedModel, embedID, err = loadEmbedModel(cfg.EmbedModelPath, cfg.EmbedModelID)
		if err != nil {
			return core.E("serving.RunServe", core.Sprintf("embeddings model %q — refusing to serve", cfg.EmbedModelPath), err)
		}
		baseListModels, baseHealthModels := host.listModels, host.healthModels
		host.listModels = func() []string { return append(append([]string{}, baseListModels()...), embedID) }
		host.healthModels = func() []string { return append(append([]string{}, baseHealthModels()...), embedID) }
		printServe(log, "serve: embeddings model %q ready (%s) — /v1/embeddings and /v1/rerank route requests naming it; other names still reach the chat model", embedID, cfg.EmbedModelPath)
	}

	admin := compat.AdminConfig{
		Health: func(_ context.Context) (compat.Health, error) {
			return compat.Health{Status: "ok", Runtime: "go-inference", Models: host.healthModels(), Time: time.Now().Unix()}, nil
		},
		Models: host.listModels,
	}
	adminMux := adminpkg.NewMux(adminpkg.Config{
		Reloader:        host.reloader,
		ServeStatus:     host.status,
		ModelController: host.controller,
		Log:             log,
	})

	opts := []ServeOption{
		WithAdminToken(cfg.AdminToken),
		WithAdminHandler(adminMux),
		WithAdminConfig(admin),
		WithAuditLog(log),
	}
	if cfg.ReadTimeout > 0 {
		opts = append(opts, WithReadHeaderTimeout(cfg.ReadTimeout))
	}
	if cfg.WriteTimeout > 0 {
		opts = append(opts, WithWriteTimeout(cfg.WriteTimeout))
	}
	if cfg.ShutdownTimeout > 0 {
		opts = append(opts, WithShutdownTimeout(cfg.ShutdownTimeout))
	}
	resolver := host.resolver
	if cfg.Welfare {
		resolver = wrapWelfareResolver(resolver, host.currentPath, log)
		printServe(log, "serve: welfare guard ON — per-turn detect + mediation on every chat route (lem_end for Lemma checkpoints); -welfare=false disables")
	}
	if outboundPolicy != nil {
		// Outermost on output: policy enforces on the final tokens (after any
		// welfare rephrase/synthetic reply the deployment would otherwise emit).
		if outboundPolicy.NeedsMediator() {
			resolver = policy.WrapResolverMediated(resolver, outboundPolicy, log, cfg.PolicyMediator)
			printServe(log, "serve: outbound policy ON — %d rule(s), hold-back %dB, mediated rewrite (timeout %s); redact/refuse/rewrite on model output, audited per enforcement; -policy disables", outboundPolicy.Len(), outboundPolicy.HoldBack(), outboundPolicy.MediateTimeout())
		} else {
			resolver = policy.WrapResolver(resolver, outboundPolicy, log)
			printServe(log, "serve: outbound policy ON — %d rule(s), hold-back %dB; redact/refuse on model output, audited per enforcement; -policy disables", outboundPolicy.Len(), outboundPolicy.HoldBack())
		}
	}
	if embedModel != nil {
		// Outermost of all: an embeddings/rerank call isn't a Chat call, so
		// there is nothing in it for the welfare/policy text guards above to
		// police — routing it past them (rather than through, like the chat
		// path) skips a wrap hop those guards' own Unwrap seam would only
		// have absorbed anyway (see serve_embed.go, welfare_guard.go).
		resolver = wrapEmbedResolver(resolver, embedID, embedModel)
	}
	return Serve(ctx, cfg.Addr, resolver, opts...)
}

// runMultiModelServe hosts the multi-model registry. The -model ModelPath becomes
// the pinned default (prepended), the extra cfg.Models load on demand under the
// memory ceiling with LRU + idle-TTL eviction, and the /v1/admin/models control
// plane mounts alongside the (default-model) serve/reload verb.
func runMultiModelServe(ctx context.Context, cfg ServeConfig, outboundPolicy *policy.Policy, log io.Writer) error {
	specs := cfg.Models
	if p := core.Trim(cfg.ModelPath); p != "" {
		// -model is the pinned default, prepended so it wins default selection.
		specs = append([]ModelSpec{{Path: p, Pinned: true, LoadOptions: modelLoadOptions(cfg)}}, specs...)
	}
	mm, err := newMultiModelResolver(specs, MultiModelOptions{
		MemoryCeiling: cfg.MemoryCeiling,
		IdleTTL:       cfg.IdleTTL,
		SweepInterval: cfg.SweepInterval,
	})
	if err != nil {
		return core.E("serving.RunServe", "multi-model registry", err)
	}
	mm.setLoader(cfg.Loader) // nil-safe: keeps the registered-metal default
	mm.setLog(log)

	// Optional per-resident-model scheduler — mirrors the single-model wiring
	// above (RunServe's schedulerResolver), but one instance PER resident model
	// rather than one shared instance: ensureResident attaches a fresh
	// scheduler.Model when a model loads, evict tears it down (CloseEngine)
	// when that model is evicted or unloaded, and closeSchedulers drains
	// whatever is still resident at serve shutdown. Unset leaves schedCfg nil
	// — ensureResident's fast path never constructs a scheduler package type,
	// so the request path is byte-for-byte unchanged, exactly as the
	// single-model path holds when its flag is unset.
	if core.Trim(cfg.Scheduler) != "" {
		mode, err := parseSchedulerMode(cfg.Scheduler)
		if err != nil {
			return err
		}
		schedCfg := schedulerServeConfig(mode)
		mm.setScheduler(&schedCfg)
		defer mm.closeSchedulers()
		printServe(log, "serve: scheduler %s — each resident model gets its own %s scheduler on load", mode, mode)
	}

	mm.startSweeper(ctx, resolveSweepInterval(cfg))

	if cfg.StateConversations {
		defer wireContinuity(ctx, cfg, mm.setOnLoad, log)()
	}

	printServe(log, "serve: multi-model — %d model(s), memory ceiling %d bytes, idle-ttl %s (per-model MTP/draft is not yet armed in this mode)", len(specs), cfg.MemoryCeiling, cfg.IdleTTL)
	printServe(log, "serve: listening on %s (default model=%s)", cfg.Addr, mm.CurrentPath())

	host := serveHost{
		resolver:     mm.openaiResolver(),
		currentPath:  mm.CurrentPath,
		setOnLoad:    mm.setOnLoad,
		reloader:     mm,
		controller:   newMultiModelController(mm),
		listModels:   mm.listedModelIDs,
		healthModels: mm.residentModelIDs,
		status: adminpkg.ServeStatus{
			ModelPath:    mm.CurrentPath(),
			Runtime:      "metal",
			LoadedAtUnix: time.Now().Unix(),
			Config: adminpkg.ServeStatusConfig{
				ContextLength: cfg.ContextLen,
				CacheMode:     cfg.KVCacheMode,
			},
		},
	}
	return hostServe(ctx, cfg, host, outboundPolicy, log)
}

// modelLoadOptions builds the loader options for the -model default in
// multi-model mode: the shared LoadOptions plus a context override.
func modelLoadOptions(cfg ServeConfig) []inference.LoadOption {
	opts := append([]inference.LoadOption(nil), cfg.LoadOptions...)
	if cfg.ContextLen > 0 {
		opts = append(opts, inference.WithContextLen(cfg.ContextLen))
	}
	return opts
}

// resolveSweepInterval picks the idle-sweep tick: an explicit SweepInterval wins;
// otherwise idle eviction ticks at min(IdleTTL, 30s) so a short TTL is honoured
// promptly and a long one does not spin. 0 when idle eviction is off.
func resolveSweepInterval(cfg ServeConfig) time.Duration {
	if cfg.SweepInterval > 0 {
		return cfg.SweepInterval
	}
	if cfg.IdleTTL <= 0 {
		return 0
	}
	if cfg.IdleTTL < 30*time.Second {
		return cfg.IdleTTL
	}
	return 30 * time.Second
}

// multiModelController adapts the multiModelResolver to the admin.ModelController
// seam, converting the resolver's ModelStatus to the admin wire shape so the
// admin package stays serving-free.
type multiModelController struct {
	r *multiModelResolver
}

// newMultiModelController wraps r as an admin.ModelController.
func newMultiModelController(r *multiModelResolver) *multiModelController {
	return &multiModelController{r: r}
}

func (c *multiModelController) ListModels() []adminpkg.ModelStatus {
	statuses := c.r.list()
	out := make([]adminpkg.ModelStatus, 0, len(statuses))
	for _, s := range statuses {
		status := adminpkg.ModelStatus{
			ID:            s.ID,
			Path:          s.Path,
			Resident:      s.Resident,
			Pinned:        s.Pinned,
			EstBytes:      s.EstBytes,
			Profiles:      s.Profiles,
			LastUsedUnix:  s.LastUsedUnix,
			SchedulerMode: s.SchedulerMode,
		}
		if s.SchedulerStats != nil {
			status.SchedulerStats = &adminpkg.SchedulerStats{
				Submitted:  s.SchedulerStats.Submitted,
				Admitted:   s.SchedulerStats.Admitted,
				Completed:  s.SchedulerStats.Completed,
				Cancelled:  s.SchedulerStats.Cancelled,
				Active:     s.SchedulerStats.Active,
				Queued:     s.SchedulerStats.Queued,
				MaxRunning: s.SchedulerStats.MaxRunning,
			}
		}
		out = append(out, status)
	}
	return out
}

func (c *multiModelController) LoadModel(id, path string, opts []inference.LoadOption, pinned bool) (string, error) {
	return c.r.loadSpec(ModelSpec{ID: id, Path: path, Pinned: pinned, LoadOptions: opts})
}

func (c *multiModelController) UnloadModel(id string) error { return c.r.unloadModel(id) }

func (c *multiModelController) SetPinned(id string, pinned bool) error {
	return c.r.setPinned(id, pinned)
}

// wireContinuity opens the conversation state store and registers the per-load
// continuity hook through setOnLoad (the single-model hot-swap resolver's or the
// multi-model registry's — both expose the same registrar). When no
// ContinuityEnabler is injected (the registered engine exposes no continuity
// attach), or the store can't open, serve degrades to stateless with an honest
// notice rather than failing. The returned cleanup (never nil) runs at serve
// shutdown: it closes the store, and when the store was the DEFAULTED one it
// removes the file — the default store is a per-run scratch cache, not an
// archive.
func wireContinuity(ctx context.Context, cfg ServeConfig, setOnLoad func(func(inference.TextModel)), log io.Writer) func() {
	if cfg.EnableContinuity == nil {
		printServe(log, "serve: conversation continuity unavailable on this engine — serving stateless")
		return func() {}
	}
	store, cleanup, where, err := openContinuityStore(ctx, cfg.StateStorePath, cfg.StateRAMBudget, log)
	if err != nil {
		printServe(log, "serve: conversation state store: %v", err)
		printServe(log, "serve: conversation continuity unavailable — serving stateless")
		return func() {}
	}
	enable := cfg.EnableContinuity
	setOnLoad(func(model inference.TextModel) {
		if err := enable(model, store); err != nil {
			printServe(log, "serve: conversation continuity unavailable (stateless serving continues): %v", err)
			return
		}
		printServe(log, "serve: conversation continuity ON — %s, no prompt replay", where)
	})
	return cleanup
}

// openContinuityStore selects the conversation state tier from the
// -state-store and -state-ram-budget flags. An explicit path is the durable
// per-project file store, so chats that asked for durability keep their .kv
// semantics untouched — -state-ram-budget is meaningless there (already
// disk-backed) and only logs a notice. Unset holds conversations in RAM: a
// long-lived server has no reason to pay a per-turn disk round-trip for a
// cache it would discard at shutdown anyway. RAM is unbounded when
// ramBudget <= 0 (byte-identical to pre-#48 behaviour); above zero it is a
// ramspill.Store instead, so the coldest chunks page out to a scratch .kv
// file once resident bytes cross the ceiling, and page back in transparently.
//
// It returns the store, a shutdown cleanup (closes/removes the backing file;
// a no-op for unbounded RAM), a phrase for the boot notice, and an error only
// when a requested durable store — or the spill scratch store — could not be
// opened; serve then degrades to stateless.
func openContinuityStore(ctx context.Context, flagPath string, ramBudget int64, log io.Writer) (store state.Store, cleanup func(), where string, err error) {
	if path := core.Trim(flagPath); path != "" {
		opened, openErr := openConversationStore(ctx, path)
		if openErr != nil {
			return nil, func() {}, "", openErr
		}
		if ramBudget > 0 {
			printServe(log, "serve: -state-ram-budget ignored — %s is already disk-backed", path)
		}
		return opened, func() { _ = opened.Close() }, "chats wake from " + path, nil
	}
	if ramBudget <= 0 {
		return state.NewInMemoryStore(nil), func() {}, "conversations held in RAM (set -state-store for a durable per-project store)", nil
	}
	spillPath := ramSpillPath()
	cold, openErr := filestore.Create(ctx, spillPath)
	if openErr != nil {
		return nil, func() {}, "", core.E("serving.openContinuityStore", "open ram-spill scratch store", openErr)
	}
	tiered, newErr := ramspill.New(ramspill.Options{Budget: ramBudget, Cold: cold, Log: log})
	if newErr != nil {
		_ = cold.Close()
		return nil, func() {}, "", core.E("serving.openContinuityStore", "build ram-spill store", newErr)
	}
	cleanup = func() {
		_ = cold.Close()
		core.Remove(spillPath)
	}
	where = core.Sprintf("conversations held in RAM under a %d-byte budget (coldest chunks spill to %s)", ramBudget, spillPath)
	return tiered, cleanup, where, nil
}

// ramSpillPath is the scratch file a budgeted RAM store pages its coldest
// chunks to — the same per-run-scratch lifecycle the old defaulted
// conversation store used: truncated fresh at Create, removed at shutdown.
func ramSpillPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "lem", "conversations.spill.kv")
}

// openConversationStore opens (or creates) the durable conversation state store
// at path — per-project state opened in place, created on first use, never
// wiped. A missing parent directory is created.
func openConversationStore(ctx context.Context, path string) (*filestore.Store, error) {
	if core.Stat(path).OK {
		return filestore.Open(ctx, path)
	}
	if dir := core.PathDir(path); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return nil, core.E("serving.openConversationStore", "mkdir store dir", r.Value.(error))
		}
	}
	return filestore.Create(ctx, path)
}

// printServe writes a serve boot notice to w (nil silences it).
func printServe(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	core.Print(w, format, args...)
}

// serveConfig holds the tunable knobs for Serve. Fields are declarative so the
// call site reads as configuration, not procedure.
type serveConfig struct {
	readHeaderTimeout time.Duration
	writeTimeout      time.Duration
	shutdownTimeout   time.Duration
	adminToken        string
	adminHandler      http.Handler
	admin             compat.AdminConfig
	audit             io.Writer
}

// ServeOption tunes Serve. Unset options fall back to the serve defaults (30s
// read-header, 5m write, 10s shutdown, no admin wall).
type ServeOption func(*serveConfig)

// WithReadHeaderTimeout sets the HTTP read-header timeout (default 30s).
func WithReadHeaderTimeout(d time.Duration) ServeOption {
	return func(c *serveConfig) { c.readHeaderTimeout = d }
}

// WithWriteTimeout sets the HTTP write timeout, which must cover a full
// streaming response (default 5m).
func WithWriteTimeout(d time.Duration) ServeOption {
	return func(c *serveConfig) { c.writeTimeout = d }
}

// WithShutdownTimeout sets the graceful-shutdown deadline applied after the
// context is cancelled (default 10s).
func WithShutdownTimeout(d time.Duration) ServeOption {
	return func(c *serveConfig) { c.shutdownTimeout = d }
}

// WithAdminToken raises the Bearer-auth wall on /v1/admin/*. An empty token (the
// default) leaves the admin subtree open — pass a token whenever an admin
// handler is mounted (see WithAdminHandler).
func WithAdminToken(token string) ServeOption {
	return func(c *serveConfig) { c.adminToken = token }
}

// WithAdminHandler mounts handler at the /v1/admin/ subtree (behind the Bearer
// wall when WithAdminToken is set). The admin subsystem (reload / download / hf
// / auth) supplies this; the foundation serve leaves it nil, so /v1/admin/* is a
// guarded-but-empty subtree.
func WithAdminHandler(handler http.Handler) ServeOption {
	return func(c *serveConfig) { c.adminHandler = handler }
}

// WithAdminConfig supplies the host-owned health / wake / sleep callbacks for
// the compatibility mux's /v1/health and /v1/runtime/* routes.
func WithAdminConfig(admin compat.AdminConfig) ServeOption {
	return func(c *serveConfig) { c.admin = admin }
}

// WithAuditLog directs admin auth-deny lines to w. nil (the default) silences
// them. cmd/lem passes os.Stderr.
func WithAuditLog(w io.Writer) ServeOption {
	return func(c *serveConfig) { c.audit = w }
}

// Serve hosts the OpenAI / Anthropic / Ollama-compatible HTTP API for resolver
// on addr and blocks until ctx is cancelled or the listener fails.
//
// It composes the compatibility mux (compat.NewMuxWithAdmin) at "/", mounts an
// optional admin handler at "/v1/admin/", and — when an admin token is set —
// wraps the whole tree in the Bearer wall so admin verbs require auth while
// inference paths pass through. On ctx cancellation it drains in-flight requests
// within the shutdown timeout. RunServe is the high-level entry; Serve is the
// resolver-agnostic HTTP layer beneath it.
//
//	err := serving.Serve(ctx, ":36911", resolver, serving.WithAdminToken(tok))
func Serve(ctx context.Context, addr string, resolver Resolver, opts ...ServeOption) error {
	cfg := serveConfig{
		readHeaderTimeout: 30 * time.Second,
		writeTimeout:      5 * time.Minute,
		shutdownTimeout:   10 * time.Second,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// http.ServeMux uses longest-prefix match, so /v1/admin/ routes hit the
	// admin handler and everything else falls through to the compat mux.
	root := http.NewServeMux()
	if cfg.adminHandler != nil {
		root.Handle("/v1/admin/", cfg.adminHandler)
	}
	root.Handle("/", compat.NewMuxWithAdmin(resolver, cfg.admin))

	// Bearer auth on /v1/admin/* only — mounted at the root so composition order
	// can never leave an admin handler unauthenticated.
	var handler http.Handler = root
	if cfg.adminToken != "" {
		handler = RequireBearerOnAdmin(root, cfg.adminToken, cfg.audit)
	}

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: cfg.readHeaderTimeout,
		WriteTimeout:      cfg.writeTimeout,
	}

	errCh := make(chan error, 1)
	go func() {
		err := srv.ListenAndServe()
		if err != nil && err != http.ErrServerClosed {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.shutdownTimeout)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	}
}
