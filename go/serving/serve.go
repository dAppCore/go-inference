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

	// Conversation continuity.
	StateConversations bool   // wake each chat from its slept state, no prompt replay
	StateStorePath     string // durable per-project store file; empty = ephemeral default, wiped each serve run

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
	var outboundPolicy *policy.Policy
	if core.Trim(cfg.PolicyPath) != "" {
		pol, err := policy.Load(cfg.PolicyPath)
		if err != nil {
			return core.E("serving.RunServe", core.Sprintf("outbound policy %q — refusing to serve unguarded", cfg.PolicyPath), err)
		}
		outboundPolicy = pol
	}

	// Reactive MTP pair resolution (the model declares, the serve reacts): an
	// explicit --draft path wins; --draft="" disables; "auto" runs the ladder.
	detection := ResolveServeDraft(cfg.ModelPath, cfg.DraftPath, cfg.DraftDetect)
	// A detected drafter is only armed when the registered engine exposes a
	// speculative loader; otherwise degrade to plain autoregressive with an
	// honest notice, faithful to lthn-mlx (a drafter that can't load never
	// blocks the serve — it surfaces the failure rather than refusing to boot).
	armDrafter := detection.Active() && cfg.SpeculativeLoader != nil
	draftPathForResolver := ""
	if armDrafter {
		draftPathForResolver = detection.DraftPath
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
		defer wireContinuity(ctx, cfg, hotSwap, log)()
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

	admin := compat.AdminConfig{
		Health: func(_ context.Context) (compat.Health, error) {
			// Report the currently-loaded model (post-reload), or no models when
			// the driver started model-less and none has been loaded yet.
			models := []string{}
			if p := hotSwap.CurrentPath(); p != "" {
				models = append(models, p)
			}
			return compat.Health{Status: "ok", Runtime: "go-inference", Models: models, Time: time.Now().Unix()}, nil
		},
		Models: func() []string {
			if p := hotSwap.CurrentPath(); p != "" {
				return []string{core.PathBase(p)}
			}
			return nil
		},
	}

	// The /v1/admin/* control plane (machine identity, serve status, hot-swap
	// reload) mounts behind the Bearer wall, driven by the same hot-swap
	// resolver serving inference.
	adminMux := adminpkg.NewMux(adminpkg.Config{
		Reloader: hotSwap,
		ServeStatus: adminpkg.ServeStatus{
			ModelPath:    cfg.ModelPath,
			Runtime:      "metal",
			LoadedAtUnix: time.Now().Unix(),
			Config: adminpkg.ServeStatusConfig{
				ContextLength: cfg.ContextLen,
				CacheMode:     cfg.KVCacheMode,
			},
		},
		Log: log,
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
	resolver := hotSwap.openaiResolver()
	if cfg.Welfare {
		resolver = wrapWelfareResolver(resolver, hotSwap, log)
		printServe(log, "serve: welfare guard ON — per-turn detect + mediation on every chat route (lem_end for Lemma checkpoints); -welfare=false disables")
	}
	if outboundPolicy != nil {
		// Outermost on output: policy enforces on the final tokens (after any
		// welfare rephrase/synthetic reply the deployment would otherwise emit).
		resolver = policy.WrapResolver(resolver, outboundPolicy, log)
		printServe(log, "serve: outbound policy ON — %d rule(s), hold-back %dB; redact/refuse on model output, audited per enforcement; -policy disables", outboundPolicy.Len(), outboundPolicy.HoldBack())
	}
	return Serve(ctx, cfg.Addr, resolver, opts...)
}

// wireContinuity opens the conversation state store and registers the per-load
// continuity hook. When no ContinuityEnabler is injected (the registered engine
// exposes no continuity attach), or the store can't open, serve degrades to
// stateless with an honest notice rather than failing. The returned cleanup
// (never nil) runs at serve shutdown: it closes the store, and when the store
// was the DEFAULTED one it removes the file — the default store is a per-run
// scratch cache, not an archive.
func wireContinuity(ctx context.Context, cfg ServeConfig, hotSwap *hotSwapResolver, log io.Writer) func() {
	if cfg.EnableContinuity == nil {
		printServe(log, "serve: conversation continuity unavailable on this engine — serving stateless")
		return func() {}
	}
	storePath, ephemeral := resolveStateStorePath(cfg.StateStorePath)
	var store *filestore.Store
	if storePath != "" {
		if opened, err := openConversationStore(ctx, storePath, ephemeral); err == nil {
			store = opened
		} else {
			printServe(log, "serve: conversation state store %s: %v", storePath, err)
		}
	}
	if store == nil {
		printServe(log, "serve: conversation continuity unavailable — serving stateless")
		return func() {}
	}
	enable := cfg.EnableContinuity
	hotSwap.setOnLoad(func(model inference.TextModel) {
		if err := enable(model, store); err != nil {
			printServe(log, "serve: conversation continuity unavailable (stateless serving continues): %v", err)
			return
		}
		if ephemeral {
			printServe(log, "serve: conversation continuity ON — chats wake from %s (ephemeral: wiped each serve run; set -state-store for durable per-project state), no prompt replay", storePath)
			return
		}
		printServe(log, "serve: conversation continuity ON — chats wake from %s, no prompt replay", storePath)
	})
	return func() {
		_ = store.Close()
		if ephemeral {
			core.Remove(storePath)
		}
	}
}

// resolveStateStorePath maps the -state-store flag to the store path and its
// lifetime: an EMPTY flag means serve made the store itself — the defaulted
// conversations.kv is ephemeral (wiped fresh at launch, removed at shutdown),
// so per-turn tail-block rewrites never accumulate across runs. Any explicit
// path — even the default's literal location — is the durable per-project
// store and is never wiped.
func resolveStateStorePath(flagPath string) (path string, ephemeral bool) {
	if p := core.Trim(flagPath); p != "" {
		return p, false
	}
	if homeR := core.UserHomeDir(); homeR.OK {
		if home, ok := homeR.Value.(string); ok {
			return core.PathJoin(home, "Lethean", "lem", "state", "conversations.kv"), true
		}
	}
	return "", false
}

// openConversationStore opens the conversation state store at path. An
// ephemeral (defaulted) store is created FRESH every time — filestore.Create's
// O_TRUNC wipes whatever a previous run (or crash) left behind. A durable
// (explicit) store opens in place, created on first use.
func openConversationStore(ctx context.Context, path string, ephemeral bool) (*filestore.Store, error) {
	if ephemeral {
		return filestore.Create(ctx, path)
	}
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
