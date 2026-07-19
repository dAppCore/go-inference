// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/serving"
	"dappco.re/go/inference/serving/continuity"
)

// runServeCommand parses the serve flags and hands them to serving.RunServe.
// The command is deliberately thin: flag parsing + the two admin-token
// subcommands + one call into the library. All serve business logic (mux,
// route handlers, admin bearer auth, reactive drafter resolution, tuned-profile
// resolution, hot-swap, conversation continuity, graceful shutdown) lives in
// dappco.re/go/inference/serving.
func runServeCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("serve"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", ":36911", "listen address (Lethean's own port — never collides with an Ollama install)")
	modelPath := fs.String("model", "", "model path to load; empty starts the driver model-less (load a model later via POST /v1/admin/serve/reload)")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside a Gemma 4 target (assistant/ pair layout, MTP/ gguf), a path forces it, '' disables")
	draftDetect := fs.Bool("draft-detect", true, "reactive drafter detection for Gemma 4 targets (false = only an explicit --draft engages MTP)")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block (verify forward = carried lead + block-1 proposals); 0 = engine default 5, tuned profile overrides when present")
	noAutoProfile := fs.Bool("no-auto-profile", false, "ignore tuned profiles from `lem tune` (run the flag/engine-default draft block)")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory (default ~/Lethean/lem/tuning)")
	contextLen := fs.Int("context", 0, "override context length; 0 uses the model's default")
	kvCacheMode := fs.String("kv-cache", "", "KV cache mode override (engine-reported; the no-cgo metal engine runs its built-in cache only — other values are noted and ignored)")
	readTimeout := fs.Duration("read-timeout", 30*time.Second, "HTTP read header timeout")
	writeTimeout := fs.Duration("write-timeout", 5*time.Minute, "HTTP write timeout (covers full streaming response)")
	shutdownTimeout := fs.Duration("shutdown-timeout", 10*time.Second, "graceful shutdown deadline after SIGINT/SIGTERM")
	printAdminToken := fs.Bool("print-admin-token", false, "print the admin Bearer token and exit (generates if absent, mode 0600 at ~/Lethean/lem/admin.token)")
	rotateAdminToken := fs.Bool("rotate-admin-token", false, "regenerate the admin Bearer token, print it, and exit")
	stateConversations := fs.Bool("state-conversations", true, "conversation continuity: wake each chat from its slept state, append only the new turn, sleep after — no prompt replay (disable with --state-conversations=false)")
	stateSharePrefix := fs.Bool("state-share-prefix", false, "cross-conversation KV sharing: a fresh chat opening with a system prompt another conversation has already served wakes that shared token span instead of re-prefilling it (needs --state-conversations; falls back to a fresh prefill byte-identically on any miss)")
	welfareOn := fs.Bool("welfare", false, "welfare guard (opt-in): per-turn hostility detect + engine-model mediation on every chat route; Lemma checkpoints additionally carry lem_end (enable with --welfare)")
	policyPath := fs.String("policy", "", "outbound policy file (JSON): deployment-owned redact/refuse rules on model OUTPUT (term/pattern matches); unset disables the layer; a load failure is fatal at boot (see serving/policy)")
	stateStorePath := fs.String("state-store", "", "conversation state store file for durable per-project state; unset = conversations held in RAM for the life of the serve process (no per-turn disk round-trip)")
	stateRAMBudget := fs.Int64("state-ram-budget", 0, "byte ceiling for the RAM conversation store (ignored when --state-store is set); 0 = unlimited; over budget the coldest conversation chunks spill to a scratch .kv file and wake back transparently")
	nativeBackend := fs.Bool("native", false, "serve via the no-cgo native token-loop contract (the default go-inference metal engine already is native)")
	modelsConfig := fs.String("models-config", "", "multi-model serving config (JSON): several models with aliases and named profiles, held resident under a memory ceiling with LRU + idle-TTL eviction; --model becomes the pinned default; empty = single-model serve")
	schedulerMode := fs.String("scheduler", "", "request scheduler between the HTTP handlers and the model: 'serial' (bounded-queue worker pool), 'batch' (continuous in-flight batching), 'interleave' (live admission-budget CB); empty = no scheduler (request path unchanged). With --models-config, each resident model gets its own scheduler instance of this mode")
	schedulerConcurrency := fs.Int("scheduler-concurrency", 0, "scheduler's concurrently running requests (interleave/CB lane count, serial pool width); 0 = the serve default (4)")
	embedModel := fs.String("embed-model", "", "embeddings/rerank model: a bert/BGE-class host encoder snapshot directory (config.json + vocab.txt + model.safetensors); served at /v1/embeddings and /v1/rerank under --embed-model-id alongside (or, with --model \"\", instead of) the chat model; empty = those routes serve only what the chat model itself implements (a clean 4xx today). A load failure is fatal at boot")
	embedModelID := fs.String("embed-model-id", "", "request name for --embed-model's `model` field; empty derives the pack's basename")
	corsOrigins := fs.String("cors", "", "browser origins allowed via CORS: comma-separated exact origins (e.g. http://localhost:4200) or '*' for any; empty (the default) sends no CORS headers — a browser app on another origin then cannot call the serve")
	captureSlug := fs.String("capture", "", "tee each completed (prompt, response) turn into this `lem data` dataset with the serving model's fingerprint; empty (the default) captures nothing — the approved privacy default is opt-in only")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s serve [--model <path>] [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Host an OpenAI / Anthropic / Ollama-compatible HTTP API for a model.\n")
		core.WriteString(stderr, "Default port 36911 is Lethean's own — an Ollama install on 11434 never collides.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		printFlagBlock(stderr, fs)
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Inference routes (all relative to the listen address):\n")
		core.WriteString(stderr, "  POST /v1/chat/completions    OpenAI chat (streaming + non-streaming)\n")
		core.WriteString(stderr, "  POST /v1/messages            Anthropic Messages\n")
		core.WriteString(stderr, "  POST /api/chat               Ollama chat\n")
		core.WriteString(stderr, "  POST /v1/embeddings          embedding vectors (chat model or --embed-model)\n")
		core.WriteString(stderr, "  POST /v1/rerank              document reranking (chat model or --embed-model)\n")
		core.WriteString(stderr, "  GET  /v1/models              list loaded models\n")
		core.WriteString(stderr, "  GET  /v1/health              process health probe\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	// Token-management subcommands — handled BEFORE the --model check so
	// operators can reveal / rotate without a model loaded.
	tokenPath := serving.AdminTokenPath()
	if *rotateAdminToken {
		tok, err := serving.GenerateAdminToken()
		if err != nil {
			core.Print(stderr, "%s serve: token rotation failed: %v", cliName(), err)
			return 1
		}
		if err := serving.WriteAdminToken(tokenPath, tok); err != nil {
			core.Print(stderr, "%s serve: token write failed: %v", cliName(), err)
			return 1
		}
		core.Print(stderr, "%s admin token (rotated):\n  %s\n  saved to %s (mode 0600)\n  any running serve still holds the old token — restart to apply", cliName(), tok, tokenPath)
		return 0
	}
	if *printAdminToken {
		tok, generated, err := serving.EnsureAdminToken(tokenPath)
		if err != nil {
			core.Print(stderr, "%s serve: token init failed: %v", cliName(), err)
			return 1
		}
		label := "loaded"
		if generated {
			label = "newly generated"
		}
		core.Print(stderr, "%s admin token (%s):\n  %s\n  at %s (mode 0600)", cliName(), label, tok, tokenPath)
		return 0
	}

	// Admin token — load existing or generate fresh. Fail-closed: if the token
	// file can't be written, serve refuses to boot rather than binding a
	// listener with an unprotected admin surface.
	adminToken, generated, err := serving.EnsureAdminToken(tokenPath)
	if err != nil {
		core.Print(stderr, "%s serve: admin token init failed (fail-closed): %v", cliName(), err)
		return 1
	}
	if generated {
		core.Print(stderr, "%s serve: fresh admin token generated at %s — reveal with `%s serve --print-admin-token`", cliName(), tokenPath, cliName())
	}

	// Multi-model config — parsed up front so a malformed file fails before the
	// listener binds. Empty leaves Models nil, i.e. the single-model serve path.
	var extraModels []serving.ModelSpec
	var mmOpts serving.MultiModelOptions
	if core.Trim(*modelsConfig) != "" {
		specs, opts, cfgErr := serving.LoadModelsConfig(*modelsConfig)
		if cfgErr != nil {
			core.Print(stderr, "%s serve: %v", cliName(), cfgErr)
			return 1
		}
		extraModels, mmOpts = specs, opts
	}

	// Cross-conversation prefix sharing is an opt-in variant of the continuity
	// enabler; default off (an operator call, made with bench receipts).
	continuityEnabler := continuity.Enable
	if *stateSharePrefix {
		continuityEnabler = continuity.EnableSharing
	}

	// Dataset capture — opt-in only (buildCaptureLoader never opens the
	// dataset store when --capture is empty, "OFF without the flag"). A
	// --capture pointing at a dataset that doesn't exist yet fails the boot
	// closed, before any listener binds, matching the admin-token/policy/
	// embed-model precedent above: a deployer who asked for capture gets it
	// or an honest refusal, never a silent no-op.
	captureLoader, captureStore, captureErr := buildCaptureLoader(*captureSlug, stderr)
	if captureErr != nil {
		core.Print(stderr, "%s serve: --capture: %v", cliName(), captureErr)
		return 1
	}
	if captureStore != nil {
		defer captureStore.Close()
		core.Print(stderr, "serve: capturing completed turns into dataset %q", core.Trim(*captureSlug))
	}

	err = serving.RunServe(ctx, serving.ServeConfig{
		Addr:                 *addr,
		ModelPath:            *modelPath,
		ContextLen:           *contextLen,
		Models:               extraModels,
		MemoryCeiling:        mmOpts.MemoryCeiling,
		IdleTTL:              mmOpts.IdleTTL,
		SweepInterval:        mmOpts.SweepInterval,
		DraftPath:            *draftPath,
		DraftDetect:          *draftDetect,
		DraftBlock:           *draftBlock,
		Loader:               captureLoader,
		SpeculativeLoader:    speculativeLoader,
		EnableContinuity:     continuityEnabler,
		NoAutoProfile:        *noAutoProfile,
		ProfileDir:           *profileDir,
		KVCacheMode:          *kvCacheMode,
		Native:               *nativeBackend,
		Scheduler:            *schedulerMode,
		SchedulerConcurrency: *schedulerConcurrency,
		StateConversations:   *stateConversations,
		Welfare:              *welfareOn,
		PolicyPath:           *policyPath,
		EmbedModelPath:       *embedModel,
		EmbedModelID:         *embedModelID,
		StateStorePath:       *stateStorePath,
		StateRAMBudget:       *stateRAMBudget,
		ReadTimeout:          *readTimeout,
		WriteTimeout:         *writeTimeout,
		ShutdownTimeout:      *shutdownTimeout,
		AdminToken:           adminToken,
		CORSOrigins:          *corsOrigins,
		Log:                  stderr,
	})
	if err != nil {
		core.Print(stderr, "%s serve: %v", cliName(), err)
		return 1
	}
	return 0
}
