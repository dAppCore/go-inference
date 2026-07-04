// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
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
	profileDir := fs.String("profile-dir", "", "tuned-profile directory (default ~/Lethean/data/tuning)")
	contextLen := fs.Int("context", 0, "override context length; 0 uses the model's default")
	kvCacheMode := fs.String("kv-cache", "", "KV cache mode (paged, fp16, q8, kq8vq4, turboquant; empty = load default)")
	readTimeout := fs.Duration("read-timeout", 30*time.Second, "HTTP read header timeout")
	writeTimeout := fs.Duration("write-timeout", 5*time.Minute, "HTTP write timeout (covers full streaming response)")
	shutdownTimeout := fs.Duration("shutdown-timeout", 10*time.Second, "graceful shutdown deadline after SIGINT/SIGTERM")
	printAdminToken := fs.Bool("print-admin-token", false, "print the admin Bearer token and exit (generates if absent, mode 0600 at ~/Lethean/data/admin.token)")
	rotateAdminToken := fs.Bool("rotate-admin-token", false, "regenerate the admin Bearer token, print it, and exit")
	stateConversations := fs.Bool("state-conversations", true, "conversation continuity: wake each chat from its slept state, append only the new turn, sleep after — no prompt replay (disable with -state-conversations=false)")
	stateStorePath := fs.String("state-store", "", "conversation state store file (default ~/Lethean/data/state/conversations.kv)")
	nativeBackend := fs.Bool("native", false, "serve via the no-cgo native token-loop contract (the default go-inference metal engine already is native)")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s serve [--model <path>] [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Host an OpenAI / Anthropic / Ollama-compatible HTTP API for a model.\n")
		core.WriteString(stderr, "Default port 36911 is Lethean's own — an Ollama install on 11434 never collides.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Inference routes (all relative to the listen address):\n")
		core.WriteString(stderr, "  POST /v1/chat/completions    OpenAI chat (streaming + non-streaming)\n")
		core.WriteString(stderr, "  POST /v1/messages            Anthropic Messages\n")
		core.WriteString(stderr, "  POST /api/chat               Ollama chat\n")
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

	loadOpts := []inference.LoadOption{}
	if *contextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(*contextLen))
	}

	err = serving.RunServe(ctx, serving.ServeConfig{
		Addr:               *addr,
		ModelPath:          *modelPath,
		LoadOptions:        loadOpts,
		DraftPath:          *draftPath,
		DraftDetect:        *draftDetect,
		DraftBlock:         *draftBlock,
		NoAutoProfile:      *noAutoProfile,
		ProfileDir:         *profileDir,
		KVCacheMode:        *kvCacheMode,
		Native:             *nativeBackend,
		StateConversations: *stateConversations,
		StateStorePath:     *stateStorePath,
		ReadTimeout:        *readTimeout,
		WriteTimeout:       *writeTimeout,
		ShutdownTimeout:    *shutdownTimeout,
		AdminToken:         adminToken,
		Log:                stderr,
	})
	if err != nil {
		core.Print(stderr, "%s serve: %v", cliName(), err)
		return 1
	}
	return 0
}
