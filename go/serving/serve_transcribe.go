// SPDX-Licence-Identifier: EUPL-1.2

// serve_transcribe.go wires an optional Whisper checkpoint into a serve as the process's ASR surface —
// the --model flag's v1 "honest shape" for Whisper (docs/superpowers/specs/2026-07-19-whisper-asr-
// design.md's Surfaces section). Unlike -embed-model (serve_embed.go: served ALONGSIDE the chat model,
// its own dedicated flag), a Whisper checkpoint reuses --model itself and takes over the WHOLE process:
// Whisper never enters the TextModel factory (model/arch/openai/whisper's own doc comment — an encoder-
// decoder ASR forward does not fit the causal-LM contract model.Load's registered path assembles), so
// there is no chat model for this process to also serve. RunServe calls detectAndLoadWhisper as its very
// first --model check, ahead of the ordinary chat hot-swap/multi-model wiring — mirrors bert's
// loadEmbedModel (serve_embed.go) in importing a concrete model/arch/* package directly, the same
// precedent for a specialised capability the generic TextModel backend registry does not (yet) carry.
package serving

import (
	"context"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/openai/whisper"
	adminpkg "dappco.re/go/inference/serving/admin"
	"dappco.re/go/inference/serving/policy"
)

// detectAndLoadWhisper reports whether modelPath is a Whisper checkpoint and, when it is, loads it fully
// — fatal at boot on a load failure, matching the outbound-policy/embed-model/admin-token "fail closed"
// precedent (a deployer who pointed --model at a Whisper checkpoint gets ASR serving or an honest
// refusal, never a silent fall-through into the chat loader's own confusing mid-Assemble failure — see
// whisper.Config.Arch's doc comment for what that failure looks like today). modelPath == "" or any
// non-Whisper directory is NOT an error here: isWhisper reports false and the caller's ordinary
// chat-model path proceeds completely unchanged — this only ever short-circuits a genuine Whisper
// directory (whisper.IsWhisperCheckpoint's cheap config.json probe decides which).
func detectAndLoadWhisper(modelPath string) (transcriber inference.Transcriber, isWhisper bool, err error) {
	path := core.Trim(modelPath)
	if path == "" || !whisper.IsWhisperCheckpoint(path) {
		return nil, false, nil
	}
	m, loadErr := whisper.Load(path)
	if loadErr != nil {
		return nil, true, core.E("serving.detectAndLoadWhisper", core.Sprintf("Whisper checkpoint %q — refusing to serve", path), loadErr)
	}
	return m, true, nil
}

// runWhisperServe hosts a Whisper checkpoint's ASR-only surface: POST /v1/audio/transcriptions answers
// via transcriber, every chat/embeddings route reports "no model loaded" exactly like an ordinary
// -model "" model-less start (newHotSwapResolver("", ...) — the same resolver the chat path builds, just
// never given a path to load), and /v1/models + /v1/health advertise the checkpoint's basename so
// operators can see what is actually being served. None of the reactive-drafter/multi-model/continuity/
// scheduler machinery applies to an encoder-decoder ASR model, so this is a deliberately short path, not
// a trimmed copy of RunServe's chat branch.
func runWhisperServe(ctx context.Context, cfg ServeConfig, transcriber inference.Transcriber, outboundPolicy *policy.Policy, log io.Writer) error {
	hotSwap := newHotSwapResolver("", "", 0, nil)
	id := core.PathBase(core.Trim(cfg.ModelPath))
	printServe(log, "serve: %q is a Whisper checkpoint — POST /v1/audio/transcriptions is live (chat/embeddings routes report no model loaded; v1 serves one purpose per process)", cfg.ModelPath)
	printServe(log, "serve: listening on %s (model=%s, whisper ASR)", cfg.Addr, cfg.ModelPath)
	host := serveHost{
		resolver:     hotSwap.openaiResolver(),
		currentPath:  hotSwap.CurrentPath,
		setOnLoad:    hotSwap.setOnLoad,
		reloader:     hotSwap,
		listModels:   func() []string { return []string{id} },
		healthModels: func() []string { return []string{cfg.ModelPath} },
		status: adminpkg.ServeStatus{
			ModelPath:    cfg.ModelPath,
			Runtime:      "whisper",
			LoadedAtUnix: time.Now().Unix(),
		},
	}
	return hostServe(ctx, cfg, host, outboundPolicy, log, transcriber)
}
