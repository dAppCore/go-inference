// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/spine"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/filestore"
	"dappco.re/go/inference/model/state/session"
)

// sessionModel is the session-capable model the -state turn loop needs: the
// loaded TextModel plus a handle factory and its info. The go-inference metal
// model (engine.TextModel) satisfies it.
type sessionModel interface {
	NewSession() inference.SessionHandle
	Info() inference.ModelInfo
}

// chatFormatter is the optional chat-framing a model may expose so a -state turn
// is rendered the way serve's conversation continuity frames every stateless
// request (fresh session → FormatChatPrompt, woken session →
// FormatChatContinuation; no prior-turn replay). The current engine/metal model
// does not implement it, so -state turns degrade to raw framing with an honest
// notice; the framing lights up when the model exposes the seam.
type chatFormatter interface {
	FormatChatPrompt(messages []inference.Message) string
	FormatChatContinuation(messages []inference.Message) string
}

// runStateTurn runs one conversation turn through the durable state system — the
// no-prompt-replay loop. If the named state exists it is woken (KV restored from
// .kv blocks, no re-prefill of prior turns) and only the new turn is appended;
// otherwise the prompt opens a fresh session. After generation the session
// sleeps back to the store, so the next invocation starts where this one ended.
// Ported from lthn-mlx's cmd/mlx generate -state.
func runStateTurn(ctx context.Context, cfg Config, loadOpts []inference.LoadOption) error {
	storePath := core.Trim(cfg.StateStore)
	if storePath == "" {
		homeR := core.UserHomeDir()
		if !homeR.OK {
			return core.E("generate.state", "resolve home for default -state-store", nil)
		}
		home, _ := homeR.Value.(string)
		storePath = core.PathJoin(home, "Lethean", "lem", "state", "agent.kv")
	}
	store, err := openStateStore(ctx, storePath)
	if err != nil {
		return core.E("generate.state", core.Sprintf("state store %s", storePath), err)
	}
	defer store.Close()

	tm, err := loadTextModel(cfg.ModelPath, loadOpts...)
	if err != nil {
		return core.E("generate.state", "load", err)
	}
	defer tm.Close()

	sm, ok := tm.(sessionModel)
	if !ok {
		return core.E("generate.state", "loaded model does not support sessions", nil)
	}
	handle := sm.NewSession()
	if handle == nil {
		return core.E("generate.state", "nil session handle", nil)
	}
	sess := session.New(handle, spineModelInfo(sm.Info(), cfg.ContextLen), nil)
	defer sess.Close()

	var formatter chatFormatter
	if !cfg.Raw {
		if f, ok := tm.(chatFormatter); ok {
			formatter = f
		} else {
			printNote(cfg.Log, "generate: model exposes no chat-framing seam — running raw-framed -state turns (the -raw contract)")
		}
	}
	return runStateSession(ctx, cfg, storePath, store, sess, formatter)
}

// runStateSession runs one -state turn against an already-open session:
// wake-if-present, generate, sleep back. formatter chat-frames the new turn when
// present (fresh → FormatChatPrompt, woken → FormatChatContinuation); a nil
// formatter is the raw contract (the prompt prefills/appends byte-for-byte).
func runStateSession(ctx context.Context, cfg Config, storePath string, store *filestore.Store, sess *session.Session, formatter chatFormatter) error {
	name := cfg.StateName
	entryURI := "mlx://agent/" + name
	indexURI := entryURI + "/index"

	woke := false
	var wakeDur, prefillDur time.Duration
	var wakeReport *agent.WakeReport
	if _, idxErr := agent.LoadStateIndex(ctx, store, indexURI); idxErr == nil {
		start := time.Now()
		report, wakeErr := sess.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: indexURI, EntryURI: entryURI})
		if wakeErr != nil {
			return core.E("generate.state", core.Sprintf("wake %s", name), wakeErr)
		}
		wakeReport = report
		wakeDur = time.Since(start)
		start = time.Now()
		// Continuation form: close the previously open model turn, render only
		// the new user turn, reopen the assistant header — no replay of the
		// retained prefix.
		turn := "\n" + cfg.Prompt
		if formatter != nil {
			turn = formatter.FormatChatContinuation([]inference.Message{{Role: "user", Content: cfg.Prompt}})
		}
		if err := sess.AppendPrompt(turn); err != nil {
			return core.E("generate.state", "append turn", err)
		}
		prefillDur = time.Since(start)
		woke = true
	} else {
		var notFound *state.URIChunkNotFoundError
		if !core.As(idxErr, &notFound) {
			return core.E("generate.state", core.Sprintf("state index %s", indexURI), idxErr)
		}
		start := time.Now()
		// Fresh form: the full chat template from empty history.
		turn := cfg.Prompt
		if formatter != nil {
			turn = formatter.FormatChatPrompt([]inference.Message{{Role: "user", Content: cfg.Prompt}})
		}
		if err := sess.Prefill(turn); err != nil {
			return core.E("generate.state", "prefill", err)
		}
		prefillDur = time.Since(start)
	}

	var out []byte
	tokens := 0
	start := time.Now()
	for tok := range sess.GenerateStream(ctx, inference.WithMaxTokens(cfg.MaxTokens), inference.WithTemperature(float32(cfg.Temp))) {
		out = append(out, tok.Text...)
		tokens++
	}
	decodeDur := time.Since(start)
	if err := sess.Err(); err != nil {
		return core.E("generate.state", "generate", err)
	}

	start = time.Now()
	sleepReport, err := sess.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: entryURI, Title: name})
	if err != nil {
		return core.E("generate.state", core.Sprintf("sleep %s", name), err)
	}
	sleepDur := time.Since(start)

	core.WriteString(cfg.Out, string(out))
	core.WriteString(cfg.Out, "\n\n")
	if woke {
		core.WriteString(cfg.Out, core.Sprintf(
			"turn: woke %d prefix tokens in %dms (no replay) · new-turn prefill %dms\n",
			wakeReport.PrefixTokens, wakeDur.Milliseconds(), prefillDur.Milliseconds()))
	} else {
		core.WriteString(cfg.Out, core.Sprintf("turn: fresh state · prefill %dms\n", prefillDur.Milliseconds()))
	}
	if decodeDur > 0 && tokens > 1 {
		core.WriteString(cfg.Out, core.Sprintf("decode %.1f tok/s (%d tok)\n", float64(tokens)/decodeDur.Seconds(), tokens))
	}
	core.WriteString(cfg.Out, core.Sprintf(
		"slept %d tokens -> %d blocks in %dms\n",
		sleepReport.TokenCount, sleepReport.BlocksWritten, sleepDur.Milliseconds()))
	core.WriteString(cfg.Out, core.Sprintf("state: %s (%s)\n", name, storePath))
	return nil
}

// spineModelInfo bridges the loaded model's inference.ModelInfo to the
// spine.ModelInfo the durable session needs, defaulting the context length.
func spineModelInfo(info inference.ModelInfo, contextLen int) spine.ModelInfo {
	if contextLen <= 0 {
		contextLen = 4096
	}
	return spine.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: contextLen,
	}
}

// openStateStore opens the append-only state file, creating it (and its parent
// directory) on first use.
func openStateStore(ctx context.Context, path string) (*filestore.Store, error) {
	if core.Stat(path).OK {
		return filestore.Open(ctx, path)
	}
	if dir := core.PathDir(path); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return nil, core.E("generate.openStateStore", "mkdir store dir", r.Value.(error))
		}
	}
	return filestore.Create(ctx, path)
}
