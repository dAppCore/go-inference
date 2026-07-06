// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// speculative_model.go is the seam that was missing: it wires the tested
// AssistantPair MTP loop (draft → verify → accept, all in assistant_load.go) to
// the inference.TextModel contract that serve and generate drive. Before this,
// a detected drafter degraded to plain autoregressive with "this engine exposes
// no speculative path" — because nothing turned the pair into a TextModel a
// SpeculativeLoader could hand back. LoadSpeculativePair is that loader.
//
// Greedy-exact: the target verify accepts exactly the tokens it would have
// argmax-produced, so at temperature 0 the speculative output is byte-identical
// to the plain decode — only faster (one target weight-read verifies a block of
// drafted tokens). That is why the generate wiring routes to it only for greedy
// requests: a sampled request would need the sampled verify lane (a follow-up).
package native

import (
	"context"
	"iter"
	"slices"
	"strings"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
)

// speculativeModel adapts a target ArchSession + drafter AssistantPair to
// inference.TextModel. The target weights are resident in one ArchSession
// (loaded once by LoadDir); the pair carries the drafter plus the target arch
// config (no second copy of the target weights). Generate/Chat run the
// speculative loop and record inference.SpeculativeMetrics for the -draft
// bench read (printMTPMetrics).
type speculativeModel struct {
	target     *ArchSession
	pair       *AssistantPair
	tok        *tokenizer.Tokenizer
	info       inference.ModelInfo
	modelType  string
	draftBlock int

	mu          sync.Mutex
	lastErr     error
	lastMetrics inference.GenerateMetrics
	lastSpec    inference.SpeculativeMetrics
}

var (
	_ inference.TextModel                  = (*speculativeModel)(nil)
	_ inference.SpeculativeMetricsProvider = (*speculativeModel)(nil)
)

// LoadSpeculativePair loads a target checkpoint + drafter as one speculative
// inference.TextModel running draftBlock-wide MTP verify forwards — the
// serving.SpeculativeLoader shape. The target weights load once via LoadDir; the
// pair adds the drafter and validates the attachment (arch/backbone match).
// draftBlock ≤ 0 falls back to the shipped default of 5.
func LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)
	maxLen := cfg.ContextLen
	if maxLen <= 0 {
		maxLen = 4096
	}
	if draftBlock <= 0 {
		draftBlock = 5
	}
	target, err := LoadDir(targetPath, maxLen)
	if err != nil {
		return nil, core.E("native.LoadSpeculativePair", "load target checkpoint", err)
	}
	pair, err := LoadAssistantPairDirs(targetPath, draftPath)
	if err != nil {
		_ = target.Close()
		return nil, core.E("native.LoadSpeculativePair", "attach drafter", err)
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetPath, "tokenizer.json"))
	if err != nil {
		_ = target.Close()
		_ = pair.Close()
		return nil, core.E("native.LoadSpeculativePair", "load tokenizer", err)
	}
	return &speculativeModel{
		target:     target,
		pair:       pair,
		tok:        tok,
		modelType:  "gemma4",
		draftBlock: draftBlock,
		info: inference.ModelInfo{
			Architecture: "gemma4",
			VocabSize:    pair.TargetArch.Vocab,
			NumLayers:    len(pair.TargetArch.Layer),
			HiddenSize:   pair.TargetArch.Hidden,
		},
	}, nil
}

// Generate streams the speculative completion of a raw prompt.
func (m *speculativeModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.speculate(ctx, m.tok.Encode(prompt), inference.ApplyGenerateOpts(opts))
}

// Chat streams the speculative completion of a multi-turn conversation, framed
// with the gemma turn template (byte-identical to the plain engine path's
// formatChatTurns — a trailing open model turn to complete).
func (m *speculativeModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.speculate(ctx, m.tok.Encode(formatSpeculativeChatTurns(messages)), inference.ApplyGenerateOpts(opts))
}

// speculate runs the AssistantPair MTP loop over the target session, decoding
// and yielding each emitted token (accepted drafts + fallbacks) and blanking a
// terminator's text so it never leaks as a literal "<end_of_turn>". It records
// the run's speculative + generate metrics for the metric providers.
func (m *speculativeModel) speculate(ctx context.Context, ids []int32, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		start := time.Now()
		if ctx == nil {
			ctx = context.Background()
		}
		if len(ids) == 0 {
			m.setErr(core.NewError("native.speculativeModel.Generate: empty prompt after tokenisation"))
			return
		}
		maxNew := cfg.MaxTokens
		if maxNew <= 0 {
			maxNew = 256
		}
		eos := int32(-1)
		if m.tok.HasEOSToken() {
			eos = m.tok.EOS()
		}
		stop := append([]int32(nil), cfg.StopTokens...)
		if eos >= 0 {
			stop = append(stop, eos)
		}
		sink := func(id int32) bool {
			if ctx.Err() != nil {
				return false
			}
			text := m.tok.DecodeToken(id)
			if speculativeTokenInSet(id, stop) {
				// A terminator's text is never content — gemma4 ships
				// <end_of_turn> as a plain vocab token, so blank it (the id
				// still surfaces for trackers), mirroring the plain decode path.
				text = ""
			}
			return yield(inference.Token{ID: id, Text: text})
		}
		res, err := m.pair.GenerateFromSessionEach(m.target, ids, maxNew, int(eos), m.draftBlock, cfg.SuppressTokens, sink)
		m.record(res, len(ids), time.Since(start), err)
	}
}

// record folds one speculative run's counters into the metric surfaces the
// bench + trace read.
func (m *speculativeModel) record(res AssistantGenerateResult, promptLen int, dur time.Duration, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		m.lastErr = err
	}
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:    promptLen,
		GeneratedTokens: len(res.Tokens),
		TotalDuration:   dur,
		DecodeDuration:  dur,
	}
	spec := inference.SpeculativeMetrics{
		DraftTokenSchedule: res.DraftTokenSchedule,
		ProposedTokens:     res.DraftTokens,
		AcceptedTokens:     res.AcceptedTokens,
		RejectedTokens:     res.RejectedTokens,
		TargetVerifyCalls:  res.TargetVerifyCalls,
		TargetCalls:        res.TargetCalls,
		DraftCalls:         res.DraftCalls,
		WallDuration:       dur,
	}
	if spec.ProposedTokens > 0 {
		spec.AcceptanceRate = float64(spec.AcceptedTokens) / float64(spec.ProposedTokens)
	}
	m.lastSpec = spec
}

// SpeculativeMetrics reports the last run's draft/verify counters (inference
// .SpeculativeMetricsProvider) — what printMTPMetrics reads for the -draft bench
// line.
func (m *speculativeModel) SpeculativeMetrics() inference.SpeculativeMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastSpec
}

// Metrics reports the last generate run's token/timing counters.
func (m *speculativeModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}

// Err reports the last run's error (OK when the last run succeeded).
func (m *speculativeModel) Err() core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastErr != nil {
		return core.Fail(m.lastErr)
	}
	return core.Ok(nil)
}

// ModelType is the target architecture identifier.
func (m *speculativeModel) ModelType() string { return m.modelType }

// Info is the target model's neutral metadata.
func (m *speculativeModel) Info() inference.ModelInfo { return m.info }

// Classify is not offered on the speculative path — speculation accelerates
// autoregressive generation, not batched prefill-only classification. Callers
// wanting Classify load the target plainly.
func (m *speculativeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("native.speculativeModel: Classify is not supported on the speculative path — load the target model plainly"))
}

// BatchGenerate is not offered on the speculative path — the MTP loop is a
// single-sequence draft/verify lane.
func (m *speculativeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("native.speculativeModel: BatchGenerate is not supported on the speculative path — load the target model plainly"))
}

// Close releases the drafter pair and the target session.
func (m *speculativeModel) Close() core.Result {
	var first error
	if m.pair != nil {
		if err := m.pair.Close(); err != nil {
			first = err
		}
	}
	if m.target != nil {
		if err := m.target.Close(); err != nil && first == nil {
			first = err
		}
	}
	if first != nil {
		return core.Fail(first)
	}
	return core.Ok(nil)
}

func (m *speculativeModel) setErr(err error) {
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

// speculativeTokenInSet reports whether id is one of set — the terminator check
// the sink uses to blank a stop token's text.
func speculativeTokenInSet(id int32, set []int32) bool {
	return slices.Contains(set, id)
}

// formatSpeculativeChatTurns renders the gemma turn template, byte-identical to
// the engine package's formatChatTurns (which is unexported there): each turn as
// "<start_of_turn>ROLE\nCONTENT<end_of_turn>\n", then a trailing open model turn
// to complete. Kept here because package native cannot reach engine's copy.
func formatSpeculativeChatTurns(messages []inference.Message) string {
	var out strings.Builder
	for _, msg := range messages {
		out.WriteString("<start_of_turn>" + speculativeChatRole(msg.Role) + "\n" + msg.Content + "<end_of_turn>\n")
	}
	return out.String() + "<start_of_turn>model\n"
}

func speculativeChatRole(role string) string {
	if role == "assistant" || role == "model" {
		return "model"
	}
	return "user"
}
