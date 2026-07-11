// SPDX-Licence-Identifier: EUPL-1.2

// Package continuity is the serve-side no-prompt-replay conversation loop
// (#370): each stateless chat request is matched to the conversation whose
// retained KV state covers its message prefix, woken (RAM-resident first,
// state-store second), appended with ONLY the new turns, and slept back after
// the turn. Without it every turn of a deep conversation re-pays the full
// prefill (measured: 12s at 32K context, 54s at 98K — decode itself holds
// ~105 tok/s); with it the prefill is paid once per conversation.
//
// Ported from lthn-mlx's conversation_continuity.go, rebuilt engine-neutral:
// sessions are the inference.SessionFactory handles wrapped by
// model/state/session for the wake/sleep machinery, and generation streams
// RAW tokens from the handle itself — byte-identical to the stateless path,
// so the compat handlers' ThinkingExtractor sees exactly the stream it
// already parses. The manager attaches through the engine's chat-interceptor
// seam and DECLINES to the stateless path whenever anything is unusual
// (media turns, explicit thinking overrides, mid-turn conversations, wake
// failures) — continuity never breaks serving, it only removes replay.
package continuity

import (
	"context"
	"iter"
	"slices"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/model/spine"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session"
)

// entryPrefix namespaces conversation state entry URIs in the store.
const entryPrefix = "lem://conversation/"

// defaultMaxResident caps RAM-resident conversations; older conversations are
// closed on eviction and wake from the store on their next turn.
const defaultMaxResident = 4

// chatFormatter is the model's chat-framing seam: a fresh conversation frames
// with the full turn template honouring the request's thinking flag, a woken
// one appends only the new turns — byte-identical to the framing the
// stateless serve path applies for the same flag.
type chatFormatter interface {
	FormatChatPromptWithThinking(messages []inference.Message, enableThinking *bool) string
	FormatChatContinuationWithThinking(messages []inference.Message, enableThinking *bool) string
}

// chatInterceptable is the engine seam Enable installs the manager into: the
// model offers every text-only Chat to the interceptor first and serves the
// request statelessly when it declines.
type chatInterceptable interface {
	SetChatInterceptor(fn func(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) (iter.Seq[inference.Token], bool))
}

type modelInfoReporter interface{ Info() inference.ModelInfo }
type maxLenReporter interface{ MaxLen() int }

// chatMetricsRecorder is the model's usage seam: continuity turns bypass the
// stateless stream that normally records Metrics, so the manager reports the
// turn's honest counts (tail prefill + generated) through it.
type chatMetricsRecorder interface {
	RecordChatMetrics(promptTokens, generated int, start, decodeStart time.Time)
}

// positionReporter measures a session's retained token position — the prefill
// delta across an append is the turn's prompt-token count.
type positionReporter interface{ Pos() int }

// Enable wires the no-prompt-replay loop into a loaded model, backed by
// store. It is the serving.ContinuityEnabler for the composition root:
//
//	serving.RunServe(ctx, serving.ServeConfig{..., EnableContinuity: continuity.Enable})
//
// Errors report which capability is missing; serve degrades to stateless.
func Enable(model inference.TextModel, store state.Store) error {
	_, err := EnableWithManager(model, store)
	return err
}

// EnableWithManager is Enable returning the manager (stats introspection in
// tests and diagnostics).
func EnableWithManager(model inference.TextModel, store state.Store) (*Manager, error) {
	if model == nil {
		return nil, core.E("continuity.Enable", "model is nil", nil)
	}
	if store == nil {
		return nil, core.E("continuity.Enable", "state store is nil", nil)
	}
	factory, ok := model.(inference.SessionFactory)
	if !ok {
		return nil, core.E("continuity.Enable", "model exposes no session factory", nil)
	}
	formatter, ok := model.(chatFormatter)
	if !ok {
		return nil, core.E("continuity.Enable", "model exposes no chat-framing seam", nil)
	}
	interceptable, ok := model.(chatInterceptable)
	if !ok {
		return nil, core.E("continuity.Enable", "model exposes no chat-interceptor seam", nil)
	}
	writer, ok := store.(state.Writer)
	if !ok {
		return nil, core.E("continuity.Enable", "state store does not implement state.Writer", nil)
	}
	var info inference.ModelInfo
	if r, ok := model.(modelInfoReporter); ok {
		info = r.Info()
	}
	contextLen := 0
	if r, ok := model.(maxLenReporter); ok {
		contextLen = r.MaxLen()
	}
	m := &Manager{
		factory:   factory,
		formatter: formatter,
		store:     store,
		writer:    writer,
		info:      spineModelInfo(info, contextLen),
		max:       defaultMaxResident,
		resident:  make(map[string]*residentConversation, defaultMaxResident),
	}
	if rec, ok := model.(chatMetricsRecorder); ok {
		m.metrics = rec
	}
	interceptable.SetChatInterceptor(m.Chat)
	return m, nil
}

// Stats counts the paths conversation turns took — the boot notice and tests
// read these.
type Stats struct {
	FreshConversations int // prefilled from scratch (no matching state)
	ResidentTurns      int // continued on a RAM-resident session
	StoreWakes         int // woken from the state store
	Sleeps             int // turns slept to the store
	StatelessFallbacks int // requests declined to the stateless path
}

// Manager keeps conversations resident across stateless chat requests.
type Manager struct {
	factory   inference.SessionFactory
	formatter chatFormatter
	store     state.Store
	writer    state.Writer
	info      spine.ModelInfo
	max       int

	metrics chatMetricsRecorder // nil when the model exposes no usage seam

	mu       sync.Mutex
	resident map[string]*residentConversation
	order    []string // oldest first, for eviction
	stats    Stats
}

// residentConversation is one conversation's retained session plus the
// previous turn's slept URIs (the incremental-sleep parent chain).
type residentConversation struct {
	handle inference.SessionHandle
	sess   *session.Session
	busy   bool
	dead   bool

	parentEntry  string
	parentBundle string
	parentIndex  string
}

func (c *residentConversation) close() {
	if c.sess != nil {
		_ = c.sess.Close()
	} else if c.handle != nil {
		_ = c.handle.Close()
	}
}

// Stats returns a snapshot of the turn-path counters.
func (m *Manager) Stats() Stats {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.stats
}

// Chat runs one continuity turn and reports whether it accepted the request.
// A false return means the caller serves the request statelessly — continuity
// never breaks serving; the stateless path is always correct, just slower.
func (m *Manager) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) (iter.Seq[inference.Token], bool) {
	if m == nil || len(messages) == 0 {
		return nil, false
	}
	if messagesCarryMedia(messages) {
		// media turns ride the stateless multimodal lane — woken KV carries
		// text turns only.
		return m.declineStat(), false
	}
	cfg := inference.ApplyGenerateOpts(opts)
	// The thinking mode is a conversation-level property (the first system
	// turn's <|think|> switch, retained in the woken prefix), so it is part
	// of the conversation key: a mid-conversation flip misses the resident
	// prefix and starts a correctly framed fresh conversation.
	thinking := cfg.EnableThinking != nil && *cfg.EnableThinking
	conv, tailStart, err := m.acquire(ctx, messages, thinking)
	if err != nil {
		core.Error("continuity declined; serving statelessly", "error", err)
		return m.declineStat(), false
	}

	// Prefill before committing to the streamed sequence so failures here
	// still fall back to the stateless path.
	turnStart := time.Now()
	posBefore := 0
	if p, ok := conv.handle.(positionReporter); ok {
		posBefore = p.Pos()
	}
	var prefillErr error
	if tailStart == 0 {
		prefillErr = conv.handle.Prefill(ctx, m.formatter.FormatChatPromptWithThinking(messages, cfg.EnableThinking))
	} else {
		prefillErr = conv.handle.AppendPrompt(ctx, m.formatter.FormatChatContinuationWithThinking(messages[tailStart:], cfg.EnableThinking))
	}
	if prefillErr != nil {
		core.Error("continuity prefill failed; serving statelessly", "error", prefillErr)
		conv.close()
		return m.declineStat(), false
	}
	promptTokens := 0
	if p, ok := conv.handle.(positionReporter); ok {
		promptTokens = p.Pos() - posBefore
	}

	return func(yield func(inference.Token) bool) {
		decodeStart := time.Now()
		reply := core.NewBuilder()
		generated := 0
		for token := range conv.handle.Generate(ctx, cfg) {
			reply.WriteString(token.Text)
			generated++
			if !yield(token) {
				break
			}
		}
		if m.metrics != nil {
			m.metrics.RecordChatMetrics(promptTokens, generated, turnStart, decodeStart)
		}
		if err := conv.handle.Err(); err != nil {
			core.Error("continuity generation failed", "error", err)
			conv.dead = true
		}
		// A client that disconnected mid-stream received exactly the tokens
		// generated so far, so its next request's prefix matches the partial
		// state — sleeping it is correct, not a compromise.
		m.finishTurn(ctx, conv, messages, reply.String(), thinking)
	}, true
}

// declineStat bumps the stateless-fallback counter and returns nil for the
// (nil, false) decline shape.
func (m *Manager) declineStat() iter.Seq[inference.Token] {
	m.mu.Lock()
	m.stats.StatelessFallbacks++
	m.mu.Unlock()
	return nil
}

// acquire resolves the session a request rides: RAM-resident match, store
// wake, or a fresh session. tailStart is the index of the first message that
// still needs prefilling (0 = the whole conversation).
func (m *Manager) acquire(ctx context.Context, messages []inference.Message, thinking bool) (*residentConversation, int, error) {
	split := conversationTurnSplit(messages)
	if split == len(messages) {
		return nil, 0, core.E("continuity", "request has no trailing user turn", nil)
	}

	if split > 0 {
		key := conversationKey(messages[:split], thinking)
		m.mu.Lock()
		if conv := m.resident[key]; conv != nil {
			if conv.busy {
				m.mu.Unlock()
				return nil, 0, core.E("continuity", "conversation is mid-turn", nil)
			}
			conv.busy = true
			delete(m.resident, key)
			m.removeOrderLocked(key)
			m.stats.ResidentTurns++
			m.mu.Unlock()
			return conv, split, nil
		}
		m.mu.Unlock()

		entryURI := entryPrefix + key
		indexURI := entryURI + "/index"
		if _, idxErr := agent.LoadStateIndex(ctx, m.store, indexURI); idxErr == nil {
			conv, err := m.newConversation()
			if err != nil {
				return nil, 0, err
			}
			if _, err := conv.sess.WakeAgentMemory(ctx, m.store, agent.WakeOptions{IndexURI: indexURI, EntryURI: entryURI}); err != nil {
				conv.close()
				return nil, 0, core.E("continuity", "wake conversation state", err)
			}
			conv.parentEntry = entryURI
			conv.parentBundle = entryURI + "/bundle"
			conv.parentIndex = indexURI
			m.mu.Lock()
			m.stats.StoreWakes++
			m.mu.Unlock()
			return conv, split, nil
		} else {
			var notFound *state.URIChunkNotFoundError
			if !core.As(idxErr, &notFound) {
				return nil, 0, core.E("continuity", "probe conversation state", idxErr)
			}
		}
	}

	conv, err := m.newConversation()
	if err != nil {
		return nil, 0, err
	}
	m.mu.Lock()
	m.stats.FreshConversations++
	m.mu.Unlock()
	return conv, 0, nil
}

// newConversation opens a fresh engine session and its state-machinery view.
func (m *Manager) newConversation() (*residentConversation, error) {
	handle := m.factory.NewSession()
	if handle == nil {
		return nil, core.E("continuity", "model returned a nil session", nil)
	}
	return &residentConversation{
		handle: handle,
		sess:   session.New(handle, m.info, nil),
		busy:   true,
	}, nil
}

// finishTurn sleeps the grown state under the key the NEXT request will look
// up (the conversation including this turn's reply), re-registers the session
// RAM-resident, and evicts beyond the cap. Sleep failure keeps the
// conversation RAM-resident only — turns keep working, durability resumes on
// the next successful sleep.
func (m *Manager) finishTurn(ctx context.Context, conv *residentConversation, messages []inference.Message, reply string, thinking bool) {
	if conv.dead {
		conv.close()
		return
	}
	full := append(slices.Clone(messages), inference.Message{Role: "assistant", Content: reply})
	key := conversationKey(full, thinking)
	entryURI := entryPrefix + key
	sleepOpts := agent.SleepOptions{EntryURI: entryURI, Title: "conversation"}
	if conv.parentEntry != "" {
		sleepOpts.ParentEntryURI = conv.parentEntry
		sleepOpts.ParentBundleURI = conv.parentBundle
		sleepOpts.ParentIndexURI = conv.parentIndex
		sleepOpts.ReuseParentPrefix = true
		// The parent IS this session's own prior sleep and the session is
		// append-only between turns — the prefix is identical by
		// construction, so the sleep captures only the new turn's blocks.
		sleepOpts.ReuseParentPrefixTrusted = true
	}
	if report, err := conv.sess.SleepAgentMemory(ctx, m.writer, sleepOpts); err != nil {
		core.Error("continuity sleep failed; conversation stays RAM-resident only", "error", err)
	} else {
		conv.parentEntry = report.EntryURI
		conv.parentBundle = report.BundleURI
		conv.parentIndex = report.IndexURI
		m.mu.Lock()
		m.stats.Sleeps++
		m.mu.Unlock()
	}

	m.mu.Lock()
	conv.busy = false
	m.resident[key] = conv
	m.order = append(m.order, key)
	for len(m.order) > m.max {
		oldest := m.order[0]
		evicted := m.resident[oldest]
		if evicted == nil || evicted.busy {
			break
		}
		m.order = m.order[1:]
		delete(m.resident, oldest)
		evicted.close()
	}
	m.mu.Unlock()
}

func (m *Manager) removeOrderLocked(key string) {
	for i, existing := range m.order {
		if existing == key {
			m.order = append(m.order[:i], m.order[i+1:]...)
			return
		}
	}
}

// conversationTurnSplit returns the index where the request's new turn
// begins: the trailing run of user/tool messages. Everything before it is the
// prefix a prior turn's retained state covers.
func conversationTurnSplit(messages []inference.Message) int {
	end := len(messages)
	for end > 0 {
		switch normaliseRole(messages[end-1].Role) {
		case "user", "tool":
			end--
		default:
			return end
		}
	}
	return end
}

// conversationKey hashes a message prefix (plus the conversation's thinking
// mode — a framing property of the retained prefix) into the state key a
// finished turn stores under and the next request looks up by.
func conversationKey(messages []inference.Message, thinking bool) string {
	builder := core.NewBuilder()
	// Presize to the exact byte budget so the prefix assembles in one
	// allocation rather than the builder's doubling growth — a deep
	// conversation is hashed twice per turn (acquire + sleep). len(Role) is an
	// upper bound for the normalised role (Trim only shrinks, Lower preserves
	// ASCII length), and Grow is a capacity hint, so the key is byte-identical.
	size := 0
	if thinking {
		size += len("think\x02")
	}
	for _, msg := range messages {
		size += len(msg.Role) + len(msg.Content) + 2
	}
	builder.Grow(size)
	if thinking {
		builder.WriteString("think\x02")
	}
	for _, msg := range messages {
		builder.WriteString(normaliseRole(msg.Role))
		builder.WriteString("\x00")
		builder.WriteString(msg.Content)
		builder.WriteString("\x01")
	}
	return bundle.HashString(builder.String())
}

// normaliseRole folds request role spellings for splitting and hashing.
func normaliseRole(role string) string {
	return core.Lower(core.Trim(role))
}

// messagesCarryMedia reports whether any turn attaches images, audio, or
// video — those ride the stateless multimodal lane.
func messagesCarryMedia(messages []inference.Message) bool {
	for _, msg := range messages {
		if len(msg.Images) > 0 || len(msg.Audios) > 0 || len(msg.Videos) > 0 {
			return true
		}
	}
	return false
}

// spineModelInfo mirrors decode/generate's mapping of the neutral model info
// onto the session package's spine.ModelInfo.
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
