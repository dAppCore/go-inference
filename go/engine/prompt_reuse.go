// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"os"
)

// prompt_reuse.go is the stateless lane's resident-session prompt cache
// (#377). The stateless text path historically opened a fresh session per
// request, so every turn of a multi-turn chat re-prefilled the whole history —
// a per-turn cost that climbs with the conversation while llama.cpp's slot
// cache pays only the new turn. A [TextModel] now keeps ONE resident session
// when the engine session implements [PromptReuseSession]; stream() prefills
// through it and the longest shared prefix is reused in place.
//
// Scope guards:
//   - single slot, TryLock — a busy resident session sends the request down
//     the fresh-session path unchanged (no queueing behind the cache);
//   - conversation continuity owns caching when wired (a chat interceptor is
//     installed) — the resident lane stands down so session KV memory is not
//     held twice;
//   - LTHN_PROMPT_REUSE=0 kills the lane (fresh session per request, the
//     pre-#377 behaviour).

// promptReuseEnabled is the lane's kill switch, read once at start-up.
// Tests flip it directly (same package) rather than racing the process env.
var promptReuseEnabled = os.Getenv("LTHN_PROMPT_REUSE") != "0"

// PromptReuseSession is the optional engine [Session] capability behind the
// resident-session prompt cache: PrefillTokensCached reuses the retained cache
// where ids share a prefix with the rows already resident, prefills only the
// divergent suffix, and reports the reused prefix length. A cold call (nothing
// reusable, or reuse unsafe) must leave the session exactly as PrefillTokens
// would — token-identical decode either way.
type PromptReuseSession interface {
	Session
	PrefillTokensCached(ids []int32) (int, error)
}

// PromptReuseCapableModel is the model-level declaration that this engine's
// sessions implement [PromptReuseSession]. The lane probes the MODEL (a cheap
// type assertion, no side effects) rather than opening a throwaway session to
// probe the capability — an engine that does not declare it never pays a
// session open for the question.
type PromptReuseCapableModel interface {
	// SessionsReusePrompts reports that sessions from OpenEngineSession
	// implement [PromptReuseSession].
	SessionsReusePrompts() bool
}

// acquireReuseSession hands back the resident reuse session when the lane is
// available: enabled, no continuity layer installed, not held by a concurrent
// request, and the engine session implements [PromptReuseSession] (probed once
// — an engine without the capability parks the lane for the model's lifetime).
// The returned release func frees the slot; ok=false means take the
// fresh-session path.
func (m *TextModel) acquireReuseSession() (PromptReuseSession, func(), bool) {
	if !promptReuseEnabled || m.reuseUnsupported.Load() || m.chatIntercept.Load() != nil {
		return nil, nil, false
	}
	if d, ok := m.tm.(PromptReuseCapableModel); !ok || !d.SessionsReusePrompts() {
		return nil, nil, false
	}
	if !m.reuseMu.TryLock() {
		return nil, nil, false
	}
	if m.reuseSess == nil {
		sess, err := m.openSession()
		if err != nil {
			m.reuseMu.Unlock()
			return nil, nil, false
		}
		rs, ok := sess.(PromptReuseSession)
		if !ok {
			_ = sess.Close()
			m.reuseUnsupported.Store(true)
			m.reuseMu.Unlock()
			return nil, nil, false
		}
		m.reuseSess = rs
	}
	return m.reuseSess, func() { m.reuseMu.Unlock() }, true
}

// dropReuseSession closes and clears the resident session so the next request
// opens a fresh one — the poisoned-state escape hatch after a failed cached
// prefill. The caller holds the slot lock.
func (m *TextModel) dropReuseSession() {
	if m.reuseSess != nil {
		_ = m.reuseSess.Close()
		m.reuseSess = nil
	}
}

// closeReuseSession releases the resident session at model close. It takes the
// slot lock, so an in-flight request finishes before the session goes away.
func (m *TextModel) closeReuseSession() {
	m.reuseMu.Lock()
	defer m.reuseMu.Unlock()
	m.dropReuseSession()
}
