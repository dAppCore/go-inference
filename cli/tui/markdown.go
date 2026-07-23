// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"crypto/sha256"
	"sync"
	"time"

	tea "dappco.re/go/html/tui"
	"dappco.re/go/html/tui/markdown"
)

type markdownCacheKey struct {
	turnID string
	hash   [sha256.Size]byte
	width  int
	theme  string
}

type markdownStats struct {
	Hits   uint64
	Misses uint64
}

// markdownRenderer caches completed turn output and the width-specific
// Glamour renderer behind one lock; Glamour reuses internal buffers and is not
// safe to invoke concurrently.
type markdownRenderer struct {
	mu        sync.Mutex
	theme     string
	cache     map[markdownCacheKey]string
	renderers map[int]*markdown.Renderer
	stats     markdownStats
}

func newMarkdownRenderer(theme string) *markdownRenderer {
	return &markdownRenderer{
		theme:     theme,
		cache:     make(map[markdownCacheKey]string),
		renderers: make(map[int]*markdown.Renderer),
	}
}

// Render returns the cached rich representation of one completed turn. An
// unusable width or renderer failure degrades to the original readable text.
func (renderer *markdownRenderer) Render(turnID, content string, width int) string {
	if renderer == nil || width <= 0 {
		return content
	}
	key := markdownCacheKey{
		turnID: turnID,
		hash:   sha256.Sum256([]byte(content)),
		width:  width,
		theme:  renderer.theme,
	}
	renderer.mu.Lock()
	defer renderer.mu.Unlock()
	if rendered, exists := renderer.cache[key]; exists {
		renderer.stats.Hits++
		return rendered
	}
	renderer.stats.Misses++
	term := renderer.renderers[width]
	if term == nil {
		built, err := markdown.New(
			markdown.WithStandardStyle("dark"),
			markdown.WithWordWrap(width),
			markdown.WithTableWrap(true),
			markdown.WithPreservedNewLines(),
			markdown.WithEmoji(),
		)
		if err != nil {
			renderer.cache[key] = content
			return content
		}
		term = built
		renderer.renderers[width] = term
	}
	rendered, err := term.Render(content)
	if err != nil || rendered == "" {
		rendered = content
	}
	renderer.cache[key] = rendered
	return rendered
}

func (renderer *markdownRenderer) Stats() markdownStats {
	if renderer == nil {
		return markdownStats{}
	}
	renderer.mu.Lock()
	defer renderer.mu.Unlock()
	return renderer.stats
}

const streamRefreshInterval = 40 * time.Millisecond

type streamRefreshMsg struct {
	SessionID string
	JobID     string
}

// waitEventOrRefresh keeps exactly one Bubble Tea command in flight. It
// returns the next stream event, or a bounded refresh signal when tokens are
// arriving faster than the terminal should re-render.
func waitEventOrRefresh(generation *generation, deadline time.Time) tea.Cmd {
	return func() tea.Msg {
		if generation == nil {
			return streamMsg{done: true}
		}
		delay := time.Until(deadline)
		if delay <= 0 {
			return streamRefreshMsg{SessionID: generation.SessionID, JobID: generation.JobID}
		}
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case event, ok := <-generation.events:
			if !ok {
				return streamMsg{SessionID: generation.SessionID, JobID: generation.JobID, done: true}
			}
			return streamMsg(event)
		case <-timer.C:
			return streamRefreshMsg{SessionID: generation.SessionID, JobID: generation.JobID}
		}
	}
}
