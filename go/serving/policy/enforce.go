// SPDX-Licence-Identifier: EUPL-1.2

package policy

import core "dappco.re/go"

// Event records one enforcement for the audit trail: which rule fired and the
// action taken. It NEVER carries the matched content — the deployment may
// consider the match itself sensitive, so only the rule index and action are
// ever surfaced.
type Event struct {
	RuleIndex int
	Action    ActionKind
}

// Enforcer applies a Policy to a SINGLE output stream. It is stateful — it holds
// back the minimal tail of the stream that could still begin a match — and is
// NOT safe for concurrent use: one Enforcer per reply (NewEnforcer per Chat).
//
// The hold-back bound is Policy.HoldBack() bytes: a term match reaches at most
// its own length, a pattern match at most its window, so a match spanning a
// token boundary can start no earlier than HoldBack()-1 bytes from the end of
// what has arrived. Bytes settled beyond that reach stream through UNCHANGED and
// promptly; only the disputable tail is withheld until the next chunk resolves
// it (or Close flushes it at end of stream).
type Enforcer struct {
	pol     *Policy
	tail    []byte // held-back bytes pending disambiguation (capacity reused)
	scan    []byte // tail+chunk join buffer (capacity reused)
	stopped bool   // a refuse rule fired; everything after is swallowed
}

// NewEnforcer returns a fresh Enforcer for one reply stream.
//
//	enf := pol.NewEnforcer()
//	for tok := range modelStream {
//	    out, events, stop := enf.Feed(tok.Text)
//	    emit(out); audit(events)
//	    if stop { break } // refuse ended the reply
//	}
//	if out, events, _ := enf.Close(); out != "" { emit(out); audit(events) }
func (p *Policy) NewEnforcer() *Enforcer { return &Enforcer{pol: p} }

// Feed processes the next chunk of model output. It returns the text to emit
// downstream (a possibly-empty run of settled, enforced bytes), the enforcement
// events that fired (nil on the clean hot path — no allocation), and stop: true
// once a refuse rule has ended the reply, after which every further Feed returns
// ("", nil, true).
//
// Byte-exactness: a chunk with no match and no match-prefix in its tail streams
// straight through as the SAME string (no copy). A match spanning this chunk and
// a later one is withheld until it is disambiguated.
func (e *Enforcer) Feed(chunk string) (out string, events []Event, stop bool) {
	if e.stopped {
		return "", nil, true
	}
	if chunk == "" {
		return "", nil, false
	}

	var buf string
	if len(e.tail) == 0 {
		buf = chunk // zero-copy: emitted runs are substrings of the original chunk
	} else {
		e.scan = append(e.scan[:0], e.tail...)
		e.scan = append(e.scan, chunk...)
		buf = string(e.scan)
	}

	holdFrom := e.pol.holdFrom(buf)
	out, events, consumed, stop := e.pol.process(buf, holdFrom, &e.stopped)
	e.tail = append(e.tail[:0], buf[consumed:]...)
	return out, events, stop
}

// Close flushes any held-back tail at end of stream. With no more input a held
// prefix cannot complete, so leftover bytes emit as-is and any match still
// wholly present is applied. After Close the Enforcer is spent.
func (e *Enforcer) Close() (out string, events []Event, stop bool) {
	if e.stopped || len(e.tail) == 0 {
		return "", nil, e.stopped
	}
	buf := string(e.tail)
	out, events, _, stop = e.pol.process(buf, len(buf), &e.stopped)
	e.tail = e.tail[:0]
	return out, events, stop
}

// match is the winning rule at a scan position: its rule index, action, and the
// byte offset one past the matched span.
type match struct {
	ruleIndex int
	action    ActionKind
	end       int
}

// process settles every match whose start lies before holdFrom (each such start
// has its full match window present, so the decision is final and byte-identical
// to running the policy over the whole text at once) and emits the settled
// prefix. It returns the emitted text, the enforcement events, the consumed byte
// count (the caller withholds buf[consumed:]), and stop when a refuse fired.
//
// The clean hot path — no match settled — returns buf[:consumed] as a substring
// (zero allocation when the caller passed the original chunk).
func (p *Policy) process(buf string, holdFrom int, stopped *bool) (out string, events []Event, consumed int, stop bool) {
	n := len(buf)
	var b core.Builder
	rewrote := false
	lastEmit := 0
	i := 0
	for i < holdFrom {
		m, ok := p.longestMatchAt(buf, i, n)
		if !ok {
			i++
			continue
		}
		b.WriteString(buf[lastEmit:i])
		rewrote = true
		events = append(events, Event{RuleIndex: m.ruleIndex, Action: m.action})
		if m.action == ActionRefuse {
			b.WriteString(p.rules[m.ruleIndex].Message)
			*stopped = true
			return b.String(), events, n, true
		}
		b.WriteString(p.rules[m.ruleIndex].Replacement)
		i = m.end
		lastEmit = i
	}
	// A match settled before holdFrom may have consumed PAST it; the settle
	// point is whichever is later.
	settle := holdFrom
	if lastEmit > settle {
		settle = lastEmit
	}
	if !rewrote {
		return buf[:settle], nil, settle, false
	}
	b.WriteString(buf[lastEmit:settle])
	return b.String(), events, settle, false
}

// longestMatchAt returns the winning rule anchored at position i, if any:
// longest match wins, and an equal-length tie resolves to the earliest rule in
// the config. Callers reach it only where the full match window is present
// (i < holdFrom, or at Close), so the decision is exact.
func (p *Policy) longestMatchAt(buf string, i, n int) (match, bool) {
	bestLen := 0
	bestIdx := -1

	// Terms: dispatched by ASCII-folded first byte, so a position that starts no
	// term costs a single slice lookup — the clean hot path.
	for _, idx := range p.firstByte[asciiLowerByte(buf[i])] {
		r := &p.rules[idx]
		L := len(r.lower)
		if i+L <= n && asciiFoldEqual(buf[i:i+L], r.lower) {
			if L > bestLen || (L == bestLen && (bestIdx < 0 || idx < bestIdx)) {
				bestLen = L
				bestIdx = idx
			}
		}
	}

	// Patterns can begin at any position, so each is probed here; the window
	// bounds the slice the regexp sees.
	for _, idx := range p.patternIdx {
		r := &p.rules[idx]
		end := i + r.window
		if end > n {
			end = n
		}
		loc := r.re.FindStringIndex(buf[i:end])
		if loc == nil || loc[0] != 0 {
			continue
		}
		L := loc[1]
		if L > bestLen || (L == bestLen && (bestIdx < 0 || idx < bestIdx)) {
			bestLen = L
			bestIdx = idx
		}
	}

	if bestIdx < 0 {
		return match{}, false
	}
	return match{ruleIndex: bestIdx, action: p.rules[bestIdx].Action, end: i + bestLen}, true
}

// holdFrom is the earliest byte offset the enforcer must withhold: the earliest
// position in the tail that could still begin a match given more input. Positions
// before it are settled; buf[holdFrom:] is held back.
//
//   - Terms: the earliest tail offset that is a PROPER prefix of some term (an
//     as-yet-incomplete match). Detected by first-byte dispatch, so ordinary
//     text that never approaches a term holds nothing back — the tail empties and
//     the next chunk streams through with no copy.
//   - Patterns: any pattern match is at most maxWindow bytes, so an incomplete
//     one can only begin within the last maxWindow-1 bytes; those are withheld
//     whenever pattern rules exist.
func (p *Policy) holdFrom(buf string) int {
	n := len(buf)
	hold := n

	if p.maxTermLen > 1 {
		start := n - (p.maxTermLen - 1)
		if start < 0 {
			start = 0
		}
		for j := start; j < n; j++ {
			if p.termPrefixAt(buf, j, n) {
				hold = j
				break
			}
		}
	}

	if p.maxWindow > 0 {
		ph := n - (p.maxWindow - 1)
		if ph < 0 {
			ph = 0
		}
		if ph < hold {
			hold = ph
		}
	}
	return hold
}

// termPrefixAt reports whether buf[j:n] is a PROPER prefix of some term — an
// incomplete term match that must be withheld pending more bytes.
func (p *Policy) termPrefixAt(buf string, j, n int) bool {
	for _, idx := range p.firstByte[asciiLowerByte(buf[j])] {
		r := &p.rules[idx]
		avail := n - j
		if avail < len(r.lower) && asciiFoldEqual(buf[j:n], r.lower[:avail]) {
			return true
		}
	}
	return false
}
