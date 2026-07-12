// SPDX-Licence-Identifier: EUPL-1.2

// Package policy is the outbound-policy layer: a deployment-owned filter on
// what the served model may EMIT. Where the welfare guard protects the model
// from the user (inbound — the model adjudicates), the policy layer enforces
// the house rules on model OUTPUT (outbound — the DEPLOYER adjudicates, via a
// config file). The engine ships the MECHANISM — term/pattern rules with
// redact/refuse actions — and never a taxonomy: there are no built-in term
// lists, only what a deployment declares.
//
// A policy is loaded from JSON (Load / Compile) and compiled ONCE — a bad
// regexp, an empty term, a refuse rule with no message, or an out-of-range
// window is rejected at LOAD, never at serve time. The file shape:
//
//		{
//		  "window": 256,
//		  "rules": [
//		    {"match": "term",    "value": "PROJECT-X", "action": "redact", "replacement": "[redacted]"},
//		    {"match": "pattern", "value": "v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+", "action": "refuse", "message": "This deployment does not discuss unreleased versions."},
//		    {"match": "term",    "value": "client",    "action": "redact"}
//		  ]
//		}
//
//	  - match "term"    — a case-insensitive literal (ASCII case fold; bytes ≥ 0x80
//	    compare exactly).
//	  - match "pattern" — a Go regexp compiled at load. Its match is bounded by a
//	    window (per-rule "window", else the file "window", else DefaultWindow) so
//	    the streaming enforcer can establish a hold-back bound; a pattern that
//	    matches the empty string is rejected (its bound cannot be established).
//	  - action "redact" — replace the matched span with "replacement" (default
//	    "[redacted]").
//	  - action "refuse" — end the reply at the match; "message" becomes the last
//	    visible text (required, non-empty).
//	  - action "rewrite" — grade G2: route the matched span through a
//	    caller-supplied mediator so the reply survives with the span transformed
//	    (see docs/policy-g2-mediated-rewrite.md and enforce.go). "replacement" is
//	    the safe-floor fallback used when the mediator errors, times out, or
//	    returns nothing — the original span is never emitted.
//
// Enforcement is streaming and byte-exact — see enforce.go for the hold-back
// design that keeps boundary-spanning matches correct while letting
// non-matching text stream through unchanged.
package policy

import (
	"regexp"
	"time"

	core "dappco.re/go"
)

// MatchKind selects how a rule's value is interpreted against the output stream.
type MatchKind string

const (
	MatchTerm    MatchKind = "term"    // case-insensitive literal (ASCII fold)
	MatchPattern MatchKind = "pattern" // Go regexp, compiled at load
)

// ActionKind selects what happens when a rule matches the output stream.
type ActionKind string

const (
	ActionRedact  ActionKind = "redact"  // replace the matched span with the replacement
	ActionRefuse  ActionKind = "refuse"  // end the reply at the match; the message is the last text
	ActionRewrite ActionKind = "rewrite" // G2: transform the matched span via a mediator; redact on failure
)

const (
	// DefaultReplacement is the redact replacement when a rule omits one.
	DefaultReplacement = "[redacted]"
	// DefaultWindow is the pattern match window (bytes) when neither the rule
	// nor the file sets one — the bound the streaming enforcer holds back.
	DefaultWindow = 256
	// MaxWindow caps any pattern window: the hold-back (and so the enforcer's
	// buffered tail) can never exceed this many bytes.
	MaxWindow = 1 << 16
	// DefaultMediateTimeout is the per-span mediator deadline (G2 rewrite) when
	// the policy file omits "mediate_timeout_ms". A rewrite hit stalls the stream
	// for at most this long before degrading to redact.
	DefaultMediateTimeout = 5 * time.Second
	// MaxMediateTimeoutMS caps "mediate_timeout_ms": a deployment cannot stall the
	// output stream on a single span for longer than this.
	MaxMediateTimeoutMS = 60000
)

// ruleJSON is the on-disk shape of one rule (see the package doc for the schema).
type ruleJSON struct {
	Match       MatchKind  `json:"match"`
	Value       string     `json:"value"`
	Action      ActionKind `json:"action"`
	Replacement *string    `json:"replacement,omitempty"` // redact only; default DefaultReplacement
	Message     string     `json:"message,omitempty"`     // refuse only; required
	Window      int        `json:"window,omitempty"`      // pattern only; match window in bytes
}

// policyJSON is the on-disk shape of a policy file.
type policyJSON struct {
	Window           int        `json:"window,omitempty"`             // default pattern window (bytes)
	MediateTimeoutMS int        `json:"mediate_timeout_ms,omitempty"` // G2 per-span mediator deadline (ms)
	Rules            []ruleJSON `json:"rules"`
}

// Rule is one compiled outbound-policy rule. The audit trail identifies a rule
// by Index (its 0-based position in the file), never by its content.
type Rule struct {
	Index       int
	Match       MatchKind
	Action      ActionKind
	Replacement string // redact replacement text
	Message     string // refuse message

	lower  string         // term, ASCII-folded to lower once at load
	re     *regexp.Regexp // pattern
	window int            // pattern effective match window (bytes)
}

// Policy is a compiled, ready-to-enforce outbound policy. It is immutable after
// Load/Compile and safe to share across concurrent streams; per-stream state
// lives in the Enforcer (NewEnforcer).
type Policy struct {
	rules          []Rule
	patternIdx     []int         // indices of pattern rules (empty for term-only policies)
	firstByte      [256][]int    // term dispatch: ASCII-folded first byte → rule indices
	maxTermLen     int           // longest term (bytes); 0 if no terms
	maxWindow      int           // largest pattern window (bytes); 0 if no patterns
	hold           int           // streaming hold-back bound (bytes) = max(maxTermLen, maxWindow), ≥ 1
	rewrites       int           // count of rewrite rules (G2); >0 means a mediator is required
	mediateTimeout time.Duration // per-span mediator deadline (G2)
}

// Load reads and compiles the policy file at path. Every fault — unreadable
// file, malformed JSON, unknown match/action, empty term, invalid or
// empty-matching regexp, missing refuse message, out-of-range window — is
// reported here, so a misconfigured deployment fails at BOOT and never serves
// unguarded.
//
//	pol, err := policy.Load("/etc/lem/policy.json")
//	if err != nil { return err } // fatal — do not serve without the policy
func Load(path string) (*Policy, error) {
	r := core.ReadFile(path)
	if !r.OK {
		return nil, core.E("policy.Load", core.Sprintf("read policy file %q", path), r.Err())
	}
	return Compile(r.Value.([]byte))
}

// Compile parses and compiles a policy from JSON bytes — the load path without
// the file read, for callers that embed or synthesise a policy. It performs the
// SAME validation as Load.
//
//	pol, err := policy.Compile([]byte(`{"rules":[{"match":"term","value":"secret","action":"redact"}]}`))
func Compile(data []byte) (*Policy, error) {
	var doc policyJSON
	if r := core.JSONUnmarshal(data, &doc); !r.OK {
		return nil, core.E("policy.Compile", "parse policy JSON", r.Err())
	}
	defWindow := doc.Window
	if defWindow == 0 {
		defWindow = DefaultWindow
	}
	if defWindow < 1 || defWindow > MaxWindow {
		return nil, core.E("policy.Compile", core.Sprintf("policy window %d out of range (1..%d)", doc.Window, MaxWindow), nil)
	}
	pol := &Policy{mediateTimeout: DefaultMediateTimeout}
	if doc.MediateTimeoutMS != 0 {
		if doc.MediateTimeoutMS < 1 || doc.MediateTimeoutMS > MaxMediateTimeoutMS {
			return nil, core.E("policy.Compile", core.Sprintf("mediate_timeout_ms %d out of range (1..%d)", doc.MediateTimeoutMS, MaxMediateTimeoutMS), nil)
		}
		pol.mediateTimeout = time.Duration(doc.MediateTimeoutMS) * time.Millisecond
	}
	for i, rj := range doc.Rules {
		rule, err := compileRule(i, rj, defWindow)
		if err != nil {
			return nil, err
		}
		pol.rules = append(pol.rules, rule)
	}
	pol.index()
	return pol, nil
}

// compileRule validates and compiles one rule, or returns a load error naming
// the offending rule index.
func compileRule(index int, rj ruleJSON, defWindow int) (Rule, error) {
	op := core.Sprintf("policy.Compile: rule #%d", index)
	rule := Rule{Index: index, Match: rj.Match, Action: rj.Action}

	switch rj.Action {
	case ActionRedact, ActionRewrite:
		// Rewrite shares redact's shape: the replacement is the safe-floor
		// fallback used when the mediator fails, and a refuse message is invalid.
		if rj.Message != "" {
			return Rule{}, core.E(op, core.Sprintf("%s rule must not carry a refuse message", rj.Action), nil)
		}
		rule.Replacement = DefaultReplacement
		if rj.Replacement != nil {
			rule.Replacement = *rj.Replacement
		}
	case ActionRefuse:
		if rj.Replacement != nil {
			return Rule{}, core.E(op, "refuse rule must not carry a redact replacement", nil)
		}
		if core.Trim(rj.Message) == "" {
			return Rule{}, core.E(op, "refuse rule requires a non-empty message", nil)
		}
		rule.Message = rj.Message
	default:
		return Rule{}, core.E(op, core.Sprintf("unknown action %q (want redact|refuse|rewrite)", rj.Action), nil)
	}

	switch rj.Match {
	case MatchTerm:
		if rj.Value == "" {
			return Rule{}, core.E(op, "term rule requires a non-empty value", nil)
		}
		if rj.Window != 0 {
			return Rule{}, core.E(op, "window is valid only for pattern rules", nil)
		}
		rule.lower = asciiLower(rj.Value)
	case MatchPattern:
		if rj.Value == "" {
			return Rule{}, core.E(op, "pattern rule requires a non-empty value", nil)
		}
		re, err := regexp.Compile(rj.Value)
		if err != nil {
			return Rule{}, core.E(op, core.Sprintf("compile pattern %q", rj.Value), err)
		}
		if re.MatchString("") {
			return Rule{}, core.E(op, core.Sprintf("pattern %q matches the empty string — anchor it or drop the optional quantifier (its match bound cannot be established)", rj.Value), nil)
		}
		w := rj.Window
		if w == 0 {
			w = defWindow
		}
		if w < 1 || w > MaxWindow {
			return Rule{}, core.E(op, core.Sprintf("pattern window %d out of range (1..%d)", rj.Window, MaxWindow), nil)
		}
		rule.re = re
		rule.window = w
	default:
		return Rule{}, core.E(op, core.Sprintf("unknown match %q (want term|pattern)", rj.Match), nil)
	}
	return rule, nil
}

// index precomputes the term first-byte dispatch, the pattern index list, and
// the streaming hold-back bound. Called once after all rules compile.
func (p *Policy) index() {
	p.hold = 1
	for i := range p.rules {
		r := &p.rules[i]
		if r.Action == ActionRewrite {
			p.rewrites++
		}
		switch r.Match {
		case MatchTerm:
			if len(r.lower) > p.maxTermLen {
				p.maxTermLen = len(r.lower)
			}
			b := asciiLowerByte(r.lower[0])
			p.firstByte[b] = append(p.firstByte[b], i)
		case MatchPattern:
			p.patternIdx = append(p.patternIdx, i)
			if r.window > p.maxWindow {
				p.maxWindow = r.window
			}
		}
	}
	if p.maxTermLen > p.hold {
		p.hold = p.maxTermLen
	}
	if p.maxWindow > p.hold {
		p.hold = p.maxWindow
	}
}

// Len reports the number of compiled rules.
func (p *Policy) Len() int { return len(p.rules) }

// HoldBack reports the streaming hold-back bound in bytes — the largest tail the
// enforcer withholds pending disambiguation (the longest term, or the largest
// pattern window). It is the concrete answer to "how far can a match reach
// across token boundaries".
func (p *Policy) HoldBack() int { return p.hold }

// NeedsMediator reports whether the policy declares any rewrite rule (grade G2).
// A policy that needs a mediator must be wired with one (NewMediatingEnforcer /
// WrapResolverMediated); the serving layer refuses to boot a rewrite policy with
// no mediator, mirroring G1's "refuse to serve unguarded" contract.
func (p *Policy) NeedsMediator() bool { return p.rewrites > 0 }

// MediateTimeout reports the per-span mediator deadline (grade G2): a rewrite hit
// stalls the output stream for at most this long before degrading to redact.
func (p *Policy) MediateTimeout() time.Duration { return p.mediateTimeout }

// asciiLower returns s with ASCII upper-case letters folded to lower. Bytes
// ≥ 0x80 are left untouched — term matching folds ASCII case only.
func asciiLower(s string) string {
	upper := false
	for k := 0; k < len(s); k++ {
		if 'A' <= s[k] && s[k] <= 'Z' {
			upper = true
			break
		}
	}
	if !upper {
		return s
	}
	b := []byte(s)
	for k := range b {
		if 'A' <= b[k] && b[k] <= 'Z' {
			b[k] += 'a' - 'A'
		}
	}
	return string(b)
}

// asciiLowerByte folds a single ASCII byte to lower case.
func asciiLowerByte(b byte) byte {
	if 'A' <= b && b <= 'Z' {
		return b + ('a' - 'A')
	}
	return b
}

// asciiFoldEqual reports whether a equals bLower under ASCII case folding. bLower
// MUST already be ASCII-lowercased (terms are folded once at load); bytes ≥ 0x80
// compare exactly.
func asciiFoldEqual(a, bLower string) bool {
	if len(a) != len(bLower) {
		return false
	}
	for k := 0; k < len(a); k++ {
		ca := a[k]
		if 'A' <= ca && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if ca != bLower[k] {
			return false
		}
	}
	return true
}
