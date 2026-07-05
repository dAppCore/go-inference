// SPDX-Licence-Identifier: EUPL-1.2

// Package enginetest is the engine conformance kit: one reusable suite every
// inference engine (metal, hip, cpu…) runs against its own implementation of
// the inference contracts. An engine imports this package IN ITS TESTS and
// hands the suite a factory; the suite exercises the lifecycle, error, and
// shape invariants that must hold for ANY conformant implementation —
// contract-level checks, deliberately independent of model quality or output
// content. Optional capabilities (KV restore, capture-with-options…) are
// probed exactly as production callers probe them: present ⇒ exercised,
// absent ⇒ skipped and reported.
//
//	func TestConformance_SessionHandle(t *testing.T) {
//	    enginetest.SessionHandle(t, func(t *testing.T) inference.SessionHandle {
//	        return newTestSession(t) // engine-provided; may use a synthetic model
//	    })
//	}
//
// The suite never loads models itself: engines choose the cheapest fixture
// that exercises their real code path (a tiny synthetic checkpoint, a fake —
// their call). Determinism- or content-sensitive assertions do not belong
// here; they stay engine-side where the fixture's properties are known.
package enginetest

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

// SessionFactory builds a fresh, unused session for one subtest. The suite
// owns the returned handle's lifecycle (it will Close it); the factory is
// called once per subtest so state never leaks between checks.
type SessionFactory func(t *testing.T) inference.SessionHandle

// drain ranges a generation to completion and returns the tokens seen.
func drain(ctx context.Context, s inference.SessionHandle, cfg inference.GenerateConfig) []inference.Token {
	var out []inference.Token
	for tok := range s.Generate(ctx, cfg) {
		out = append(out, tok)
	}
	return out
}
