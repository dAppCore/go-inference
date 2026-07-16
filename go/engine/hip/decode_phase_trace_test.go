// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

func TestHIPDecodePhaseTrace_Good(t *testing.T) {
	session := &hipEngineSession{tokens: make([]int32, 8)}
	var tracer engine.DecodePhaseTracer = session
	stop := tracer.BeginDecodePhaseTrace()

	session.mu.Lock()
	session.tokens = append(session.tokens, 1, 2, 3)
	session.mu.Unlock()
	budget := stop()

	core.AssertEqual(t, 3, budget.Tokens)
	core.AssertTrue(t, budget.TotalPerToken > 0)
	core.AssertEqual(t, time.Duration(0), budget.GPUPerToken)
	core.AssertEqual(t, 0, len(budget.Phases))
}

func TestHIPDecodePhaseTrace_Bad(t *testing.T) {
	session := &hipEngineSession{tokens: make([]int32, 8)}
	budget := session.BeginDecodePhaseTrace()()
	core.AssertEqual(t, 0, budget.Tokens)
	core.AssertEqual(t, time.Duration(0), budget.TotalPerToken)
}

func TestHIPDecodePhaseTrace_Ugly(t *testing.T) {
	var session *hipEngineSession
	budget := session.BeginDecodePhaseTrace()()
	core.AssertEqual(t, 0, budget.Tokens)
	core.AssertEqual(t, time.Duration(0), budget.TotalPerToken)
}
