// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

var _ engine.DecodePhaseTracer = (*hipEngineSession)(nil)

// BeginDecodePhaseTrace measures the complete retained-session decode wall.
// GPUPerToken and Phases remain zero until the HIP driver can report portable
// device-event spans; a zero sub-split is more useful than a fabricated one.
func (s *hipEngineSession) BeginDecodePhaseTrace() func() inference.DecodePhaseBudget {
	if s == nil {
		return func() inference.DecodePhaseBudget { return inference.DecodePhaseBudget{} }
	}
	s.mu.Lock()
	startPos := len(s.tokens)
	s.mu.Unlock()
	start := time.Now()
	return func() inference.DecodePhaseBudget {
		s.mu.Lock()
		tokens := len(s.tokens) - startPos
		s.mu.Unlock()
		if tokens <= 0 {
			return inference.DecodePhaseBudget{}
		}
		return inference.DecodePhaseBudget{
			Tokens:        tokens,
			TotalPerToken: time.Since(start) / time.Duration(tokens),
		}
	}
}
