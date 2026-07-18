// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

var controllerDecisionBenchmarkSink core.Result

func BenchmarkController_Decide(b *testing.B) {
	controllerResult := NewController(defaultPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	controller := controllerResult.Value.(*Controller)
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	candidate := Candidate{RunID: "run-1", Provider: "codex", QueuedAt: now}
	runtime := Runtime{Now: now}
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		controllerDecisionBenchmarkSink = controller.Decide(candidate, runtime)
	}
}
