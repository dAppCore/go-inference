// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

var orchestratorSnapshotSink core.Result

func BenchmarkOrchestrator_Snapshot(b *testing.B) {
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	store := newOrchestratorTestStore(at)
	for index := 0; index < 32; index++ {
		run := storeTestRun(work.RunCompleted)
		run.ID = core.Sprintf("run-%03d", index)
		run.WorkID = "work-benchmark"
		store.runs[run.ID] = run
	}
	orchestrator := &Orchestrator{store: store}
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		orchestratorSnapshotSink = orchestrator.Snapshot(context.Background(), "work-benchmark")
	}
}
