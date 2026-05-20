// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"testing"

	core "dappco.re/go"
)

func TestDefaultTuningWorkloads_Good(t *testing.T) {
	workloads := DefaultTuningWorkloads()
	if len(workloads) < 4 {
		t.Fatalf("DefaultTuningWorkloads() len = %d, want at least 4", len(workloads))
	}
	if workloads[0] != TuningWorkloadChat {
		t.Fatalf("first workload = %q, want %q", workloads[0], TuningWorkloadChat)
	}

	workloads[0] = TuningWorkloadThroughput
	next := DefaultTuningWorkloads()
	if next[0] != TuningWorkloadChat {
		t.Fatalf("DefaultTuningWorkloads() returned shared slice, first = %q", next[0])
	}
}

func TestMachineDiscoveryReport_JSONIncludesUnavailable_Bad(t *testing.T) {
	report := MachineDiscoveryReport{
		Runtime:   RuntimeIdentity{Backend: "metal"},
		Available: false,
	}

	data := core.JSONMarshalString(report)
	if !core.Contains(data, `"available":false`) {
		t.Fatalf("JSON = %s, want explicit available:false", data)
	}
}

func TestScoreTuningMeasurements_Good(t *testing.T) {
	score := ScoreTuningMeasurements(TuningWorkloadAgentState, TuningMeasurements{
		PrefillTokensPerSec:     900,
		DecodeTokensPerSec:      120,
		PromptCacheHitRate:      0.75,
		KVRestoreMilliseconds:   4,
		StateBundleMilliseconds: 2,
		PeakMemoryBytes:         8 << 30,
	})

	if score.Workload != TuningWorkloadAgentState {
		t.Fatalf("score.Workload = %q, want %q", score.Workload, TuningWorkloadAgentState)
	}
	if score.Score <= score.DecodeTokensPerSec {
		t.Fatalf("agent-state score = %f, want cache/restore benefit above decode tps %f", score.Score, score.DecodeTokensPerSec)
	}
	if score.Labels["state_restore"] != "enabled" {
		t.Fatalf("score labels = %+v, want state_restore enabled", score.Labels)
	}
}

func TestScoreTuningMeasurements_LowLatencyFirstToken_Good(t *testing.T) {
	score := ScoreTuningMeasurements(TuningWorkloadLowLatency, TuningMeasurements{
		DecodeTokensPerSec:     80,
		FirstTokenMilliseconds: 20,
		TotalMilliseconds:      120,
		CorrectnessSmokeResult: "passed",
		CorrectnessSmokeChecks: 2,
	})

	if score.FirstTokenMilliseconds != 20 {
		t.Fatalf("FirstTokenMilliseconds = %f, want 20", score.FirstTokenMilliseconds)
	}
	if score.Score <= score.DecodeTokensPerSec {
		t.Fatalf("low-latency score = %f, want first-token benefit above decode tps %f", score.Score, score.DecodeTokensPerSec)
	}
	if score.Labels["first_token"] != "measured" {
		t.Fatalf("labels = %+v, want first_token measured", score.Labels)
	}
}

func TestPlanModelReplace_Good(t *testing.T) {
	current := ModelIdentity{Path: "/models/qwen", Hash: "abc", Architecture: "qwen3", QuantBits: 4}
	runtime := RuntimeIdentity{Backend: "metal", CacheMode: "paged"}
	adapter := AdapterIdentity{Hash: "lora1"}

	reuse := PlanModelReplace(ModelReplaceRequest{
		CurrentModel:   current,
		NextModel:      current,
		CurrentRuntime: runtime,
		NextRuntime:    runtime,
		CurrentAdapter: adapter,
		NextAdapter:    adapter,
	})
	if reuse.Action != ModelReplaceReuseState || !reuse.Compatible {
		t.Fatalf("reuse plan = %+v, want compatible reuse_state", reuse)
	}

	next := current
	next.Hash = "def"
	next.Path = "/models/qwen-new"
	summary := PlanModelReplace(ModelReplaceRequest{
		CurrentModel:   current,
		NextModel:      next,
		CurrentRuntime: runtime,
		NextRuntime:    runtime,
	})
	if summary.Action != ModelReplaceSummaryWindow || summary.Compatible {
		t.Fatalf("summary plan = %+v, want incompatible summary_window", summary)
	}
}
