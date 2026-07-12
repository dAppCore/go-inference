// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"
	"time"
)

func TestHIPLogitSpreadSummary_Top20AndMoments(t *testing.T) {
	logits := make([]float32, 24)
	for index := range logits {
		logits[index] = float32(index - 12)
	}
	summary := hipSummarizeLogitSpread("host", "sampler-input", 0, logits)
	if summary.Arm != "host" || summary.Stage != "sampler-input" || summary.Step != 0 || summary.Count != 24 {
		t.Fatalf("summary identity = %#v", summary)
	}
	if summary.Max != 11 || summary.Mean != -0.5 || math.Abs(summary.StdDev-6.92218655) > 1e-5 {
		t.Fatalf("summary moments = max %g mean %g stddev %g", summary.Max, summary.Mean, summary.StdDev)
	}
	if len(summary.Top) != 20 || summary.Top[0].TokenID != 23 || summary.Top[0].Logit != 11 || summary.Top[19].TokenID != 4 {
		t.Fatalf("summary top = %#v", summary.Top)
	}
}

func TestHIPLogitSpreadSummary_IgnoresNonFiniteValues(t *testing.T) {
	summary := hipSummarizeLogitSpread("oracle", "sampler-input", 5, []float32{1, float32(math.NaN()), 3, float32(math.Inf(1))})
	if summary.Count != 2 || summary.Max != 3 || summary.Mean != 2 || summary.StdDev != 1 {
		t.Fatalf("finite summary = %#v", summary)
	}
}

func TestHIPLogitSpreadReceipts_DisabledIsZeroCost(t *testing.T) {
	hipLogitSpreadReceiptsActive.Store(nil)
	if got := hipActiveLogitSpreadReceipts(); got != nil {
		t.Fatalf("inactive spread receipts = %#v, want nil", got)
	}
}

func TestHIPDecodeRouteMetrics_RecordAndSnapshot(t *testing.T) {
	metrics := newHIPDecodeRouteMetrics()
	metrics.setLayer(5, "full_attention")
	metrics.record("rocm_attention_heads", hipDecodeRouteDevice, 3*time.Millisecond)
	metrics.record("kv_host_restore", hipDecodeRouteHost, 7*time.Millisecond)

	snapshot := metrics.snapshot()
	if len(snapshot) != 2 {
		t.Fatalf("route count = %d, want 2", len(snapshot))
	}
	if got := snapshot[0]; got.Layer != 5 || got.LayerType != "full_attention" || got.Route != hipDecodeRouteHost || got.Op != "kv_host_restore" || got.Calls != 1 || got.WallTime != 7*time.Millisecond {
		t.Fatalf("host route = %#v", got)
	}
	if got := snapshot[1]; got.Layer != 5 || got.LayerType != "full_attention" || got.Route != hipDecodeRouteDevice || got.Op != "rocm_attention_heads" || got.Calls != 1 || got.WallTime != 3*time.Millisecond {
		t.Fatalf("device route = %#v", got)
	}
}

func TestHIPDecodeRouteMetrics_DisabledIsZeroCost(t *testing.T) {
	hipDecodeRouteMetricsActive.Store(nil)
	if got := hipActiveDecodeRouteMetrics(); got != nil {
		t.Fatalf("inactive metrics = %#v, want nil", got)
	}
}

func TestHIPFeedbackReceipts_RecordAndSnapshot(t *testing.T) {
	receipts := newHIPFeedbackReceipts()
	receipts.record(0, 17, 17, 9, 9)
	receipts.record(1, 23, 23, 10, 10)

	snapshot := receipts.snapshot()
	if len(snapshot) != 2 {
		t.Fatalf("receipt count = %d, want 2", len(snapshot))
	}
	if got := snapshot[0]; got.Step != 0 || got.DeviceArgmax != 17 || got.FedToken != 17 || got.Position != 9 || got.KVWriteIndex != 9 {
		t.Fatalf("first receipt = %#v", got)
	}
	if got := snapshot[1]; got.Step != 1 || got.DeviceArgmax != 23 || got.FedToken != 23 || got.Position != 10 || got.KVWriteIndex != 10 {
		t.Fatalf("second receipt = %#v", got)
	}
}

func TestHIPFeedbackReceipts_DisabledIsZeroCost(t *testing.T) {
	hipFeedbackReceiptsActive.Store(nil)
	if got := hipActiveFeedbackReceipts(); got != nil {
		t.Fatalf("inactive receipts = %#v, want nil", got)
	}
}
