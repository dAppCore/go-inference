// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"
	"time"
)

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
