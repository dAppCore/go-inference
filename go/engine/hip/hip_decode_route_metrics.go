// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
)

const hipDecodeRouteMetricsEnv = "GO_ROCM_HIP_DECODE_ROUTE_METRICS"
const hipFeedbackReceiptsEnv = "GO_ROCM_HIP_FEEDBACK_RECEIPTS"

type hipDecodeRoute string

const (
	hipDecodeRouteDevice hipDecodeRoute = "device-kernel"
	hipDecodeRouteHost   hipDecodeRoute = "host-fallback"
)

type hipDecodeRouteMetric struct {
	Op        string
	Layer     int
	LayerType string
	Route     hipDecodeRoute
	Calls     uint64
	WallTime  time.Duration
}

type hipDecodeRouteMetricKey struct {
	op        string
	layer     int
	layerType string
	route     hipDecodeRoute
}

type hipDecodeRouteMetrics struct {
	mu        sync.Mutex
	layer     int
	layerType string
	entries   map[hipDecodeRouteMetricKey]hipDecodeRouteMetric
}

var hipDecodeRouteMetricsActive atomic.Pointer[hipDecodeRouteMetrics]
var hipDecodeRouteMetricsArmed atomic.Bool

type hipFeedbackReceipt struct {
	Step         int
	DeviceArgmax int32
	FedToken     int32
	Position     int
	KVWriteIndex int
}

type hipFeedbackReceipts struct {
	mu      sync.Mutex
	entries []hipFeedbackReceipt
}

var hipFeedbackReceiptsActive atomic.Pointer[hipFeedbackReceipts]
var hipFeedbackReceiptsArmed atomic.Bool

func newHIPFeedbackReceipts() *hipFeedbackReceipts {
	return &hipFeedbackReceipts{entries: make([]hipFeedbackReceipt, 0, 128)}
}

func hipActiveFeedbackReceipts() *hipFeedbackReceipts {
	return hipFeedbackReceiptsActive.Load()
}

func hipBeginFeedbackReceipts() *hipFeedbackReceipts {
	if core.Getenv(hipFeedbackReceiptsEnv) == "" || !hipFeedbackReceiptsArmed.CompareAndSwap(false, true) {
		return nil
	}
	receipts := newHIPFeedbackReceipts()
	hipFeedbackReceiptsActive.Store(receipts)
	return receipts
}

func hipFinishFeedbackReceipts(receipts *hipFeedbackReceipts) {
	if receipts == nil {
		return
	}
	hipFeedbackReceiptsActive.CompareAndSwap(receipts, nil)
	core.Println("HIP_FEEDBACK_RECEIPTS_BEGIN")
	core.Println("step\tdevice_argmax\tfed_token\tposition\tkv_write_index")
	for _, entry := range receipts.snapshot() {
		core.Println(core.Sprintf("%d\t%d\t%d\t%d\t%d", entry.Step, entry.DeviceArgmax, entry.FedToken, entry.Position, entry.KVWriteIndex))
	}
	core.Println("HIP_FEEDBACK_RECEIPTS_END")
}

func (receipts *hipFeedbackReceipts) record(step int, deviceArgmax, fedToken int32, position, kvWriteIndex int) {
	if receipts == nil {
		return
	}
	receipts.mu.Lock()
	receipts.entries = append(receipts.entries, hipFeedbackReceipt{
		Step:         step,
		DeviceArgmax: deviceArgmax,
		FedToken:     fedToken,
		Position:     position,
		KVWriteIndex: kvWriteIndex,
	})
	receipts.mu.Unlock()
}

func (receipts *hipFeedbackReceipts) snapshot() []hipFeedbackReceipt {
	if receipts == nil {
		return nil
	}
	receipts.mu.Lock()
	out := append([]hipFeedbackReceipt(nil), receipts.entries...)
	receipts.mu.Unlock()
	return out
}

func newHIPDecodeRouteMetrics() *hipDecodeRouteMetrics {
	return &hipDecodeRouteMetrics{layer: -1, entries: make(map[hipDecodeRouteMetricKey]hipDecodeRouteMetric, 64)}
}

func hipActiveDecodeRouteMetrics() *hipDecodeRouteMetrics {
	return hipDecodeRouteMetricsActive.Load()
}

func hipBeginDecodeRouteMetrics() *hipDecodeRouteMetrics {
	if core.Getenv(hipDecodeRouteMetricsEnv) == "" || !hipDecodeRouteMetricsArmed.CompareAndSwap(false, true) {
		return nil
	}
	metrics := newHIPDecodeRouteMetrics()
	hipDecodeRouteMetricsActive.Store(metrics)
	return metrics
}

func hipFinishDecodeRouteMetrics(metrics *hipDecodeRouteMetrics) {
	if metrics == nil {
		return
	}
	hipDecodeRouteMetricsActive.CompareAndSwap(metrics, nil)
	core.Println("HIP_DECODE_ROUTE_TABLE_BEGIN")
	core.Println("layer\tlayer_type\troute\top\tcalls\twall_ms")
	for _, entry := range metrics.snapshot() {
		core.Println(core.Sprintf("%d\t%s\t%s\t%s\t%d\t%.3f", entry.Layer, entry.LayerType, entry.Route, entry.Op, entry.Calls, float64(entry.WallTime)/float64(time.Millisecond)))
	}
	core.Println("HIP_DECODE_ROUTE_TABLE_END")
}

func (metrics *hipDecodeRouteMetrics) setLayer(layer int, layerType string) {
	if metrics == nil {
		return
	}
	metrics.mu.Lock()
	metrics.layer = layer
	metrics.layerType = layerType
	metrics.mu.Unlock()
}

func (metrics *hipDecodeRouteMetrics) record(op string, route hipDecodeRoute, elapsed time.Duration) {
	if metrics == nil {
		return
	}
	metrics.mu.Lock()
	key := hipDecodeRouteMetricKey{op: op, layer: metrics.layer, layerType: metrics.layerType, route: route}
	entry := metrics.entries[key]
	entry.Op, entry.Layer, entry.LayerType, entry.Route = op, key.layer, key.layerType, route
	entry.Calls++
	entry.WallTime += elapsed
	metrics.entries[key] = entry
	metrics.mu.Unlock()
}

func (metrics *hipDecodeRouteMetrics) snapshot() []hipDecodeRouteMetric {
	if metrics == nil {
		return nil
	}
	metrics.mu.Lock()
	out := make([]hipDecodeRouteMetric, 0, len(metrics.entries))
	for _, entry := range metrics.entries {
		out = append(out, entry)
	}
	metrics.mu.Unlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].WallTime != out[j].WallTime {
			return out[i].WallTime > out[j].WallTime
		}
		if out[i].Layer != out[j].Layer {
			return out[i].Layer < out[j].Layer
		}
		return out[i].Op < out[j].Op
	})
	return out
}
