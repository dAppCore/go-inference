package mlservice

import (
	core "dappco.re/go"
	"dappco.re/go/inference/datapipe"
)

func TestStatus_PrintStatus_Good(t *core.T) {
	queries := map[string][]map[string]any{
		"training_status": {{"model": "m", "status": "running", "iteration": float64(1), "total_iters": float64(2), "pct": float64(50)}},
		"training_loss":   {{"model": "m", "loss": float64(0.5)}},
	}
	influx, _ := newFakeInflux(t, queries, 0)
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintStatus(influx, buf))
	core.AssertContains(t, buf.String(), "running")
}

func TestStatus_PrintStatus_Bad(t *core.T) {
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintStatus(influx, buf))
	core.AssertContains(t, buf.String(), "no data")
}

func TestStatus_PrintStatus_Ugly(t *core.T) {
	influx, _ := newFakeInflux(t, nil, 0)
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintStatus(influx, buf))
	core.AssertContains(t, buf.String(), "Generation:")
}
