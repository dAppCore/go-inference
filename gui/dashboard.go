package main

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/gui/internal/lem"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// DashboardService bridges pkg/lem CLI functions for the desktop UI.
// Provides real-time status, model inventory, and scoring progress
// to the frontend via Wails bindings.
type DashboardService struct {
	influx *lem.InfluxClient
	dbPath string
	mu     sync.RWMutex

	// Cached state (refreshed periodically).
	trainingStatus  []TrainingRow
	generationStats GenerationStats
	modelInventory  []ModelInfo
	lastRefresh     time.Time
}

// TrainingRow represents a single model's training progress.
type TrainingRow struct {
	Model      string  `json:"model"`
	RunID      string  `json:"runId"`
	Status     string  `json:"status"`
	Iteration  int     `json:"iteration"`
	TotalIters int     `json:"totalIters"`
	Pct        float64 `json:"pct"`
	Loss       float64 `json:"loss"`
}

// GenerationStats shows golden set and expansion progress.
type GenerationStats struct {
	GoldenCompleted    int     `json:"goldenCompleted"`
	GoldenTarget       int     `json:"goldenTarget"`
	GoldenPct          float64 `json:"goldenPct"`
	ExpansionCompleted int     `json:"expansionCompleted"`
	ExpansionTarget    int     `json:"expansionTarget"`
	ExpansionPct       float64 `json:"expansionPct"`
}

// ModelInfo represents a model in the inventory.
type ModelInfo struct {
	Name       string  `json:"name"`
	Tag        string  `json:"tag"`
	Accuracy   float64 `json:"accuracy"`
	Iterations int     `json:"iterations"`
	Status     string  `json:"status"`
}

// AgentStatus represents the scoring agent's current state.
type AgentStatus struct {
	Running     bool   `json:"running"`
	CurrentTask string `json:"currentTask"`
	Scored      int    `json:"scored"`
	Remaining   int    `json:"remaining"`
	LastScore   string `json:"lastScore"`
}

// DashboardSnapshot is the complete UI state sent to the frontend.
type DashboardSnapshot struct {
	Training   []TrainingRow   `json:"training"`
	Generation GenerationStats `json:"generation"`
	Models     []ModelInfo     `json:"models"`
	Agent      AgentStatus     `json:"agent"`
	DBPath     string          `json:"dbPath"`
	UpdatedAt  string          `json:"updatedAt"`
}

// NewDashboardService creates a DashboardService.
func NewDashboardService(influxURL, influxDB, dbPath string) *DashboardService {
	return &DashboardService{
		influx: lem.NewInfluxClient(influxURL, influxDB),
		dbPath: dbPath,
	}
}

// ServiceName returns the Wails service name.
func (d *DashboardService) ServiceName() string {
	return "DashboardService"
}

// ServiceStartup is called when the Wails app starts.
func (d *DashboardService) ServiceStartup(ctx context.Context, options application.ServiceOptions) core.Result {
	core.Print(core.Stderr(), "DashboardService started\n")
	go d.refreshLoop(ctx)
	return core.Ok(nil)
}

// GetSnapshot returns the complete dashboard state.
func (d *DashboardService) GetSnapshot() DashboardSnapshot {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return DashboardSnapshot{
		Training:   d.trainingStatus,
		Generation: d.generationStats,
		Models:     d.modelInventory,
		DBPath:     d.dbPath,
		UpdatedAt:  d.lastRefresh.Format(time.RFC3339),
	}
}

// GetTraining returns current training status.
func (d *DashboardService) GetTraining() []TrainingRow {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.trainingStatus
}

// GetGeneration returns generation progress.
func (d *DashboardService) GetGeneration() GenerationStats {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.generationStats
}

// GetModels returns the model inventory.
func (d *DashboardService) GetModels() []ModelInfo {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.modelInventory
}

// Refresh forces an immediate data refresh.
func (d *DashboardService) Refresh() core.Result {
	return d.refresh()
}

// RunQuery executes an ad-hoc SQL query against DuckDB.
func (d *DashboardService) RunQuery(sql string) core.Result {
	if d.dbPath == "" {
		return core.Fail(core.Errorf("no database configured"))
	}
	db, err := lem.OpenDB(d.dbPath)
	if err != nil {
		return core.Fail(core.Errorf("open db: %w", err))
	}
	defer db.Close()

	rows, err := db.QueryRows(sql)
	if err != nil {
		return core.Fail(core.Errorf("query: %w", err))
	}
	return core.Ok(rows)
}

func (d *DashboardService) refreshLoop(ctx context.Context) {
	// Initial refresh.
	if r := d.refresh(); !r.OK {
		core.Print(core.Stderr(), "Dashboard refresh error: %s\n", r.Error())
	}

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if r := d.refresh(); !r.OK {
				core.Print(core.Stderr(), "Dashboard refresh error: %s\n", r.Error())
			}
		}
	}
}

func (d *DashboardService) refresh() core.Result {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Query training status from InfluxDB.
	rows, err := d.influx.QuerySQL(`
		SELECT model, run_id, status, iteration, total_iters, pct
		FROM training_status
		ORDER BY time DESC LIMIT 10
	`)
	if err == nil {
		d.trainingStatus = nil
		for _, row := range rows {
			d.trainingStatus = append(d.trainingStatus, TrainingRow{
				Model:      strVal(row, "model"),
				RunID:      strVal(row, "run_id"),
				Status:     strVal(row, "status"),
				Iteration:  intVal(row, "iteration"),
				TotalIters: intVal(row, "total_iters"),
				Pct:        floatVal(row, "pct"),
			})
		}
	}

	// Query latest loss per model.
	lossRows, err := d.influx.QuerySQL(`
		SELECT model, loss FROM training_loss
		WHERE loss_type = 'train'
		ORDER BY time DESC LIMIT 10
	`)
	if err == nil {
		lossMap := make(map[string]float64)
		for _, row := range lossRows {
			model := strVal(row, "model")
			if _, exists := lossMap[model]; !exists {
				lossMap[model] = floatVal(row, "loss")
			}
		}
		for i, t := range d.trainingStatus {
			if loss, ok := lossMap[t.Model]; ok {
				d.trainingStatus[i].Loss = loss
			}
		}
	}

	// Query golden set progress.
	goldenRows, err := d.influx.QuerySQL(`
		SELECT completed, target, pct FROM golden_gen_progress
		ORDER BY time DESC LIMIT 1
	`)
	if err == nil && len(goldenRows) > 0 {
		d.generationStats.GoldenCompleted = intVal(goldenRows[0], "completed")
		d.generationStats.GoldenTarget = intVal(goldenRows[0], "target")
		d.generationStats.GoldenPct = floatVal(goldenRows[0], "pct")
	}

	// Query expansion progress.
	expRows, err := d.influx.QuerySQL(`
		SELECT completed, target, pct FROM expansion_progress
		ORDER BY time DESC LIMIT 1
	`)
	if err == nil && len(expRows) > 0 {
		d.generationStats.ExpansionCompleted = intVal(expRows[0], "completed")
		d.generationStats.ExpansionTarget = intVal(expRows[0], "target")
		d.generationStats.ExpansionPct = floatVal(expRows[0], "pct")
	}

	// Query model capability scores.
	capRows, err := d.influx.QuerySQL(`
		SELECT model, label, accuracy, iteration FROM capability_score
		WHERE category = 'overall'
		ORDER BY time DESC LIMIT 20
	`)
	if err == nil {
		d.modelInventory = nil
		seen := make(map[string]bool)
		for _, row := range capRows {
			label := strVal(row, "label")
			if seen[label] {
				continue
			}
			seen[label] = true
			d.modelInventory = append(d.modelInventory, ModelInfo{
				Name:       strVal(row, "model"),
				Tag:        label,
				Accuracy:   floatVal(row, "accuracy"),
				Iterations: intVal(row, "iteration"),
				Status:     "scored",
			})
		}
	}

	d.lastRefresh = time.Now()
	return core.Ok(nil)
}

func strVal(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		return core.Sprintf("%v", v)
	}
	return ""
}

func intVal(m map[string]interface{}, key string) int {
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		}
	}
	return 0
}

func floatVal(m map[string]interface{}, key string) float64 {
	if v, ok := m[key]; ok {
		if f, ok := v.(float64); ok {
			return f
		}
	}
	return 0
}
