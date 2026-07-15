package main

import (
	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
	"time"
)

// --- AX-7 canonical triplets ---

func TestDashboard_NewDashboardService_Good(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "training", "/tmp/db.duckdb")
	name := service.ServiceName()
	snapshot := service.GetSnapshot()

	core.AssertEqual(t, "DashboardService", name)
	core.AssertEqual(t, "/tmp/db.duckdb", snapshot.DBPath)
}

func TestDashboard_NewDashboardService_Bad(t *core.T) {
	service := NewDashboardService("", "", "")
	snapshot := service.GetSnapshot()
	models := service.GetModels()

	core.AssertEqual(t, "", snapshot.DBPath)
	core.AssertEmpty(t, models)
}

func TestDashboard_NewDashboardService_Ugly(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "training", "")
	generation := service.GetGeneration()
	training := service.GetTraining()

	core.AssertEqual(t, GenerationStats{}, generation)
	core.AssertEmpty(t, training)
}

func TestDashboard_DashboardService_ServiceName_Good(t *core.T) {
	service := &DashboardService{}
	got := service.ServiceName()
	want := "DashboardService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestDashboard_DashboardService_ServiceName_Bad(t *core.T) {
	var service *DashboardService
	got := service.ServiceName()
	want := "DashboardService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestDashboard_DashboardService_ServiceName_Ugly(t *core.T) {
	service := NewDashboardService("", "", "")
	first := service.ServiceName()
	second := service.ServiceName()

	core.AssertEqual(t, first, second)
	core.AssertEqual(t, "DashboardService", first)
}

func TestDashboard_DashboardService_ServiceStartup_Good(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "training", "")
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := service.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "DashboardService", service.ServiceName())
}

func TestDashboard_DashboardService_ServiceStartup_Bad(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "training", "")
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := service.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertTrue(t, err.OK)
	core.AssertEmpty(t, service.GetModels())
}

func TestDashboard_DashboardService_ServiceStartup_Ugly(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "training", "")
	err := service.ServiceStartup(core.Background(), application.ServiceOptions{})
	snapshot := service.GetSnapshot()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "", snapshot.DBPath)
}

func TestDashboard_DashboardService_GetSnapshot_Good(t *core.T) {
	service := &DashboardService{dbPath: "/tmp/db.duckdb", lastRefresh: time.Unix(1, 0)}
	service.modelInventory = []ModelInfo{{Name: "model"}}
	snapshot := service.GetSnapshot()

	core.AssertEqual(t, "/tmp/db.duckdb", snapshot.DBPath)
	core.AssertLen(t, snapshot.Models, 1)
}

func TestDashboard_DashboardService_GetSnapshot_Bad(t *core.T) {
	service := &DashboardService{}
	snapshot := service.GetSnapshot()
	got := snapshot.UpdatedAt

	core.AssertEqual(t, "", snapshot.DBPath)
	core.AssertNotEqual(t, "", got)
}

func TestDashboard_DashboardService_GetSnapshot_Ugly(t *core.T) {
	service := &DashboardService{generationStats: GenerationStats{GoldenCompleted: 1}}
	snapshot := service.GetSnapshot()
	got := snapshot.Generation.GoldenCompleted

	core.AssertEqual(t, 1, got)
	core.AssertEmpty(t, snapshot.Training)
}

func TestDashboard_DashboardService_GetTraining_Good(t *core.T) {
	service := &DashboardService{trainingStatus: []TrainingRow{{Model: "m"}}}
	training := service.GetTraining()
	got := training[0].Model

	core.AssertLen(t, training, 1)
	core.AssertEqual(t, "m", got)
}

func TestDashboard_DashboardService_GetTraining_Bad(t *core.T) {
	service := &DashboardService{}
	training := service.GetTraining()
	got := len(training)

	core.AssertEqual(t, 0, got)
	core.AssertEmpty(t, training)
}

func TestDashboard_DashboardService_GetTraining_Ugly(t *core.T) {
	service := &DashboardService{trainingStatus: []TrainingRow{{Model: "m", Loss: 0.5}}}
	training := service.GetTraining()
	training[0].Loss = 0.1

	core.AssertEqual(t, 0.1, training[0].Loss)
	core.AssertEqual(t, 0.1, service.trainingStatus[0].Loss)
}

func TestDashboard_DashboardService_GetGeneration_Good(t *core.T) {
	service := &DashboardService{generationStats: GenerationStats{GoldenCompleted: 3, GoldenTarget: 10}}
	generation := service.GetGeneration()
	got := generation.GoldenCompleted

	core.AssertEqual(t, 3, got)
	core.AssertEqual(t, 10, generation.GoldenTarget)
}

func TestDashboard_DashboardService_GetGeneration_Bad(t *core.T) {
	service := &DashboardService{}
	generation := service.GetGeneration()
	got := generation.GoldenTarget

	core.AssertEqual(t, 0, got)
	core.AssertEqual(t, GenerationStats{}, generation)
}

func TestDashboard_DashboardService_GetGeneration_Ugly(t *core.T) {
	service := &DashboardService{generationStats: GenerationStats{ExpansionPct: 99.5}}
	generation := service.GetGeneration()
	got := generation.ExpansionPct

	core.AssertEqual(t, 99.5, got)
	core.AssertEqual(t, 0, generation.GoldenCompleted)
}

func TestDashboard_DashboardService_GetModels_Good(t *core.T) {
	service := &DashboardService{modelInventory: []ModelInfo{{Name: "m", Status: "scored"}}}
	models := service.GetModels()
	got := models[0].Status

	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "scored", got)
}

func TestDashboard_DashboardService_GetModels_Bad(t *core.T) {
	service := &DashboardService{}
	models := service.GetModels()
	got := len(models)

	core.AssertEqual(t, 0, got)
	core.AssertEmpty(t, models)
}

func TestDashboard_DashboardService_GetModels_Ugly(t *core.T) {
	service := &DashboardService{modelInventory: []ModelInfo{{Name: "m", Accuracy: 0.9}}}
	models := service.GetModels()
	models[0].Accuracy = 0.1

	core.AssertEqual(t, 0.1, models[0].Accuracy)
	core.AssertEqual(t, 0.1, service.modelInventory[0].Accuracy)
}

func TestDashboard_DashboardService_Refresh_Good(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "training", "")
	err := service.Refresh()
	snapshot := service.GetSnapshot()

	core.AssertTrue(t, err.OK)
	core.AssertNotEqual(t, "", snapshot.UpdatedAt)
}

func TestDashboard_DashboardService_Refresh_Bad(t *core.T) {
	var service DashboardService
	core.AssertPanics(t, func() {
		_ = service.Refresh()
	})
	core.AssertEqual(t, "", service.dbPath)
}

func TestDashboard_DashboardService_Refresh_Ugly(t *core.T) {
	service := NewDashboardService("http://127.0.0.1:1", "", "")
	err := service.Refresh()
	generation := service.GetGeneration()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, GenerationStats{}, generation)
}

func TestDashboard_DashboardService_RunQuery_Good(t *core.T) {
	service := &DashboardService{}
	r := service.RunQuery("select 1")

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "no database configured")
}

func TestDashboard_DashboardService_RunQuery_Bad(t *core.T) {
	service := &DashboardService{dbPath: ""}
	r := service.RunQuery("")

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "no database configured")
}

func TestDashboard_DashboardService_RunQuery_Ugly(t *core.T) {
	service := &DashboardService{dbPath: "/path/that/does/not/exist.duckdb"}
	r := service.RunQuery("select 1")

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "open db")
}
