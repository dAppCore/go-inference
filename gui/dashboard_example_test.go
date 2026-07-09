package main

import (
	"context"

	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

func ExampleNewDashboardService() {
	service := NewDashboardService("", "metrics", "lem.duckdb")

	core.Println(service.ServiceName())
	// Output:
	// DashboardService
}

func ExampleDashboardService_ServiceName() {
	service := NewDashboardService("", "", "")

	core.Println(service.ServiceName())
	// Output:
	// DashboardService
}

func ExampleDashboardService_ServiceStartup() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	service := NewDashboardService("", "", "")
	err := service.ServiceStartup(ctx, application.ServiceOptions{})

	core.Println(err.OK)
	// Output:
	// true
}

func ExampleDashboardService_GetSnapshot() {
	service := NewDashboardService("", "", "lem.duckdb")
	snapshot := service.GetSnapshot()

	core.Println(snapshot.DBPath)
	core.Println(len(snapshot.Models))
	// Output:
	// lem.duckdb
	// 0
}

func ExampleDashboardService_GetTraining() {
	service := NewDashboardService("", "", "")

	core.Println(len(service.GetTraining()))
	// Output:
	// 0
}

func ExampleDashboardService_GetGeneration() {
	service := NewDashboardService("", "", "")
	generation := service.GetGeneration()

	core.Println(generation.GoldenCompleted)
	core.Println(generation.ExpansionCompleted)
	// Output:
	// 0
	// 0
}

func ExampleDashboardService_GetModels() {
	service := NewDashboardService("", "", "")

	core.Println(len(service.GetModels()))
	// Output:
	// 0
}

func ExampleDashboardService_Refresh() {
	service := NewDashboardService("", "", "")
	err := service.Refresh()

	core.Println(err.OK)
	// Output:
	// true
}

func ExampleDashboardService_RunQuery() {
	service := NewDashboardService("", "", "")
	_, err := service.RunQuery("select 1")

	core.Println(err != nil)
	// Output:
	// true
}
