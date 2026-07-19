// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/work"
	coreio "dappco.re/go/io"
)

type exampleWorkspaceRunner struct{}

func (exampleWorkspaceRunner) Run(context.Context, Command) core.Result {
	return core.Ok("")
}

type exampleWorkspaceServer struct{}

func (exampleWorkspaceServer) Start(context.Context) core.Result {
	return core.Ok(gitserver.Health{Running: true})
}

func (exampleWorkspaceServer) EnsureRepository(context.Context, string) core.Result {
	return core.Ok(gitserver.Repository{Name: "example", CloneURL: "/tmp/example.git"})
}

func (exampleWorkspaceServer) Health(context.Context) core.Result {
	return core.Ok(gitserver.Health{Running: true})
}

func (exampleWorkspaceServer) Close() core.Result {
	return core.Ok(nil)
}

func ExampleNewManager() {
	result := NewManager(ManagerOptions{
		Root:   "/tmp",
		Files:  coreio.NewMemoryMedium(),
		Git:    exampleWorkspaceRunner{},
		Server: exampleWorkspaceServer{},
		IDs:    func() string { return "example" },
		Now:    time.Now,
	})
	core.Println(result.OK)
	// Output: true
}

func ExampleRecoveryReceipt() {
	receipt := RecoveryReceipt{Kind: "review", ReviewID: "review-1", Worktree: "/tmp/review-1"}
	core.Println(receipt.Kind, receipt.ReviewID, receipt.Worktree)
	// Output: review review-1 /tmp/review-1
}

func ExampleManager_Recovery() {
	var manager *Manager
	result := manager.Recovery("run-1")
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_AbandonRecovery() {
	var manager *Manager
	result := manager.AbandonRecovery(context.Background(), RecoveryReceipt{})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_ReviewSource() {
	var manager *Manager
	result := manager.ReviewSource(context.Background(), "/tmp/project")
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_Register() {
	var manager *Manager
	result := manager.Register(context.Background(), RegisterRequest{Confirmed: true})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_PrepareRun() {
	var manager *Manager
	result := manager.PrepareRun(context.Background(), work.Project{}, work.Run{})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_CaptureRun() {
	var manager *Manager
	result := manager.CaptureRun(context.Background(), RunWorkspace{})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_ReconstructRun() {
	var manager *Manager
	result := manager.ReconstructRun(context.Background(), work.Project{}, work.Run{})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_ReleaseRun() {
	var manager *Manager
	result := manager.ReleaseRun(context.Background(), RunWorkspace{})
	core.Println(result.OK)
	// Output: false
}
