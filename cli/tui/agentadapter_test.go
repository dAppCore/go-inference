// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

func TestAgentAdapter_Capabilities_Good(t *testing.T) {
	engine := &fixtureNativeAgentEngine{capabilities: []work.Capability{
		{Name: "dispatch", Available: true}, {Name: "cancel", Available: true},
		{Name: "answer", Available: true}, {Name: "retry", Available: true},
		{Name: "resume", Available: true}, {Name: "queue.start", Available: true},
		{Name: "queue.stop", Available: true}, {Name: "changes.review", Available: true},
		{Name: "accept", Available: true}, {Name: "reject", Available: true},
	}}
	adapter := requireAgentAdapter(t, engine)

	capabilities := adapter.Capabilities()
	available := make([]agentFeature, 0, 10)
	unavailableReasons := make(map[agentFeature]string)
	for _, capability := range capabilities {
		if capability.Available {
			available = append(available, capability.Feature)
		} else {
			unavailableReasons[capability.Feature] = capability.Reason
		}
	}
	want := []agentFeature{
		agentFeatureDispatch, agentFeatureCancel, agentFeatureAnswer, agentFeatureRetry,
		agentFeatureResume, agentFeatureQueueStart, agentFeatureQueueStop,
		agentFeatureChangesReview, agentFeatureAccept, agentFeatureReject,
	}
	if core.JSONMarshalString(available) != core.JSONMarshalString(want) {
		t.Fatalf("available capabilities = %#v, want %#v", available, want)
	}
	for _, feature := range []agentFeature{agentFeatureSetup, agentFeatureProvider, agentFeatureTemplate, agentFeaturePlan, agentFeatureSession, agentFeatureHandoff, agentFeatureScan, agentFeatureAudit, agentFeaturePipeline, agentFeatureMonitor, agentFeatureHarvest, agentFeatureBrainRecall, agentFeatureBrainRemember, agentFeatureMessage, agentFeatureFleet, agentFeatureForge, agentFeatureRemote, agentFeatureQA, agentFeatureReview, agentFeaturePRCreate, agentFeaturePRMerge} {
		if core.Trim(unavailableReasons[feature]) == "" {
			t.Fatalf("future capability %q has no specific unavailable reason", feature)
		}
	}
}

func TestAgentAdapter_Snapshot_Good(t *testing.T) {
	at := time.Date(2026, time.July, 18, 11, 0, 0, 0, time.UTC)
	engine := &fixtureNativeAgentEngine{snapshot: work.Snapshot{
		Projects: []work.Project{{ID: "work-1", SourcePath: "/src/one", SourceBranch: "main", RepositoryName: "private-one"}},
		Runs: []work.Run{
			{ID: "run-1", WorkID: "work-1", ProjectID: "work-1", Provider: "codex", Model: "gpt-5", Branch: "lem/work-1/1", Status: work.RunRunning},
			{ID: "run-2", WorkID: "work-2", Provider: "claude", Model: "opus", Status: work.RunWaiting},
		},
		Events:    []work.Event{{ID: "event-2", RunID: "run-1", WorkID: "work-1", Kind: "started", Title: "started", CreatedAt: at.Add(2 * time.Second)}},
		Logs:      []work.LogChunk{{RunID: "run-1", Sequence: 4, Stream: "stdout", Text: "building", CreatedAt: at.Add(time.Second)}},
		Questions: []work.Question{{ID: "question-1", RunID: "run-2", Text: "Which target?", CreatedAt: at.Add(3 * time.Second)}},
	}}
	adapter := requireAgentAdapter(t, engine)

	result := adapter.Snapshot(context.Background())
	if !result.OK {
		t.Fatalf("Snapshot failed: %s", result.Error())
	}
	snapshot, ok := result.Value.(agentSnapshot)
	if !ok {
		t.Fatalf("Snapshot value = %T, want agentSnapshot", result.Value)
	}
	if len(snapshot.Work) != 2 || snapshot.Work[0].ExternalID != "work-1" || snapshot.Work[0].Repo != "/src/one" || snapshot.Work[0].Agent != "codex" || snapshot.Work[1].ExternalID != "work-2" || snapshot.Work[1].Question != "Which target?" {
		t.Fatalf("mapped work = %#v", snapshot.Work)
	}
	if len(snapshot.Events) != 3 || snapshot.Events[0].Kind != "log.stdout" || snapshot.Events[1].ExternalID != "event-2" || snapshot.Events[2].Kind != "question" {
		t.Fatalf("ordered events/logs/questions = %#v", snapshot.Events)
	}
	if engine.snapshotWorkID != "" {
		t.Fatalf("Snapshot work ID = %q, want all work", engine.snapshotWorkID)
	}
}

func TestAgentAdapter_ProjectAndDispatchReview_Good(t *testing.T) {
	projectReview := orchestrator.ProjectReview{
		Work:           work.Item{ID: "work-1", Title: "Ship it", Task: "Implement the slice", Repository: "/src/project"},
		Source:         workspace.SourceReview{Path: "/src/project", Root: "/src/project", Branch: "main", Revision: "abc123", IncludedHash: "hash", Included: []string{"go.mod", "main.go"}},
		RepositoryName: "work-1", RequiresGitEnable: true,
	}
	dispatchReview := orchestrator.DispatchReview{
		Request: work.DispatchRequest{Work: projectReview.Work, Provider: "codex", Model: "gpt-5", ConfirmedSourceRevision: "abc123"},
		Project: work.Project{ID: "work-1", SourcePath: "/src/project", SourceRevision: "abc123", RepositoryName: "work-1"},
		Source:  projectReview.Source, WorktreePath: "/private/runs/pending-run/worktree", Warning: "native host access",
	}
	engine := &fixtureNativeAgentEngine{projectReview: projectReview, registeredProject: dispatchReview.Project, dispatchReview: dispatchReview}
	adapter := requireAgentAdapter(t, engine)
	request := agentReviewRequest{
		Feature: agentFeatureDispatch, WorkID: "work-1", Provider: "codex", Model: "gpt-5",
		Work: agentWorkRequest{ID: "work-1", Title: "Ship it", Task: "Implement the slice", Repository: "/src/project"},
	}

	reviewResult := adapter.Review(context.Background(), request)
	if !reviewResult.OK {
		t.Fatalf("project Review failed: %s", reviewResult.Error())
	}
	review := reviewResult.Value.(agentReview)
	if !review.ConfirmRequired || !review.GitConfirmRequired || !strings.Contains(review.Body, "/src/project") || !strings.Contains(review.Warning, "Git") {
		t.Fatalf("project review = %#v", review)
	}
	withoutGit := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureDispatch, Review: review, Confirmed: true})
	if withoutGit.OK || engine.registerCalls != 0 {
		t.Fatalf("registration without separate Git confirmation = %#v, calls=%d", withoutGit, engine.registerCalls)
	}

	registered := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureDispatch, Review: review, Confirmed: true, EnableGit: true})
	if !registered.OK {
		t.Fatalf("register project: %s", registered.Error())
	}
	launch := registered.Value.(agentReview)
	if !launch.ConfirmRequired || launch.GitConfirmRequired || !strings.Contains(launch.Body, "codex") || !strings.Contains(launch.Body, "/private/runs/pending-run/worktree") || !strings.Contains(launch.Warning, "native host access") {
		t.Fatalf("dispatch review = %#v", launch)
	}
	if engine.registerConfirmed != true || engine.registerCalls != 1 || engine.reviewDispatchCalls != 1 {
		t.Fatalf("project/dispatch calls = register %d confirmed %v, review %d", engine.registerCalls, engine.registerConfirmed, engine.reviewDispatchCalls)
	}

	dispatched := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureDispatch, Review: launch, Confirmed: true})
	if !dispatched.OK || engine.dispatchCalls != 1 {
		t.Fatalf("dispatch = %#v, calls=%d", dispatched, engine.dispatchCalls)
	}
}

func TestAgentAdapter_Actions_Good(t *testing.T) {
	changeReview := workspace.ChangeReview{WorkID: "work-1", RunID: "run-1", Diff: "+change", CommitLog: "abc change", Validation: []workspace.ValidationResult{{Passed: true}}}
	engine := &fixtureNativeAgentEngine{changeReview: changeReview}
	adapter := requireAgentAdapter(t, engine)
	workRequest := agentWorkRequest{ID: "work-1", Title: "Work one", Task: "Task one", Repository: "/src/one"}

	actions := []agentRequest{
		{Feature: agentFeatureCancel, WorkID: "run-1"},
		{Feature: agentFeatureAnswer, WorkID: "run-1", Input: "Use target A"},
		{Feature: agentFeatureRetry, WorkID: "run-1", Work: workRequest},
		{Feature: agentFeatureResume, WorkID: "run-1", Input: "answer-1", Provider: "codex", Model: "gpt-5", Work: workRequest},
		{Feature: agentFeatureQueueStart},
		{Feature: agentFeatureQueueStop},
		{Feature: agentFeatureReject, WorkID: "run-1"},
	}
	for _, request := range actions {
		if result := adapter.Run(context.Background(), request); !result.OK {
			t.Fatalf("Run(%s): %s", request.Feature, result.Error())
		} else if _, ok := result.Value.(agentActionReceipt); !ok {
			t.Fatalf("Run(%s) value = %T, want private receipt", request.Feature, result.Value)
		}
	}

	reviewed := adapter.Review(context.Background(), agentReviewRequest{Feature: agentFeatureChangesReview, WorkID: "run-1"})
	if !reviewed.OK {
		t.Fatalf("change Review: %s", reviewed.Error())
	}
	review := reviewed.Value.(agentReview)
	if !review.ConfirmRequired || !strings.Contains(review.Body, "+change") {
		t.Fatalf("change review = %#v", review)
	}
	accepted := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureAccept, Review: review, Confirmed: true})
	if !accepted.OK {
		t.Fatalf("Accept: %s", accepted.Error())
	}
	wantCalls := []string{"cancel", "answer", "retry", "resume", "queue.start", "queue.stop", "reject", "changes.review", "accept"}
	if core.JSONMarshalString(engine.calls) != core.JSONMarshalString(wantCalls) {
		t.Fatalf("mapped action order = %#v, want %#v", engine.calls, wantCalls)
	}
}

func TestAgentAdapter_Close_Ugly(t *testing.T) {
	engine := &fixtureNativeAgentEngine{}
	adapter := requireAgentAdapter(t, engine)
	if result := adapter.Close(); !result.OK {
		t.Fatalf("first Close: %s", result.Error())
	}
	if result := adapter.Close(); !result.OK {
		t.Fatalf("second Close: %s", result.Error())
	}
	if engine.closeCalls != 1 {
		t.Fatalf("engine Close calls = %d, want 1", engine.closeCalls)
	}
}

func requireAgentAdapter(t *testing.T, engine nativeAgentEngine) agentProvider {
	t.Helper()
	result := newAgentAdapter(engine)
	if !result.OK {
		t.Fatalf("newAgentAdapter: %s", result.Error())
	}
	return result.Value.(agentProvider)
}

type fixtureNativeAgentEngine struct {
	capabilities        []work.Capability
	snapshot            work.Snapshot
	projectReview       orchestrator.ProjectReview
	registeredProject   work.Project
	dispatchReview      orchestrator.DispatchReview
	changeReview        workspace.ChangeReview
	snapshotWorkID      string
	calls               []string
	registerCalls       int
	registerConfirmed   bool
	reviewDispatchCalls int
	dispatchCalls       int
	closeCalls          int
}

func (engine *fixtureNativeAgentEngine) Capabilities() []work.Capability {
	return append([]work.Capability(nil), engine.capabilities...)
}

func (engine *fixtureNativeAgentEngine) Snapshot(_ context.Context, workID string) core.Result {
	engine.snapshotWorkID = workID
	return core.Ok(engine.snapshot)
}

func (engine *fixtureNativeAgentEngine) ReviewProject(context.Context, work.Item) core.Result {
	return core.Ok(engine.projectReview)
}

func (engine *fixtureNativeAgentEngine) RegisterProject(_ context.Context, _ orchestrator.ProjectReview, confirmed bool) core.Result {
	engine.registerCalls++
	engine.registerConfirmed = confirmed
	return core.Ok(engine.registeredProject)
}

func (engine *fixtureNativeAgentEngine) ReviewDispatch(context.Context, work.DispatchRequest) core.Result {
	engine.reviewDispatchCalls++
	return core.Ok(engine.dispatchReview)
}

func (engine *fixtureNativeAgentEngine) Dispatch(context.Context, orchestrator.DispatchReview) core.Result {
	engine.dispatchCalls++
	return core.Ok(work.Run{ID: "run-1", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) Cancel(context.Context, string) core.Result {
	engine.calls = append(engine.calls, "cancel")
	return core.Ok(work.Run{ID: "run-1", WorkID: "work-1", Status: work.RunCancelled})
}

func (engine *fixtureNativeAgentEngine) Answer(context.Context, string, string) core.Result {
	engine.calls = append(engine.calls, "answer")
	return core.Ok(work.Answer{ID: "answer-1", ResumeRunID: "run-2"})
}

func (engine *fixtureNativeAgentEngine) Retry(context.Context, work.Item, string) core.Result {
	engine.calls = append(engine.calls, "retry")
	return core.Ok(work.Run{ID: "run-2", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) Resume(context.Context, work.ResumeRequest) core.Result {
	engine.calls = append(engine.calls, "resume")
	return core.Ok(work.Run{ID: "run-3", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) StartQueue(context.Context) core.Result {
	engine.calls = append(engine.calls, "queue.start")
	return core.Ok(work.QueueState{ID: "default", Status: work.QueueAccepting})
}

func (engine *fixtureNativeAgentEngine) StopQueue(context.Context) core.Result {
	engine.calls = append(engine.calls, "queue.stop")
	return core.Ok(work.QueueState{ID: "default", Status: work.QueueFrozen})
}

func (engine *fixtureNativeAgentEngine) ReviewChanges(context.Context, string) core.Result {
	engine.calls = append(engine.calls, "changes.review")
	return core.Ok(engine.changeReview)
}

func (engine *fixtureNativeAgentEngine) Accept(context.Context, workspace.AcceptRequest) core.Result {
	engine.calls = append(engine.calls, "accept")
	return core.Ok(work.Acceptance{ID: "accept-1", WorkID: "work-1", RunID: "run-1", Status: "accepted"})
}

func (engine *fixtureNativeAgentEngine) Reject(context.Context, string) core.Result {
	engine.calls = append(engine.calls, "reject")
	return core.Ok(work.Acceptance{ID: "reject-1", WorkID: "work-1", RunID: "run-1", Status: "rejected"})
}

func (engine *fixtureNativeAgentEngine) Close() core.Result {
	engine.closeCalls++
	return core.Ok(nil)
}
