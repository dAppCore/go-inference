// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
	coreio "dappco.re/go/io"
	commandexec "dappco.re/go/process/exec"
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

func TestAgentAdapterSnapshotMapsAnsweredWaitingResumeIdentity(t *testing.T) {
	at := time.Date(2026, time.July, 19, 10, 0, 0, 0, time.UTC)
	projection := agentAnswerProjection{AnswerID: "answer-1", QuestionID: "question-1", ResumeRunID: "resume-1"}
	adapter := requireAgentAdapter(t, &fixtureNativeAgentEngine{snapshot: work.Snapshot{
		Runs:      []work.Run{{ID: "waiting-1", WorkID: "work-1", Status: work.RunWaiting}},
		Questions: []work.Question{{ID: projection.QuestionID, RunID: "waiting-1", Text: "Which target?", CreatedAt: at}},
		Events: []work.Event{{
			ID: "answer:" + projection.AnswerID, RunID: "waiting-1", WorkID: "work-1", Kind: "answered",
			DetailJSON: core.JSONMarshalString(projection), CreatedAt: at.Add(time.Second),
		}},
	}})

	result := adapter.Snapshot(context.Background())
	core.AssertTrue(t, result.OK, result.Error())
	mapped := result.Value.(agentSnapshot)
	core.AssertEqual(t, 1, len(mapped.Work))
	core.AssertEqual(t, projection.QuestionID, mapped.Work[0].QuestionID)
	core.AssertEqual(t, projection.AnswerID, mapped.Work[0].AnswerID)
	core.AssertEqual(t, projection.ResumeRunID, mapped.Work[0].ResumeRunID)
}

func TestAgentAdapter_SnapshotProjectWithoutRunDoesNotCreateWork(t *testing.T) {
	adapter := requireAgentAdapter(t, &fixtureNativeAgentEngine{snapshot: work.Snapshot{Projects: []work.Project{{ID: "project-only", RepositoryName: "private-project"}}}})
	result := adapter.Snapshot(context.Background())
	if !result.OK {
		t.Fatalf("Snapshot: %s", result.Error())
	}
	snapshot := result.Value.(agentSnapshot)
	if len(snapshot.Work) != 0 {
		t.Fatalf("project-only snapshot created Work: %#v", snapshot.Work)
	}
}

func TestAgentAdapter_SnapshotKeepsHistoricalQuestionWithoutStaleAttention(t *testing.T) {
	at := time.Date(2026, time.July, 18, 15, 0, 0, 0, time.UTC)
	adapter := requireAgentAdapter(t, &fixtureNativeAgentEngine{snapshot: work.Snapshot{
		Runs: []work.Run{
			{ID: "run-parent", WorkID: "work-1", Status: work.RunWaiting},
			{ID: "run-child", WorkID: "work-1", Status: work.RunInterrupted},
		},
		Questions: []work.Question{{ID: "question-parent", RunID: "run-parent", Text: "Which target?", CreatedAt: at}},
	}})

	result := adapter.Snapshot(context.Background())
	if !result.OK {
		t.Fatalf("Snapshot: %s", result.Error())
	}
	snapshot := result.Value.(agentSnapshot)
	if len(snapshot.Work) != 1 || snapshot.Work[0].NativeRunID != "run-child" || snapshot.Work[0].Question != "" || snapshot.Work[0].QuestionID != "" {
		t.Fatalf("selected child attention = %#v", snapshot.Work)
	}
	if len(snapshot.Events) != 1 || snapshot.Events[0].ExternalID != "question-parent" || snapshot.Events[0].RunID != "run-parent" || snapshot.Events[0].WorkID != "work-1" || snapshot.Events[0].Kind != "question" || snapshot.Events[0].Detail != "Which target?" {
		t.Fatalf("historical question timeline = %#v", snapshot.Events)
	}
}

func TestAgentAdapter_SnapshotReviewUsesFreshReviewPresentation(t *testing.T) {
	review := workspace.ChangeReview{
		WorkID: "work-1", RunID: "run-1", SourceBranch: "main", SourceRevision: "source-123",
		AgentBase: "source-123", AgentTip: "agent-456", IntegrationBranch: "lem/integration/run-1",
		IntegrationPath: "/private/reviews/run-1", ResultRevision: "result-789",
		CommitLog: "agent-456 implement reviewed change", Diff: "diff --git a/a.go b/a.go\n+reviewed",
		Validation: []workspace.ValidationResult{{
			Command:  workspace.Command{Dir: "/src/project", Executable: "go", Args: []string{"test", "./..."}},
			ExitCode: 0, Output: "ok all packages", Receipt: "receipt-sha256", Passed: true,
		}},
		Conflicts: []string{},
	}
	engine := &fixtureNativeAgentEngine{
		changeReview: review,
		snapshot: work.Snapshot{
			Runs: []work.Run{{ID: review.RunID, WorkID: review.WorkID, Status: work.RunCompleted}},
			Acceptances: []work.Acceptance{{
				ID: "review-1", WorkID: review.WorkID, RunID: review.RunID,
				Status: "prepared", ValidationJSON: core.JSONMarshalString(review),
			}},
		},
	}
	adapter := requireAgentAdapter(t, engine)
	freshResult := adapter.Review(context.Background(), agentReviewRequest{Feature: agentFeatureChangesReview, WorkID: review.RunID})
	snapshotResult := adapter.Snapshot(context.Background())
	if !freshResult.OK || !snapshotResult.OK {
		t.Fatalf("fresh/snapshot review = %#v / %#v", freshResult, snapshotResult)
	}
	fresh := freshResult.Value.(agentReview)
	decoded := snapshotResult.Value.(agentSnapshot).Work[0].Review
	if decoded.Body != fresh.Body || decoded.Warning != fresh.Warning || core.JSONMarshalString(decoded.Payload) != core.JSONMarshalString(review) {
		t.Fatalf("snapshot review differs from fresh review:\nfresh=%#v\ndecoded=%#v", fresh, decoded)
	}
	for _, want := range []string{
		"Source branch: main", "Source revision: source-123", "Agent base: source-123",
		"Agent tip: agent-456", "Result revision: result-789", "agent-456 implement reviewed change",
		"+reviewed", "PASSED", "go test ./...", "receipt-sha256", "ok all packages", "Conflicts:\nnone",
	} {
		if !strings.Contains(decoded.Body, want) {
			t.Fatalf("snapshot review missing %q:\n%s", want, decoded.Body)
		}
	}

	noValidation := review
	noValidation.Validation = nil
	engine.changeReview = noValidation
	engine.snapshot.Acceptances[0].ValidationJSON = core.JSONMarshalString(noValidation)
	freshResult = adapter.Review(context.Background(), agentReviewRequest{Feature: agentFeatureChangesReview, WorkID: review.RunID})
	snapshotResult = adapter.Snapshot(context.Background())
	fresh = freshResult.Value.(agentReview)
	decoded = snapshotResult.Value.(agentSnapshot).Work[0].Review
	if decoded.Body != fresh.Body || decoded.Warning != fresh.Warning || !decoded.NeedsAcknowledgement || !strings.Contains(decoded.Body, "No validation command configured") {
		t.Fatalf("no-validation snapshot review = %#v, fresh %#v", decoded, fresh)
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
	if core.JSONMarshalString(engine.reviewedProject) != core.JSONMarshalString(projectReview.Work) {
		t.Fatalf("ReviewProject item = %#v, want %#v", engine.reviewedProject, projectReview.Work)
	}
	if core.JSONMarshalString(engine.registeredReview) != core.JSONMarshalString(projectReview) || !engine.registerConfirmed {
		t.Fatalf("RegisterProject args = review %#v confirmed %v, want %#v true", engine.registeredReview, engine.registerConfirmed, projectReview)
	}
	wantDispatchRequest := work.DispatchRequest{Work: projectReview.Work, Provider: "codex", Model: "gpt-5", ConfirmedSourceRevision: dispatchReview.Project.SourceRevision}
	if core.JSONMarshalString(engine.reviewedDispatch) != core.JSONMarshalString(wantDispatchRequest) {
		t.Fatalf("ReviewDispatch request = %#v, want %#v", engine.reviewedDispatch, wantDispatchRequest)
	}

	dispatched := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureDispatch, Review: launch, Confirmed: true})
	if !dispatched.OK || engine.dispatchCalls != 1 {
		t.Fatalf("dispatch = %#v, calls=%d", dispatched, engine.dispatchCalls)
	}
	if core.JSONMarshalString(engine.dispatchedReview) != core.JSONMarshalString(dispatchReview) {
		t.Fatalf("Dispatch review = %#v, want %#v", engine.dispatchedReview, dispatchReview)
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
	retryReview := adapter.Review(context.Background(), agentReviewRequest{Feature: agentFeatureRetry, WorkID: "run-1", Work: workRequest})
	if !retryReview.OK {
		t.Fatalf("retry Review: %s", retryReview.Error())
	}
	if retried := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureRetry, Review: retryReview.Value.(agentReview), Confirmed: true}); !retried.OK {
		t.Fatalf("confirmed Retry: %s", retried.Error())
	}
	resumeReview := adapter.Review(context.Background(), agentReviewRequest{Feature: agentFeatureResume, WorkID: "run-1", Input: "answer-1", Provider: "codex", Model: "gpt-5", Work: workRequest})
	if !resumeReview.OK {
		t.Fatalf("resume Review: %s", resumeReview.Error())
	}
	if resumed := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureResume, Review: resumeReview.Value.(agentReview), Confirmed: true}); !resumed.OK {
		t.Fatalf("confirmed Resume: %s", resumed.Error())
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
	wantCalls := []string{"cancel", "answer", "queue.start", "queue.stop", "reject", "retry.review", "retry", "resume.review", "resume", "changes.review", "accept"}
	if core.JSONMarshalString(engine.calls) != core.JSONMarshalString(wantCalls) {
		t.Fatalf("mapped action order = %#v, want %#v", engine.calls, wantCalls)
	}
	if engine.cancelRunID != "run-1" || engine.answerRunID != "run-1" || engine.answerText != "Use target A" {
		t.Fatalf("cancel/answer args = %q / %q %q", engine.cancelRunID, engine.answerRunID, engine.answerText)
	}
	if core.JSONMarshalString(engine.retryItem) != core.JSONMarshalString(nativeWorkItem(workRequest, workRequest.ID, "")) || engine.retryParent != "run-1" {
		t.Fatalf("Retry args = item %#v parent %q", engine.retryItem, engine.retryParent)
	}
	wantResume := work.ResumeRequest{Work: nativeWorkItem(workRequest, workRequest.ID, ""), ParentRunID: "run-1", AnswerID: "answer-1", Provider: "codex", Model: "gpt-5"}
	if core.JSONMarshalString(engine.resumeRequest) != core.JSONMarshalString(wantResume) {
		t.Fatalf("Resume request = %#v, want %#v", engine.resumeRequest, wantResume)
	}
	if engine.reviewChangesRunID != "run-1" || engine.rejectRunID != "run-1" {
		t.Fatalf("review/reject run IDs = %q / %q", engine.reviewChangesRunID, engine.rejectRunID)
	}
	wantAccept := workspace.AcceptRequest{Review: changeReview, Confirmed: true}
	if core.JSONMarshalString(engine.acceptRequest) != core.JSONMarshalString(wantAccept) {
		t.Fatalf("Accept request = %#v, want %#v", engine.acceptRequest, wantAccept)
	}
}

func TestAgentAdapterChildActionsRequireReviewedConfirmation(t *testing.T) {
	engine := &fixtureNativeAgentEngine{}
	adapter := requireAgentAdapter(t, engine)
	request := agentReviewRequest{
		Feature: agentFeatureRetry, WorkID: "run-parent",
		Work: agentWorkRequest{ID: "work-1", Title: "Work one", Task: "Task one", Repository: "/src/one"},
	}
	reviewed := adapter.Review(context.Background(), request)
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	review := reviewed.Value.(agentReview)
	core.AssertTrue(t, review.ConfirmRequired)
	core.AssertEqual(t, agentFeatureRetry, review.Feature)
	core.AssertFalse(t, adapter.Run(context.Background(), agentRequest{Feature: agentFeatureRetry, Review: review}).OK)
	confirmed := adapter.Run(context.Background(), agentRequest{Feature: agentFeatureRetry, Review: review, Confirmed: true})
	core.AssertTrue(t, confirmed.OK, confirmed.Error())
}

func TestAgentAdapter_Close_Ugly(t *testing.T) {
	engine := &fixtureNativeAgentEngine{closeFailures: 1}
	adapter := requireAgentAdapter(t, engine)
	if result := adapter.Close(); result.OK {
		t.Fatal("first Close unexpectedly ignored injected ownership cleanup failure")
	}
	if result := adapter.Close(); !result.OK {
		t.Fatalf("retried Close: %s", result.Error())
	}
	if result := adapter.Close(); !result.OK {
		t.Fatalf("idempotent Close: %s", result.Error())
	}
	if engine.closeCalls != 2 {
		t.Fatalf("engine Close calls = %d, want 2", engine.closeCalls)
	}
}

func TestAgentAdapter_CloseConcurrentCapabilities_Ugly(t *testing.T) {
	engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}
	adapter := requireAgentAdapter(t, engine)
	var workers sync.WaitGroup
	start := make(chan struct{})
	for index := 0; index < 32; index++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			<-start
			for iteration := 0; iteration < 200; iteration++ {
				if len(adapter.Capabilities()) == 0 {
					t.Error("Capabilities returned an empty catalog")
					return
				}
			}
		}()
	}
	workers.Add(1)
	go func() {
		defer workers.Done()
		<-start
		if result := adapter.Close(); !result.OK {
			t.Errorf("Close: %s", result.Error())
		}
	}()
	close(start)
	workers.Wait()
	if engine.closeCalls != 1 {
		t.Fatalf("engine Close calls = %d, want 1", engine.closeCalls)
	}
}

func TestAgentAdapter_DurableInterruptedResumeAfterRestartReceipt(t *testing.T) {
	root, repository, agentStore := openTestAgentStore(t)
	initialRepositoryClosed := false
	t.Cleanup(func() {
		if initialRepositoryClosed {
			return
		}
		if result := repository.Close(); !result.OK {
			t.Errorf("cleanup initial repository/store: %v", result.Value)
		}
	})
	at := time.Date(2026, time.July, 18, 18, 0, 0, 0, time.UTC)
	task := "Resume the durable interrupted run"
	project := testDurableAgentProject(t, at)
	parent := testAgentRun("run-interrupted", work.RunInterrupted, at)
	parent.ProcessID = 0
	parent.SourceRevision = project.SourceRevision
	parent.DurableRevision = project.SourceRevision
	parent.ExecutionRevision = project.SourceRevision
	parent.Worktree = t.TempDir()
	event := work.Event{
		ID: "event-interrupted", RunID: parent.ID, WorkID: parent.WorkID,
		Kind: "queued", Title: "queued", Detail: task, CreatedAt: at,
	}
	logs := []work.LogChunk{
		{RunID: parent.ID, Sequence: 1, Stream: "stdout", Text: "durable-one", CreatedAt: at.Add(time.Second)},
		{RunID: parent.ID, Sequence: 2, Stream: "stderr", Text: "durable-two", CreatedAt: at.Add(2 * time.Second)},
	}
	initial := newApp("", 0, 64)
	if result := initial.attachWork(repository, requireAgentAdapter(t, &fixtureNativeAgentEngine{})); !result.OK {
		t.Fatalf("attach initial Work: %v", result.Value)
	}
	initial.work.ids = sequenceIDs(parent.WorkID)
	if result := initial.work.CreateWork("Durable restart", task, project.SourcePath); !result.OK {
		t.Fatalf("create durable Work: %v", result.Value)
	}
	if result := agentStore.Commit(orchestrator.Commit{
		Project: &project, Run: &parent, CreateRun: true, Event: &event, Logs: logs,
	}); !result.OK {
		t.Fatalf("persist interrupted parent: %v", result.Value)
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close initial repository/store: %v", result.Value)
	}
	initialRepositoryClosed = true

	reopenedResult := openDuckRepository(root + "/lem.duckdb")
	if !reopenedResult.OK {
		t.Fatalf("reopen durable repository: %v", reopenedResult.Value)
	}
	reopened := reopenedResult.Value.(workspaceRepository)
	t.Cleanup(func() {
		if result := reopened.Close(); !result.OK {
			t.Errorf("cleanup reopened repository/store: %v", result.Value)
		}
	})
	reopenedStore := requireAgentValue[*duckAgentStore](t, "newDuckAgentStore reopened", newDuckAgentStore(reopened))
	reopenedSnapshot := requireAgentValue[work.Snapshot](t, "reopened Snapshot", reopenedStore.Snapshot(parent.WorkID))
	if len(reopenedSnapshot.Runs) != 1 || reopenedSnapshot.Runs[0].ID != parent.ID || reopenedSnapshot.Runs[0].Status != work.RunInterrupted {
		t.Fatalf("reopened parent = %#v", reopenedSnapshot.Runs)
	}
	if len(reopenedSnapshot.Logs) != 2 || reopenedSnapshot.Logs[0].Text != "durable-one" || reopenedSnapshot.Logs[1].Text != "durable-two" || reopenedSnapshot.Logs[0].Sequence >= reopenedSnapshot.Logs[1].Sequence {
		t.Fatalf("reopened ordered logs = %#v", reopenedSnapshot.Logs)
	}
	storedParent := requireAgentValue[work.Run](t, "parent before UI Resume", reopenedStore.Run(parent.ID))
	engine := newDurableResumeReceiptEngine(t, reopenedStore, at.Add(time.Hour))
	if _, available := engine.(nativeChildReviewEngine); !available {
		_ = engine.Close()
		t.Skip("standalone inference dependency predates reviewed child continuation APIs")
	}
	var adapter agentProvider
	t.Cleanup(func() {
		var result core.Result
		if adapter != nil {
			result = adapter.Close()
		} else {
			result = engine.Close()
		}
		if !result.OK {
			t.Errorf("cleanup restarted native adapter/orchestrator: %v", result.Value)
		}
	})
	adapter = requireAgentAdapter(t, engine)
	restarted := newApp("", 0, 64)
	if result := restarted.attachWork(reopened, adapter); !result.OK {
		t.Fatalf("attach restarted Work: %v", result.Value)
	}
	driveCorrectiveCommand(t, &restarted, restarted.requestAgentSnapshot())
	selected, ok := restarted.work.Selected()
	if !ok || selected.ID != parent.WorkID || selected.Status != string(work.RunInterrupted) {
		t.Fatalf("restarted selected Work = %#v, selected=%v", selected, ok)
	}
	if result := restarted.queueAgentAction(agentFeatureResume); !result.OK {
		t.Fatalf("queue interrupted UI Resume: %v", result.Value)
	}
	driveCorrectiveCommand(t, &restarted, restarted.takeAgentCommand())
	if restarted.activeOverlay != overlayLaunchReview {
		t.Fatalf("interrupted Resume did not present launch review: overlay=%d err=%q", restarted.activeOverlay, restarted.errText)
	}
	beforeConfirm := requireAgentValue[work.Snapshot](t, "Snapshot before explicit Resume confirmation", reopenedStore.Snapshot(parent.WorkID))
	if len(beforeConfirm.Runs) != 1 {
		t.Fatalf("interrupted Resume created child before confirmation: %#v", beforeConfirm.Runs)
	}
	model, command := restarted.Update(tea.KeyMsg{Type: tea.KeyEnter})
	restarted = model.(app)
	next := driveCorrectiveCommand(t, &restarted, command)
	if next != nil {
		driveCorrectiveCommand(t, &restarted, next)
	}

	resumedSnapshot := requireAgentValue[work.Snapshot](t, "Snapshot after UI Resume", reopenedStore.Snapshot(parent.WorkID))
	var child work.Run
	for _, candidate := range resumedSnapshot.Runs {
		if candidate.ParentRunID == parent.ID {
			child = candidate
		}
	}
	if child.ID == "" || child.ID == parent.ID || child.Status != work.RunQueued || child.Attempt != parent.Attempt+1 || child.Branch != parent.Branch || child.Worktree != parent.Worktree {
		t.Fatalf("durable resumed child = %#v", child)
	}
	afterParent := requireAgentValue[work.Run](t, "parent after UI Resume", reopenedStore.Run(parent.ID))
	if core.JSONMarshalString(afterParent) != core.JSONMarshalString(storedParent) {
		t.Fatalf("UI Resume mutated durable parent:\nbefore=%#v\nafter=%#v", storedParent, afterParent)
	}
	if len(resumedSnapshot.Logs) != 2 || resumedSnapshot.Logs[0] != reopenedSnapshot.Logs[0] || resumedSnapshot.Logs[1] != reopenedSnapshot.Logs[1] {
		t.Fatalf("UI Resume changed durable parent logs: %#v", resumedSnapshot.Logs)
	}
}

func TestAgentAdapterDurableAnsweredWaitingResumeAfterRestartReceipt(t *testing.T) {
	root, repository, agentStore := openTestAgentStore(t)
	initialRepositoryClosed := false
	t.Cleanup(func() {
		if !initialRepositoryClosed {
			_ = repository.Close()
		}
	})
	at := time.Date(2026, time.July, 19, 11, 0, 0, 0, time.UTC)
	task := "Resume the durable answered waiting run"
	project := testDurableAgentProject(t, at)
	parent := testAgentRun("run-answered-waiting", work.RunWaiting, at)
	parent.ProcessID = 0
	parent.SourceRevision = project.SourceRevision
	parent.DurableRevision = project.SourceRevision
	parent.ExecutionRevision = project.SourceRevision
	parent.Worktree = t.TempDir()
	question := work.Question{ID: "question-reopened", RunID: parent.ID, Text: "Which target?", CreatedAt: at.Add(time.Second)}
	answer := work.Answer{
		ID: "answer-reopened", QuestionID: question.ID, ResumeRunID: "resume-reopened",
		Text: "Use target A", CreatedAt: at.Add(2 * time.Second),
	}
	event := work.Event{
		ID: "event-answered-waiting", RunID: parent.ID, WorkID: parent.WorkID,
		Kind: "queued", Title: "queued", Detail: task, CreatedAt: at,
	}
	initial := newApp("", 0, 64)
	if result := initial.attachWork(repository, requireAgentAdapter(t, &fixtureNativeAgentEngine{})); !result.OK {
		t.Fatalf("attach initial answered Work: %s", result.Error())
	}
	initial.work.ids = sequenceIDs(parent.WorkID)
	if result := initial.work.CreateWork("Answered restart", task, project.SourcePath); !result.OK {
		t.Fatalf("create durable answered Work: %s", result.Error())
	}
	if result := agentStore.Commit(orchestrator.Commit{
		Project: &project, Run: &parent, CreateRun: true, Event: &event, Question: &question, Answer: &answer,
	}); !result.OK {
		t.Fatalf("persist answered waiting parent: %s", result.Error())
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close answered repository/store: %s", result.Error())
	}
	initialRepositoryClosed = true

	reopenedResult := openDuckRepository(core.PathJoin(root, "lem.duckdb"))
	if !reopenedResult.OK {
		t.Fatalf("reopen answered repository: %s", reopenedResult.Error())
	}
	reopened := reopenedResult.Value.(workspaceRepository)
	t.Cleanup(func() { _ = reopened.Close() })
	reopenedStore := requireAgentValue[*duckAgentStore](t, "newDuckAgentStore answered reopened", newDuckAgentStore(reopened))
	engine := newDurableResumeReceiptEngine(t, reopenedStore, at.Add(time.Hour))
	if _, available := engine.(nativeChildReviewEngine); !available {
		_ = engine.Close()
		t.Skip("standalone inference dependency predates reviewed child continuation APIs")
	}
	adapter := requireAgentAdapter(t, engine)
	t.Cleanup(func() { _ = adapter.Close() })
	restarted := newApp("", 0, 64)
	if result := restarted.attachWork(reopened, adapter); !result.OK {
		t.Fatalf("attach restarted answered Work: %s", result.Error())
	}
	driveCorrectiveCommand(t, &restarted, restarted.requestAgentSnapshot())
	selected, ok := restarted.work.Selected()
	if !ok || selected.ID != parent.WorkID || selected.Status != string(work.RunWaiting) {
		t.Fatalf("reopened answered selection = %#v selected=%v", selected, ok)
	}
	state := restarted.work.AgentState(selected)
	if state.NativeRunID != parent.ID || state.QuestionID != question.ID || state.AnswerID != answer.ID || state.ResumeRunID != answer.ResumeRunID {
		t.Fatalf("reopened durable answer projection = %#v", state)
	}
	if result := restarted.queueAgentAction(agentFeatureResume); !result.OK {
		t.Fatalf("queue answered Resume: %s", result.Error())
	}
	driveCorrectiveCommand(t, &restarted, restarted.takeAgentCommand())
	if restarted.activeOverlay != overlayLaunchReview {
		t.Fatalf("answered Resume did not present launch review: overlay=%d err=%q", restarted.activeOverlay, restarted.errText)
	}
	beforeConfirm := requireAgentValue[work.Snapshot](t, "answered Snapshot before confirmation", reopenedStore.Snapshot(parent.WorkID))
	if len(beforeConfirm.Runs) != 1 {
		t.Fatalf("answered Resume created child before confirmation: %#v", beforeConfirm.Runs)
	}
	model, confirm := restarted.Update(tea.KeyMsg{Type: tea.KeyEnter})
	restarted = model.(app)
	next := driveCorrectiveCommand(t, &restarted, confirm)
	if next != nil {
		driveCorrectiveCommand(t, &restarted, next)
	}
	after := requireAgentValue[work.Snapshot](t, "answered Snapshot after confirmation", reopenedStore.Snapshot(parent.WorkID))
	var child work.Run
	for _, candidate := range after.Runs {
		if candidate.ParentRunID == parent.ID {
			child = candidate
		}
	}
	if child.ID != answer.ResumeRunID || child.Status != work.RunQueued || child.Attempt != parent.Attempt+1 || child.Branch != parent.Branch || child.Worktree != parent.Worktree {
		t.Fatalf("durable answered resumed child = %#v", child)
	}
}

func newDurableResumeReceiptEngine(t *testing.T, store orchestrator.Store, at time.Time) nativeAgentEngine {
	t.Helper()
	workspaceRoot := t.TempDir()
	files, filesErr := coreio.NewSandboxed(workspaceRoot)
	if filesErr != nil {
		t.Fatalf("construct durable receipt workspace medium: %v", filesErr)
	}
	managerResult := workspace.NewManager(workspace.ManagerOptions{
		Root: workspaceRoot, Files: files, Git: agentAdapterDirectGitRunner{}, Server: &fixtureGitServer{},
		IDs: func() string { return "durable-review" }, Now: func() time.Time { return at },
	})
	if !managerResult.OK {
		t.Fatalf("construct durable receipt workspace manager: %s", managerResult.Error())
	}
	registryResult := provider.NewRegistry(&fixtureNativeProvider{name: "codex", available: true})
	if !registryResult.OK {
		t.Fatalf("construct durable receipt provider registry: %v", registryResult.Value)
	}
	policy := queue.Policy{
		Version: 1,
		Dispatch: queue.DispatchConfig{
			DefaultAgent: "codex", GlobalConcurrency: 1, TimeoutMinutes: 1,
		},
		Concurrency: map[string]queue.ConcurrencyLimit{"codex": {Total: 1}},
		Rates:       map[string]queue.RateConfig{},
		Providers:   map[string]queue.NativeConfig{"codex": {Executable: "codex"}},
	}
	queueResult := queue.NewController(policy, work.QueueState{ID: "default", Status: work.QueueFrozen}, nil)
	if !queueResult.OK {
		t.Fatalf("construct durable receipt queue: %v", queueResult.Value)
	}
	engineResult := orchestrator.New(orchestrator.Options{
		Store: store, GitServer: &fixtureGitServer{}, Workspaces: managerResult.Value.(*workspace.Manager),
		Providers: registryResult.Value.(*provider.Registry), Queue: queueResult.Value.(*queue.Controller),
		Launcher: &fixtureAgentLauncher{}, Clock: agentClock{now: func() time.Time { return at }},
		IDs: &fixtureAgentIdentifiers{},
	})
	if !engineResult.OK {
		t.Fatalf("construct durable receipt orchestrator: %v", engineResult.Value)
	}
	return engineResult.Value.(nativeAgentEngine)
}

func testDurableAgentProject(t *testing.T, at time.Time) work.Project {
	t.Helper()
	source := t.TempDir()
	canonical := core.PathEvalSymlinks(source)
	if !canonical.OK {
		t.Fatalf("canonicalize durable review source: %s", canonical.Error())
	}
	source = canonical.Value.(string)
	agentAdapterGit(t, source, "init", "-b", "main")
	agentAdapterGit(t, source, "config", "user.name", "LEM Test")
	agentAdapterGit(t, source, "config", "user.email", "lem@example.invalid")
	if result := core.WriteFile(core.PathJoin(source, "README.md"), []byte("durable review source\n"), 0o600); !result.OK {
		t.Fatalf("write durable review source: %s", result.Error())
	}
	agentAdapterGit(t, source, "add", "README.md")
	agentAdapterGit(t, source, "commit", "-m", "durable review source")
	revision := core.Trim(agentAdapterGit(t, source, "rev-parse", "HEAD"))
	project := testAgentProject(at)
	project.SourcePath = source
	project.RepositoryRoot = source
	project.SourceBranch = "main"
	project.SourceRevision = revision
	return project
}

func agentAdapterGit(t *testing.T, directory string, arguments ...string) string {
	t.Helper()
	result := (agentAdapterDirectGitRunner{}).Run(context.Background(), workspace.Command{
		Dir: directory, Executable: "git", Args: arguments,
		Environment: []string{"GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", "LC_ALL=C"},
	})
	if !result.OK {
		t.Fatalf("git %s: %s", core.Join(" ", arguments...), result.Error())
	}
	return result.Value.(string)
}

type agentAdapterDirectGitRunner struct{}

func (agentAdapterDirectGitRunner) Run(ctx context.Context, command workspace.Command) core.Result {
	stdout := core.NewBuffer()
	stderr := core.NewBuffer()
	result := commandexec.Command(ctx, command.Executable, command.Args...).
		WithDir(command.Dir).
		WithEnv(command.Environment).
		WithStdout(stdout).
		WithStderr(stderr).
		Run()
	output := core.Concat(stdout.String(), stderr.String())
	if !result.OK {
		return core.Fail(core.E("test.agentAdapterDirectGitRunner", output, result.Err()))
	}
	return core.Ok(output)
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
	reviewedProject     work.Item
	registeredReview    orchestrator.ProjectReview
	registerProject     func(context.Context, orchestrator.ProjectReview, bool) core.Result
	reviewedDispatch    work.DispatchRequest
	dispatchedReview    orchestrator.DispatchReview
	cancelRunID         string
	answerRunID         string
	answerText          string
	retryItem           work.Item
	retryParent         string
	resumeRequest       work.ResumeRequest
	reviewChangesRunID  string
	acceptRequest       workspace.AcceptRequest
	rejectRunID         string
	calls               []string
	registerCalls       int
	registerConfirmed   bool
	reviewDispatchCalls int
	dispatchCalls       int
	closeCalls          int
	closeFailures       int
}

func (engine *fixtureNativeAgentEngine) Capabilities() []work.Capability {
	return append([]work.Capability(nil), engine.capabilities...)
}

func (engine *fixtureNativeAgentEngine) Snapshot(_ context.Context, workID string) core.Result {
	engine.snapshotWorkID = workID
	return core.Ok(engine.snapshot)
}

func (engine *fixtureNativeAgentEngine) ReviewProject(_ context.Context, item work.Item) core.Result {
	engine.reviewedProject = item
	return core.Ok(engine.projectReview)
}

func (engine *fixtureNativeAgentEngine) RegisterProject(ctx context.Context, review orchestrator.ProjectReview, confirmed bool) core.Result {
	engine.registerCalls++
	engine.registerConfirmed = confirmed
	engine.registeredReview = review
	if engine.registerProject != nil {
		return engine.registerProject(ctx, review, confirmed)
	}
	return core.Ok(engine.registeredProject)
}

func (engine *fixtureNativeAgentEngine) ReviewDispatch(_ context.Context, request work.DispatchRequest) core.Result {
	engine.reviewDispatchCalls++
	engine.reviewedDispatch = request
	return core.Ok(engine.dispatchReview)
}

func (engine *fixtureNativeAgentEngine) Dispatch(_ context.Context, review orchestrator.DispatchReview) core.Result {
	engine.dispatchCalls++
	engine.dispatchedReview = review
	return core.Ok(work.Run{ID: "run-1", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) Cancel(_ context.Context, runID string) core.Result {
	engine.calls = append(engine.calls, "cancel")
	engine.cancelRunID = runID
	return core.Ok(work.Run{ID: "run-1", WorkID: "work-1", Status: work.RunCancelled})
}

func (engine *fixtureNativeAgentEngine) Answer(_ context.Context, runID, text string) core.Result {
	engine.calls = append(engine.calls, "answer")
	engine.answerRunID = runID
	engine.answerText = text
	return core.Ok(work.Answer{ID: "answer-1", ResumeRunID: "run-2"})
}

func (engine *fixtureNativeAgentEngine) Retry(_ context.Context, item work.Item, parentRunID string) core.Result {
	engine.calls = append(engine.calls, "retry.legacy")
	engine.retryItem = item
	engine.retryParent = parentRunID
	return core.Ok(work.Run{ID: "run-2", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) Resume(_ context.Context, request work.ResumeRequest) core.Result {
	engine.calls = append(engine.calls, "resume.legacy")
	engine.resumeRequest = request
	return core.Ok(work.Run{ID: "run-3", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) ReviewRetry(_ context.Context, item work.Item, parentRunID string) core.Result {
	engine.calls = append(engine.calls, "retry.review")
	engine.retryItem = item
	engine.retryParent = parentRunID
	return core.Ok(fixtureChildReview("retry", "run-2", "fake", "test-model"))
}

func (engine *fixtureNativeAgentEngine) ConfirmRetry(_ context.Context, _ any) core.Result {
	engine.calls = append(engine.calls, "retry")
	return core.Ok(work.Run{ID: "run-2", WorkID: "work-1", Status: work.RunQueued})
}

func (engine *fixtureNativeAgentEngine) ReviewResume(_ context.Context, request work.ResumeRequest) core.Result {
	engine.calls = append(engine.calls, "resume.review")
	engine.resumeRequest = request
	return core.Ok(fixtureChildReview("resume", "run-3", request.Provider, request.Model))
}

func (engine *fixtureNativeAgentEngine) ConfirmResume(_ context.Context, _ any) core.Result {
	engine.calls = append(engine.calls, "resume")
	return core.Ok(work.Run{ID: "run-3", WorkID: "work-1", Status: work.RunQueued})
}

func fixtureChildReview(action, runID, providerName, model string) agentChildReviewProjection {
	review := agentChildReviewProjection{
		Action: action, RunID: runID, Provider: providerName, Model: model,
		WorktreePath: "/private/runs/parent/worktree", Branch: "lem/work-1/1",
		Warning: "native host access",
	}
	review.Project.RepositoryName = "private-one"
	review.Source.Path, review.Source.Branch, review.Source.Revision = "/src/one", "main", "abc123"
	review.Command.Receipt = core.Concat(providerName, " run <redacted>")
	review.Queue.Reason = "queue frozen"
	return review
}

func (engine *fixtureNativeAgentEngine) StartQueue(context.Context) core.Result {
	engine.calls = append(engine.calls, "queue.start")
	return core.Ok(work.QueueState{ID: "default", Status: work.QueueAccepting})
}

func (engine *fixtureNativeAgentEngine) StopQueue(context.Context) core.Result {
	engine.calls = append(engine.calls, "queue.stop")
	return core.Ok(work.QueueState{ID: "default", Status: work.QueueFrozen})
}

func (engine *fixtureNativeAgentEngine) ReviewChanges(_ context.Context, runID string) core.Result {
	engine.calls = append(engine.calls, "changes.review")
	engine.reviewChangesRunID = runID
	return core.Ok(engine.changeReview)
}

func (engine *fixtureNativeAgentEngine) Accept(_ context.Context, request workspace.AcceptRequest) core.Result {
	engine.calls = append(engine.calls, "accept")
	engine.acceptRequest = request
	return core.Ok(work.Acceptance{ID: "accept-1", WorkID: "work-1", RunID: "run-1", Status: "accepted"})
}

func (engine *fixtureNativeAgentEngine) Reject(_ context.Context, runID string) core.Result {
	engine.calls = append(engine.calls, "reject")
	engine.rejectRunID = runID
	return core.Ok(work.Acceptance{ID: "reject-1", WorkID: "work-1", RunID: "run-1", Status: "rejected"})
}

func (engine *fixtureNativeAgentEngine) Close() core.Result {
	engine.closeCalls++
	if engine.closeFailures > 0 {
		engine.closeFailures--
		return core.Fail(core.E("test.nativeAgentEngine.Close", "injected ownership cleanup failure", nil))
	}
	return core.Ok(nil)
}

func nativeFixtureCapabilities() []work.Capability {
	return []work.Capability{
		{Name: "dispatch", Available: true}, {Name: "cancel", Available: true},
		{Name: "answer", Available: true}, {Name: "retry", Available: true},
		{Name: "resume", Available: true}, {Name: "queue.start", Available: true},
		{Name: "queue.stop", Available: true}, {Name: "changes.review", Available: true},
		{Name: "accept", Available: true}, {Name: "reject", Available: true},
	}
}
