// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"sort"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
)

// nativeAgentEngine is the reusable orchestration surface consumed at the
// single private adapter boundary. Bubble Tea code only sees the private
// request, review, snapshot, and receipt values declared in agentcap.go.
type nativeAgentEngine interface {
	Capabilities() []work.Capability
	Snapshot(context.Context, string) core.Result
	ReviewProject(context.Context, work.Item) core.Result
	RegisterProject(context.Context, orchestrator.ProjectReview, bool) core.Result
	ReviewDispatch(context.Context, work.DispatchRequest) core.Result
	Dispatch(context.Context, orchestrator.DispatchReview) core.Result
	Cancel(context.Context, string) core.Result
	Answer(context.Context, string, string) core.Result
	Retry(context.Context, work.Item, string) core.Result
	Resume(context.Context, work.ResumeRequest) core.Result
	StartQueue(context.Context) core.Result
	StopQueue(context.Context) core.Result
	ReviewChanges(context.Context, string) core.Result
	Accept(context.Context, workspace.AcceptRequest) core.Result
	Reject(context.Context, string) core.Result
	Close() core.Result
}

type nativeAgentAdapter struct {
	engine       nativeAgentEngine
	availability *agentAvailability
	closeOnce    sync.Once
	closeResult  core.Result
}

type agentProjectRegistration struct {
	Review   orchestrator.ProjectReview
	Provider string
	Model    string
}

func newAgentAdapter(engine nativeAgentEngine) core.Result {
	return newAgentAdapterWithAvailability(engine, nil)
}

func newAgentAdapterWithAvailability(engine nativeAgentEngine, availability *agentAvailability) core.Result {
	if engine == nil {
		return core.Fail(core.E("tui.newAgentAdapter", "native agent engine is required", nil))
	}
	return core.Ok(agentProvider(&nativeAgentAdapter{engine: engine, availability: availability, closeResult: core.Ok(nil)}))
}

func (adapter *nativeAgentAdapter) Capabilities() []agentCapability {
	if adapter == nil || adapter.engine == nil {
		return newUnavailableAgentProvider("native agent engine is unavailable").Capabilities()
	}
	reported := make(map[agentFeature]work.Capability)
	for _, capability := range adapter.engine.Capabilities() {
		feature := agentFeature(core.Trim(capability.Name))
		if isNativeAgentFeature(feature) {
			reported[feature] = capability
		}
	}
	catalog := agentFeatureCatalog("")
	for index := range catalog {
		feature := catalog[index].Feature
		if capability, ok := reported[feature]; ok {
			catalog[index].Available = capability.Available
			catalog[index].Reason = core.Trim(capability.Reason)
			if !catalog[index].Available && catalog[index].Reason == "" {
				catalog[index].Reason = "native agent engine reports this capability unavailable"
			}
			if reason := adapter.availability.reason(feature); reason != "" {
				catalog[index].Available = false
				catalog[index].Reason = reason
			}
			continue
		}
		if isNativeAgentFeature(feature) {
			catalog[index].Reason = "native agent engine does not report this capability"
			continue
		}
		catalog[index].Reason = futureAgentFeatureReason(feature)
	}
	return catalog
}

func isNativeAgentFeature(feature agentFeature) bool {
	switch feature {
	case agentFeatureDispatch, agentFeatureCancel, agentFeatureAnswer, agentFeatureRetry,
		agentFeatureResume, agentFeatureQueueStart, agentFeatureQueueStop,
		agentFeatureChangesReview, agentFeatureAccept, agentFeatureReject:
		return true
	default:
		return false
	}
}

func futureAgentFeatureReason(feature agentFeature) string {
	reasons := map[agentFeature]string{
		agentFeatureSetup:         "project setup is performed through reviewed dispatch registration",
		agentFeatureProvider:      "provider policy is configured in agents.yaml",
		agentFeatureTemplate:      "agent templates are not part of the native execution slice",
		agentFeaturePlan:          "agent planning is not part of the native execution slice",
		agentFeatureSession:       "agent sessions are not part of the native execution slice",
		agentFeatureHandoff:       "agent handoff is not part of the native execution slice",
		agentFeatureScan:          "agent scanning is not part of the native execution slice",
		agentFeatureAudit:         "agent audit is not part of the native execution slice",
		agentFeaturePipeline:      "agent pipelines are not part of the native execution slice",
		agentFeatureMonitor:       "agent monitoring is not part of the native execution slice",
		agentFeatureHarvest:       "agent harvesting is not part of the native execution slice",
		agentFeatureBrainRecall:   "Brain recall is not connected to the native execution slice",
		agentFeatureBrainRemember: "Brain memory is not connected to the native execution slice",
		agentFeatureMessage:       "agent messaging is not part of the native execution slice",
		agentFeatureFleet:         "agent fleet controls are not part of the native execution slice",
		agentFeatureForge:         "Forge controls are not part of the native execution slice",
		agentFeatureRemote:        "remote agents are not part of the native execution slice",
		agentFeatureQA:            "standalone QA is represented by reviewed validation commands",
		agentFeatureReview:        "generic review is represented by Review Changes",
		agentFeaturePRCreate:      "pull-request creation is not part of the native execution slice",
		agentFeaturePRMerge:       "pull-request merging is not part of the native execution slice",
	}
	if reason := reasons[feature]; reason != "" {
		return reason
	}
	return "capability is not part of the native execution slice"
}

func (adapter *nativeAgentAdapter) Snapshot(ctx context.Context) core.Result {
	if adapter == nil || adapter.engine == nil {
		return core.Fail(core.E("tui.agentAdapter.Snapshot", "native agent engine is unavailable", nil))
	}
	result := adapter.engine.Snapshot(ctx, "")
	if !result.OK {
		return result
	}
	snapshot, ok := result.Value.(work.Snapshot)
	if !ok {
		return core.Fail(core.E("tui.agentAdapter.Snapshot", core.Sprintf("native agent snapshot has type %T", result.Value), nil))
	}
	return core.Ok(mapAgentSnapshot(snapshot))
}

func mapAgentSnapshot(snapshot work.Snapshot) agentSnapshot {
	mapped := agentSnapshot{
		Work:   make([]agentWorkSnapshot, 0, len(snapshot.Projects)+len(snapshot.Runs)),
		Events: make([]agentEventSnapshot, 0, len(snapshot.Events)+len(snapshot.Logs)+len(snapshot.Questions)),
	}
	workIndex := make(map[string]int)
	projects := make(map[string]work.Project, len(snapshot.Projects))
	for _, project := range snapshot.Projects {
		projects[project.ID] = project
		if project.ID == "" {
			continue
		}
		workIndex[project.ID] = len(mapped.Work)
		mapped.Work = append(mapped.Work, agentWorkSnapshot{
			ExternalID: project.ID,
			Title:      firstAgentText(project.RepositoryName, project.ID),
			Status:     "active",
			Repo:       project.SourcePath,
			Branch:     project.SourceBranch,
		})
	}
	runWork := make(map[string]string, len(snapshot.Runs))
	for _, run := range snapshot.Runs {
		runWork[run.ID] = run.WorkID
		index, exists := workIndex[run.WorkID]
		if !exists {
			index = len(mapped.Work)
			workIndex[run.WorkID] = index
			mapped.Work = append(mapped.Work, agentWorkSnapshot{ExternalID: run.WorkID, Title: run.WorkID})
		}
		item := &mapped.Work[index]
		project := projects[run.ProjectID]
		if item.Repo == "" {
			item.Repo = project.SourcePath
		}
		if item.Branch == "" {
			item.Branch = project.SourceBranch
		}
		if run.Branch != "" {
			item.Branch = run.Branch
		}
		item.Status = string(run.Status)
		item.Agent = run.Provider
		item.Runtime = run.Model
	}
	for _, event := range snapshot.Events {
		workID := event.WorkID
		if workID == "" {
			workID = runWork[event.RunID]
		}
		mapped.Events = append(mapped.Events, agentEventSnapshot{
			ExternalID: event.ID, WorkID: workID, Kind: event.Kind,
			Title: event.Title, Detail: event.Detail, CreatedAt: event.CreatedAt,
		})
	}
	for _, log := range snapshot.Logs {
		mapped.Events = append(mapped.Events, agentEventSnapshot{
			ExternalID: core.Sprintf("log:%s:%d", log.RunID, log.Sequence),
			WorkID:     runWork[log.RunID],
			Kind:       core.Concat("log.", core.Lower(core.Trim(log.Stream))),
			Title:      core.Concat(core.Upper(core.Trim(log.Stream)), " output"),
			Detail:     log.Text,
			CreatedAt:  log.CreatedAt,
		})
	}
	for _, question := range snapshot.Questions {
		workID := runWork[question.RunID]
		mapped.Events = append(mapped.Events, agentEventSnapshot{
			ExternalID: question.ID, WorkID: workID, Kind: "question",
			Title: "Agent question", Detail: question.Text, CreatedAt: question.CreatedAt,
		})
		if index, exists := workIndex[workID]; exists {
			mapped.Work[index].Question = question.Text
		}
	}
	sort.SliceStable(mapped.Events, func(left, right int) bool {
		if mapped.Events[left].CreatedAt.Equal(mapped.Events[right].CreatedAt) {
			return mapped.Events[left].ExternalID < mapped.Events[right].ExternalID
		}
		return mapped.Events[left].CreatedAt.Before(mapped.Events[right].CreatedAt)
	})
	return mapped
}

func firstAgentText(values ...string) string {
	if value := firstNonEmptyAgentText(values...); value != "" {
		return value
	}
	return "Agent work"
}

func firstNonEmptyAgentText(values ...string) string {
	for _, value := range values {
		if value = core.Trim(value); value != "" {
			return value
		}
	}
	return ""
}

func (adapter *nativeAgentAdapter) Review(ctx context.Context, request agentReviewRequest) core.Result {
	if adapter == nil || adapter.engine == nil {
		return core.Fail(core.E("tui.agentAdapter.Review", "native agent engine is unavailable", nil))
	}
	switch request.Feature {
	case agentFeatureDispatch:
		item := nativeWorkItem(request.Work, request.WorkID, request.Input)
		reviewResult := adapter.engine.ReviewProject(ctx, item)
		if !reviewResult.OK {
			return reviewResult
		}
		review, ok := reviewResult.Value.(orchestrator.ProjectReview)
		if !ok {
			return core.Fail(core.E("tui.agentAdapter.Review", core.Sprintf("project review has type %T", reviewResult.Value), nil))
		}
		warning := "Registering starts the private Git service and seeds an internal repository."
		if review.RequiresGitEnable {
			warning = "This directory is not a Git repository. Enabling Git requires a separate explicit confirmation before private registration."
		}
		return core.Ok(agentReview{
			Feature: agentFeatureDispatch, Title: "Review project registration",
			Body: core.Sprintf("Source: %s\nBranch: %s\nRevision: %s\nPrivate repository: %s\nIncluded files: %d",
				review.Source.Path, review.Source.Branch, review.Source.Revision, review.RepositoryName, len(review.Source.Included)),
			Warning: warning, ConfirmRequired: true, GitConfirmRequired: review.RequiresGitEnable,
			Payload: agentProjectRegistration{Review: review, Provider: request.Provider, Model: request.Model},
		})
	case agentFeatureChangesReview:
		reviewResult := adapter.engine.ReviewChanges(ctx, request.WorkID)
		if !reviewResult.OK {
			return reviewResult
		}
		review, ok := reviewResult.Value.(workspace.ChangeReview)
		if !ok {
			return core.Fail(core.E("tui.agentAdapter.Review", core.Sprintf("change review has type %T", reviewResult.Value), nil))
		}
		warning := "Accept applies the reviewed result to the source only after explicit final confirmation."
		if len(review.Conflicts) > 0 {
			warning = "Integration conflicts must be resolved by a later reviewed attempt; the source remains unchanged."
		} else if len(review.Validation) == 0 {
			warning = "No validation command is configured; acceptance requires explicit acknowledgement."
		}
		return core.Ok(agentReview{
			Feature: agentFeatureChangesReview, Title: "Review agent changes",
			Body: core.Sprintf("Commits:\n%s\n\nDiff:\n%s\n\nValidation checks: %d\nConflicts: %d",
				review.CommitLog, review.Diff, len(review.Validation), len(review.Conflicts)),
			Warning: warning, ConfirmRequired: true, Payload: review,
		})
	default:
		return core.Fail(core.E("tui.agentAdapter.Review", core.Concat("agent feature does not support review: ", string(request.Feature)), nil))
	}
}

func (adapter *nativeAgentAdapter) Run(ctx context.Context, request agentRequest) core.Result {
	if adapter == nil || adapter.engine == nil {
		return core.Fail(core.E("tui.agentAdapter.Run", "native agent engine is unavailable", nil))
	}
	var result core.Result
	switch request.Feature {
	case agentFeatureDispatch:
		return adapter.runDispatch(ctx, request)
	case agentFeatureCancel:
		result = adapter.engine.Cancel(ctx, request.WorkID)
	case agentFeatureAnswer:
		result = adapter.engine.Answer(ctx, request.WorkID, request.Input)
	case agentFeatureRetry:
		result = adapter.engine.Retry(ctx, nativeWorkItem(request.Work, request.Work.ID, ""), request.WorkID)
	case agentFeatureResume:
		result = adapter.engine.Resume(ctx, work.ResumeRequest{
			Work: nativeWorkItem(request.Work, request.Work.ID, ""), ParentRunID: request.WorkID,
			AnswerID: request.Input, Provider: request.Provider, Model: request.Model,
		})
	case agentFeatureQueueStart:
		result = adapter.engine.StartQueue(ctx)
	case agentFeatureQueueStop:
		result = adapter.engine.StopQueue(ctx)
	case agentFeatureAccept:
		review, ok := request.Review.Payload.(workspace.ChangeReview)
		if !ok || request.Review.Feature != agentFeatureChangesReview {
			return core.Fail(core.E("tui.agentAdapter.Run", "accept requires a reviewed change receipt", nil))
		}
		result = adapter.engine.Accept(ctx, workspace.AcceptRequest{Review: review, Confirmed: request.Confirmed})
	case agentFeatureReject:
		result = adapter.engine.Reject(ctx, request.WorkID)
	default:
		return core.Fail(core.E("tui.agentAdapter.Run", core.Concat("unsupported native agent action: ", string(request.Feature)), nil))
	}
	return privateAgentReceipt(request, result)
}

func (adapter *nativeAgentAdapter) runDispatch(ctx context.Context, request agentRequest) core.Result {
	switch payload := request.Review.Payload.(type) {
	case agentProjectRegistration:
		if !request.Confirmed {
			return core.Fail(core.E("tui.agentAdapter.Run", "project registration requires explicit confirmation", nil))
		}
		if payload.Review.RequiresGitEnable != request.EnableGit {
			if payload.Review.RequiresGitEnable {
				return core.Fail(core.E("tui.agentAdapter.Run", "project registration requires separate Git enable confirmation", nil))
			}
			return core.Fail(core.E("tui.agentAdapter.Run", "Git enable confirmation is invalid for an existing repository", nil))
		}
		registered := adapter.engine.RegisterProject(ctx, payload.Review, true)
		if !registered.OK {
			return registered
		}
		project, ok := registered.Value.(work.Project)
		if !ok {
			return core.Fail(core.E("tui.agentAdapter.Run", core.Sprintf("registered project has type %T", registered.Value), nil))
		}
		dispatchResult := adapter.engine.ReviewDispatch(ctx, work.DispatchRequest{
			Work: payload.Review.Work, Provider: payload.Provider, Model: payload.Model,
			ConfirmedSourceRevision: project.SourceRevision,
		})
		if !dispatchResult.OK {
			return dispatchResult
		}
		dispatch, ok := dispatchResult.Value.(orchestrator.DispatchReview)
		if !ok {
			return core.Fail(core.E("tui.agentAdapter.Run", core.Sprintf("dispatch review has type %T", dispatchResult.Value), nil))
		}
		return core.Ok(mapDispatchReview(dispatch))
	case orchestrator.DispatchReview:
		if !request.Confirmed {
			return core.Fail(core.E("tui.agentAdapter.Run", "dispatch requires explicit launch confirmation", nil))
		}
		return privateAgentReceipt(request, adapter.engine.Dispatch(ctx, payload))
	default:
		return core.Fail(core.E("tui.agentAdapter.Run", "dispatch requires a current reviewed receipt", nil))
	}
}

func mapDispatchReview(review orchestrator.DispatchReview) agentReview {
	queueStatus := "ready for admission"
	if reason := core.Trim(review.Queue.Reason); reason != "" {
		queueStatus = reason
	}
	return agentReview{
		Feature: agentFeatureDispatch, Title: "Review native agent launch",
		Body: core.Sprintf("Provider: %s\nModel: %s\nCommand: %s\nSource: %s\nBranch: %s\nRevision: %s\nPrivate repository: %s\nWorktree: %s\nQueue: %s",
			review.Request.Provider, review.Request.Model, review.Command.Receipt, review.Source.Path,
			review.Source.Branch, review.Source.Revision, review.Project.RepositoryName, review.WorktreePath, queueStatus),
		Warning: review.Warning, ConfirmRequired: true, Payload: review,
	}
}

func nativeWorkItem(request agentWorkRequest, fallbackID, fallbackInput string) work.Item {
	id := firstNonEmptyAgentText(request.ID, request.ExternalID, fallbackID)
	title := firstNonEmptyAgentText(request.Title, id)
	task := core.Trim(request.Task)
	repository := firstNonEmptyAgentText(request.Repository, fallbackInput)
	return work.Item{ID: id, ExternalID: request.ExternalID, Title: title, Task: task, Repository: repository}
}

func privateAgentReceipt(request agentRequest, result core.Result) core.Result {
	if !result.OK {
		return result
	}
	receipt := agentActionReceipt{Feature: request.Feature, WorkID: request.WorkID}
	switch value := result.Value.(type) {
	case work.Run:
		receipt.WorkID = value.WorkID
		receipt.RunID = value.ID
		receipt.Status = string(value.Status)
	case work.Answer:
		receipt.RunID = value.ResumeRunID
		receipt.Status = "answered"
		receipt.Detail = value.ID
	case work.QueueState:
		receipt.Status = string(value.Status)
		receipt.Detail = value.Reason
	case work.Acceptance:
		receipt.WorkID = value.WorkID
		receipt.RunID = value.RunID
		receipt.Status = value.Status
	case nil:
		receipt.Status = "completed"
	default:
		receipt.Status = "completed"
		receipt.Detail = core.Sprintf("%v", value)
	}
	return core.Ok(receipt)
}

func (adapter *nativeAgentAdapter) Close() core.Result {
	if adapter == nil {
		return core.Ok(nil)
	}
	adapter.closeOnce.Do(func() {
		if adapter.engine != nil {
			adapter.closeResult = adapter.engine.Close()
		}
	})
	return adapter.closeResult
}
