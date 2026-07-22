// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"time"

	core "dappco.re/go"
)

const defaultAgentUnavailableReason = "agent capability not installed; the go/agent provider has not been connected"

type agentFeature string

const (
	agentFeatureDispatch        agentFeature = "dispatch"
	agentFeatureCancel          agentFeature = "cancel"
	agentFeatureAnswer          agentFeature = "answer"
	agentFeatureRetry           agentFeature = "retry"
	agentFeatureResume          agentFeature = "resume"
	agentFeatureRecoveryAbandon agentFeature = "recovery.abandon"
	agentFeatureQueueStart      agentFeature = "queue.start"
	agentFeatureQueueStop       agentFeature = "queue.stop"
	agentFeatureSetup           agentFeature = "setup"
	agentFeatureProvider        agentFeature = "provider"
	agentFeatureTemplate        agentFeature = "template"
	agentFeaturePlan            agentFeature = "plan"
	agentFeatureSession         agentFeature = "session"
	agentFeatureHandoff         agentFeature = "handoff"
	agentFeatureScan            agentFeature = "scan"
	agentFeatureAudit           agentFeature = "audit"
	agentFeaturePipeline        agentFeature = "pipeline"
	agentFeatureMonitor         agentFeature = "monitor"
	agentFeatureHarvest         agentFeature = "harvest"
	agentFeatureBrainRecall     agentFeature = "brain.recall"
	agentFeatureBrainRemember   agentFeature = "brain.remember"
	agentFeatureMessage         agentFeature = "message"
	agentFeatureFleet           agentFeature = "fleet"
	agentFeatureForge           agentFeature = "forge"
	agentFeatureRemote          agentFeature = "remote"
	agentFeatureQA              agentFeature = "qa"
	agentFeatureReview          agentFeature = "review"
	agentFeatureChangesReview   agentFeature = "changes.review"
	agentFeatureAccept          agentFeature = "accept"
	agentFeatureReject          agentFeature = "reject"
	agentFeaturePRCreate        agentFeature = "pr.create"
	agentFeaturePRMerge         agentFeature = "pr.merge"
)

type agentCapability struct {
	Feature   agentFeature
	Available bool
	Reason    string
}

type agentWorkSnapshot struct {
	ExternalID    string
	NativeRunID   string
	QuestionID    string
	AnswerID      string
	ResumeRunID   string
	QueueStatus   string
	QueueReason   string
	ReviewID      string
	ReviewStatus  string
	Review        agentReview
	Title         string
	Status        string
	Agent         string
	Repo          string
	Branch        string
	Runtime       string
	Question      string
	PRURL         string
	Recovery      agentPendingRecovery
	RecoveryCount int
}

type agentRecoveryReceipt struct {
	Kind           string
	ProjectID      string
	WorkID         string
	RunID          string
	RunNumber      int
	WorkspaceRunID string
	ReviewID       string
	Branch         string
	Worktree       string
}

type agentPendingRecovery struct {
	EventID string
	Receipt agentRecoveryReceipt
}

type agentRecoveryOutcome struct {
	RecoveryEventID string
	Receipt         agentRecoveryReceipt
	Error           string
}

type agentEventSnapshot struct {
	ExternalID string
	WorkID     string
	RunID      string
	Sequence   int64
	Stream     string
	Kind       string
	Title      string
	Detail     string
	CreatedAt  time.Time
}

type agentSnapshot struct {
	Work        []agentWorkSnapshot
	Events      []agentEventSnapshot
	QueueStatus string
	QueueReason string
}

type agentRequest struct {
	Feature    agentFeature
	WorkID     string
	RunID      string
	QuestionID string
	Provider   string
	Model      string
	Input      string
	Work       agentWorkRequest
	Review     agentReview
	Recovery   agentPendingRecovery
	Confirmed  bool
	EnableGit  bool
}

type agentWorkRequest struct {
	ID         string
	ExternalID string
	Title      string
	Task       string
	Repository string
}

type agentReviewRequest struct {
	Feature  agentFeature
	WorkID   string
	Provider string
	Model    string
	Input    string
	Work     agentWorkRequest
	Recovery agentPendingRecovery
}

type agentReview struct {
	Feature              agentFeature
	Title                string
	Body                 string
	Warning              string
	ConfirmRequired      bool
	GitConfirmRequired   bool
	NeedsAcknowledgement bool
	AcceptanceAllowed    bool
	Payload              any
}

type agentActionReceipt struct {
	Feature agentFeature
	WorkID  string
	RunID   string
	Status  string
	Detail  string
}

type agentProvider interface {
	Capabilities() []agentCapability
	Snapshot(ctx context.Context) core.Result
	Review(ctx context.Context, request agentReviewRequest) core.Result
	Run(ctx context.Context, request agentRequest) core.Result
	Close() core.Result
}

type agentFeatureGroup struct {
	Title    string
	Features []agentFeature
}

var agentFeatureGroups = []agentFeatureGroup{
	{Title: "EXECUTION", Features: []agentFeature{agentFeatureDispatch, agentFeatureCancel, agentFeatureAnswer, agentFeatureRetry, agentFeatureResume, agentFeatureRecoveryAbandon}},
	{Title: "QUEUE + SETUP", Features: []agentFeature{agentFeatureQueueStart, agentFeatureQueueStop, agentFeatureSetup, agentFeatureProvider, agentFeatureTemplate}},
	{Title: "PLANS + SESSIONS", Features: []agentFeature{agentFeaturePlan, agentFeatureSession, agentFeatureHandoff}},
	{Title: "SCAN + MONITOR", Features: []agentFeature{agentFeatureScan, agentFeatureAudit, agentFeaturePipeline, agentFeatureMonitor, agentFeatureHarvest}},
	{Title: "BRAIN + MESSAGES", Features: []agentFeature{agentFeatureBrainRecall, agentFeatureBrainRemember, agentFeatureMessage}},
	{Title: "FLEET + FORGE", Features: []agentFeature{agentFeatureFleet, agentFeatureForge, agentFeatureRemote}},
	{Title: "QA + REVIEW + PR", Features: []agentFeature{agentFeatureQA, agentFeatureReview, agentFeatureChangesReview, agentFeatureAccept, agentFeatureReject, agentFeaturePRCreate, agentFeaturePRMerge}},
}

func agentFeatureCatalog(reason string) []agentCapability {
	catalog := make([]agentCapability, 0, 32)
	for _, group := range agentFeatureGroups {
		for _, feature := range group.Features {
			catalog = append(catalog, agentCapability{Feature: feature, Reason: reason})
		}
	}
	return catalog
}

func agentFeatureTitle(feature agentFeature) string {
	switch feature {
	case agentFeatureQueueStart:
		return "Start queue"
	case agentFeatureQueueStop:
		return "Stop queue"
	case agentFeatureBrainRecall:
		return "Recall from Brain"
	case agentFeatureBrainRemember:
		return "Remember in Brain"
	case agentFeaturePRCreate:
		return "Create pull request"
	case agentFeaturePRMerge:
		return "Merge pull request"
	case agentFeatureChangesReview:
		return "Review changes"
	case agentFeatureRecoveryAbandon:
		return "Abandon retained recovery"
	case agentFeatureQA:
		return "Run QA"
	default:
		name := core.Replace(string(feature), ".", " ")
		if name == "" {
			return "Unknown agent action"
		}
		return core.Upper(name[:1]) + name[1:]
	}
}

type unavailableAgentProvider struct {
	reason string
}

func newUnavailableAgentProvider(reason string) agentProvider {
	reason = core.Trim(reason)
	if reason == "" {
		reason = defaultAgentUnavailableReason
	}
	return &unavailableAgentProvider{reason: reason}
}

func (provider *unavailableAgentProvider) Capabilities() []agentCapability {
	reason := defaultAgentUnavailableReason
	if provider != nil && provider.reason != "" {
		reason = provider.reason
	}
	return agentFeatureCatalog(reason)
}

func (*unavailableAgentProvider) Snapshot(context.Context) core.Result {
	return core.Ok(agentSnapshot{Work: []agentWorkSnapshot{}, Events: []agentEventSnapshot{}})
}

func (provider *unavailableAgentProvider) Review(_ context.Context, request agentReviewRequest) core.Result {
	reason := defaultAgentUnavailableReason
	if provider != nil && provider.reason != "" {
		reason = provider.reason
	}
	return core.Fail(core.E(
		"tui.agentProvider.Review",
		core.Concat("agent review ", string(request.Feature), " is unavailable: ", reason),
		nil,
	))
}

func (provider *unavailableAgentProvider) Run(_ context.Context, request agentRequest) core.Result {
	reason := defaultAgentUnavailableReason
	if provider != nil && provider.reason != "" {
		reason = provider.reason
	}
	return core.Fail(core.E(
		"tui.agentProvider.Run",
		core.Concat("agent action ", string(request.Feature), " is unavailable: ", reason),
		nil,
	))
}

func (*unavailableAgentProvider) Close() core.Result { return core.Ok(nil) }
