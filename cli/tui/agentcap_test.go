// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
)

func TestAgentFeatureCatalog_Good(t *testing.T) {
	reason := "agent runtime is not installed"
	want := []agentFeature{
		agentFeatureDispatch, agentFeatureCancel, agentFeatureAnswer, agentFeatureRetry,
		agentFeatureResume, agentFeatureRecoveryAbandon, agentFeatureQueueStart, agentFeatureQueueStop, agentFeatureSetup,
		agentFeatureProvider, agentFeatureTemplate, agentFeaturePlan, agentFeatureSession,
		agentFeatureHandoff, agentFeatureScan, agentFeatureAudit, agentFeaturePipeline,
		agentFeatureMonitor, agentFeatureHarvest, agentFeatureBrainRecall, agentFeatureBrainRemember,
		agentFeatureMessage, agentFeatureFleet, agentFeatureForge, agentFeatureRemote,
		agentFeatureQA, agentFeatureReview, agentFeaturePRCreate, agentFeaturePRMerge,
		agentFeatureChangesReview, agentFeatureAccept, agentFeatureReject,
	}
	catalog := agentFeatureCatalog(reason)
	if len(catalog) != len(want) {
		t.Fatalf("agentFeatureCatalog length = %d, want %d", len(catalog), len(want))
	}
	seen := make(map[agentFeature]int, len(catalog))
	for _, capability := range catalog {
		seen[capability.Feature]++
		if capability.Available || capability.Reason != reason {
			t.Fatalf("catalog capability = %#v", capability)
		}
	}
	for _, feature := range want {
		if seen[feature] != 1 {
			t.Fatalf("feature %q appears %d times", feature, seen[feature])
		}
	}
}

func TestUnavailableAgentProvider_Bad(t *testing.T) {
	provider := newUnavailableAgentProvider("port go/agent to enable execution")
	snapshot := provider.Snapshot(context.Background())
	if !snapshot.OK {
		t.Fatalf("Snapshot failed: %v", snapshot.Value)
	}
	value, ok := snapshot.Value.(agentSnapshot)
	if !ok || value.Work == nil || value.Events == nil || len(value.Work) != 0 || len(value.Events) != 0 {
		t.Fatalf("Snapshot = %#v (%T), want empty typed snapshot", snapshot.Value, snapshot.Value)
	}
	run := provider.Run(context.Background(), agentRequest{Feature: agentFeatureDispatch, WorkID: "work-1"})
	if run.OK || !strings.Contains(run.Error(), string(agentFeatureDispatch)) || !strings.Contains(run.Error(), "port go/agent") {
		t.Fatalf("Run = %#v", run)
	}
	review := provider.Review(context.Background(), agentReviewRequest{Feature: agentFeatureDispatch, WorkID: "work-1"})
	if review.OK || !strings.Contains(review.Error(), string(agentFeatureDispatch)) || !strings.Contains(review.Error(), "port go/agent") {
		t.Fatalf("Review = %#v", review)
	}
	if result := provider.Close(); !result.OK {
		t.Fatalf("first Close: %v", result.Value)
	}
	if result := provider.Close(); !result.OK {
		t.Fatalf("second Close: %v", result.Value)
	}
}

func TestAgentProviderBoundary_Ugly(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	createdAt := time.Date(2026, time.July, 17, 16, 0, 0, 0, time.UTC)
	provider := &fixtureAgentProvider{snapshot: agentSnapshot{
		Work: []agentWorkSnapshot{
			{ExternalID: "agent-2", Title: "Second", Status: "waiting", Agent: "beta", Repo: "repo-b", Question: "Which target?"},
			{ExternalID: "agent-1", Title: "First", Status: "active", Agent: "alpha", Repo: "repo-a", Branch: "feat/first"},
		},
		Events: []agentEventSnapshot{
			{ExternalID: "event-3", WorkID: "agent-2", Kind: "question", Title: "Needs input", CreatedAt: createdAt.Add(3 * time.Minute)},
			{ExternalID: "event-1", WorkID: "agent-1", Kind: "started", Title: "Started", CreatedAt: createdAt.Add(time.Minute)},
			{ExternalID: "event-2", WorkID: "agent-1", Kind: "log", Title: "Checked tree", CreatedAt: createdAt.Add(2 * time.Minute)},
		},
	}}
	opened := newWorkPanel(repository, provider, sequenceIDs("work-a", "work-b"), func() time.Time { return createdAt })
	if !opened.OK {
		t.Fatalf("newWorkPanel: %v", opened.Value)
	}
	panel := opened.Value.(*workPanel)
	for pass := 0; pass < 2; pass++ {
		if result := panel.Refresh(context.Background()); !result.OK {
			t.Fatalf("Refresh pass %d: %v", pass, result.Value)
		}
	}
	if items := panel.Items(); len(items) != 0 {
		t.Fatalf("provider snapshot created Work rows: %#v", items)
	}
	if provider.snapshots != 2 {
		t.Fatalf("Snapshot calls = %d, want 2", provider.snapshots)
	}
}

type fixtureAgentProvider struct {
	snapshot   agentSnapshot
	caps       []agentCapability
	snapshots  int
	runs       int
	closeCalls int
}

func (provider *fixtureAgentProvider) Capabilities() []agentCapability {
	if provider.caps != nil {
		return append([]agentCapability(nil), provider.caps...)
	}
	capabilities := agentFeatureCatalog("")
	for index := range capabilities {
		capabilities[index].Available = true
	}
	return capabilities
}

func (provider *fixtureAgentProvider) Snapshot(context.Context) core.Result {
	provider.snapshots++
	return core.Ok(provider.snapshot)
}

func (*fixtureAgentProvider) Review(context.Context, agentReviewRequest) core.Result {
	return core.Ok(agentReview{})
}

func (provider *fixtureAgentProvider) Run(context.Context, agentRequest) core.Result {
	provider.runs++
	return core.Ok(nil)
}

func (provider *fixtureAgentProvider) Close() core.Result {
	provider.closeCalls++
	return core.Ok(nil)
}
