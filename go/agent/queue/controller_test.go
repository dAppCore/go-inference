// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

func controllerTestPolicy() Policy {
	return Policy{
		Version: 1,
		Dispatch: DispatchConfig{
			DefaultAgent:      "codex",
			GlobalConcurrency: 4,
			TimeoutMinutes:    60,
			Validation:        []Command{{Command: "go", Args: []string{"test", "./..."}}},
		},
		Concurrency: map[string]ConcurrencyLimit{
			"codex": {Total: 2, Models: map[string]int{"gpt-5.6": 1}},
		},
		Rates: map[string]RateConfig{
			"codex": {ResetUTC: "00:00", DailyLimit: 2},
		},
		Providers: map[string]NativeConfig{
			"codex": {Executable: "codex", CredentialEnv: []string{"OPENAI_API_KEY"}, Flags: []string{"--search"}},
		},
	}
}

func mustQueueController(t *testing.T, policy Policy, state work.QueueState, providers []work.ProviderState) *Controller {
	t.Helper()
	result := NewController(policy, state, providers)
	if !result.OK {
		t.Fatalf("NewController failed: %s", result.Error())
	}
	controller, ok := result.Value.(*Controller)
	if !ok {
		t.Fatalf("NewController returned %T", result.Value)
	}
	return controller
}

func TestController_NewController_Good(t *testing.T) {
	policy := controllerTestPolicy()
	initial := work.QueueState{ID: "default", Status: work.QueueFrozen, Reason: "startup"}
	providers := []work.ProviderState{{Provider: "codex", BackoffReason: "quota"}}

	result := NewController(policy, initial, providers)
	core.AssertTrue(t, result.OK, result.Error())
	controller := result.Value.(*Controller)

	policy.Dispatch.Validation[0].Args[0] = "mutated"
	policy.Concurrency["codex"].Models["gpt-5.6"] = 9
	configured := policy.Providers["codex"]
	configured.CredentialEnv[0] = "MUTATED"
	policy.Providers["codex"] = configured
	providers[0].BackoffReason = "mutated"

	core.AssertEqual(t, "test", controller.policy.Dispatch.Validation[0].Args[0])
	core.AssertEqual(t, 1, controller.policy.Concurrency["codex"].Models["gpt-5.6"])
	core.AssertEqual(t, "OPENAI_API_KEY", controller.policy.Providers["codex"].CredentialEnv[0])
	core.AssertEqual(t, "quota", controller.providers["codex"].BackoffReason)
}

func TestController_NewController_Bad(t *testing.T) {
	tests := []struct {
		name      string
		policy    Policy
		state     work.QueueState
		providers []work.ProviderState
	}{
		{name: "negative global concurrency", policy: func() Policy { policy := controllerTestPolicy(); policy.Dispatch.GlobalConcurrency = -1; return policy }(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}},
		{name: "empty concurrency provider", policy: func() Policy {
			policy := controllerTestPolicy()
			policy.Concurrency[""] = ConcurrencyLimit{Total: 1}
			return policy
		}(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}},
		{name: "empty model", policy: func() Policy {
			policy := controllerTestPolicy()
			policy.Concurrency["codex"] = ConcurrencyLimit{Total: 2, Models: map[string]int{"": 1}}
			return policy
		}(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}},
		{name: "empty rate provider", policy: func() Policy {
			policy := controllerTestPolicy()
			policy.Rates[""] = RateConfig{ResetUTC: "00:00"}
			return policy
		}(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}},
		{name: "empty native provider", policy: func() Policy { policy := controllerTestPolicy(); policy.Providers[""] = NativeConfig{}; return policy }(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}},
		{name: "wrong queue ID", policy: controllerTestPolicy(), state: work.QueueState{ID: "another", Status: work.QueueFrozen}},
		{name: "unknown queue status", policy: controllerTestPolicy(), state: work.QueueState{ID: "default", Status: work.QueueStatus("lost")}},
		{name: "empty provider state", policy: controllerTestPolicy(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}, providers: []work.ProviderState{{}}},
		{name: "duplicate provider state", policy: controllerTestPolicy(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}, providers: []work.ProviderState{{Provider: "codex"}, {Provider: "codex"}}},
		{name: "negative admissions", policy: controllerTestPolicy(), state: work.QueueState{ID: "default", Status: work.QueueFrozen}, providers: []work.ProviderState{{Provider: "codex", WindowAdmissions: -1}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := NewController(test.policy, test.state, test.providers)
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestController_NewController_Ugly(t *testing.T) {
	policy := controllerTestPolicy()
	result := NewController(policy, work.QueueState{}, nil)
	core.AssertTrue(t, result.OK, result.Error())
	controller := result.Value.(*Controller)
	core.AssertEqual(t, "default", controller.state.ID)
	core.AssertEqual(t, work.QueueFrozen, controller.state.Status)
	core.AssertTrue(t, controller.state.Reason != "")
}

func TestController_Controller_Start_Good(t *testing.T) {
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueFrozen}, nil)
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

	result := controller.Start(at)
	core.AssertTrue(t, result.OK, result.Error())
	state := result.Value.(work.QueueState)
	core.AssertEqual(t, work.QueueAccepting, state.Status)
	core.AssertEqual(t, "", state.Reason)
	core.AssertEqual(t, at, state.UpdatedAt)
}

func TestController_Controller_Start_Bad(t *testing.T) {
	var controller *Controller
	result := controller.Start(time.Now())
	core.AssertFalse(t, result.OK)
}

func TestController_Controller_Start_Ugly(t *testing.T) {
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting, UpdatedAt: at}, nil)

	result := controller.Start(at.Add(time.Hour))
	core.AssertTrue(t, result.OK, result.Error())
	state := result.Value.(work.QueueState)
	core.AssertEqual(t, at, state.UpdatedAt)
	core.AssertFalse(t, controller.Start(time.Time{}).OK)
}

func TestController_Controller_Stop_Good(t *testing.T) {
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

	result := controller.Stop(2, at)
	core.AssertTrue(t, result.OK, result.Error())
	state := result.Value.(work.QueueState)
	core.AssertEqual(t, work.QueueDraining, state.Status)
	core.AssertContains(t, state.Reason, "2")

	result = controller.Stop(0, at.Add(time.Minute))
	core.AssertTrue(t, result.OK, result.Error())
	state = result.Value.(work.QueueState)
	core.AssertEqual(t, work.QueueFrozen, state.Status)
}

func TestController_Controller_Stop_Bad(t *testing.T) {
	var controller *Controller
	result := controller.Stop(0, time.Now())
	core.AssertFalse(t, result.OK)
}

func TestController_Controller_Stop_Ugly(t *testing.T) {
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	core.AssertFalse(t, controller.Stop(-1, time.Now()).OK)
	core.AssertFalse(t, controller.Stop(0, time.Time{}).OK)
}

func TestController_Controller_Decide_Good(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	candidate := Candidate{RunID: "run-1", Provider: "codex", Model: "gpt-5.6", QueuedAt: now.Add(-time.Minute)}
	runtime := Runtime{
		Queued: []work.Run{{ID: "run-1", Status: work.RunQueued, QueuedAt: candidate.QueuedAt}},
		Running: []work.Run{
			{ID: "done", Provider: "codex", Model: "gpt-5.6", Status: work.RunCompleted},
			{ID: "claude", Provider: "claude", Model: "sonnet", Status: work.RunRunning},
			{ID: "other-model", Provider: "codex", Model: "gpt-5.5", Status: work.RunRunning},
		},
		Now: now,
	}

	result := controller.Decide(candidate, runtime)
	core.AssertTrue(t, result.OK, result.Error())
	decision := result.Value.(Decision)
	core.AssertTrue(t, decision.Allowed)
	core.AssertEqual(t, "", decision.Reason)
	core.AssertTrue(t, decision.NotBefore.IsZero())
}

func TestController_Controller_Decide_Bad(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	candidate := Candidate{RunID: "run-2", Provider: "codex", Model: "gpt-5.6", QueuedAt: now}

	tests := []struct {
		name      string
		policy    Policy
		queue     work.QueueState
		providers []work.ProviderState
		runtime   Runtime
		reason    string
		notBefore time.Time
	}{
		{
			name: "frozen queue", policy: controllerTestPolicy(),
			queue:   work.QueueState{ID: "default", Status: work.QueueFrozen, Reason: "startup"},
			runtime: Runtime{Now: now}, reason: "startup",
		},
		{
			name: "global concurrency", policy: func() Policy { policy := controllerTestPolicy(); policy.Dispatch.GlobalConcurrency = 1; return policy }(),
			queue:   work.QueueState{ID: "default", Status: work.QueueAccepting},
			runtime: Runtime{Now: now, Running: []work.Run{{ID: "other", Provider: "claude", Status: work.RunRunning}}}, reason: "global",
		},
		{
			name: "provider concurrency", policy: func() Policy {
				policy := controllerTestPolicy()
				policy.Concurrency["codex"] = ConcurrencyLimit{Total: 1}
				return policy
			}(),
			queue:   work.QueueState{ID: "default", Status: work.QueueAccepting},
			runtime: Runtime{Now: now, Running: []work.Run{{ID: "other", Provider: "codex", Status: work.RunPreparing}}}, reason: "provider",
		},
		{
			name: "model concurrency", policy: controllerTestPolicy(),
			queue:   work.QueueState{ID: "default", Status: work.QueueAccepting},
			runtime: Runtime{Now: now, Running: []work.Run{{ID: "other", Provider: "codex", Model: "gpt-5.6", Status: work.RunRunning}}}, reason: "model",
		},
		{
			name: "provider backoff", policy: controllerTestPolicy(),
			queue:     work.QueueState{ID: "default", Status: work.QueueAccepting},
			providers: []work.ProviderState{{Provider: "codex", BackoffReason: "quota", BackoffUntil: now.Add(time.Hour)}},
			runtime:   Runtime{Now: now}, reason: "quota", notBefore: now.Add(time.Hour),
		},
		{
			name: "provider backoff without stored reason", policy: controllerTestPolicy(),
			queue:     work.QueueState{ID: "default", Status: work.QueueAccepting},
			providers: []work.ProviderState{{Provider: "codex", BackoffUntil: now.Add(time.Hour)}},
			runtime:   Runtime{Now: now}, reason: "backoff", notBefore: now.Add(time.Hour),
		},
		{
			name: "minimum delay", policy: func() Policy {
				policy := controllerTestPolicy()
				policy.Rates["codex"] = RateConfig{ResetUTC: "00:00", MinDelay: 60}
				return policy
			}(),
			queue:     work.QueueState{ID: "default", Status: work.QueueAccepting},
			providers: []work.ProviderState{{Provider: "codex", LastStartedAt: now.Add(-30 * time.Second)}},
			runtime:   Runtime{Now: now}, reason: "pacing", notBefore: now.Add(30 * time.Second),
		},
		{
			name: "sustained delay", policy: func() Policy {
				policy := controllerTestPolicy()
				policy.Rates["codex"] = RateConfig{ResetUTC: "00:00", SustainedDelay: 300}
				return policy
			}(),
			queue:     work.QueueState{ID: "default", Status: work.QueueAccepting},
			providers: []work.ProviderState{{Provider: "codex", LastStartedAt: now.Add(-100 * time.Second)}},
			runtime:   Runtime{Now: now}, reason: "pacing", notBefore: now.Add(200 * time.Second),
		},
		{
			name: "burst delay", policy: func() Policy {
				policy := controllerTestPolicy()
				policy.Rates["codex"] = RateConfig{ResetUTC: "13:00", SustainedDelay: 300, BurstWindow: 2, BurstDelay: 30}
				return policy
			}(),
			queue:     work.QueueState{ID: "default", Status: work.QueueAccepting},
			providers: []work.ProviderState{{Provider: "codex", LastStartedAt: now.Add(-10 * time.Second)}},
			runtime:   Runtime{Now: now}, reason: "pacing", notBefore: now.Add(20 * time.Second),
		},
		{
			name: "daily quota", policy: controllerTestPolicy(),
			queue:     work.QueueState{ID: "default", Status: work.QueueAccepting},
			providers: []work.ProviderState{{Provider: "codex", WindowStartedAt: time.Date(2026, 7, 18, 0, 0, 0, 0, time.UTC), WindowAdmissions: 2}},
			runtime:   Runtime{Now: now}, reason: "daily", notBefore: time.Date(2026, 7, 19, 0, 0, 0, 0, time.UTC),
		},
		{
			name: "FIFO", policy: controllerTestPolicy(),
			queue:   work.QueueState{ID: "default", Status: work.QueueAccepting},
			runtime: Runtime{Now: now, Queued: []work.Run{{ID: "run-1", Status: work.RunQueued, QueuedAt: now.Add(-time.Minute)}}}, reason: "run-1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			controller := mustQueueController(t, test.policy, test.queue, test.providers)
			result := controller.Decide(candidate, test.runtime)
			core.AssertTrue(t, result.OK, result.Error())
			decision := result.Value.(Decision)
			core.AssertFalse(t, decision.Allowed)
			core.AssertContains(t, decision.Reason, test.reason)
			core.AssertEqual(t, test.notBefore, decision.NotBefore)
		})
	}
}

func TestController_Controller_Decide_Ugly(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	validCandidate := Candidate{RunID: "run-1", Provider: "codex", QueuedAt: now}
	validRuntime := Runtime{Now: now}
	var controller *Controller
	core.AssertFalse(t, controller.Decide(validCandidate, validRuntime).OK)

	controller = mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	core.AssertFalse(t, controller.Decide(Candidate{}, validRuntime).OK)
	core.AssertFalse(t, controller.Decide(validCandidate, Runtime{}).OK)
}

func TestController_Controller_RecordStart_Good(t *testing.T) {
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)

	result := controller.RecordStart("codex", "run-1", at)
	core.AssertTrue(t, result.OK, result.Error())
	state := result.Value.(work.ProviderState)
	core.AssertEqual(t, "run-1", state.LastRunID)
	core.AssertEqual(t, at, state.LastStartedAt)
	core.AssertEqual(t, 1, state.WindowAdmissions)
	core.AssertEqual(t, time.Date(2026, 7, 18, 0, 0, 0, 0, time.UTC), state.WindowStartedAt)

	result = controller.RecordStart("codex", "run-2", at.Add(time.Hour))
	core.AssertTrue(t, result.OK, result.Error())
	state = result.Value.(work.ProviderState)
	core.AssertEqual(t, 2, state.WindowAdmissions)
}

func TestController_Controller_RecordStart_Bad(t *testing.T) {
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	core.AssertFalse(t, controller.RecordStart("", "run-1", time.Now()).OK)
	core.AssertFalse(t, controller.RecordStart("codex", "", time.Now()).OK)
}

func TestController_Controller_RecordStart_Ugly(t *testing.T) {
	at := time.Date(2026, 7, 19, 1, 0, 0, 0, time.UTC)
	providers := []work.ProviderState{{
		Provider:         "codex",
		WindowStartedAt:  time.Date(2026, 7, 18, 0, 0, 0, 0, time.UTC),
		WindowAdmissions: 8,
	}}
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, providers)
	result := controller.RecordStart("codex", "run-new-day", at)
	core.AssertTrue(t, result.OK, result.Error())
	state := result.Value.(work.ProviderState)
	core.AssertEqual(t, 1, state.WindowAdmissions)
	core.AssertEqual(t, time.Date(2026, 7, 19, 0, 0, 0, 0, time.UTC), state.WindowStartedAt)
	core.AssertFalse(t, controller.RecordStart("codex", "run-zero", time.Time{}).OK)

	beforeResetPolicy := controllerTestPolicy()
	beforeResetPolicy.Rates["codex"] = RateConfig{ResetUTC: "18:00"}
	beforeResetProviders := []work.ProviderState{{Provider: "codex", BackoffReason: "old", BackoffUntil: at.Add(-time.Minute)}}
	beforeReset := mustQueueController(t, beforeResetPolicy, work.QueueState{ID: "default", Status: work.QueueAccepting}, beforeResetProviders)
	result = beforeReset.RecordStart("codex", "run-before-reset", time.Date(2026, 7, 19, 12, 0, 0, 0, time.UTC))
	core.AssertTrue(t, result.OK, result.Error())
	state = result.Value.(work.ProviderState)
	core.AssertEqual(t, time.Date(2026, 7, 18, 18, 0, 0, 0, time.UTC), state.WindowStartedAt)
	core.AssertEqual(t, "", state.BackoffReason)
	core.AssertTrue(t, state.BackoffUntil.IsZero())

	var nilController *Controller
	core.AssertFalse(t, nilController.RecordStart("codex", "run-1", at).OK)
}

func TestController_Controller_RecordBackoff_Good(t *testing.T) {
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	until := at.Add(30 * time.Minute)
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)

	result := controller.RecordBackoff("codex", "provider retry-after", until, at)
	core.AssertTrue(t, result.OK, result.Error())
	state := result.Value.(work.ProviderState)
	core.AssertEqual(t, "codex", state.Provider)
	core.AssertEqual(t, "provider retry-after", state.BackoffReason)
	core.AssertEqual(t, until, state.BackoffUntil)
	core.AssertEqual(t, at, state.UpdatedAt)
}

func TestController_Controller_RecordBackoff_Bad(t *testing.T) {
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	at := time.Now()
	core.AssertFalse(t, controller.RecordBackoff("", "quota", at.Add(time.Hour), at).OK)
	core.AssertFalse(t, controller.RecordBackoff("codex", "", at.Add(time.Hour), at).OK)
}

func TestController_Controller_RecordBackoff_Ugly(t *testing.T) {
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueAccepting}, nil)
	core.AssertFalse(t, controller.RecordBackoff("codex", "quota", at, at).OK)
	core.AssertFalse(t, controller.RecordBackoff("codex", "quota", at.Add(time.Hour), time.Time{}).OK)

	var nilController *Controller
	core.AssertFalse(t, nilController.RecordBackoff("codex", "quota", at.Add(time.Hour), at).OK)
}

func TestController_Controller_Restore_Good(t *testing.T) {
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueFrozen}, nil)
	result := controller.Restore(work.QueueState{
		ID: "default", Status: work.QueueAccepting, UpdatedAt: at,
	}, []work.ProviderState{{
		Provider: "codex", BackoffReason: "quota", BackoffUntil: at.Add(time.Hour), UpdatedAt: at,
	}})
	core.AssertTrue(t, result.OK, result.Error())
	decision := controller.Decide(Candidate{
		RunID: "run-restore", Provider: "codex", QueuedAt: at,
	}, Runtime{Now: at})
	core.AssertTrue(t, decision.OK, decision.Error())
	core.AssertFalse(t, decision.Value.(Decision).Allowed)
	core.AssertEqual(t, at.Add(time.Hour), decision.Value.(Decision).NotBefore)
}

func TestController_Controller_Restore_Bad(t *testing.T) {
	var controller *Controller
	core.AssertFalse(t, controller.Restore(work.QueueState{}, nil).OK)
	controller = mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueFrozen}, nil)
	core.AssertFalse(t, controller.Restore(work.QueueState{ID: "other", Status: work.QueueFrozen}, nil).OK)
}

func TestController_Controller_Restore_Ugly(t *testing.T) {
	controller := mustQueueController(t, controllerTestPolicy(), work.QueueState{ID: "default", Status: work.QueueFrozen}, nil)
	providers := []work.ProviderState{{Provider: "codex"}, {Provider: "codex"}}
	result := controller.Restore(work.QueueState{ID: "default", Status: work.QueueFrozen}, providers)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "duplicated")
}
