// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

// Candidate is one queued run being considered for admission.
type Candidate struct {
	RunID    string
	Provider string
	Model    string
	QueuedAt time.Time
}

// Runtime is the durable queue view used for a pure admission decision.
type Runtime struct {
	Queued  []work.Run
	Running []work.Run
	Now     time.Time
}

// Decision reports whether a candidate may start and when to reconsider it.
type Decision struct {
	Allowed   bool
	Reason    string
	NotBefore time.Time
}

// Controller owns queue admission policy and durable queue/provider state.
type Controller struct {
	mu        sync.Mutex
	policy    Policy
	state     work.QueueState
	providers map[string]work.ProviderState
}

type quotaWindow struct {
	Start time.Time
	End   time.Time
}

// NewController validates and isolates policy plus recovered durable state.
func NewController(policy Policy, initial work.QueueState, providers []work.ProviderState) core.Result {
	policyResult := validatePolicy(policy)
	if !policyResult.OK {
		return policyResult
	}

	initial.ID = core.Trim(initial.ID)
	if initial.ID == "" {
		initial.ID = "default"
	}
	if initial.ID != "default" {
		return core.Fail(core.NewError("agent queue state ID must be default"))
	}
	if initial.Status == "" {
		initial.Status = work.QueueFrozen
	}
	if !validQueueStatus(initial.Status) {
		return core.Fail(core.Errorf("agent queue status %q is invalid", initial.Status))
	}
	initial.Reason = core.Trim(initial.Reason)
	if initial.Status == work.QueueFrozen && initial.Reason == "" {
		initial.Reason = "queue requires explicit start"
	}

	providerStates := make(map[string]work.ProviderState, len(providers))
	for _, providerState := range providers {
		providerState.Provider = core.Trim(providerState.Provider)
		if providerState.Provider == "" {
			return core.Fail(core.NewError("agent queue provider state requires a provider"))
		}
		if _, exists := providerStates[providerState.Provider]; exists {
			return core.Fail(core.Errorf("agent queue provider state for %s is duplicated", providerState.Provider))
		}
		if providerState.WindowAdmissions < 0 {
			return core.Fail(core.Errorf("agent queue admissions for %s cannot be negative", providerState.Provider))
		}
		providerStates[providerState.Provider] = providerState
	}

	return core.Ok(&Controller{
		policy:    clonePolicy(policyResult.Value.(Policy)),
		state:     initial,
		providers: providerStates,
	})
}

// Start enables admission and returns the state that must be persisted.
func (controller *Controller) Start(at time.Time) core.Result {
	if controller == nil {
		return core.Fail(core.NewError("agent queue controller is required"))
	}
	if at.IsZero() {
		return core.Fail(core.NewError("agent queue start time is required"))
	}

	controller.mu.Lock()
	defer controller.mu.Unlock()
	if controller.state.Status == work.QueueAccepting {
		return core.Ok(controller.state)
	}
	controller.state.Status = work.QueueAccepting
	controller.state.Reason = ""
	controller.state.UpdatedAt = at
	return core.Ok(controller.state)
}

// Stop prevents new admissions and drains active runs when necessary.
func (controller *Controller) Stop(active int, at time.Time) core.Result {
	if controller == nil {
		return core.Fail(core.NewError("agent queue controller is required"))
	}
	if active < 0 {
		return core.Fail(core.NewError("agent queue active run count cannot be negative"))
	}
	if at.IsZero() {
		return core.Fail(core.NewError("agent queue stop time is required"))
	}

	controller.mu.Lock()
	defer controller.mu.Unlock()
	if active > 0 {
		controller.state.Status = work.QueueDraining
		controller.state.Reason = core.Sprintf("%d active runs draining", active)
	} else {
		controller.state.Status = work.QueueFrozen
		controller.state.Reason = "queue stopped"
	}
	controller.state.UpdatedAt = at
	return core.Ok(controller.state)
}

// Decide evaluates one candidate without mutating queue or provider state.
func (controller *Controller) Decide(candidate Candidate, runtime Runtime) core.Result {
	if controller == nil {
		return core.Fail(core.NewError("agent queue controller is required"))
	}
	candidate.RunID = core.Trim(candidate.RunID)
	candidate.Provider = core.Trim(candidate.Provider)
	candidate.Model = core.Trim(candidate.Model)
	if candidate.RunID == "" || candidate.Provider == "" || candidate.QueuedAt.IsZero() {
		return core.Fail(core.NewError("agent queue candidate requires run ID, provider, and queued time"))
	}
	if runtime.Now.IsZero() {
		return core.Fail(core.NewError("agent queue runtime time is required"))
	}

	controller.mu.Lock()
	policy := controller.policy
	queueState := controller.state
	providerState := controller.providers[candidate.Provider]
	controller.mu.Unlock()

	if queueState.Status != work.QueueAccepting {
		reason := core.Trim(queueState.Reason)
		if reason == "" {
			reason = core.Concat("queue is ", string(queueState.Status))
		}
		return core.Ok(Decision{Reason: reason})
	}

	activeTotal := activeRunCount(runtime.Running, "", "")
	if policy.Dispatch.GlobalConcurrency > 0 && activeTotal >= policy.Dispatch.GlobalConcurrency {
		return core.Ok(Decision{Reason: core.Sprintf("global concurrency %d/%d", activeTotal, policy.Dispatch.GlobalConcurrency)})
	}

	if limit, exists := policy.Concurrency[candidate.Provider]; exists {
		providerTotal := activeRunCount(runtime.Running, candidate.Provider, "")
		if limit.Total > 0 && providerTotal >= limit.Total {
			return core.Ok(Decision{Reason: core.Sprintf("provider %s concurrency %d/%d", candidate.Provider, providerTotal, limit.Total)})
		}
		if candidate.Model != "" {
			if modelLimit, limited := limit.Models[candidate.Model]; limited && modelLimit > 0 {
				modelTotal := activeRunCount(runtime.Running, candidate.Provider, candidate.Model)
				if modelTotal >= modelLimit {
					return core.Ok(Decision{Reason: core.Sprintf("model %s/%s concurrency %d/%d", candidate.Provider, candidate.Model, modelTotal, modelLimit)})
				}
			}
		}
	}

	if !providerState.BackoffUntil.IsZero() && runtime.Now.Before(providerState.BackoffUntil) {
		reason := core.Trim(providerState.BackoffReason)
		if reason == "" {
			reason = core.Concat("provider ", candidate.Provider, " backoff")
		}
		return core.Ok(Decision{Reason: reason, NotBefore: providerState.BackoffUntil})
	}

	rate := policy.Rates[candidate.Provider]
	delay := admissionDelay(rate, runtime.Now)
	if delay > 0 && !providerState.LastStartedAt.IsZero() {
		notBefore := providerState.LastStartedAt.Add(delay)
		if runtime.Now.Before(notBefore) {
			return core.Ok(Decision{
				Reason:    core.Sprintf("provider %s pacing", candidate.Provider),
				NotBefore: notBefore,
			})
		}
	}

	if rate.DailyLimit > 0 {
		window := quotaWindowFor(rate, runtime.Now)
		if providerState.WindowAdmissions >= rate.DailyLimit && providerState.WindowStartedAt.Equal(window.Start) {
			return core.Ok(Decision{
				Reason:    core.Sprintf("provider %s daily quota %d/%d", candidate.Provider, providerState.WindowAdmissions, rate.DailyLimit),
				NotBefore: window.End,
			})
		}
	}

	if earlierRunID := earlierQueuedRun(candidate, runtime.Queued); earlierRunID != "" {
		return core.Ok(Decision{Reason: core.Concat("queued behind earlier run ", earlierRunID)})
	}
	return core.Ok(Decision{Allowed: true})
}

// RecordStart updates provider timing and quota state after durable admission.
func (controller *Controller) RecordStart(provider, runID string, at time.Time) core.Result {
	if controller == nil {
		return core.Fail(core.NewError("agent queue controller is required"))
	}
	provider = core.Trim(provider)
	runID = core.Trim(runID)
	if provider == "" || runID == "" {
		return core.Fail(core.NewError("agent queue start requires provider and run ID"))
	}
	if at.IsZero() {
		return core.Fail(core.NewError("agent queue start time is required"))
	}

	controller.mu.Lock()
	defer controller.mu.Unlock()
	state := controller.providers[provider]
	state.Provider = provider
	window := quotaWindowFor(controller.policy.Rates[provider], at)
	if !state.WindowStartedAt.Equal(window.Start) {
		state.WindowStartedAt = window.Start
		state.WindowAdmissions = 0
	}
	state.WindowAdmissions++
	state.LastRunID = runID
	state.LastStartedAt = at
	state.UpdatedAt = at
	if !state.BackoffUntil.IsZero() && !at.Before(state.BackoffUntil) {
		state.BackoffReason = ""
		state.BackoffUntil = time.Time{}
	}
	controller.providers[provider] = state
	return core.Ok(state)
}

// RecordBackoff updates a provider retry deadline after a rate-limit event.
func (controller *Controller) RecordBackoff(provider, reason string, until, at time.Time) core.Result {
	if controller == nil {
		return core.Fail(core.NewError("agent queue controller is required"))
	}
	provider = core.Trim(provider)
	reason = core.Trim(reason)
	if provider == "" || reason == "" {
		return core.Fail(core.NewError("agent queue backoff requires provider and reason"))
	}
	if at.IsZero() || !until.After(at) {
		return core.Fail(core.NewError("agent queue backoff deadline must be after its update time"))
	}

	controller.mu.Lock()
	defer controller.mu.Unlock()
	state := controller.providers[provider]
	state.Provider = provider
	state.BackoffReason = reason
	state.BackoffUntil = until
	state.UpdatedAt = at
	controller.providers[provider] = state
	return core.Ok(state)
}

func validQueueStatus(status work.QueueStatus) bool {
	return status == work.QueueFrozen || status == work.QueueAccepting || status == work.QueueDraining
}

func activeRunCount(runs []work.Run, provider, model string) int {
	count := 0
	for _, run := range runs {
		if run.Status != work.RunPreparing && run.Status != work.RunRunning && run.Status != work.RunCancelling {
			continue
		}
		if provider != "" && core.Trim(run.Provider) != provider {
			continue
		}
		if model != "" && core.Trim(run.Model) != model {
			continue
		}
		count++
	}
	return count
}

func admissionDelay(rate RateConfig, at time.Time) time.Duration {
	delaySeconds := rate.SustainedDelay
	window := quotaWindowFor(rate, at)
	if rate.BurstWindow > 0 && window.End.Sub(at.UTC()) <= time.Duration(rate.BurstWindow)*time.Hour {
		delaySeconds = rate.BurstDelay
	}
	if rate.MinDelay > delaySeconds {
		delaySeconds = rate.MinDelay
	}
	return time.Duration(delaySeconds) * time.Second
}

func quotaWindowFor(rate RateConfig, at time.Time) quotaWindow {
	resetUTC := core.Trim(rate.ResetUTC)
	if resetUTC == "" {
		resetUTC = "00:00"
	}
	parts := core.SplitN(resetUTC, ":", 2)
	hour := 0
	minute := 0
	if len(parts) == 2 {
		if hourResult := core.Atoi(parts[0]); hourResult.OK {
			hour = hourResult.Value.(int)
		}
		if minuteResult := core.Atoi(parts[1]); minuteResult.OK {
			minute = minuteResult.Value.(int)
		}
	}
	now := at.UTC()
	start := time.Date(now.Year(), now.Month(), now.Day(), hour, minute, 0, 0, time.UTC)
	if now.Before(start) {
		start = start.AddDate(0, 0, -1)
	}
	return quotaWindow{Start: start, End: start.AddDate(0, 0, 1)}
}

func earlierQueuedRun(candidate Candidate, queued []work.Run) string {
	for _, run := range queued {
		if run.ID == candidate.RunID || run.Status != work.RunQueued || run.QueuedAt.IsZero() {
			continue
		}
		if run.QueuedAt.Before(candidate.QueuedAt) || run.QueuedAt.Equal(candidate.QueuedAt) && run.ID < candidate.RunID {
			return run.ID
		}
	}
	return ""
}
