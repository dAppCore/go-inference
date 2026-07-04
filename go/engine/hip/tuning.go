// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/json"
	"errors"
	"hash/fnv"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"dappco.re/go/inference"
)

const rocmTuningMachineHashLabel = "machine_hash"

var (
	_ inference.MachineDiscoverer = (*rocmBackend)(nil)
	_ inference.TuningPlanner     = (*rocmBackend)(nil)
)

// DiscoverMachine reports the local ROCm runtime and optional model-pack
// metadata without loading model weights.
func DiscoverMachine(ctx context.Context, req inference.MachineDiscoveryRequest) (*inference.MachineDiscoveryReport, error) {
	return (&rocmBackend{}).DiscoverMachine(ctx, req)
}

func (b *rocmBackend) DiscoverMachine(ctx context.Context, req inference.MachineDiscoveryRequest) (*inference.MachineDiscoveryReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	caps := b.Capabilities()
	device := rocmMachineDiscoveryDevice(b)
	machineHash := rocmTuningMachineHash(device)
	device.Labels = rocmTuningLabelsWithMachineHash(device.Labels, machineHash)
	workloads := rocmTuningWorkloadsOrDefault(req.Workloads)
	report := &inference.MachineDiscoveryReport{
		Runtime:      rocmTuningRuntimeIdentity(caps.Runtime, device, nil, ""),
		Device:       device,
		Available:    caps.Available,
		Capabilities: rocmTuningCloneCapabilities(caps.Capabilities),
		CacheModes:   append([]string(nil), caps.CacheModes...),
		Workloads:    workloads,
		Labels:       rocmTuningLabelsWithMachineHash(req.Labels, machineHash),
	}
	if report.Labels == nil {
		report.Labels = map[string]string{}
	}
	report.Labels["backend"] = rocmTuningFirstNonEmptyString(report.Runtime.Backend, "rocm")
	report.Labels["production_requires_env_gate"] = "false"
	report.Labels["production_requires_cli_flag"] = "false"
	report.Labels["reactive_registry_planning"] = "true"
	if !req.IncludeModels && len(req.ModelDirs) == 0 {
		return report, nil
	}
	maxModels := req.MaxModels
	for _, dir := range req.ModelDirs {
		for discovered := range inference.Discover(dir) {
			if err := ctx.Err(); err != nil {
				return report, err
			}
			report.Models = append(report.Models, discovered)
			if req.IncludeCandidates {
				modelIdentity := rocmTuningModelIdentityFromDiscovered(discovered)
				if inspection, err := b.InspectModelPack(ctx, discovered.Path); err == nil {
					modelIdentity = rocmTuningModelIdentityFromInspection(inspection, modelIdentity)
				} else {
					report.Warnings = append(report.Warnings, err.Error())
				}
				planLabels := cloneStringMap(req.Labels)
				if planLabels == nil {
					planLabels = map[string]string{}
				}
				planLabels["discovery_model_path"] = discovered.Path
				plan, err := b.PlanTuning(ctx, inference.TuningPlanRequest{
					Runtime:   report.Runtime,
					Device:    report.Device,
					Model:     modelIdentity,
					Workloads: workloads,
					Budget:    inference.TuningBudget{MaxCandidates: len(workloads)},
					Labels:    planLabels,
				})
				if err != nil {
					report.Warnings = append(report.Warnings, err.Error())
				} else {
					report.Candidates = append(report.Candidates, plan.Candidates...)
				}
			}
			if maxModels > 0 && len(report.Models) >= maxModels {
				return report, nil
			}
		}
	}
	return report, nil
}

// PlanLocalTuning proposes metadata-only ROCm load candidates for a model.
func PlanLocalTuning(ctx context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
	plan, err := (&rocmBackend{}).PlanTuning(ctx, req)
	if err != nil {
		return inference.TuningPlan{}, err
	}
	return *plan, nil
}

func (b *rocmBackend) PlanTuning(ctx context.Context, req inference.TuningPlanRequest) (*inference.TuningPlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	caps := b.Capabilities()
	device := req.Device
	if rocmTuningDeviceInfoIsZero(device) {
		device = rocmMachineDiscoveryDevice(b)
	}
	machineHash := rocmTuningMachineHash(device)
	device.Labels = rocmTuningLabelsWithMachineHash(device.Labels, machineHash)
	model := req.Model
	var inspection *inference.ModelPackInspection
	if strings.TrimSpace(model.Path) != "" {
		if inspected, err := b.InspectModelPack(ctx, model.Path); err == nil {
			inspection = inspected
			model = rocmTuningModelIdentityFromInspection(inspected, model)
		}
	}
	profile, hasProfile := rocmTuningModelProfile(model, inspection)
	if hasProfile {
		model = rocmTuningModelIdentityWithProfile(model, profile)
	}
	fastLane := DefaultProductionFastLane()
	cacheMode := rocmTuningNativeCandidateCacheMode(fastLane.CacheMode)
	runtime := rocmTuningRuntimeIdentity(req.Runtime, device, &profile, cacheMode)
	if runtime.Backend == "" {
		runtime.Backend = rocmTuningFirstNonEmptyString(caps.Runtime.Backend, "rocm")
	}
	workloads := rocmTuningWorkloadsOrDefault(req.Workloads)
	candidateCap := len(workloads)
	if req.Budget.MaxCandidates > 0 && req.Budget.MaxCandidates < candidateCap {
		candidateCap = req.Budget.MaxCandidates
	}
	plan := &inference.TuningPlan{
		Runtime:     runtime,
		Device:      device,
		Model:       model,
		Adapter:     req.Adapter,
		Workloads:   workloads,
		Candidates:  make([]inference.TuningCandidate, 0, candidateCap),
		Recommended: make(map[inference.TuningWorkload]string, candidateCap),
		Labels:      rocmTuningLabelsWithMachineHash(req.Labels, machineHash),
	}
	if !hasProfile {
		plan.Warnings = append(plan.Warnings, "model did not resolve to a ROCm registry profile")
	}
	for _, workload := range workloads {
		candidate := rocmTuningCandidateForWorkload(workload, model, req.Adapter, runtime, profile, hasProfile, fastLane, req.Labels)
		plan.Candidates = append(plan.Candidates, candidate)
		if plan.Recommended[workload] == "" {
			plan.Recommended[workload] = candidate.ID
		}
		if req.Budget.MaxCandidates > 0 && len(plan.Candidates) >= req.Budget.MaxCandidates {
			break
		}
	}
	if len(plan.Recommended) == 0 {
		plan.Recommended = nil
	}
	return plan, nil
}

// TuningCandidateLoadConfig converts a selected tuning candidate into the
// ROCm-specific config plus backend-neutral load options needed by
// LoadModelWithConfig.
func TuningCandidateLoadConfig(candidate inference.TuningCandidate) (ROCmLoadConfig, []inference.LoadOption) {
	return TuningCandidateROCmLoadConfig(candidate), TuningCandidateLoadOptions(candidate)
}

// TuningCandidateLoadOptions converts the backend-neutral portion of a selected
// tuning candidate into go-inference load options.
func TuningCandidateLoadOptions(candidate inference.TuningCandidate) []inference.LoadOption {
	opts := make([]inference.LoadOption, 0, 4)
	if candidate.Runtime.Backend != "" {
		opts = append(opts, inference.WithBackend(candidate.Runtime.Backend))
	}
	if candidate.ContextLength > 0 {
		opts = append(opts, inference.WithContextLen(candidate.ContextLength))
	}
	if candidate.ParallelSlots > 0 {
		opts = append(opts, inference.WithParallelSlots(candidate.ParallelSlots))
	}
	if candidate.Adapter.Path != "" {
		opts = append(opts, inference.WithAdapterPath(candidate.Adapter.Path))
	}
	return opts
}

// TuningCandidateROCmLoadConfig converts the ROCm-specific portion of a
// selected tuning candidate into native load config.
func TuningCandidateROCmLoadConfig(candidate inference.TuningCandidate) ROCmLoadConfig {
	cacheMode := rocmTuningCandidateLoadCacheMode(rocmTuningFirstNonEmptyString(candidate.CacheMode, candidate.Runtime.CacheMode))
	if cacheMode == "" {
		return ROCmLoadConfig{}
	}
	return ROCmLoadConfig{
		CacheMode:    cacheMode,
		DeviceKVMode: cacheMode,
	}
}

// LoadModelWithTuningCandidate loads a model using candidate-derived settings.
// Explicit opts are applied after candidate-derived opts so callers can override
// a persisted profile when needed.
func LoadModelWithTuningCandidate(path string, candidate inference.TuningCandidate, opts ...inference.LoadOption) (inference.TextModel, error) {
	if strings.TrimSpace(path) == "" {
		path = candidate.Model.Path
	}
	cfg, candidateOpts := TuningCandidateLoadConfig(candidate)
	merged := make([]inference.LoadOption, 0, len(candidateOpts)+len(opts))
	merged = append(merged, candidateOpts...)
	merged = append(merged, opts...)
	return LoadModelWithConfig(path, cfg, merged...)
}

// CurrentMachineProfileHash returns the discovery hash used to key persisted
// tuning profiles for this machine.
func CurrentMachineProfileHash(ctx context.Context) (string, error) {
	report, err := DiscoverMachine(ctx, inference.MachineDiscoveryRequest{})
	if err != nil {
		return "", err
	}
	if report.Labels != nil && report.Labels[rocmTuningMachineHashLabel] != "" {
		return report.Labels[rocmTuningMachineHashLabel], nil
	}
	if report.Device.Labels != nil && report.Device.Labels[rocmTuningMachineHashLabel] != "" {
		return report.Device.Labels[rocmTuningMachineHashLabel], nil
	}
	return "", errors.New("current ROCm machine hash unavailable")
}

// SelectTuningResult returns the highest-scoring successful tuning result.
func SelectTuningResult(results []inference.TuningResult) (inference.TuningResult, bool) {
	var best inference.TuningResult
	found := false
	for _, result := range results {
		if result.Error != "" {
			continue
		}
		if !found || result.Score.Score > best.Score.Score {
			best = result
			found = true
		}
	}
	return best, found
}

// TuningSelectionLabels records why one tuning result won a measured sweep.
func TuningSelectionLabels(results []inference.TuningResult, selected inference.TuningResult) map[string]string {
	labels := map[string]string{
		"source":           "go-rocm tune-run",
		"selection_policy": "highest_successful_score",
		"selection_reason": "selected highest successful score from measured tuning candidates",
		"selected_score":   strconv.FormatFloat(selected.Score.Score, 'f', 6, 64),
	}
	if selected.Candidate.ID != "" {
		labels["selected_candidate_id"] = selected.Candidate.ID
	}
	if selected.Measurements.DecodeTokensPerSec > 0 {
		labels["selected_decode_tokens_per_sec"] = strconv.FormatFloat(selected.Measurements.DecodeTokensPerSec, 'f', 6, 64)
	}
	if selected.Measurements.LoadMilliseconds > 0 {
		labels["selected_load_milliseconds"] = strconv.FormatFloat(selected.Measurements.LoadMilliseconds, 'f', 6, 64)
	}
	if selected.Measurements.FirstTokenMilliseconds > 0 {
		labels["selected_first_token_milliseconds"] = strconv.FormatFloat(selected.Measurements.FirstTokenMilliseconds, 'f', 6, 64)
	}
	if selected.Measurements.KVRestoreMilliseconds > 0 {
		labels["selected_restore_milliseconds"] = strconv.FormatFloat(selected.Measurements.KVRestoreMilliseconds, 'f', 6, 64)
	}
	if selected.Measurements.PeakMemoryBytes > 0 {
		labels["selected_peak_memory_bytes"] = strconv.FormatUint(selected.Measurements.PeakMemoryBytes, 10)
	}
	if selected.Measurements.CorrectnessSmokeResult != "" {
		labels["selected_correctness_smoke_result"] = selected.Measurements.CorrectnessSmokeResult
	}
	if selected.Measurements.CorrectnessSmokeChecks > 0 {
		labels["selected_correctness_smoke_checks"] = strconv.Itoa(selected.Measurements.CorrectnessSmokeChecks)
	}
	successful := 0
	failed := 0
	var runnerUp inference.TuningResult
	hasRunnerUp := false
	for _, result := range results {
		if result.Error != "" {
			failed++
			continue
		}
		successful++
		if result.Candidate.ID == selected.Candidate.ID && result.Score.Score == selected.Score.Score {
			continue
		}
		if !hasRunnerUp || result.Score.Score > runnerUp.Score.Score {
			runnerUp = result
			hasRunnerUp = true
		}
	}
	labels["successful_candidates"] = strconv.Itoa(successful)
	labels["failed_candidates"] = strconv.Itoa(failed)
	if hasRunnerUp {
		if runnerUp.Candidate.ID != "" {
			labels["runner_up_candidate_id"] = runnerUp.Candidate.ID
		}
		labels["runner_up_score"] = strconv.FormatFloat(runnerUp.Score.Score, 'f', 6, 64)
		labels["selection_score_delta"] = strconv.FormatFloat(selected.Score.Score-runnerUp.Score.Score, 'f', 6, 64)
	}
	return labels
}

// BuildTuningProfile creates a durable inference.TuningProfile from a selected
// result, filling any missing candidate identity from the source plan.
func BuildTuningProfile(plan inference.TuningPlan, modelPath, machineHash string, workload inference.TuningWorkload, result inference.TuningResult, labels map[string]string, createdAt time.Time) inference.TuningProfile {
	candidate := result.Candidate
	if candidate.Model.Path == "" && plan.Model.Path != "" {
		candidate.Model = plan.Model
	}
	if candidate.Model.Path == "" {
		candidate.Model.Path = modelPath
	}
	if candidate.Runtime.Backend == "" {
		candidate.Runtime = plan.Runtime
	}
	if candidate.Adapter.Path == "" && plan.Adapter.Path != "" {
		candidate.Adapter = plan.Adapter
	}
	if candidate.Workload == "" {
		candidate.Workload = workload
	}
	score := result.Score
	if score.Workload == "" {
		score.Workload = workload
	}
	profileLabels := cloneStringMap(labels)
	if profileLabels == nil {
		profileLabels = map[string]string{}
	}
	if profileLabels["source"] == "" {
		profileLabels["source"] = "go-rocm tune-run"
	}
	return inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: machineHash,
			Runtime:     candidate.Runtime,
			Model:       candidate.Model,
			Adapter:     candidate.Adapter,
			Workload:    workload,
		},
		Candidate:     candidate,
		Measurements:  result.Measurements,
		Score:         score,
		CreatedAtUnix: createdAt.Unix(),
		Labels:        profileLabels,
	}
}

// TuningProfilePath returns the conventional profile JSON path for a built
// profile inside profileDir.
func TuningProfilePath(profileDir string, profile inference.TuningProfile) string {
	modelName := filepath.Base(profile.Key.Model.Path)
	if modelName == "." || modelName == string(filepath.Separator) {
		modelName = ""
	}
	if modelName == "" {
		modelName = profile.Candidate.Model.Architecture
	}
	if modelName == "" {
		modelName = profile.Key.Model.Architecture
	}
	machineHash := profile.Key.MachineHash
	if parts := strings.SplitN(machineHash, ":", 2); len(parts) == 2 {
		machineHash = parts[1]
	}
	name := strings.Join([]string{
		rocmTuningProfileFilePart(string(profile.Key.Workload), "workload", 32),
		rocmTuningProfileFilePart(machineHash, "machine", 12),
		rocmTuningProfileFilePart(modelName, "model", 48),
		rocmTuningProfileFilePart(profile.Candidate.ID, "candidate", 48),
	}, "-") + ".json"
	return filepath.Join(profileDir, name)
}

// WriteTuningProfile persists a profile as pretty JSON with private file
// permissions.
func WriteTuningProfile(path string, profile inference.TuningProfile) error {
	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

// ModelIdentityFromTuningProfile overlays candidate model metadata on the
// persisted profile key.
func ModelIdentityFromTuningProfile(profile inference.TuningProfile) inference.ModelIdentity {
	return rocmTuningMergeModelIdentity(profile.Key.Model, profile.Candidate.Model)
}

// RuntimeIdentityFromTuningProfile overlays candidate runtime metadata on the
// persisted profile key.
func RuntimeIdentityFromTuningProfile(profile inference.TuningProfile) inference.RuntimeIdentity {
	identity := profile.Key.Runtime
	candidate := profile.Candidate.Runtime
	if candidate.Backend != "" {
		identity.Backend = candidate.Backend
	}
	if candidate.Device != "" {
		identity.Device = candidate.Device
	}
	if candidate.Version != "" {
		identity.Version = candidate.Version
	}
	if candidate.CacheMode != "" {
		identity.CacheMode = candidate.CacheMode
	}
	if candidate.NativeRuntime {
		identity.NativeRuntime = candidate.NativeRuntime
	}
	if len(candidate.Labels) > 0 {
		identity.Labels = cloneStringMap(candidate.Labels)
	}
	return identity
}

// AdapterIdentityFromTuningProfile overlays candidate adapter metadata on the
// persisted profile key.
func AdapterIdentityFromTuningProfile(profile inference.TuningProfile) inference.AdapterIdentity {
	identity := profile.Key.Adapter
	candidate := profile.Candidate.Adapter
	if candidate.Path != "" {
		identity.Path = candidate.Path
	}
	if candidate.Hash != "" {
		identity.Hash = candidate.Hash
	}
	if candidate.Format != "" {
		identity.Format = candidate.Format
	}
	if candidate.Rank != 0 {
		identity.Rank = candidate.Rank
	}
	if candidate.Alpha != 0 {
		identity.Alpha = candidate.Alpha
	}
	if len(candidate.TargetKeys) > 0 {
		identity.TargetKeys = append([]string(nil), candidate.TargetKeys...)
	}
	if candidate.BaseModelHash != "" {
		identity.BaseModelHash = candidate.BaseModelHash
	}
	if len(candidate.Labels) > 0 {
		identity.Labels = cloneStringMap(candidate.Labels)
	}
	return identity
}

func rocmTuningModelProfile(model inference.ModelIdentity, inspection *inference.ModelPackInspection) (ROCmModelProfile, bool) {
	if inspection != nil {
		if profile, ok := ResolveROCmModelProfileForInspection(inspection); ok {
			return profile, true
		}
	}
	return ResolveROCmModelProfile(model.Path, model)
}

func rocmTuningModelIdentityWithProfile(model inference.ModelIdentity, profile ROCmModelProfile) inference.ModelIdentity {
	if profile.Model.Architecture != "" {
		model = rocmTuningMergeModelIdentity(model, profile.Model)
	}
	model.Labels = ApplyROCmModelProfileLabels(model.Labels, profile)
	return model
}

func rocmTuningModelIdentityFromInspection(inspection *inference.ModelPackInspection, fallback inference.ModelIdentity) inference.ModelIdentity {
	if inspection == nil {
		return fallback
	}
	model := rocmTuningMergeModelIdentity(fallback, inspection.Model)
	if model.Path == "" {
		model.Path = inspection.Path
	}
	labels := cloneStringMap(inspection.Labels)
	for key, value := range model.Labels {
		if value != "" {
			if labels == nil {
				labels = map[string]string{}
			}
			labels[key] = value
		}
	}
	model.Labels = labels
	return model
}

func rocmTuningModelIdentityFromDiscovered(discovered inference.DiscoveredModel) inference.ModelIdentity {
	return inference.ModelIdentity{
		Path:         discovered.Path,
		Architecture: normalizeROCmArchitecture(discovered.ModelType),
		QuantBits:    discovered.QuantBits,
		QuantGroup:   discovered.QuantGroup,
		QuantType:    rocmTuningFirstNonEmptyString(discovered.QuantType, discovered.QuantFamily),
		Labels: map[string]string{
			"format":     discovered.Format,
			"num_files":  strconv.Itoa(discovered.NumFiles),
			"model_type": discovered.ModelType,
		},
	}
}

func rocmTuningMergeModelIdentity(base, overlay inference.ModelIdentity) inference.ModelIdentity {
	if overlay.ID != "" {
		base.ID = overlay.ID
	}
	if overlay.Path != "" {
		base.Path = overlay.Path
	}
	if overlay.Architecture != "" {
		base.Architecture = overlay.Architecture
	}
	if overlay.Revision != "" {
		base.Revision = overlay.Revision
	}
	if overlay.Hash != "" {
		base.Hash = overlay.Hash
	}
	if overlay.QuantBits > 0 {
		base.QuantBits = overlay.QuantBits
	}
	if overlay.QuantGroup > 0 {
		base.QuantGroup = overlay.QuantGroup
	}
	if overlay.QuantType != "" {
		base.QuantType = overlay.QuantType
	}
	if overlay.ContextLength > 0 {
		base.ContextLength = overlay.ContextLength
	}
	if overlay.NumLayers > 0 {
		base.NumLayers = overlay.NumLayers
	}
	if overlay.HiddenSize > 0 {
		base.HiddenSize = overlay.HiddenSize
	}
	if overlay.VocabSize > 0 {
		base.VocabSize = overlay.VocabSize
	}
	labels := cloneStringMap(base.Labels)
	for key, value := range overlay.Labels {
		if value != "" {
			if labels == nil {
				labels = map[string]string{}
			}
			labels[key] = value
		}
	}
	base.Labels = labels
	return base
}

func rocmTuningRuntimeIdentity(runtime inference.RuntimeIdentity, device inference.MachineDeviceInfo, profile *ROCmModelProfile, cacheMode string) inference.RuntimeIdentity {
	if runtime.Backend == "" {
		runtime.Backend = "rocm"
	}
	if runtime.Device == "" {
		runtime.Device = rocmTuningFirstNonEmptyString(device.Architecture, device.Name, "rocm")
	}
	if cacheMode != "" {
		runtime.CacheMode = cacheMode
	}
	labels := cloneStringMap(runtime.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels["backend"] = runtime.Backend
	labels["production_requires_env_gate"] = "false"
	labels["production_requires_cli_flag"] = "false"
	if profile != nil && profile.Matched() {
		runtime.NativeRuntime = profile.LoadStatus.NativeRuntime
		labels = ApplyROCmModelProfileLabels(labels, *profile)
	}
	runtime.Labels = labels
	return runtime
}

func rocmTuningCandidateForWorkload(workload inference.TuningWorkload, model inference.ModelIdentity, adapter inference.AdapterIdentity, runtime inference.RuntimeIdentity, profile ROCmModelProfile, hasProfile bool, fastLane ProductionFastLane, requestLabels map[string]string) inference.TuningCandidate {
	contextLength := rocmTuningFirstPositiveInt(model.ContextLength, fastLane.ContextLength, 4096)
	cacheMode := rocmTuningNativeCandidateCacheMode(fastLane.CacheMode)
	labels := rocmTuningCandidateLabels(runtime.Backend, &profile, hasProfile, fastLane, requestLabels)
	reasons := []string{"registry-derived ROCm discovery candidate; optional tune smoke can validate it before persistence"}
	if !hasProfile || !profile.Matched() {
		labels["engine_profile_matched"] = "false"
		labels["candidate_status"] = "registry_profile_unmatched"
		reasons = append(reasons, "model did not resolve to a ROCm registry profile")
	} else if profile.LoadStatus.Reason != "" {
		reasons = append(reasons, profile.LoadStatus.Reason)
	}
	candidate := inference.TuningCandidate{
		Workload:             workload,
		Model:                model,
		Adapter:              adapter,
		Runtime:              rocmTuningRuntimeIdentity(runtime, inference.MachineDeviceInfo{}, &profile, cacheMode),
		ContextLength:        contextLength,
		ParallelSlots:        1,
		PromptCache:          false,
		PromptCacheMinTokens: 128,
		CachePolicy:          "default",
		CacheMode:            cacheMode,
		BatchSize:            1,
		PrefillChunkSize:     1024,
		ExpectedQuantization: rocmTuningFirstPositiveInt(model.QuantBits, fastLane.QuantBits),
		Reasons:              reasons,
		Labels:               labels,
	}
	candidate.Runtime.Labels = cloneStringMap(labels)
	switch workload {
	case inference.TuningWorkloadLowLatency:
		candidate.ContextLength = rocmTuningMinPositive(candidate.ContextLength, 32768)
		candidate.BatchSize = 1
		candidate.ParallelSlots = 1
		candidate.PrefillChunkSize = rocmTuningMinPositive(candidate.PrefillChunkSize, 512)
		candidate.Reasons = append(candidate.Reasons, "low-latency profile keeps batch and prefill chunks small")
	case inference.TuningWorkloadThroughput:
		candidate.BatchSize = 4
		candidate.ParallelSlots = 2
		candidate.PrefillChunkSize = rocmTuningMaxPositive(candidate.PrefillChunkSize, 2048)
		candidate.Reasons = append(candidate.Reasons, "throughput profile favours larger batches where memory permits")
	case inference.TuningWorkloadLongContext:
		candidate.PromptCache = true
		candidate.CachePolicy = "full"
		candidate.PrefillChunkSize = rocmTuningMaxPositive(candidate.PrefillChunkSize, 2048)
		candidate.Reasons = append(candidate.Reasons, "long-context profile favours full prompt-cache retention")
	case inference.TuningWorkloadAgentState:
		candidate.PromptCache = true
		candidate.CachePolicy = "stateful"
		candidate.Labels["state_restore"] = "candidate"
		candidate.Labels["reactive_state_continuity"] = "candidate"
		candidate.Runtime.Labels["state_restore"] = "candidate"
		candidate.Runtime.Labels["reactive_state_continuity"] = "candidate"
		candidate.Reasons = append(candidate.Reasons, "agent-state profile measures prompt-cache and state restore")
	case inference.TuningWorkloadCoding:
		candidate.Reasons = append(candidate.Reasons, "coding profile keeps the production fast-lane context and native cache mode")
	default:
		candidate.Reasons = append(candidate.Reasons, "chat profile uses the production fast-lane defaults")
	}
	candidate.ID = inference.CandidateID(candidate.Workload, candidate.CacheMode, candidate.ContextLength, candidate.BatchSize)
	return candidate
}

func rocmTuningCandidateLabels(backendName string, profile *ROCmModelProfile, hasProfile bool, fastLane ProductionFastLane, requestLabels map[string]string) map[string]string {
	labels := map[string]string{
		"candidate_source":             "go-rocm PlanTuning",
		"candidate_contract":           "go-inference.tuning-candidate",
		"backend":                      rocmTuningFirstNonEmptyString(backendName, "rocm"),
		"production_fast_lane":         fastLane.Name,
		"production_default":           strconv.FormatBool(fastLane.EnabledByDefault),
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"production_cache_mode":        fastLane.CacheMode,
		"candidate_cache_mode_source":  "native-compatible-fast-lane",
		"candidate_cache_mode_bound":   "true",
		"reactive_registry_planning":   "true",
	}
	if hasProfile && profile != nil && profile.Matched() {
		labels = ApplyROCmModelProfileLabels(labels, *profile)
		if profile.LoadStatus.Status != "" {
			labels["engine_load_status"] = string(profile.LoadStatus.Status)
			labels["engine_load_target"] = profile.LoadStatus.Target
			labels["engine_load_text_generate"] = strconv.FormatBool(profile.LoadStatus.TextGenerate)
			labels["candidate_status"] = string(profile.LoadStatus.Status)
		}
		if profile.EngineFeatures.ChatTemplateID != "" {
			labels["chat_template_id"] = profile.EngineFeatures.ChatTemplateID
		}
		if profile.EngineFeatures.ReasoningParserID != "" {
			labels["reasoning_parser_id"] = profile.EngineFeatures.ReasoningParserID
		}
	}
	for key, value := range requestLabels {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmTuningWorkloadsOrDefault(workloads []inference.TuningWorkload) []inference.TuningWorkload {
	if len(workloads) == 0 {
		return inference.DefaultTuningWorkloads()
	}
	return append([]inference.TuningWorkload(nil), workloads...)
}

func rocmTuningNativeCandidateCacheMode(productionCacheMode string) string {
	mode := strings.ToLower(strings.TrimSpace(productionCacheMode))
	mode = strings.ReplaceAll(mode, "_", "-")
	switch mode {
	case "fp16", "q8", "k-q8-v-q4":
		return mode
	case "kq8vq4":
		return "k-q8-v-q4"
	default:
		return "k-q8-v-q4"
	}
}

func rocmTuningCandidateLoadCacheMode(raw string) string {
	trimmed := strings.TrimSpace(raw)
	mode := strings.ToLower(trimmed)
	mode = strings.ReplaceAll(mode, "_", "-")
	switch mode {
	case "fp16", "q8", "k-q8-v-q4":
		return mode
	case "kq8vq4":
		return "k-q8-v-q4"
	default:
		return trimmed
	}
}

func rocmTuningProfileFilePart(value, fallback string, maxLen int) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var builder strings.Builder
	lastDash := false
	for i := 0; i < len(value); i++ {
		b := value[i]
		if (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') {
			builder.WriteByte(b)
			lastDash = false
			continue
		}
		if builder.Len() > 0 && !lastDash {
			builder.WriteByte('-')
			lastDash = true
		}
	}
	part := rocmTuningTrimProfileFileDashes(builder.String())
	if part == "" {
		part = fallback
	}
	if maxLen > 0 && len(part) > maxLen {
		part = rocmTuningTrimProfileFileDashes(part[:maxLen])
	}
	if part == "" {
		return fallback
	}
	return part
}

func rocmTuningTrimProfileFileDashes(value string) string {
	for len(value) > 0 && value[len(value)-1] == '-' {
		value = value[:len(value)-1]
	}
	return value
}

func rocmTuningLabelsWithMachineHash(labels map[string]string, machineHash string) map[string]string {
	out := cloneStringMap(labels)
	if machineHash == "" {
		return out
	}
	if out == nil {
		out = map[string]string{}
	}
	out[rocmTuningMachineHashLabel] = machineHash
	return out
}

func rocmTuningMachineHash(device inference.MachineDeviceInfo) string {
	h := fnv.New64a()
	write := func(value string) {
		if value == "" {
			return
		}
		_, _ = h.Write([]byte(value))
		_, _ = h.Write([]byte{0})
	}
	write(device.Name)
	write(device.Architecture)
	write(strconv.FormatUint(device.MemorySize, 10))
	write(strconv.FormatUint(device.MaxRecommendedWorkingSetSize, 10))
	if h.Sum64() == fnv.New64a().Sum64() {
		return ""
	}
	return strconv.FormatUint(h.Sum64(), 16)
}

func rocmTuningDeviceInfoIsZero(device inference.MachineDeviceInfo) bool {
	return device.Name == "" &&
		device.Architecture == "" &&
		device.MaxBufferLength == 0 &&
		device.MaxRecommendedWorkingSetSize == 0 &&
		device.MemorySize == 0 &&
		len(device.Labels) == 0
}

func rocmTuningCloneCapabilities(in []inference.Capability) []inference.Capability {
	if len(in) == 0 {
		return nil
	}
	out := make([]inference.Capability, len(in))
	for i, capability := range in {
		out[i] = capability
		out[i].Labels = cloneStringMap(capability.Labels)
	}
	return out
}

func rocmTuningFirstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func rocmTuningFirstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func rocmTuningMaxPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a > b {
		return a
	}
	return b
}

func rocmTuningMinPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}
