// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// CaptureState implements the shared StatefulModel metadata bundle surface.
// Durable KV bytes remain URI-first through SleepState/WakeState; this method
// captures the portable envelope and sampler/runtime metadata only.
func (m *rocmModel) CaptureState(ctx context.Context, prompt string, opts ...inference.GenerateOption) (bundle *inference.StateBundle, err error) {
	if m == nil {
		return nil, core.E("rocm.CaptureState", "model is nil", nil)
	}
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cfg := m.applyGenerateOpts(opts)
	promptTokens, err := m.resolveGenerateGemma4Context(prompt, &cfg, "rocm.CaptureState")
	if err != nil {
		return nil, err
	}
	metrics := m.Metrics()
	model := m.modelIdentity()
	labels := map[string]string{
		"backend":              "rocm",
		"state_bundle":         "metadata_only",
		"state_bundle_kv_refs": "use_sleep_state",
	}
	for key, value := range m.kernelStatus().Labels() {
		labels[key] = value
	}
	labels = rocmApplyGemma4StateArtifactLabels(labels, model)
	adapter := m.ActiveAdapter()
	rocmAddStateBundleAdapterLabels(labels, adapter)
	return &inference.StateBundle{
		Version:         "rocm-state-bundle-v1",
		CreatedAtUnix:   time.Now().Unix(),
		Model:           model,
		Adapter:         adapter,
		Sampler:         rocmSamplerConfig(cfg),
		Runtime:         inference.RuntimeIdentity{Backend: "rocm", NativeRuntime: true, Labels: m.kernelStatus().Labels()},
		PromptHash:      rocmPromptHash(prompt),
		PromptTokens:    promptTokens,
		GeneratedTokens: metrics.GeneratedTokens,
		Labels:          labels,
	}, nil
}

// RestoreState validates a portable metadata bundle and installs a matching
// StateSession envelope. KV payload restore still requires WakeState with a
// concrete state store.
func (m *rocmModel) RestoreState(ctx context.Context, bundle *inference.StateBundle) (err error) {
	if m == nil {
		return core.E("rocm.RestoreState", "model is nil", nil)
	}
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if bundle == nil {
		return core.E("rocm.RestoreState", "state bundle is nil", nil)
	}
	if err := checkROCmStateModelCompatibility("rocm.RestoreState", m.modelIdentity(), bundle.Model); err != nil {
		return err
	}
	if err := checkROCmAdapterModelCompatibility("rocm.RestoreState", m.modelIdentity(), bundle.Adapter); err != nil {
		return err
	}
	labels := mergeStringMaps(bundle.Labels, map[string]string{
		"backend":          "rocm",
		"kv_restore":       "metadata_only",
		"state_bundle":     "restored",
		"state_bundle_kv":  "use_wake_state",
		"state_bundle_ref": core.Sprintf("%d", len(bundle.KVRefs)),
	})
	rocmAddStateBundleAdapterLabels(labels, bundle.Adapter)
	next := NewStateSession(bundle.Model, bundle.Tokenizer, labels)
	m.stateMutex.Lock()
	previous := m.state
	if previous != nil {
		if err := previous.Close(); err != nil {
			m.stateMutex.Unlock()
			return core.E("rocm.RestoreState", "close previous state runtime", err)
		}
	}
	m.state = next
	m.stateMutex.Unlock()
	return nil
}

func rocmSamplerConfig(cfg inference.GenerateConfig) inference.SamplerConfig {
	return inference.SamplerConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		RepeatPenalty: cfg.RepeatPenalty,
		StopTokens:    append([]int32(nil), cfg.StopTokens...),
		ReturnLogits:  cfg.ReturnLogits,
	}
}

func rocmPromptHash(prompt string) string {
	sum := sha256.Sum256([]byte(prompt))
	return "sha256:" + hex.EncodeToString(sum[:])
}
