// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ROCmLoadConfig carries ROCm-specific load decisions that are intentionally
// narrower than the backend-neutral go-inference LoadConfig.
type ROCmLoadConfig struct {
	CacheMode    string `json:"cache_mode,omitempty"`
	DeviceKVMode string `json:"device_kv_mode,omitempty"`
}

// LoadModelWithConfig loads a model with ROCm-specific native runtime settings.
func LoadModelWithConfig(path string, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return (&rocmBackend{}).LoadModelWithConfig(path, cfg, opts...)
}

func (b *rocmBackend) LoadModelWithConfig(path string, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return b.loadModelWithROCmConfig(path, inference.ApplyLoadOpts(opts), cfg)
}

func (cfg ROCmLoadConfig) active() bool {
	return strings.TrimSpace(cfg.CacheMode) != "" || strings.TrimSpace(cfg.DeviceKVMode) != ""
}

func (cfg ROCmLoadConfig) deviceKVMode() (string, error) {
	raw := firstNonEmptyString(strings.TrimSpace(cfg.DeviceKVMode), strings.TrimSpace(cfg.CacheMode))
	if raw == "" {
		return "", nil
	}
	mode, ok := normalizeROCmDeviceKVMode(raw)
	if !ok {
		return "", core.E("rocm.LoadModel", core.Sprintf("unsupported ROCm device KV cache mode %q", raw), nil)
	}
	return mode, nil
}

func normalizeROCmDeviceKVMode(raw string) (string, bool) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	mode = strings.ReplaceAll(mode, "_", "-")
	switch mode {
	case rocmKVCacheModeFP16, rocmKVCacheModeQ8:
		return mode, true
	case "kq8vq4", rocmKVCacheModeKQ8VQ4:
		return rocmKVCacheModeKQ8VQ4, true
	default:
		return "", false
	}
}

func rocmApplyNativeLoadDeviceKVModeLabels(labels map[string]string, mode string) map[string]string {
	if strings.TrimSpace(mode) == "" {
		return labels
	}
	if labels == nil {
		labels = map[string]string{}
	}
	labels["kv_cache_mode"] = mode
	labels["device_kv_mode"] = mode
	labels["kv_cache_source"] = "load_config"
	return labels
}
