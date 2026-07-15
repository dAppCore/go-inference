// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"
)

// RoPEParameters are the backend-neutral rotary-position settings Gemma-4
// declares per attention class.
type RoPEParameters struct {
	PartialRotaryFactor float64
	RopeTheta           float64
	RopeType            string
	Factor              float64
}

// RoPEPolicy is the model-owned rotary-position surface runtimes consume.
type RoPEPolicy struct {
	Parameters map[string]RoPEParameters
}

func DefaultRoPEParameters(globalPartialRotaryFactor float64) map[string]RoPEParameters {
	return map[string]RoPEParameters{
		LayerTypeFullAttention: {
			PartialRotaryFactor: positiveFloat(globalPartialRotaryFactor),
			RopeTheta:           1000000,
			RopeType:            "proportional",
			Factor:              1,
		},
		LayerTypeSlidingAttention: {
			PartialRotaryFactor: 1,
			RopeTheta:           10000,
			RopeType:            "default",
			Factor:              1,
		},
	}
}

func RoPEPolicyOf(cfg TextConfig) RoPEPolicy {
	globalPartialRotaryFactor := GlobalPartialRotaryFactorOf(cfg)
	return RoPEPolicy{
		Parameters: MergeRoPEParameters(DefaultRoPEParameters(globalPartialRotaryFactor), cfg.RoPEParameters),
	}
}

func GlobalPartialRotaryFactorOf(cfg TextConfig) float64 {
	if cfg.GlobalPartialRotaryFactor > 0 {
		return cfg.GlobalPartialRotaryFactor
	}
	if params, ok := cfg.RoPEParameters[LayerTypeFullAttention]; ok && params.PartialRotaryFactor > 0 {
		return params.PartialRotaryFactor
	}
	return 0
}

func CloneRoPEParameters(src map[string]RoPEParameters) map[string]RoPEParameters {
	if len(src) == 0 {
		return nil
	}
	cloned := make(map[string]RoPEParameters, len(src))
	for attentionType, params := range src {
		if attentionType != "" {
			cloned[attentionType] = params
		}
	}
	if len(cloned) == 0 {
		return nil
	}
	return cloned
}

// OverlayRoPEParameters applies non-zero/non-empty overlay fields onto base.
func OverlayRoPEParameters(base, overlay map[string]RoPEParameters) map[string]RoPEParameters {
	if len(base) == 0 && len(overlay) == 0 {
		return nil
	}
	merged := CloneRoPEParameters(base)
	if merged == nil {
		merged = make(map[string]RoPEParameters, len(overlay))
	}
	for attentionType, params := range overlay {
		if attentionType == "" {
			continue
		}
		current := merged[attentionType]
		if params.PartialRotaryFactor != 0 {
			current.PartialRotaryFactor = params.PartialRotaryFactor
		}
		if params.RopeTheta != 0 {
			current.RopeTheta = params.RopeTheta
		}
		if params.RopeType != "" {
			current.RopeType = params.RopeType
		}
		if params.Factor != 0 {
			current.Factor = params.Factor
		}
		merged[attentionType] = current
	}
	if len(merged) == 0 {
		return nil
	}
	return merged
}

// MergeRoPEParameters fills missing fields from defaults and keeps additional
// declared attention classes intact.
func MergeRoPEParameters(defaults, overrides map[string]RoPEParameters) map[string]RoPEParameters {
	if len(defaults) == 0 && len(overrides) == 0 {
		return nil
	}
	merged := CloneRoPEParameters(defaults)
	if merged == nil {
		merged = make(map[string]RoPEParameters, len(overrides))
	}
	for attentionType, params := range overrides {
		if attentionType == "" {
			continue
		}
		if defaultsForType, ok := merged[attentionType]; ok {
			if params.PartialRotaryFactor == 0 {
				params.PartialRotaryFactor = defaultsForType.PartialRotaryFactor
			}
			if params.RopeTheta == 0 {
				params.RopeTheta = defaultsForType.RopeTheta
			}
			if params.RopeType == "" {
				params.RopeType = defaultsForType.RopeType
			}
			if params.Factor == 0 {
				params.Factor = defaultsForType.Factor
			}
		} else if params.Factor == 0 {
			params.Factor = 1
		}
		merged[attentionType] = params
	}
	if len(merged) == 0 {
		return nil
	}
	return merged
}

func ApplyRoPEPolicyLabels(labels map[string]string, policy RoPEPolicy) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	for attentionType, params := range policy.Parameters {
		labelType := ropeLabelType(attentionType)
		if labelType == "" {
			continue
		}
		if params.RopeTheta > 0 {
			setRoPELabel(labels, labelType, "theta", formatRoPEFloat(params.RopeTheta))
		}
		if params.PartialRotaryFactor > 0 {
			setRoPELabel(labels, labelType, "partial_rotary_factor", formatRoPEFloat(params.PartialRotaryFactor))
		}
		if params.RopeType != "" {
			setRoPELabel(labels, labelType, "type", params.RopeType)
		}
		if params.Factor > 0 {
			setRoPELabel(labels, labelType, "factor", formatRoPEFloat(params.Factor))
		}
	}
	return labels
}

func setRoPELabel(labels map[string]string, labelType, suffix, value string) {
	labels["attention_rope_"+labelType+"_"+suffix] = value
	labels["gemma4_attention_rope_"+labelType+"_"+suffix] = value
}

func ropeLabelType(attentionType string) string {
	attentionType = strings.TrimSpace(attentionType)
	attentionType = strings.TrimSuffix(attentionType, "_attention")
	return attentionType
}

func formatRoPEFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func positiveFloat(value float64) float64 {
	if value > 0 {
		return value
	}
	return 0
}
