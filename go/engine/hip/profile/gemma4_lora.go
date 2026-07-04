// SPDX-Licence-Identifier: EUPL-1.2

package profile

import "strings"

// LoRATargetPolicy describes the loader-neutral adapter target policy a model
// family owns.
type LoRATargetPolicy struct {
	DefaultTargets  []string          `json:"default_targets,omitempty"`
	SafeTargets     []string          `json:"safe_targets,omitempty"`
	ExtendedTargets []string          `json:"extended_targets,omitempty"`
	TargetPaths     map[string]string `json:"target_paths,omitempty"`
}

var gemma4LoRADefaultTargets = []string{"q_proj", "v_proj", "o_proj"}
var gemma4LoRASafeTargets = []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
var gemma4LoRAExtendedTargets = []string{
	"router.proj",
	"per_layer_input_gate",
	"per_layer_projection",
}
var gemma4LoRATargets = append(cloneStringSlice(gemma4LoRASafeTargets), gemma4LoRAExtendedTargets...)
var gemma4LoRATargetPaths = map[string]string{
	"q_proj":               "self_attn.q_proj",
	"self_attn.q_proj":     "self_attn.q_proj",
	"k_proj":               "self_attn.k_proj",
	"self_attn.k_proj":     "self_attn.k_proj",
	"v_proj":               "self_attn.v_proj",
	"self_attn.v_proj":     "self_attn.v_proj",
	"o_proj":               "self_attn.o_proj",
	"self_attn.o_proj":     "self_attn.o_proj",
	"gate_proj":            "mlp.gate_proj",
	"mlp.gate_proj":        "mlp.gate_proj",
	"up_proj":              "mlp.up_proj",
	"mlp.up_proj":          "mlp.up_proj",
	"down_proj":            "mlp.down_proj",
	"mlp.down_proj":        "mlp.down_proj",
	"router.proj":          "router.proj",
	"per_layer_input_gate": "per_layer_input_gate",
	"per_layer_projection": "per_layer_projection",
}

// LoRATargetPolicyForArchitecture returns the adapter target policy declared
// by the active architecture registry.
func LoRATargetPolicyForArchitecture(architecture string) (LoRATargetPolicy, bool) {
	settings, ok := LookupArchitectureProfile(architecture)
	if !ok {
		return LoRATargetPolicy{}, false
	}
	return LoRATargetPolicyForProfile(settings)
}

// LoRATargetPolicyForProfile returns the adapter target policy carried by an
// architecture profile.
func LoRATargetPolicyForProfile(settings ArchitectureProfile) (LoRATargetPolicy, bool) {
	settings = CloneGemma4ArchitectureSettings(settings)
	policy := LoRATargetPolicy{
		DefaultTargets:  cleanLoRATargets(settings.LoRADefaultTargets),
		SafeTargets:     safeLoRATargetsFromProfile(settings.LoRATargets, settings.LoRAExtendedTargets, settings.LoRATargetPaths),
		ExtendedTargets: cleanLoRATargets(settings.LoRAExtendedTargets),
		TargetPaths:     cloneStringMap(settings.LoRATargetPaths),
	}
	if len(policy.DefaultTargets) == 0 && len(policy.SafeTargets) == 0 &&
		len(policy.ExtendedTargets) == 0 && len(policy.TargetPaths) == 0 {
		return LoRATargetPolicy{}, false
	}
	if len(policy.DefaultTargets) == 0 {
		policy.DefaultTargets = cloneStringSlice(policy.SafeTargets)
	}
	if len(policy.SafeTargets) == 0 {
		policy.SafeTargets = cloneStringSlice(policy.DefaultTargets)
	}
	return CloneLoRATargetPolicy(policy), true
}

// Gemma4LoRATargetPolicyForArchitecture returns the model-owned Gemma-4 LoRA
// target policy for target architectures. The attached assistant drafter
// deliberately has no standalone adapter targets.
func Gemma4LoRATargetPolicyForArchitecture(architecture string) (LoRATargetPolicy, bool) {
	if !Gemma4LoRATargetArchitecture(architecture) {
		return LoRATargetPolicy{}, false
	}
	return LoRATargetPolicyForArchitecture(architecture)
}

// CloneLoRATargetPolicy returns a deep copy of policy.
func CloneLoRATargetPolicy(policy LoRATargetPolicy) LoRATargetPolicy {
	return LoRATargetPolicy{
		DefaultTargets:  cloneStringSlice(policy.DefaultTargets),
		SafeTargets:     cloneStringSlice(policy.SafeTargets),
		ExtendedTargets: cloneStringSlice(policy.ExtendedTargets),
		TargetPaths:     cloneStringMap(policy.TargetPaths),
	}
}

// Gemma4LoRADefaultTargets returns the narrow default adapter target set for
// Gemma-4 target models.
func Gemma4LoRADefaultTargets(architecture string) []string {
	if !Gemma4LoRATargetArchitecture(architecture) {
		return nil
	}
	return DefaultLoRATargets(architecture)
}

// Gemma4LoRATargetPath canonicalizes a Gemma-4 LoRA target key to its model
// projection path.
func Gemma4LoRATargetPath(architecture, target string) (string, bool) {
	if !Gemma4LoRATargetArchitecture(architecture) {
		return "", false
	}
	return LoRATargetPath(architecture, target)
}

// Gemma4LoRASafeTarget reports whether target is enabled without the extended
// target opt-in.
func Gemma4LoRASafeTarget(architecture, target string) bool {
	if !Gemma4LoRATargetArchitecture(architecture) {
		return false
	}
	return SafeLoRATarget(architecture, target)
}

// Gemma4LoRAExtendedTarget reports whether target is registered as an
// explicit-opt-in extended Gemma-4 target.
func Gemma4LoRAExtendedTarget(architecture, target string) bool {
	if !Gemma4LoRATargetArchitecture(architecture) {
		return false
	}
	return LoRAExtendedTarget(architecture, target)
}

// Gemma4LoRACanonicalTarget canonicalizes a possibly layer-qualified target to
// the model projection path used by adapter metadata.
func Gemma4LoRACanonicalTarget(architecture, target string) (string, bool) {
	if !Gemma4LoRATargetArchitecture(architecture) {
		return "", false
	}
	return LoRACanonicalTarget(architecture, target)
}

// DefaultLoRATargets returns the registered narrow default LoRA target set for
// architecture. Nil means the architecture is unknown or declares no adapter
// targets.
func DefaultLoRATargets(architecture string) []string {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return nil
	}
	return cloneStringSlice(policy.DefaultTargets)
}

// LoRATargetPath canonicalizes a LoRA target key through the architecture
// registry's target-path map.
func LoRATargetPath(architecture, target string) (string, bool) {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return "", false
	}
	path, ok := policy.TargetPaths[strings.TrimSpace(target)]
	return path, ok
}

// SafeLoRATarget reports whether target can be enabled without an extended
// target opt-in.
func SafeLoRATarget(architecture, target string) bool {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return false
	}
	target = strings.TrimSpace(target)
	path, ok := policy.TargetPaths[target]
	if !ok {
		return false
	}
	if loRATargetListMatches(policy.ExtendedTargets, policy.TargetPaths, target, path) {
		return false
	}
	return loRATargetListMatches(policy.SafeTargets, policy.TargetPaths, target, path)
}

// LoRAExtendedTarget reports whether target is registered as an explicit
// opt-in LoRA target for architecture.
func LoRAExtendedTarget(architecture, target string) bool {
	policy, ok := LoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return false
	}
	target = strings.TrimSpace(target)
	path, ok := policy.TargetPaths[target]
	if !ok {
		return false
	}
	return loRATargetListMatches(policy.ExtendedTargets, policy.TargetPaths, target, path)
}

// LoRACanonicalTarget canonicalizes a possibly layer-qualified target to the
// projection path used by adapter metadata.
func LoRACanonicalTarget(architecture, target string) (string, bool) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", false
	}
	parts := strings.Split(target, ".")
	if len(parts) >= 2 {
		short := parts[len(parts)-2] + "." + parts[len(parts)-1]
		if canonical, ok := LoRATargetPath(architecture, short); ok {
			return joinLoRACanonicalTarget(parts[:len(parts)-2], canonical), true
		}
		if len(parts) == 2 {
			return "", false
		}
	}
	short := parts[len(parts)-1]
	if canonical, ok := LoRATargetPath(architecture, short); ok {
		return joinLoRACanonicalTarget(parts[:len(parts)-1], canonical), true
	}
	return "", false
}

// Gemma4LoRATargetArchitecture reports whether architecture is a Gemma-4
// target model that can own LoRA adapters.
func Gemma4LoRATargetArchitecture(architecture string) bool {
	switch Gemma4ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified":
		return true
	default:
		return false
	}
}

func joinLoRACanonicalTarget(prefix []string, canonical string) string {
	if len(prefix) == 0 {
		return canonical
	}
	parts := make([]string, 0, len(prefix)+strings.Count(canonical, ".")+1)
	parts = append(parts, prefix...)
	parts = append(parts, strings.Split(canonical, ".")...)
	return strings.Join(parts, ".")
}

func safeLoRATargetsFromProfile(targets, extendedTargets []string, paths map[string]string) []string {
	targets = cleanLoRATargets(targets)
	extendedTargets = cleanLoRATargets(extendedTargets)
	out := make([]string, 0, len(targets))
	for _, target := range targets {
		path := target
		if canonical, ok := paths[target]; ok && canonical != "" {
			path = canonical
		}
		if containsString(extendedTargets, target) || containsString(extendedTargets, path) {
			continue
		}
		out = append(out, target)
	}
	return out
}

func cleanLoRATargets(targets []string) []string {
	if len(targets) == 0 {
		return nil
	}
	out := make([]string, 0, len(targets))
	seen := map[string]struct{}{}
	for _, target := range targets {
		target = strings.TrimSpace(target)
		if target == "" {
			continue
		}
		if _, ok := seen[target]; ok {
			continue
		}
		seen[target] = struct{}{}
		out = append(out, target)
	}
	return out
}

func containsString(values []string, target string) bool {
	target = strings.TrimSpace(target)
	if target == "" {
		return false
	}
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func loRATargetListMatches(values []string, paths map[string]string, target, path string) bool {
	if containsString(values, target) || containsString(values, path) {
		return true
	}
	for _, value := range values {
		canonical, ok := paths[value]
		if !ok {
			continue
		}
		if canonical == target || canonical == path {
			return true
		}
	}
	return false
}
