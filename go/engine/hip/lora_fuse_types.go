// SPDX-Licence-Identifier: EUPL-1.2

package hip

import "dappco.re/go/inference"

const LoRAFuseProvenanceFile = "adapter_provenance.json"

type LoRAFuseOptions struct {
	BasePath     string                    `json:"base_path"`
	AdapterPath  string                    `json:"adapter_path"`
	OutputPath   string                    `json:"output_path"`
	Architecture string                    `json:"architecture,omitempty"`
	Adapter      inference.AdapterIdentity `json:"adapter,omitempty"`
	Labels       map[string]string         `json:"labels,omitempty"`
}

type LoRAFuseResult struct {
	OutputPath      string                    `json:"output_path"`
	WeightFiles     []string                  `json:"weight_files,omitempty"`
	ProvenancePath  string                    `json:"provenance_path,omitempty"`
	Adapter         inference.AdapterIdentity `json:"adapter,omitempty"`
	FusedWeights    int                       `json:"fused_weights"`
	FusedWeightKeys []string                  `json:"fused_weight_keys,omitempty"`
	FusedLayers     []string                  `json:"fused_layers,omitempty"`
	Labels          map[string]string         `json:"labels,omitempty"`
}

type LoRAFuseProvenance struct {
	Version         int                       `json:"version"`
	SourcePath      string                    `json:"source_path"`
	OutputPath      string                    `json:"output_path"`
	WeightFiles     []string                  `json:"weight_files,omitempty"`
	Adapter         inference.AdapterIdentity `json:"adapter,omitempty"`
	FusedWeightKeys []string                  `json:"fused_weight_keys,omitempty"`
	FusedLayers     []string                  `json:"fused_layers,omitempty"`
	Labels          map[string]string         `json:"labels,omitempty"`
}
