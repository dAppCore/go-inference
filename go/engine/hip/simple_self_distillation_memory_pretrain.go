// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/memorypretrain"
)

// NativeSimpleSelfDistillationMemoryPretrainingConfig configures the ROCm
// package-local SSD+SFT AdamW pass followed by an offline hierarchical-memory
// bank build.
type NativeSimpleSelfDistillationMemoryPretrainingConfig struct {
	SSDAdamW NativeSimpleSelfDistillationAdamWConfig
	Embedder memorypretrain.Embedder
	Bank     memorypretrain.BuildConfig
	BankPath string
}

// NativeSimpleSelfDistillationMemoryPretrainingResult records the SSD training
// step and the offline hierarchical-memory bank built from generated samples.
type NativeSimpleSelfDistillationMemoryPretrainingResult struct {
	SSD        *SimpleSelfDistillationResult `json:"-"`
	NativeLoss bool                          `json:"native_loss"`
	Bank       *memorypretrain.Bank          `json:"-"`
	BankPath   string                        `json:"bank_path,omitempty"`
	Records    int                           `json:"records"`
	Labels     map[string]string             `json:"labels,omitempty"`
}

// RunModelNativeSimpleSelfDistillationMemoryPretraining runs local ROCm SSD
// generation plus SFT AdamW, then builds an offline hierarchical-memory bank
// from the accepted generated samples. It does not make rocmModel implement a
// public trainer interface and does not perform HIP layer injection.
func RunModelNativeSimpleSelfDistillationMemoryPretraining(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg NativeSimpleSelfDistillationMemoryPretrainingConfig) (*NativeSimpleSelfDistillationMemoryPretrainingResult, error) {
	if cfg.Embedder == nil {
		return nil, core.NewError("rocm: SSD memory pretraining embedder is nil")
	}
	ssd, nativeLoss, err := RunModelNativeSimpleSelfDistillationAdamWUpdatePass(ctx, model, dataset, cfg.SSDAdamW)
	result := &NativeSimpleSelfDistillationMemoryPretrainingResult{
		SSD:        ssd,
		NativeLoss: nativeLoss,
		BankPath:   cfg.BankPath,
		Labels:     nativeSimpleSelfDistillationMemoryPretrainingLabels(nativeLoss, cfg.BankPath),
	}
	addSimpleSelfDistillationMemoryPretrainingOptimizerLabels(result.Labels, ssd)
	if err != nil {
		return result, err
	}
	records, err := simpleSelfDistillationMemoryPretrainingRecords(ssd)
	if err != nil {
		return result, err
	}
	bank, err := memorypretrain.BuildBankFromCorpus(ctx, cfg.Embedder, records, cfg.Bank)
	result.Records = len(records)
	if err != nil {
		return result, err
	}
	result.Bank = bank
	result.Labels["memory_pretraining_bank_records"] = core.Sprintf("%d", len(records))
	result.Labels["memory_pretraining_bank_dimension"] = core.Sprintf("%d", bank.Dimension)
	if cfg.BankPath != "" {
		if err := bank.Save(cfg.BankPath); err != nil {
			return result, err
		}
	}
	return result, nil
}

func simpleSelfDistillationMemoryPretrainingRecords(result *SimpleSelfDistillationResult) ([]memorypretrain.CorpusRecord, error) {
	if result == nil {
		return nil, core.NewError("rocm: SSD memory pretraining result is nil")
	}
	if len(result.Samples) == 0 {
		return nil, core.NewError("rocm: SSD memory pretraining samples are required")
	}
	records := make([]memorypretrain.CorpusRecord, 0, len(result.Samples))
	for index, sample := range result.Samples {
		text := simpleSelfDistillationMemoryPretrainingText(sample)
		if text == "" {
			continue
		}
		meta := rocmCloneLabels(sample.Labels)
		if meta == nil {
			meta = make(map[string]string, 4)
		}
		meta["memory_pretraining_source"] = "simple_self_distillation"
		meta["memory_pretraining_source_index"] = core.Sprintf("%d", index)
		records = append(records, memorypretrain.CorpusRecord{
			ID:   core.Sprintf("ssd-%d", index),
			Text: text,
			Meta: meta,
		})
	}
	if len(records) == 0 {
		return nil, core.NewError("rocm: SSD memory pretraining samples produced no records")
	}
	return records, nil
}

func simpleSelfDistillationMemoryPretrainingText(sample SimpleSelfDistillationSample) string {
	switch {
	case sample.Prompt != "" && sample.Response != "":
		return sample.Prompt + "\n" + sample.Response
	case sample.Response != "":
		return sample.Response
	default:
		return sample.Prompt
	}
}

func nativeSimpleSelfDistillationMemoryPretrainingLabels(nativeLoss bool, bankPath string) map[string]string {
	labels := map[string]string{
		"memory_pretraining":               "hierarchical",
		"memory_pretraining_bank_builder":  "hierarchical_kmeans",
		"memory_pretraining_bank_runtime":  "cpu_native",
		"memory_pretraining_hip_injection": "pending",
		"memory_pretraining_injection":     "additive",
		"memory_pretraining_source":        "simple_self_distillation",
		"memory_pretraining_stage":         "ssd_sft_adamw_memory_bank_build",
		"ssd_native_loss_ready":            boolLabel(nativeLoss),
		"trainer_interface":                "not_implemented",
	}
	if bankPath != "" {
		labels["memory_pretraining_bank_file"] = bankPath
	}
	return labels
}

func addSimpleSelfDistillationMemoryPretrainingOptimizerLabels(labels map[string]string, ssd *SimpleSelfDistillationResult) {
	if labels == nil || ssd == nil || ssd.SFT == nil {
		return
	}
	for _, key := range []string{
		"optimizer_track",
		"optimizer_track_container",
		"optimizer_track_format",
		"optimizer_track_offset",
		"optimizer_track_path",
		"optimizer_track_payload_bytes",
		"optimizer_track_step",
		"optimizer_track_frames",
		"optimizer_track_list_helper",
		"optimizer_track_find_helper",
		"optimizer_track_load_step_helper",
	} {
		if value := ssd.SFT.Labels[key]; value != "" {
			labels["memory_pretraining_"+key] = value
		}
	}
}
