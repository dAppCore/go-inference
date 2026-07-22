// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPLoRAModel_MetalHeadAdapterArtifact_Good(t *testing.T) {
	path := t.TempDir()
	wantA := []float32{0.25, -0.5}
	wantB := []float32{0.75, 1.25, -1.5}

	core.RequireNoError(t, saveMetalHeadLoRAAdapter(path, wantA, wantB, 3, 2, 1, 4))
	adapter, err := loadMetalHeadLoRAAdapter(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, wantA, adapter.a)
	core.AssertEqual(t, wantB, adapter.b)
	core.AssertEqual(t, 1, adapter.rank)
	core.AssertEqual(t, 2, adapter.hiddenSize)
	core.AssertEqual(t, 3, adapter.vocabSize)
	core.AssertEqual(t, float32(4), adapter.alpha)

	_, err = os.Stat(filepath.Join(path, "adapter.safetensors"))
	core.RequireNoError(t, err)
	_, err = os.Stat(filepath.Join(path, "adapter_config.json"))
	core.RequireNoError(t, err)
}

func TestHIPLoRAModel_MetalHeadAdapterArtifact_Bad(t *testing.T) {
	path := t.TempDir()
	core.RequireNoError(t, os.WriteFile(filepath.Join(path, "adapter_config.json"), []byte(`{"rank":1,"alpha":4}`), 0o644))

	_, err := loadMetalHeadLoRAAdapter(path)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "load adapter.safetensors")
}

func TestHIPLoRAModel_Gemma4MetalHeadAdapter_Good(t *testing.T) {
	path := t.TempDir()
	core.RequireNoError(t, saveMetalHeadLoRAAdapter(path, []float32{0.25, -0.5}, []float32{0.75, 1.25, -1.5}, 3, 2, 1, 4))
	model := &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text", HiddenSize: 2, VocabSize: 3}}

	adapter, identity, err := model.loadGemma4HeadLoRAAdapter(path)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{0.25, -0.5}, adapter.a)
	core.AssertEqual(t, []float32{0.75, 1.25, -1.5}, adapter.b)
	core.AssertEqual(t, "hip_gemma4_lm_head", identity.Labels["adapter_runtime"])
	core.AssertEqual(t, []string{"lm_head.weight"}, identity.TargetKeys)
}

func TestHIPLoRAModel_Gemma4MetalHeadAdapter_Bad(t *testing.T) {
	path := t.TempDir()
	core.RequireNoError(t, saveMetalHeadLoRAAdapter(path, []float32{0.25, -0.5}, []float32{0.75, 1.25, -1.5}, 3, 2, 1, 4))
	model := &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text", HiddenSize: 4, VocabSize: 3}}

	_, _, err := model.loadGemma4HeadLoRAAdapter(path)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "hidden size mismatch")
}

func TestHIPLoRAModel_Gemma4ForwardConfig_Good(t *testing.T) {
	adapter := &hipLoadedSmallLoRAAdapter{a: []float32{1, 2}, b: []float32{1, 2, 3}, rank: 1, alpha: 2}
	model := &hipLoadedModel{
		gemma4LoRA: adapter,
		q4ConfigOK: true,
		q4Layers:   1,
		q4Config: hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{
			HiddenSize: 2,
			VocabSize:  3,
		}}},
	}

	cfg, err := model.cachedGemma4Q4ForwardConfig(1)

	core.RequireNoError(t, err)
	core.AssertTrue(t, cfg.HeadLoRA == adapter)
	core.AssertTrue(t, cfg.usesDenseProjectionWeights())
}

func TestHIPLoRAModel_TinyAdapterValidation_Bad(t *testing.T) {
	cfg := hipLoadedTinyLMConfig{HiddenSize: 2, VocabSize: 3}
	valid := func() hipTinyLoRAAdapterFile {
		return hipTinyLoRAAdapterFile{
			Format:     rocmTinyLoRAFormat,
			Target:     "output.weight",
			Rank:       1,
			Alpha:      1,
			HiddenSize: cfg.HiddenSize,
			VocabSize:  cfg.VocabSize,
			LoRAA:      []float32{1, 0},
			LoRAB:      []float32{0, 1, 2},
			Bias:       []float32{0, 0, 0},
		}
	}
	_, err := validateTinyLoRAAdapterFile(valid(), cfg)
	core.RequireNoError(t, err)

	tests := []struct {
		name   string
		mutate func(*hipTinyLoRAAdapterFile)
		want   string
	}{
		{
			name: "unsupported format",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Format = "unsupported"
			},
			want: "unsupported adapter format",
		},
		{
			name: "unsupported target",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Target = "attention.q_proj"
			},
			want: "unsupported adapter target",
		},
		{
			name: "hidden size mismatch",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.HiddenSize = cfg.HiddenSize + 1
			},
			want: "adapter hidden size mismatch",
		},
		{
			name: "vocab size mismatch",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.VocabSize = cfg.VocabSize + 1
			},
			want: "adapter vocab size mismatch",
		},
		{
			name: "zero rank",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Rank = 0
			},
			want: "adapter rank must be positive",
		},
		{
			name: "zero alpha",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Alpha = 0
			},
			want: "adapter alpha must be positive and finite",
		},
		{
			name: "nan alpha",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Alpha = float32(math.NaN())
			},
			want: "adapter alpha must be positive and finite",
		},
		{
			name: "inf alpha",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Alpha = float32(math.Inf(1))
			},
			want: "adapter alpha must be positive and finite",
		},
		{
			name: "lora a length",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.LoRAA = file.LoRAA[:1]
			},
			want: "adapter LoRA A length must match rank*hidden",
		},
		{
			name: "lora b length",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.LoRAB = file.LoRAB[:2]
			},
			want: "adapter LoRA B length must match vocab*rank",
		},
		{
			name: "bias length",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Bias = []float32{1, 2}
			},
			want: "adapter bias length must match vocab",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file := valid()
			tt.mutate(&file)

			_, err := validateTinyLoRAAdapterFile(file, cfg)

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPLoRAModel_SmallAdapterValidation_Bad(t *testing.T) {
	cfg := hipLoadedSmallDecodeConfig{Architecture: "qwen3", HiddenSize: 2, VocabSize: 3}
	valid := func() hipTinyLoRAAdapterFile {
		return hipTinyLoRAAdapterFile{
			Format:     rocmSmallLoRAFormat,
			Target:     "lm_head.weight",
			Rank:       1,
			Alpha:      1,
			HiddenSize: cfg.HiddenSize,
			VocabSize:  cfg.VocabSize,
			LoRAA:      []float32{1, 0},
			LoRAB:      []float32{0, 1, 2},
			Bias:       []float32{0, 0, 0},
		}
	}
	_, err := validateSmallLoRAAdapterFile(valid(), cfg)
	core.RequireNoError(t, err)

	tests := []struct {
		name   string
		mutate func(*hipTinyLoRAAdapterFile)
		want   string
	}{
		{
			name: "unsupported format",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Format = "unsupported"
			},
			want: "unsupported small LM-head adapter format",
		},
		{
			name: "unsupported delegated target",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Target = "model.layers.0.mlp"
			},
			want: "unsupported adapter target",
		},
		{
			name: "delegated hidden mismatch",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.HiddenSize = cfg.HiddenSize + 1
			},
			want: "adapter hidden size mismatch",
		},
		{
			name: "delegated vocab mismatch",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.VocabSize = cfg.VocabSize + 1
			},
			want: "adapter vocab size mismatch",
		},
		{
			name: "delegated alpha",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.Alpha = float32(math.Inf(-1))
			},
			want: "adapter alpha must be positive and finite",
		},
		{
			name: "delegated lora b length",
			mutate: func(file *hipTinyLoRAAdapterFile) {
				file.LoRAB = file.LoRAB[:2]
			},
			want: "adapter LoRA B length must match vocab*rank",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file := valid()
			tt.mutate(&file)

			_, err := validateSmallLoRAAdapterFile(file, cfg)

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPLoRAModel_SmallAdapterStatusUsesDenseRoute_Good(t *testing.T) {
	core.AssertEqual(t, "experimental_qwen_gemma_small_decode", hipSmallDecodeLoRAModelStatus("qwen3"))
	core.AssertEqual(t, "experimental_qwen_gemma_small_decode", hipSmallDecodeLoRAModelStatus("gemma4_text"))
	core.AssertEqual(t, "experimental_dense_small_decode", hipSmallDecodeLoRAModelStatus("mistral"))
	core.AssertEqual(t, "experimental_dense_small_decode", hipSmallDecodeLoRAModelStatus("phi"))
	core.AssertEqual(t, "experimental_dense_small_decode", hipSmallDecodeLoRAModelStatus("glm4"))
	core.AssertEqual(t, "experimental_dense_small_decode", hipSmallDecodeLoRAModelStatus("hermes"))
	core.AssertEqual(t, "experimental_dense_small_decode", hipSmallDecodeLoRAModelStatus("granite"))
}

func TestHIPLoRAModel_ClassifierAdapterValidation_Bad(t *testing.T) {
	cfg := hipLoadedSequenceClassifierConfig{
		HiddenSize:         2,
		NumLabels:          2,
		WeightTensor:       "classifier.weight",
		PositiveLabelIndex: 1,
	}
	valid := func() hipClassifierLoRAAdapterFile {
		return hipClassifierLoRAAdapterFile{
			Format:     rocmClassifierLoRAFormat,
			Target:     "classifier.weight",
			Rank:       1,
			Alpha:      1,
			HiddenSize: cfg.HiddenSize,
			NumLabels:  cfg.NumLabels,
			LoRAA:      []float32{1, 0},
			LoRAB:      []float32{0, 1},
			Bias:       []float32{0, 0},
		}
	}
	_, err := validateClassifierLoRAAdapterFile(valid(), cfg)
	core.RequireNoError(t, err)

	tests := []struct {
		name   string
		mutate func(*hipClassifierLoRAAdapterFile)
		want   string
	}{
		{
			name: "unsupported format",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Format = "rocm-tiny-lora"
			},
			want: "unsupported classifier adapter format",
		},
		{
			name: "unsupported target",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Target = "pooler.dense.weight"
			},
			want: "unsupported classifier adapter target",
		},
		{
			name: "hidden size mismatch",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.HiddenSize = cfg.HiddenSize + 1
			},
			want: "classifier adapter hidden size mismatch",
		},
		{
			name: "label count mismatch",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.NumLabels = cfg.NumLabels + 1
			},
			want: "classifier adapter label count mismatch",
		},
		{
			name: "zero rank",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Rank = 0
			},
			want: "classifier adapter rank must be positive",
		},
		{
			name: "negative alpha",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Alpha = -1
			},
			want: "classifier adapter alpha must be positive and finite",
		},
		{
			name: "nan alpha",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Alpha = float32(math.NaN())
			},
			want: "classifier adapter alpha must be positive and finite",
		},
		{
			name: "inf alpha",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Alpha = float32(math.Inf(1))
			},
			want: "classifier adapter alpha must be positive and finite",
		},
		{
			name: "lora a length",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.LoRAA = file.LoRAA[:1]
			},
			want: "classifier adapter LoRA A length must match rank*hidden",
		},
		{
			name: "lora b length",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.LoRAB = file.LoRAB[:1]
			},
			want: "classifier adapter LoRA B length must match labels*rank",
		},
		{
			name: "bias length",
			mutate: func(file *hipClassifierLoRAAdapterFile) {
				file.Bias = []float32{1}
			},
			want: "classifier adapter bias length must match label count",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file := valid()
			tt.mutate(&file)

			_, err := validateClassifierLoRAAdapterFile(file, cfg)

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPLoRAModel_HelperValidation_Bad(t *testing.T) {
	t.Run("merge classifier bias row count", func(t *testing.T) {
		_, err := mergeClassifierLoRABias(nil, nil, 0)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "classifier row count must be positive")
	})
	t.Run("merge classifier base bias length", func(t *testing.T) {
		_, err := mergeClassifierLoRABias([]float32{1}, nil, 2)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "classifier base bias length must match label count")
	})
	t.Run("merge classifier adapter bias length", func(t *testing.T) {
		_, err := mergeClassifierLoRABias(nil, []float32{1}, 2)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "classifier adapter bias length must match label count")
	})
	t.Run("attention hidden size", func(t *testing.T) {
		_, err := hipTinyAttentionWeightedOutput([]float32{1, 2}, []float32{1}, 0)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "hidden size must be positive")
	})
	t.Run("attention empty weights", func(t *testing.T) {
		_, err := hipTinyAttentionWeightedOutput([]float32{1, 2}, nil, 2)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "attention values must align with weights and hidden size")
	})
	t.Run("attention value alignment", func(t *testing.T) {
		_, err := hipTinyAttentionWeightedOutput([]float32{1, 2, 3}, []float32{1, 1}, 2)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "attention values must align with weights and hidden size")
	})
}

func TestHIPLoRAModel_RunProjectionRequiresActiveAdapter_Bad(t *testing.T) {
	model := &hipLoadedModel{}

	_, _, _, err := model.runTinyLoRAProjection(context.Background(), hipLoadedTinyLMConfig{}, []float32{1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "active LoRA adapter is required")

	_, _, _, err = model.runSmallLoRAProjection(context.Background(), hipLoadedSmallDecodeConfig{}, []float32{1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "active small LM-head LoRA adapter is required")

	_, err = model.runSequenceClassifierLoRAProjection(context.Background(), hipLoadedSequenceClassifierConfig{}, []float32{1}, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "active classifier LoRA adapter is required")
}

func TestHIPLoRAModel_LoadedWeightHelpersValidation_Bad(t *testing.T) {
	t.Run("tiny output weights nil driver", func(t *testing.T) {
		_, err := (*hipLoadedModel)(nil).loadedTinyOutputWeights(hipLoadedTinyLMConfig{
			OutputWeightEncoding: hipTinyOutputWeightEncodingFP32,
			OutputWeightBytes:    4,
			VocabSize:            1,
			HiddenSize:           1,
		})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "HIP driver is nil")
	})
	t.Run("tiny output weights copy failure", func(t *testing.T) {
		model := &hipLoadedModel{driver: &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed")}}
		_, err := model.loadedTinyOutputWeights(hipLoadedTinyLMConfig{
			OutputWeightPointer:  0x1000,
			OutputWeightBytes:    4,
			OutputWeightEncoding: hipTinyOutputWeightEncodingFP32,
			VocabSize:            1,
			HiddenSize:           1,
		})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "copy output weights")
	})
	t.Run("tiny output weights shape mismatch", func(t *testing.T) {
		payload, err := hipFloat32Payload([]float32{1, 2})
		core.RequireNoError(t, err)
		driver, pointer := hipLoRAModelTestDevicePayload(t, payload)
		model := &hipLoadedModel{driver: driver}

		_, err = model.loadedTinyOutputWeights(hipLoadedTinyLMConfig{
			OutputWeightPointer:  pointer,
			OutputWeightBytes:    uint64(len(payload)),
			OutputWeightEncoding: hipTinyOutputWeightEncodingFP32,
			VocabSize:            2,
			HiddenSize:           2,
		})

		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "output weight length must match vocab*hidden")
	})
	t.Run("small lm head tensor required", func(t *testing.T) {
		model := &hipLoadedModel{driver: &fakeHIPDriver{available: true}}
		_, err := model.loadedSmallLMHeadWeightsF32(hipLoadedSmallDecodeConfig{VocabSize: 1, HiddenSize: 1})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "LM head weights tensor is required")
	})
	t.Run("small lm head copy failure", func(t *testing.T) {
		model := &hipLoadedModel{driver: &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed")}}
		_, err := model.loadedSmallLMHeadWeightsF32(hipLoadedSmallDecodeConfig{
			LMHeadPointer: 0x1000,
			LMHeadBytes:   2,
			VocabSize:     1,
			HiddenSize:    1,
		})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "copy LM head weights")
	})
	t.Run("small lm head shape mismatch", func(t *testing.T) {
		payload, err := hipUint16Payload([]uint16{0x3c00})
		core.RequireNoError(t, err)
		driver, pointer := hipLoRAModelTestDevicePayload(t, payload)
		model := &hipLoadedModel{driver: driver}

		_, err = model.loadedSmallLMHeadWeightsF32(hipLoadedSmallDecodeConfig{
			LMHeadPointer: pointer,
			LMHeadBytes:   uint64(len(payload)),
			VocabSize:     2,
			HiddenSize:    2,
		})

		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "LM head weight length must match vocab*hidden")
	})
	t.Run("sequence classifier empty base weights", func(t *testing.T) {
		_, err := hipSequenceClassifierWeightsF32(hipLoadedSequenceClassifierWeights{})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "classifier base weights are required")
	})
	t.Run("sequence classifier f32 base weights must be finite", func(t *testing.T) {
		_, err := hipSequenceClassifierWeightsF32(hipLoadedSequenceClassifierWeights{F32: []float32{float32(math.NaN())}})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "classifier base weight values must be finite")
	})
	t.Run("sequence classifier fp16 base weights must be finite", func(t *testing.T) {
		_, err := hipSequenceClassifierWeightsF32(hipLoadedSequenceClassifierWeights{FP16: []uint16{0x7e00}})
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "classifier base weight values must be finite")
	})
	t.Run("loaded f32 tensor values must be finite", func(t *testing.T) {
		payload, err := hipFloat32Payload([]float32{float32(math.Inf(1))})
		core.RequireNoError(t, err)
		driver, pointer := hipLoRAModelTestDevicePayload(t, payload)
		model := &hipLoadedModel{driver: driver}

		_, err = model.loadedF32TensorPayload("rocm.hip.Test", "test tensor", pointer, uint64(len(payload)), 1)
		core.AssertError(t, err)
		core.AssertContains(t, err.Error(), "test tensor values must be finite")
	})
}

func hipLoRAModelTestDevicePayload(t *testing.T, payload []byte) (*fakeHIPDriver, nativeDevicePointer) {
	t.Helper()
	driver := &fakeHIPDriver{available: true}
	pointer, err := driver.Malloc(uint64(len(payload)))
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(pointer, payload))
	return driver, pointer
}
