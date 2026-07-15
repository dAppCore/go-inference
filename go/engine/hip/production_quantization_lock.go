// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

const officialGemma4E2BLicenceURL = "https://ai.google.dev/gemma/docs/gemma_4_license"

// ProductionQuantizationFileLock pins one file inside a quantized target pack.
// BlobID records the Hugging Face cache/git blob identity; SHA256 is the
// content hash used for local verification.
type ProductionQuantizationFileLock struct {
	Name   string `json:"name"`
	BlobID string `json:"blob_id,omitempty"`
	SHA256 string `json:"sha256"`
	Bytes  uint64 `json:"bytes,omitempty"`
}

// ProductionQuantizationPackLock records MLX-community Gemma 4 E2B derivatives
// that sit beside the official Google E2B source locks. These are not a
// promotion signal; they make the app quantization ladder and bench/R&D pack
// matrix auditable for the ROCm runtime.
type ProductionQuantizationPackLock struct {
	Name              string `json:"name"`
	ModelID           string `json:"model_id"`
	Revision          string `json:"revision"`
	SourceCheckedAt   string `json:"source_checked_at"`
	SourceURL         string `json:"source_url"`
	BaseModelID       string `json:"base_model_id"`
	BaseRevision      string `json:"base_revision"`
	ConversionTool    string `json:"conversion_tool"`
	ConversionCommand string `json:"conversion_command"`
	AccuracySmoke     string `json:"accuracy_smoke"`
	Licence           string `json:"licence"`
	LicenceURL        string `json:"licence_url"`

	QuantBits  int    `json:"quant_bits"`
	QuantGroup int    `json:"quant_group"`
	QuantMode  string `json:"quant_mode"`

	ReadmeBlobID            string                           `json:"readme_blob_id,omitempty"`
	ReadmeSHA256            string                           `json:"readme_sha256"`
	ConfigBlobID            string                           `json:"config_blob_id,omitempty"`
	ConfigSHA256            string                           `json:"config_sha256"`
	ProcessorConfigBlobID   string                           `json:"processor_config_blob_id,omitempty"`
	ProcessorConfigSHA256   string                           `json:"processor_config_sha256"`
	TokenizerBlobID         string                           `json:"tokenizer_blob_id,omitempty"`
	TokenizerSHA256         string                           `json:"tokenizer_sha256"`
	TokenizerConfigBlobID   string                           `json:"tokenizer_config_blob_id,omitempty"`
	TokenizerConfigSHA256   string                           `json:"tokenizer_config_sha256"`
	GenerationConfigBlobID  string                           `json:"generation_config_blob_id,omitempty"`
	GenerationConfigSHA256  string                           `json:"generation_config_sha256"`
	ChatTemplateBlobID      string                           `json:"chat_template_blob_id,omitempty"`
	ChatTemplateSHA256      string                           `json:"chat_template_sha256"`
	SafetensorsIndexPresent bool                             `json:"safetensors_index_present"`
	SafetensorsIndexBlobID  string                           `json:"safetensors_index_blob_id,omitempty"`
	SafetensorsIndexSHA256  string                           `json:"safetensors_index_sha256"`
	SafetensorsIndexBytes   uint64                           `json:"safetensors_index_bytes,omitempty"`
	WeightFiles             []ProductionQuantizationFileLock `json:"weight_files"`
}

// DefaultProductionQuantizationPackLocks returns the local MLX-community
// derivatives that back the app-facing Gemma 4 E2B quantization ladder plus the
// planned ROCm FP research packs.
func DefaultProductionQuantizationPackLocks() []ProductionQuantizationPackLock {
	locks := []ProductionQuantizationPackLock{
		productionQuantizationPackLock(productionQuantizationPackLockInput{
			name: "research-mxfp4", modelID: "mlx-community/gemma-4-e2b-it-mxfp4",
			revision: "6505f8b409be66c5a6d767e21b7d2bed277fcaa4", bits: 4, group: 32, mode: "mxfp4",
			command: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-mxfp4 (MXFP4; exact upstream conversion flags not recorded)",
			smoke:   "bench/R&D lock only; MXFP4 remains a research pack until retained-workflow quality and memory evidence promote it",
			readme:  "a77b4db96f0e1067216103be91d53b544c7e96bae001736226a2a15fa851be82",
			config:  "614e876b4efcaff13ce4c7a3f96a5b9de86325e3d2ab9c622606ced688f1b8b7",
			index:   "682ab3c507de77072844c5dff4fbb35dfa46fec9fc4b6f3ae014b3f42e78d51b", indexBytes: 211538,
			weights: []ProductionQuantizationFileLock{{Name: "model.safetensors", BlobID: "d9209536088aa473de0f28bc5d590a15f2af845d59b32e38bbb0a45e8750889c", SHA256: "d9209536088aa473de0f28bc5d590a15f2af845d59b32e38bbb0a45e8750889c", Bytes: 4263396466}},
		}),
		productionQuantizationPackLock(productionQuantizationPackLockInput{
			name: "research-mxfp8", modelID: "mlx-community/gemma-4-e2b-it-mxfp8",
			revision: "58034520e7459bf1e5be508e46906aa943683ee4", bits: 8, group: 32, mode: "mxfp8",
			command: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-mxfp8 (MXFP8; exact upstream conversion flags not recorded)",
			smoke:   "bench/R&D lock only; MXFP8 remains a research pack until retained-workflow quality and memory evidence promote it",
			readme:  "e26522311415e53896517e66fe70be411012327cc5275e48067170119dc07756",
			config:  "d6be5b24cbc974d492804737716ade8d2575eb849ec90a1d316bb64e99838104",
			index:   "3dd5efc67da447bc266f6f9e727450b54377cb8563181a947ff727dbf9d1eae1", indexBytes: 237768,
			weights: []ProductionQuantizationFileLock{
				{Name: "model-00001-of-00002.safetensors", BlobID: "d6e4ec568ad5301f74e46772b745aeeffedf4f4cc3f87e2eeeab5e0cba812592", SHA256: "d6e4ec568ad5301f74e46772b745aeeffedf4f4cc3f87e2eeeab5e0cba812592", Bytes: 5367071866},
				{Name: "model-00002-of-00002.safetensors", BlobID: "56ab229f33c37fc325c6c07cad8bbf87e3306ead53b90f36ebf34a1353530629", SHA256: "56ab229f33c37fc325c6c07cad8bbf87e3306ead53b90f36ebf34a1353530629", Bytes: 387549560},
			},
		}),
		productionQuantizationPackLock(productionQuantizationPackLockInput{
			name: "quality", modelID: "mlx-community/gemma-4-e2b-it-8bit",
			revision: "48ef0737faea4e72556670e49da0ba421027a545", bits: ProductionLaneQualityQuantBits, group: 64, mode: "affine",
			command: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-8bit --q-bits 8 --q-group-size 64",
			smoke:   "metadata lock only; official target native-load, retained-state, and long-output quality gates remain pending",
			readme:  "306177431807e9ff28450b718b022ce411c422f34d44e8d64461901b99beb13d",
			config:  "5cdd5627ab3ecf52086cc79b2c14c45a277d273069f1d73bf17a3a5136afe3db",
			index:   "cba1620cfe01e35a14cbebddcc32415d55292529795565d1d11e9cb9cf669f50", indexBytes: 270064,
			weights: []ProductionQuantizationFileLock{
				{Name: "model-00001-of-00002.safetensors", BlobID: "fe889fb027f0b79758af4a7da6a27c6c7bc715680bbdd5af9797bd8355d86820", SHA256: "fe889fb027f0b79758af4a7da6a27c6c7bc715680bbdd5af9797bd8355d86820", Bytes: 5367135201},
				{Name: "model-00002-of-00002.safetensors", BlobID: "83bb2a3420d473d416ffcb3cf9c93bacce064981fb22ea20cb6111a178d2679b", SHA256: "83bb2a3420d473d416ffcb3cf9c93bacce064981fb22ea20cb6111a178d2679b", Bytes: 532432577},
			},
		}),
		productionQuantizationPackLock(productionQuantizationPackLockInput{
			name: "default", modelID: ProductionLaneModelID,
			revision: "40d43b05f94ee798c0e40fe19fcd9ef49928486b", bits: ProductionLaneProductDefaultQuantBits, group: 64, mode: "affine",
			command: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-6bit --q-bits 6 --q-group-size 64",
			smoke:   "metadata lock only; official target native-load, retained-state, and long-output quality gates remain pending",
			readme:  "9293f5a79db1e170557902c0a7b87d309a8f70c28be42f3a298ee6f2ce006ca4",
			config:  "32e50a33a18172e79c86b7a78aff7e79c7544031199d672a2a65e526a8bf0199",
			index:   "7e6bdf16f05a9d296179d9fe93ae18b52177e84a6e78d46f126e2fa6f6b02414", indexBytes: 230329,
			weights: []ProductionQuantizationFileLock{{Name: "model.safetensors", BlobID: "1ce6f5c8d5daf306e71824cfc752020b70fc9262ff201a577d18d62cc446d5bc", SHA256: "1ce6f5c8d5daf306e71824cfc752020b70fc9262ff201a577d18d62cc446d5bc", Bytes: 4740335854}},
		}),
		productionQuantizationPackLock(productionQuantizationPackLockInput{
			name: "constrained", modelID: ProductionLaneArchivedBaselineModelID,
			revision: "99d9a53ff828d365a8ecae538e45f80a08d612cd", bits: ProductionLaneConstrainedQuantBits, group: 64, mode: "affine",
			command: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-4bit --q-bits 4 --q-group-size 64",
			smoke:   "archived q4 control; historical retained-state benchmark baseline accepted before official q6/q8 promotion",
			readme:  "0d0e79f7c5427656411c4ce41fb2a69889bd4f5011ef1885a3b8af9cf6ce8167",
			config:  "6d12c87861fff3871d3a745011b0d852be6513f3ce594ae1e8d643dae9d3b9a8",
			index:   "a8aa7359c747a0d59368dbff9a1029da86bda139ccc0ae1f1e938db75de7d5ce", indexBytes: 230329,
			weights: []ProductionQuantizationFileLock{{Name: "model.safetensors", BlobID: "e9bea0584546fafb5ff83a1132a6c4662a8498cc6a5bcda52fc6ca562b7bafab", SHA256: "e9bea0584546fafb5ff83a1132a6c4662a8498cc6a5bcda52fc6ca562b7bafab", Bytes: 3581101896}},
		}),
		productionQuantizationPackLock(productionQuantizationPackLockInput{
			name: "quality-control-bf16", modelID: "mlx-community/gemma-4-e2b-it-bf16",
			revision: "22a2753af6114b0c364f09921771b458e40b9e09", bits: 16, group: 0, mode: "bf16",
			command: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-bf16",
			smoke:   "quality-control lock only; BF16 is the unquantised comparison target and requires native validation before promotion",
			readme:  "157c751ee86bfe06c986860228d6500d2719a36d8696d43e166279eed67a6c50",
			config:  "29b810ed760b55104943a3cc3b6f8b9ca079e6e00b09585d85aec54863a42fb4",
			index:   "3c147c85c7d2d964452007af9056a78c0ca916dffc06fec1e7c218f28b30bd4f", indexBytes: 205473,
			weights: []ProductionQuantizationFileLock{
				{Name: "model-00001-of-00003.safetensors", BlobID: "ff4c28c7f1b0a841697cdd10fc7b45d434c2edeb6e02360e8a56ed88fa7b1cef", SHA256: "ff4c28c7f1b0a841697cdd10fc7b45d434c2edeb6e02360e8a56ed88fa7b1cef", Bytes: 4569831590},
				{Name: "model-00002-of-00003.safetensors", BlobID: "b2d44b0ee3454db90d6d10b4006b0270be0729094809570c9b366f3a35ca7655", SHA256: "b2d44b0ee3454db90d6d10b4006b0270be0729094809570c9b366f3a35ca7655", Bytes: 5366705230},
				{Name: "model-00003-of-00003.safetensors", BlobID: "2fb5cbee871ebe7dcfaebef771c3013dd6cee51d9c8e0023d5d7c32cb0e9e244", SHA256: "2fb5cbee871ebe7dcfaebef771c3013dd6cee51d9c8e0023d5d7c32cb0e9e244", Bytes: 310074804},
			},
		}),
	}
	return cloneProductionQuantizationPackLocks(locks)
}

type productionQuantizationPackLockInput struct {
	name, modelID, revision string
	bits, group             int
	mode, command, smoke    string
	readme, config, index   string
	indexBytes              uint64
	weights                 []ProductionQuantizationFileLock
}

func productionQuantizationPackLock(input productionQuantizationPackLockInput) ProductionQuantizationPackLock {
	tokenizerSHA := "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
	tokenizerConfigSHA := "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df"
	generationConfigSHA := "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de"
	chatTemplateSHA := "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082"
	if input.modelID == ProductionLaneArchivedBaselineModelID {
		chatTemplateSHA = "781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf"
	}
	return ProductionQuantizationPackLock{
		Name:              input.name,
		ModelID:           input.modelID,
		Revision:          input.revision,
		SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
		SourceURL:         "https://huggingface.co/" + input.modelID,
		BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
		BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
		ConversionTool:    "mlx-vlm 0.4.3",
		ConversionCommand: input.command,
		AccuracySmoke:     input.smoke,
		Licence:           "apache-2.0",
		LicenceURL:        officialGemma4E2BLicenceURL,
		QuantBits:         input.bits,
		QuantGroup:        input.group,
		QuantMode:         input.mode,

		ReadmeSHA256:            input.readme,
		ConfigSHA256:            input.config,
		ProcessorConfigSHA256:   "1bd0d00776284f369c1eff5fb631e865dfcdca861e0b7d60dbef27fcf37436a8",
		TokenizerSHA256:         tokenizerSHA,
		TokenizerConfigSHA256:   tokenizerConfigSHA,
		GenerationConfigSHA256:  generationConfigSHA,
		ChatTemplateSHA256:      chatTemplateSHA,
		SafetensorsIndexPresent: true,
		SafetensorsIndexSHA256:  input.index,
		SafetensorsIndexBytes:   input.indexBytes,
		WeightFiles:             append([]ProductionQuantizationFileLock(nil), input.weights...),
	}
}

func cloneProductionQuantizationPackLocks(locks []ProductionQuantizationPackLock) []ProductionQuantizationPackLock {
	clone := make([]ProductionQuantizationPackLock, len(locks))
	for i, lock := range locks {
		clone[i] = lock
		clone[i].WeightFiles = append([]ProductionQuantizationFileLock(nil), lock.WeightFiles...)
	}
	return clone
}
