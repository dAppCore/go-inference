// SPDX-Licence-Identifier: EUPL-1.2

package hip

import rocmmodel "dappco.re/go/inference/engine/hip/model"

const (
	ROCmModelPackFileManifestContract = rocmmodel.ModelPackFileManifestContract

	ROCmModelPackFilesStatusReady         = rocmmodel.ModelPackFilesStatusReady
	ROCmModelPackFilesStatusMissing       = rocmmodel.ModelPackFilesStatusMissing
	ROCmModelPackFilesStatusAmbiguousGGUF = rocmmodel.ModelPackFilesStatusAmbiguousGGUF

	ROCmModelPackFormatGGUF        = rocmmodel.ModelPackFormatGGUF
	ROCmModelPackFormatSafetensors = rocmmodel.ModelPackFormatSafetensors
	ROCmModelPackFormatMixed       = rocmmodel.ModelPackFormatMixed
	ROCmModelPackFormatMissing     = rocmmodel.ModelPackFormatMissing
)

type ROCmModelPackWeightFile = rocmmodel.ModelPackWeightFile
type ROCmModelPackFileManifest = rocmmodel.ModelPackFileManifest

// ResolveROCmModelRoot returns the directory that owns model metadata and
// weights. File paths resolve to their parent directory; directory paths resolve
// to themselves, matching go-mlx's backend-level model-root contract.
func ResolveROCmModelRoot(path string) (string, error) {
	return rocmmodel.ResolveModelPackRoot(path)
}

// InspectROCmModelPackFiles discovers local model-pack files without parsing
// tensor payloads. It is safe for CLI/API preflight before selecting a runtime.
func InspectROCmModelPackFiles(path string) (ROCmModelPackFileManifest, error) {
	return rocmmodel.InspectModelPackFiles(path)
}

// ROCmModelLoadWeightFiles returns the go-mlx-compatible preferred load-file
// set: all safetensors shards when present, a single GGUF when unambiguous, or
// an empty list for missing/ambiguous packs.
func ROCmModelLoadWeightFiles(path string) ([]ROCmModelPackWeightFile, error) {
	manifest, err := InspectROCmModelPackFiles(path)
	if err != nil {
		return nil, err
	}
	return manifest.LoadWeightFiles, nil
}

// ROCmModelLoadWeightPaths returns the preferred load-file paths for callers
// that do not need the full manifest.
func ROCmModelLoadWeightPaths(path string) ([]string, error) {
	manifest, err := InspectROCmModelPackFiles(path)
	if err != nil {
		return nil, err
	}
	return manifest.LoadWeightPaths(), nil
}
