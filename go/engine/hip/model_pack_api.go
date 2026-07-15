// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	"dappco.re/go/inference"
)

// InspectModelPack validates a local model pack without loading tensors.
func InspectModelPack(ctx context.Context, path string) (*inference.ModelPackInspection, error) {
	return (&rocmBackend{}).InspectModelPack(ctx, path)
}
