// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"context"

	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

// ROCmCacheProfileReporter exposes the live runtime cache profile used by
// reactive model-route consumers.
type ROCmCacheProfileReporter interface {
	CacheProfile(context.Context) (rocmmodel.CacheProfile, error)
}
