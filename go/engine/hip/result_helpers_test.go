// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64

package hip

import (
	core "dappco.re/go"
)

// resultValue unwraps a core.Result back into the (value, error) shape the
// test suite was written against before LoadModel/Classify/BatchGenerate
// migrated to core.Result (see native.go). Kept test-side only so migrated
// call sites read identically to their pre-migration form; production code
// uses r.OK/r.Value directly.
//
//	results, err := resultValue[[]inference.ClassifyResult](model.Classify(ctx, prompts))
func resultValue[T any](r core.Result) (T, error) {
	v, ok := core.Cast[T](r)
	if !ok {
		return v, resultError(r)
	}
	return v, nil
}
