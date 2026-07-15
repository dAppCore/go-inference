// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

// Registers the architectures the native registry tests load (LoadDir / LoadTokenModelDir dispatch
// through model.LookupLoader). The library itself imports no arch — only this test binary and the
// serve cmd blank-import the loaders they need. Replaces the registration the deleted per-arch
// loaders used to pull in transitively.
import (
	_ "dappco.re/go/inference/model/gemma4"
	_ "dappco.re/go/inference/model/arch/mistralai/mistral"
)
