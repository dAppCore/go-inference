// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package tui

// The engine registration lives in the lem main package; the test binary
// links it here so the live drive can load a real checkpoint.
import (
	_ "dappco.re/go/inference/engine/metal"
	_ "dappco.re/go/inference/model/builtin"
)
