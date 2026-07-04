// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"

const (
	Gemma4RuntimeMLXAffine    = modelgemma4.RuntimeMLXAffine
	Gemma4RuntimeBF16         = modelgemma4.RuntimeBF16
	Gemma4RuntimeGGUF         = modelgemma4.RuntimeGGUF
	Gemma4RuntimePlanned      = modelgemma4.RuntimePlanned
	Gemma4GenerateLinked      = modelgemma4.GenerateLinked
	Gemma4GenerateLoadOnly    = modelgemma4.GenerateLoadOnly
	Gemma4GeneratePlannedOnly = modelgemma4.GeneratePlannedOnly
)

type Gemma4SizeQuantSupport = modelgemma4.SizeQuantSupport

type Gemma4QuantModeSupport = modelgemma4.QuantModeSupport

func DefaultGemma4SizeQuantSupport() []Gemma4SizeQuantSupport {
	return modelgemma4.DefaultSizeQuantSupport()
}

func Gemma4SizeQuantSupportBySize(size string) (Gemma4SizeQuantSupport, bool) {
	return modelgemma4.SizeQuantSupportBySize(size)
}

func Gemma4QuantModeSupportBySize(size, mode string) (Gemma4QuantModeSupport, bool) {
	return modelgemma4.QuantModeSupportBySize(size, mode)
}
