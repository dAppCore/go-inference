// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

import "dappco.re/go/inference/model/gemma4/audio"

func newSystemHIPAudioGEMM() audio.GEMM { return nil }
