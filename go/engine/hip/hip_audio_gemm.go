// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"unsafe"

	"dappco.re/go/inference/model/gemma4/audio"
)

type nativeHIPAudioGEMM interface {
	AudioGEMMAvailable() bool
	AudioMatMul(a, b []float32, m, k, n int, transposeB bool) ([]float32, error)
}

type hipAudioGEMM struct {
	driver nativeHIPAudioGEMM
}

func newHIPAudioGEMM(driver nativeHIPDriver) audio.GEMM {
	gemm, ok := driver.(nativeHIPAudioGEMM)
	if !ok || !driver.Available() || !gemm.AudioGEMMAvailable() {
		return nil
	}
	return hipAudioGEMM{driver: gemm}
}

func newSystemHIPAudioGEMM() audio.GEMM {
	return newHIPAudioGEMM(newSystemHIPDriver())
}

func (gemm hipAudioGEMM) MatMul(a, b []float32, m, k, n int, transposeB bool) ([]float32, bool) {
	out, err := gemm.driver.AudioMatMul(a, b, m, k, n, transposeB)
	return out, err == nil
}

func hipAudioFloat32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
}
