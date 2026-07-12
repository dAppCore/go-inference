// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !cgo && !rocm_legacy_server

package hip

func (unavailableHIPDriver) AudioGEMMAvailable() bool { return false }

func (unavailableHIPDriver) AudioMatMul([]float32, []float32, int, int, int, bool) ([]float32, error) {
	return nil, nil
}
