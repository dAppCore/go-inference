// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "dappco.re/go/inference/model"

// affineQMV is the no-cgo backend's affine quant compute — pkg/native's entry in
// the backend cross-section of the quant compute (pkg/model). It is a bf16-
// activation 4-bit (group-size/bits per call) decode matvec via QMVBF16
// (affine_qmv_bfloat16_t, driven directly, no mlx-c). Registered as backend
// "native", kind "affine"; the metal backend registers "metal"/"affine" the same
// way, so a model declaring quantization.kind="affine" decodes on either through
// one registry.
type affineQMV struct{}

func (affineQMV) Kind() string { return "affine" }
func (affineQMV) Bits() int    { return 0 } // the model's config declares the bit-width

func (affineQMV) MatVec(x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	return QMVBF16(x, packed, scales, biases, outDim, inDim, groupSize, bits)
}

func init() { model.RegisterBackendQuant("native", affineQMV{}) }
