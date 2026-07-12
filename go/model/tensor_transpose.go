// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TransposeTensor2D converts the Hugging Face Conv1D [in,out] convention to Linear [out,in].
func TransposeTensor2D(t safetensors.Tensor) (safetensors.Tensor, error) {
	if len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 {
		return safetensors.Tensor{}, core.NewError("model.TransposeTensor2D: tensor must be 2-D")
	}
	width := len(t.Data) / (t.Shape[0] * t.Shape[1])
	if width <= 0 || width*t.Shape[0]*t.Shape[1] != len(t.Data) {
		return safetensors.Tensor{}, core.NewError("model.TransposeTensor2D: invalid payload size")
	}
	out := make([]byte, len(t.Data))
	for row := range t.Shape[0] {
		for col := range t.Shape[1] {
			src := (row*t.Shape[1] + col) * width
			dst := (col*t.Shape[0] + row) * width
			copy(out[dst:dst+width], t.Data[src:src+width])
		}
	}
	t.Data, t.Shape = out, []int{t.Shape[1], t.Shape[0]}
	return t, nil
}
