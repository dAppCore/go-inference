// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import "dappco.re/go/inference/safetensors"

// DecodeFloat32 decodes a safetensors tensor's raw byte payload to float32
// values according to its dtype. Delegates to the safetensors format
// package, which owns the value codecs (F32 reinterpret, F16/BF16 upcast,
// F64 downcast).
//
//	info := tensors["model.embed_tokens.weight"]
//	raw := GetTensorData(info, data)
//	values, err := modelmgmt.DecodeFloat32(info.Dtype, raw, len(info.Shape))
//	if err != nil { return err }
func DecodeFloat32(dtype string, raw []byte, elements int) ([]float32, error) {
	return safetensors.DecodeFloat32(dtype, raw, elements)
}

// EncodeFloat32 encodes values as little-endian F32 safetensors bytes — the
// on-disk data-section layout WriteSafetensors expects for an "F32" tensor.
// Delegates to the safetensors format package.
//
//	tensorData["merged.weight"] = modelmgmt.EncodeFloat32(merged)
func EncodeFloat32(values []float32) []byte {
	return safetensors.EncodeFloat32(values)
}
