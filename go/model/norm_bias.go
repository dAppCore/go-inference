// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"

	core "dappco.re/go"
)

// norm_bias.go folds the gemma-style "(1 + weight)" RMSNorm convention into the norm weight at load.
// gemma / gemma2 / gemma3 store the learned norm weight `w` and apply (1+w)·rms(x) at decode (metal
// precomputes NormScaled = AddScalar(weight, 1.0) and the gemma forward uses it); folding +1 here lets
// the engine's PLAIN RMSNorm kernel reproduce it byte-for-byte. An ArchSpec turns it on via
// WeightNames.NormBiasOne, so a gemma-family registration is the only place the convention lives — the
// shared Assemble and the decode stay plain. (gemma4 keeps the same convention; mistral does NOT, so
// the flag is opt-in per arch.)

// foldNormBiasOne returns the norm weight bytes with +1 added to every element, in the tensor's dtype.
// bf16 and f32 are supported (the dtypes gemma norms ship in); any other dtype is a loud error so the
// convention is never silently mis-applied.
func foldNormBiasOne(data []byte, dtype string) ([]byte, error) {
	out := make([]byte, len(data))
	switch dtype {
	case "BF16", "bfloat16":
		if len(data)%2 != 0 {
			return nil, core.NewError("model.foldNormBiasOne: bf16 byte length is odd")
		}
		for i := 0; i < len(data); i += 2 {
			b := uint16(data[i]) | uint16(data[i+1])<<8
			f := math.Float32frombits(uint32(b)<<16) + 1
			bits := math.Float32bits(f)
			r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) // round-to-nearest-even to bf16
			out[i], out[i+1] = byte(r), byte(r>>8)
		}
	case "F32", "float32":
		if len(data)%4 != 0 {
			return nil, core.NewError("model.foldNormBiasOne: f32 byte length not a multiple of 4")
		}
		for i := 0; i < len(data); i += 4 {
			bits := uint32(data[i]) | uint32(data[i+1])<<8 | uint32(data[i+2])<<16 | uint32(data[i+3])<<24
			nb := math.Float32bits(math.Float32frombits(bits) + 1)
			out[i], out[i+1], out[i+2], out[i+3] = byte(nb), byte(nb>>8), byte(nb>>16), byte(nb>>24)
		}
	default:
		return nil, core.NewError("model.foldNormBiasOne: unsupported norm dtype " + dtype)
	}
	return out, nil
}
