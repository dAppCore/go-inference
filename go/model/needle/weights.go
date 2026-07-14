// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// Weights is a checkpoint's tensors widened to f32 and keyed by their
// safetensors name (e.g. "model.encoder.layers.0.self_attn.q_proj.weight").
// The reference trades the memory of full-f32 weights for a single, obvious
// numeric path; a 26M-param model is a few hundred MB of f32, which is fine for
// a host oracle.
type Weights struct {
	data   map[string][]float32
	shapes map[string][]int
}

// widenBF16 converts little-endian bf16 bytes to f32. bf16 is the top 16 bits of
// an f32, so the widen is a left shift by 16 with no rounding — exact.
//
//	widenBF16([]byte{0x80, 0x3f}) // []float32{1}
func widenBF16(raw []byte) []float32 {
	n := len(raw) / 2
	out := make([]float32, n)
	for i := range n {
		bits := uint16(raw[2*i]) | uint16(raw[2*i+1])<<8
		out[i] = math.Float32frombits(uint32(bits) << 16)
	}
	return out
}

// widenF32 reinterprets little-endian f32 bytes as []float32.
func widenF32(raw []byte) []float32 {
	n := len(raw) / 4
	out := make([]float32, n)
	for i := range n {
		out[i] = math.Float32frombits(
			uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
	}
	return out
}

// LoadWeights reads a safetensors file and widens every tensor to f32. Only the
// bf16 and f32 dtypes Needle uses are accepted; any other dtype is a loud error
// rather than a silently mis-read tensor.
//
//	w, err := needle.LoadWeights("/models/needle/model.safetensors")
func LoadWeights(path string) (*Weights, error) {
	tensors, err := safetensors.Load(path)
	if err != nil {
		return nil, core.E("needle.LoadWeights", "load "+path, err)
	}
	w := &Weights{
		data:   make(map[string][]float32, len(tensors)),
		shapes: make(map[string][]int, len(tensors)),
	}
	for name, t := range tensors {
		switch t.Dtype {
		case "BF16":
			w.data[name] = widenBF16(t.Data)
		case "F32":
			w.data[name] = widenF32(t.Data)
		default:
			return nil, core.E("needle.LoadWeights", "unsupported dtype "+t.Dtype+" for "+name, nil)
		}
		w.shapes[name] = t.Shape
	}
	return w, nil
}

// get returns a tensor by name, or nil if absent. Callers that need a tensor to
// exist should route through mustGet.
func (w *Weights) get(name string) []float32 { return w.data[name] }

// mustGet returns a tensor by name and reports whether it was present, so the
// forward pass fails loudly on a missing weight rather than reading zeros.
func (w *Weights) mustGet(name string) ([]float32, bool) {
	v, ok := w.data[name]
	return v, ok
}

// shape returns a tensor's dimensions, or nil if absent.
//
//	w.shape("model.embed_tokens.weight") // []int{8192, 512}
func (w *Weights) shape(name string) []int { return w.shapes[name] }
