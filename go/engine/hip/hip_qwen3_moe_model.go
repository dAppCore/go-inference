// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"

	core "dappco.re/go"
	sharedmodel "dappco.re/go/inference/model"
)

// hipQwen3MoEModel implements the shared model.TokenModel contract (Embedder + Backend +
// LMHead + Vocab) over the qwen3_moe device-resident weight set. It deliberately does
// NOT implement model.SessionModel: a persistent incremental KV cache across
// DecodeForward calls is a performance refinement out of this v1's scope (see
// hip_qwen3_moe_layer.go's doc) — the shared model.Generate family already falls back to
// the whole-sequence path automatically for a plain TokenModel (model/token.go's
// generateUntilTransform), the same fallback every non-SessionModel backend gets.
type hipQwen3MoEModel struct {
	driver  nativeHIPDriver
	cfg     hipQwen3MoEConfig
	weights *hipQwen3MoEWeights
}

var (
	_ sharedmodel.Embedder    = (*hipQwen3MoEModel)(nil)
	_ sharedmodel.Backend     = (*hipQwen3MoEModel)(nil)
	_ sharedmodel.LMHead      = (*hipQwen3MoEModel)(nil)
	_ sharedmodel.TokenModel  = (*hipQwen3MoEModel)(nil)
)

// Embed looks up token id's row in the host-resident embedding table — a plain slice
// copy, never a kernel launch (the table is not device-resident; see
// hipQwen3MoEWeights' doc).
func (m *hipQwen3MoEModel) Embed(id int32) ([]byte, error) {
	if m == nil || m.weights == nil {
		return nil, core.NewError("rocm.hip.Qwen3MoE: model is not initialised")
	}
	if id < 0 || int(id) >= m.cfg.VocabSize {
		return nil, core.NewError("rocm.hip.Qwen3MoE: token id out of vocabulary range")
	}
	start := int(id) * m.cfg.HiddenSize
	row := m.weights.Embed[start : start+m.cfg.HiddenSize]
	return hipQwen3MoEFloat32ToBF16(row), nil
}

// DecodeForward runs T input embeddings through the full qwen3_moe layer stack —
// whole-sequence, rebuilding the KV cache for this call (see hipQwen3MoEHiddenForward).
func (m *hipQwen3MoEModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	if m == nil || m.weights == nil {
		return nil, core.NewError("rocm.hip.Qwen3MoE: model is not initialised")
	}
	if len(inputs) == 0 {
		return nil, core.NewError("rocm.hip.Qwen3MoE: empty input sequence")
	}
	hidden := make([][]float32, len(inputs))
	for i, in := range inputs {
		if len(in) != m.cfg.HiddenSize*2 {
			return nil, core.NewError("rocm.hip.Qwen3MoE: input embedding length must be hidden size bf16 bytes")
		}
		hidden[i] = hipQwen3MoEBF16ToFloat32(in)
	}
	if err := hipQwen3MoEHiddenForward(context.Background(), m.driver, m.cfg, m.weights, hidden); err != nil {
		return nil, err
	}
	out := make([][]byte, len(hidden))
	for i, h := range hidden {
		out[i] = hipQwen3MoEFloat32ToBF16(h)
	}
	return out, nil
}

// Head maps a final hidden state to vocab logits: final norm + LM head projection.
// qwen3_moe declares no logit soft-cap.
func (m *hipQwen3MoEModel) Head(hidden []byte) ([]byte, error) {
	if m == nil || m.weights == nil {
		return nil, core.NewError("rocm.hip.Qwen3MoE: model is not initialised")
	}
	if len(hidden) != m.cfg.HiddenSize*2 {
		return nil, core.NewError("rocm.hip.Qwen3MoE: hidden state length must be hidden size bf16 bytes")
	}
	h := hipQwen3MoEBF16ToFloat32(hidden)
	normed, err := hipRunRMSNormKernelWithDeviceWeight(context.Background(), m.driver, h, m.weights.FinalNorm.Pointer(), m.weights.FinalNorm.SizeBytes(), m.cfg.HiddenSize, m.cfg.Epsilon)
	if err != nil {
		return nil, err
	}
	logits, err := hipRunProjectionKernelWithDeviceWeightEncoding(context.Background(), m.driver, normed, m.weights.LMHead.Pointer(), m.weights.LMHead.SizeBytes(), m.cfg.VocabSize, m.cfg.HiddenSize, hipProjectionWeightEncodingF32)
	if err != nil {
		return nil, err
	}
	return hipQwen3MoEFloat32ToBF16(logits), nil
}

// Vocab reports the model's vocabulary size.
func (m *hipQwen3MoEModel) Vocab() int {
	if m == nil {
		return 0
	}
	return m.cfg.VocabSize
}

// Close releases the model's device-resident weights.
func (m *hipQwen3MoEModel) Close() error {
	if m == nil {
		return nil
	}
	m.weights.Close()
	return nil
}

// hipQwen3MoEFloat32ToBF16 encodes host float32 values as row-major bf16 bytes — the
// model.TokenModel seam contract ("everything crosses the seam as bf16 []byte",
// model/token.go). Truncates each value's low mantissa bits (the standard bf16 encode:
// keep the top 16 bits of the IEEE-754 float32 bit pattern).
func hipQwen3MoEFloat32ToBF16(values []float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		u := uint16(math.Float32bits(v) >> 16)
		out[2*i] = byte(u)
		out[2*i+1] = byte(u >> 8)
	}
	return out
}

// hipQwen3MoEBF16ToFloat32 decodes row-major bf16 bytes to host float32 values — the
// inverse of hipQwen3MoEFloat32ToBF16 (widen: left-shift the bf16 bit pattern into the
// top 16 bits of a float32, low mantissa bits zero).
func hipQwen3MoEBF16ToFloat32(data []byte) []float32 {
	out := make([]float32, len(data)/2)
	for i := range out {
		u := uint16(data[2*i]) | uint16(data[2*i+1])<<8
		out[i] = math.Float32frombits(uint32(u) << 16)
	}
	return out
}
