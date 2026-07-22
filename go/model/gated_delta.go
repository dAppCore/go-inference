// SPDX-Licence-Identifier: EUPL-1.2

package model

// gated_delta.go is the NEUTRAL home for the gated-delta (GatedDeltaNet linear-attention) mixer's
// config + weights — the data half of a named MixerKind (MixerGatedDelta), owned by the factory root
// like LoadedMoE/LoadedAudio, NOT by an arch package. It moved here from model/arch/Qwen/qwen3: the
// mixer is a GENERIC flash-linear-attention block (the causal conv is mamba2's, the recurrence is
// deltanet's — both shared primitives), so an arch only SELECTS it, it does not own it. qwen3 keeps a
// thin alias while its callers migrate; the alias + the mis-homed override are deleted once the arch
// factory builds and decodes the kind directly (retiring the composed engine — the parallel-engine fork, #50).

// GatedDeltaConfig is one gated-delta layer's geometry. The delta state is square, so KeyHeadDim ==
// ValueHeadDim == HeadDim; q/k use KeyHeads (GQA-repeated up to ValueHeads), v uses ValueHeads.
type GatedDeltaConfig struct {
	KeyHeads, ValueHeads, HeadDim, ConvKernel int
	Eps                                       float32
}

func (c GatedDeltaConfig) qDim() int    { return c.KeyHeads * c.HeadDim }
func (c GatedDeltaConfig) vDim() int    { return c.ValueHeads * c.HeadDim }
func (c GatedDeltaConfig) convDim() int { return 2*c.qDim() + c.vDim() } // q | k | v

// QDim, VDim and ConvDim are the exported projection-shape formulas a caller outside this package
// needs to target a gated-delta layer's input projections (in_proj_qkv/z/a/b) without duplicating the
// arithmetic.
func (c GatedDeltaConfig) QDim() int    { return c.qDim() }
func (c GatedDeltaConfig) VDim() int    { return c.vDim() }
func (c GatedDeltaConfig) ConvDim() int { return c.convDim() }

// GatedDeltaWeights is one layer's f32 weights (the loader widens the bf16 checkpoint). InProjQKV is
// [convDim, D]; ConvWeight [convDim, K]; InProjA/InProjB [ValueHeads, D]; ALog/DtBias [ValueHeads];
// InProjZ [vDim, D]; Norm [HeadDim] (per-value-head gated RMSNorm); OutProj [D, vDim].
type GatedDeltaWeights struct {
	InProjQKV  []float32
	ConvWeight []float32
	ConvBias   []float32
	InProjA    []float32
	ALog       []float32
	DtBias     []float32
	InProjB    []float32
	InProjZ    []float32
	Norm       []float32
	OutProj    []float32
	// Packed forms in a quant checkpoint (nil ⇒ the dense f32 field is used). The five projections
	// (in_proj_qkv/a/b/z + out_proj) dispatch to the quant matvec seam; the conv/A_log/norm/dt_bias stay
	// host f32 (small, unquantised) so the recurrent state math is exact.
	InProjQKVQ *QuantWeight
	InProjAQ   *QuantWeight
	InProjBQ   *QuantWeight
	InProjZQ   *QuantWeight
	OutProjQ   *QuantWeight
	// bf16-resident forms in a dense bf16 checkpoint (zero-copy views, never widened; nil ⇒ the f32 or
	// packed field is used). Same dispatch shape as the packed forms, through the bf16 matvec seam.
	InProjQKVB *BF16Weight
	InProjAB   *BF16Weight
	InProjBB   *BF16Weight
	InProjZB   *BF16Weight
	OutProjB   *BF16Weight
}
