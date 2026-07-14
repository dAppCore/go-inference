// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "dappco.re/go/inference/model"

// archICBOpCaps are the engine-side inputs to the per-layer ICB op layout: the
// buffer-presence self-heals for hand-built callers that bypass model.Assemble
// (the slice-2/3 pattern), the stack-level op selections already resolved at the
// record boundary, and which fused kernels this build/GPU offers. Family knowledge
// never lands here — the declared per-layer selections come in on the
// model.LayerSpec slice.
type archICBOpCaps struct {
	// layer-0 norm weight buffers present — the self-heal a hand-built caller gets
	// when it did not declare the norm selections on its specs.
	qNormBuf, kNormBuf, postAttnBuf, postFFBuf bool
	// stack-level op selections resolved at the record boundary: the value-norm ones
	// weight was built (Arch.ValueNorm), the per-layer-input plan is present
	// (Arch.PerLayerInput*), a per-layer output scalar is declared or its buffer built
	// (LayerSpec.LayerScalar).
	valueNorm, ple, layerScalar bool
	// engine kernel availability: fused gelu(gate)·up (1 op vs the composed 10),
	// fused per-head QK-norm+RoPE, fused input-RMSNorm+projection recording, fused
	// residual-RMSNorm (post-norm + residual add in one op).
	fusedGELU, fusedQKRope, fusedRMSProj, fusedResRMS bool
}

// archICBOpLayout is the RESOLVED per-layer ICB op layout: which optional per-layer
// op selections the stack records (family-DECLARED on model.LayerSpec, with the
// buffer-presence self-heal) and what each layer therefore costs in ICB command
// slots — the one home both the ICB sizing and the record loop consume (#57).
//
// opsPerLayer is load-bearing beyond sizing: the fine-grained replay carves the
// recorded ICB into per-layer ranges of exactly this stride, so the record loop
// keeps the op count UNIFORM across layers by contract (sharer layers record
// discarded ops; global 2-pass SDPA and q8 store ops are accounted stack-globally
// in total, not per layer). The count arithmetic mirrors the record loop's emit
// sequence by construction; the recorded-count guard in decodeForwardArchICBCore
// fails loud if the two ever diverge.
type archICBOpLayout struct {
	hasQN, hasKN, hasPA, hasPF bool // per-layer norm ops the stack records
	// kRopeBindIdx is the buffer index the per-token rebind uses for the K cache-row
	// write: the plain rope op writes the cache at index 1; the fused kNorm+rope op
	// writes it at index 2 (its `out`).
	kRopeBindIdx uint
	opsPerLayer  int // ICB command slots per layer — the replay's per-layer carve stride
}

// resolveArchICBOpLayout resolves the per-layer op layout from the family-declared
// op selections (layer 0 speaks for the stack — the layout is uniform by contract)
// and the engine's capabilities. Pure: no Metal calls, unit-testable without a GPU.
func resolveArchICBOpLayout(specs []model.LayerSpec, caps archICBOpCaps) archICBOpLayout {
	lay := archICBOpLayout{
		hasQN:        specs[0].AttentionQNorm || caps.qNormBuf,
		hasKN:        specs[0].AttentionKNorm || caps.kNormBuf,
		hasPA:        specs[0].PostAttnNorm || caps.postAttnBuf,
		hasPF:        specs[0].PostFFNorm || caps.postFFBuf,
		kRopeBindIdx: 1,
	}
	extra := 0
	for _, h := range []bool{lay.hasQN, lay.hasKN, lay.hasPA, lay.hasPF} {
		if h {
			extra++
		}
	}
	if caps.valueNorm { // value-norm adds one op/layer (owner: the V row; sharer: discarded)
		extra++
	}
	ops := 24 + extra
	if caps.fusedGELU { // fused gelu is 1 command vs the composed chain's 10
		ops -= 9
	}
	// fused QK-norm+rope collapses (qNorm + ropeQ) and (kNorm + ropeK) from 2 ops to
	// 1 each when the layer has QK-norm; the fused K op moves the cache-row bind to 2.
	if caps.fusedQKRope && lay.hasQN {
		ops-- // qNorm+ropeQ
	}
	if caps.fusedQKRope && lay.hasKN {
		ops-- // kNorm+ropeK
		lay.kRopeBindIdx = 2
	}
	if caps.ple {
		if caps.fusedGELU {
			ops += 5 // gate matmul, fused gelu*pli, proj matmul, rms, residual add
		} else {
			ops += 14 // gate matmul, 10-op gelu*pli chain, proj matmul, rms, residual add
		}
	}
	if caps.layerScalar { // one multiply on the layer's residual output
		ops++
	}
	// fused input-RMSNorm+projection folds the attn-input rms and the mlp-input rms
	// INTO their following projections (Q/K/V read inBuf+attnNormW; gate/up read
	// hBuf+mlpNormW), removing both setRMS ops.
	if caps.fusedRMSProj {
		ops -= 2
	}
	// fused residual-RMSNorm folds each post-norm + its residual add into one op
	// (out = res + rms(branch)).
	if caps.fusedResRMS {
		if lay.hasPA {
			ops--
		}
		if lay.hasPF {
			ops--
		}
	}
	lay.opsPerLayer = ops
	return lay
}

// total is the whole-stack ICB command count: the uniform per-layer stride plus the
// stack-global additions — each GLOBAL layer recording the 2-pass SDPA costs one op
// over the single-pass layout, and each q8 owner layer records two quantise-store
// ops (K row, V row).
func (l archICBOpLayout) total(nLayers, nGlobal2Pass, nQ8Store int) int {
	return l.opsPerLayer*nLayers + nGlobal2Pass + nQ8Store
}
