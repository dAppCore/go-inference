// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// tq_kv_state.go — the STATE-lane TurboQuant KV carrier (#48 follow-on,
// docs/design-tq-moe-hybrid.md): TQ for sessions the arch ICB cannot record.
// A MoE stack (host router mid-token) or a hybrid stack (gated-delta host
// recurrence) decodes via stepToken; its qualifying GLOBAL attention owners
// hold the SAME codes+γ caches the recorded lane uses, written and read per
// token by encAttnHalfKVTQ through the live encoder — the same #41/#48
// kernels, the same wire format (turboquant_device.go stays the format
// authority), no ICB. MoE is an FFN property, not a cache kind: the layer's
// attention side is standard K/V rows.
//
// Cache-kind selection is per LAYER KIND, never per arch family:
//   - GLOBAL attention owner, geometry OK, un-gated, un-shared, no sinks → TQ
//   - sliding owners → native bf16 ring (bounded residency — nothing to win)
//   - sharers → owner forced native (v1: the shared-attention emitters
//     encAttnHalfShared/…Paged are not TQ-wired on this lane)
//   - MixerGatedDelta → native recurrent state (no KV rows exist)
//   - gated full attention (attn_output_gate) → arch-wide decline upstream
//     (its KV lives in the gated/fused lane's resident state, not lb caches)
//
// The paged pool is declined wholesale for an armed state carrier (the
// attention-sinks precedent): paged TQ would need a per-page code-addressing
// kernel family that does not exist. Sequential prefill rides stepToken and
// is TQ-correct by construction; the batched dense pass declines up front.

// archStateKVTQ is the state-owned TurboQuant carrier: the per-layer
// enablement/γ/stride set (the archICBKVTQ shape, shared with the recorded
// carrier so addressing is uniform), the code caches, and the fixed staging +
// 2-pass scratch the per-token attention half encodes through.
type archStateKVTQ struct {
	set     archICBKVTQ
	kCaches []metal.MTLBuffer // per-layer packed code caches (TQ owners only; nil elsewhere)
	vCaches []metal.MTLBuffer
	// kStage/vStage: the fixed bf16 staging rows (largest enabled kv geometry).
	// K rope/norm and the V projection land here; one lthn_tq_kv_store each
	// rotates+quantises staging → the code cache row + γ row at pos.
	kStage, vStage metal.MTLBuffer
	// qRot: the 2-pass lane's Πq scratch (largest q geometry). nil below the
	// knee — the single-pass kernel fuses the q rotation itself (#48).
	qRot metal.MTLBuffer
}

func (t *archStateKVTQ) on(li int) bool { return t != nil && t.set.on(li) }

func (t *archStateKVTQ) any() bool { return t != nil && t.set.any() }

// release frees the carrier's own device buffers. Π and centroid tables are
// process-memoised and shared across sessions — never released here (the
// releaseTQPlanes contract).
func (t *archStateKVTQ) release() {
	if t == nil {
		return
	}
	releaseDeviceBuffers(t.kCaches...)
	releaseDeviceBuffers(t.vCaches...)
	releaseDeviceBuffers(t.set.kGammas...)
	releaseDeviceBuffers(t.set.vGammas...)
	releaseDeviceBuffers(t.kStage, t.vStage, t.qRot)
	for i := range t.kCaches {
		t.kCaches[i], t.vCaches[i] = nil, nil
	}
	for i := range t.set.kGammas {
		t.set.kGammas[i], t.set.vGammas[i] = nil, nil
	}
	t.kStage, t.vStage, t.qRot = nil, nil, nil
	for i := range t.set.enabled {
		t.set.enabled[i] = false
	}
}

// tqStateArmed reports whether this state carries the state-lane TQ set — the
// per-layer branch gate in stepTokenEncode and the wholesale decline gate for
// the paged pool and the batched dense pass.
func (s *archDecodeState) tqStateArmed() bool { return s != nil && s.kvTQState.any() }

// hasKVTQAny reports whether EITHER TurboQuant carrier is armed on this
// session — the recorded-ICB set or the state-lane set. Every lane that
// declines TQ (prompt reuse, CaptureKV, snapshot views, laneSet) gates here
// so a second carrier can never slip past a decline written for the first.
func (s *ArchSession) hasKVTQAny() bool {
	if s == nil {
		return false
	}
	return s.state.icb.hasKVTQ() || s.state.tqStateArmed()
}

// archSpecsRequireStepToken reports whether the SPECS alone disqualify the
// recorded arch ICB — a MoE layer (host router) or a gated-delta mixer (host
// recurrence). These stacks decode via stepToken, so a TurboQuant request
// arms the STATE carrier instead of the recorded one. Mirrors the spec-level
// half of ArchSession.icbEligible (trace, the session-level half, is checked
// by the constructor).
func archSpecsRequireStepToken(specs []model.LayerSpec) bool {
	for li := range specs {
		if specs[li].MoE || specs[li].Mixer == model.MixerGatedDelta {
			return true
		}
	}
	return false
}

// allocArchStateKVTQ builds the state-lane carrier for an armed TQ mode over
// stepToken-decoded specs: qualifying GLOBAL owners allocate code caches + γ
// planes (maxLen rows — TQ owners are global-only, never ring-sized), every
// other owner keeps its native lb path untouched. Returns nil when no layer
// qualifies — the constructor refuses loudly rather than pretending (the
// allocArchICBCachesTQ idiom). Store/read pipelines for every enabled
// geometry are resolved HERE so an unservable metallib fails the session
// build, not the first decode step.
func allocArchStateKVTQ(specs []model.LayerSpec, lb []archLayerBufs, nHeads, nKVHeads, headDim, maxLen int, attnOutputGate bool, tq *tqKVConfig) (*archStateKVTQ, error) {
	if tq == nil || attnOutputGate {
		// gated full attention keeps its own lane-resident KV — arming TQ under
		// it would put a codes cache behind a lane that reads bf16 (upstream
		// tqKVArchServable already refused; this is the construction belt).
		return nil, nil
	}
	// ANY sharer forces its owner native on this lane: the shared-attention
	// emitters are not TQ-wired (v1) — a sharer reading an owner's packed
	// codes as bf16 rows is the exact mixed-cache failure this file forbids.
	shared := make([]bool, len(specs))
	for li := range specs {
		if !specs[li].OwnsCache() && specs[li].Mixer == model.MixerAttention &&
			specs[li].KVShareFrom != li && specs[li].KVShareFrom >= 0 && specs[li].KVShareFrom < len(specs) {
			shared[specs[li].KVShareFrom] = true
		}
	}
	t := &archStateKVTQ{
		set: archICBKVTQ{
			enabled:       make([]bool, len(specs)),
			kBits:         tq.kBits,
			vBits:         tq.vBits,
			kGammas:       make([]metal.MTLBuffer, len(specs)),
			vGammas:       make([]metal.MTLBuffer, len(specs)),
			kRowBytes:     make([]int, len(specs)),
			vRowBytes:     make([]int, len(specs)),
			gammaRowBytes: make([]int, len(specs)),
		},
		kCaches: make([]metal.MTLBuffer, len(specs)),
		vCaches: make([]metal.MTLBuffer, len(specs)),
	}
	maxKvd, maxQd := 0, 0
	for li := range specs {
		sp := specs[li]
		if !sp.OwnsCache() || sp.Mixer != model.MixerAttention || sp.Attention != model.GlobalAttention ||
			shared[li] || !tqKVGeometryOK(tq.kBits, tq.vBits, headDimOf(sp, headDim)) {
			continue
		}
		if li < len(lb) && lb[li].sinks.buf != nil {
			continue // sinks carry no TQ read lane (upstream declined; belt here)
		}
		lkv, lhd := kvHeadsOf(sp, nKVHeads), headDimOf(sp, headDim)
		kRow := lkv * tqBytesPerRow(tq.kBits, lhd)
		vRow := lkv * tqBytesPerRow(tq.vBits, lhd)
		t.kCaches[li] = device.NewBufferWithLengthOptions(uint(maxLen*kRow), metal.MTLResourceStorageModeShared)
		t.vCaches[li] = device.NewBufferWithLengthOptions(uint(maxLen*vRow), metal.MTLResourceStorageModeShared)
		t.set.kGammas[li] = device.NewBufferWithLengthOptions(uint(maxLen*lkv*4), metal.MTLResourceStorageModeShared)
		t.set.vGammas[li] = device.NewBufferWithLengthOptions(uint(maxLen*lkv*4), metal.MTLResourceStorageModeShared)
		t.set.kRowBytes[li] = kRow
		t.set.vRowBytes[li] = vRow
		t.set.gammaRowBytes[li] = lkv * 4
		t.set.enabled[li] = true
		if kvd := lkv * lhd; kvd > maxKvd {
			maxKvd = kvd
		}
		// stepToken's q width for this layer (the scratch is shared across the
		// stack; size to the largest TQ layer's geometry).
		if qd := nHeads * lhd; qd > maxQd {
			maxQd = qd
		}
	}
	if !t.set.any() {
		return nil, nil
	}
	t.kStage = device.NewBufferWithLengthOptions(uint(maxKvd*bf16Size), metal.MTLResourceStorageModeShared)
	t.vStage = device.NewBufferWithLengthOptions(uint(maxKvd*bf16Size), metal.MTLResourceStorageModeShared)
	if maxLen >= sdpa2PassMinKV {
		// the 2-pass lane stages Πq (the O(output) once-per-step fold — fusing
		// it into pass 1 would redo O(d²) per block, see kernels/lthn_tq_kv.metal)
		t.qRot = device.NewBufferWithLengthOptions(uint(maxQd*bf16Size), metal.MTLResourceStorageModeShared)
	}
	// Resolve every pipeline this carrier can dispatch so a missing kernel
	// fails the build loudly (never the first token).
	if _, err := tqKVStorePipeline(tq.kBits); err != nil {
		t.release()
		return nil, err
	}
	if _, err := tqKVStorePipeline(tq.vBits); err != nil {
		t.release()
		return nil, err
	}
	for li := range specs {
		if !t.set.enabled[li] {
			continue
		}
		lhd := headDimOf(specs[li], headDim)
		if _, err := sdpaVectorTQPipeline(lhd, tq.kBits, tq.vBits); err != nil {
			t.release()
			return nil, err
		}
		if maxLen >= sdpa2PassMinKV {
			if _, err := tqRotRowsPipeline(false); err != nil {
				t.release()
				return nil, err
			}
			if _, err := sdpaVector2Pass2TQPipeline(lhd); err != nil {
				t.release()
				return nil, err
			}
		}
	}
	return t, nil
}

// encAttnHalfKVTQ encodes the TurboQuant attention half for GLOBAL owner li
// into enc — stepToken's twin of the recorded lane's TQ block, and the
// codes-cache sibling of encAttnHalfKVInputAt: entry RMS + q projection +
// QK-norm/RoPE exactly as the native half; K rope/norm and the V projection
// land in the FIXED staging rows; one lthn_tq_kv_store each quantises staging
// into the code cache + γ row at pos; the SDPA reads codes — the FUSED
// single-pass kernel below the 2-pass knee (q raw in, final unrotated output
// out, #48), the rot→pass1→pass2 trio at/past it — then the o-projection and
// residual, byte-identical to the native tail. Metal's in-encoder hazard
// tracking orders store→SDPA (both touch the codes), so the token attends its
// own just-landed row exactly as the recorded replay does.
func (s *archDecodeState) encAttnHalfKVTQ(enc metal.MTLComputeCommandEncoder, li int, in metal.MTLBuffer, pos, rotaryDim int, base float32, ropeFreqs metal.MTLBuffer) error {
	t := s.kvTQState
	if !t.on(li) {
		return core.NewError("native.encAttnHalfKVTQ: layer is not a TurboQuant owner")
	}
	sp := s.specs[li]
	lkv, lhd := kvHeadsOf(sp, s.nKVHeads), headDimOf(sp, s.headDim)
	sc, proj := s.asc, s.lb[li].proj
	attnNormW, postAttnNorm, qNorm, kNorm := s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm
	kBits, vBits := t.set.kBits, t.set.vBits
	pi := tqKVPiBuffer(lhd)
	kCent, vCent := tqKVCentroidsBuffer(lhd, kBits), tqKVCentroidsBuffer(lhd, vBits)
	sink := encSink{enc}

	// entry rms — the native half's exact op.
	if err := encRMSNormBF16At(enc, in, attnNormW.buf, sc.normed, 0, attnNormW.off, 0, s.dModel, s.eps); err != nil {
		return err
	}
	// query: project, (per-head QK-norm), rotate IN PLACE — sc.q holds the
	// roped q the fused TQ kernel reads RAW (it computes Πq itself, #48).
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, s.offBuf, 0, ropeFreqs, s.nHeads, lhd, rotaryDim, base, s.ropeScale, s.eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, s.nHeads, lhd, s.eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, s.offBuf, 0, ropeFreqs, s.nHeads, lhd, rotaryDim, base, s.ropeScale); err != nil {
			return err
		}
	}
	// key: project into the STAGING row (not a cache — the cache holds codes),
	// norm+rope there, then rotate+quantise staging → code row + γ row at pos.
	if err := proj.project(enc, sc.normed, t.kStage, 0, projK); err != nil {
		return err
	}
	if gpuHasGeluKernel() && kNorm.buf != nil {
		if err := encQKNormRopeAt(enc, t.kStage, kNorm.buf, t.kStage, 0, kNorm.off, 0, s.offBuf, 0, ropeFreqs, lkv, lhd, rotaryDim, base, s.ropeScale, s.eps); err != nil {
			return err
		}
	} else {
		if kNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, t.kStage, kNorm.buf, t.kStage, 0, kNorm.off, 0, lkv, lhd, s.eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, t.kStage, t.kStage, 0, 0, s.offBuf, 0, ropeFreqs, lkv, lhd, rotaryDim, base, s.ropeScale); err != nil {
			return err
		}
	}
	storeK, err := tqKVStorePipeline(kBits)
	if err != nil {
		return err
	}
	emitTQKVStore(sink, storeK, t.kStage, pi, kCent, t.kCaches[li], uint(pos*t.set.kRowBytes[li]), t.set.kGammas[li], uint(pos*t.set.gammaRowBytes[li]), lkv, lhd)
	// value: project into staging (gemma4 K==V layers project via wK — the
	// native half's vIdx carve), value-norm there, quantise-store.
	vIdx := projV
	if !proj.hasV() {
		vIdx = projK
	}
	if err := proj.project(enc, sc.normed, t.vStage, 0, vIdx); err != nil {
		return err
	}
	if s.valueNormOnes != nil {
		if err := encRMSNormRowsBF16(enc, t.vStage, s.valueNormOnes, t.vStage, 0, 0, 0, lkv, lhd, s.eps); err != nil {
			return err
		}
	}
	storeV, err := tqKVStorePipeline(vBits)
	if err != nil {
		return err
	}
	emitTQKVStore(sink, storeV, t.vStage, pi, vCent, t.vCaches[li], uint(pos*t.set.vRowBytes[li]), t.set.vGammas[li], uint(pos*t.set.gammaRowBytes[li]), lkv, lhd)

	// SDPA over codes [0..pos] — per-token single/2-pass routing, the
	// encSDPADecodeAt knee with the TQ kernel set. TQ owners are GLOBAL: the
	// whole seq-major cache, no ring, no window offset.
	n := pos + 1
	kb, vb := int64(t.set.kRowBytes[li]/lkv), int64(t.set.vRowBytes[li]/lkv)
	if n >= sdpa2PassMinKV && sc.p2Partials != nil && t.qRot != nil && !sdpa2PassDisabledForTest {
		blocks := sdpa2PassBlocks(n, lkv)
		rotPSO, rerr := tqRotRowsPipeline(false)
		if rerr != nil {
			return rerr
		}
		pso1, p1err := sdpaVector2Pass1TQPipeline(lhd, kBits, vBits, blocks)
		if p1err != nil {
			return p1err
		}
		pso2, p2err := sdpaVector2Pass2TQPipeline(lhd)
		if p2err != nil {
			return p2err
		}
		emitTQRotRows(sink, rotPSO, sc.q, pi, t.qRot, s.nHeads, lhd)
		emitSDPAVector2Pass1TQ(sink, pso1, t.qRot, t.kCaches[li], t.vCaches[li], sc.p2Partials, sc.p2Sums, sc.p2Maxs,
			t.set.kGammas[li], t.set.vGammas[li], kCent, vCent, nil,
			s.nHeads, lkv, n, int(blocks), kb, int64(lkv)*kb, vb, int64(lkv)*vb, s.scale)
		emitSDPA2Pass2TQ(sink, pso2, sc.p2Partials, sc.p2Sums, sc.p2Maxs, sc.attn, s.nHeads, int(blocks), pi)
	} else {
		pso, serr := sdpaVectorTQPipeline(lhd, kBits, vBits)
		if serr != nil {
			return serr
		}
		emitSDPAVectorTQ(sink, pso, sc.q, t.kCaches[li], t.vCaches[li], sc.attn,
			t.set.kGammas[li], t.set.vGammas[li], kCent, vCent, nil,
			s.nHeads, lkv, n, kb, int64(lkv)*kb, vb, int64(lkv)*vb, s.scale, pi)
	}
	// output projection + residual — the native tail, byte-identical.
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNormAt(enc, in, 0, sc.attnOut, 0, sc.normed, s.hBuf, 0, postAttnNorm, s.dModel, s.eps)
}
