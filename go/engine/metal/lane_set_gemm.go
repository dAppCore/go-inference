// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// lane_set_gemm.go — the weight-read-once GEMM batching rung on top of the merged
// laneSet (docs/design-continuous-batching.md pinned gap #1). The merged laneSet's
// Step phase 2 replays every advancing lane's recorded ICB into one command buffer:
// K lanes cost ONE submission (the 2.58× CB-count win), but each lane's ICB still
// runs its OWN per-layer projections, so every weight matrix is read K times per
// layer. This forward reads each weight ONCE for all K lanes.
//
// The seam is the projection. A dense decode layer's seven matmuls (qkv/o + mlp
// gate/up/down) are the only weight-heavy ops; everything else — rms, qk-norm+rope,
// value-norm, SDPA, the residuals, the swiglu — is a cheap per-row elementwise/
// reduction. So the forward keeps EVERY non-projection op per-lane on its OWN
// single-row kernel (byte-identical to the ICB the merged path replays — the
// size-specialised rms kernel a rows kernel would drift by ulps, decode_step.go:314),
// and lifts ONLY the projections to the batched projector.projectRows: it gathers the
// K lanes' per-lane rms outputs into one [K,D] slab, sweeps the weight ONCE (the
// batched bf16 gemv is byte-identical to K single-row gemvs — attention.go:650), and
// scatters the [K,N] result back to the lanes. Attention stays per-lane: each lane
// ropes/stores/attends over its OWN icb KV caches at its OWN position, exactly as its
// ICB replay does. Byte-identity is per-op by construction — only the projection's
// DISPATCH shape changes (one weight sweep vs K), never a row's accumulation order.
//
// SCOPE (byte-identity envelope, gemmEligible): projections whose batched dispatch
// is byte-identical per row (projector.rowsByteTier) — only bf16's batched gemv
// qualifies. Genuinely-quantised weights do NOT: the register-tiled lthn_qmv_rows
// kernel re-orders the quantised dot vs the per-row qmv_impl the replay records (a
// ~1 ulp value-dependent drift — the hd-256 fold divergence, proven 2026-07-13),
// so a quant model DECLINES the fold and keeps the merged per-lane ICB replay. The
// q8 KV rung below rides ON TOP of a byte-identical projection path (so it is
// receipted on bf16 weights); q8 KV over quant weights waits on a byte-identical
// batched quant kernel (a metallib change). Kill switch LTHN_CB_GEMM=0 disables the
// forward entirely (the merged 2.58× path, byte-for-byte).

// gemmForwardEnabled reports whether the weight-read-once GEMM forward is armed.
// LTHN_CB_GEMM=0 forces the per-lane ICB replay (the merged 2.58× path). Any other
// value (including unset) arms it. Read once and cached on the lane set.
func (ls *laneSet) gemmForwardEnabled() bool {
	if ls.gemmMode == 0 {
		ls.gemmMode = 1
		if os.Getenv("LTHN_CB_GEMM") == "0" {
			ls.gemmMode = 2 // disabled
		}
	}
	return ls.gemmMode == 1
}

// gemmEligible reports whether every advancing lane can take the batched-GEMM
// forward. All lanes share the model, so this is really a model-feature check made
// once per Step: recorded-ICB, byte-tier projections (bf16 at any K; quant on
// fast-twin dims — see the gate below), q8 KV only when the fold's q8 mirror
// pipelines resolve (the #367 rung), and ≥2 lanes (one lane has nothing to batch —
// the merged replay is already optimal). The PLE / sliding-window / KV-share arms
// were once excluded pending receipts; they are receipted now — per-feature and
// combined fixtures (TestLaneSetGEMM{Sliding,KVShare,MixedHeadDim,PLE,LayerScalar,
// SplitRope,AllArms}ByteIdentity) plus full-identity on real E2B
// (TestLaneSetGEMME2BByteIdentityHiddens; the historical fires-and-diverges was a
// slab under-sizing on MatFormer deep layers — see gemmDims — plus the live-n SDPA
// routing, both fixed 2026-07-13).
func (ls *laneSet) gemmEligible(advancing []*decodeLane) bool {
	if len(advancing) < 2 {
		return false
	}
	for _, lane := range advancing {
		s := lane.sess.state
		icb := s.icb
		// icb != nil already guarantees no MoE / no trace (icbEligible).
		if icb == nil || !icb.hasFinalOut {
			return false
		}
		// q8 KV: the fold mirrors the recorded quantise-store + q8-read SDPA ops
		// per lane (the #367 staging rung) — eligible only when every pipeline
		// the mirror needs resolves on this metallib.
		if icb.kvQ8 != nil && !gemmQ8Eligible(s) {
			return false
		}
		if !gpuHasGeluKernel() {
			return false // the forward mirrors the fused qk-norm+rope / rms-residual path
		}
		for li := range s.specs {
			lb := s.lb[li]
			// BYTE-IDENTITY / SAFETY GATE: a projection is batched only when its
			// batched dispatch reproduces the per-lane replay byte for byte
			// (projector.rowsByteTier). bf16's batched gemv qualifies at any K.
			// Quant qualifies per weight: the register-tiled lthn_qmv_rows is
			// qmv_fast_impl's M-variant, byte-identical exactly where the per-row
			// oracle routes fast (outDim%8==0 && inDim%512==0 — production dims);
			// any weight off that envelope declines and the lane set keeps the
			// byte-identical merged replay. History: the kernel's packs=1
			// predecessor drifted ~1 ulp value-dependently vs the replay (proven
			// 2026-07-13 — THE hd-256 fold divergence, root-caused to the
			// projection kernel, not the q8 KV ops; 9b6b9d2 had armed it by
			// default, a live exposure on production dense-quant). The fast-twin
			// kernel + per-weight plan check replaced the interim blanket quant
			// decline.
			if !lb.proj.rowsByteTier(len(advancing)) {
				return false
			}
			if !lb.proj.rowsCapable() {
				return false
			}
			// qk-norm rides the fused qk-norm+rope kernel; without gelu the model would
			// need the rms-then-rope branch this forward doesn't mirror — keep it on the replay.
			if (lb.qNorm.buf != nil || lb.kNorm.buf != nil) && !gpuHasGeluKernel() {
				return false
			}
		}
	}
	return true
}

// gemmQ8Blocks reproduces the recorder's fixed 2-pass fan for q8 SDPA reads —
// blocks bake into the RECORDED pipelines from maxLen and the most-starved
// GLOBAL layer's KV heads (decode_forward_arch_icb.go), so the fold must run
// the identical fan for byte-identity with the replay. 0 = single-pass layout.
func gemmQ8Blocks(s archDecodeState) int {
	if s.maxLen < sdpa2PassMinKV {
		return 0
	}
	minKV := 0
	for li := range s.specs {
		if s.specs[li].Attention != model.GlobalAttention {
			continue
		}
		if kv := kvHeadsOf(s.specs[li], s.nKVHeads); minKV == 0 || kv < minKV {
			minKV = kv
		}
	}
	return int(sdpa2PassBlocks(s.maxLen, minKV))
}

// gemmQ8Eligible probes every pipeline the fold's q8 mirror needs: the
// quantise-store op and, per q8-read layer head-dim, the 1-pass q8 SDPA (plus
// the fixed-fan 2-pass pair on global layers when the recorded layout uses it).
// The Once-backed getters make repeat probes cheap.
func gemmQ8Eligible(s archDecodeState) bool {
	if _, err := kvQ8StorePipelineICB(); err != nil {
		return false
	}
	blocks := gemmQ8Blocks(s)
	for li := range s.specs {
		ownerIdx := li
		if !s.specs[li].OwnsCache() {
			ownerIdx = s.specs[li].KVShareFrom
		}
		if s.icb == nil || s.icb.kvQ8 == nil || !s.icb.kvQ8.on(ownerIdx) {
			continue
		}
		hd := headDimOf(s.specs[li], s.headDim)
		if _, err := sdpaVectorQ8PipelineICB(hd); err != nil {
			return false
		}
		if blocks > 0 && s.specs[li].Attention == model.GlobalAttention {
			if _, err := sdpaVector2Pass1Q8PipelineICB(hd, int32(blocks)); err != nil {
				return false
			}
			if _, err := sdpaVector2Pass2PipelineForHeadDim(hd); err != nil {
				return false
			}
		}
	}
	return true
}

// gemmDims scans the model's layers for the widest per-layer projection dims, so the
// staging slabs are sized once for the largest layer: gemma4 global layers carry a
// wider head than sliding ones, and the MatFormer E-series carries a wider FFN on its
// deep layers (E2B: 12288 on layers 15-34 vs the config's 6144 — sizing gate/up/gated
// from s.dFF alone sent row 1's batched projection past the slab, THE lifted-envelope
// E2B divergence: deep layers own no caches, so the corruption left no cache trail).
// Every lane shares the arch, so one lane answers.
func (s *archDecodeState) gemmDims() (maxQDim, maxKVDim, maxFF int) {
	maxFF = s.dFF
	for li := range s.specs {
		lhd := headDimOf(s.specs[li], s.headDim)
		lkv := kvHeadsOf(s.specs[li], s.nKVHeads)
		if q := s.nHeads * lhd; q > maxQDim {
			maxQDim = q
		}
		if kv := lkv * lhd; kv > maxKVDim {
			maxKVDim = kv
		}
		if li < len(s.lb) && s.lb[li].dFF > maxFF {
			maxFF = s.lb[li].dFF
		}
	}
	return maxQDim, maxKVDim, maxFF
}

// gemmSlabs is the K-row staging the batched-GEMM forward gathers into and scatters
// out of — allocated once per lane set (Shared storage: the host writes the input
// embeddings and reads the final hiddens). x/h ping-pong the running hidden across
// layers; the rest stage one layer's projection I/O.
type gemmSlabs struct {
	k                              int
	dModel, maxQDim, maxKVDim, dFF int
	x, h, normed                   metal.MTLBuffer // [K,dModel]
	q, attn                        metal.MTLBuffer // [K,maxQDim]
	kProj, vProj                   metal.MTLBuffer // [K,maxKVDim]
	attnOut, down, mlpNormed       metal.MTLBuffer // [K,dModel]
	gate, up, gated                metal.MTLBuffer // [K,dFF]
}

// ensureGemmSlabs (re)allocates the staging for K lanes and the model's dims.
func (ls *laneSet) ensureGemmSlabs(k, dModel, maxQDim, maxKVDim, dFF int) (*gemmSlabs, error) {
	g := ls.gemm
	if g != nil && g.k >= k && g.dModel == dModel && g.maxQDim >= maxQDim && g.maxKVDim >= maxKVDim && g.dFF >= dFF {
		return g, nil
	}
	if g != nil {
		g.release()
	}
	ng := &gemmSlabs{k: k, dModel: dModel, maxQDim: maxQDim, maxKVDim: maxKVDim, dFF: dFF}
	alloc := func(rowElems int) (metal.MTLBuffer, error) {
		n := uint(k * rowElems * bf16Size)
		buf := device.NewBufferWithLengthOptions(n, metal.MTLResourceStorageModeShared)
		if buf == nil {
			return nil, core.NewError("native.laneSet.ensureGemmSlabs: buffer alloc failed")
		}
		return buf, nil
	}
	var err error
	bufs := []struct {
		dst  *metal.MTLBuffer
		elem int
	}{
		{&ng.x, dModel}, {&ng.h, dModel}, {&ng.normed, dModel},
		{&ng.q, maxQDim}, {&ng.attn, maxQDim},
		{&ng.kProj, maxKVDim}, {&ng.vProj, maxKVDim},
		{&ng.attnOut, dModel}, {&ng.down, dModel}, {&ng.mlpNormed, dModel},
		{&ng.gate, dFF}, {&ng.up, dFF}, {&ng.gated, dFF},
	}
	for _, b := range bufs {
		if *b.dst, err = alloc(b.elem); err != nil {
			ng.release()
			return nil, err
		}
	}
	ls.gemm = ng
	return ng, nil
}

func (g *gemmSlabs) release() {
	if g == nil {
		return
	}
	releaseDeviceBuffers(g.x, g.h, g.normed, g.q, g.attn, g.kProj, g.vProj, g.attnOut, g.down, g.mlpNormed, g.gate, g.up, g.gated)
	*g = gemmSlabs{}
}

// slabRowBytes writes host bytes into row i of a slab (Shared storage).
func slabWriteRow(buf metal.MTLBuffer, i, rowElems int, src []byte) {
	dst := unsafe.Slice((*byte)(bufferContentsFast(buf)), (i+1)*rowElems*bf16Size)
	copy(dst[i*rowElems*bf16Size:], src)
}

// slabReadRow reads row i of a slab into dst (Shared storage).
func slabReadRow(buf metal.MTLBuffer, i, rowElems int, dst []byte) {
	src := unsafe.Slice((*byte)(bufferContentsFast(buf)), (i+1)*rowElems*bf16Size)
	copy(dst, src[i*rowElems*bf16Size:(i+1)*rowElems*bf16Size])
}

// batchedGEMMForward is Step's phase 2 with the projections swept once for all K
// lanes. It advances every lane in `advancing` by one token: writes each lane's
// pending-token embedding (+PLE) into the input slab, runs the layer stack with
// batched projections + per-lane attention, and copies each lane's post-stack hidden
// into lane.hidden — exactly the bytes icb.copyLastOutInto would leave, so the next
// Step's per-lane head reads an identical hidden. Runs inside the caller's autorelease
// pool. Returns ok=false (no encode, no mutation) if a lane's PLE/emb prep fails a
// precondition so the caller can fall back to the replay.
func (ls *laneSet) batchedGEMMForward(advancing []*decodeLane) (err error) {
	k := len(advancing)
	s0 := advancing[0].sess.state
	dModel := s0.dModel
	maxQDim, maxKVDim, maxFF := s0.gemmDims()
	g, err := ls.ensureGemmSlabs(k, dModel, maxQDim, maxKVDim, maxFF)
	if err != nil {
		return err
	}

	// Per-lane input: embed the pending token (+PLE), stage into the input slab and
	// pin each lane's position into its ICB offset buffer for the rope.
	for i, lane := range advancing {
		emb, eerr := lane.sess.embedID(lane.pendingToken)
		if eerr != nil {
			return eerr
		}
		slabWriteRow(g.x, i, dModel, emb)
		if lane.hasPLE {
			pli, perr := lane.sess.perLayerInput(lane.pendingToken, emb)
			if perr != nil {
				return perr
			}
			// pin the lane's PLE tensor into its ICB input buffer (bound by the gate)
			icb := lane.sess.state.icb
			want := icb.nLayers * icb.plePliDim * bf16Size
			if len(pli) != want || icb.pleInputPtr == nil {
				return core.NewError("native.batchedGEMMForward: PLE tensor size mismatch")
			}
			copy(unsafe.Slice(icb.pleInputPtr, want), pli)
		}
		*lane.sess.state.icb.offPtr = int32(lane.pos)
	}

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)

	for li := range s0.specs {
		if err = ls.gemmLayer(enc, advancing, g, li); err != nil {
			endEncodingFast(enc)
			return err
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	ls.fwdCount++
	ls.gemmFwdCount++

	// Scatter each lane's post-stack hidden (now in g.x) back to lane.hidden and
	// advance its position — the same post-conditions the replay leaves.
	for i, lane := range advancing {
		if cap(lane.hidden) < dModel*bf16Size {
			lane.hidden = make([]byte, dModel*bf16Size)
		}
		lane.hidden = lane.hidden[:dModel*bf16Size]
		slabReadRow(g.x, i, dModel, lane.hidden)
		lane.pos++
	}
	return nil
}

// gemmLayer encodes one decode layer for all K lanes: batched projections, per-lane
// attention over each lane's own KV, batched MLP. The running hidden enters in g.x
// and leaves in g.x (g.h is the mid-layer residual carrier), so the next layer reads
// g.x — the same ping-pong the ICB's two slabs run, just K rows wide.
func (ls *laneSet) gemmLayer(enc metal.MTLComputeCommandEncoder, advancing []*decodeLane, g *gemmSlabs, li int) error {
	k := len(advancing)
	s0 := advancing[0].sess.state
	dModel, eps := s0.dModel, s0.eps
	lb := s0.lb[li]
	proj := lb.proj

	lhd := headDimOf(s0.specs[li], s0.headDim)
	lkv := kvHeadsOf(s0.specs[li], s0.nKVHeads)
	qDim, kvDim := s0.nHeads*lhd, lkv*lhd

	// per-attention-type rope params (mirrors stepToken lines 1384-1395)
	slideW, rbase, rotDim := 0, s0.base, s0.rotaryDim
	layerRopeFreqs := s0.ropeFreqs
	if s0.specs[li].Attention == model.SlidingAttention {
		slideW, rbase, rotDim = s0.slidingWindow, s0.localBase, s0.rotaryDimLocal
	} else if s0.globalRopeFreqs != nil {
		layerRopeFreqs, rotDim = s0.globalRopeFreqs, lhd
	}

	// --- entry RMS (per lane, byte-identical single-row kernel) → g.normed ---
	for i := range advancing {
		xOff := uint(i * dModel * bf16Size)
		if err := encRMSNormBF16At(enc, g.x, lb.anw.buf, g.normed, xOff, lb.anw.off, xOff, dModel, eps); err != nil {
			return err
		}
	}
	// A cache-owning layer projects + stores its own K/V; a KV-share layer
	// (gemma4 sliding-tower sharers) projects only Q and attends the OWNER's cache
	// (mirrors stepToken's encAttnHalfKV vs encAttnHalfShared split).
	ownsCache := s0.specs[li].OwnsCache()
	ownerIdx := li
	if !ownsCache {
		ownerIdx = s0.specs[li].KVShareFrom
	}

	// --- batched projections (one weight sweep each): Q always; K/V only on owners ---
	if err := projectRowsRequired(proj, enc, g.normed, g.q, 0, 0, k, projQ); err != nil {
		return err
	}
	vSrc := g.vProj
	if ownsCache {
		if err := projectRowsRequired(proj, enc, g.normed, g.kProj, 0, 0, k, projK); err != nil {
			return err
		}
		if !proj.hasV() {
			vSrc = g.kProj // gemma4 K==V: V rides the k-proj output (value-normed), no weight
		} else if err := projectRowsRequired(proj, enc, g.normed, g.vProj, 0, 0, k, projV); err != nil {
			return err
		}
	}

	// --- per-lane attention: rope Q, (owner: rope+store K, value-norm+store V), SDPA → g.attn ---
	for i, lane := range advancing {
		icb := lane.sess.state.icb
		asc := lane.sess.state.asc
		attendK, attendV := icb.kCaches[ownerIdx], icb.vCaches[ownerIdx]
		ownRowStride := icb.rowBytes[li]
		ownRows := icb.cacheRows[li]
		pos := lane.pos
		offBuf := icb.offBuf
		qOff := uint(i * qDim * bf16Size)
		vOff := uint(i * kvDim * bf16Size) // K/V slab row stride matches their projection (kvDim)

		// Q: gemma4 fuses qk-norm+rope in one op; models without qk-norm rope plain.
		if lb.qNorm.buf != nil {
			if err := encQKNormRopeAt(enc, g.q, lb.qNorm.buf, g.q, qOff, lb.qNorm.off, qOff, offBuf, 0, layerRopeFreqs, s0.nHeads, lhd, rotDim, rbase, s0.scale, eps); err != nil {
				return err
			}
		} else if err := encRopeDecodeAt(enc, g.q, g.q, qOff, qOff, offBuf, 0, layerRopeFreqs, s0.nHeads, lhd, rotDim, rbase, s0.scale); err != nil {
			return err
		}

		if ownsCache {
			// STORE slot — prepareStepRebind's formula (line 785-788): pos%cacheRows
			// when bounded (sliding ring), else linear pos.
			slot := pos
			if ownRows > 0 {
				slot = pos % ownRows
			}
			if icb.kvQ8 != nil && icb.kvQ8.on(li) {
				// q8 owner (#367 staging rung): mirror the recorded sequence —
				// rope/norm in bf16 staging, then ONE quantise-store per row into
				// the int8 cache row + f32 scale row. The fold's slab rows ARE the
				// staging. V stores FIRST from the pre-rope row: on K==V layers
				// vSrc aliases the k slab row, and roping K in place would
				// corrupt the value source (the bf16 path dodges this by roping
				// in-cache; q8 cannot rope an int8 row).
				storePSO, err := kvQ8StorePipelineICB()
				if err != nil {
					return err
				}
				cacheOff := uint(slot * kvDim)                    // int8: 1 byte/elem
				scOff := uint(slot * (kvDim / kvQ8GroupSize) * 4) // f32 scale row
				vRow, vRowOff := vSrc, vOff
				if s0.valueNormOnes != nil {
					dst, dstOff := vSrc, vOff
					if !proj.hasV() { // aliased with the k slab row — norm into the unused vProj row
						dst, dstOff = g.vProj, vOff
					}
					if err := encRMSNormRowsBF16(enc, vSrc, s0.valueNormOnes, dst, vOff, 0, dstOff, lkv, lhd, eps); err != nil {
						return err
					}
					vRow, vRowOff = dst, dstOff
				}
				emitKVQ8StoreAt(encSink{enc}, storePSO, vRow, vRowOff, attendV, cacheOff, icb.kvQ8.vScales[li], scOff, kvDim)
				// K: norm+rope the slab row in place (bf16 staging), then quantise-store.
				if lb.kNorm.buf != nil {
					if err := encQKNormRopeAt(enc, g.kProj, lb.kNorm.buf, g.kProj, vOff, lb.kNorm.off, vOff, offBuf, 0, layerRopeFreqs, lkv, lhd, rotDim, rbase, s0.scale, eps); err != nil {
						return err
					}
				} else if err := encRopeDecodeAt(enc, g.kProj, g.kProj, vOff, vOff, offBuf, 0, layerRopeFreqs, lkv, lhd, rotDim, rbase, s0.scale); err != nil {
					return err
				}
				emitKVQ8StoreAt(encSink{enc}, storePSO, g.kProj, vOff, attendK, cacheOff, icb.kvQ8.kScales[li], scOff, kvDim)
			} else {
				rowOff := uint(slot * ownRowStride)
				// K: store the k-proj row into the cache slot, then rope IN PLACE there —
				// exactly encAttnHalfKVInputAt (project straight into cache, rope in place),
				// so partial rotary's untouched tail keeps the projected value in the cache.
				if err := encCopyBF16Contig(enc, g.kProj, attendK, vOff, rowOff, kvDim); err != nil {
					return err
				}
				if lb.kNorm.buf != nil {
					if err := encQKNormRopeAt(enc, attendK, lb.kNorm.buf, attendK, rowOff, lb.kNorm.off, rowOff, offBuf, 0, layerRopeFreqs, lkv, lhd, rotDim, rbase, s0.scale, eps); err != nil {
						return err
					}
				} else if err := encRopeDecodeAt(enc, attendK, attendK, rowOff, rowOff, offBuf, 0, layerRopeFreqs, lkv, lhd, rotDim, rbase, s0.scale); err != nil {
					return err
				}
				// V: value-norm (gemma4 no-scale rms) FROM the v source slab row INTO the slot.
				if s0.valueNormOnes != nil {
					if err := encRMSNormRowsBF16(enc, vSrc, s0.valueNormOnes, attendV, vOff, 0, rowOff, lkv, lhd, eps); err != nil {
						return err
					}
				} else if err := encCopyBF16Contig(enc, vSrc, attendV, vOff, rowOff, kvDim); err != nil {
					return err
				}
			}
		}

		// READ window — global attends [0,pos+1); sliding attends its live ring from
		// offset 0 (the bounded ring IS the window). Owners and sharers read the same
		// way (encAttnHalfKV / encAttnHalfShared both bind the cache at 0, n live rows).
		n := pos + 1
		if slideW > 0 && n > slideW {
			n = slideW
		}
		if icb.kvQ8 != nil && icb.kvQ8.on(ownerIdx) { // sharers of a q8 owner read q8 too
			kSc, vSc := icb.kvQ8.kScales[ownerIdx], icb.kvQ8.vScales[ownerIdx]
			blocks := gemmQ8Blocks(lane.sess.state)
			if blocks > 0 && s0.specs[li].Attention == model.GlobalAttention {
				// Mirror the recorder's FIXED 2-pass fan (blocks bake from maxLen,
				// not the live n) so the reduction order matches the replay.
				pso1, err := sdpaVector2Pass1Q8PipelineICB(lhd, int32(blocks))
				if err != nil {
					return err
				}
				pso2, err := sdpaVector2Pass2PipelineForHeadDim(lhd)
				if err != nil {
					return err
				}
				sink := encSink{enc}
				emitSDPAVector2Pass1Q8At(sink, pso1, g.q, qOff, attendK, attendV, asc.p2Partials, asc.p2Sums, asc.p2Maxs, kSc, vSc, 0, 0, nil, s0.nHeads, lkv, n, blocks, int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s0.scale)
				emitSDPA2Pass2At(sink, pso2, asc.p2Partials, asc.p2Sums, asc.p2Maxs, g.attn, qOff, 1, s0.nHeads, blocks)
			} else {
				pso, err := sdpaVectorQ8PipelineICB(lhd)
				if err != nil {
					return err
				}
				emitSDPAVectorQ8At(encSink{enc}, pso, g.q, qOff, attendK, attendV, g.attn, qOff, kSc, vSc, 0, 0, nil, s0.nHeads, lkv, n, int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s0.scale)
			}
		} else if blocks := gemmQ8Blocks(lane.sess.state); blocks > 0 && s0.specs[li].Attention == model.GlobalAttention {
			// Mirror the recorder's FIXED 2-pass fan on plain bf16 reads too — the
			// recorded ICB bakes its global-layer SDPA layout (and block count) from
			// maxLen, not the live n. Routing by live n here (single-pass below the
			// knee, blocks-from-n above it) changes the reduction bracketing vs the
			// replay: value-dependent bf16 drift on global layers — THE lifted-envelope
			// E2B divergence (2026-07-13; only shared-region globals leave no cache
			// trail, which is why caches stayed equal while hiddens diverged).
			pso1, perr := sdpaVector2Pass1PipelineForHeadDim(lhd, int32(blocks))
			if perr != nil {
				return perr
			}
			pso2, perr := sdpaVector2Pass2PipelineForHeadDim(lhd)
			if perr != nil {
				return perr
			}
			sink := encSink{enc}
			emitSDPA2Pass1At(sink, pso1, g.q, qOff, attendK, attendV, asc.p2Partials, asc.p2Sums, asc.p2Maxs, 0, 1, s0.nHeads, lkv, n, blocks, int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s0.scale)
			emitSDPA2Pass2At(sink, pso2, asc.p2Partials, asc.p2Sums, asc.p2Maxs, g.attn, qOff, 1, s0.nHeads, blocks)
		} else if err := encSDPADecodeAt(enc, asc, g.q, qOff, attendK, attendV, g.attn, qOff, s0.nHeads, lkv, lhd, n, int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s0.scale, 0); err != nil {
			return err
		}
	}
	// --- batched O projection (one weight sweep) → g.attnOut ---
	if err := projectRowsRequired(proj, enc, g.attn, g.attnOut, 0, 0, k, projO); err != nil {
		return err
	}
	// --- per-lane residual (+ gemma4 post-attention norm) → g.h ---
	for i := range advancing {
		xOff := uint(i * dModel * bf16Size)
		if err := encResidualMaybeNormAt(enc, g.x, xOff, g.attnOut, xOff, g.normed, g.h, xOff, lb.postAttnNorm, dModel, eps); err != nil {
			return err
		}
	}

	// --- MLP: per-lane entry RMS → batched gate/up → per-lane swiglu → batched down → per-lane residual ---
	lff := s0.dFF
	if lb.dFF > 0 {
		lff = lb.dFF
	}
	for i := range advancing {
		xOff := uint(i * dModel * bf16Size)
		if err := encRMSNormBF16At(enc, g.h, lb.mnw.buf, g.mlpNormed, xOff, lb.mnw.off, xOff, dModel, eps); err != nil {
			return err
		}
	}
	if err := projectRowsRequired(proj, enc, g.mlpNormed, g.gate, 0, 0, k, projGate); err != nil {
		return err
	}
	if err := projectRowsRequired(proj, enc, g.mlpNormed, g.up, 0, 0, k, projUp); err != nil {
		return err
	}
	for i := range advancing {
		off := uint(i * lff * bf16Size)
		if err := encGeluGateMulFusedTo(enc, g.gate, g.up, g.gated, off, off, off, lff); err != nil {
			return err
		}
	}
	if err := projectRowsRequired(proj, enc, g.gated, g.down, 0, 0, k, projDown); err != nil {
		return err
	}
	for i, lane := range advancing {
		xOff := uint(i * dModel * bf16Size)
		if err := encResidualMaybeNormAt(enc, g.h, xOff, g.down, xOff, g.mlpNormed, g.x, xOff, lb.postFFNorm, dModel, eps); err != nil {
			return err
		}
		// gemma4 per-layer-input gate (E2B/E4B) + per-layer output scalar, per lane in place.
		if err := ls.gemmLayerEpilogue(enc, lane, li, g.x, xOff); err != nil {
			return err
		}
	}
	return nil
}

// gemmLayerEpilogue encodes one lane's gemma4 PLE gate + per-layer output scalar on
// its layer output row (in place), mirroring stepToken's live gate chain so the bytes
// match the lane's ICB. Skipped cleanly for dense (non-PLE) models.
func (ls *laneSet) gemmLayerEpilogue(enc metal.MTLComputeCommandEncoder, lane *decodeLane, li int, out metal.MTLBuffer, outOff uint) error {
	s := lane.sess.state
	if len(s.ple) > li && len(s.ple[li].postNorm) > 0 {
		pl := s.ple[li]
		if len(pl.postNorm) != s.dModel*bf16Size {
			return core.NewError("native.gemmLayerEpilogue: PLE post norm size mismatch")
		}
		icb := s.icb
		pliOff := uint(li * s.pliDim * bf16Size)
		sc := s.perLayerInputGateScratch()
		if pl.bits == 0 {
			if len(pl.gate.Packed) != s.pliDim*s.dModel*bf16Size || len(pl.proj.Packed) != s.dModel*s.pliDim*bf16Size {
				return core.NewError("native.gemmLayerEpilogue: PLE bf16 weight size mismatch")
			}
			if err := encPerLayerInputGateBF16ScratchAt(enc, sc, out, outOff, residentBytes(pl.gate.Packed), icb.pleInput, residentBytes(pl.proj.Packed), residentBytes(pl.postNorm), out, outOff, pliOff, s.dModel, s.pliDim, s.eps); err != nil {
				return err
			}
		} else {
			gateGroupSize, gateBits, err := validatePerLayerInputGateQuantWeight("gate", pl.gate, s.pliDim, s.dModel, pl.groupSize, pl.bits)
			if err != nil {
				return err
			}
			projGroupSize, projBits, err := validatePerLayerInputGateQuantWeight("projection", pl.proj, s.dModel, s.pliDim, pl.groupSize, pl.bits)
			if err != nil {
				return err
			}
			gatePacked, gateScales, gateBiases := quantWeightViews(pl.gate)
			projPacked, projScales, projBiases := quantWeightViews(pl.proj)
			if err := encPerLayerInputGateQuantScratchAt(enc, sc, out, outOff, gatePacked, gateScales, gateBiases, icb.pleInput, projPacked, projScales, projBiases, residentBytes(pl.postNorm), out, outOff, pliOff, s.dModel, s.pliDim, gateGroupSize, gateBits, projGroupSize, projBits, s.eps); err != nil {
				return err
			}
		}
	}
	if s.lb[li].layerScalar != nil {
		return encMulBF16To(enc, out, s.lb[li].layerScalar, out, outOff, 0, outOff, s.dModel)
	}
	return nil
}
