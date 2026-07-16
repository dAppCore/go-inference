// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"
	"runtime"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/deltanet"
	"dappco.re/go/inference/model/arch/mamba2"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// gated_delta.go is the Qwen 3.6 GatedDeltaNet linear-attention block (the "linear_attention" layers of
// the hybrid schedule), mirroring metal's GatedDeltaMixer.Forward exactly so native can serve it:
//
//	in_proj_qkv → causal depthwise conv (ring) → SiLU → split q|k|v → GQA-repeat(q,k: key→value heads)
//	→ l2norm(q) → α=exp(−exp(A_log)·softplus(a+dt_bias)), β=sigmoid(b)
//	→ GatedDeltaRule → gated RMSNorm: RMSNorm(o)·SiLU(z) → out_proj
//
// The causal conv is reused from mamba2 (the shared FLA primitive metal keeps in flakernel), the
// recurrence from pkg/model/deltanet. Pure Go host f32; the conv-state ring + the delta state thread for
// decode. Projections go through ProjMatMul (host matNT default; native injects a device GEMM).

// GatedDeltaConfig is the per-layer geometry. The delta state is square, so KeyHeadDim == ValueHeadDim ==
// HeadDim; q/k use KeyHeads (GQA-repeated up to ValueHeads), v uses ValueHeads.
type GatedDeltaConfig struct {
	KeyHeads, ValueHeads, HeadDim, ConvKernel int
	Eps                                       float32
}

func (c GatedDeltaConfig) qDim() int    { return c.KeyHeads * c.HeadDim }
func (c GatedDeltaConfig) vDim() int    { return c.ValueHeads * c.HeadDim }
func (c GatedDeltaConfig) convDim() int { return 2*c.qDim() + c.vDim() } // q | k | v

// QDim, VDim and ConvDim are the exported forms of qDim/vDim/convDim — the projection-shape formulas a
// caller OUTSIDE this package needs to target a gated-delta layer's input projections (in_proj_qkv/z/a/b)
// without duplicating the arithmetic. Exported for composed's device-fusion seam (see
// composed.ResidualNormMLPProjGatedDeltaInputDevice), which reads a NEXT layer's gated-delta geometry to
// size the fused command buffer's projection outputs.
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
	InProjQKVQ *model.QuantWeight
	InProjAQ   *model.QuantWeight
	InProjBQ   *model.QuantWeight
	InProjZQ   *model.QuantWeight
	OutProjQ   *model.QuantWeight
}

// ProjMatMul is the device-GEMM seam for the gated-delta projections (host matNT default; native injects
// its steel GEMM). AX-8: the lib declares the hook, the backend sets it.
var ProjMatMul func(x, w []float32, M, K, N int) ([]float32, error)

// ProjMatMulInto is the OPTIONAL write-into sibling of ProjMatMul: a backend that can target a
// caller-owned output buffer sets this so the projection GEMM skips its per-call output alloc (the
// dominant per-token decode cost). nil ⇒ not injected — the caller falls back to ProjMatMul, then the
// host matNTInto. AX-8: the lib declares the hook, the backend sets it. Into is preferred when set and
// the legacy ProjMatMul stays the fallback, so a backend that wired only the old hook keeps working.
var ProjMatMulInto func(out, x, w []float32, M, K, N int) ([]float32, error)

// ProjQuantMatMulInto is the quant twin of ProjMatMulInto: out[M,N] = x[M,K] @ dequant(w)ᵀ for an MLX
// affine-packed weight (packed uint32 codes + bf16 scales/biases). N=outDim, K=inDim, groupSize divides K,
// bits a shipped kernel width. Same AX-8 shape: qwen3 declares the hook and runs the host dequant-row
// fallback (matNTQuantHost); the native backend binds it to the metallib's affine_qmv / affine_qmm_t float
// kernels. nil, or a device error, falls to the host path — correct on any build, not memory-heavy.
var ProjQuantMatMulInto func(out, x []float32, packed, scales, biases []byte, M, K, N, groupSize, bits int) ([]float32, error)

// matNTQuant computes out[M,N] = x[M,K] @ dequant(qw)ᵀ for a packed gated-delta projection, preferring the
// device quant seam and falling back to the host dequant-row path — the gated-delta counterpart of
// composed.matNTQuant, with the same deterministic-on-device-failure contract.
func matNTQuant(out, x []float32, qw *model.QuantWeight, M, K, N int) []float32 {
	if ProjQuantMatMulInto != nil {
		if res, err := ProjQuantMatMulInto(out, x, qw.Packed, qw.Scales, qw.Biases, M, K, N, qw.GroupSize, qw.Bits); err == nil {
			return res
		}
	}
	return matNTQuantHost(out, x, qw, M, K, N)
}

// matNTQuantHost dequantises each output row [K] of the packed weight and dots it with every input row —
// never widening the whole [N,K] weight. The host reference for a non-metal build / a device error; f64
// accumulation in ascending-k order so it matches matNTCols' tier.
func matNTQuantHost(out, x []float32, qw *model.QuantWeight, M, K, N int) []float32 {
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	wordsPerRow := mlxaffine.PackedWords(K, qw.Bits)
	groupsPerRow := K / qw.GroupSize
	for n := range N {
		wr, err := mlxaffine.DequantizeTensor(
			qw.Packed[n*wordsPerRow*4:(n+1)*wordsPerRow*4],
			qw.Scales[n*groupsPerRow*2:(n+1)*groupsPerRow*2],
			qw.Biases[n*groupsPerRow*2:(n+1)*groupsPerRow*2],
			1, K, qw.Bits, qw.GroupSize)
		if err != nil {
			return out
		}
		for m := range M {
			xr := x[m*K : m*K+K]
			var acc float64
			for k := range K {
				acc += float64(xr[k]) * float64(wr[k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

// GatedDeltaInputDevice fuses the four x-reading input projections — in_proj_qkv, in_proj_z,
// in_proj_a, in_proj_b — into ONE command buffer. They all read the same x [L,D], so the GEMMs are
// independent (no barrier between them) and pay ONE command-buffer round-trip where projMatMulInto
// pays four. in_proj_a/b are sub-floor standalone (a few KMACs each) but ride the fused CB for free.
// The backend allocates and returns the four outputs (qkv [L,convDim], z [L,vDim], a/b [L,VH]); a nil
// hook (or an error) leaves the per-projection path in charge — deterministic per build either way.
// AX-8: the lib declares the hook, the backend binds it; the lib never imports the backend.
var GatedDeltaInputDevice func(x, qkvW, zW, aW, bW []float32, L, D, convDim, vDim, VH int) (qkv, z, a, b []float32, err error)

// deviceMinWork is the M·K·N floor below which the projections ignore the device hooks — a tiny
// GEMV (the gated-delta in_proj_a/b are [ValueHeads, D] = a few KMACs) pays a full command-buffer
// round-trip for microseconds of compute, so sub-MMAC shapes stay on the host path. Mirrors
// composed.deviceMinWork.
const deviceMinWork = 1 << 20

func projMatMul(x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMul != nil && M*K*N >= deviceMinWork {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNT(x, w, M, K, N), nil
}

// projMatMulInto runs y = x[M,K] @ w[N,K]ᵀ into out (reused when cap(out) ≥ M·N, else a fresh slab).
// It prefers the write-into backend hook, then the legacy fresh-buffer hook (out is ignored there —
// correctness kept, no reuse), then the host matNTInto. The RETURNED slice is authoritative (it may be a
// freshly grown/allocated buffer); callers store it back into their scratch to retain the growth.
func projMatMulInto(out, x, w []float32, M, K, N int) ([]float32, error) {
	if M*K*N >= deviceMinWork {
		if ProjMatMulInto != nil {
			return ProjMatMulInto(out, x, w, M, K, N)
		}
		if ProjMatMul != nil {
			return ProjMatMul(x, w, M, K, N)
		}
	}
	return matNTInto(out, x, w, M, K, N), nil
}

// matNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ), f32 host.
func matNT(in, w []float32, M, K, N int) []float32 {
	return matNTInto(nil, in, w, M, K, N)
}

// matNTInto is matNT writing into out, reusing it when cap(out) ≥ M·N (else it allocates a fresh M·N
// slab). Identical f64 accumulation + write order to the fresh-buffer form — only WHERE the result lands
// changes, so the output is bit-identical.
//
// Large shapes shard the OUTPUT COLUMNS across CPU cores (mirrors composed's matNTInto): each
// out[m·N+n] keeps exactly the serial per-element k-accumulation order, so the sharded form stays
// bit-identical; small shapes stay serial below the fan-out floor.
func matNTInto(out, in, w []float32, M, K, N int) []float32 {
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	if M*K*N < matNTParMinWork {
		matNTCols(out, in, w, M, K, N, 0, N)
		return out
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > N {
		workers = N
	}
	span := (N + workers - 1) / workers
	var wg core.WaitGroup
	for lo := 0; lo < N; lo += span {
		hi := lo + span
		if hi > N {
			hi = N
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			matNTCols(out, in, w, M, K, N, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return out
}

// matNTParMinWork is the M·K·N floor below which matNTInto stays serial — under ~1 MMAC the
// fan-out/join overhead exceeds the compute it spreads.
const matNTParMinWork = 1 << 20

// matNTCols is the serial kernel over output columns [n0,n1) — the one accumulation-order-defining
// loop both the serial and sharded paths run.
func matNTCols(out, in, w []float32, M, K, N, n0, n1 int) {
	for m := range M {
		for n := n0; n < n1; n++ {
			var acc float64
			for k := range K {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
}

func gdSilu(v float64) float64 { return v / (1 + math.Exp(-v)) }

func gdSoftplus(v float64) float64 {
	if v > 20 {
		return v
	}
	return math.Log1p(math.Exp(v))
}

// GatedDeltaScratch holds the reusable projection-output buffers for GatedDeltaForwardScratchF32. A
// caller that steps one sequence (a decode session — single-goroutine) keeps one Scratch and passes it
// every token so the five projection GEMMs (qkv, a, b, z, out) write into resident buffers instead of
// allocating — the dominant per-token cost. NEVER share a Scratch across concurrently-stepped sessions:
// the buffers are mutable and unsynchronised (they mirror the recurrent conv/delta state's ownership —
// per-session, threaded, never on the shared weights). Buffers grow to fit and are reused thereafter.
// Device carries the engine's resident recurrent-state handle when the device block path is engaged
// (GatedDeltaBlockDevice) — opaque to this package, same per-session lifetime as the scratch.
type GatedDeltaScratch struct {
	qkv, aProj, bProj, zProj, out []float32
	Device                        any
}

// GatedDeltaBlockDevice is the device seam for the WHOLE post-projection gated-delta block — causal
// conv ring + SiLU + split + norms, the α/β gate transform, the delta-rule recurrence and the gated
// RMSNorm·SiLU(z) — in one command buffer with the recurrent state RESIDENT on device across calls
// (stowed on sc.Device). a and b are the RAW projection outputs (the hook's own gate transform
// applies α/β). On success the state advances device-side and the caller carries NO host state
// slices; the engine exports through GatedDeltaDeviceStateExport for snapshots. AX-8: the lib
// declares the hook, the backend binds it; nil ⇒ the host path below is the implementation.
var GatedDeltaBlockDevice func(sc *GatedDeltaScratch, qkv, z, a, b []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L int) (gated []float32, err error)

// GatedDeltaDeviceStateExport reads a device-resident recurrent state (a sc.Device handle) back
// into host-layout slices (conv ring [(K-1),convDim], delta [Hv,Dk,Dv]) — the snapshot/clone seam.
// ok=false means the handle is absent or not authoritative and the caller's host slices stand.
var GatedDeltaDeviceStateExport func(dev any) (conv, delta []float32, ok bool)

// GatedDeltaBlockDeviceTry engages the device block when the backend bound it and this sequence
// threads a real scratch (state continuity lives on sc — a nil sc caller cannot carry the resident
// handle between calls, so it must stay on the host path). engaged=false ⇒ the caller runs the host
// block; engaged=true with err ⇒ the device holds this sequence's authoritative state and a host
// fallback off the stale priors would corrupt it — propagate, never fall back mid-sequence.
func GatedDeltaBlockDeviceTry(sc *GatedDeltaScratch, qkv, z, a, b []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L int) (gated []float32, engaged bool, err error) {
	if GatedDeltaBlockDevice == nil || sc == nil {
		return nil, false, nil
	}
	gated, err = GatedDeltaBlockDevice(sc, qkv, z, a, b, w, cfg, priorConv, priorDelta, L)
	if err == nil {
		return gated, true, nil
	}
	if sc.Device != nil {
		return nil, true, err
	}
	return nil, false, nil // never engaged (first-call decline): the host path serves
}

// GatedDeltaForwardF32 is GatedDeltaForwardScratchF32 with a fresh (nil) scratch — every projection
// allocates, the behaviour before the write-into seam. Kept for existing callers and the engine backend
// parity tests; bit-identical to the scratch path.
func GatedDeltaForwardF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int) (out, newConv, newDelta []float32, err error) {
	return GatedDeltaForwardScratchF32(x, w, cfg, priorConv, priorDelta, L, D, nil)
}

// GatedDeltaForwardScratchF32 runs one chunk of the Qwen 3.6 gated-delta block over x [L, D], threading
// the [conv-state ring, delta state] across calls (priorConv [(K-1),convDim], priorDelta [ValueHeads,
// HeadDim,HeadDim]; both nil ⇒ fresh). Returns out [L, D] and the advanced (newConv, newDelta). When sc
// is non-nil the five projection outputs write into its buffers (reused across calls); nil ⇒ each
// projection allocates fresh (the GatedDeltaForwardF32 path). The recurrent state (newConv/newDelta) is
// always freshly allocated — it is carried information, not scratch. It is GatedDeltaForwardScratchNoProjF32
// plus the out_proj — the projection is split out so the composed session can instead fold it into the
// FFN-tail command buffer (see composed.projMixer / ResidualNormMLPProjDevice).
func GatedDeltaForwardScratchF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int, sc *GatedDeltaScratch) (out, newConv, newDelta []float32, err error) {
	if sc == nil {
		sc = &GatedDeltaScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	gated, vDim, newConv, newDelta, err := GatedDeltaForwardScratchNoProjF32(x, w, cfg, priorConv, priorDelta, L, D, sc)
	if err != nil {
		return nil, nil, nil, err
	}
	_ = vDim
	out, err = GatedDeltaOutProjF32(gated, w, cfg, L, D, sc)
	if err != nil {
		return nil, nil, nil, err
	}
	return out, newConv, newDelta, nil
}

// GatedDeltaOutProjF32 applies the block's final projection out [L,D] = gated [L,vDim] @ OutProjᵀ —
// packed weights through the quant matvec seam, dense through the device/host GEMM ladder — writing
// into sc.out (reused across tokens). Split out of GatedDeltaForwardScratchF32 so a caller running
// the post-projection block on the device seam (GatedDeltaBlockDeviceTry) applies the identical
// projection to the device block's output.
func GatedDeltaOutProjF32(gated []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, L, D int, sc *GatedDeltaScratch) ([]float32, error) {
	if sc == nil {
		sc = &GatedDeltaScratch{} // throwaway: the output allocates fresh
	}
	vDim := cfg.vDim()
	var out []float32
	var err error
	if w.OutProjQ != nil { // packed out_proj through the quant matvec seam
		out = matNTQuant(sc.out, gated, w.OutProjQ, L, vDim, D)
	} else {
		out, err = projMatMulInto(sc.out, gated, w.OutProj, L, vDim, D)
		if err != nil {
			return nil, err
		}
	}
	sc.out = out
	return out, nil
}

// GatedDeltaForwardScratchNoProjF32 is GatedDeltaForwardScratchF32 up to but NOT including out_proj: it
// returns the gated pre-projection hidden [L, vDim] (per (token, value-head) RMSNorm(o)·SiLU(z)), the value
// dim vDim, and the advanced (newConv, newDelta). The composed session uses it to fold out_proj into the
// FFN-tail command buffer (composed.projMixer); GatedDeltaForwardScratchF32 wraps it with the out_proj GEMM.
// The state advances identically — only the final projection is deferred to the caller. sc's projection
// buffers (qkv/aProj/bProj/zProj) are reused as in the wrapper; nil ⇒ each allocates fresh. It computes its
// own input projections then hands off to GatedDeltaForwardScratchFromInputF32, which a device fusion
// (composed.ResidualNormMLPProjGatedDeltaInputDevice) also feeds directly — the two entry points differ
// only in HOW qkv/z/a/b were produced.
func GatedDeltaForwardScratchNoProjF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int, sc *GatedDeltaScratch) (gated []float32, vDim int, newConv, newDelta []float32, err error) {
	if sc == nil {
		sc = &GatedDeltaScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	qkv, zProj, alpha, beta, err := GatedDeltaInputProjectF32(x, w, cfg, L, D, sc)
	if err != nil {
		return nil, 0, nil, nil, err
	}
	return GatedDeltaForwardScratchFromInputF32(qkv, zProj, alpha, beta, w, cfg, priorConv, priorDelta, L, D, sc)
}

// GatedDeltaInputProjectF32 computes the four x-reading input projections (in_proj_qkv/z/a/b) —
// exactly the front half of GatedDeltaForwardScratchNoProjF32, split out so a caller that owns the
// state threading (composed's mixer) can route the post-projection block to the device seam
// (GatedDeltaBlockDeviceTry) instead of the host stages. Same dispatch ladder: packed weights per
// projection through the quant matvec seam; dense through the fused input hook above the floor,
// else the per-projection device/host GEMMs. alpha/beta are the RAW projection outputs.
func GatedDeltaInputProjectF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, L, D int, sc *GatedDeltaScratch) (qkv, zProj, alpha, beta []float32, err error) {
	if sc == nil {
		sc = &GatedDeltaScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	if w == nil {
		return nil, nil, nil, nil, core.NewError("qwen3.GatedDeltaForwardF32: nil weights")
	}
	KH, VH := cfg.KeyHeads, cfg.ValueHeads
	if KH <= 0 || VH <= 0 || cfg.HeadDim <= 0 || VH%KH != 0 || len(x) != L*D {
		return nil, nil, nil, nil, core.NewError("qwen3.GatedDeltaForwardF32: bad geometry or x size")
	}
	vDim := cfg.vDim()
	convDim := cfg.convDim()

	// Input fuse: in_proj_qkv/z/a/b all read x [L,D]. When the backend supplies the fused hook and the
	// dominant qkv projection crosses the device floor, all four are computed up front in ONE command
	// buffer (a/b ride free inside it); z's value depends only on x, so computing it here — before the
	// conv/recurrence that consume it downstream — is identical to computing it at its use site. A nil
	// hook or a device error leaves inputFused false and each projection runs at its per-call site below.
	if w.InProjQKVQ != nil {
		// Packed input projections: each dispatches to the quant matvec seam (the f32 GatedDeltaInputDevice
		// fuse takes f32 weights — bypassed). z is computed here before the conv/recurrence exactly as the
		// dense per-projection path does; identical downstream.
		sc.qkv = matNTQuant(sc.qkv, x, w.InProjQKVQ, L, D, convDim)
		qkv = sc.qkv
		sc.aProj = matNTQuant(sc.aProj, x, w.InProjAQ, L, D, VH)
		alpha = sc.aProj
		sc.bProj = matNTQuant(sc.bProj, x, w.InProjBQ, L, D, VH)
		beta = sc.bProj
		sc.zProj = matNTQuant(sc.zProj, x, w.InProjZQ, L, D, vDim)
		zProj = sc.zProj
	} else {
		inputFused := false
		if GatedDeltaInputDevice != nil && L*D*convDim >= deviceMinWork {
			if fqkv, fz, fa, fb, ferr := GatedDeltaInputDevice(x, w.InProjQKV, w.InProjZ, w.InProjA, w.InProjB, L, D, convDim, vDim, VH); ferr == nil {
				qkv, zProj, alpha, beta = fqkv, fz, fa, fb
				inputFused = true
			}
		}
		if !inputFused {
			qkv, err = projMatMulInto(sc.qkv, x, w.InProjQKV, L, D, convDim)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			sc.qkv = qkv
			alpha, err = projMatMulInto(sc.aProj, x, w.InProjA, L, D, VH)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			sc.aProj = alpha
			beta, err = projMatMulInto(sc.bProj, x, w.InProjB, L, D, VH)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			sc.bProj = beta
			zProj, err = projMatMulInto(sc.zProj, x, w.InProjZ, L, D, vDim)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			sc.zProj = zProj
		}
	}
	return qkv, zProj, alpha, beta, nil
}

// GatedDeltaForwardScratchFromInputF32 is GatedDeltaForwardScratchNoProjF32 from ALREADY-COMPUTED input
// projections (qkv, zProj, alpha, beta) — used when a device fusion
// (composed.ResidualNormMLPProjGatedDeltaInputDevice) folds the PREVIOUS layer's tail + THIS layer's input
// RMSNorm + in_proj_qkv/z/a/b into one command buffer, so this call skips the input projection step
// entirely. From here on the computation — causal conv, split q|k|v, l2-norm, the α/β gate transform, the
// delta-rule recurrence and the gated output norm — is identical to GatedDeltaForwardScratchNoProjF32;
// state threading is the same (priorConv/priorDelta in, newConv/newDelta out). alpha and beta are mutated
// in place (the α/β gate transform below), so callers must not reuse them after this call.
func GatedDeltaForwardScratchFromInputF32(qkv, zProj, alpha, beta []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int, sc *GatedDeltaScratch) (gated []float32, vDim int, newConv, newDelta []float32, err error) {
	if sc == nil {
		sc = &GatedDeltaScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	if w == nil {
		return nil, 0, nil, nil, core.NewError("qwen3.GatedDeltaForwardF32: nil weights")
	}
	KH, VH, HD, K := cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.ConvKernel
	if KH <= 0 || VH <= 0 || HD <= 0 || VH%KH != 0 {
		return nil, 0, nil, nil, core.NewError("qwen3.GatedDeltaForwardF32: bad geometry")
	}
	qDim, vDim, convDim := cfg.qDim(), cfg.vDim(), cfg.convDim()
	rep := VH / KH
	scale := float32(1.0 / math.Sqrt(float64(HD)))

	convOut, newConv, err := mamba2.CausalConv1dF32(qkv, w.ConvWeight, w.ConvBias, priorConv, L, convDim, K)
	if err != nil {
		return nil, 0, nil, nil, err
	}
	for i := range convOut {
		convOut[i] = float32(gdSilu(float64(convOut[i])))
	}

	// split q|k|v, GQA-repeat q,k (value head vh reads key head vh/rep), l2-normalise q.
	// q,k,v are read-only inputs to the recurrence (q is l2-normalised in place over its own window),
	// so one backing slab carved into three non-overlapping capped windows is bit-identical to three
	// makes and saves 2 allocs/token.
	qkvN := L * VH * HD
	qkvBuf := make([]float32, 3*qkvN)
	q := qkvBuf[0:qkvN:qkvN]
	k := qkvBuf[qkvN : 2*qkvN : 2*qkvN]
	v := qkvBuf[2*qkvN : 3*qkvN : 3*qkvN]
	for t := range L {
		base := t * convDim
		for vh := range VH {
			kh := vh / rep
			copy(q[(t*VH+vh)*HD:(t*VH+vh+1)*HD], convOut[base+kh*HD:base+kh*HD+HD])
			copy(k[(t*VH+vh)*HD:(t*VH+vh+1)*HD], convOut[base+qDim+kh*HD:base+qDim+kh*HD+HD])
		}
		copy(v[t*VH*HD:(t+1)*VH*HD], convOut[base+2*qDim:base+2*qDim+vDim])
	}
	for row := 0; row < L*VH; row++ { // l2-normalise q over HD (kernel l2-norms k itself)
		var ss float64
		for i := range HD {
			qv := float64(q[row*HD+i])
			ss += qv * qv
		}
		inv := float32(1.0 / math.Sqrt(ss+1e-6))
		for i := range HD {
			q[row*HD+i] *= inv
		}
	}

	// α = exp(−exp(A_log)·softplus(a+dt_bias)) ∈ (0,1] ; β = sigmoid(b). Per (token, value-head). The
	// two projection outputs are each read once and then dead, and α/β are the same [L,VH] shape, so
	// map α over alpha and β over beta in place — the element-wise transform (output i depends only
	// on input i) makes this bit-identical and needs no separate α/β buffer.
	for i := 0; i < L*VH; i++ {
		h := i % VH
		dt := gdSoftplus(float64(alpha[i]) + float64(w.DtBias[h]))
		aDecay := -math.Exp(float64(w.ALog[h]))
		beta[i] = float32(1.0 / (1.0 + math.Exp(-float64(beta[i]))))
		alpha[i] = float32(math.Exp(aDecay * dt))
	}

	o, newDelta, err := deltanet.GatedDeltaRuleF32(q, k, v, beta, alpha, priorDelta, L, VH, HD, scale, 0)
	if err != nil {
		return nil, 0, nil, nil, err
	}

	// gated RMSNorm: per (token, value-head) RMSNorm(o over HD)·SiLU(z), then out-proj. o is [L,VH,HD]
	// = [L·vDim], the gated shape, and is dead after this stage; each row's o is fully read (ss) then
	// each element is read once more immediately before its own write, so the gated result is written
	// in place over o — bit-identical, one fewer alloc per token.
	gated = o
	for row := 0; row < L*VH; row++ {
		var ss float64
		for i := range HD {
			ov := float64(o[row*HD+i])
			ss += ov * ov
		}
		rms := math.Sqrt(ss/float64(HD) + float64(cfg.Eps))
		for i := range HD {
			normed := float64(o[row*HD+i]) / rms * float64(w.Norm[i])
			gated[row*HD+i] = float32(normed * gdSilu(float64(zProj[row*HD+i])))
		}
	}
	return gated, vDim, newConv, newDelta, nil
}
