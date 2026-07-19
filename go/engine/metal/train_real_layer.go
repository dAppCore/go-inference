// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// train_real_layer.go is the REAL-ARCH successor to the #31 simplified-layer reference
// (train_lora_layer.go): the host-side forward + per-projection LoRA backward of ONE gemma4 decode
// layer AS THE ENGINE COMPUTES IT (encAttnHalfKV + encMLPHalfBF16 + the PLE gate + layer scalar —
// decode_step.go / decode_forward_arch.go are the ground truth), grown ONE feature at a time, each
// finite-difference-gated in train_real_layer_test.go before the next lands. Where the simplified
// reference composed the GPU-backed block backwards (train_backward.go — steel-GEMM matmuls, so its
// composed FD gates need the metal runtime), this file is PURE HOST: f32 storage, f64 accumulation,
// no kernel, so every FD gate here runs on any checkout — the property that lets the trainer wiring
// (#40 stage 3) be gated by tests that cannot silently skip.
//
// Feature ladder (each rung = one commit + its FD gate):
//  1. base — pre-norm GQA attention (causal, sliding-window aware) + standard rope + pre-norm
//     gated-GELU MLP: the simplified layer re-derived host-pure, FD-gated per projection target.
//  2. per-head QK-norm.            3. post-attention / post-feed-forward sandwich norms.
//  4. value-norm + K==V.           5. per-layer-input (PLE) gate.
//  6. rope variants (partial / proportional pairing / sliding) + layer scalar.

// RealTrainLayerF32 bundles one real gemma4 layer's frozen f32 weights and geometry for the host
// training reference — the widened (bf16→f32) form of DecodeLayerWeights plus the per-layer spec
// fields the engine resolves (model.LayerSpec). Projection weights are row-major [out,in]. Rope is
// carried as an EXPLICIT per-pair spectrum so one form covers the engine's base-derived rope
// (pairs (j, j+rotaryDim/2) over the first rotaryDim dims) and the gemma4 proportional global rope
// (pairs (j, j+headDim/2) over the whole head, unrotated pairs holding inv-freq 0) — see
// rope_freqs.go proportionalRopePeriods.
type RealTrainLayerF32 struct {
	AttnNormW []float32 // [DModel] pre-attention RMSNorm
	WQ        []float32 // [Heads·HeadDim, DModel]
	// WK/WV are the key/value projections. A NIL WV declares the gemma4 K==V layer
	// (LayerSpec.AttentionKEqV): the value is the KEY projection's RAW output (pre-QK-norm,
	// pre-rope — the decode's copy-before-k-norms), and the layer carries no v_proj target.
	WK, WV []float32 // [KVHeads·HeadDim, DModel]
	WO     []float32 // [DModel, Heads·HeadDim]
	// ValueNorm applies gemma4's NO-SCALE per-head RMSNorm to V after the projection
	// (Arch.ValueNorm — metal's RMSNormNoScale, a ones weight through the rows kernel).
	ValueNorm bool
	// QNormW/KNormW are gemma4's per-head query/key RMSNorm weights ([HeadDim], shared across
	// heads), applied between the projection and rope (encAttnHalfKV's QK-norm station). nil =
	// the layer carries no QK-norm (LayerSpec.AttentionQNorm/AttentionKNorm false).
	QNormW, KNormW []float32
	// PostAttnNormW is the gemma4 post-attention "sandwich" norm ([DModel]), applied to the
	// attention BRANCH output (Wo·attn) before the residual add — encResidualMaybeNorm's norm.
	// nil = plain residual (LayerSpec.PostAttnNorm false).
	PostAttnNormW []float32
	MLPNormW      []float32 // [DModel] pre-feed-forward RMSNorm
	WGate []float32 // [DFF, DModel]
	WUp   []float32 // [DFF, DModel]
	WDown []float32 // [DModel, DFF]
	// PostFFNormW is the gemma4 post-feed-forward sandwich norm ([DModel]) on the MLP branch
	// output (Wdown·…) before its residual add. nil = plain residual (LayerSpec.PostFFNorm false).
	PostFFNormW []float32

	// The gemma4 per-layer-input (PLE) tower (E2B/E4B — PerLayerInputGateBF16's maths), applied
	// to the layer output after the MLP residual:
	//
	//	gate = WGate_pli·h ; mult = gelu(gate)·pli ; h = h + rms(WProj_pli·mult, postNorm)
	//
	// PLEInput is the layer's slice of the PerLayerInputs tensor ([T, PLIDim]) — a FROZEN
	// function of the token id + main embedding (per_layer_input.go), so it is a CONSTANT of the
	// layer backward (no gradient path from any layer parameter reaches it). All five fields are
	// set together or all absent (a dense layer / a PLE-free model).
	PLEGateW     []float32 // [PLIDim, DModel]
	PLEProjW     []float32 // [DModel, PLIDim]
	PLEPostNormW []float32 // [DModel]
	PLEInput     []float32 // [T, PLIDim]
	PLIDim       int

	T, DModel, DFF          int // rows (tokens), hidden, feed-forward width
	Heads, KVHeads, HeadDim int // GQA head geometry (Heads % KVHeads == 0)

	// Rope: pair j = (j, j+RopePairHalf) rotates by angle RopeScale·pos·RopeInvFreq[j] for
	// j < len(RopeInvFreq); every other dim passes through. Standard full rotary is
	// RopePairHalf = HeadDim/2 with the base-derived spectrum (realRopeInvFreqs).
	RopeInvFreq  []float32
	RopePairHalf int
	RopeScale    float32

	AttnScale float32
	Window    int // sliding window: row i attends [max(0, i−Window+1), i]; 0 = global (all of [0, i])
	Eps       float32
}

// realRopeInvFreqs builds the standard base-derived rope spectrum: invFreq[j] = base^(−2j/rotDim),
// length rotDim/2 — the engine's rope_single kernel maths (and ropeForwardF32's). With
// RopePairHalf = rotDim/2 this reproduces the base-derived partial-rotary rope exactly.
func realRopeInvFreqs(rotDim int, base float32) []float32 {
	inv := make([]float32, rotDim/2)
	for j := range inv {
		inv[j] = float32(math.Pow(float64(base), -2*float64(j)/float64(rotDim)))
	}
	return inv
}

// validate checks the layer's shape contract once, so every entry point shares one refusal site.
func (L *RealTrainLayerF32) validate() error {
	if L == nil {
		return core.NewError("native.RealTrainLayerF32: nil layer")
	}
	if L.T <= 0 || L.DModel <= 0 || L.DFF <= 0 || L.Heads <= 0 || L.KVHeads <= 0 || L.HeadDim <= 0 {
		return core.NewError("native.RealTrainLayerF32: dimensions must be positive")
	}
	if L.Heads%L.KVHeads != 0 {
		return core.NewError("native.RealTrainLayerF32: Heads must be a multiple of KVHeads")
	}
	qDim, kvDim := L.Heads*L.HeadDim, L.KVHeads*L.HeadDim
	if len(L.AttnNormW) != L.DModel || len(L.MLPNormW) != L.DModel {
		return core.NewError("native.RealTrainLayerF32: norm weights must be [DModel]")
	}
	if len(L.WQ) != qDim*L.DModel || len(L.WK) != kvDim*L.DModel || len(L.WO) != L.DModel*qDim {
		return core.NewError("native.RealTrainLayerF32: attention projection weight size mismatch")
	}
	if L.WV != nil && len(L.WV) != kvDim*L.DModel { // nil WV = the K==V layer (no v_proj)
		return core.NewError("native.RealTrainLayerF32: WV must be [KVHeads·HeadDim, DModel] (or nil for a K==V layer)")
	}
	if len(L.WGate) != L.DFF*L.DModel || len(L.WUp) != L.DFF*L.DModel || len(L.WDown) != L.DModel*L.DFF {
		return core.NewError("native.RealTrainLayerF32: MLP projection weight size mismatch")
	}
	if L.RopePairHalf <= 0 || L.RopePairHalf > L.HeadDim/2 || len(L.RopeInvFreq) > L.RopePairHalf {
		return core.NewError("native.RealTrainLayerF32: RopePairHalf must be in (0, HeadDim/2] with len(RopeInvFreq) ≤ RopePairHalf")
	}
	if (len(L.QNormW) != 0 && len(L.QNormW) != L.HeadDim) || (len(L.KNormW) != 0 && len(L.KNormW) != L.HeadDim) {
		return core.NewError("native.RealTrainLayerF32: QNormW/KNormW must be [HeadDim] when set (per-head, shared across heads)")
	}
	if (len(L.PostAttnNormW) != 0 && len(L.PostAttnNormW) != L.DModel) || (len(L.PostFFNormW) != 0 && len(L.PostFFNormW) != L.DModel) {
		return core.NewError("native.RealTrainLayerF32: PostAttnNormW/PostFFNormW must be [DModel] when set")
	}
	if L.hasPLE() {
		if L.PLIDim <= 0 || len(L.PLEGateW) != L.PLIDim*L.DModel || len(L.PLEProjW) != L.DModel*L.PLIDim ||
			len(L.PLEPostNormW) != L.DModel || len(L.PLEInput) != L.T*L.PLIDim {
			return core.NewError("native.RealTrainLayerF32: the PLE tower needs PLEGateW [PLIDim,DModel], PLEProjW [DModel,PLIDim], PLEPostNormW [DModel] and PLEInput [T,PLIDim] together")
		}
	} else if len(L.PLEGateW) != 0 || len(L.PLEProjW) != 0 || len(L.PLEPostNormW) != 0 || len(L.PLEInput) != 0 || L.PLIDim != 0 {
		return core.NewError("native.RealTrainLayerF32: a partial PLE tower — set all of PLEGateW/PLEProjW/PLEPostNormW/PLEInput/PLIDim or none")
	}
	return nil
}

// hasPLE reports whether this layer carries the per-layer-input tower (every field set).
func (L *RealTrainLayerF32) hasPLE() bool {
	return L.PLIDim > 0 && len(L.PLEGateW) > 0 && len(L.PLEProjW) > 0 && len(L.PLEPostNormW) > 0 && len(L.PLEInput) > 0
}

// projDims returns the [out,in] dimensions of a canonical projection target within L, or an error
// for a target this layer does not carry.
func (L *RealTrainLayerF32) projDims(target string) (out, in int, err error) {
	switch target {
	case ProjQ:
		return L.Heads * L.HeadDim, L.DModel, nil
	case ProjV:
		if L.WV == nil {
			return 0, 0, core.NewError("native.RealTrainLayerF32: this layer shares K==V (no v_proj) — target " + ProjK + " trains both paths")
		}
		return L.KVHeads * L.HeadDim, L.DModel, nil
	case ProjK:
		return L.KVHeads * L.HeadDim, L.DModel, nil
	case ProjO:
		return L.DModel, L.Heads * L.HeadDim, nil
	case ProjGate, ProjUp:
		return L.DFF, L.DModel, nil
	case ProjDown:
		return L.DModel, L.DFF, nil
	}
	return 0, 0, core.NewError(core.Concat("native.RealTrainLayerF32: unknown projection target ", core.Sprintf("%q", target),
		" (supported: ", ProjQ, " ", ProjK, " ", ProjV, " ", ProjO, " ", ProjGate, " ", ProjUp, " ", ProjDown, ")"))
}

// setProjWeight swaps the named projection's weight slice and returns the previous one — the FD
// harness substitutes the LoRA effective weight into the forward with it (and restores after).
func (L *RealTrainLayerF32) setProjWeight(target string, w []float32) (prev []float32) {
	switch target {
	case ProjQ:
		prev, L.WQ = L.WQ, w
	case ProjK:
		prev, L.WK = L.WK, w
	case ProjV:
		prev, L.WV = L.WV, w
	case ProjO:
		prev, L.WO = L.WO, w
	case ProjGate:
		prev, L.WGate = L.WGate, w
	case ProjUp:
		prev, L.WUp = L.WUp, w
	case ProjDown:
		prev, L.WDown = L.WDown, w
	}
	return prev
}

// projWeight returns the frozen weight slice of a canonical projection target within L.
func (L *RealTrainLayerF32) projWeight(target string) []float32 {
	switch target {
	case ProjQ:
		return L.WQ
	case ProjK:
		return L.WK
	case ProjV:
		return L.WV
	case ProjO:
		return L.WO
	case ProjGate:
		return L.WGate
	case ProjUp:
		return L.WUp
	case ProjDown:
		return L.WDown
	}
	return nil
}

// ---- pure-host linear algebra (f32 storage, f64 accumulation) ----

// hostLinearF32 computes y[M,N] = x[M,K] · w[N,K]ᵀ — the projection forward (w row-major
// out×in, the storage every projection uses). Pure host, f64 accumulation.
func hostLinearF32(x, w []float32, M, K, N int) []float32 {
	y := make([]float32, M*N)
	for m := range M {
		xr := x[m*K : (m+1)*K]
		for n := range N {
			wr := w[n*K : (n+1)*K]
			var acc float64
			for k := range K {
				acc += float64(xr[k]) * float64(wr[k])
			}
			y[m*N+n] = float32(acc)
		}
	}
	return y
}

// hostLinearBackwardF32 is the VJP of hostLinearF32: dx[M,K] = dy·w, dw[N,K] = dyᵀ·x.
// Pure host, f64 accumulation — the no-kernel twin of LinearBackwardF32.
func hostLinearBackwardF32(dy, x, w []float32, M, K, N int) (dx, dw []float32) {
	dx = make([]float32, M*K)
	for m := range M {
		dyr := dy[m*N : (m+1)*N]
		dxr := dx[m*K : (m+1)*K]
		for k := range K {
			var acc float64
			for n := range N {
				acc += float64(dyr[n]) * float64(w[n*K+k])
			}
			dxr[k] = float32(acc)
		}
	}
	dw = make([]float32, N*K)
	for n := range N {
		for k := range K {
			var acc float64
			for m := range M {
				acc += float64(dy[m*N+n]) * float64(x[m*K+k])
			}
			dw[n*K+k] = float32(acc)
		}
	}
	return dx, dw
}

// realRopeForwardF32 rotates one head-major [nHeads, headDim] row in place-copy form at position pos
// under the layer's explicit spectrum: pair (j, j+pairHalf) rotates by angle scale·pos·invFreq[j]
// for j < len(invFreq); all other dims pass through.
func realRopeForwardF32(x []float32, pos, nHeads, headDim, pairHalf int, invFreq []float32, scale float32) []float32 {
	y := make([]float32, len(x))
	copy(y, x)
	for head := range nHeads {
		off := head * headDim
		for j, f := range invFreq {
			ang := float64(scale) * float64(pos) * float64(f)
			c, s := math.Cos(ang), math.Sin(ang)
			a, b := float64(x[off+j]), float64(x[off+j+pairHalf])
			y[off+j] = float32(a*c - b*s)
			y[off+j+pairHalf] = float32(a*s + b*c)
		}
	}
	return y
}

// realRopeBackwardF32 is the VJP of realRopeForwardF32 — the rotation is orthogonal, so the backward
// is the inverse rotation (by −angle) of the upstream gradient.
func realRopeBackwardF32(dy []float32, pos, nHeads, headDim, pairHalf int, invFreq []float32, scale float32) []float32 {
	dx := make([]float32, len(dy))
	copy(dx, dy)
	for head := range nHeads {
		off := head * headDim
		for j, f := range invFreq {
			ang := float64(scale) * float64(pos) * float64(f)
			c, s := math.Cos(ang), math.Sin(ang)
			a, b := float64(dy[off+j]), float64(dy[off+j+pairHalf])
			dx[off+j] = float32(a*c + b*s)
			dx[off+j+pairHalf] = float32(-a*s + b*c)
		}
	}
	return dx
}

// onesF32 returns a ones vector of length n — the no-scale RMSNorm's implicit weight.
func onesF32(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = 1
	}
	return w
}

// attnLow returns the first attendable position for row i under the layer's window:
// global (Window ≤ 0) attends [0, i]; a sliding layer attends the last Window positions
// [max(0, i−Window+1), i] — the ring-cache live window (decode_step.go: n = min(pos+1, slideW)).
func (L *RealTrainLayerF32) attnLow(i int) int {
	if L.Window <= 0 {
		return 0
	}
	if lo := i - L.Window + 1; lo > 0 {
		return lo
	}
	return 0
}

// hostSDPAProbsF32 computes the causal (window-masked) softmax attention probabilities for one head:
// P[i,j] = softmax_j(q_i·k_j·scale) over j ∈ [attnLow(i), i], zero elsewhere. [T,T], f64 reduction.
func hostSDPAProbsF32(q, k []float32, L *RealTrainLayerF32) []float32 {
	T, d := L.T, L.HeadDim
	p := make([]float32, T*T)
	for i := range T {
		lo := L.attnLow(i)
		mx := math.Inf(-1)
		scores := make([]float64, i-lo+1)
		for j := lo; j <= i; j++ {
			var acc float64
			for c := range d {
				acc += float64(q[i*d+c]) * float64(k[j*d+c])
			}
			acc *= float64(L.AttnScale)
			scores[j-lo] = acc
			if acc > mx {
				mx = acc
			}
		}
		var sum float64
		for j := lo; j <= i; j++ {
			e := math.Exp(scores[j-lo] - mx)
			scores[j-lo] = e
			sum += e
		}
		for j := lo; j <= i; j++ {
			p[i*T+j] = float32(scores[j-lo] / sum)
		}
	}
	return p
}

// realLayerTape holds the forward intermediates the backward consumes — one forward, one tape,
// shared by RealLayerForwardF32 (which returns only the output) and the backward (which walks it
// in reverse).
type realLayerTape struct {
	normed         []float32 // [T,DModel] pre-attn rms output
	q0, k0         []float32 // raw projections [T,qDim] / [T,kvDim]
	qn, kn         []float32 // post-QK-norm, pre-rope (alias q0/k0 when the layer has no QK-norm)
	v0             []float32 // raw value rows (own projection, or the k0 copy on a K==V layer)
	qr, kr, v      []float32 // rope'd q/k and the post-value-norm v (v aliases v0 when unnormed)
	probs          [][]float32 // per query head [T,T]
	o              []float32 // attention output [T,qDim]
	attnBranch     []float32 // the branch added to the residual (post post-attn-norm when present)
	attnBranchPre  []float32 // the branch BEFORE a post-attn norm (the norm backward needs its input)
	h1             []float32 // [T,DModel] residual after attention
	mlpNormed      []float32
	gate, up       []float32
	gated          []float32
	mlpBranch      []float32 // the branch added to the residual (post post-ff-norm when present)
	mlpBranchPre   []float32
	h2             []float32 // [T,DModel] residual after the MLP half (the PLE gate's input)
	pleGate        []float32 // [T,PLIDim] the PLE gate pre-activations (nil without the tower)
	pleProj        []float32 // [T,DModel] the PLE projection output, pre-post-norm
	out            []float32 // [T,DModel] layer output
}

// realLayerForwardTape runs the layer forward with the (possibly LoRA-substituted) weights and
// records the tape. wQ..wDown default to the layer's frozen weights; a LoRA target passes its
// effective weight in the matching slot.
func realLayerForwardTape(h []float32, L *RealTrainLayerF32, wQ, wK, wV, wO, wGate, wUp, wDown []float32) (*realLayerTape, error) {
	if err := L.validate(); err != nil {
		return nil, err
	}
	if len(h) != L.T*L.DModel {
		return nil, core.NewError("native.RealTrainLayerF32: h must be [T,DModel]")
	}
	T, D, d := L.T, L.DModel, L.HeadDim
	qDim, kvDim := L.Heads*d, L.KVHeads*d
	tp := &realLayerTape{}

	// attention half: pre-norm → q/k/v projections → per-head QK-norm → rope(q,k) → windowed GQA
	// SDPA → o-proj → residual (the encAttnHalfKV station order).
	tp.normed = rmsNormForwardF32(h, L.AttnNormW, T, D, L.Eps)
	tp.q0 = hostLinearF32(tp.normed, wQ, T, D, qDim)
	tp.k0 = hostLinearF32(tp.normed, wK, T, D, kvDim)
	if wV != nil {
		tp.v0 = hostLinearF32(tp.normed, wV, T, D, kvDim)
	} else {
		// K==V: the value is the KEY projection's RAW output, copied BEFORE the k norms/rope
		// (decode_step.go's copy-before-k-norms) — under a k_proj LoRA both paths see the
		// effective weight, so dWK will accumulate the K and V contributions.
		tp.v0 = append([]float32(nil), tp.k0...)
	}
	tp.v = tp.v0
	if L.ValueNorm {
		// gemma4 value-norm: NO-SCALE per-head RMSNorm — a ones weight through the plain rms.
		tp.v = rmsNormForwardF32(tp.v0, onesF32(d), T*L.KVHeads, d, L.Eps)
	}
	tp.qn, tp.kn = tp.q0, tp.k0
	if len(L.QNormW) > 0 {
		// per-head RMSNorm with the shared [HeadDim] weight: the head-major layout makes every
		// head's d-vector a contiguous row, so this is the plain rms over T·Heads rows of width d.
		tp.qn = rmsNormForwardF32(tp.q0, L.QNormW, T*L.Heads, d, L.Eps)
	}
	if len(L.KNormW) > 0 {
		tp.kn = rmsNormForwardF32(tp.k0, L.KNormW, T*L.KVHeads, d, L.Eps)
	}
	tp.qr = make([]float32, T*qDim)
	tp.kr = make([]float32, T*kvDim)
	for i := range T {
		copy(tp.qr[i*qDim:(i+1)*qDim], realRopeForwardF32(tp.qn[i*qDim:(i+1)*qDim], i, L.Heads, d, L.RopePairHalf, L.RopeInvFreq, L.RopeScale))
		copy(tp.kr[i*kvDim:(i+1)*kvDim], realRopeForwardF32(tp.kn[i*kvDim:(i+1)*kvDim], i, L.KVHeads, d, L.RopePairHalf, L.RopeInvFreq, L.RopeScale))
	}
	gqa := L.Heads / L.KVHeads
	tp.o = make([]float32, T*qDim)
	tp.probs = make([][]float32, L.Heads)
	for hh := range L.Heads {
		hk := hh / gqa
		qh := gatherHeadF32(tp.qr, T, L.Heads, d, hh)
		kh := gatherHeadF32(tp.kr, T, L.KVHeads, d, hk)
		vh := gatherHeadF32(tp.v, T, L.KVHeads, d, hk)
		p := hostSDPAProbsF32(qh, kh, L)
		tp.probs[hh] = p
		oh := make([]float32, T*d)
		for i := range T {
			for j := L.attnLow(i); j <= i; j++ {
				pij := float64(p[i*T+j])
				for c := range d {
					oh[i*d+c] += float32(pij * float64(vh[j*d+c]))
				}
			}
		}
		scatterAddHeadF32(tp.o, oh, T, L.Heads, d, hh)
	}
	tp.attnBranchPre = hostLinearF32(tp.o, wO, T, qDim, D)
	tp.attnBranch = tp.attnBranchPre
	if len(L.PostAttnNormW) > 0 { // gemma4 sandwich: norm the BRANCH, then add the residual
		tp.attnBranch = rmsNormForwardF32(tp.attnBranchPre, L.PostAttnNormW, T, D, L.Eps)
	}
	tp.h1 = make([]float32, T*D)
	for i := range tp.h1 {
		tp.h1[i] = h[i] + tp.attnBranch[i]
	}

	// MLP half: pre-norm → gate/up → gelu·up → down → residual.
	tp.mlpNormed = rmsNormForwardF32(tp.h1, L.MLPNormW, T, D, L.Eps)
	tp.gate = hostLinearF32(tp.mlpNormed, wGate, T, D, L.DFF)
	tp.up = hostLinearF32(tp.mlpNormed, wUp, T, D, L.DFF)
	tp.gated = make([]float32, T*L.DFF)
	for i := range tp.gated {
		tp.gated[i] = float32(geluTanh(float64(tp.gate[i])) * float64(tp.up[i]))
	}
	tp.mlpBranchPre = hostLinearF32(tp.gated, wDown, T, L.DFF, D)
	tp.mlpBranch = tp.mlpBranchPre
	if len(L.PostFFNormW) > 0 {
		tp.mlpBranch = rmsNormForwardF32(tp.mlpBranchPre, L.PostFFNormW, T, D, L.Eps)
	}
	tp.h2 = make([]float32, T*D)
	for i := range tp.h2 {
		tp.h2[i] = tp.h1[i] + tp.mlpBranch[i]
	}
	tp.out = tp.h2

	// PLE gate (E2B/E4B): h = h + rms(WProj·(gelu(WGate·h)·pli), postNorm) — the
	// PerLayerInputGateBF16 chain with the layer's frozen per-layer-input rows.
	if L.hasPLE() {
		tp.pleGate = hostLinearF32(tp.h2, L.PLEGateW, T, D, L.PLIDim)
		mult := make([]float32, T*L.PLIDim)
		for i := range mult {
			mult[i] = float32(geluTanh(float64(tp.pleGate[i])) * float64(L.PLEInput[i]))
		}
		tp.pleProj = hostLinearF32(mult, L.PLEProjW, T, L.PLIDim, D)
		pleBranch := rmsNormForwardF32(tp.pleProj, L.PLEPostNormW, T, D, L.Eps)
		tp.out = make([]float32, T*D)
		for i := range tp.out {
			tp.out[i] = tp.h2[i] + pleBranch[i]
		}
	}
	return tp, nil
}

// RealLayerForwardF32 is the pure-host forward of one real gemma4 layer over h [T,DModel] — the
// engine layer maths (encAttnHalfKV + encMLPHalfBF16) re-derived in f32/f64 with the layer's frozen
// weights. The reference forward every FD gate in train_real_layer_test.go differentiates.
func RealLayerForwardF32(h []float32, L *RealTrainLayerF32) ([]float32, error) {
	tp, err := realLayerForwardTape(h, L, L.WQ, L.WK, L.WV, L.WO, L.WGate, L.WUp, L.WDown)
	if err != nil {
		return nil, err
	}
	return tp.out, nil
}

// realLayerGrads holds the per-projection weight gradients and dH of one real-layer backward.
type realLayerGrads struct {
	dWQ, dWK, dWV, dWO, dWGate, dWUp, dWDown []float32
	dH                                       []float32
}

// realLayerBackward walks the tape in reverse: given dout [T,DModel] it returns every projection's
// weight gradient plus dH — the gradient to the layer input. Pure host, f64 accumulation.
func realLayerBackward(dout, h []float32, L *RealTrainLayerF32, tp *realLayerTape, wQ, wK, wV, wO, wGate, wUp, wDown []float32) (*realLayerGrads, error) {
	if len(dout) != L.T*L.DModel {
		return nil, core.NewError("native.RealTrainLayerF32: dout must be [T,DModel]")
	}
	T, D, d := L.T, L.DModel, L.HeadDim
	qDim, kvDim := L.Heads*d, L.KVHeads*d
	g := &realLayerGrads{}

	// PLE gate backward first (the gate is the layer's LAST station): out = h2 + rms(proj(mult)),
	// so dH2 = dout + the gate-branch VJP. The PLE weights and the per-layer input are FROZEN —
	// only the flow back to h2 matters (dPli is discarded: the input is a constant of the layer).
	dH2 := dout
	if L.hasPLE() {
		dPleProj, _, perr := RMSNormBackwardF32(dout, tp.pleProj, L.PLEPostNormW, T, D, L.Eps)
		if perr != nil {
			return nil, perr
		}
		// proj forward was mult[T,PLIDim]·WProj[D,PLIDim]ᵀ: M=T, K=PLIDim, N=D → dMult = dy·WProj.
		dMult := make([]float32, T*L.PLIDim)
		for m := range T {
			dyr := dPleProj[m*D : (m+1)*D]
			for k := range L.PLIDim {
				var acc float64
				for n := range D {
					acc += float64(dyr[n]) * float64(L.PLEProjW[n*L.PLIDim+k])
				}
				dMult[m*L.PLIDim+k] = float32(acc)
			}
		}
		dPleGate, _, gerr := GeluGateMulBackwardF32(dMult, tp.pleGate, L.PLEInput, T*L.PLIDim)
		if gerr != nil {
			return nil, gerr
		}
		// gate forward was h2[T,D]·WGate[PLIDim,D]ᵀ: M=T, K=D, N=PLIDim → dH2gate = dy·WGate.
		dH2 = make([]float32, T*D)
		for m := range T {
			dyr := dPleGate[m*L.PLIDim : (m+1)*L.PLIDim]
			for k := range D {
				var acc float64
				for n := range L.PLIDim {
					acc += float64(dyr[n]) * float64(L.PLEGateW[n*D+k])
				}
				dH2[m*D+k] = dout[m*D+k] + float32(acc) // residual + gate branch
			}
		}
	}

	// MLP half backward: residual → (post-FF sandwich norm) → down → gelu·up → gate/up projections
	// → pre-FF norm → residual join.
	dMlpBranchPre := dH2 // h2 = h1 + mlpBranch
	if len(L.PostFFNormW) > 0 {
		var err error
		dMlpBranchPre, _, err = RMSNormBackwardF32(dH2, tp.mlpBranchPre, L.PostFFNormW, T, D, L.Eps)
		if err != nil {
			return nil, err
		}
	}
	// down forward was gated[T,DFF]·wDown[D,DFF]ᵀ → [T,D]: M=T, K=DFF, N=D.
	var dGated []float32
	dGated, g.dWDown = hostLinearBackwardF32(dMlpBranchPre, tp.gated, wDown, T, L.DFF, D)
	dGate, dUp, err := GeluGateMulBackwardF32(dGated, tp.gate, tp.up, T*L.DFF)
	if err != nil {
		return nil, err
	}
	dMlpNormedG, dWGate := hostLinearBackwardF32(dGate, tp.mlpNormed, wGate, T, D, L.DFF)
	dMlpNormedU, dWUp := hostLinearBackwardF32(dUp, tp.mlpNormed, wUp, T, D, L.DFF)
	g.dWGate, g.dWUp = dWGate, dWUp
	dMlpNormed := make([]float32, T*D)
	for i := range dMlpNormed {
		dMlpNormed[i] = dMlpNormedG[i] + dMlpNormedU[i]
	}
	dH1Norm, _, err := RMSNormBackwardF32(dMlpNormed, tp.h1, L.MLPNormW, T, D, L.Eps)
	if err != nil {
		return nil, err
	}
	dH1 := make([]float32, T*D)
	for i := range dH1 {
		dH1[i] = dH2[i] + dH1Norm[i]
	}

	// attention half backward: residual → (post-attention sandwich norm) → o-proj → SDPA core
	// (windowed, GQA) → rope → QK-norm → q/k/v projections → pre-attn norm → residual join.
	dAttnBranchPre := dH1 // h1 = h + attnBranch
	if len(L.PostAttnNormW) > 0 {
		var nerr error
		dAttnBranchPre, _, nerr = RMSNormBackwardF32(dH1, tp.attnBranchPre, L.PostAttnNormW, T, D, L.Eps)
		if nerr != nil {
			return nil, nerr
		}
	}
	// o-proj forward was o[T,qDim]·wO[D,qDim]ᵀ → [T,D]: M=T, K=qDim, N=D, so
	// dO = dy·wO → [T,qDim] and dWO = dyᵀ·o → [D,qDim].
	var dO []float32
	dO, g.dWO = hostLinearBackwardF32(dAttnBranchPre, tp.o, wO, T, qDim, D)
	gqa := L.Heads / L.KVHeads
	dQr := make([]float32, T*qDim)
	dKr := make([]float32, T*kvDim)
	dV := make([]float32, T*kvDim)
	for hh := range L.Heads {
		hk := hh / gqa
		qh := gatherHeadF32(tp.qr, T, L.Heads, d, hh)
		kh := gatherHeadF32(tp.kr, T, L.KVHeads, d, hk)
		vh := gatherHeadF32(tp.v, T, L.KVHeads, d, hk)
		doh := gatherHeadF32(dO, T, L.Heads, d, hh)
		p := tp.probs[hh]

		dqh := make([]float32, T*d)
		dkh := make([]float32, T*d)
		dvh := make([]float32, T*d)
		for i := range T {
			lo := L.attnLow(i)
			// dV_j += P[i,j]·dO_i ; dP[i,j] = dO_i·v_j
			dp := make([]float64, i-lo+1)
			var dot float64 // Σ_j P[i,j]·dP[i,j] (softmax VJP reduction)
			for j := lo; j <= i; j++ {
				var acc float64
				for c := range d {
					acc += float64(doh[i*d+c]) * float64(vh[j*d+c])
					dvh[j*d+c] += float32(float64(p[i*T+j]) * float64(doh[i*d+c]))
				}
				dp[j-lo] = acc
				dot += float64(p[i*T+j]) * acc
			}
			// dS[i,j] = P[i,j]·(dP[i,j] − dot); dQ_i += dS·K_j·scale; dK_j += dS·Q_i·scale
			for j := lo; j <= i; j++ {
				ds := float64(p[i*T+j]) * (dp[j-lo] - dot) * float64(L.AttnScale)
				for c := range d {
					dqh[i*d+c] += float32(ds * float64(kh[j*d+c]))
					dkh[j*d+c] += float32(ds * float64(qh[i*d+c]))
				}
			}
		}
		scatterAddHeadF32(dQr, dqh, T, L.Heads, d, hh)
		scatterAddHeadF32(dKr, dkh, T, L.KVHeads, d, hk)
		scatterAddHeadF32(dV, dvh, T, L.KVHeads, d, hk)
	}
	// rope backward lands on the POST-QK-norm values; the QK-norm backward (when the layer carries
	// one) then maps that onto the raw projections — QKNormBackwardF32, the gemma4 station between
	// the projection VJP and the rope VJP.
	dQ0 := make([]float32, T*qDim)
	dK0 := make([]float32, T*kvDim)
	for i := range T {
		copy(dQ0[i*qDim:(i+1)*qDim], realRopeBackwardF32(dQr[i*qDim:(i+1)*qDim], i, L.Heads, d, L.RopePairHalf, L.RopeInvFreq, L.RopeScale))
		copy(dK0[i*kvDim:(i+1)*kvDim], realRopeBackwardF32(dKr[i*kvDim:(i+1)*kvDim], i, L.KVHeads, d, L.RopePairHalf, L.RopeInvFreq, L.RopeScale))
	}
	if len(L.QNormW) > 0 {
		dQ0, _, err = QKNormBackwardF32(dQ0, tp.q0, L.QNormW, T, L.Heads, d, L.Eps)
		if err != nil {
			return nil, err
		}
	}
	if len(L.KNormW) > 0 {
		dK0, _, err = QKNormBackwardF32(dK0, tp.k0, L.KNormW, T, L.KVHeads, d, L.Eps)
		if err != nil {
			return nil, err
		}
	}
	// value path: the SDPA's dV lands on the POST-value-norm rows; the no-scale norm backward
	// (ones weight) maps it onto the raw value rows, which belong to the v_proj — or, on a K==V
	// layer, to the RAW key projection (the pre-norm copy), where it JOINS dK0 so the shared
	// weight's gradient accumulates both paths.
	dV0 := dV
	if L.ValueNorm {
		dV0, _, err = RMSNormBackwardF32(dV, tp.v0, onesF32(d), T*L.KVHeads, d, L.Eps)
		if err != nil {
			return nil, err
		}
	}
	dNormedQ, dWQ := hostLinearBackwardF32(dQ0, tp.normed, wQ, T, D, qDim)
	g.dWQ = dWQ
	var dNormedV []float32
	if wV != nil {
		dNormedV, g.dWV = hostLinearBackwardF32(dV0, tp.normed, wV, T, D, kvDim)
	} else {
		for i := range dK0 {
			dK0[i] += dV0[i] // K==V: the value gradient joins the raw key-projection gradient
		}
	}
	dNormedK, dWK := hostLinearBackwardF32(dK0, tp.normed, wK, T, D, kvDim)
	g.dWK = dWK
	dNormed := make([]float32, T*D)
	for i := range dNormed {
		dNormed[i] = dNormedQ[i] + dNormedK[i]
		if dNormedV != nil {
			dNormed[i] += dNormedV[i]
		}
	}
	dHNorm, _, err := RMSNormBackwardF32(dNormed, h, L.AttnNormW, T, D, L.Eps)
	if err != nil {
		return nil, err
	}
	g.dH = make([]float32, T*D)
	for i := range g.dH {
		g.dH[i] = dH1[i] + dHNorm[i]
	}
	return g, nil
}

// RealLayerProjLoRABackwardF32 is the host-side backward of ONE real gemma4 layer with a LoRA on the
// named projection target: given the layer's frozen input h [T,DModel] (the residual stream from
// ForwardCaptureHiddens) and the upstream gradient dout [T,DModel], it substitutes the LoRA's
// effective weight at the target, backpropagates through the REAL layer (windowed GQA attention +
// gated-GELU MLP, growing the gemma4 feature set rung by rung), and returns the factor gradients
// plus dH — the gradient to the layer below. Pure host f32/f64; finite-difference-gated per target
// and per feature in train_real_layer_test.go. The trainer wiring (#40 stage 3) builds on this and
// only on shapes whose every feature gate is green.
func RealLayerProjLoRABackwardF32(dout, h []float32, L *RealTrainLayerF32, target string, a, b []float32, rank int, scaling float32) (dA, dB, dH []float32, err error) {
	if err := L.validate(); err != nil {
		return nil, nil, nil, err
	}
	out, in, err := L.projDims(target)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(dout) != L.T*L.DModel || len(h) != L.T*L.DModel {
		return nil, nil, nil, core.NewError("native.RealLayerProjLoRABackwardF32: dout/h must be [T,DModel]")
	}
	eff, err := LoRAEffectiveWeightF32(L.projWeight(target), a, b, out, in, rank, scaling)
	if err != nil {
		return nil, nil, nil, err
	}
	wQ, wK, wV, wO := L.WQ, L.WK, L.WV, L.WO
	wGate, wUp, wDown := L.WGate, L.WUp, L.WDown
	switch target {
	case ProjQ:
		wQ = eff
	case ProjK:
		wK = eff
	case ProjV:
		wV = eff
	case ProjO:
		wO = eff
	case ProjGate:
		wGate = eff
	case ProjUp:
		wUp = eff
	case ProjDown:
		wDown = eff
	}
	tp, err := realLayerForwardTape(h, L, wQ, wK, wV, wO, wGate, wUp, wDown)
	if err != nil {
		return nil, nil, nil, err
	}
	g, err := realLayerBackward(dout, h, L, tp, wQ, wK, wV, wO, wGate, wUp, wDown)
	if err != nil {
		return nil, nil, nil, err
	}
	var dW []float32
	switch target {
	case ProjQ:
		dW = g.dWQ
	case ProjK:
		dW = g.dWK
	case ProjV:
		dW = g.dWV
	case ProjO:
		dW = g.dWO
	case ProjGate:
		dW = g.dWGate
	case ProjUp:
		dW = g.dWUp
	case ProjDown:
		dW = g.dWDown
	}
	dA, dB, err = LoRAFactorGradsF32(dW, a, b, out, in, rank, scaling)
	if err != nil {
		return nil, nil, nil, err
	}
	return dA, dB, g.dH, nil
}
