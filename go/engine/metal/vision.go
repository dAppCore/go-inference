// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"

	core "dappco.re/go"
)

// vision.go ports the gemma4 SigLIP vision tower forward to the no-cgo native path. The decode path
// holds byte-parity with mlx-c because error drift compounds over thousands of autoregressive
// tokens; the vision tower instead runs ONCE per image at prefill and emits soft-token rows into the
// text stream, so it composes native primitives and uses native Steel GEMM for multi-row projections
// instead of cgo. Attention still stays decomposed (matmul + softmax + matmul), so the output is
// numerically EQUIVALENT to pkg/metal's tower within a measured tolerance (vision_test.go pins it),
// not bit-identical — a deliberate trade: there is no autoregressive feedback at prefill for the
// small fp difference to compound. Engine-neutral: this file names no model; the SigLIP geometry
// arrives as a VisionConfig.

// VisionConfig is the engine-neutral SigLIP tower geometry the forward reads — dimensions only, the
// loader fills it from the checkpoint's own declared dims (the vision-side sibling of model.Arch).
// No model name lives here: the same fields describe any patch-embedded vision transformer.
type VisionConfig struct {
	Hidden                int     // encoder width (gemma4-E4B: 768)
	PatchDim              int     // channels·patch·patch — the flattened patch-projection input (3·16·16 = 768)
	NumLayers             int     // encoder layer count
	NumHeads              int     // attention query heads
	NumKVHeads            int     // attention kv heads (GQA; == NumHeads for SigLIP)
	HeadDim               int     // per-head width (Hidden/NumHeads = 64)
	PatchSize             int     // raw image patch-conv kernel and stride
	NumChannels           int     // raw image channel count
	GridH                 int     // patch grid rows (for 2-D rope + spatial pooling)
	GridW                 int     // patch grid cols
	PositionEmbeddingSize int     // slots in a flat or split-axis position embedding table
	RopeBase              float32 // 2-D rope theta
	RMSNormEps            float32
	PoolKernel            int  // spatial pooling kernel (gemma4 default 3)
	Standardize           bool // post-pool (x-bias)·scale
	// EmbeddingScale is √Hidden, multiplied into the pooled rows (cached to skip a per-pass sqrt).
	EmbeddingScale  float32
	ImageTokenID    int32
	ImageBeginToken string
	ImageToken      string
	ImageEndToken   string
	VideoTokenID    int32
	VideoToken      string
}

type visionSDPAScratchKey struct {
	L, nHeads, nKVHeads, headDim int
}

var visionSDPAScratchPools sync.Map

type visionSDPAScratch struct {
	L, nHeads, nKVHeads, headDim   int
	q, k, v, vt, scores, probs, oh []float32
}

func visionSDPAScratchPoolFor(key visionSDPAScratchKey) *sync.Pool {
	if v, ok := visionSDPAScratchPools.Load(key); ok {
		return v.(*sync.Pool)
	}
	pool := new(sync.Pool)
	if v, loaded := visionSDPAScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

func visionSDPAScratchReady(s *visionSDPAScratch, key visionSDPAScratchKey) bool {
	return s != nil &&
		s.L == key.L && s.nHeads == key.nHeads && s.nKVHeads == key.nKVHeads && s.headDim == key.headDim &&
		len(s.q) == key.nHeads*key.L*key.headDim && len(s.k) == key.nKVHeads*key.L*key.headDim &&
		len(s.v) == key.nKVHeads*key.L*key.headDim && len(s.vt) == key.L*key.headDim &&
		len(s.scores) == key.L*key.L && len(s.probs) == key.L*key.L && len(s.oh) == key.L*key.headDim
}

func newVisionSDPAScratch(L, nHeads, nKVHeads, headDim int) *visionSDPAScratch {
	qLen := nHeads * L * headDim
	kvLen := nKVHeads * L * headDim
	scoreLen := L * L
	outLen := L * headDim
	return &visionSDPAScratch{
		L: L, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
		q:      make([]float32, qLen),
		k:      make([]float32, kvLen),
		v:      make([]float32, kvLen),
		vt:     make([]float32, outLen),
		scores: make([]float32, scoreLen),
		probs:  make([]float32, scoreLen),
		oh:     make([]float32, outLen),
	}
}

func getVisionSDPAScratch(L, nHeads, nKVHeads, headDim int) *visionSDPAScratch {
	key := visionSDPAScratchKey{L: L, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim}
	pool := visionSDPAScratchPoolFor(key)
	if v := pool.Get(); v != nil {
		s := v.(*visionSDPAScratch)
		if visionSDPAScratchReady(s, key) {
			return s
		}
	}
	return newVisionSDPAScratch(L, nHeads, nKVHeads, headDim)
}

func putVisionSDPAScratch(s *visionSDPAScratch) {
	if s == nil {
		return
	}
	key := visionSDPAScratchKey{L: s.L, nHeads: s.nHeads, nKVHeads: s.nKVHeads, headDim: s.headDim}
	if visionSDPAScratchReady(s, key) {
		visionSDPAScratchPoolFor(key).Put(s)
	}
}

// MatRowsBF16 is the multi-row sibling of MatVecBF16: out[L, outDim] = in[L, inDim] @ Wᵀ where W is
// row-major (outDim × inDim) bf16, all raw bf16 bytes. It uses native's fused Steel BF16 GEMM so the
// fixed weight is streamed once across all rows, matching pkg/metal's multi-row projection route while
// staying CGO-free. The BF16 output is byte-identical to the old looped MatVecBF16 reference.
//
//	out, err := native.MatRowsBF16(weightBytes, inBytes, L, outDim, inDim)
func MatRowsBF16(w, in []byte, L, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(w) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.MatRowsBF16: len(w) must equal outDim*inDim*2 bytes")
	}
	if len(in) != L*inDim*bf16Size {
		return nil, core.NewError("native.MatRowsBF16: len(in) must equal L*inDim*2 bytes")
	}
	if L == 0 || outDim == 0 || inDim == 0 {
		return make([]byte, L*outDim*bf16Size), nil
	}
	return MatMulBF16NT(in, w, L, inDim, outDim)
}

// scaleSiglipPatchesBF16 applies the SigLIP input normalisation (x-0.5)·2 to bf16 patch pixels,
// host-side. This is BYTE-IDENTICAL to metal's on-device AddScalar(-0.5)+MulScalar(2): the ×2 is an
// exact bf16 exponent bump (mantissa unchanged, no rounding), so the only rounding either way is the
// one after the subtract, and round-to-nearest-even commutes with the doubling. Host-side keeps the
// cheap per-pixel affine off the GPU; the heavy patch projection stays on-device (MatRowsBF16).
func scaleSiglipPatchesBF16(pixels []byte) []byte {
	out := make([]byte, len(pixels))
	for i := 0; i+1 < len(pixels); i += bf16Size {
		x := bf16ToF32(pixels[i], pixels[i+1])
		h := f32ToBF16((x - 0.5) * 2.0)
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// VisionPatchEmbed runs the SigLIP patch embedding: scale the (pre-patchified) pixel patches by the
// SigLIP convention (x-0.5)·2, project them to the encoder width, and add the learned position
// embeddings. pixels is [L, patchDim] bf16; weight is the patch projection [hidden, patchDim] bf16 —
// a non-overlapping patch conv IS exactly this linear projection, so the conv-weight checkpoint and
// the linear-weight checkpoint feed the same matmul. posEmb is the per-patch position embedding rows
// [L, hidden] bf16 already arranged for this grid, or nil when the tower uses only 2-D rope. Returns
// the [L, hidden] bf16 patch rows that open the encoder. Composed from the proven byte-identical
// MatRowsBF16 + AddBF16, so it inherits their equivalence to pkg/metal's patch embedder.
func VisionPatchEmbed(pixels, weight, posEmb []byte, L, patchDim, hidden int) ([]byte, error) {
	if len(pixels) != L*patchDim*bf16Size {
		return nil, core.NewError("native.VisionPatchEmbed: len(pixels) must equal L*patchDim*2 bytes")
	}
	if len(weight) != hidden*patchDim*bf16Size {
		return nil, core.NewError("native.VisionPatchEmbed: len(weight) must equal hidden*patchDim*2 bytes")
	}
	proj, err := MatRowsBF16(weight, scaleSiglipPatchesBF16(pixels), L, hidden, patchDim)
	if err != nil {
		return nil, err
	}
	if posEmb == nil {
		return proj, nil
	}
	if len(posEmb) != L*hidden*bf16Size {
		return nil, core.NewError("native.VisionPatchEmbed: len(posEmb) must equal L*hidden*2 bytes")
	}
	return AddBF16(proj, posEmb)
}

// matRowsF32 is the float32 multi-row matmul out[L,outDim] = in[L,inDim] @ Wᵀ (W row-major
// [outDim,inDim]), dispatched through the same native Steel NT/split-K route as pkg/metal. The
// attention core's two products run through it so the scores and the softmax stay in fp32.
func matRowsF32(w, in []float32, L, outDim, inDim int) ([]float32, error) {
	out := make([]float32, L*outDim)
	if err := matRowsF32Into(out, w, in, L, outDim, inDim); err != nil {
		return nil, err
	}
	return out, nil
}

func matRowsF32Into(out, w, in []float32, L, outDim, inDim int) error {
	if len(w) != outDim*inDim || len(in) != L*inDim {
		return core.NewError("native.matRowsF32: size mismatch (w=outDim*inDim, in=L*inDim)")
	}
	if len(out) != L*outDim {
		return core.NewError("native.matRowsF32: len(out) must equal L*outDim")
	}
	if L == 0 || outDim == 0 || inDim == 0 {
		return nil
	}
	return matMulF32NTInto(out, in, w, L, inDim, outDim, false)
}

// bf16HeadF32 reads one [L,headDim] head out of a [heads,L,headDim] bf16 buffer as fp32.
func bf16HeadF32(b []byte, head, L, headDim int) []float32 {
	out := make([]float32, L*headDim)
	bf16HeadF32Into(out, b, head, L, headDim)
	return out
}

func bf16HeadF32Into(out []float32, b []byte, head, L, headDim int) {
	base := head * L * headDim * bf16Size
	for i := range out {
		o := base + i*bf16Size
		out[i] = bf16ToF32(b[o], b[o+1])
	}
}

// transposeF32 returns the [cols,rows] transpose of a row-major [rows,cols] fp32 matrix.
func transposeF32(m []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	transposeF32Into(out, m, rows, cols)
	return out
}

func transposeF32Into(out, m []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = m[r*cols+c]
		}
	}
}

// VisionSDPA computes full (non-causal, no-mask) bidirectional attention by DECOMPOSITION — the
// composition stand-in for the fused steel attention the vision tower's encoder would otherwise need
// wrapping. q is [nHeads,L,headDim] bf16, k/v are [nKVHeads,L,headDim] bf16 (B=1), out is
// [nHeads,L,headDim] bf16. Per query head: scores[L,L] = q·kᵀ·scale (fp32) → row softmax (fp32) →
// out = scores·v (fp32) → bf16. GQA maps each query head to kv head h/(nHeads/nKVHeads). Keeping the
// scores and softmax in fp32 (the precision the fused kernel keeps) bounds the deviation; the matmuls
// and softmax run on-device. Numerically equivalent to
// pkg/metal.ScaledDotProductAttention within a measured tolerance (vision_test.go), not bit-identical.
func VisionSDPA(q, k, v []byte, L, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.VisionSDPA: nHeads must be a multiple of nKVHeads")
	}
	if len(q) != nHeads*L*headDim*bf16Size {
		return nil, core.NewError("native.VisionSDPA: len(q) must equal nHeads*L*headDim*2 bytes")
	}
	if len(k) != nKVHeads*L*headDim*bf16Size || len(v) != len(k) {
		return nil, core.NewError("native.VisionSDPA: len(k)/len(v) must equal nKVHeads*L*headDim*2 bytes")
	}
	grp := nHeads / nKVHeads
	out := make([]byte, nHeads*L*headDim*bf16Size)
	scratch := getVisionSDPAScratch(L, nHeads, nKVHeads, headDim)
	defer putVisionSDPAScratch(scratch)
	for kvh := 0; kvh < nKVHeads; kvh++ {
		off := kvh * L * headDim
		bf16HeadF32Into(scratch.k[off:off+L*headDim], k, kvh, L, headDim)
		bf16HeadF32Into(scratch.v[off:off+L*headDim], v, kvh, L, headDim)
	}
	for h := 0; h < nHeads; h++ {
		kvh := h / grp
		qOff := h * L * headDim
		kvOff := kvh * L * headDim
		qh := scratch.q[qOff : qOff+L*headDim]
		kh := scratch.k[kvOff : kvOff+L*headDim]
		vh := scratch.v[kvOff : kvOff+L*headDim]
		scores := scratch.scores[:L*L]
		probs := scratch.probs[:L*L]
		oh := scratch.oh[:L*headDim]
		vt := scratch.vt[:L*headDim]

		bf16HeadF32Into(qh, q, h, L, headDim)

		// scores[i,j] = qh[i] · kh[j]  →  matRowsF32(W=kh[L,d], in=qh[L,d]) = qh @ khᵀ
		if err := matRowsF32Into(scores, kh, qh, L, L, headDim); err != nil {
			return nil, err
		}
		for i := range scores {
			scores[i] *= scale
		}
		if err := softmaxF32Into(probs, scores, L, false); err != nil {
			return nil, err
		}

		// out[i,o] = Σ_j scores[i,j]·vh[j,o]  →  matRowsF32(W=vhᵀ[d,L], in=scores[L,L])
		transposeF32Into(vt, vh, L, headDim)
		if err := matRowsF32Into(oh, vt, probs, L, headDim, L); err != nil {
			return nil, err
		}
		base := h * L * headDim * bf16Size
		for i, val := range oh {
			hh := f32ToBF16(val)
			out[base+i*bf16Size], out[base+i*bf16Size+1] = byte(hh), byte(hh>>8)
		}
	}
	return out, nil
}

// bf16ToF32Slice / f32ToBf16Slice convert a whole bf16 byte buffer to/from fp32 — the host-side edge
// where the per-head norms and the 2-D rope run before they hand bf16 back to the device matmuls.
func bf16ToF32Slice(b []byte) []float32 {
	out := make([]float32, len(b)/bf16Size)
	for i := range out {
		out[i] = bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
	}
	return out
}

func f32ToBf16Slice(f []float32) []byte {
	b := make([]byte, len(f)*bf16Size)
	for i, v := range f {
		h := f32ToBF16(v)
		b[i*bf16Size], b[i*bf16Size+1] = byte(h), byte(h>>8)
	}
	return b
}

// rmsNormVec RMS-normalises v in place (x·rsqrt(mean(x²)+eps)), then scales by w when non-nil — the
// plain gemma RMSNorm (no +1 bias), matching native's RMSNorm kernel and metal's RMSNormNoScale for
// the value path. Run per head over the headDim axis.
func rmsNormVec(v, w []float32, eps float32) {
	var ss float32
	for _, x := range v {
		ss += x * x
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(v))+eps)))
	for i := range v {
		v[i] *= inv
		if w != nil {
			v[i] *= w[i]
		}
	}
}

// ropePartRotate applies one rotate-half RoPE block to a length-m slice for a single grid coordinate:
// out[d] = part[d]·cos(θ_d) + rot[d]·sin(θ_d), rot = [-part[half:], part[:half]], θ_d = coord·invFreq[d%（m/2)].
// Lifted from metal's gemma4VisionRotatePart + gemma4Vision2DRoPETables (the 2-D vision RoPE).
func ropePartRotate(out, part []float32, coord float64, invFreq []float64, m int) {
	half := m / 2
	for d := 0; d < m; d++ {
		angle := coord * invFreq[d%half]
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		var rot float32
		if d < half {
			rot = -part[half+d]
		} else {
			rot = part[d-half]
		}
		out[d] = part[d]*c + rot*s
	}
}

// vision2DRoPEHeadMajor applies the gemma4 vision 2-D RoPE to x [L, N, headDim] (fp32, per-patch
// per-head, after QK-norm) and transposes to head-major [N, L, headDim]. The first rotatedPerDim =
// 2·(headDim/4) dims rotate with the patch X coordinate, the next rotatedPerDim with Y, any tail
// passes through — exactly metal's gemma4VisionApply2DRoPE. base==0 ⇒ no rotation (only the
// transpose). gridH·gridW must equal L.
func vision2DRoPEHeadMajor(x []float32, L, N, headDim, gridH, gridW int, base float32) []float32 {
	out := make([]float32, N*L*headDim)
	rotatedPerDim := 2 * (headDim / 4)
	rotatedTotal := rotatedPerDim * 2
	doRoPE := base != 0 && rotatedPerDim >= 2 && gridW > 0
	var invFreq []float64
	if doRoPE {
		half := rotatedPerDim / 2
		invFreq = make([]float64, half)
		for i := 0; i < half; i++ {
			invFreq[i] = 1.0 / math.Pow(float64(base), float64(2*i)/float64(rotatedPerDim))
		}
	}
	for pos := 0; pos < L; pos++ {
		cx, cy := float64(pos%gridW), float64(pos/gridW)
		for h := 0; h < N; h++ {
			in := x[(pos*N+h)*headDim : (pos*N+h)*headDim+headDim]
			o := out[(h*L+pos)*headDim : (h*L+pos)*headDim+headDim]
			if !doRoPE {
				copy(o, in)
				continue
			}
			ropePartRotate(o[0:rotatedPerDim], in[0:rotatedPerDim], cx, invFreq, rotatedPerDim)
			ropePartRotate(o[rotatedPerDim:rotatedTotal], in[rotatedPerDim:rotatedTotal], cy, invFreq, rotatedPerDim)
			for d := rotatedTotal; d < headDim; d++ {
				o[d] = in[d]
			}
		}
	}
	return out
}

// qkNormRoPEHeadMajor takes a [L, N·headDim] bf16 projection, applies the per-head QK-norm (RMSNorm
// with normW) then the 2-D RoPE, and returns head-major [N, L, headDim] bf16 ready for VisionSDPA.
func qkNormRoPEHeadMajor(proj, normW []byte, L, N, headDim, gridH, gridW int, base, eps float32) []byte {
	f := bf16ToF32Slice(proj) // [L, N, headDim]
	w := bf16ToF32Slice(normW)
	for i := 0; i < L*N; i++ {
		rmsNormVec(f[i*headDim:i*headDim+headDim], w, eps)
	}
	return f32ToBf16Slice(vision2DRoPEHeadMajor(f, L, N, headDim, gridH, gridW, base))
}

// vNormHeadMajor takes a [L, N·headDim] bf16 V projection, applies the no-scale per-head RMSNorm
// (metal's RMSNormNoScale), and transposes to head-major [N, L, headDim] bf16.
func vNormHeadMajor(proj []byte, L, N, headDim int, eps float32) []byte {
	f := bf16ToF32Slice(proj) // [L, N, headDim]
	out := make([]float32, N*L*headDim)
	for pos := 0; pos < L; pos++ {
		for h := 0; h < N; h++ {
			v := f[(pos*N+h)*headDim : (pos*N+h)*headDim+headDim]
			rmsNormVec(v, nil, eps)
			copy(out[(h*L+pos)*headDim:(h*L+pos)*headDim+headDim], v)
		}
	}
	return f32ToBf16Slice(out)
}

// VisionLayerWeights is one SigLIP encoder layer's weights as bf16 byte views — the native-side,
// engine-neutral mirror of gemma4.LoadedVisionLayer (an adapter fills it; native imports no model).
// The four norms are [hidden]; QNorm/KNorm are [headDim]; the projections are row-major bf16.
type VisionLayerWeights struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm []byte
	WQ, WK, WV, WO                                 []byte
	BQ, BK, BV, BO                                 []byte
	QNorm, KNorm                                   []byte
	WGate, WUp, WDown                              []byte
	BGate, BUp, BDown                              []byte
}

// visionAttention runs the SigLIP attention subblock on a pre-normed [L, hidden] input: Q/K/V
// projections (on-device) → per-head QK-norm + 2-D RoPE (host) → decomposed full attention
// (VisionSDPA) → output projection. Returns [L, hidden] bf16.
func visionAttention(normed []byte, w *VisionLayerWeights, cfg VisionConfig) ([]byte, error) {
	qDim, kvDim := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	qP, err := visionDenseLinearRows(w.WQ, w.BQ, normed, cfg.GridH*cfg.GridW, qDim, cfg.Hidden, "q projection")
	if err != nil {
		return nil, err
	}
	kP, err := visionDenseLinearRows(w.WK, w.BK, normed, cfg.GridH*cfg.GridW, kvDim, cfg.Hidden, "k projection")
	if err != nil {
		return nil, err
	}
	vP, err := visionDenseLinearRows(w.WV, w.BV, normed, cfg.GridH*cfg.GridW, kvDim, cfg.Hidden, "v projection")
	if err != nil {
		return nil, err
	}
	L := cfg.GridH * cfg.GridW
	q := qkNormRoPEHeadMajor(qP, w.QNorm, L, cfg.NumHeads, cfg.HeadDim, cfg.GridH, cfg.GridW, cfg.RopeBase, cfg.RMSNormEps)
	k := qkNormRoPEHeadMajor(kP, w.KNorm, L, cfg.NumKVHeads, cfg.HeadDim, cfg.GridH, cfg.GridW, cfg.RopeBase, cfg.RMSNormEps)
	v := vNormHeadMajor(vP, L, cfg.NumKVHeads, cfg.HeadDim, cfg.RMSNormEps)

	// The actual gemma4 vision loader (buildGemma4VisionModel) hardcodes the attention scale to 1.0
	// (Gemma4VisionAttention.Attention = 1.0) — NOT 1/√headDim. The QK-norm makes the usual scaling
	// unnecessary. Taken from the real code, not derived.
	attn, err := VisionSDPA(q, k, v, L, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, 1.0)
	if err != nil {
		return nil, err
	}

	// head-major [N, L, headDim] → token-major [L, N·headDim] for the output projection.
	af := bf16ToF32Slice(attn)
	tok := make([]float32, L*qDim)
	for h := 0; h < cfg.NumHeads; h++ {
		for pos := 0; pos < L; pos++ {
			copy(tok[(pos*cfg.NumHeads+h)*cfg.HeadDim:(pos*cfg.NumHeads+h)*cfg.HeadDim+cfg.HeadDim],
				af[(h*L+pos)*cfg.HeadDim:(h*L+pos)*cfg.HeadDim+cfg.HeadDim])
		}
	}
	return visionDenseLinearRows(w.WO, w.BO, f32ToBf16Slice(tok), L, cfg.Hidden, qDim, "output projection")
}

// visionMLP runs the gated-GeLU feed-forward on [L, hidden] bf16: gate/up projections → gelu(gate)·up
// → down projection. The gelu·gate·up runs in fp32 (gemma's tanh-approx gelu) then back to bf16.
func visionMLP(ffIn []byte, w *VisionLayerWeights, L, hidden int) ([]byte, error) {
	ffDim := len(w.WGate) / bf16Size / hidden
	gate, err := visionDenseLinearRows(w.WGate, w.BGate, ffIn, L, ffDim, hidden, "gate projection")
	if err != nil {
		return nil, err
	}
	up, err := visionDenseLinearRows(w.WUp, w.BUp, ffIn, L, ffDim, hidden, "up projection")
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMul(bf16ToF32Slice(gate), bf16ToF32Slice(up))
	if err != nil {
		return nil, err
	}
	return visionDenseLinearRows(w.WDown, w.BDown, f32ToBf16Slice(gated), L, hidden, ffDim, "down projection")
}

// VisionEncoderLayer runs one pre-norm SigLIP encoder block — the faithful re-expression of metal's
// Gemma4VisionEncoderLayer.Forward composed from native's validated ops: InputNorm → attention
// subblock → PostAttnNorm → residual → PreFFNorm → gated MLP → PostFFNorm → residual. x and the
// result are [L, hidden] bf16 (L = GridH·GridW). Numerically equivalent to metal within the measured
// vision tolerance, not bit-identical (the attention softmax + the host norms/rope are fp32).
func VisionEncoderLayer(x []byte, w *VisionLayerWeights, cfg VisionConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	L := cfg.GridH * cfg.GridW
	if len(x) != L*cfg.Hidden*bf16Size {
		return nil, core.NewError("native.VisionEncoderLayer: len(x) must equal GridH*GridW*Hidden*2 bytes")
	}
	normed, err := RMSNormBF16(x, w.InputNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	attnOut, err := visionAttention(normed, w, cfg)
	if err != nil {
		return nil, err
	}
	attnNormed, err := RMSNormBF16(attnOut, w.PostAttnNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	h, err := AddBF16(x, attnNormed)
	if err != nil {
		return nil, err
	}
	ffIn, err := RMSNormBF16(h, w.PreFFNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	ff, err := visionMLP(ffIn, w, L, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	ffNormed, err := RMSNormBF16(ff, w.PostFFNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	return AddBF16(h, ffNormed)
}

// geluTanhScalar is gemma's gelu_pytorch_tanh activation (the vision MLP + projector activation),
// matching metal's GeluActivation.
func geluTanhScalar(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608028654*(x+0.044715*x*x*x)))))
}

// visionGridForPatchCount factors a patch count into the most-square (gridH, gridW) with both
// divisible by poolKernel when it is >1 — a verbatim port of metal's gemma4VisionGridForPatchCount.
// The grid drives the 2-D RoPE coordinates and the spatial pooler.
func visionGridForPatchCount(patches, poolKernel int) (int, int) {
	if patches <= 0 {
		return 0, 0
	}
	bestH, bestW, bestDelta := 1, patches, patches
	for h := 1; h*h <= patches; h++ {
		if patches%h != 0 {
			continue
		}
		w := patches / h
		if poolKernel > 1 && (h%poolKernel != 0 || w%poolKernel != 0) {
			continue
		}
		delta := w - h
		if delta < 0 {
			delta = -delta
		}
		if delta < bestDelta {
			bestH, bestW, bestDelta = h, w, delta
		}
	}
	return bestH, bestW
}

// VisionProjectorLinear is one vision-projector linear. Weight is either dense
// bf16 or affine-packed quant data when Scales/Biases are present.
type VisionProjectorLinear struct {
	Weight         []byte
	Scales, Biases []byte
	Bias           []byte
	OutDim, InDim  int
	GroupSize      int
	Bits           int
}

// VisionProjectorWeights is the vision-to-text projector's weight views: a
// single projection, or fc1+fc2 with a gelu between. Eps is the projector's
// RMSNormNoScale epsilon.
type VisionProjectorWeights struct {
	Projection       VisionProjectorLinear
	Linear1, Linear2 VisionProjectorLinear
	Eps              float32
}

// VisionWeights is the whole SigLIP tower + projector as bf16 byte views — the native-side neutral
// mirror of gemma4.LoadedVision (an adapter fills it; native imports no model). PositionEmbeddings,
// PostLayernorm and StdBias/StdScale are nil when the checkpoint omits them.
type VisionWeights struct {
	PatchEmbedding     []byte
	PatchConvWeight    []byte
	PositionEmbeddings []byte
	PostLayernorm      []byte
	StdBias, StdScale  []byte
	Layers             []VisionLayerWeights
	Projector          VisionProjectorWeights
}

// visionPooler runs metal's Gemma4VisionPooler.Forward host-side: spatial mean-pool by the k×k grid
// (poolByGrid) when the grid divides evenly, else group-pool over k², else the flat reshape, then the
// √Hidden embedding scale. Input/output bf16; input is [gridH·gridW, H].
func visionPooler(hidden []byte, gridH, gridW, H, k int, embScale float32) []byte {
	f := bf16ToF32Slice(hidden)
	L := gridH * gridW
	var pooled []float32
	switch {
	case k > 1 && gridH%k == 0 && gridW%k == 0 && L == gridH*gridW:
		rows, cols := gridH/k, gridW/k
		pooled = make([]float32, rows*cols*H)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				for hh := 0; hh < H; hh++ {
					var acc float32
					for dy := 0; dy < k; dy++ {
						for dx := 0; dx < k; dx++ {
							acc += f[((y*k+dy)*gridW+(x*k+dx))*H+hh]
						}
					}
					pooled[(y*cols+x)*H+hh] = acc / float32(k*k)
				}
			}
		}
	case k > 1 && L%(k*k) == 0:
		outLen := L / (k * k)
		pooled = make([]float32, outLen*H)
		for o := 0; o < outLen; o++ {
			for hh := 0; hh < H; hh++ {
				var acc float32
				for g := 0; g < k*k; g++ {
					acc += f[(o*k*k+g)*H+hh]
				}
				pooled[o*H+hh] = acc / float32(k*k)
			}
		}
	default:
		pooled = f
	}
	for i := range pooled {
		pooled[i] *= embScale
	}
	return f32ToBf16Slice(pooled)
}

// visionStandardize applies the post-pool (x-bias)·scale when the tower carries std weights, host-side
// over the H axis (metal's Subtract+Mul). nil std ⇒ pass through.
func visionStandardize(pooled, stdBias, stdScale []byte, H int) []byte {
	if stdBias == nil || stdScale == nil {
		return pooled
	}
	f, b, s := bf16ToF32Slice(pooled), bf16ToF32Slice(stdBias), bf16ToF32Slice(stdScale)
	for r := 0; r < len(f)/H; r++ {
		for hh := 0; hh < H; hh++ {
			f[r*H+hh] = (f[r*H+hh] - b[hh]) * s[hh]
		}
	}
	return f32ToBf16Slice(f)
}

// visionProjector maps pooled vision rows [*, H] into the text hidden size — metal's
// Gemma4MultiModalProjector.Forward: RMSNormNoScale then a single projection, or fc1→gelu→fc2.
func visionProjector(rows []byte, w *VisionProjectorWeights, H int) ([]byte, error) {
	L := len(rows) / (H * bf16Size)
	f := bf16ToF32Slice(rows)
	for i := 0; i < L; i++ {
		rmsNormVec(f[i*H:i*H+H], nil, w.Eps)
	}
	normed := f32ToBf16Slice(f)
	switch {
	case w.Projection.Weight != nil:
		return visionProjectorLinearRows(normed, w.Projection, L, H, "projection")
	case w.Linear1.Weight != nil && w.Linear2.Weight != nil:
		h1, err := visionProjectorLinearRows(normed, w.Linear1, L, H, "linear1")
		if err != nil {
			return nil, err
		}
		inter := len(h1) / bf16Size / L
		g := bf16ToF32Slice(h1)
		for i := range g {
			g[i] = geluTanhScalar(g[i])
		}
		return visionProjectorLinearRows(f32ToBf16Slice(g), w.Linear2, L, inter, "linear2")
	default:
		return normed, nil
	}
}

func visionProjectorLinearRows(in []byte, w VisionProjectorLinear, rows, inDim int, label string) ([]byte, error) {
	if len(w.Scales) > 0 {
		if w.InDim != inDim || w.OutDim <= 0 || w.GroupSize <= 0 || w.Bits <= 0 {
			return nil, core.NewError("native.VisionProjector: invalid quant " + label + " geometry")
		}
		if len(w.Biases) == 0 {
			return nil, core.NewError("native.VisionProjector: quant " + label + " missing biases")
		}
		out := make([]byte, rows*w.OutDim*bf16Size)
		for r := 0; r < rows; r++ {
			rowIn := in[r*inDim*bf16Size : (r+1)*inDim*bf16Size]
			rowOut := out[r*w.OutDim*bf16Size : (r+1)*w.OutDim*bf16Size]
			if _, err := QMVBF16Into(rowOut, rowIn, w.Weight, w.Scales, w.Biases, w.OutDim, inDim, w.GroupSize, w.Bits); err != nil {
				return nil, err
			}
		}
		return addVisionLinearBiasRows(out, w.Bias, rows, w.OutDim, "native.VisionProjector "+label)
	}
	outDim := w.OutDim
	if outDim <= 0 {
		outDim = len(w.Weight) / bf16Size / inDim
	}
	return visionDenseLinearRows(w.Weight, w.Bias, in, rows, outDim, inDim, "projector "+label)
}

func visionDenseLinearRows(weight, bias, in []byte, rows, outDim, inDim int, label string) ([]byte, error) {
	out, err := MatRowsBF16(weight, in, rows, outDim, inDim)
	if err != nil {
		return nil, err
	}
	return addVisionLinearBiasRows(out, bias, rows, outDim, "native.Vision "+label)
}

func addVisionLinearBiasRows(out, bias []byte, rows, outDim int, label string) ([]byte, error) {
	if bias == nil {
		return out, nil
	}
	if len(bias) != outDim*bf16Size {
		return nil, core.NewError(label + ": bias length must equal outDim*2 bytes")
	}
	if len(out) != rows*outDim*bf16Size {
		return nil, core.NewError(label + ": output length must equal rows*outDim*2 bytes")
	}
	b := bf16ToF32Slice(bias)
	f := bf16ToF32Slice(out)
	for r := 0; r < rows; r++ {
		row := f[r*outDim : (r+1)*outDim]
		for c := range row {
			row[c] += b[c]
		}
	}
	return f32ToBf16Slice(f), nil
}

func visionPositionEmbeddings(table []byte, L, hidden, gridH, gridW, slots int) ([]byte, error) {
	if table == nil {
		return nil, nil
	}
	rowBytes := hidden * bf16Size
	if slots > 0 {
		splitBytes := 2 * slots * rowBytes
		if len(table) == splitBytes {
			if gridH > slots || gridW > slots {
				return nil, core.NewError("native.VisionTower: split position embeddings shorter than grid")
			}
			out := make([]byte, L*rowBytes)
			for y := 0; y < gridH; y++ {
				for x := 0; x < gridW; x++ {
					pos := y*gridW + x
					if pos >= L {
						break
					}
					xBase := x * rowBytes
					yBase := (slots + y) * rowBytes
					dst := pos * rowBytes
					for h := 0; h < hidden; h++ {
						xi := xBase + h*bf16Size
						yi := yBase + h*bf16Size
						v := bf16ToF32(table[xi], table[xi+1]) + bf16ToF32(table[yi], table[yi+1])
						b := f32ToBF16(v)
						out[dst+h*bf16Size], out[dst+h*bf16Size+1] = byte(b), byte(b>>8)
					}
				}
			}
			return out, nil
		}
	}
	need := L * rowBytes
	if len(table) < need {
		return nil, core.NewError("native.VisionTower: position embeddings shorter than patch count")
	}
	return append([]byte(nil), table[:need]...), nil
}

func visionPatchConvEmbedNHWC(pixels []float32, weight []byte, height, width, channels, hidden, patch int) ([]byte, int, int, error) {
	if height <= 0 || width <= 0 || channels <= 0 || hidden <= 0 || patch <= 0 {
		return nil, 0, 0, core.NewError("native.VisionPatchEmbedNHWC: invalid geometry")
	}
	if len(pixels) != height*width*channels {
		return nil, 0, 0, core.NewError("native.VisionPatchEmbedNHWC: pixels must be height*width*channels")
	}
	patchDim := patch * patch * channels
	if len(weight) != hidden*patchDim*bf16Size {
		return nil, 0, 0, core.NewError("native.VisionPatchEmbedNHWC: conv weight must be hidden*patch*patch*channels bf16 bytes")
	}
	gridH, gridW := height/patch, width/patch
	if gridH <= 0 || gridW <= 0 {
		return nil, 0, 0, core.NewError("native.VisionPatchEmbedNHWC: image smaller than patch")
	}
	out := make([]byte, gridH*gridW*hidden*bf16Size)
	row := 0
	for gy := 0; gy < gridH; gy++ {
		for gx := 0; gx < gridW; gx++ {
			for h := 0; h < hidden; h++ {
				var acc float32
				wBase := h * patchDim * bf16Size
				for py := 0; py < patch; py++ {
					y := gy*patch + py
					for px := 0; px < patch; px++ {
						x := gx*patch + px
						for c := 0; c < channels; c++ {
							pix := pixels[(y*width+x)*channels+c]
							wi := wBase + ((py*patch+px)*channels+c)*bf16Size
							acc += (pix - 0.5) * 2 * bf16ToF32(weight[wi], weight[wi+1])
						}
					}
				}
				b := f32ToBF16(acc)
				dst := (row*hidden + h) * bf16Size
				out[dst], out[dst+1] = byte(b), byte(b>>8)
			}
			row++
		}
	}
	return out, gridH, gridW, nil
}

func visionPatchGeometry(cfg VisionConfig) (int, int, error) {
	channels := cfg.NumChannels
	if channels <= 0 {
		channels = 3
	}
	patch := cfg.PatchSize
	if patch <= 0 {
		if cfg.PatchDim <= 0 || cfg.PatchDim%channels != 0 {
			return 0, 0, core.NewError("native.VisionPatchEmbedNHWC: cfg.PatchSize or valid cfg.PatchDim must be set")
		}
		side := int(math.Round(math.Sqrt(float64(cfg.PatchDim / channels))))
		if side <= 0 || side*side*channels != cfg.PatchDim {
			return 0, 0, core.NewError("native.VisionPatchEmbedNHWC: cfg.PatchDim is not channels*patch*patch")
		}
		patch = side
	}
	if cfg.PatchDim > 0 && patch*patch*channels != cfg.PatchDim {
		return 0, 0, core.NewError("native.VisionPatchEmbedNHWC: patch geometry does not match cfg.PatchDim")
	}
	return patch, channels, nil
}

func visionAddBF16Host(x, y []byte, label string) ([]byte, error) {
	if len(x) != len(y) {
		return nil, core.NewError(label + ": add inputs must have equal bf16 byte length")
	}
	out := make([]byte, len(x))
	for i := 0; i < len(x); i += bf16Size {
		v := bf16ToF32(x[i], x[i+1]) + bf16ToF32(y[i], y[i+1])
		b := f32ToBF16(v)
		out[i], out[i+1] = byte(b), byte(b>>8)
	}
	return out, nil
}

// VisionPatchEmbedNHWC runs metal's raw NHWC patch-embed path for one image:
// scale pixels by (x-0.5)*2, apply the patch conv with stride=patch, then add
// optional position embeddings. It returns [gridH*gridW, hidden] bf16 rows.
func VisionPatchEmbedNHWC(pixels []float32, height, width int, w *VisionWeights, cfg VisionConfig) ([]byte, int, int, error) {
	if w == nil {
		return nil, 0, 0, core.NewError("native.VisionPatchEmbedNHWC: weights must be non-nil")
	}
	if cfg.Hidden <= 0 {
		return nil, 0, 0, core.NewError("native.VisionPatchEmbedNHWC: cfg.Hidden must be set")
	}
	patch, channels, err := visionPatchGeometry(cfg)
	if err != nil {
		return nil, 0, 0, err
	}
	conv := w.PatchConvWeight
	if conv == nil {
		conv = w.PatchEmbedding
	}
	h, gridH, gridW, err := visionPatchConvEmbedNHWC(pixels, conv, height, width, channels, cfg.Hidden, patch)
	if err != nil {
		return nil, 0, 0, err
	}
	posEmb, err := visionPositionEmbeddings(w.PositionEmbeddings, gridH*gridW, cfg.Hidden, gridH, gridW, cfg.PositionEmbeddingSize)
	if err != nil {
		return nil, 0, 0, err
	}
	if posEmb != nil {
		h, err = visionAddBF16Host(h, posEmb, "native.VisionPatchEmbedNHWC")
		if err != nil {
			return nil, 0, 0, err
		}
	}
	return h, gridH, gridW, nil
}

// VisionTowerNHWC runs the whole vision tower from a raw NHWC float32 image,
// matching metal's raw-image conv patchify route before entering the shared
// native encoder/pool/projector tail.
func VisionTowerNHWC(pixels []float32, height, width int, w *VisionWeights, cfg VisionConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	h, gridH, gridW, err := VisionPatchEmbedNHWC(pixels, height, width, w, cfg)
	if err != nil {
		return nil, err
	}
	lcfg := cfg
	lcfg.GridH, lcfg.GridW = gridH, gridW
	return visionTowerProjected(h, w, lcfg)
}

func visionTowerProjected(h []byte, w *VisionWeights, cfg VisionConfig) ([]byte, error) {
	L := cfg.GridH * cfg.GridW
	if L <= 0 {
		return nil, core.NewError("native.VisionTower: empty patch grid")
	}
	var err error
	for i := range w.Layers {
		if h, err = VisionEncoderLayer(h, &w.Layers[i], cfg); err != nil {
			return nil, err
		}
	}
	if w.PostLayernorm != nil {
		if h, err = RMSNormBF16(h, w.PostLayernorm, L, cfg.Hidden, cfg.RMSNormEps); err != nil {
			return nil, err
		}
	}
	embScale := cfg.EmbeddingScale
	if embScale == 0 && cfg.Hidden > 0 {
		embScale = float32(math.Sqrt(float64(cfg.Hidden)))
	}
	pooled := visionStandardize(visionPooler(h, cfg.GridH, cfg.GridW, cfg.Hidden, cfg.PoolKernel, embScale), w.StdBias, w.StdScale, cfg.Hidden)
	return visionProjector(pooled, &w.Projector, cfg.Hidden)
}

// VisionTower runs the whole gemma4 SigLIP vision forward on pre-patchified pixel patches [L, patchDim]
// bf16, returning the projected soft-token rows [*, textHidden] bf16 — the faithful port of metal's
// Gemma4VisionModel.Forward: patch embed (+ flat or split-axis 2-D position table) → encoder layers → post-layernorm
// → spatial pooler → standardize → projector. The grid is derived from the patch count exactly as
// metal does.
func VisionTower(patches []byte, w *VisionWeights, cfg VisionConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if cfg.PatchDim == 0 || cfg.Hidden == 0 {
		return nil, core.NewError("native.VisionTower: cfg.PatchDim and cfg.Hidden must be set")
	}
	if w == nil {
		return nil, core.NewError("native.VisionTower: weights must be non-nil")
	}
	L := len(patches) / (cfg.PatchDim * bf16Size)
	gridH, gridW := visionGridForPatchCount(L, cfg.PoolKernel)
	lcfg := cfg
	lcfg.GridH, lcfg.GridW = gridH, gridW

	posEmb, err := visionPositionEmbeddings(w.PositionEmbeddings, L, cfg.Hidden, gridH, gridW, cfg.PositionEmbeddingSize)
	if err != nil {
		return nil, err
	}
	h, err := VisionPatchEmbed(patches, w.PatchEmbedding, posEmb, L, cfg.PatchDim, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	return visionTowerProjected(h, w, lcfg)
}

// VisionInjectFeatures splices the vision soft-token rows into the text embedding stream at the
// image-placeholder positions — the port of metal's injectGemma4TokenFeatures (B=1). embeddings is
// the [L, H] bf16 token-embedding stream; tokenIDs are the L token ids; features are the [N, H] bf16
// vision rows (N must equal the count of imageTokenID positions). Each image-token position takes the
// next feature row in order; the rest pass through. Returns the spliced [L, H] stream. The features'
// H must match the embedding H (the projector already mapped vision → text hidden).
func VisionInjectFeatures(embeddings []byte, tokenIDs []int32, features []byte, imageTokenID int32, H int) ([]byte, error) {
	return injectTokenFeatures(embeddings, tokenIDs, features, imageTokenID, H, "Vision")
}

// AudioInjectFeatures splices Gemma-4 audio soft-token rows into the text
// embedding stream at audio-placeholder positions, matching the same B=1
// contract as Metal's injectGemma4TokenFeatures.
func AudioInjectFeatures(embeddings []byte, tokenIDs []int32, features []byte, audioTokenID int32, H int) ([]byte, error) {
	return injectTokenFeatures(embeddings, tokenIDs, features, audioTokenID, H, "Audio")
}

func injectTokenFeatures(embeddings []byte, tokenIDs []int32, features []byte, tokenID int32, H int, label string) ([]byte, error) {
	if H <= 0 {
		return nil, core.NewError("native." + label + "InjectFeatures: hidden size must be positive")
	}
	row := H * bf16Size
	if len(embeddings)%row != 0 {
		return nil, core.NewError("native." + label + "InjectFeatures: embedding rows must align to hidden size")
	}
	if len(features)%row != 0 {
		return nil, core.NewError("native." + label + "InjectFeatures: feature rows must align to hidden size")
	}
	L := len(embeddings) / row
	if len(tokenIDs) != L {
		return nil, core.NewError("native." + label + "InjectFeatures: token ids must match embedding rows")
	}
	nFeat := len(features) / row
	slots := 0
	for _, id := range tokenIDs {
		if id == tokenID {
			slots++
		}
	}
	if slots != nFeat {
		return nil, core.NewError("native." + label + "InjectFeatures: feature count must equal token slots")
	}
	out := append([]byte(nil), embeddings...)
	featureIdx := 0
	for pos, id := range tokenIDs {
		if id != tokenID {
			continue
		}
		copy(out[pos*row:pos*row+row], features[featureIdx*row:featureIdx*row+row])
		featureIdx++
	}
	return out, nil
}
