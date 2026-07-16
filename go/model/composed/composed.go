// SPDX-Licence-Identifier: EUPL-1.2

// Package composed is the native (no-cgo) config-composed hybrid transformer — a pre-norm SwiGLU stack
// whose per-layer attention slot is a config-dispatched sequence Mixer (gated-delta for the
// linear_attention layers, full attention for the rest). It is the native port of metal's
// composed.ComposedModel, the orchestration that turns the FLA mixer math into a servable model: the
// Qwen 3.6 hybrid (gemma4's peer for local inference) runs here. A ComposedSession threads each layer's
// own state — recurrent (conv + delta) for a gated-delta mixer, a KV cache for an attention mixer — so a
// streaming decode reproduces a one-pass prefill exactly. Pure Go host f32; the mixers' projections use
// their own device-GEMM seams.
package composed

import (
	"io"
	"math"
	"runtime"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// Mixer is one layer's sequence mixer (the attention slot). A mixer owns its WEIGHTS (shared across
// sessions); its STATE is threaded by the session, passed in and returned, so one model serves many
// concurrent sessions. prior is nil for a fresh sequence.
type Mixer interface {
	// Forward mixes hidden [L,D] (L tokens) and returns out [L,D] plus the advanced state. The state is
	// opaque to the session (gated-delta carries conv+delta; attention carries a KV cache).
	Forward(hidden []float32, L, D int, prior any) (out []float32, next any, err error)
	// Kind reports the mixer family ("gated_deltanet", "full_attention") for diagnostics + cache typing.
	Kind() string
	// CloneState returns a deep copy of an opaque mixer state such that advancing the
	// copy never mutates the original (and vice-versa). prior==nil ⇒ return nil.
	CloneState(prior any) any
}

// FFN is a layer's feed-forward slot: a dense SwiGLU MLP or a Mixture-of-Experts (qwen3_6_moe). Both map
// hidden [L,D] → [L,D].
type FFN interface {
	forward(x []float32, L, D int) []float32
}

// MLP is a per-layer SwiGLU feed-forward: out = (SiLU(x·Gateᵀ) ⊙ x·Upᵀ)·Downᵀ. Gate/Up are [FF,D],
// Down is [D,FF]. GateQ/UpQ/DownQ are their packed forms in a quant checkpoint (nil ⇒ dense f32); when
// set, forward dispatches the three matmuls to the quant matvec seam instead of the f32 matNT.
type MLP struct {
	Gate, Up, Down    []float32
	GateQ, UpQ, DownQ *model.QuantWeight
	GateB, UpB, DownB *model.BF16Weight
	FF                int
}

// Layer is one pre-norm block: InputNorm → Mixer → residual, PostAttnNorm → MLP → residual.
type Layer struct {
	InputNorm    []float32 // [D] plain RMSNorm (qwen is not gemma)
	Mixer        Mixer
	PostAttnNorm []float32 // [D]
	MLP          FFN       // dense SwiGLU or MoE
}

// ComposedModel is the loaded hybrid stack: token embedding, the per-layer blocks, the final norm and the
// LM head (tied to Embed when Output is nil). All f32 (the loader widens the bf16 checkpoint).
type ComposedModel struct {
	Embed            []float32          // [Vocab, D] (nil ⇒ EmbedQ packed or EmbedB bf16-resident)
	EmbedQ           *model.QuantWeight // packed embedding table (dequantised one row per token at gather; never widened whole)
	EmbedB           *model.BF16Weight  // bf16-resident embedding table (#26 — widened one row per token at gather)
	Layers           []Layer
	NormF            []float32          // [D] final RMSNorm
	Output           []float32          // [Vocab, D] (nil ⇒ tied to Embed, or OutputQ packed / OutputB bf16)
	OutputQ          *model.QuantWeight // packed untied LM head (served by the quant matvec)
	OutputB          *model.BF16Weight  // bf16-resident untied LM head (served by the bf16 matvec seam)
	D                int
	Vocab            int
	Eps              float32
	LayerNorm        bool
	ParallelResidual bool
	LogitScale       float32
	EmbedScale       float32
	LogitsScaling    float32
	ResidualScale    float32
	// Quantised marks a checkpoint whose projections are kept PACKED: forwardEmb then runs the plain
	// per-projection path (the mixer's own Forward + the host FFN tail), where the big matmuls dispatch to
	// the quant matvec seam. The f32 device tail-fusion hooks (ResidualNormMLPProj*Device) are bypassed —
	// they take f32 weights, and quant fused tails are a later slice.
	// BF16Resident marks a dense checkpoint whose 2-D projections stayed the checkpoint's own bf16
	// bytes (#26): forwardEmb routes it down the per-layer mixer path exactly like Quantised (the f32
	// fold ladder's hooks take f32 weights — bypassed).
	BF16Resident bool
	Quantised bool
	// mmap is the checkpoint mapping this model's zero-copy packed weights VIEW (a QuantWeight.Packed slice
	// aliases it), owned so the model unmaps it on Close/finalize; nil when no weight aliases a mapping (a
	// dense model, an all-1-bit pack repacked to owned heap, or the copying LoadComposed path).
	mmap io.Closer
	// mmapAliased is set by loadComposed when at least one packed weight is a view into the input tensors —
	// the signal model.LoadComposedDir reads (via RetainMmap) to hand this model the checkpoint mapping.
	mmapAliased bool
	// Vision is the loaded vision tower + projector/merger (see vision.go, vision_loader.go) — nil for a
	// text-only checkpoint, or one whose safetensors carry no vision_tower.*/multi_modal_projector.*
	// tensors. Non-nil is the AcceptsImageInput() live probe: a text-only quant of a vision family loads
	// with Vision nil exactly as it always did (loadComposed's vision step is additive-only).
	Vision *visionTower
	// ImageTokenID is the vocabulary id one image soft-token placeholder occupies (config.json's top-level
	// image_token_id) — 0 when Vision is nil. Read by ComposedTokenModel.ImagePlaceholderTokenID.
	ImageTokenID int32
	// VisionBeginToken/VisionToken/VisionEndToken are the literal spellings that wrap an image's
	// soft-token run in the rendered prompt text (the Qwen-VL family's <|vision_start|>/<|image_pad|>/
	// <|vision_end|> convention) — empty when Vision is nil. Read by
	// ComposedTokenModel.ImagePlaceholderBlock.
	VisionBeginToken, VisionToken, VisionEndToken string
}

// retain gives this model ownership of the checkpoint mapping c when its packed weights ALIAS c (a
// zero-copy quant load), returning true; the model then unmaps c on Close (a finalizer is the fallback
// for a caller that never calls Close). Returns false — leaving c for the caller to unmap now — when no
// weight aliases c (dense, copied, or all-1-bit-repacked), so an unused mapping is never held.
func (m *ComposedModel) retain(c io.Closer) bool {
	if m == nil || !m.mmapAliased || c == nil {
		return false
	}
	m.mmap = c
	// The finalizer can only run once m is unreachable, which means every session and stepper that holds
	// *ComposedModel is gone too — so no live QuantWeight.Packed slice still aliases the mapping. That is
	// the safety guarantee for the unmap: no use-after-unmap is possible when the finalizer fires.
	runtime.SetFinalizer(m, (*ComposedModel).finalizeUnmap)
	return true
}

func (m *ComposedModel) finalizeUnmap() { _ = m.Close() }

// Close unmaps the checkpoint mapping backing this model's zero-copy packed weights. After Close the
// model's QuantWeight.Packed slices alias unmapped memory, so call ONLY when the model and all its
// sessions/steppers are done. Idempotent, and a no-op on a model that owns no mapping (dense/copied).
func (m *ComposedModel) Close() error {
	if m == nil || m.mmap == nil {
		return nil
	}
	c := m.mmap
	m.mmap = nil
	runtime.SetFinalizer(m, nil) // an explicit Close cancels the fallback finalizer
	return c.Close()
}

func silu(v float64) float64 { return v / (1 + math.Exp(-v)) }

// ProjMatMulInto is the device-GEMM seam for the composed stack's OWN projections — the attention
// mixer's q/k/v/o, the MLP/MoE matmuls, and the LM head (the largest single matmul of every decode
// step). Same AX-8 shape as qwen3/mamba2/rwkv7's hooks: the lib declares it and runs the host matNT
// by default; importing the native backend binds it to the steel GEMM. The device kernel accumulates
// in f32 (host is f64), so binding trades at the same numeric tier the gated-delta projections
// already serve at — the composed -state contract (a token-prefix snapshot, recomputed on wake) is
// deterministic per build either way.
var ProjMatMulInto func(out, x, w []float32, M, K, N int) ([]float32, error)

// ProjQuantMatMulInto is the quant twin of ProjMatMulInto: out[M,N] = x[M,K] @ dequant(w)ᵀ for an MLX
// affine-packed weight (packed uint32 codes + bf16 scales/biases, one scale+bias per group per row).
// N=outDim, K=inDim, groupSize divides K, bits is a shipped kernel width (2/3/4/5/6/8 — Bonsai's 1-bit is
// repacked to 2-bit at load). Same AX-8 shape as ProjMatMulInto: composed declares the hook and runs the
// host dequant-row fallback by default; importing the native backend binds it to the metallib's affine_qmv
// (M=1 decode) / affine_qmm_t (M>1 prefill) bf16 kernels — the composed serve boundary is already bf16 and
// bf16's mantissa is finer than the packed weight, so the weight quantisation, not the activation dtype,
// bounds the error. nil, or a device error, falls to matNTQuantHost — correct on any build, memory-safe
// (never widens the whole weight), but not fast.
var ProjQuantMatMulInto func(out, x []float32, packed, scales, biases []byte, M, K, N, groupSize, bits int) ([]float32, error)

// matNTQuant computes out[M,N] = x[M,K] @ dequant(qw)ᵀ for a packed projection weight, preferring the
// device quant seam and falling back to the host dequant-row path — a device failure (missing kernel, no
// Metal device) is deterministic for the rest of the process either way, mirroring matNTInto. qw's
// (OutDim, InDim) equal (N, K).
func matNTQuant(out, x []float32, qw *model.QuantWeight, M, K, N int) []float32 {
	if ProjQuantMatMulInto != nil {
		if res, err := ProjQuantMatMulInto(out, x, qw.Packed, qw.Scales, qw.Biases, M, K, N, qw.GroupSize, qw.Bits); err == nil {
			return res
		}
	}
	return matNTQuantHost(out, x, qw, M, K, N)
}

// ProjBF16MatMulInto is the dense bf16 matvec seam (#26) — the BF16Weight twin of
// ProjQuantMatMulInto: out[M,N] = x[M,K] @ wᵀ over the checkpoint's own bf16 bytes, no widening.
// AX-8: the lib declares the hook and runs the row-widen host fallback; the backend binds the
// device gemv. nil, or a device error, falls to matNTBF16Host.
var ProjBF16MatMulInto func(out, x []float32, w *model.BF16Weight, M, K, N int) ([]float32, error)

// matNTBF16 dispatches a dense bf16 projection: the device seam when bound, else the host
// row-widen reference — the exact matNTQuant contract for the unquantised form.
func matNTBF16(out, x []float32, bw *model.BF16Weight, M, K, N int) []float32 {
	if ProjBF16MatMulInto != nil {
		if res, err := ProjBF16MatMulInto(out, x, bw, M, K, N); err == nil {
			return res
		}
	}
	return matNTBF16Host(out, x, bw, M, K, N)
}

// matNTBF16Host widens one weight ROW at a time and dots it with each input row — never
// materialising the whole [N,K] f32 weight (the widening this seam exists to kill). f64
// accumulation in ascending-k order, matching matNTCols' tier; the correctness floor for
// non-metal builds and device declines.
func matNTBF16Host(out, x []float32, bw *model.BF16Weight, M, K, N int) []float32 {
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	row := make([]float32, K)
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			u := uint32(bw.Data[(n*K+k)*2]) | uint32(bw.Data[(n*K+k)*2+1])<<8
			row[k] = math.Float32frombits(u << 16)
		}
		for m := 0; m < M; m++ {
			xr := x[m*K : m*K+K]
			var acc float64
			for k := 0; k < K; k++ {
				acc += float64(xr[k]) * float64(row[k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

// matNTQuantHost is the host reference for a packed projection: for each output column n it dequantises
// that weight ROW (qw's row n, [K]) and dots it with each input row — never materialising the whole [N,K]
// f32 weight (a 27B head widened is ~5 GB). It is the device seam's error fallback and the whole quant
// forward on a non-metal build: correct, memory-safe, but slow (per-row dequant per call) — a correctness
// floor, not a serving path. f64 accumulation in ascending-k order, matching matNTCols so each out[m,n] is
// the same tier as the dense host GEMM (the device qmv differs only at quantisation precision — the parity
// gate's tolerance).
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
		if err != nil { // geometry is validated at load; a mismatch here means a caller passed the wrong K/N
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

// embedRow writes token id's embedding row [D] into dst (len D). Dense: a slice copy. Packed: a per-row
// host dequant of the embedding table — one row per token (cheap), never widening the whole [Vocab, D]
// table (~5 GB widened even at 4-bit). EmbedScale is applied by the caller.
func (m *ComposedModel) embedRow(dst []float32, id int) error {
	if m.EmbedQ != nil {
		wordsPerRow := mlxaffine.PackedWords(m.D, m.EmbedQ.Bits)
		groupsPerRow := m.D / m.EmbedQ.GroupSize
		wr, err := mlxaffine.DequantizeTensor(
			m.EmbedQ.Packed[id*wordsPerRow*4:(id+1)*wordsPerRow*4],
			m.EmbedQ.Scales[id*groupsPerRow*2:(id+1)*groupsPerRow*2],
			m.EmbedQ.Biases[id*groupsPerRow*2:(id+1)*groupsPerRow*2],
			1, m.D, m.EmbedQ.Bits, m.EmbedQ.GroupSize)
		if err != nil {
			return err
		}
		copy(dst, wr)
		return nil
	}
	if m.EmbedB != nil {
		base := id * m.D * 2
		for i := 0; i < m.D; i++ {
			u := uint32(m.EmbedB.Data[base+2*i]) | uint32(m.EmbedB.Data[base+2*i+1])<<8
			dst[i] = math.Float32frombits(u << 16)
		}
		return nil
	}
	copy(dst, m.Embed[id*m.D:id*m.D+m.D])
	return nil
}

// MLPDevice is the fused-SwiGLU device hook: gate/up GEMMs, the silu-and-multiply glue, and the
// down GEMM encoded into ONE command buffer with device-resident intermediates — one round-trip
// where the per-projection hook pays three. Same AX-8 shape as ProjMatMulInto; nil runs the
// per-projection path.
var MLPDevice func(gate, up, down, x []float32, L, D, FF int) ([]float32, error)

// ResidualNormMLPDevice is the fused pre-norm FFN-tail hook: the mixer-output residual add, the post-attn
// RMSNorm, the SwiGLU MLP and the MLP residual add encoded into ONE command buffer with device-resident
// intermediates — where the host path pays three passes over [L,D] (add, norm, add) bracketing the MLP's
// own command buffer. Given the pre-mixer hidden h and the mixer output mixOut, it returns the new hidden
// h = (h+mixOut) + SwiGLU(RMSNorm(h+mixOut, normW)). Same AX-8 shape as MLPDevice; nil — or a MoE FFN, or
// a sub-floor shape — runs the host add/norm/MLP/add path. Arch-neutral: every pre-norm SwiGLU stack has
// exactly this tail, so the backend primitive is named for the op, not for this stack.
var ResidualNormMLPDevice func(h, mixOut, normW, gate, up, down []float32, L, D, FF int, eps float32) ([]float32, error)

// ResidualNormMLPQuantDevice is ResidualNormMLPDevice's PACKED-weight twin (#8-B): the same fused
// pre-norm FFN tail — mixer residual, post-attn RMSNorm, SwiGLU MLP, MLP residual in ONE command
// buffer — with gate/up/down consumed as the checkpoint's own affine-packed codes (model.QuantWeight,
// never widened). This is the fold the quant bypass in forwardEmb could not take: without it a packed
// layer pays three separate quant-seam round trips (gate, up, down) bracketed by host glue. Residuals
// are plain adds, so the wiring only routes here at residualScale == 1 (the Qwen hybrids); nil — or a
// MoE FFN, a sub-floor shape, or a scaled-residual arch — keeps the host tail. Same AX-8 shape as the
// f32 hook: composed declares, the engine binds.
var ResidualNormMLPQuantDevice func(h, mixOut, normW []float32, gate, up, down *model.QuantWeight, L, D, FF int, eps float32) ([]float32, error)

// ResidualNormMLPProjDevice is ResidualNormMLPDevice with the mixer's FINAL projection (attention o_proj /
// gated-delta out_proj) folded onto the front: given the mixer's pre-projection hidden mixerHidden [L,mixCols]
// and that projection's weight projW [D,mixCols], it computes mixOut = mixerHidden@projWᵀ, then the same
// (h+mixOut) → RMSNorm → SwiGLU → (+mlpOut) tail, all in ONE command buffer. The projection's output never
// crosses the host floor — where the unfused path runs it as its own command buffer (a commit+wait per
// layer) before the tail's. The session uses this only for a mixer that implements projMixer AND a dense MLP
// above the device floor; nil — or a MoE FFN, or an error — falls back to the mixer's own projection + the
// standard tail (see forwardEmb). Same AX-8 shape as ResidualNormMLPDevice; arch-neutral (named for the op,
// not the mixer).
var ResidualNormMLPProjDevice func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error)

// ResidualNormMLPProjHeadDevice is ResidualNormMLPProjDevice with the MODEL's own terminal stage folded
// onto the BACK of that same tail instead of a next layer's input (there is no next layer — this hook
// only ever applies to the LAST layer): given the tail's output y [L,D], it also runs the final RMSNorm
// (NormF) over y's LAST row and the LM head GEMM (tied Embed or untied Output) against it, returning y
// [L,D] (still needed — the DecodeStepper/DecodeForward contract returns hidden, not logits) AND logits
// [Vocab] for that last row only — mirroring headLogits' single-row contract exactly. This is the mirror,
// at the OTHER end of the stack, of the next-layer INPUT fuses (…AttnInputDevice / …GatedDeltaInputDevice):
// those fold the NEXT layer's input norm+projections on; this folds the model's OWN output norm+head on,
// collapsing the head's own command buffer that ran after forwardEmb returned (N+1 → N per token). Same
// AX-8 shape; nil, an error, or any non-last layer all leave the plain proj-fused tail running with the
// head computed separately as today (see forwardEmb + ComposedSession.headLogits).
var ResidualNormMLPProjHeadDevice func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32, normF, head []float32, Vocab int) (y, logits []float32, err error)

// projMixer is the OPTIONAL capability of a mixer whose output is produced by a single final GEMM (the
// attention o_proj, the gated-delta out_proj): it runs the mixer up to but NOT including that projection and
// hands back the pre-projection hidden [L,mixCols] plus the projection weight projW [D,mixCols], so the
// session can fold the projection into the FFN-tail command buffer (one CB where the mixer-owned projection
// would otherwise be a second). forwardNoProj advances the mixer state exactly as Forward does — the ONLY
// difference is that the caller now owns the final projection. A mixer without a single-GEMM output does not
// implement this and the session runs the standard Forward → tail path.
type projMixer interface {
	forwardNoProj(hidden []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error)
}

// deviceMinWork is the M·K·N floor below which matNTInto ignores the device hook — a tiny GEMV's
// command-buffer round-trip outweighs its compute, so sub-MMAC shapes stay on the host path.
const deviceMinWork = 1 << 20

// matNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ), f32 host.
func matNT(in, w []float32, M, K, N int) []float32 {
	return matNTInto(nil, in, w, M, K, N)
}

// matNTInto is matNT writing into out, reusing it when cap(out) ≥ M·N (else it allocates a fresh M·N
// slab). Identical f64 accumulation + write order to the fresh-buffer form — only WHERE the result lands
// changes, so the output is bit-identical. The attention mixer threads a caller-owned scratch through
// this for its per-token q/k/v/o projections.
//
// Large shapes shard the OUTPUT COLUMNS across CPU cores: every out[m·N+n] keeps exactly the serial
// per-element k-accumulation order (only which goroutine computes it changes), so the sharded form is
// bit-identical too — the composed -state byte-identity contract holds. Small shapes stay serial: the
// goroutine fan-out costs more than it saves below the work floor (tests' toy shapes, tiny projections).
func matNTInto(out, in, w []float32, M, K, N int) []float32 {
	if ProjMatMulInto != nil && M*K*N >= deviceMinWork {
		if res, err := ProjMatMulInto(out, in, w, M, K, N); err == nil {
			return res
		}
		// A device failure (missing kernel, no Metal device) falls through to the
		// host path — deterministic for the rest of the process either way.
	}
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

// rmsNormRowsPlain RMS-norms each of the `rows` rows of x [rows,d] by the shared plain weight w [d].
func rmsNormRowsPlain(x, w []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		xr := x[r*d : (r+1)*d]
		var ss float64
		for i := range d {
			ss += float64(xr[i]) * float64(xr[i])
		}
		rms := math.Sqrt(ss/float64(d) + float64(eps))
		for i := range d {
			out[r*d+i] = float32(float64(xr[i]) / rms * float64(w[i]))
		}
	}
	return out
}

func layerNormRowsPlain(x, w []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		xr := x[r*d : (r+1)*d]
		var mean float64
		for _, value := range xr {
			mean += float64(value)
		}
		mean /= float64(d)
		var variance float64
		for _, value := range xr {
			delta := float64(value) - mean
			variance += delta * delta
		}
		inv := 1 / math.Sqrt(variance/float64(d)+float64(eps))
		for i := range d {
			out[r*d+i] = float32((float64(xr[i]) - mean) * inv * float64(w[i]))
		}
	}
	return out
}

func (m *ComposedModel) normalise(x, w []float32, rows int) []float32 {
	if m.LayerNorm {
		return layerNormRowsPlain(x, w, rows, m.D, m.Eps)
	}
	return rmsNormRowsPlain(x, w, rows, m.D, m.Eps)
}

// swiglu runs the SwiGLU MLP over x [L,D] → [L,D].
func (mlp *MLP) forward(x []float32, L, D int) []float32 {
	if mlp.GateB != nil { // bf16-resident MLP (#26): gate/up/down through the bf16 matvec seam
		g := matNTBF16(nil, x, mlp.GateB, L, D, mlp.FF)
		u := matNTBF16(nil, x, mlp.UpB, L, D, mlp.FF)
		h := make([]float32, L*mlp.FF)
		for i := range h {
			h[i] = float32(silu(float64(g[i])) * float64(u[i]))
		}
		return matNTBF16(nil, h, mlp.DownB, L, mlp.FF, D)
	}
	if mlp.GateQ != nil { // packed MLP: gate/up/down through the quant matvec seam (the fused MLPDevice is f32-only)
		g := matNTQuant(nil, x, mlp.GateQ, L, D, mlp.FF)
		u := matNTQuant(nil, x, mlp.UpQ, L, D, mlp.FF)
		h := make([]float32, L*mlp.FF)
		for i := range h {
			h[i] = float32(silu(float64(g[i])) * float64(u[i]))
		}
		return matNTQuant(nil, h, mlp.DownQ, L, mlp.FF, D)
	}
	if MLPDevice != nil && L*D*mlp.FF >= deviceMinWork {
		if out, err := MLPDevice(mlp.Gate, mlp.Up, mlp.Down, x, L, D, mlp.FF); err == nil {
			return out
		}
	}
	g := matNT(x, mlp.Gate, L, D, mlp.FF) // [L,FF]
	u := matNT(x, mlp.Up, L, D, mlp.FF)   // [L,FF]
	h := make([]float32, L*mlp.FF)
	for i := range h {
		h[i] = float32(silu(float64(g[i])) * float64(u[i]))
	}
	return matNT(h, mlp.Down, L, mlp.FF, D) // [L,D]
}

// ComposedSession is a recurrent decode session over a ComposedModel: per-layer mixer state, threaded
// across forward calls. Single-goroutine.
type ComposedSession struct {
	m      *ComposedModel
	states []any // per-layer opaque mixer state; nil ⇒ fresh

	// pendingHeadLogits is set by forwardEmb only when the LAST layer's tail fused the model's final
	// RMSNorm + LM head GEMM in (ResidualNormMLPProjHeadDevice succeeded) — the vocab logits for that
	// call's last row, ready for the caller to reuse instead of paying a second head command buffer.
	// Reset to nil at the top of every forwardEmb call, so a stale value never survives past the call
	// that didn't produce one. See PendingHeadLogits.
	pendingHeadLogits []float32
}

// PendingHeadLogits returns the vocab logits the MOST RECENT forwardEmb call (Forward/forward) computed
// as a side effect of fusing the model's final RMSNorm + LM head GEMM onto the LAST layer's tail command
// buffer — nil when that fuse didn't apply (hook unbound, a sub-floor MLP, a device error, or the model
// has no layers). A stepper that wants the fused fast path reads this right after its own forwardEmb call
// (see composed/token_model.go's composedStepper.Step); the native backend's parity test reads it too.
func (s *ComposedSession) PendingHeadLogits() []float32 { return s.pendingHeadLogits }

// NewSession builds a fresh session (each layer's mixer state starts empty).
func NewSession(m *ComposedModel) *ComposedSession {
	return &ComposedSession{m: m, states: make([]any, len(m.Layers))}
}

// Snapshot returns a deep copy of the session's per-layer recurrent state — a rollback point. Each
// non-nil layer state is cloned via that layer's own Mixer.CloneState, so a later advance of the live
// session (forwardEmb always REPLACES s.states[li] with a freshly-produced next state rather than
// mutating the old one) can never reach back and perturb an already-taken snapshot.
func (s *ComposedSession) Snapshot() []any {
	snap := make([]any, len(s.states))
	for li, st := range s.states {
		if st == nil {
			continue
		}
		snap[li] = s.m.Layers[li].Mixer.CloneState(st)
	}
	return snap
}

// Restore rolls the session back to a prior Snapshot: s.states becomes an independent deep copy of snap
// (via the same per-layer Mixer.CloneState Snapshot itself uses), so neither the caller's snap slice nor
// its element states end up aliased by s.states — the caller may keep advancing the restored session,
// take a fresh Snapshot, or Restore(snap) again, without any of those perturbing one another.
func (s *ComposedSession) Restore(snap []any) {
	states := make([]any, len(snap))
	for li, st := range snap {
		if st == nil {
			continue
		}
		states[li] = s.m.Layers[li].Mixer.CloneState(st)
	}
	s.states = states
}

// pendingAttnQKV carries a NEXT layer's ALREADY-COMPUTED input RMSNorm + q/k/v projections, folded into
// the PREVIOUS layer's proj-fused tail command buffer by ResidualNormMLPProjAttnInputDevice — the
// input-side mirror of the o_proj fuse. The layer at that index resumes via attnMixer.forwardFromQKV
// instead of recomputing its input norm + projections.
type pendingAttnQKV struct{ q, k, v []float32 }

// pendingGatedDeltaInput is pendingAttnQKV's gated-delta counterpart — set by
// ResidualNormMLPProjGatedDeltaInputDevice; the layer resumes via gatedDeltaMixer.forwardFromInput.
type pendingGatedDeltaInput struct{ qkv, z, a, b []float32 }
type pendingMamba2Input struct{ projected []float32 }
type pendingRWKV7Input struct{ r, w, k, v, a, b []float32 }

// ComposedChainBeginDevice / ComposedChainEndDevice bracket a whole-token chained forward (#26):
// Begin uploads the hidden once and returns the opaque context every chained layer encodes into;
// End commits, waits ONCE, and returns the final hidden. AX-8: declared here, bound by the backend.
var ComposedChainBeginDevice func(h []float32, L, D int) (ctx any, err error)
var ComposedChainEndDevice func(ctx any) (y []float32, err error)

// chainable reports whether EVERY layer of this model can ride the chained device path — the
// all-or-nothing v1 gate (mixed models keep the per-layer folds). Form-agnostic (#26 QUANT): each
// layer qualifies by carrying EITHER its bf16 form OR its packed form ready for the chain step —
// a model may mix bf16-resident and quantised layers, so long as every layer is one or the other.
func (s *ComposedSession) chainable() bool {
	if ComposedChainBeginDevice == nil || ComposedChainEndDevice == nil || s.m.residualScale() != 1 || s.m.LayerNorm {
		return false
	}
	for li := range s.m.Layers {
		layer := &s.m.Layers[li]
		mlp, isDense := layer.MLP.(*MLP)
		if !isDense {
			return false
		}
		switch mx := layer.Mixer.(type) {
		case *gatedDeltaMixer:
			if !mx.chainableBF16(mlp) && !mx.chainableQuant(mlp) {
				return false
			}
		case *attnMixer:
			if !mx.chainableBF16(mlp) && !mx.chainableQuant(mlp) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// forwardChain runs the whole stack as ONE chained device forward: one upload, one wait. Each
// layer dispatches to its bf16 or packed chain step depending on which weight form it carries
// (chainable already proved every layer is ready in one form or the other). Any error mid-chain
// aborts the forward — the device owns every layer's state by then.
func (s *ComposedSession) forwardChain(h []float32, L int) ([]float32, error) {
	D, eps := s.m.D, s.m.Eps
	ctx, err := ComposedChainBeginDevice(h, L, D)
	if err != nil {
		return nil, err
	}
	for li := range s.m.Layers {
		layer := &s.m.Layers[li]
		mlp := layer.MLP.(*MLP)
		switch mx := layer.Mixer.(type) {
		case *gatedDeltaMixer:
			var next any
			var cerr error
			if mx.chainableBF16(mlp) {
				next, cerr = mx.chainBF16Layer(ctx, layer.InputNorm, layer.PostAttnNorm, mlp, eps, s.states[li])
			} else {
				next, cerr = mx.chainQuantLayer(ctx, layer.InputNorm, layer.PostAttnNorm, mlp, eps, s.states[li])
			}
			if cerr != nil {
				return nil, cerr
			}
			s.states[li] = next
		case *attnMixer:
			var next any
			var cerr error
			if mx.chainableBF16(mlp) {
				next, cerr = mx.chainBF16Layer(ctx, layer.InputNorm, layer.PostAttnNorm, mlp, L, D, eps, s.states[li])
			} else {
				next, cerr = mx.chainQuantLayer(ctx, layer.InputNorm, layer.PostAttnNorm, mlp, L, D, eps, s.states[li])
			}
			if cerr != nil {
				return nil, cerr
			}
			s.states[li] = next
		}
	}
	return ComposedChainEndDevice(ctx)
}

// forwardEmb runs L input embeddings [L,D] through the stack, advancing each layer's mixer state, and
// returns the output hiddens [L,D]. Serves both prefill (L>1) and decode (L=1).
func (s *ComposedSession) forwardEmb(h []float32, L int) ([]float32, error) {
	D, eps := s.m.D, s.m.Eps
	s.pendingHeadLogits = nil // cleared every call; set only when the last layer's tail fuses the head in

	// pendingAttn / pendingGD carry a NEXT layer's already-computed input RMSNorm + input projections —
	// folded into the PREVIOUS layer's proj-fused tail command buffer, the symmetric collapse to the
	// o_proj fuse below. At most one is ever set (a layer has exactly one mixer kind); both nil ⇒ this
	// layer computes its input norm + projections fresh (always true for layer 0 — no predecessor tail).
	if (s.m.BF16Resident || s.m.Quantised) && s.chainable() {
		return s.forwardChain(h, L)
	}

	var pendingAttn *pendingAttnQKV
	var pendingGD *pendingGatedDeltaInput
	var pendingMamba2 *pendingMamba2Input
	var pendingRWKV7 *pendingRWKV7Input

	for li := range s.m.Layers {
		layer := &s.m.Layers[li]
		if s.m.ParallelResidual {
			normed := s.m.normalise(h, layer.InputNorm, L)
			mixOut, next, err := layer.Mixer.Forward(normed, L, D, s.states[li])
			if err != nil {
				return nil, err
			}
			s.states[li] = next
			mlpOut := layer.MLP.forward(normed, L, D)
			scale := s.m.residualScale()
			for i := range h {
				h[i] += scale * (mixOut[i] + mlpOut[i])
			}
			continue
		}

		if s.m.Quantised || s.m.BF16Resident {
			// Whole-layer device fold (#18 S3): a packed gated-delta layer with a dense packed SwiGLU
			// at residual scale 1 rides ONE command buffer — input norm, the five packed projections,
			// the gated-delta block (state device-resident) and the FFN tail; x is the only upload, y
			// the only readback. engaged=false (hook unbound, geometry unservable, first-call decline)
			// leaves the per-stage quant branch below in charge.
			if gm, isGD := layer.Mixer.(*gatedDeltaMixer); isGD && s.m.residualScale() == 1 {
				if mlp, isDense := layer.MLP.(*MLP); isDense {
					switch {
					case mlp.GateQ != nil && mlp.UpQ != nil && mlp.DownQ != nil:
						y, next, engaged, qerr := gm.forwardQuantLayer(h, layer.InputNorm, layer.PostAttnNorm, mlp.GateQ, mlp.UpQ, mlp.DownQ, mlp.FF, L, D, eps, s.states[li])
						if engaged {
							if qerr != nil {
								return nil, qerr
							}
							s.states[li] = next
							h = y
							continue
						}
					case mlp.GateB != nil && mlp.UpB != nil && mlp.DownB != nil:
						y, next, engaged, berr := gm.forwardBF16Layer(h, layer.InputNorm, layer.PostAttnNorm, mlp.GateB, mlp.UpB, mlp.DownB, mlp.FF, L, D, eps, s.states[li])
						if engaged {
							if berr != nil {
								return nil, berr
							}
							s.states[li] = next
							h = y
							continue
						}
					}
				}
			}
			// Attention fold (#26): a dense bf16 attention layer rides [norm+q/k/v CB] → host
			// rope/cache/SDPA → [o_proj+FFN-tail CB] — two round-trips where the per-stage path
			// pays seven. engaged=false falls through to the per-stage branch below.
			if am, isAttn := layer.Mixer.(*attnMixer); isAttn && s.m.residualScale() == 1 && !s.m.LayerNorm {
				if mlp, isDense := layer.MLP.(*MLP); isDense {
					y, next, engaged, aerr := am.forwardBF16Layer(h, layer.InputNorm, layer.PostAttnNorm, mlp, L, D, eps, s.states[li])
					if !engaged {
						y, next, engaged, aerr = am.forwardQuantLayer(h, layer.InputNorm, layer.PostAttnNorm, mlp, L, D, eps, s.states[li])
					}
					if engaged {
						if aerr != nil {
							return nil, aerr
						}
						s.states[li] = next
						h = y
						continue
					}
				}
			}
			// Packed checkpoint: the mixer's own Forward (its q/k/v/o or in_proj/out_proj dispatch to the
			// quant matvec seam) then the FFN tail. The tail rides the fused packed-weight device fold
			// (ResidualNormMLPQuantDevice — residual + norm + SwiGLU-over-codes + residual in ONE command
			// buffer, #8-B) when the layer is a plain packed SwiGLU at residual scale 1; a MoE FFN (its
			// routed + shared experts dispatch packed experts per swigluExpertQuantInto), a sub-floor
			// shape, a scaled-residual arch or a device decline keeps the host tail, whose gate/up/down
			// dispatch to the per-projection seam exactly as before.
			normed := rmsNormRowsPlain(h, layer.InputNorm, L, D, eps)
			mixOut, next, err := layer.Mixer.Forward(normed, L, D, s.states[li])
			if err != nil {
				return nil, err
			}
			s.states[li] = next
			if mlp, ok := layer.MLP.(*MLP); ok && ResidualNormMLPQuantDevice != nil &&
				mlp.GateQ != nil && mlp.UpQ != nil && mlp.DownQ != nil &&
				s.m.residualScale() == 1 && L*D*mlp.FF >= deviceMinWork {
				if y, derr := ResidualNormMLPQuantDevice(h, mixOut, layer.PostAttnNorm, mlp.GateQ, mlp.UpQ, mlp.DownQ, L, D, mlp.FF, eps); derr == nil {
					h = y
					continue
				}
			}
			h = tailHost(h, mixOut, layer.PostAttnNorm, layer.MLP, L, D, eps, s.m.residualScale())
			continue
		}

		// Resolve this layer's (mixerHidden, projW, mixCols, next-state): either resume from a pending
		// input-fuse (skipping this layer's own RMSNorm + projection step — both mixer kinds implement
		// projMixer, so a pending resume always lands here) or run the mixer's forwardNoProj off a
		// freshly-computed input RMSNorm.
		var mixerHidden, projW []float32
		var mixCols int
		var next any
		var err error
		isProj := false

		switch {
		case pendingAttn != nil:
			p := pendingAttn
			pendingAttn = nil
			am := layer.Mixer.(*attnMixer) // guaranteed: only ever set for an *attnMixer next layer
			mixerHidden, projW, mixCols, next, err = am.forwardFromQKV(p.q, p.k, p.v, L, D, s.states[li])
			isProj = true
		case pendingGD != nil:
			p := pendingGD
			pendingGD = nil
			gm := layer.Mixer.(*gatedDeltaMixer) // guaranteed: only ever set for a *gatedDeltaMixer next layer
			mixerHidden, projW, mixCols, next, err = gm.forwardFromInput(p.qkv, p.z, p.a, p.b, L, D, s.states[li])
			isProj = true
		case pendingMamba2 != nil:
			p := pendingMamba2
			pendingMamba2 = nil
			mm := layer.Mixer.(*mamba2Mixer)
			mixerHidden, projW, mixCols, next, err = mm.forwardFromInput(p.projected, L, D, s.states[li])
			isProj = true
		case pendingRWKV7 != nil:
			p := pendingRWKV7
			pendingRWKV7 = nil
			rm := layer.Mixer.(*rwkv7Mixer)
			mixerHidden, projW, mixCols, next, err = rm.forwardFromInput(p.r, p.w, p.k, p.v, p.a, p.b, L, D, s.states[li])
			isProj = true
		default:
			if pm, ok := layer.Mixer.(projMixer); ok {
				normed := rmsNormRowsPlain(h, layer.InputNorm, L, D, eps)
				mixerHidden, projW, mixCols, next, err = pm.forwardNoProj(normed, L, D, s.states[li])
				isProj = true
			}
		}
		if err != nil {
			return nil, err
		}

		if isProj {
			s.states[li] = next
			if mlp, isDense := layer.MLP.(*MLP); isDense && ResidualNormMLPProjDevice != nil && L*D*mlp.FF >= deviceMinWork {
				// Input-fuse: when the NEXT layer exists and is itself a full-attention or gated-delta
				// mixer, fold its input RMSNorm + input projections onto the BACK of this layer's
				// proj+tail command buffer — the symmetric collapse to the o_proj fuse. y never crosses
				// the host floor before feeding the next layer's projections; only y and the projections
				// do. The projections ride free whatever their own size (the tail CB is already paid
				// for); a nil hook, an unmatched next-mixer kind, the last layer, or a device error all
				// fall through to the plain proj-fused tail below.
				if li+1 < len(s.m.Layers) {
					nextLayer := &s.m.Layers[li+1]
					switch nm := nextLayer.Mixer.(type) {
					case *attnMixer:
						if ResidualNormMLPProjAttnInputDevice != nil {
							qCols := nm.cfg.Heads * nm.cfg.HeadDim
							if nm.cfg.OutputGate {
								qCols *= 2
							}
							kvCols := nm.cfg.KVHeads * nm.cfg.HeadDim
							if y, q, k, v, ferr := ResidualNormMLPProjAttnInputDevice(
								mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
								nextLayer.InputNorm, nm.w.QProj, nm.w.KProj, nm.w.VProj, qCols, kvCols,
							); ferr == nil {
								h = y
								pendingAttn = &pendingAttnQKV{q: q, k: k, v: v}
								continue
							}
						}
					case *gatedDeltaMixer:
						if ResidualNormMLPProjGatedDeltaInputDevice != nil {
							convDim, vDim, VH := nm.cfg.ConvDim(), nm.cfg.VDim(), nm.cfg.ValueHeads
							if y, qkv, z, a, b, ferr := ResidualNormMLPProjGatedDeltaInputDevice(
								mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
								nextLayer.InputNorm, nm.w.InProjQKV, nm.w.InProjZ, nm.w.InProjA, nm.w.InProjB, convDim, vDim, VH,
							); ferr == nil {
								h = y
								pendingGD = &pendingGatedDeltaInput{qkv: qkv, z: z, a: a, b: b}
								continue
							}
						}
					case *mamba2Mixer:
						if ResidualNormMLPProjMamba2InputDevice != nil {
							projDim := 2*nm.cfg.NumHeads*nm.cfg.HeadDim + 2*nm.cfg.NumGroups*nm.cfg.StateDim + nm.cfg.NumHeads
							if y, projected, ferr := ResidualNormMLPProjMamba2InputDevice(
								mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
								nextLayer.InputNorm, nm.w.InProj, projDim,
							); ferr == nil {
								h = y
								pendingMamba2 = &pendingMamba2Input{projected: projected}
								continue
							}
						}
					case *rwkv7Mixer:
						if ResidualNormMLPProjRWKV7InputDevice != nil {
							hk := nm.cfg.NumHeads * nm.cfg.KeyDim
							hv := nm.cfg.NumHeads * nm.cfg.ValueDim
							if y, r, w, k, v, a, b, ferr := ResidualNormMLPProjRWKV7InputDevice(
								mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
								nextLayer.InputNorm, nm.w.RProj, nm.w.WProj, nm.w.KProj, nm.w.VProj, nm.w.AProj, nm.w.BProj, hk, hv,
							); ferr == nil {
								h = y
								pendingRWKV7 = &pendingRWKV7Input{r: r, w: w, k: k, v: v, a: a, b: b}
								continue
							}
						}
					}
				} else if ResidualNormMLPProjHeadDevice != nil {
					// Terminal collapse: this IS the last layer, so there is no next-layer input to
					// fuse — fold the MODEL's own final RMSNorm + LM head GEMM onto the back of this
					// tail instead. The head's own separate command buffer (today's
					// ComposedSession.headLogits, called by the caller once this returns) disappears —
					// N+1 → N command buffers per token. A nil hook or a device error leaves the plain
					// proj-fused tail below to run and the head computed separately as today.
					headW := s.m.Output
					if headW == nil {
						headW = s.m.Embed // tied head
					}
					if y, logits, ferr := ResidualNormMLPProjHeadDevice(
						mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
						s.m.NormF, headW, s.m.Vocab,
					); ferr == nil {
						h = y
						s.pendingHeadLogits = logits
						continue
					}
				}
				// Plain proj-fused tail (no next-input fuse, no head fuse, or either failed): o_proj/
				// out_proj → residual → post-attn RMSNorm → SwiGLU → residual, in one command buffer.
				if y, ferr := ResidualNormMLPProjDevice(mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps); ferr == nil {
					h = y
					continue
				}
			}
			// Fallback — covers a failed proj-fused tail attempt above AND a projMixer layer whose MLP
			// isn't dense/above-floor (including one reached via a pending resume): complete the mixer's
			// own projection on the host, then the plain device tail-fuse, else host tail.
			mixOut := matNT(mixerHidden, projW, L, mixCols, D)
			if mlp, ok := layer.MLP.(*MLP); ok && ResidualNormMLPDevice != nil && L*D*mlp.FF >= deviceMinWork {
				if y, err := ResidualNormMLPDevice(h, mixOut, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mlp.FF, eps); err == nil {
					h = y
					continue
				}
			}
			h = tailHost(h, mixOut, layer.PostAttnNorm, layer.MLP, L, D, eps, s.m.residualScale())
			continue
		}

		// Standard path: the mixer doesn't implement projMixer at all — it owns its final projection
		// outright; the FFN tail fuses on its own device CB (h += mixOut → RMSNorm → SwiGLU → h += mlpOut)
		// for a dense MLP above the floor, else host.
		normed := rmsNormRowsPlain(h, layer.InputNorm, L, D, eps)
		mixOut, next, err := layer.Mixer.Forward(normed, L, D, s.states[li])
		if err != nil {
			return nil, err
		}
		s.states[li] = next
		if mlp, ok := layer.MLP.(*MLP); ok && ResidualNormMLPDevice != nil && L*D*mlp.FF >= deviceMinWork {
			if y, err := ResidualNormMLPDevice(h, mixOut, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mlp.FF, eps); err == nil {
				h = y
				continue
			}
		}
		h = tailHost(h, mixOut, layer.PostAttnNorm, layer.MLP, L, D, eps, s.m.residualScale())
	}
	return h, nil
}

// tailHost runs the FFN tail on the host — the mixer-output residual add, the post-attn RMSNorm, the FFN
// (dense SwiGLU or MoE) and the MLP residual add — returning the new hidden. It is the shared fallback for
// both the projection-fused and standard paths when the device tail hook is nil or errors; a MoE FFN always
// lands here (no fused-MLP device kernel).
func tailHost(h, mixOut, normW []float32, ffn FFN, L, D int, eps, residualScale float32) []float32 {
	for i := range h {
		h[i] += residualScale * mixOut[i] // mixer residual
	}
	normed := rmsNormRowsPlain(h, normW, L, D, eps)
	mlpOut := ffn.forward(normed, L, D)
	for i := range h {
		h[i] += residualScale * mlpOut[i] // MLP residual
	}
	return h
}

func (m *ComposedModel) residualScale() float32 {
	if m.ResidualScale == 0 {
		return 1
	}
	return m.ResidualScale
}

// forward embeds tokens then runs the stack.
func (s *ComposedSession) forward(tokens []int32) ([]float32, error) {
	L, D := len(tokens), s.m.D
	h := make([]float32, L*D)
	for t, tok := range tokens {
		if int(tok) < 0 || int(tok) >= s.m.Vocab {
			return nil, core.NewError("composed.forward: token out of range")
		}
		if err := s.m.embedRow(h[t*D:(t+1)*D], int(tok)); err != nil {
			return nil, err
		}
		if s.m.EmbedScale != 0 && s.m.EmbedScale != 1 {
			for i := t * D; i < (t+1)*D; i++ {
				h[i] *= s.m.EmbedScale
			}
		}
	}
	return s.forwardEmb(h, L)
}

// Forward prefills tokens and returns the per-position hiddens [L,D] (state advanced).
func (s *ComposedSession) Forward(tokens []int32) ([]float32, error) { return s.forward(tokens) }

// headLogits maps a single hidden [D] to vocab logits via the final norm + LM head.
func (s *ComposedSession) headLogits(hidden []float32) []float32 {
	normed := s.m.normalise(hidden, s.m.NormF, 1)
	var logits []float32
	switch {
	case s.m.OutputQ != nil: // packed untied head
		logits = matNTQuant(nil, normed, s.m.OutputQ, 1, s.m.D, s.m.Vocab)
	case s.m.OutputB != nil: // bf16-resident untied head (#26)
		logits = matNTBF16(nil, normed, s.m.OutputB, 1, s.m.D, s.m.Vocab)
	case s.m.Output == nil && s.m.EmbedQ != nil: // packed embed, tied head
		logits = matNTQuant(nil, normed, s.m.EmbedQ, 1, s.m.D, s.m.Vocab)
	case s.m.Output == nil && s.m.EmbedB != nil: // bf16-resident embed, tied head (#26)
		logits = matNTBF16(nil, normed, s.m.EmbedB, 1, s.m.D, s.m.Vocab)
	default:
		head := s.m.Output
		if head == nil {
			head = s.m.Embed
		}
		logits = matNT(normed, head, 1, s.m.D, s.m.Vocab)
	}
	if s.m.LogitScale != 0 {
		for i := range logits {
			logits[i] *= s.m.LogitScale
		}
	}
	if s.m.LogitsScaling != 0 && s.m.LogitsScaling != 1 {
		for i := range logits {
			logits[i] /= s.m.LogitsScaling
		}
	}
	return logits
}

// HeadLogitsHost is headLogits, exported: the reference final-RMSNorm + LM-head computation
// ResidualNormMLPProjHeadDevice's fused path is checked against, reachable across the package boundary
// without a throwaway session of one's own (headLogits needs none of a session's per-layer mixer state —
// it is a pure function of hidden + the model's own weights). Exists for the native backend's
// device-vs-host parity test.
func HeadLogitsHost(m *ComposedModel, hidden []float32) []float32 {
	return NewSession(m).headLogits(hidden)
}

// Generate greedily decodes up to maxNew tokens after prefilling prompt, threading every layer's mixer
// state. eosID < 0 disables early stop.
func (s *ComposedSession) Generate(prompt []int32, maxNew, eosID int) ([]int32, error) {
	if len(prompt) == 0 || maxNew <= 0 {
		return nil, core.NewError("composed.Generate: empty prompt or maxNew<=0")
	}
	h, err := s.forward(prompt)
	if err != nil {
		return nil, err
	}
	D := s.m.D
	last := h[(len(prompt)-1)*D:]
	gen := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		next := argmaxF32(s.headLogits(last))
		gen = append(gen, next)
		if eosID >= 0 && int(next) == eosID {
			break
		}
		h1, err := s.forward([]int32{next})
		if err != nil {
			return nil, err
		}
		last = h1
	}
	return gen, nil
}

func argmaxF32(v []float32) int32 {
	best, bi := v[0], int32(0)
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best, bi = v[i], int32(i)
		}
	}
	return bi
}
