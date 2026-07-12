// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// audio.go ports the gemma4 Conformer audio tower to the no-cgo native path — the faithful
// translation of metal's audio_encoder.go, composed from native's byte-parity kernels (on-device
// matmuls + the byte-identical Conv2d/LayerNorm/RMSNorm/SiLU/Clip helpers). The blocks are
// BYTE-IDENTICAL to pkg/metal (eqBytes-verified), NOT a tolerance match — see audio_test.go. Per-
// linear activation clamps (ClipPair) are byte-identical when the checkpoint stores them in the
// model dtype (bf16); f32 clamp arrays would promote the projection to fp32 in metal — handle that
// at load if a checkpoint is found to use them. Engine-neutral: no model name; geometry arrives as
// AudioConfig. Shares the bf16↔fp32 + rmsNormVec + MatRowsBF16 helpers with vision.go.
//
// Split with the engine-neutral host tower (model/gemma4/audio, #44 convergence): that package is the
// pure-host REFERENCE forward (no GPU dependency) plus the hip binding; this file (+ audio_f32.go,
// audio_attention.go, audio_encoder.go) is the GPU-dispatch SPECIALISATION, byte-identical to the real
// metal.Forward — its clamp/activation/matmul calls route through on-device kernels (RMSNormBF16,
// MatRowsBF16, MatMulF32NT, Sigmoid, …), so its composed forwards cannot ride the shared home's plain
// host loops without floating-point drift and stay local. The one piece that WAS provably duplicate and
// genuinely GPU-free — the mel front-end — converged onto the shared home (audio_features.go is now a
// thin alias/delegate). A handful of small pure-index-remap/host-arithmetic leaves inside this GPU tower
// (audioBlockContextF32, audioRelShiftF32Into, audioBlockedMask, clampF32, reluF32, halveValidity, …in
// audio_f32.go/audio_attention.go) are ALSO algorithmically identical to their shared-home counterparts,
// but stay local rather than being exported from the reference package purely to serve this one GPU
// consumer — the honest-partial-convergence call for #44 (see the worktree report). Four confirmed-dead
// duplicate helpers (audioContextSize method, audioClamp, audioActivate, rmsRowsHost — never called
// outside their own tests, fully superseded by audioContextSizeOf/clampF32|clampBF16/
// audioActivateF32|audioActivateBF16/the GPU RMSNorm path) were deleted outright.

// AudioConfig is the engine-neutral Conformer geometry the forward reads. ClipMin/ClipMax are the
// ±gradient-clipping clamp every module borrows (ClipMin==ClipMax ⇒ no clamp). Act is the FF/conv
// activation ("silu"/"swish"/""→SiLU, "relu", "gelu"/"gelu_pytorch_tanh").
type AudioConfig struct {
	Hidden     int
	FFInter    int
	Channels   int // LightConv conv channels (== Hidden for gemma4 audio)
	KernelSize int // LightConv depthwise conv1d kernel
	Eps        float32
	Act        string
	FFResidual float32
	ClipMin    float32
	ClipMax    float32

	// Relative-position attention geometry (the chunked Conformer attention).
	NumHeads      int
	HeadDim       int
	ChunkSize     int
	PastHorizon   int // ContextLeft-1
	FutureHorizon int // ContextRight
	KScale        float32
	LogitCap      float32 // tanh soft-cap
	InvalidLogit  float32 // masked-position fill
}

// AudioFeedForwardWeights is one Conformer FeedForward's bf16 weight views: pre/post RMSNorm [hidden]
// and the two linears FFW1 [inter,hidden], FFW2 [hidden,inter]. (gemma4 audio FF linears carry no
// per-linear input/output clip — the FF-level gradient clamp is the active one.)
type AudioFeedForwardWeights struct {
	PreNorm, PostNorm  []byte
	FFW1, FFW2         []byte
	FFW1Clip, FFW2Clip ClipPair // optional per-linear activation clamps (zero value = none)
}

// clampBF16 is the byte-parity bf16 clamp to [min,max] — metal.Clip is a SELECT (no arithmetic), so
// the host comparison on bf16 values gives identical bytes: in-range elements keep their original
// bytes, clipped elements become bf16(min)/bf16(max). min==max ⇒ pass-through.
func clampBF16(b []byte, min, max float32) []byte {
	if min == max {
		return b
	}
	out := make([]byte, len(b))
	copy(out, b)
	for i := 0; i+1 < len(b); i += bf16Size {
		v := bf16ToF32(b[i], b[i+1])
		var h uint16
		switch {
		case v < min:
			h = f32ToBF16(min)
		case v > max:
			h = f32ToBF16(max)
		default:
			continue
		}
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// ClipBound is one optional per-linear activation clamp (metal's input_min/input_max or
// output_min/output_max scalars on a Gemma4AudioClippableLinear). Present=false leaves the activation
// untouched — byte-for-byte the metal path when the clamp array is nil (the checkpoint omits it).
type ClipBound struct {
	Min, Max float32
	Present  bool
}

// applyBF16 clamps when present (metal.Clip is a select, so clampBF16 is byte-identical).
func (c ClipBound) applyBF16(b []byte) []byte {
	if !c.Present {
		return b
	}
	return clampBF16(b, c.Min, c.Max)
}

// ClipPair is a clippable linear's input + output clamps — the no-cgo equivalent of
// Gemma4AudioClippableLinear's {InputMin,InputMax} / {OutputMin,OutputMax}. Zero value = no clamp.
type ClipPair struct{ In, Out ClipBound }

// clippedMatRowsBF16 is ClippableLinear.Forward: clip input → MatRowsBF16 → clip output, each clamp
// applied only when present (matching metal's nil-guarded Clip).
func clippedMatRowsBF16(weight, x []byte, L, outDim, inDim int, clip ClipPair) ([]byte, error) {
	out, err := MatRowsBF16(weight, clip.In.applyBF16(x), L, outDim, inDim)
	if err != nil {
		return nil, err
	}
	return clip.Out.applyBF16(out), nil
}

// mulScalarBF16 multiplies every bf16 element by the f32 scalar s, rounding once to bf16 — the same
// bf16-in / f32-scalar / bf16-out computation as metal.MulScalar (verified eqBytes).
func mulScalarBF16(b []byte, s float32) []byte {
	out := make([]byte, len(b))
	for i := 0; i+1 < len(b); i += bf16Size {
		h := f32ToBF16(bf16ToF32(b[i], b[i+1]) * s)
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// audioActivateBF16 applies the Conformer activation as a byte-parity bf16 op, matching metal's
// gemma4AudioActivate (SiLU = Mul(x, Sigmoid(x)); ReLU = Maximum(x,0); GeLU = the tanh approx).
func audioActivateBF16(b []byte, act string) ([]byte, error) {
	switch act {
	case "relu":
		return reluBF16(b), nil
	case "gelu", "gelu_pytorch_tanh":
		return GeluBF16(b)
	default: // silu / swish / ""
		return SiLUBF16(b)
	}
}

// AudioFeedForward is the all-bf16 FeedForward — DEPRECATED / NOT byte-identical to the real
// Gemma4AudioFeedForward.Forward. The audio tower's GC clamp scalars are f32, so metal.Clip promotes
// the activation to fp32 and the whole FF runs in fp32 (audio_f32.go); this bf16 path only matches
// data-dependently (it diverges at some scales). The tower uses AudioFeedForwardF32. Retained only as
// a bf16 reference; do not use it where byte-identity matters.
func AudioFeedForward(x []byte, w *AudioFeedForwardWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if cfg.Hidden == 0 || cfg.FFInter == 0 {
		return nil, core.NewError("native.AudioFeedForward: cfg.Hidden and cfg.FFInter must be set")
	}
	L := len(x) / (cfg.Hidden * bf16Size)

	pre, err := RMSNormBF16(clampBF16(x, cfg.ClipMin, cfg.ClipMax), w.PreNorm, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	up, err := clippedMatRowsBF16(w.FFW1, pre, L, cfg.FFInter, cfg.Hidden, w.FFW1Clip)
	if err != nil {
		return nil, err
	}
	act, err := audioActivateBF16(up, cfg.Act)
	if err != nil {
		return nil, err
	}
	down, err := clippedMatRowsBF16(w.FFW2, act, L, cfg.Hidden, cfg.FFInter, w.FFW2Clip)
	if err != nil {
		return nil, err
	}
	post, err := RMSNormBF16(clampBF16(down, cfg.ClipMin, cfg.ClipMax), w.PostNorm, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	return AddBF16(mulScalarBF16(post, cfg.FFResidual), x) // residual on the original input
}

// reluBF16 is metal's ReLU (Maximum(x, 0)) as a byte-identical bf16 select: x≥0 keeps its bytes,
// x<0 becomes bf16 0. No arithmetic, so it equals metal byte-for-byte.
func reluBF16(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	for i := 0; i+1 < len(b); i += bf16Size {
		// bf16 sign bit is the top bit of the high byte; negative (and not -0) → 0.
		if b[i+1]&0x80 != 0 {
			out[i], out[i+1] = 0, 0
		}
	}
	return out
}

// AudioSubsampleWeights is the subsampler's bf16 views: two conv layers (weight [outC,3,3,inC] +
// scale-only LayerNorm weight/bias [outC]) and the input projection [hidden, F1·outC1].
type AudioSubsampleWeights struct {
	Conv0, Norm0W, Norm0B []byte
	Conv1, Norm1W, Norm1B []byte
	InputProj             []byte
	InputProjClip         ClipPair // optional per-linear activation clamps (zero value = none)
}

// AudioSubsampleConfig is the subsampler geometry (B=1): mel input dims + the two conv output channel
// counts + the encoder width.
type AudioSubsampleConfig struct {
	Frames, MelBins int
	OutC0, OutC1    int
	Hidden          int
	Eps             float32
}

// convOut returns the strided-conv output length for (in, kernel 3, stride 2, pad 1).
func convOut(in int) int { return (in+2-3)/2 + 1 }

// AudioSubsample is the all-bf16 subsampler — DEPRECATED / NOT byte-identical to the real
// Gemma4AudioSubSampleConvProjection.Forward. metal's ReLU is Maximum(x, FromValue(0)) with an f32
// zero, so it promotes the activation to fp32 at the first ReLU and the rest of the subsampler (and
// the whole tower) runs fp32; this bf16 path only matches data-dependently. The tower uses
// AudioSubsampleF32 (audio_f32.go). Retained only as a bf16 reference.
func AudioSubsample(features []byte, w *AudioSubsampleWeights, cfg AudioSubsampleConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(features) != cfg.Frames*cfg.MelBins*bf16Size {
		return nil, core.NewError("native.AudioSubsample: len(features) must equal Frames*MelBins*2 bytes")
	}
	t0, f0 := convOut(cfg.Frames), convOut(cfg.MelBins)
	h0, err := Conv2dBF16(features, w.Conv0, 1, cfg.Frames, cfg.MelBins, 1, cfg.OutC0, 3, 3, 2, 2, 1, 1)
	if err != nil {
		return nil, err
	}
	if h0, err = LayerNormBF16(h0, w.Norm0W, w.Norm0B, t0*f0, cfg.OutC0, cfg.Eps); err != nil {
		return nil, err
	}
	h0 = reluBF16(h0)

	t1, f1 := convOut(t0), convOut(f0)
	h1, err := Conv2dBF16(h0, w.Conv1, 1, t0, f0, cfg.OutC0, cfg.OutC1, 3, 3, 2, 2, 1, 1)
	if err != nil {
		return nil, err
	}
	if h1, err = LayerNormBF16(h1, w.Norm1W, w.Norm1B, t1*f1, cfg.OutC1, cfg.Eps); err != nil {
		return nil, err
	}
	h1 = reluBF16(h1)

	// flatten [t1, f1, outC1] → [t1, f1·outC1] is a contiguous reinterpret; InputProj maps to hidden.
	return clippedMatRowsBF16(w.InputProj, h1, t1, cfg.Hidden, f1*cfg.OutC1, w.InputProjClip)
}

// AudioLightConvWeights is one Conformer LightConv module's bf16 views: pre/conv RMSNorm, the GLU
// expand (LinearStart [2·channels, hidden]) and contract (LinearEnd [hidden, channels]) linears, and
// the depthwise conv1d weight [channels, kernel] (flattened from torch's [channels, kernel, 1]).
type AudioLightConvWeights struct {
	PreNorm, ConvNorm  []byte
	LinearStart        []byte
	LinearEnd          []byte
	DepthwiseWeight    []byte
	StartClip, EndClip ClipPair // optional per-linear activation clamps (zero value = none)
}

// sliceColsBF16 extracts columns [c0:c1) from each row of an [rows,cols] bf16 buffer — a byte-copy
// (byte-identical to metal.SliceAxis on the last axis).
func sliceColsBF16(b []byte, rows, cols, c0, c1 int) []byte {
	w := (c1 - c0) * bf16Size
	out := make([]byte, rows*w)
	for r := range rows {
		copy(out[r*w:r*w+w], b[(r*cols+c0)*bf16Size:(r*cols+c1)*bf16Size])
	}
	return out
}

// depthwiseConv1dBF16 is the causal depthwise conv1d over time, bf16: out[t,c] = Σ_k in[t-(K-1)+k,c]·
// dw[c,k] (left-pad K-1, in[<0]=0), fp32 accumulation rounded to bf16 — matching metal's
// PadAxis+Conv1d(groups=channels). in is [L,ch], dw is [ch,K], out is [L,ch].
func depthwiseConv1dBF16(in, dw []byte, L, ch, K int) []byte {
	inF, dwF := bf16ToF32Slice(in), bf16ToF32Slice(dw)
	out := make([]byte, L*ch*bf16Size)
	for t := range L {
		for c := range ch {
			var acc float32
			for k := range K {
				if src := t - (K - 1) + k; src >= 0 {
					acc += inF[src*ch+c] * dwF[c*K+k]
				}
			}
			h := f32ToBF16(acc)
			o := (t*ch + c) * bf16Size
			out[o], out[o+1] = byte(h), byte(h>>8)
		}
	}
	return out
}

// AudioLightConv is the all-bf16 LightConv — DEPRECATED / NOT byte-identical to the real
// Gemma4AudioLightConv.Forward. After the conv's f32 GC clamp the module runs in fp32 (audio_f32.go);
// this bf16 path only matches data-dependently. The tower uses AudioLightConvF32. Retained only as a
// bf16 reference; do not use it where byte-identity matters.
func AudioLightConv(x []byte, w *AudioLightConvWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	ch, K := cfg.Channels, cfg.KernelSize
	if cfg.Hidden == 0 || ch == 0 || K == 0 {
		return nil, core.NewError("native.AudioLightConv: cfg.Hidden, Channels, KernelSize must be set")
	}
	L := len(x) / (cfg.Hidden * bf16Size)

	pre, err := RMSNormBF16(x, w.PreNorm, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	start, err := clippedMatRowsBF16(w.LinearStart, pre, L, 2*ch, cfg.Hidden, w.StartClip) // [L, 2·ch]
	if err != nil {
		return nil, err
	}
	// GLU: gate · sigmoid(gateIn) — gate = cols [0:ch], gateIn = cols [ch:2ch].
	sig, err := SigmoidBF16(sliceColsBF16(start, L, 2*ch, ch, 2*ch))
	if err != nil {
		return nil, err
	}
	glu, err := MulBF16(sliceColsBF16(start, L, 2*ch, 0, ch), sig)
	if err != nil {
		return nil, err
	}

	conv := clampBF16(depthwiseConv1dBF16(glu, w.DepthwiseWeight, L, ch, K), cfg.ClipMin, cfg.ClipMax)
	normed, err := RMSNormBF16(conv, w.ConvNorm, L, ch, cfg.Eps)
	if err != nil {
		return nil, err
	}
	act, err := audioActivateBF16(normed, cfg.Act)
	if err != nil {
		return nil, err
	}
	end, err := clippedMatRowsBF16(w.LinearEnd, act, L, cfg.Hidden, ch, w.EndClip)
	if err != nil {
		return nil, err
	}
	return AddBF16(end, x)
}
