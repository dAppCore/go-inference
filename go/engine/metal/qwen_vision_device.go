// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/Qwen/qwen35"
)

// qwen_vision_device.go is the #59 follow-up (design doc §6): a DEVICE-RESIDENT bf16 mirror of the
// host f32 qwen35 vision tower, uploaded once at load and run entirely through the engine's existing
// bf16 kernels — the same kernels + upload-once idiom gemma4's device SigLIP tower (vision.go) uses,
// mirrored for qwen's LayerNorm-with-bias / fused-QKV / merger shape. Every weight downcasts f32→bf16
// (round-to-nearest-even, f32ToBF16) exactly once at load; the host f32 arrays are then freed
// (freeQwenVisionHostWeights), so the tower's steady-state RSS is the bf16 copy only, not both. The
// host tier (qwen_vision_encoder.go) stays byte-for-byte as the #59 v1 port left it — the correctness
// oracle a parity receipt runs against (qwen_vision_device_test.go) and the tower's only tier when the
// device mirror is disabled (LTHN_QWEN_VISION_DEVICE=0) or, in principle, unavailable.
//
// What moves device: patch embed, every block's Q/K/V/O projections, the O(L²) attention core
// (VisionSDPA — gemma's own bidirectional-attention kernel, reused as-is: qwen's tower is ALSO
// bidirectional/non-causal, and VisionSDPA's GQA ratio + scale are already parameters, not gemma
// constants), the GELU MLP's two projections + GeluBF16 activation, and the merger's LayerNorm +
// both linears + GELU. What stays host: the per-head QK-norm + 2-D rope (HeadDim-sized, reuses the
// EXISTING qwen35-specific scalar helpers below unchanged — gemma's own device tower keeps this same
// slice of work host-side for the identical reason: too small to be worth a command-buffer round
// trip), the merger's spatial-block gather (a pure byte reshuffle, no arithmetic, no device twin
// anywhere in this engine), and the GUESSED layout's SwiGLU gate — see qwenVisionSiluGateMulBF16.

// qwenVisionDeviceEnabled reports whether the device-resident bf16 tower engages — the kill-switch
// this engine's other wall-clock-adaptive levers use (CLAUDE.md "one lever per commit": compare
// LTHN_MTP_REENGAGE=0). A function, not a package var, so a test can toggle it per-run with
// t.Setenv without needing a fresh process. LTHN_QWEN_VISION_DEVICE=0 forces the v1 host-only path
// (unchanged f32 weights, never freed) — the safety valve if the device tier ever regresses.
func qwenVisionDeviceEnabled() bool { return os.Getenv("LTHN_QWEN_VISION_DEVICE") != "0" }

// qwenVisionDeviceLinear mirrors qwen35.VisionLinear as bf16 device bytes: W is row-major [Out,In],
// B is [Out] or nil when the source carried no bias — the SAME optionality qwen35.VisionLinear uses.
type qwenVisionDeviceLinear struct {
	W       []byte
	B       []byte
	Out, In int
}

// qwenVisionDeviceAttn mirrors qwen35.VisionAttnWeights. QNorm/KNorm stay float32 (HeadDim-sized,
// applied per-head on the host — see qwenVisionSplitHeadsRoped — so a bf16 round-trip would only cost
// precision for zero device benefit); they alias the tower's own norm slices directly, which is safe
// because freeQwenVisionHostWeights deliberately leaves these two fields alone (they are tiny — a
// few HeadDim vectors per block — unlike the GEMM weights the memory story is actually about).
type qwenVisionDeviceAttn struct {
	Q, K, V, O   qwenVisionDeviceLinear
	QNorm, KNorm []float32
}

// qwenVisionDeviceMLP mirrors qwen35.VisionMLPWeights: exactly one of (Gate,Up,Down) or (FC1,FC2) is
// populated, selected by GELU — copied verbatim from the source block, never both.
type qwenVisionDeviceMLP struct {
	Gate, Up, Down qwenVisionDeviceLinear
	FC1, FC2       qwenVisionDeviceLinear
	GELU           bool
}

// qwenVisionDeviceBlock mirrors qwen35.VisionBlock.
type qwenVisionDeviceBlock struct {
	Norm1W, Norm1B []byte
	Attn           qwenVisionDeviceAttn
	Norm2W, Norm2B []byte
	MLP            qwenVisionDeviceMLP
}

// qwenVisionDeviceMerger mirrors qwen35.VisionMerger.
type qwenVisionDeviceMerger struct {
	NormW, NormB []byte
	L1, L2       qwenVisionDeviceLinear
}

// qwenVisionDeviceTower is the whole bf16 device mirror of a qwen35.VisionTower — built once
// (buildQwenVisionDeviceTower), stored on the source tower's own DeviceSeam field (its lifetime is
// then exactly the tower's — no separate cache to evict on model unload), and consumed by
// QwenVisionTowerForwardDevice. Cfg is the SAME qwen35.VisionTowerConfig value (plain geometry, no
// device state, so it is shared rather than copied).
type qwenVisionDeviceTower struct {
	Patch    qwenVisionDeviceLinear
	PosEmbed []byte
	Blocks   []qwenVisionDeviceBlock
	Merger   qwenVisionDeviceMerger
	Cfg      qwen35.VisionTowerConfig
}

// qwenVisionDeviceLinearFrom downcasts one f32 VisionLinear to its bf16 device mirror.
func qwenVisionDeviceLinearFrom(l qwen35.VisionLinear) qwenVisionDeviceLinear {
	d := qwenVisionDeviceLinear{Out: l.Out, In: l.In}
	if len(l.W) > 0 {
		d.W = f32ToBf16Slice(l.W)
	}
	if len(l.B) > 0 {
		d.B = f32ToBf16Slice(l.B)
	}
	return d
}

// buildQwenVisionDeviceTower downcasts every weight in tower to its bf16 mirror — a pure host
// conversion (f32ToBf16Slice, round-to-nearest-even), no device call, so it always succeeds; the
// first actual kernel dispatch against the returned buffers is what pins them resident (residentBytes,
// the #60 owned-weight precedent — see the design doc's lifetime note for the process-scoped-cache
// boundary this shares with every other off-shard weight in this engine).
func buildQwenVisionDeviceTower(tower *qwen35.VisionTower) *qwenVisionDeviceTower {
	dt := &qwenVisionDeviceTower{
		Patch: qwenVisionDeviceLinearFrom(tower.Patch),
		Cfg:   tower.Cfg,
	}
	if len(tower.PosEmbed) > 0 {
		dt.PosEmbed = f32ToBf16Slice(tower.PosEmbed)
	}
	dt.Blocks = make([]qwenVisionDeviceBlock, len(tower.Blocks))
	for i := range tower.Blocks {
		b := &tower.Blocks[i]
		dt.Blocks[i] = qwenVisionDeviceBlock{
			Norm1W: f32ToBf16Slice(b.Norm1W), Norm1B: f32ToBf16Slice(b.Norm1B),
			Norm2W: f32ToBf16Slice(b.Norm2W), Norm2B: f32ToBf16Slice(b.Norm2B),
			Attn: qwenVisionDeviceAttn{
				Q: qwenVisionDeviceLinearFrom(b.Attn.Q), K: qwenVisionDeviceLinearFrom(b.Attn.K),
				V: qwenVisionDeviceLinearFrom(b.Attn.V), O: qwenVisionDeviceLinearFrom(b.Attn.O),
				QNorm: b.Attn.QNorm, KNorm: b.Attn.KNorm,
			},
			MLP: qwenVisionDeviceMLP{
				Gate: qwenVisionDeviceLinearFrom(b.MLP.Gate), Up: qwenVisionDeviceLinearFrom(b.MLP.Up), Down: qwenVisionDeviceLinearFrom(b.MLP.Down),
				FC1: qwenVisionDeviceLinearFrom(b.MLP.FC1), FC2: qwenVisionDeviceLinearFrom(b.MLP.FC2),
				GELU: b.MLP.GELU,
			},
		}
	}
	dt.Merger = qwenVisionDeviceMerger{
		NormW: f32ToBf16Slice(tower.Merger.NormW), NormB: f32ToBf16Slice(tower.Merger.NormB),
		L1: qwenVisionDeviceLinearFrom(tower.Merger.L1), L2: qwenVisionDeviceLinearFrom(tower.Merger.L2),
	}
	return dt
}

// qwenVisionDeviceTowerBytes sums the bf16 mirror's total weight-byte footprint — the measured half
// of the memory story (design doc §6): the f32 host copy freeQwenVisionHostWeights releases is
// approximately double this (the same element counts, 4 bytes/element instead of 2).
func qwenVisionDeviceTowerBytes(dt *qwenVisionDeviceTower) int {
	if dt == nil {
		return 0
	}
	linearBytes := func(l qwenVisionDeviceLinear) int { return len(l.W) + len(l.B) }
	n := linearBytes(dt.Patch) + len(dt.PosEmbed)
	for i := range dt.Blocks {
		b := &dt.Blocks[i]
		n += len(b.Norm1W) + len(b.Norm1B) + len(b.Norm2W) + len(b.Norm2B)
		n += linearBytes(b.Attn.Q) + linearBytes(b.Attn.K) + linearBytes(b.Attn.V) + linearBytes(b.Attn.O)
		n += linearBytes(b.MLP.Gate) + linearBytes(b.MLP.Up) + linearBytes(b.MLP.Down)
		n += linearBytes(b.MLP.FC1) + linearBytes(b.MLP.FC2)
	}
	n += len(dt.Merger.NormW) + len(dt.Merger.NormB) + linearBytes(dt.Merger.L1) + linearBytes(dt.Merger.L2)
	return n
}

// qwenVisionDeviceWarm eagerly resident-binds every bf16 weight — "upload at load" rather than
// deferring the first Metal buffer creation to the first image turn, so a live wall-time receipt
// measures encode time only, not first-use upload latency. Best-effort: a bind failure here (device
// unavailable) is silently skipped, and the SAME call inside the real kernel dispatch will then
// surface the error loudly on the first actual encode instead.
func qwenVisionDeviceWarm(dt *qwenVisionDeviceTower) {
	if dt == nil || ensureInit() != nil {
		return
	}
	warmLinear := func(l qwenVisionDeviceLinear) {
		if len(l.W) > 0 {
			residentBytes(l.W)
		}
		if len(l.B) > 0 {
			residentBytes(l.B)
		}
	}
	warmLinear(dt.Patch)
	if len(dt.PosEmbed) > 0 {
		residentBytes(dt.PosEmbed)
	}
	for i := range dt.Blocks {
		b := &dt.Blocks[i]
		for _, w := range [][]byte{b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B} {
			if len(w) > 0 {
				residentBytes(w)
			}
		}
		warmLinear(b.Attn.Q)
		warmLinear(b.Attn.K)
		warmLinear(b.Attn.V)
		warmLinear(b.Attn.O)
		warmLinear(b.MLP.Gate)
		warmLinear(b.MLP.Up)
		warmLinear(b.MLP.Down)
		warmLinear(b.MLP.FC1)
		warmLinear(b.MLP.FC2)
	}
	for _, w := range [][]byte{dt.Merger.NormW, dt.Merger.NormB} {
		if len(w) > 0 {
			residentBytes(w)
		}
	}
	warmLinear(dt.Merger.L1)
	warmLinear(dt.Merger.L2)
}

// freeQwenVisionHostWeights drops every large f32 weight slice on tower IN PLACE, once its bf16
// device mirror is built and cached on tower.DeviceSeam — the memory half of the #59 device-tower
// follow-up: the tower's host RSS after this call is the small metadata (Cfg, Preprocess, the
// tiny QNorm/KNorm vectors the device mirror still shares) plus the bf16 mirror, not a second full
// f32 copy sitting alongside it. QwenVisionTowerForward on a freed tower returns a clean, named error
// (the guard at its top) instead of computing over emptied slices — this tower instance is
// device-only from here on, so a host-vs-device parity check always loads an INDEPENDENT, un-freed
// tower for its host side (qwen_vision_device_test.go never calls this on a tower it still wants to
// run QwenVisionTowerForward against).
func freeQwenVisionHostWeights(tower *qwen35.VisionTower) {
	free := func(l *qwen35.VisionLinear) { l.W, l.B = nil, nil }
	free(&tower.Patch)
	tower.PosEmbed = nil
	for i := range tower.Blocks {
		b := &tower.Blocks[i]
		b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B = nil, nil, nil, nil
		free(&b.Attn.Q)
		free(&b.Attn.K)
		free(&b.Attn.V)
		free(&b.Attn.O)
		// Attn.QNorm/KNorm deliberately NOT freed — the device mirror aliases them directly (see
		// qwenVisionDeviceAttn's doc comment); they are HeadDim-sized, negligible against the GEMM
		// weights above.
		free(&b.MLP.Gate)
		free(&b.MLP.Up)
		free(&b.MLP.Down)
		free(&b.MLP.FC1)
		free(&b.MLP.FC2)
	}
	tower.Merger.NormW, tower.Merger.NormB = nil, nil
	free(&tower.Merger.L1)
	free(&tower.Merger.L2)
}

// qwenVisionLayerNormBF16 is LayerNormBF16 with qwen35's OWN "nil/empty bias means no bias" LayerNorm
// convention restored: LayerNormBF16 itself requires a real length-axisSize bias buffer (its "no bias"
// idiom elsewhere in this engine is an explicit ZERO vector — see its doc comment re: gemma4's audio
// subsampler), whereas qwen35.VisionBlock/VisionMerger norms may carry a nil bias (the REAL layout
// always populates one; the GUESSED layout's loader does not always). Zero bf16 is the all-zero byte
// pattern (same as zero f32), so substituting a fresh zero buffer is exact, not an approximation.
func qwenVisionLayerNormBF16(x, weight, bias []byte, rows, axisSize int, eps float32) ([]byte, error) {
	if len(bias) == 0 {
		bias = make([]byte, axisSize*bf16Size)
	}
	return LayerNormBF16(x, weight, bias, rows, axisSize, eps)
}

// qwenVisionAddBiasBF16 adds bias (length dim, bf16) to every one of rows rows in x ([rows,dim] bf16)
// IN PLACE — a no-op when bias is empty (an absent optional bias, matching qwen35.VisionLinear.B's own
// nil-is-absent convention). Host-side: gemma's own device tower computes its linear biases the same
// way (vision.go's addVisionLinearBiasRows) — a per-row broadcast add has no existing device kernel in
// this engine (AddBF16 is equal-shape only) and is small (O(rows·dim), the same tier as a norm).
func qwenVisionAddBiasBF16(x, bias []byte, rows, dim int) {
	if len(bias) == 0 {
		return
	}
	for r := range rows {
		row := x[r*dim*bf16Size : (r+1)*dim*bf16Size]
		for c := range dim {
			v := bf16ToF32(row[2*c], row[2*c+1]) + bf16ToF32(bias[2*c], bias[2*c+1])
			h := f32ToBF16(v)
			row[2*c], row[2*c+1] = byte(h), byte(h>>8)
		}
	}
}

// qwenVisionSplitHeadsRoped reshapes a token-major bf16 buffer [L, heads·headDim] into the head-major
// [heads, L, headDim] layout VisionSDPA consumes, applying the optional per-head RMS norm (norm — nil
// for the REAL layout, populated for the GUESSED layout, exactly qwenVisionRMSNormHead's existing
// no-op-when-empty contract) and the 2-D rotary embedding in the SAME per-head pass — both host f64,
// reusing qwen_vision_encoder.go's EXISTING scalar helpers unchanged (no new host maths, only new
// bf16↔f32 data-movement glue around them, the same edge gemma's own qkNormRoPEHeadMajor sits at).
func qwenVisionSplitHeadsRoped(tokenMajor []byte, norm []float32, heads, L, headDim, gridW int, invFreq []float64, eps float32) []byte {
	out := make([]byte, len(tokenMajor))
	head := make([]float32, headDim)
	rowBytes := headDim * bf16Size
	for t := range L {
		row, col := t/gridW, t%gridW
		for h := range heads {
			src := (t*heads + h) * rowBytes
			for d := range headDim {
				head[d] = bf16ToF32(tokenMajor[src+2*d], tokenMajor[src+2*d+1])
			}
			qwenVisionRMSNormHead(head, norm, eps)
			qwenVisionRope2D(head, row, col, headDim, invFreq)
			dst := (h*L + t) * rowBytes
			for d := range headDim {
				b := f32ToBF16(head[d])
				out[dst+2*d], out[dst+2*d+1] = byte(b), byte(b>>8)
			}
		}
	}
	return out
}

// qwenVisionSplitHeadsBF16 reshapes a token-major bf16 buffer [L, heads·headDim] into head-major
// [heads, L, headDim] — the value path's plain transpose (no norm, no rope, unlike Q/K).
func qwenVisionSplitHeadsBF16(tokenMajor []byte, heads, L, headDim int) []byte {
	out := make([]byte, len(tokenMajor))
	rowBytes := headDim * bf16Size
	for t := range L {
		for h := range heads {
			src := (t*heads + h) * rowBytes
			dst := (h*L + t) * rowBytes
			copy(out[dst:dst+rowBytes], tokenMajor[src:src+rowBytes])
		}
	}
	return out
}

// qwenVisionMergeHeadsBF16 is the inverse reshape: head-major [heads, L, headDim] bf16 (VisionSDPA's
// output layout) back to token-major [L, heads·headDim] bf16 for the O projection.
func qwenVisionMergeHeadsBF16(headMajor []byte, heads, L, headDim int) []byte {
	out := make([]byte, len(headMajor))
	rowBytes := headDim * bf16Size
	for h := range heads {
		for t := range L {
			src := (h*L + t) * rowBytes
			dst := (t*heads + h) * rowBytes
			copy(out[dst:dst+rowBytes], headMajor[src:src+rowBytes])
		}
	}
	return out
}

// qwenVisionMergeSpatialBF16 is qwenVisionMergeSpatial's bf16 byte-wide twin: gathers x
// [gridH*gridW,hidden] (row-major raster order) into [(gridH/M)*(gridW/M), hidden*M*M], each output
// row concatenating one M×M spatial block of input rows. A pure memory reshuffle — no arithmetic, so
// it stays host regardless of dtype; there is no device gather primitive of this shape in this engine
// (and gemma4's tower has no merger to have set a precedent for one).
func qwenVisionMergeSpatialBF16(x []byte, gridH, gridW, hidden, m int) []byte {
	rowBytes := hidden * bf16Size
	outRows, outCols := (gridH/m)*(gridW/m), hidden*m*m
	out := make([]byte, outRows*outCols*bf16Size)
	idx := 0
	for by := 0; by < gridH; by += m {
		for bx := 0; bx < gridW; bx += m {
			dst := out[idx*outCols*bf16Size : (idx+1)*outCols*bf16Size]
			col := 0
			for dy := range m {
				for dx := range m {
					src := x[((by+dy)*gridW+(bx+dx))*rowBytes : ((by+dy)*gridW+(bx+dx))*rowBytes+rowBytes]
					copy(dst[col:col+rowBytes], src)
					col += rowBytes
				}
			}
			idx++
		}
	}
	return out
}

// qwenVisionSiluGateMulBF16 computes g ← silu(g)·u in place (bf16↔f64↔bf16, mirroring
// qwenVisionSilu's existing f64 formula exactly) — the GUESSED layout's SwiGLU gate. No existing
// device kernel computes this activation: GeluGateMul/GeluGateMulBF16 are GELU-gated (gemma's MLP
// nonlinearity, wrong shape for SwiGLU), and this engine's other SiLU kernel (MoEExpertsQuantSiLU) is
// a decode-path MoE primitive with no vision-tower-shaped sibling. Writing a genuinely new metallib
// kernel for it is not justified here: it is O(L·FF) elementwise (the same tier as the bias-add/gather
// glue already host-side above, not a compute-bound GEMM), and the GUESSED layout is a compatibility
// lane no live receipt exercises — the real 27B checkpoint ships the REAL layout's plain GELU MLP
// (qwenVisionDeviceMLPForward's GELU branch), which is fully device end-to-end.
func qwenVisionSiluGateMulBF16(g, u []byte) {
	for i := 0; i+1 < len(g); i += bf16Size {
		gv := float64(bf16ToF32(g[i], g[i+1]))
		uv := bf16ToF32(u[i], u[i+1])
		r := f32ToBF16(float32(qwenVisionSilu(gv) * float64(uv)))
		g[i], g[i+1] = byte(r), byte(r>>8)
	}
}

// qwenVisionDeviceAttentionForward is qwenVisionAttentionForward's device mirror: Q/K/V projections
// (device GEMM, MatRowsBF16) → per-head norm+rope (host, qwenVisionSplitHeadsRoped) → bidirectional
// attention (device, VisionSDPA — gemma's own kernel, reused unmodified: its GQA ratio and scale are
// call parameters, and qwen's tower is bidirectional/non-causal exactly like gemma's SigLIP) → O
// projection (device GEMM).
func qwenVisionDeviceAttentionForward(x []byte, w *qwenVisionDeviceAttn, L, gridW int, cfg qwen35.VisionTowerConfig) ([]byte, error) {
	H, KVH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	if H <= 0 || KVH <= 0 || HD <= 0 || H%KVH != 0 || gridW <= 0 {
		return nil, core.NewError("native.qwenVisionDeviceAttentionForward: bad attention/grid geometry")
	}
	q, err := MatRowsBF16(w.Q.W, x, L, w.Q.Out, w.Q.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceAttentionForward", "q projection", err)
	}
	qwenVisionAddBiasBF16(q, w.Q.B, L, w.Q.Out)
	k, err := MatRowsBF16(w.K.W, x, L, w.K.Out, w.K.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceAttentionForward", "k projection", err)
	}
	qwenVisionAddBiasBF16(k, w.K.B, L, w.K.Out)
	v, err := MatRowsBF16(w.V.W, x, L, w.V.Out, w.V.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceAttentionForward", "v projection", err)
	}
	qwenVisionAddBiasBF16(v, w.V.B, L, w.V.Out)

	theta := float64(cfg.RopeTheta)
	if theta == 0 {
		theta = 10000
	}
	invFreq := qwenVisionRotaryTable(HD, theta)
	qHM := qwenVisionSplitHeadsRoped(q, w.QNorm, H, L, HD, gridW, invFreq, cfg.Eps)
	kHM := qwenVisionSplitHeadsRoped(k, w.KNorm, KVH, L, HD, gridW, invFreq, cfg.Eps)
	vHM := qwenVisionSplitHeadsBF16(v, KVH, L, HD)

	scale := float32(1.0 / math.Sqrt(float64(HD)))
	attnHM, err := VisionSDPA(qHM, kHM, vHM, L, H, KVH, HD, scale)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceAttentionForward", "sdpa", err)
	}
	attnTM := qwenVisionMergeHeadsBF16(attnHM, H, L, HD)
	out, err := MatRowsBF16(w.O.W, attnTM, L, w.O.Out, w.O.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceAttentionForward", "o projection", err)
	}
	qwenVisionAddBiasBF16(out, w.O.B, L, w.O.Out)
	return out, nil
}

// qwenVisionDeviceMLPForward is qwenVisionMLPForward's device mirror. GELU branch (the REAL layout,
// what the live 27B checkpoint ships): FC1 (device GEMM) → GeluBF16 (device, tanh-approximation —
// byte-parity-intended with qwenVisionMLPForward's geluTanhScalar, per qwen_vision_encoder.go's own
// header note) → FC2 (device GEMM). SwiGLU branch (the GUESSED compatibility layout): Gate/Up (device
// GEMM) → qwenVisionSiluGateMulBF16 (host, see its doc comment) → Down (device GEMM).
func qwenVisionDeviceMLPForward(x []byte, w *qwenVisionDeviceMLP, L int) ([]byte, error) {
	if w.GELU {
		h, err := MatRowsBF16(w.FC1.W, x, L, w.FC1.Out, w.FC1.In)
		if err != nil {
			return nil, core.E("native.qwenVisionDeviceMLPForward", "fc1", err)
		}
		qwenVisionAddBiasBF16(h, w.FC1.B, L, w.FC1.Out)
		if h, err = GeluBF16(h); err != nil {
			return nil, core.E("native.qwenVisionDeviceMLPForward", "gelu", err)
		}
		out, err := MatRowsBF16(w.FC2.W, h, L, w.FC2.Out, w.FC2.In)
		if err != nil {
			return nil, core.E("native.qwenVisionDeviceMLPForward", "fc2", err)
		}
		qwenVisionAddBiasBF16(out, w.FC2.B, L, w.FC2.Out)
		return out, nil
	}
	g, err := MatRowsBF16(w.Gate.W, x, L, w.Gate.Out, w.Gate.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceMLPForward", "gate", err)
	}
	qwenVisionAddBiasBF16(g, w.Gate.B, L, w.Gate.Out)
	u, err := MatRowsBF16(w.Up.W, x, L, w.Up.Out, w.Up.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceMLPForward", "up", err)
	}
	qwenVisionAddBiasBF16(u, w.Up.B, L, w.Up.Out)
	qwenVisionSiluGateMulBF16(g, u)
	out, err := MatRowsBF16(w.Down.W, g, L, w.Down.Out, w.Down.In)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceMLPForward", "down", err)
	}
	qwenVisionAddBiasBF16(out, w.Down.B, L, w.Down.Out)
	return out, nil
}

// qwenVisionDeviceBlockForward is qwenVisionBlockForward's device mirror: LayerNormBF16 (device,
// weight+bias — the same kernel gemma's audio subsampler already proves byte-parity for) → attention
// → AddBF16 residual (device) → LayerNormBF16 → MLP → AddBF16 residual.
func qwenVisionDeviceBlockForward(b *qwenVisionDeviceBlock, x []byte, L, gridW int, cfg qwen35.VisionTowerConfig) ([]byte, error) {
	normed, err := qwenVisionLayerNormBF16(x, b.Norm1W, b.Norm1B, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceBlockForward", "norm1", err)
	}
	attnOut, err := qwenVisionDeviceAttentionForward(normed, &b.Attn, L, gridW, cfg)
	if err != nil {
		return nil, err
	}
	h, err := AddBF16(x, attnOut)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceBlockForward", "attn residual", err)
	}
	normed2, err := qwenVisionLayerNormBF16(h, b.Norm2W, b.Norm2B, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceBlockForward", "norm2", err)
	}
	mlpOut, err := qwenVisionDeviceMLPForward(normed2, &b.MLP, L)
	if err != nil {
		return nil, err
	}
	out, err := AddBF16(h, mlpOut)
	if err != nil {
		return nil, core.E("native.qwenVisionDeviceBlockForward", "mlp residual", err)
	}
	return out, nil
}

// qwenVisionDeviceMergerForward is qwenVisionMergerForward's device mirror: LayerNormBF16 (device) →
// the spatial merge gather (host, qwenVisionMergeSpatialBF16 — see its doc comment) → Linear1 (device)
// → GeluBF16 (device) → Linear2 (device) into text-hidden width.
func qwenVisionDeviceMergerForward(m *qwenVisionDeviceMerger, x []byte, gridH, gridW int, cfg qwen35.VisionTowerConfig) (features []byte, softTokens int, err error) {
	M := cfg.MergeSize
	if M <= 0 {
		M = 1
	}
	if gridH%M != 0 || gridW%M != 0 {
		return nil, 0, core.NewError(core.Sprintf("native.qwenVisionDeviceMergerForward: grid %dx%d not divisible by merge size %d", gridH, gridW, M))
	}
	normed, err := qwenVisionLayerNormBF16(x, m.NormW, m.NormB, gridH*gridW, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, 0, core.E("native.qwenVisionDeviceMergerForward", "norm", err)
	}
	merged := qwenVisionMergeSpatialBF16(normed, gridH, gridW, cfg.Hidden, M)
	outRows := (gridH / M) * (gridW / M)
	h1, err := MatRowsBF16(m.L1.W, merged, outRows, m.L1.Out, m.L1.In)
	if err != nil {
		return nil, 0, core.E("native.qwenVisionDeviceMergerForward", "linear1", err)
	}
	qwenVisionAddBiasBF16(h1, m.L1.B, outRows, m.L1.Out)
	if h1, err = GeluBF16(h1); err != nil {
		return nil, 0, core.E("native.qwenVisionDeviceMergerForward", "gelu", err)
	}
	features, err = MatRowsBF16(m.L2.W, h1, outRows, m.L2.Out, m.L2.In)
	if err != nil {
		return nil, 0, core.E("native.qwenVisionDeviceMergerForward", "linear2", err)
	}
	qwenVisionAddBiasBF16(features, m.L2.B, outRows, m.L2.Out)
	return features, outRows, nil
}

// QwenVisionTowerForwardDevice is QwenVisionTowerForward's device-resident mirror: the SAME stage
// order (patch embed → optional learned position add → N bidirectional blocks → merger) computed
// through bf16 device kernels throughout, returning bf16 feature BYTES directly (ready for
// TokenEmbeddingsWithFeatures — projectQwenImage skips the f32 round-trip QwenVisionTowerForward's
// callers pay).
func QwenVisionTowerForwardDevice(patches []float32, gridH, gridW int, dt *qwenVisionDeviceTower) (features []byte, softTokens int, err error) {
	if dt == nil {
		return nil, 0, core.NewError("native.QwenVisionTowerForwardDevice: nil device tower")
	}
	cfg := dt.Cfg
	L := gridH * gridW
	if L <= 0 {
		return nil, 0, core.NewError("native.QwenVisionTowerForwardDevice: empty patch grid")
	}
	if len(patches) != L*cfg.PatchDim {
		return nil, 0, core.NewError(core.Sprintf("native.QwenVisionTowerForwardDevice: patch buffer len %d != L·PatchDim %d", len(patches), L*cfg.PatchDim))
	}
	h, err := MatRowsBF16(dt.Patch.W, f32ToBf16Slice(patches), L, cfg.Hidden, cfg.PatchDim)
	if err != nil {
		return nil, 0, core.E("native.QwenVisionTowerForwardDevice", "patch embed", err)
	}
	qwenVisionAddBiasBF16(h, dt.Patch.B, L, cfg.Hidden)
	if len(dt.PosEmbed) > 0 {
		pos := dt.PosEmbed
		if len(pos) != L*cfg.Hidden*bf16Size {
			interp, perr := qwen35.InterpolatePosEmbed(bf16ToF32Slice(dt.PosEmbed), cfg.Hidden, gridH, gridW)
			if perr != nil {
				return nil, 0, core.E("native.QwenVisionTowerForwardDevice", "pos_embed interpolation", perr)
			}
			pos = f32ToBf16Slice(interp)
		}
		if h, err = AddBF16(h, pos); err != nil {
			return nil, 0, core.E("native.QwenVisionTowerForwardDevice", "pos_embed add", err)
		}
	}
	for i := range dt.Blocks {
		if h, err = qwenVisionDeviceBlockForward(&dt.Blocks[i], h, L, gridW, cfg); err != nil {
			return nil, 0, core.E("native.QwenVisionTowerForwardDevice", core.Sprintf("block %d", i), err)
		}
	}
	return qwenVisionDeviceMergerForward(&dt.Merger, h, gridH, gridW, cfg)
}
