// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"math"
	"runtime"
	"sync"

	core "dappco.re/go"
)

// mathops.go holds the primitive row-wise operations every tower/decoder file in this package
// shares: linear projections, the two LayerNorm shapes (weight-only RMSNorm for the decoder,
// weight+bias LayerNorm for SAM/CLIP), softmax, and the three activations DeepSeek-OCR's three
// sub-networks each pick a different one of (SAM's MLPBlock: exact erf GELU; CLIP's
// NoTPFeedForward: quick/sigmoid GELU; the decoder's SiLU-gated MLP/MoE experts: SiLU) — mirrors
// whisper/attention.go's f64-accumulation house convention for precision.
//
// PARALLELISM: this is still host-f32 correctness-first (device/GPU fusion is NOT this lane, per
// the design) — but a real checkpoint's per-layer cost (12 decoder layers x up to 64 routed
// experts, 12+24 vision-tower blocks over up to 4096 tokens) is far too large for one CPU core to
// run in gate-practical time (a live-checkpoint run confirmed >10 minutes without finishing even
// the prompt prefill on a single core). parallelFor spreads each hot loop's ALREADY-INDEPENDENT
// iterations (one row/token per iteration, writing to a disjoint output slice, reading only
// shared read-only weights) across GOMAXPROCS goroutines — ordinary host-CPU concurrency, the same
// category of lever as SIMD or cache-blocking would be, not a device-fusion optimisation.

// parallelFor calls fn(i) for every i in [0,n), split across GOMAXPROCS goroutines when n is
// large enough to be worth the dispatch overhead (small n runs inline). The caller's fn MUST
// write only to a disjoint region per i (e.g. out[i*width:(i+1)*width]) and must not share any
// OTHER mutable scratch buffer across iterations — every call site in this package that used a
// single reused scratch slice across a loop being parallelised had that scratch moved inside the
// loop body first (a shared buffer written from multiple goroutines is a data race, not merely a
// slow path).
func parallelFor(n int, fn func(i int)) {
	if n <= 1 {
		for i := range n {
			fn(i)
		}
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := range n {
			fn(i)
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		end := min(start+chunk, n)
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				fn(i)
			}
		}(start, end)
	}
	wg.Wait()
}

// linear computes y[T,Out] = x[T,In]·Wᵥᵀ + b (row-major [Out,In] weight, the PyTorch/safetensors
// convention; b nil ⇒ no bias). Rows are independent (parallelFor — see the file doc comment).
func linear(x, w []float32, in, out int, b []float32) []float32 {
	t := len(x) / in
	y := make([]float32, t*out)
	parallelFor(t, func(i int) {
		xi := x[i*in : (i+1)*in]
		yi := y[i*out : (i+1)*out]
		for o := range out {
			var acc float64
			wr := w[o*in : (o+1)*in]
			for k := range in {
				acc += float64(xi[k]) * float64(wr[k])
			}
			if b != nil {
				acc += float64(b[o])
			}
			yi[o] = float32(acc)
		}
	})
	return y
}

// layerNormBias applies full LayerNorm (mean-centred, weight AND bias) over the last dimension D
// of x[T,D] — the SAM/CLIP tower shape (unlike the decoder's weight-only RMSNorm below).
func layerNormBias(x, w, b []float32, d int, eps float32) []float32 {
	t := len(x) / d
	out := make([]float32, t*d)
	for i := range t {
		row := x[i*d : (i+1)*d]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(d)
		var vsum float64
		for _, v := range row {
			delta := float64(v) - mean
			vsum += delta * delta
		}
		inv := 1.0 / math.Sqrt(vsum/float64(d)+float64(eps))
		orow := out[i*d : (i+1)*d]
		for j, v := range row {
			orow[j] = float32((float64(v)-mean)*inv*float64(w[j]) + float64(b[j]))
		}
	}
	return out
}

// rmsNorm applies weight-only RMSNorm over the last dimension D of x[T,D] — DeepseekV2RMSNorm's
// shape (the decoder's norm; no bias, no mean-centring — "equivalent to T5LayerNorm" per its own
// doc comment).
func rmsNorm(x, w []float32, d int, eps float32) []float32 {
	t := len(x) / d
	out := make([]float32, t*d)
	for i := range t {
		row := x[i*d : (i+1)*d]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+float64(eps))
		orow := out[i*d : (i+1)*d]
		for j, v := range row {
			orow[j] = float32(float64(v) * inv * float64(w[j]))
		}
	}
	return out
}

// addRows returns a+b elementwise (the residual add).
func addRows(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// geluExact is the exact erf-based GELU — SAM's MLPBlock activation (nn.GELU()'s default,
// approximate="none"): 0.5x(1+erf(x/√2)).
func geluExact(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Erf(xf/math.Sqrt2)))
}

// quickGELU is CLIP's NoTPFeedForward activation: x·sigmoid(1.702x) — deepencoder.py's
// quick_gelu, distinct from SAM's exact erf GELU above (a different sub-network, a different
// activation — never conflate the two).
func quickGELU(x float32) float32 {
	xf := float64(x)
	return float32(xf / (1 + math.Exp(-1.702*xf)))
}

// silu is the decoder MLP/MoE-expert activation (config.json's implied hidden_act default
// "silu" — see Config's doc comment): x·sigmoid(x), aka swish.
func silu(x float32) float32 {
	xf := float64(x)
	return float32(xf / (1 + math.Exp(-xf)))
}

func geluRow(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = geluExact(v)
	}
	return out
}

func quickGELURow(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = quickGELU(v)
	}
	return out
}

// softmaxInPlace normalises row (already the raw scores) to a probability distribution, f64
// accumulation, numerically stable (max-subtracted).
func softmaxInPlace(row []float64) {
	var maxV = math.Inf(-1)
	for _, v := range row {
		if v > maxV {
			maxV = v
		}
	}
	var sum float64
	for i, v := range row {
		e := math.Exp(v - maxV)
		row[i] = e
		sum += e
	}
	for i := range row {
		row[i] /= sum
	}
}

// tensorF32 widens a bf16/f16/f32 safetensors tensor to a flat f32 slice — DeepSeek-OCR ships
// BF16 (torch_dtype "bfloat16" in config.json), but this widens any of the three the same way
// whisper.tensorF32/mamba2.tensorF32 do, for the same reason (Hub conversions circulate).
func tensorF32(dtype string, data []byte) ([]float32, error) {
	switch dtype {
	case "BF16", "bfloat16":
		if len(data)%2 != 0 {
			return nil, core.NewError("deepseekvl2.tensorF32: bf16 byte length odd")
		}
		out := make([]float32, len(data)/2)
		for i := range out {
			b := uint16(data[2*i]) | uint16(data[2*i+1])<<8
			out[i] = math.Float32frombits(uint32(b) << 16)
		}
		return out, nil
	case "F16", "float16":
		if len(data)%2 != 0 {
			return nil, core.NewError("deepseekvl2.tensorF32: f16 byte length odd")
		}
		out := make([]float32, len(data)/2)
		for i := range out {
			b := uint16(data[2*i]) | uint16(data[2*i+1])<<8
			out[i] = float16ToFloat32(b)
		}
		return out, nil
	case "F32", "float32":
		if len(data)%4 != 0 {
			return nil, core.NewError("deepseekvl2.tensorF32: f32 byte length not /4")
		}
		out := make([]float32, len(data)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(data[4*i]) | uint32(data[4*i+1])<<8 | uint32(data[4*i+2])<<16 | uint32(data[4*i+3])<<24)
		}
		return out, nil
	}
	return nil, core.NewError("deepseekvl2.tensorF32: unsupported dtype " + dtype)
}

// float16ToFloat32 widens one IEEE-754 binary16 value (as its raw bits) to float32 — identical
// bit-manipulation to whisper.float16ToFloat32 (arch/openai/whisper/weights.go), duplicated
// rather than imported: two independent "own loader" arch packages, neither depends on the
// other (AX-8 — no arch-to-arch imports).
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			for frac&0x400 == 0 {
				frac <<= 1
				exp--
			}
			exp++
			frac &= 0x3ff
			bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
		}
	case 0x1f:
		bits = (sign << 31) | 0x7f800000 | (frac << 13)
	default:
		bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}
