// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"testing"
)

// requantRelL2 dequantises a 4-bit affine tensor from its packed codes + per-group
// scale/bias, re-quantises each group to nbits (fresh per-group min/scale), and returns
// the rel-L2 error of that re-quant plus the element count. Layout-free: group(i)=i/gs
// holds because groupSize divides every tensor's column count, so the flat index maps to
// the right group regardless of the [rows×cols] shape.
func requantRelL2(packed, scales, biases []byte, gs, bits, nbits int) (float64, int) {
	if gs <= 0 || bits != 4 || len(packed) == 0 {
		return 0, 0
	}
	nElem := len(packed) * 8 / bits // 4-bit: 2 codes/byte
	levels := float64(int64(1)<<uint(nbits) - 1)
	var errSq, refSq float64
	nGroups := (nElem + gs - 1) / gs
	vals := make([]float64, 0, gs)
	for g := 0; g < nGroups; g++ {
		if (g*2 + 1) >= len(scales) {
			break
		}
		sc := float64(bf16ToF32(scales[g*2], scales[g*2+1]))
		bi := float64(bf16ToF32(biases[g*2], biases[g*2+1]))
		vals = vals[:0]
		lo, hi := math.Inf(1), math.Inf(-1)
		for j := 0; j < gs; j++ {
			i := g*gs + j
			if i >= nElem {
				break
			}
			b := packed[i/2]
			var code byte
			if i%2 == 0 {
				code = b & 0xF
			} else {
				code = b >> 4
			}
			v := float64(code)*sc + bi
			vals = append(vals, v)
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		nscale := (hi - lo) / levels
		for _, v := range vals {
			vq := v
			if nscale > 0 {
				vq = lo + math.Round((v-lo)/nscale)*nscale
			}
			d := v - vq
			errSq += d * d
			refSq += v * v
		}
	}
	if refSq == 0 {
		return 0, nElem
	}
	return math.Sqrt(errSq / refSq), nElem
}

// TestWeightQuantSensitivityRealE2B is the #367 opportunity landscape: per weight-tensor
// KIND, the rel-L2 error of re-quantising the real e2b-4bit weights to 3 and 2 bits, beside
// each kind's share of the 4-bit weight bytes. Kinds with LOW 3-bit error tolerate 3-bit —
// adaptive candidates. It says WHERE the sub-4bit headroom is (and the byte win) before the
// definitive decode argmax-flip test. Static-metric caveat (the FFN taught it over-estimates):
// this ranks, the flip test decides.
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestWeightQuantSensitivityRealE2B -v ./engine/metal/
func TestWeightQuantSensitivityRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit weight-quant sensitivity landscape (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}

	kinds := []struct {
		name string
		get  func(QuantizedLayerWeights) QuantWeight
	}{
		{"Q", func(l QuantizedLayerWeights) QuantWeight { return l.Q }},
		{"K", func(l QuantizedLayerWeights) QuantWeight { return l.K }},
		{"V", func(l QuantizedLayerWeights) QuantWeight { return l.V }},
		{"O", func(l QuantizedLayerWeights) QuantWeight { return l.O }},
		{"Gate", func(l QuantizedLayerWeights) QuantWeight { return l.Gate }},
		{"Up", func(l QuantizedLayerWeights) QuantWeight { return l.Up }},
		{"Down", func(l QuantizedLayerWeights) QuantWeight { return l.Down }},
	}
	const mb = 1.0 / (1024 * 1024)
	t.Logf("=== #367 weight-quant sensitivity — real e2b-4bit (%d layers) ===", len(qm.Layers))
	var totBytes, tol3Bytes float64
	for _, k := range kinds {
		var e3, e2 float64
		var bytes, n int
		for l := range qm.Layers {
			qw := k.get(qm.Layers[l])
			if len(qw.Packed) == 0 {
				continue
			}
			r3, _ := requantRelL2(qw.Packed, qw.Scales, qw.Biases, qw.GroupSize, qw.Bits, 3)
			r2, _ := requantRelL2(qw.Packed, qw.Scales, qw.Biases, qw.GroupSize, qw.Bits, 2)
			e3 += r3
			e2 += r2
			bytes += len(qw.Packed)
			n++
		}
		if n == 0 {
			continue
		}
		totBytes += float64(bytes)
		tol := ""
		if e3/float64(n) < 0.05 {
			tol = "  <- tolerates 3-bit"
			tol3Bytes += float64(bytes)
		}
		t.Logf("  %-5s 3-bit rel-L2 %.4f · 2-bit %.4f · %2d layers · %6.1f MB (4bit)%s",
			k.name, e3/float64(n), e2/float64(n), n, float64(bytes)*mb, tol)
	}
	t.Logf("  quantised-weight bytes: %.0f MB @4bit; tensors tolerating 3-bit (<5%% rel-L2) hold %.0f MB -> save ~%.0f MB (25%%)",
		totBytes*mb, tol3Bytes*mb, tol3Bytes*mb*0.25)
	t.Logf("  => low 3-bit-error kinds are the adaptive-quant candidates; the decode argmax-flip test decides the real assignment")
}
