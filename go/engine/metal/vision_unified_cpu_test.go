// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

// vision_unified_cpu_test.go gates the HOST-side (no-GPU) helpers of the
// encoder-free unified embedder: the RIFF/WAVE audio decoder (DecodeWAVMono16k
// + its Kaiser-sinc resampleTo16k/kaiserI0) and the PNG/JPEG-to-model-patch
// packer (UnifiedVisionImagePatches). All pure Go — no metallib, no engine —
// so they run on every build, unlike the runtime-gated projection parity in
// vision_unified_test.go.

// wavLE16/wavLE32 write little-endian integers into a byte slice, the RIFF
// container's native encoding.
func wavLE16(b []byte, v int) { b[0], b[1] = byte(v), byte(v>>8) }
func wavLE32(b []byte, v int) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

// wavChunk frames a RIFF sub-chunk: id + le32(size) + body, word-aligned with a
// trailing pad byte when the body length is odd (the alignment the decoder's
// `off = body + size + (size&1)` scan must honour).
func wavChunk(id string, body []byte) []byte {
	head := make([]byte, 8)
	copy(head, id)
	wavLE32(head[4:], len(body))
	out := append(head, body...)
	if len(body)&1 == 1 {
		out = append(out, 0)
	}
	return out
}

// wavRIFF wraps pre-framed chunks in the RIFF/WAVE container.
func wavRIFF(chunks ...[]byte) []byte {
	body := []byte("WAVE")
	for _, c := range chunks {
		body = append(body, c...)
	}
	head := make([]byte, 8)
	copy(head, "RIFF")
	wavLE32(head[4:], len(body))
	return append(head, body...)
}

// wavFmt builds a canonical 16-byte fmt chunk body for the given PCM geometry.
func wavFmt(format, channels, rate, bits int) []byte {
	b := make([]byte, 16)
	wavLE16(b[0:], format)
	wavLE16(b[2:], channels)
	wavLE32(b[4:], rate)
	wavLE32(b[8:], rate*channels*bits/8) // byte rate
	wavLE16(b[12:], channels*bits/8)     // block align
	wavLE16(b[14:], bits)
	return b
}

// wavPCM16 encodes interleaved int16 samples as little-endian bytes.
func wavPCM16(samples ...int16) []byte {
	b := make([]byte, len(samples)*2)
	for i, s := range samples {
		b[2*i], b[2*i+1] = byte(uint16(s)), byte(uint16(s)>>8)
	}
	return b
}

// wavFile composes a complete mono/stereo 16-bit PCM WAV from interleaved samples.
func wavFile(channels, rate int, samples ...int16) []byte {
	return wavRIFF(wavChunk("fmt ", wavFmt(1, channels, rate, 16)), wavChunk("data", wavPCM16(samples...)))
}

func TestDecodeWAVMono16kGood(t *testing.T) {
	// Mono 16 kHz decodes straight through (rate == 16000, no resample) with each
	// sample scaled by 1/32768.
	mono := wavFile(1, 16000, 100, -100, 32767, -32768)
	got, err := DecodeWAVMono16k(mono)
	if err != nil {
		t.Fatalf("mono decode: %v", err)
	}
	want := []float32{100.0 / 32768, -100.0 / 32768, 32767.0 / 32768, -32768.0 / 32768}
	if len(got) != len(want) {
		t.Fatalf("mono samples = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("mono[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Stereo 16 kHz downmixes by averaging the L/R frame.
	stereo := wavFile(2, 16000, 100, 200, -400, -800)
	got, err = DecodeWAVMono16k(stereo)
	if err != nil {
		t.Fatalf("stereo decode: %v", err)
	}
	wantStereo := []float32{(100.0/32768 + 200.0/32768) / 2, (-400.0/32768 - 800.0/32768) / 2}
	if len(got) != len(wantStereo) {
		t.Fatalf("stereo frames = %d, want %d", len(got), len(wantStereo))
	}
	for i := range wantStereo {
		if math.Abs(float64(got[i]-wantStereo[i])) > 1e-6 {
			t.Fatalf("stereo[%d] = %v, want %v", i, got[i], wantStereo[i])
		}
	}
}

func TestDecodeWAVMono16kResamples(t *testing.T) {
	// 8 kHz upsamples to ~2× length; 48 kHz downsamples to ~1/3. The resampler
	// output count is round(len/ratio); we assert the length contract and that
	// samples stay finite and bounded (the Kaiser window has unity passband gain).
	for _, tc := range []struct {
		name    string
		rate    int
		samples int
	}{
		{"upsample8k", 8000, 64},
		{"downsample48k", 48000, 96},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pcm := make([]int16, tc.samples)
			for i := range pcm {
				pcm[i] = int16(8000 * math.Sin(float64(i)*0.3))
			}
			got, err := DecodeWAVMono16k(wavFile(1, tc.rate, pcm...))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			wantN := int(float64(tc.samples)/(float64(tc.rate)/16000) + 0.5)
			if len(got) != wantN {
				t.Fatalf("resampled len = %d, want %d", len(got), wantN)
			}
			for i, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)) > 1.5 {
					t.Fatalf("resampled[%d] = %v out of range", i, v)
				}
			}
		})
	}
}

func TestDecodeWAVMono16kSkipsOddChunk(t *testing.T) {
	// An odd-length unknown chunk between fmt and data forces the word-alignment
	// branch (`off = body + size + (size&1)`); a clean decode proves the pad byte
	// is skipped and the data chunk is still found.
	wav := wavRIFF(
		wavChunk("fmt ", wavFmt(1, 1, 16000, 16)),
		wavChunk("LIST", []byte{1, 2, 3}), // odd body -> padded
		wavChunk("data", wavPCM16(1000, -1000)),
	)
	got, err := DecodeWAVMono16k(wav)
	if err != nil {
		t.Fatalf("decode with odd chunk: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("samples after odd-chunk skip = %d, want 2", len(got))
	}
}

func TestDecodeWAVMono16kBad(t *testing.T) {
	valid := wavFile(1, 16000, 1, 2)
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"tooShort", []byte("RIFFxxxxWAVE")},
		{"notRIFF", append([]byte("XXXXyyyyWAVE"), make([]byte, 40)...)},
		{"notWAVE", append([]byte("RIFFyyyyXXXX"), make([]byte, 40)...)},
		{"shortFmt", wavRIFF(wavChunk("fmt ", make([]byte, 12)), wavChunk("data", wavPCM16(1, 2)))},
		{"nonPCM", wavRIFF(wavChunk("fmt ", wavFmt(2, 1, 16000, 16)), wavChunk("data", wavPCM16(1, 2)))},
		{"not16bit", wavRIFF(wavChunk("fmt ", wavFmt(1, 1, 16000, 8)), wavChunk("data", wavPCM16(1, 2)))},
		{"zeroChannels", wavRIFF(wavChunk("fmt ", wavFmt(1, 0, 16000, 16)), wavChunk("data", wavPCM16(1, 2)))},
		{"badRate", wavRIFF(wavChunk("fmt ", wavFmt(1, 1, 0, 16)), wavChunk("data", wavPCM16(1, 2)))},
		{"noData", wavRIFF(wavChunk("fmt ", wavFmt(1, 1, 16000, 16)))},
		{"emptyFrames", wavRIFF(wavChunk("fmt ", wavFmt(1, 2, 16000, 16)), wavChunk("data", wavPCM16(1)))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := DecodeWAVMono16k(tc.data); err == nil {
				t.Fatalf("%s: expected error, got nil", tc.name)
			}
		})
	}
	// The valid fixture must actually decode, or the negatives prove nothing.
	if _, err := DecodeWAVMono16k(valid); err != nil {
		t.Fatalf("control fixture failed to decode: %v", err)
	}
}

func TestResampleTo16kLengthAndEdges(t *testing.T) {
	// Direct exercise of the unexported resampler: length contract in both
	// directions plus the empty-input clamp (outN floored to 1).
	up := resampleTo16k(make([]float32, 100), 8000)
	if len(up) != 200 {
		t.Fatalf("upsample len = %d, want 200", len(up))
	}
	down := resampleTo16k(make([]float32, 300), 48000)
	if len(down) != 100 {
		t.Fatalf("downsample len = %d, want 100", len(down))
	}
	empty := resampleTo16k(nil, 8000)
	if len(empty) != 1 {
		t.Fatalf("empty-input resample len = %d, want 1 (clamped)", len(empty))
	}
	// A DC (constant) signal must pass through the unity-gain window unchanged.
	dc := make([]float32, 64)
	for i := range dc {
		dc[i] = 0.5
	}
	for i, v := range resampleTo16k(dc, 8000) {
		if math.Abs(float64(v-0.5)) > 1e-3 {
			t.Fatalf("DC resample[%d] = %v, want ~0.5", i, v)
		}
	}
}

// unifiedTestPNG builds a w×h solid-colour PNG — the host-decodable input to
// UnifiedVisionImagePatches.
func unifiedTestPNG(t *testing.T, w, h int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestUnifiedVisionImagePatchesGood(t *testing.T) {
	// A 96×96 image is an exact 2×2 grid of 48px model patches (PatchSize 16 ×
	// PoolingKernel 3), so it needs no resize: expect 4 patches, matching soft
	// tokens, patchDim = 48²·3 bytes each, and (col,row) position indices.
	png := unifiedTestPNG(t, 96, 96, color.RGBA{R: 200, G: 100, B: 50, A: 255})
	patches, positions, n, err := UnifiedVisionImagePatches(png, &VisionImageFeatureConfig{})
	if err != nil {
		t.Fatalf("UnifiedVisionImagePatches: %v", err)
	}
	if n != 4 {
		t.Fatalf("model patches = %d, want 4", n)
	}
	const patchDim = 48 * 48 * 3
	if len(patches) != n*patchDim*bf16Size {
		t.Fatalf("patch bytes = %d, want %d", len(patches), n*patchDim*bf16Size)
	}
	if len(positions) != n*2 {
		t.Fatalf("positions len = %d, want %d", len(positions), n*2)
	}
	// Row-major grid, positions are (col, row).
	wantPos := []int32{0, 0, 1, 0, 0, 1, 1, 1}
	for i, want := range wantPos {
		if positions[i] != want {
			t.Fatalf("positions = %v, want %v", positions, wantPos)
		}
	}
	// A solid image has finite, in-range patch bytes (rescaled to [0,1]).
	for i := 0; i+1 < len(patches); i += bf16Size {
		v := bf16ToF32(patches[i], patches[i+1])
		if math.IsNaN(float64(v)) || v < 0 || v > 1.001 {
			t.Fatalf("patch value %v out of [0,1]", v)
		}
	}
}

func TestUnifiedVisionImagePatchesMaxTokEnv(t *testing.T) {
	// The #351 LTHN_VISION_MAXTOK instrument shrinks the soft-token budget: a
	// budget of 1 collapses the 96×96 image to a single 48px model patch. The
	// budget is only consulted through the aspect-preserving sizing path, so
	// DoResize forces that path even for a patch-aligned image.
	t.Setenv("LTHN_VISION_MAXTOK", "1")
	png := unifiedTestPNG(t, 96, 96, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	_, _, n, err := UnifiedVisionImagePatches(png, &VisionImageFeatureConfig{DoResize: true})
	if err != nil {
		t.Fatalf("UnifiedVisionImagePatches maxtok=1: %v", err)
	}
	if n != 1 {
		t.Fatalf("maxtok=1 produced %d patches, want 1", n)
	}
}

func TestUnifiedVisionImagePatchesBad(t *testing.T) {
	// Undecodable bytes surface the VisionImagePixels decode error unchanged.
	if _, _, _, err := UnifiedVisionImagePatches([]byte("not an image"), &VisionImageFeatureConfig{}); err == nil {
		t.Fatal("expected decode error on garbage bytes")
	}
}
