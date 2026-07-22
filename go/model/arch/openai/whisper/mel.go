// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"

	core "dappco.re/go"
)

// mel.go is Whisper's OWN log-mel front end — NOT a reuse of engine/metal's gemma4-derived
// audio.FeatureExtractor (model/gemma4/audio/features.go). The two are genuinely different DSP
// pipelines, confirmed against transformers' feature_extraction_whisper.py source before a line of this
// file was written:
//
//   - mel scale + normalisation: Whisper's filterbank is Slaney-scale + Slaney-normalised
//     (mel_filter_bank(..., norm="slaney", mel_scale="slaney")); gemma4's is HTK triangular, unnormalised.
//   - framing: Whisper centres each STFT frame via reflect-padding (torch.stft's center=True default,
//     pad n_fft//2 on both sides); gemma4 semicausally prepends frame/2 zeros ONCE at the start.
//   - power vs magnitude: Whisper squares the spectrum (power=2.0); gemma4 takes bare magnitude.
//   - compression: Whisper is log10 with a global dynamic-range clamp (max-8dB) then a fixed (x+4)/4
//     affine normalise; gemma4 is natural log with a small additive floor (+ optional per-bin z-score).
//
// Given that, this package does NOT reimplement the Slaney filterbank formula either: every published
// Whisper checkpoint ships the exact filter matrix it was computed with in preprocessor_config.json's
// "mel_filters" field ([NumMelBins][1+NFFT/2], row-major) — read verbatim (the "never guessed" rule),
// bit-verified against a freshly-computed transformers mel_filter_bank(...) matrix to ~1e-9 before this
// file was written (they're the same deterministic function of fixed n_fft/n_mels/sampling_rate, just
// serialised once). The rest of the pipeline (reflect-pad, periodic Hann, spectrum, power, log10/clamp/
// normalise) is ported from _torch_extract_fbank_features, the path transformers actually runs whenever
// torch is present (the numpy fallback in the same file is algorithmically equivalent but unused there).
//
// The spectrum step is a DIRECT DFT (dftPower/dftTwiddles below), not an FFT: Whisper's n_fft is fixed
// at 400, not a power of two, so a radix-2 FFT (gemma4's rfft, which validates a power-of-two fft_length
// at construction) does not apply here — see dftTwiddles' doc comment.

// FeatureConfig is the feature_extractor geometry + filterbank a Whisper checkpoint's
// preprocessor_config.json carries.
type FeatureConfig struct {
	NFFT         int
	HopLength    int
	NSamples     int // fixed window length in samples (chunk_length * sampling_rate — 480000 = 30s@16kHz)
	SamplingRate int
	NumMelBins   int         // feature_size
	MelFilters   [][]float64 // [NumMelBins][NFFT/2+1], read verbatim off the checkpoint
}

// featureConfigJSON mirrors the subset of preprocessor_config.json this package reads.
type featureConfigJSON struct {
	NFFT         int         `json:"n_fft"`
	HopLength    int         `json:"hop_length"`
	NSamples     int         `json:"n_samples"`
	SamplingRate int         `json:"sampling_rate"`
	FeatureSize  int         `json:"feature_size"`
	MelFilters   [][]float64 `json:"mel_filters"`
}

// LoadFeatureConfig reads preprocessor_config.json from a Whisper checkpoint directory.
func LoadFeatureConfig(dir string) (*FeatureConfig, error) {
	path := core.PathJoin(dir, "preprocessor_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("whisper.LoadFeatureConfig", "read "+path, resultErr(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.NewError("whisper.LoadFeatureConfig: " + path + " read returned non-byte data")
	}
	var fc featureConfigJSON
	if r := core.JSONUnmarshal(data, &fc); !r.OK {
		return nil, core.NewError("whisper.LoadFeatureConfig: parse " + path)
	}
	if fc.NFFT <= 0 || fc.HopLength <= 0 || fc.NSamples <= 0 || fc.SamplingRate <= 0 || fc.FeatureSize <= 0 {
		return nil, core.NewError("whisper.LoadFeatureConfig: " + path + " is missing n_fft/hop_length/n_samples/sampling_rate/feature_size")
	}
	if len(fc.MelFilters) != fc.FeatureSize {
		return nil, core.NewError(core.Sprintf("whisper.LoadFeatureConfig: mel_filters has %d rows, want feature_size %d", len(fc.MelFilters), fc.FeatureSize))
	}
	bins := fc.NFFT/2 + 1
	for i, row := range fc.MelFilters {
		if len(row) != bins {
			return nil, core.NewError(core.Sprintf("whisper.LoadFeatureConfig: mel_filters row %d has %d columns, want %d (n_fft/2+1)", i, len(row), bins))
		}
	}
	return &FeatureConfig{
		NFFT: fc.NFFT, HopLength: fc.HopLength, NSamples: fc.NSamples, SamplingRate: fc.SamplingRate,
		NumMelBins: fc.FeatureSize, MelFilters: fc.MelFilters,
	}, nil
}

// resultErr pulls the error out of a failed core.Result for wrapping, tolerating a Result whose Value is
// not an error (mirrors decode/generate/multimodal.go's helper of the same name in a sibling package).
func resultErr(r core.Result) error {
	if err, ok := r.Value.(error); ok {
		return err
	}
	return nil
}

// hannWindow returns the periodic Hann window of length n (torch.hann_window's default
// periodic=True: w[i] = 0.5 - 0.5·cos(2πi/n), the sum-of-cosines divided by n not n-1).
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	return w
}

// reflectPad centre-pads x by p samples on each side using numpy/torch "reflect" semantics (the edge
// sample itself is NOT repeated): left[i] = x[p-i] for i in [0,p), right[i] = x[N-2-i] for i in [0,p).
// Requires len(x) > p (true whenever NSamples ≥ NFFT, always the case for a real feature config).
func reflectPad(x []float64, p int) []float64 {
	n := len(x)
	out := make([]float64, n+2*p)
	for i := range p {
		out[i] = x[p-i]
		out[n+p+i] = x[n-2-i]
	}
	copy(out[p:p+n], x)
	return out
}

// ExtractLogMel converts a waveform into log-mel features: [NumMelBins][frames], frames =
// len(samples)/HopLength (3000 for a full 30 s/10 ms clip at the stock config). This is the general DSP
// primitive — it does NOT enforce the model's fixed 30 s/NSamples window (that is Model.Transcribe's
// policy in transcribe.go: pad short clips with silence to exactly NSamples, hard-refuse long ones —
// keeping the "pad/refuse to a fixed window" POLICY separate from the mel ALGORITHM below is what lets
// mel_test.go gate the algorithm against a small, fast, committable fixture instead of a full 480 000-
// sample clip). Ported from _torch_extract_fbank_features step for step: reflect-pad centre STFT →
// periodic Hann → power spectrum → the checkpoint's own Slaney mel filters → log10 → global (max-8)
// dynamic-range clamp → (x+4)/4 normalise. Accumulates in float64 throughout (the global max/clamp pass
// needs full-clip precision) and narrows to float32 only in the returned rows — matching the host-f32
// convention every other stage of this forward (attention.go/encoder.go/decoder.go) uses.
func (cfg *FeatureConfig) ExtractLogMel(samples []float32) ([][]float32, error) {
	if cfg == nil {
		return nil, core.NewError("whisper.ExtractLogMel: nil feature config")
	}
	if len(samples) < cfg.NFFT {
		return nil, core.NewError(core.Sprintf("whisper.ExtractLogMel: got %d samples, need at least NFFT (%d) to produce one frame", len(samples), cfg.NFFT))
	}
	wave := make([]float64, len(samples))
	for i, s := range samples {
		wave[i] = float64(s)
	}
	pad := cfg.NFFT / 2
	padded := reflectPad(wave, pad)
	window := hannWindow(cfg.NFFT)

	totalFrames := 1 + (len(padded)-cfg.NFFT)/cfg.HopLength
	frames := totalFrames - 1 // drop the last frame — stft[..., :-1] in the reference
	if frames <= 0 {
		return nil, core.NewError("whisper.ExtractLogMel: waveform too short to produce any mel frames")
	}

	bins := cfg.NFFT/2 + 1
	logSpec := make([][]float64, cfg.NumMelBins)
	for m := range logSpec {
		logSpec[m] = make([]float64, frames)
	}

	twiddleCos, twiddleSin := dftTwiddles(cfg.NFFT, bins)
	segment := make([]float64, cfg.NFFT)
	power := make([]float64, bins)
	globalMax := math.Inf(-1)
	for f := 0; f < frames; f++ {
		start := f * cfg.HopLength
		for n := range cfg.NFFT {
			segment[n] = padded[start+n] * window[n]
		}
		dftPower(segment, twiddleCos, twiddleSin, bins, power)
		for m := 0; m < cfg.NumMelBins; m++ {
			filter := cfg.MelFilters[m]
			var acc float64
			for b := range bins {
				acc += filter[b] * power[b]
			}
			if acc < 1e-10 {
				acc = 1e-10
			}
			v := math.Log10(acc)
			logSpec[m][f] = v
			if v > globalMax {
				globalMax = v
			}
		}
	}
	floor := globalMax - 8.0
	out := make([][]float32, cfg.NumMelBins)
	for m := range logSpec {
		row := logSpec[m]
		orow := make([]float32, frames)
		for f := range row {
			v := row[f]
			if v < floor {
				v = floor
			}
			orow[f] = float32((v + 4.0) / 4.0)
		}
		out[m] = orow
	}
	return out, nil
}

// dftTwiddles precomputes cos(-2πkt/n)/sin(-2πkt/n) for every (bin k, sample t) pair — a direct DFT,
// NOT a radix-2 FFT: Whisper's n_fft is FIXED at 400, which is not a power of two (400 = 2⁴·25), so the
// power-of-two-only bit-reversal FFT model/gemma4/audio.rfft uses (correct there — gemma4's front end
// validates fft_length is a power of two at construction) panics out of bounds on Whisper's real n_fft.
// A direct O(n·bins) DFT has no such restriction and is one-time work per transcription (bins·n ≈ 80 000
// multiply-adds to build, reused unchanged across every frame) — plenty fast for a correctness-first host
// forward. Returned tables are [bins][n] flat, row-major per bin.
func dftTwiddles(n, bins int) (cosT, sinT []float64) {
	cosT = make([]float64, bins*n)
	sinT = make([]float64, bins*n)
	for k := range bins {
		w := -2 * math.Pi * float64(k) / float64(n)
		row := k * n
		for t := range n {
			s, c := math.Sincos(w * float64(t))
			cosT[row+t] = c
			sinT[row+t] = s
		}
	}
	return cosT, sinT
}

// dftPower computes |DFT(segment)|² for bins [0,bins) into power, using the precomputed twiddle tables
// from dftTwiddles(len(segment), bins).
func dftPower(segment, cosT, sinT []float64, bins int, power []float64) {
	n := len(segment)
	for k := range bins {
		row := k * n
		var re, im float64
		for t, x := range segment {
			re += x * cosT[row+t]
			im += x * sinT[row+t]
		}
		power[k] = re*re + im*im
	}
}
