// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

// encoder.go is WhisperEncoder ported host-side: conv1(k3,s1,p1)+GELU → conv2(k3,s2,p1)+GELU (the 2×
// time subsample) → transpose to time-major → + the FULL fixed position table (never sliced — Whisper
// always runs exactly MaxSourcePositions=1500 positions, enforced upstream by the 30 s window) → N
// pre-LN transformer layers (non-causal self-attention + FFN) → one final top-level LayerNorm. Ported
// directly from modeling_whisper.py's WhisperEncoder.forward — see that file's read notes in this
// package's git history for the exact line-by-line correspondence.

// conv1D runs a single Conv1d(inC,outC,kernel,stride,padding) over input[inC][T], weight [outC,inC,kernel]
// row-major (the PyTorch/safetensors layout), bias [outC]. Returns output[outC][outT].
func conv1D(input [][]float32, weight, bias []float32, inC, outC, kernel, stride, padding int) [][]float32 {
	T := len(input[0])
	outT := (T+2*padding-kernel)/stride + 1
	out := make([][]float32, outC)
	for oc := range outC {
		row := make([]float32, outT)
		wBase := oc * inC * kernel
		for ot := range outT {
			var acc float64
			start := ot*stride - padding
			for ic := range inC {
				irow := input[ic]
				wrow := weight[wBase+ic*kernel : wBase+ic*kernel+kernel]
				for k := range kernel {
					pos := start + k
					if pos < 0 || pos >= T {
						continue // zero padding
					}
					acc += float64(wrow[k]) * float64(irow[pos])
				}
			}
			acc += float64(bias[oc])
			row[ot] = float32(acc)
		}
		out[oc] = row
	}
	return out
}

func geluConv(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, row := range x {
		out[i] = geluRow(row)
	}
	return out
}

// EncodeAudio runs the full encoder stack over log-mel features [NumMelBins][frames] (frames must equal
// MaxSourcePositions*2 — the conv stack's fixed 2× subsample; callers pad/refuse before extraction so
// this is always exactly 3000 for the stock 30 s/10 ms config), returning the encoder output
// [MaxSourcePositions][DModel] flat time-major.
func EncodeAudio(melFeatures [][]float32, w *Weights, cfg *Config) ([]float32, error) {
	if w == nil || cfg == nil {
		return nil, core.NewError("whisper.EncodeAudio: nil weights/config")
	}
	if len(melFeatures) != cfg.NumMelBins {
		return nil, core.NewError(core.Sprintf("whisper.EncodeAudio: got %d mel bins, want %d", len(melFeatures), cfg.NumMelBins))
	}
	expected := cfg.MaxSourcePositions * 2
	if len(melFeatures[0]) != expected {
		return nil, core.NewError(core.Sprintf("whisper.EncodeAudio: got %d mel frames, want exactly %d (%d×2 — pad short audio, refuse long audio before calling)", len(melFeatures[0]), expected, cfg.MaxSourcePositions))
	}

	c1 := geluConv(conv1D(melFeatures, w.Conv1Weight, w.Conv1Bias, cfg.NumMelBins, cfg.DModel, 3, 1, 1))
	c2 := geluConv(conv1D(c1, w.Conv2Weight, w.Conv2Bias, cfg.DModel, cfg.DModel, 3, 2, 1))
	Tout := len(c2[0])
	if Tout != cfg.MaxSourcePositions {
		return nil, core.NewError(core.Sprintf("whisper.EncodeAudio: conv stack produced %d positions, want %d", Tout, cfg.MaxSourcePositions))
	}

	D := cfg.DModel
	hidden := make([]float32, Tout*D)
	for t := range Tout {
		for d := range D {
			hidden[t*D+d] = c2[d][t] + w.EncoderPos[t*D+d]
		}
	}

	for _, layer := range w.EncoderLayers {
		residual := hidden
		normed := layerNormForward(hidden, layer.SelfAttnNorm, Tout, D)
		attnOut, err := selfAttentionForward(normed, Tout, D, cfg.EncoderAttentionHeads, false, layer.SelfAttn)
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, attnOut)

		residual = hidden
		normed = layerNormForward(hidden, layer.FinalNorm, Tout, D)
		ff := linearForward(geluRow(linearForward(normed, layer.FC1, Tout)), layer.FC2, Tout)
		hidden = addRows(residual, ff)
	}
	return layerNormForward(hidden, w.EncoderFinalNorm, Tout, D), nil
}

func addRows(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}
