// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

// hipDecodeWAVMono16k converts 16-bit PCM RIFF/WAVE input to the mono 16 kHz
// float32 waveform consumed by Gemma 4's Conformer audio tower.
func hipDecodeWAVMono16k(data []byte) ([]float32, error) {
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, core.NewError("hip.DecodeWAVMono16k: not a RIFF/WAVE file")
	}
	var formatOK bool
	var channels, sampleRate int
	var audioOffset, audioBytes int
	for offset := 12; offset+8 <= len(data); {
		chunkID := string(data[offset : offset+4])
		chunkBytes := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		body := offset + 8
		if chunkBytes > len(data)-body {
			chunkBytes = len(data) - body
		}
		switch chunkID {
		case "fmt ":
			if chunkBytes < 16 {
				return nil, core.NewError("hip.DecodeWAVMono16k: short fmt chunk")
			}
			format := binary.LittleEndian.Uint16(data[body : body+2])
			channels = int(binary.LittleEndian.Uint16(data[body+2 : body+4]))
			sampleRate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
			bits := binary.LittleEndian.Uint16(data[body+14 : body+16])
			if format != 1 || bits != 16 {
				return nil, core.NewError("hip.DecodeWAVMono16k: want 16-bit PCM (format 1)")
			}
			if channels < 1 || channels > 8 || sampleRate <= 0 {
				return nil, core.NewError("hip.DecodeWAVMono16k: malformed channel count or sample rate")
			}
			formatOK = true
		case "data":
			audioOffset, audioBytes = body, chunkBytes
		}
		offset = body + chunkBytes + (chunkBytes & 1)
	}
	if !formatOK || audioBytes < 2 {
		return nil, core.NewError("hip.DecodeWAVMono16k: missing fmt or data chunk")
	}
	frameBytes := channels * 2
	frames := audioBytes / frameBytes
	if frames == 0 {
		return nil, core.NewError("hip.DecodeWAVMono16k: empty audio data")
	}
	mono := make([]float32, frames)
	for frame := range frames {
		var sum float32
		base := audioOffset + frame*frameBytes
		for channel := range channels {
			sample := int16(binary.LittleEndian.Uint16(data[base+channel*2 : base+channel*2+2]))
			sum += float32(sample) / 32768
		}
		mono[frame] = sum / float32(channels)
	}
	if sampleRate == 16000 {
		return mono, nil
	}
	return hipResampleAudioTo16k(mono, sampleRate), nil
}

func hipResampleAudioTo16k(input []float32, sourceRate int) []float32 {
	const destinationRate = 16000
	ratio := float64(sourceRate) / destinationRate
	cutoff := 1.0
	if sourceRate > destinationRate {
		cutoff = 1 / ratio
	}
	const taps = 32
	const beta = 8.0
	i0Beta := hipKaiserI0(beta)
	outputCount := int(float64(len(input))/ratio + 0.5)
	if outputCount <= 0 {
		outputCount = 1
	}
	output := make([]float32, outputCount)
	for outputIndex := range outputCount {
		center := float64(outputIndex) * ratio
		low := int(math.Ceil(center)) - taps
		high := int(math.Floor(center)) + taps
		var sum, weightSum float64
		for inputIndex := max(low, 0); inputIndex <= min(high, len(input)-1); inputIndex++ {
			x := (float64(inputIndex) - center) * cutoff
			sinc := 1.0
			if x != 0 {
				piX := math.Pi * x
				sinc = math.Sin(piX) / piX
			}
			windowPosition := (float64(inputIndex) - center) / taps
			if windowPosition < -1 || windowPosition > 1 {
				continue
			}
			window := hipKaiserI0(beta*math.Sqrt(1-windowPosition*windowPosition)) / i0Beta
			weight := sinc * window
			sum += float64(input[inputIndex]) * weight
			weightSum += weight
		}
		if weightSum != 0 {
			sum /= weightSum
		}
		output[outputIndex] = float32(sum)
	}
	return output
}

func hipKaiserI0(x float64) float64 {
	sum, term := 1.0, 1.0
	half := x / 2
	for k := 1; k <= 24; k++ {
		term *= (half / float64(k)) * (half / float64(k))
		sum += term
		if term < 1e-12*sum {
			break
		}
	}
	return sum
}
