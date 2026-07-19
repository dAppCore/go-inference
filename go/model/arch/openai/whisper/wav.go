// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

// wav.go decodes the fixed WAV contract `lem transcribe` documents: 16-bit PCM, mono, 16 kHz (the
// existing --audio ingestion precedent — decode/generate/multimodal.go's doc comment on the --audio
// flag). Deliberately NOT resampling/downmixing (unlike engine/metal's DecodeWAVMono16k /
// engine/hip's hipDecodeWAVMono16k, which accept arbitrary rate/channel counts and convert): a model/
// arch/* package must not import engine/* (AX-8, "lib never imports consumer"; CLAUDE.md: "Backends
// import this [go/*.go]; never the reverse"), so this is a small, independent, STRICT parser — any
// shape outside the documented contract is a clear refusal, not a silent best-effort conversion.

// DecodeWAV16Mono parses a RIFF/WAVE file and returns its samples as [-1,1] float32 — strictly 16-bit
// PCM, mono, 16 kHz; anything else is a clear, named refusal (never resampled/downmixed).
func DecodeWAV16Mono(data []byte) ([]float32, error) {
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, core.NewError("whisper.DecodeWAV16Mono: not a RIFF/WAVE file")
	}
	le16 := func(off int) int { return int(data[off]) | int(data[off+1])<<8 }
	le32 := func(off int) int {
		return int(data[off]) | int(data[off+1])<<8 | int(data[off+2])<<16 | int(data[off+3])<<24
	}
	var fmtOK bool
	var channels, rate, bits int
	var dataOff, dataLen int
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := le32(off + 4)
		if size < 0 {
			return nil, core.NewError("whisper.DecodeWAV16Mono: negative chunk size")
		}
		body := off + 8
		if body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, core.NewError("whisper.DecodeWAV16Mono: short fmt chunk")
			}
			format := le16(body)
			channels, rate, bits = le16(body+2), le32(body+4), le16(body+14)
			if format != 1 {
				return nil, core.NewError("whisper.DecodeWAV16Mono: want PCM (format 1), got format " + core.Sprintf("%d", format))
			}
			fmtOK = true
		case "data":
			dataOff, dataLen = body, size
		}
		off = body + size + (size & 1) // chunks are word-aligned
	}
	if !fmtOK {
		return nil, core.NewError("whisper.DecodeWAV16Mono: missing fmt chunk")
	}
	if dataLen < 2 {
		return nil, core.NewError("whisper.DecodeWAV16Mono: missing or empty data chunk")
	}
	if bits != 16 {
		return nil, core.NewError(core.Sprintf("whisper.DecodeWAV16Mono: want 16-bit PCM, got %d-bit — re-encode to 16-bit PCM mono 16 kHz", bits))
	}
	if channels != 1 {
		return nil, core.NewError(core.Sprintf("whisper.DecodeWAV16Mono: want mono, got %d channels — re-encode to 16-bit PCM mono 16 kHz", channels))
	}
	if rate != 16000 {
		return nil, core.NewError(core.Sprintf("whisper.DecodeWAV16Mono: want 16 kHz, got %d Hz — re-encode to 16-bit PCM mono 16 kHz", rate))
	}
	n := dataLen / 2
	if n == 0 {
		return nil, core.NewError("whisper.DecodeWAV16Mono: empty audio data")
	}
	samples := make([]float32, n)
	for i := range n {
		s := int16(uint16(data[dataOff+2*i]) | uint16(data[dataOff+2*i+1])<<8)
		samples[i] = float32(s) / 32768
	}
	return samples, nil
}
