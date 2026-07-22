// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"encoding/binary"
	"testing"
)

// buildWAV assembles a minimal RIFF/WAVE file: PCM format, the given channel count/rate/bit depth, and
// samples as raw little-endian bytes (already at the target bit depth).
func buildWAV(channels, rate, bits int, sampleBytes []byte) []byte {
	blockAlign := channels * bits / 8
	byteRate := rate * blockAlign
	dataLen := len(sampleBytes)
	riffLen := 4 + (8 + 16) + (8 + dataLen)

	buf := make([]byte, 0, 8+riffLen)
	buf = append(buf, "RIFF"...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(riffLen))
	buf = append(buf, "WAVE"...)
	buf = append(buf, "fmt "...)
	buf = binary.LittleEndian.AppendUint32(buf, 16)
	buf = binary.LittleEndian.AppendUint16(buf, 1) // PCM
	buf = binary.LittleEndian.AppendUint16(buf, uint16(channels))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(rate))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(byteRate))
	buf = binary.LittleEndian.AppendUint16(buf, uint16(blockAlign))
	buf = binary.LittleEndian.AppendUint16(buf, uint16(bits))
	buf = append(buf, "data"...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(dataLen))
	buf = append(buf, sampleBytes...)
	return buf
}

func int16LEBytes(samples []int16) []byte {
	b := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

// TestDecodeWAV16Mono_Good parses a valid 16-bit PCM mono 16 kHz clip and normalises to [-1,1].
func TestDecodeWAV16Mono_Good(t *testing.T) {
	raw := buildWAV(1, 16000, 16, int16LEBytes([]int16{0, 16384, -32768, 32767}))
	samples, err := DecodeWAV16Mono(raw)
	if err != nil {
		t.Fatalf("DecodeWAV16Mono: %v", err)
	}
	if len(samples) != 4 {
		t.Fatalf("len(samples) = %d, want 4", len(samples))
	}
	if samples[0] != 0 {
		t.Fatalf("samples[0] = %v, want 0", samples[0])
	}
	if samples[2] != -1 {
		t.Fatalf("samples[2] = %v, want -1.0 (int16 min / 32768)", samples[2])
	}
}

// TestDecodeWAV16Mono_Bad proves stereo is refused, not silently downmixed (unlike engine/metal's
// DecodeWAVMono16k — see wav.go's doc comment on why this package deliberately does not convert).
func TestDecodeWAV16Mono_Bad(t *testing.T) {
	raw := buildWAV(2, 16000, 16, int16LEBytes([]int16{0, 0, 1, 1}))
	if _, err := DecodeWAV16Mono(raw); err == nil {
		t.Fatal("DecodeWAV16Mono accepted stereo audio")
	}
}

// TestDecodeWAV16Mono_Ugly proves a non-RIFF file is refused with a clear message, not a panic on the
// 44-byte header-length assumption.
func TestDecodeWAV16Mono_Ugly(t *testing.T) {
	if _, err := DecodeWAV16Mono([]byte("not a wav file at all")); err == nil {
		t.Fatal("DecodeWAV16Mono accepted a non-RIFF file")
	}
	if _, err := DecodeWAV16Mono(nil); err == nil {
		t.Fatal("DecodeWAV16Mono accepted nil data")
	}
}

func TestDecodeWAV16Mono_WrongRate_Bad(t *testing.T) {
	raw := buildWAV(1, 44100, 16, int16LEBytes([]int16{0, 1}))
	if _, err := DecodeWAV16Mono(raw); err == nil {
		t.Fatal("DecodeWAV16Mono accepted 44.1 kHz audio")
	}
}

func TestDecodeWAV16Mono_WrongBitDepth_Bad(t *testing.T) {
	// 8-bit PCM: blockAlign/byteRate computed for 8 bits, one byte per sample.
	raw := buildWAV(1, 16000, 8, []byte{128, 200})
	if _, err := DecodeWAV16Mono(raw); err == nil {
		t.Fatal("DecodeWAV16Mono accepted 8-bit audio")
	}
}

func TestDecodeWAV16Mono_NonPCMFormat_Bad(t *testing.T) {
	raw := buildWAV(1, 16000, 16, int16LEBytes([]int16{0, 1}))
	// flip the format tag (byte offset 20, little-endian uint16) from 1 (PCM) to 3 (IEEE float).
	raw[20] = 3
	if _, err := DecodeWAV16Mono(raw); err == nil {
		t.Fatal("DecodeWAV16Mono accepted a non-PCM format tag")
	}
}

func TestDecodeWAV16Mono_EmptyData_Ugly(t *testing.T) {
	raw := buildWAV(1, 16000, 16, nil)
	if _, err := DecodeWAV16Mono(raw); err == nil {
		t.Fatal("DecodeWAV16Mono accepted an empty data chunk")
	}
}
