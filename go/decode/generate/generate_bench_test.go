// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"testing"

	core "dappco.re/go"
)

// Bench sinks — package-level so the compiler cannot elide the benchmarked call.
var (
	benchPrefix string
	benchBytes  []byte
)

// benchLongMultibytePrompt builds a prompt longer than warmPrefixChars whose cut
// boundary lands inside a 3-byte rune, so warmPrefix must run its UTF-8 backoff.
func benchLongMultibytePrompt() string {
	buf := make([]byte, 0, warmPrefixChars+1200)
	for i := 0; i < warmPrefixChars-1; i++ {
		buf = append(buf, 'a')
	}
	buf = append(buf, "日"...) // 3-byte rune straddling the warmPrefixChars cut
	for i := 0; i < 1000; i++ {
		buf = append(buf, 'b')
	}
	return string(buf)
}

// warmPrefix on an over-long prompt: exercises the rune-boundary backoff scan.
// Returns a substring of the input, so the steady-state expectation is 0 allocs.
func BenchmarkWarmPrefix_LongMultibyte(b *testing.B) {
	prompt := benchLongMultibytePrompt()
	b.ReportAllocs()
	for b.Loop() {
		benchPrefix = warmPrefix(prompt)
	}
}

// warmPrefix on a short prompt: the len <= warmPrefixChars fast path, returned
// unchanged (0 allocs).
func BenchmarkWarmPrefix_ShortPassthrough(b *testing.B) {
	const prompt = "Write a haiku about the sea."
	b.ReportAllocs()
	for b.Loop() {
		benchPrefix = warmPrefix(prompt)
	}
}

// decodeImageDataURL on a realistic ~1 KiB base64 "data:" image URL — the pure-CPU
// decode path serve and the CLI share (no file I/O, no model). The allocation
// floor is the decoded output the caller keeps.
func BenchmarkDecodeImageDataURL(b *testing.B) {
	raw := make([]byte, 1024)
	for i := range raw {
		raw[i] = byte(i * 7)
	}
	url := "data:image/png;base64," + core.Base64Encode(raw)
	b.ReportAllocs()
	for b.Loop() {
		out, err := decodeImageDataURL(url)
		if err != nil {
			b.Fatal(err)
		}
		benchBytes = out
	}
}

// resolveImageInputs over a small set of base64 "data:" URLs — the loop + cap
// checks + per-source decode, all pure CPU (data: URLs never touch the disk).
func BenchmarkResolveImageInputs_DataURLs(b *testing.B) {
	raw := make([]byte, 256)
	for i := range raw {
		raw[i] = byte(i)
	}
	url := "data:image/png;base64," + core.Base64Encode(raw)
	sources := []string{url, url, url}
	b.ReportAllocs()
	for b.Loop() {
		out, err := resolveImageInputs(sources)
		if err != nil {
			b.Fatal(err)
		}
		benchBytes = out[0]
	}
}
