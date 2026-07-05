// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import "testing"

// BenchmarkEscapeLp_Plain measures the common ingest path: tag values that
// contain none of the line-protocol special characters, so every Replace is a
// no-op and the function should not allocate.
func BenchmarkEscapeLp_Plain(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = EscapeLp("gemma4-philosophy-calm")
	}
}

// BenchmarkEscapeLp_Escaped measures the worst case: a value carrying commas,
// equals signs and spaces, so all three Replace passes allocate.
func BenchmarkEscapeLp_Escaped(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = EscapeLp("model=gemma4, voice=calm narrator")
	}
}
