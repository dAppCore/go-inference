// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"testing"
)

var parserBenchmarkSink []Output

func BenchmarkParseLine(b *testing.B) {
	result := DefaultRegistry(nil, nil)
	if !result.OK {
		b.Fatal(result.Error())
	}
	registry := result.Value.(*Registry)
	adapter := registry.Adapter("codex").Value.(Adapter)
	line := `{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		parserBenchmarkSink = adapter.ParseLine("stdout", line)
	}
}
