// SPDX-License-Identifier: EUPL-1.2

package work

import (
	"testing"

	core "dappco.re/go"
)

var transitionBenchmarkSink core.Result

func BenchmarkTransition(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		transitionBenchmarkSink = Transition(RunRunning, RunCompleted)
	}
}
