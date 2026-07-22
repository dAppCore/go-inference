// SPDX-Licence-Identifier: EUPL-1.2
package ernie45

import (
	"dappco.re/go/inference/model"
	"testing"
)

var sink model.Arch

func BenchmarkConfig(b *testing.B) {
	s, _ := model.LookupArch("ernie4_5")
	c, _ := s.Parse([]byte(realConfig))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink, _ = c.Arch()
	}
}
