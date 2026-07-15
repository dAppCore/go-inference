// SPDX-Licence-Identifier: EUPL-1.2
package hunyuan

import (
	"dappco.re/go/inference/model"
	"testing"
)

var sink model.Arch

func BenchmarkConfig(b *testing.B) {
	s, _ := model.LookupArch("hunyuan_v1_dense")
	c, _ := s.Parse([]byte(realConfig))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink, _ = c.Arch()
	}
}
