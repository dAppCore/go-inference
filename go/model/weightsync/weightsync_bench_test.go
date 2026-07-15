// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the weight-swap coordinator (AX-11). The Update
// success path (one per RL checkpoint) and Begin/End (a pair per live
// generation) fold only scalar version/counter state under a mutex, so with a
// no-op Applier they must not allocate. These benches pin that to zero.
//
// Run: go test -bench=. -benchmem -run='^$' ./weightsync/
package weightsync

import (
	"context"
	"testing"
)

// benchApplier is an allocation-free Applier so each bench isolates the
// coordinator's own per-call buffering (none, on the success path) from the
// real GPU weight apply.
type benchApplier struct{}

func (benchApplier) Stage(context.Context, uint64, string) error { return nil }
func (benchApplier) Activate(context.Context, uint64) error      { return nil }

var benchErr error

func BenchmarkCoordinator_Update(b *testing.B) {
	co := New(benchApplier{})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErr = co.Update(ctx, uint64(i)+1, "ckpt") // versions strictly advance
	}
}

func BenchmarkCoordinator_BeginEnd(b *testing.B) {
	co := New(benchApplier{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		co.Begin()
		co.End()
	}
}
