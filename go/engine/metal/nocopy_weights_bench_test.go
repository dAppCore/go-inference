// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"
)

func BenchmarkShardBufferResidentRangeEviction(b *testing.B) {
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const entries = 64
	shard := make([]byte, entries+1)
	other := make([]byte, entries+1)
	shardBase := uintptr(unsafe.Pointer(&shard[0]))
	otherBase := uintptr(unsafe.Pointer(&other[0]))
	bases := []uintptr{shardBase}
	ends := []uintptr{shardBase + uintptr(len(shard))}
	shardKeys := make([]uintptr, entries)
	otherKeys := make([]uintptr, entries)
	for i := range entries {
		shardKeys[i] = shardBase + uintptr(i)
		otherKeys[i] = otherBase + uintptr(i)
	}

	residentBufMu.Lock()
	residentBufs = make(map[uintptr]residentBuf, entries*2)
	for i := range entries {
		residentBufs[shardKeys[i]] = residentBuf{pin: shard[i : i+1]}
		residentBufs[otherKeys[i]] = residentBuf{pin: other[i : i+1]}
	}
	residentBufMu.Unlock()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evictResidentBufsForRanges(bases, ends)

		b.StopTimer()
		residentBufMu.Lock()
		for j := range entries {
			residentBufs[shardKeys[j]] = residentBuf{pin: shard[j : j+1]}
			residentBufs[otherKeys[j]] = residentBuf{pin: other[j : j+1]}
		}
		residentBufMu.Unlock()
		b.StartTimer()
	}
}
