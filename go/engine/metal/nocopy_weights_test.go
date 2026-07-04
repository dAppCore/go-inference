// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"

	"dappco.re/go/inference/safetensors"
)

func TestShardBuffersEmptyWeightIsOptional(t *testing.T) {
	var sb shardBuffers
	got, err := sb.bufFor(nil)
	if err != nil {
		t.Fatalf("bufFor(nil): %v", err)
	}
	if got.buf != nil || got.off != 0 {
		t.Fatalf("bufFor(nil) = %+v, want zero bufView", got)
	}
}

func TestShardBuffersRejectsForeignWeight(t *testing.T) {
	weight := []byte{1, 2, 3, 4}
	sb := shardBuffers{bases: []uintptr{1}, ends: []uintptr{2}}
	if _, err := sb.bufFor(weight); err == nil {
		t.Fatal("expected bufFor to reject a weight outside mapped shards")
	}
}

func TestShardBuffersCloseIsNilSafe(t *testing.T) {
	var sb *shardBuffers
	if err := sb.Close(); err != nil {
		t.Fatalf("nil Close: %v", err)
	}
	if err := (&shardBuffers{}).Close(); err != nil {
		t.Fatalf("empty Close: %v", err)
	}
}

func TestShardBuffersCloseEvictsResidentBuffersForShardRanges(t *testing.T) {
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	shard := []byte{1, 2, 3, 4, 5, 6}
	other := []byte{7, 8, 9, 10}
	shardBase := uintptr(unsafe.Pointer(&shard[0]))
	otherBase := uintptr(unsafe.Pointer(&other[0]))
	shardKey := shardBase + 1

	residentBufMu.Lock()
	residentBufs[shardKey] = residentBuf{pin: shard[1:3]}
	residentBufs[otherBase] = residentBuf{pin: other}
	residentBufMu.Unlock()

	sb := &shardBuffers{
		dm:    &safetensors.DirMapping{},
		bases: []uintptr{shardBase},
		ends:  []uintptr{shardBase + uintptr(len(shard))},
	}
	if err := sb.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	residentBufMu.Lock()
	_, shardOK := residentBufs[shardKey]
	_, otherOK := residentBufs[otherBase]
	residentBufMu.Unlock()
	if shardOK {
		t.Fatal("Close left a resident buffer whose key belongs to the closing shard range")
	}
	if !otherOK {
		t.Fatal("Close evicted an unrelated resident buffer outside the closing shard range")
	}
}
