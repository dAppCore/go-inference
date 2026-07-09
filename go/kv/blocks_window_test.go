// SPDX-Licence-Identifier: EUPL-1.2

package kv

import "testing"

// TestBlocks_SliceKVSnapshotTensor_GoodBad covers the clone=true public wrapper
// sliceKVSnapshotTensor (blocks.go) — it forwards to the covered
// sliceKVSnapshotTensorOpt with clone=true. Good returns an independent clone of
// the requested row range; Bad trips the shape guard with a zero seqLen.
func TestBlocks_SliceKVSnapshotTensor_GoodBad(t *testing.T) {
	values := cvtF32(4, 2) // 8 values, rows of 2

	got, err := sliceKVSnapshotTensor(values, 1, 3, 2, 4)
	if err != nil {
		t.Fatalf("sliceKVSnapshotTensor() error = %v", err)
	}
	want := []float32{2, 3, 4, 5} // rows [1,3) of headDim 2
	if len(got) != len(want) {
		t.Fatalf("slice len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("slice = %v, want %v", got, want)
		}
	}
	// Clone independence: mutating the clone must not touch the source.
	got[0] = -1
	if values[2] == -1 {
		t.Fatal("sliceKVSnapshotTensor returned a view, want an independent clone")
	}

	// Bad: zero seqLen trips the shape guard.
	if _, err := sliceKVSnapshotTensor(values, 0, 1, 2, 0); err == nil {
		t.Fatal("sliceKVSnapshotTensor(seqLen 0) error = nil, want shape error")
	}
}

// TestBlocks_SliceKVSnapshotRawTensor_GoodBad covers the clone=true public
// wrapper sliceKVSnapshotRawTensor (blocks.go). Good clones the requested
// row range out of an f16 raw payload; Bad passes an unsupported dtype.
func TestBlocks_SliceKVSnapshotRawTensor_GoodBad(t *testing.T) {
	raw := cvtRawF16(4, 2) // 8 f16 values across 4 rows of 2

	got, err := sliceKVSnapshotRawTensor(raw, "float16", 1, 3, 4, 8)
	if err != nil {
		t.Fatalf("sliceKVSnapshotRawTensor() error = %v", err)
	}
	// rows [1,3) of headDim 2 → 4 f16 values → 8 bytes.
	if len(got) != 8 {
		t.Fatalf("raw slice len = %d, want 8 bytes", len(got))
	}
	got[0] ^= 0xff
	if raw[4] == got[0] {
		t.Fatal("sliceKVSnapshotRawTensor returned a view, want an independent clone")
	}

	// Bad: an unsupported dtype trips the dtype guard.
	if _, err := sliceKVSnapshotRawTensor(raw, "nonsense", 0, 1, 4, 8); err == nil {
		t.Fatal("sliceKVSnapshotRawTensor(bad dtype) error = nil, want dtype error")
	}
}

// TestBlocks_SliceKVSnapshotLayerRawTensor_GoodBad covers the clone=true public
// wrapper sliceKVSnapshotLayerRawTensor (blocks.go). Good slices a
// [B,H,L,D] native slab down the L axis; Bad passes a non-4D shape.
func TestBlocks_SliceKVSnapshotLayerRawTensor_GoodBad(t *testing.T) {
	// B=1, H=1, L=4, D=2 → 8 f16 values → 16 bytes.
	raw := cvtRawF16(4, 2)
	shape := []int32{1, 1, 4, 2}

	got, outShape, err := sliceKVSnapshotLayerRawTensor(raw, "float16", shape, 1, 3)
	if err != nil {
		t.Fatalf("sliceKVSnapshotLayerRawTensor() error = %v", err)
	}
	if len(outShape) != 4 || outShape[2] != 2 {
		t.Fatalf("outShape = %v, want L dimension 2", outShape)
	}
	// take=2 rows × D=2 × 2 bytes = 8 bytes.
	if len(got) != 8 {
		t.Fatalf("layer raw slice len = %d, want 8 bytes", len(got))
	}

	// Bad: a non-4D shape trips the layer-raw guard.
	if _, _, err := sliceKVSnapshotLayerRawTensor(raw, "float16", []int32{4, 2}, 0, 1); err == nil {
		t.Fatal("sliceKVSnapshotLayerRawTensor(non-4D shape) error = nil, want shape error")
	}
}
