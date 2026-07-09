// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// gatherRoutesParityCase drives ONE all-routes gather through both dispatch paths —
// MLX's affine_gather_qmv (shape/stride constant buffers) and the lean fc-specialised
// lthn_gather_qmv (#280) — on identical inputs, returning both output slabs.
func gatherRoutesParityCase(t *testing.T, outDim, inDim, groupSize, bits, numExperts, expertRows, rowBase int, batchedX bool, routes int) ([]byte, []byte) {
	t.Helper()
	packed := make([]byte, numExperts*expertRows*inDim*bits/8)
	for i := range packed {
		packed[i] = byte((i*131 + 17) % 256)
	}
	nSB := numExperts * expertRows * (inDim / groupSize)
	scales := toBF16Bytes(syntheticFloat32(nSB, 11))
	biases := toBF16Bytes(syntheticFloat32(nSB, 13))

	xRows := 1
	lhs := make([]uint32, routes)
	if batchedX {
		xRows = routes
		for i := range lhs {
			lhs[i] = uint32((i * 2) % xRows)
		}
	}
	x := toBF16Bytes(syntheticFloat32(xRows*inDim, 7))
	rhs := make([]uint32, routes)
	for i := range rhs {
		rhs[i] = uint32((i*3 + 1) % numExperts)
	}

	mlxPSO, err := gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits)
	if err != nil {
		t.Skipf("MLX gather pipeline unavailable (gs=%d b=%d): %v", groupSize, bits, err)
	}
	metaKey := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: outDim, inDim: inDim, groupSize: groupSize, bits: bits, expertRows: expertRows, routes: routes, xRows: xRows, batchedX: batchedX}
	meta, err := gatherQMVAllRoutesMetadata(numExperts, outDim, inDim, groupSize, bits, expertRows, routes, xRows, batchedX)
	if err != nil {
		t.Fatalf("gatherQMVAllRoutesMetadata: %v", err)
	}

	outLen := routes * outDim * bf16Size
	run := func(lean bool) []byte {
		wasDisabled := leanGatherDisabled
		leanGatherDisabled = !lean
		defer func() { leanGatherDisabled = wasDisabled }()
		before := leanGatherDispatches.Load()

		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&packed[0]), uint(len(packed)), metal.MTLResourceStorageModeShared)
		sBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&scales[0]), uint(len(scales)), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&biases[0]), uint(len(biases)), metal.MTLResourceStorageModeShared)
		lhsBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&lhs[0]), uint(len(lhs)*4), metal.MTLResourceStorageModeShared)
		rhsBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&rhs[0]), uint(len(rhs)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitGatherQMVAllRoutes(encSink{enc}, mlxPSO, meta, metaKey, xBuf, 0, wBuf, 0, sBuf, 0, bBuf, 0, lhsBuf, rhsBuf, 0, outBuf, 0, outDim, inDim, groupSize, bits, rowBase, routes)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		engaged := leanGatherDispatches.Load() > before
		if lean && !engaged {
			t.Fatalf("lean gather lane did not engage (gs=%d b=%d fast=%v batchedX=%v)", groupSize, bits, outDim%8 == 0 && inDim%512 == 0, batchedX)
		}
		if !lean && engaged {
			t.Fatal("lean gather lane engaged while disabled — the A/B is vacuous")
		}
		return append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), outLen)...)
	}

	return run(false), run(true)
}

// TestQmvGather_emitGatherQMVAllRoutes_LeanMatchesMLX gates #280: the lean
// fc-specialised lthn_gather_qmv dispatch must produce BYTE-IDENTICAL output to
// MLX's affine_gather_qmv shape/stride-buffer path — the dot arithmetic is the
// same qmv[_fast]_impl by construction, so any drift is an addressing bug. The
// sweep covers every instantiated width on both variants (fast and plain), the
// shared-x and batched-x lhs modes, and the fused gate_up slab's rowBase offset.
func TestQmvGather_emitGatherQMVAllRoutes_LeanMatchesMLX(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded — run `task metallib:kernels`")
	}

	const numExperts, routes = 4, 3
	for _, bits := range []int{2, 3, 4, 5, 6, 8} {
		for _, c := range []struct {
			name                string
			outDim, inDim, gs   int
			expertRows, rowBase int
			batchedX            bool
		}{
			{"fast shared-x", 32, 512, 64, 32, 0, false},
			{"fast batched-x", 32, 512, 64, 32, 0, true},
			{"plain shared-x", 24, 256, 32, 24, 0, false},
			{"fast gate_up rowBase", 32, 512, 64, 64, 32, false},
		} {
			if c.inDim*bits%32 != 0 {
				continue // the packed-row alignment guard rejects these upstream
			}
			mlx, lean := gatherRoutesParityCase(t, c.outDim, c.inDim, c.gs, bits, numExperts, c.expertRows, c.rowBase, c.batchedX, routes)
			if !bytes.Equal(mlx, lean) {
				for i := 0; i+1 < len(mlx); i += 2 {
					if mlx[i] != lean[i] || mlx[i+1] != lean[i+1] {
						t.Logf("b%d %s: first diff at elem %d: mlx % x (%.9g) lean % x (%.9g)", bits, c.name, i/2,
							mlx[i:i+2], bf16ToF32(mlx[i], mlx[i+1]), lean[i:i+2], bf16ToF32(lean[i], lean[i+1]))
						break
					}
				}
				t.Fatalf("b%d %s: lean gather != MLX gather (cosine=%.7f)", bits, c.name, cosineBF16(lean, mlx))
			}
		}
	}
	t.Logf("lean fc-specialised gather matches MLX byte-for-byte at every width, both lhs modes, rowBase slab offset")
}
