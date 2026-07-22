// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestVerifyStackICBReplay_KVQ8StoreParity_Good is #71's store micro: the
// skip-op discriminator pinned the E4B first-replay NaN to the recorded
// lthn_kv_q8_store_rows_bf16 commands (removing ONLY them cures; identity
// rebinds and every other class stay poisoned). This records one store at the
// live fold's exact layer shapes — staged bf16 rows quantised into int8 cache
// rows + f32 group scales at the batch-base offset — replays it, and compares
// the landed cache+scale bytes against the live encKVQ8StoreRows twin.
func TestVerifyStackICBReplay_KVQ8StoreParity_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const rows = 5
	const basePos = 26
	cases := []struct {
		name  string
		kvDim int
	}{
		{"e4b-owner-hd256", 512},
		{"e4b-owner-hd512", 1024},
		{"e2b-ctrl", 256},
	}
	rng := uint32(0x1234abcd)
	next := func() uint32 { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng }

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stageF := make([]float32, rows*tc.kvDim)
			for i := range stageF {
				stageF[i] = (float32(next()%4000) - 2000) / 250
			}
			stageB := toBF16Bytes(stageF)
			stage := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&stageB[0]), uint(len(stageB)), metal.MTLResourceStorageModeShared)

			cacheLen := (basePos + rows) * tc.kvDim
			scaleLen := (basePos + rows) * (tc.kvDim / kvQ8GroupSize) * 4
			cacheOff := uint(basePos * tc.kvDim)
			scaleOff := uint(basePos * (tc.kvDim / kvQ8GroupSize) * 4)
			mk := func() (metal.MTLBuffer, metal.MTLBuffer) {
				return device.NewBufferWithLengthOptions(uint(cacheLen), metal.MTLResourceStorageModeShared),
					device.NewBufferWithLengthOptions(uint(scaleLen), metal.MTLResourceStorageModeShared)
			}

			// live twin.
			lCache, lScales := mk()
			var liveErr error
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				liveErr = encKVQ8StoreRows(enc, stage, lCache, cacheOff, lScales, scaleOff, rows, tc.kvDim)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})
			if liveErr != nil {
				t.Fatalf("live encKVQ8StoreRows: %v", liveErr)
			}

			// recorded lane.
			rCache, rScales := mk()
			rec := newVerifyStackRecorder(3, rows, verifyStackKey{}, nil)
			if rec == nil {
				t.Fatal("newVerifyStackRecorder returned nil")
			}
			rec.setLayer(1)
			rec.layerEntry()
			rec.recKVQ8StoreRows(stage, rCache, cacheOff, rScales, scaleOff, rows, tc.kvDim)
			rec.markRebindLast(rCache, cacheOff, vsRebindGlobalRow, tc.kvDim, 0, 0)
			rec.markRebindLast(rScales, scaleOff, vsRebindGlobalRow, (tc.kvDim/kvQ8GroupSize)*4, 0, 0)
			if rec.failed {
				t.Fatal("recorder failed to record the store")
			}
			vs := rec.finish()
			if vs == nil {
				t.Fatal("finish returned nil")
			}
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := concurrentComputeEncoderFast(cb)
				vs.executeInto(enc, basePos, nil)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})

			lc := unsafe.Slice((*byte)(lCache.Contents()), cacheLen)
			rc := unsafe.Slice((*byte)(rCache.Contents()), cacheLen)
			ls := unsafe.Slice((*byte)(lScales.Contents()), scaleLen)
			rs := unsafe.Slice((*byte)(rScales.Contents()), scaleLen)
			if !bytes.Equal(lc[cacheOff:], rc[cacheOff:]) {
				diff := -1
				for i := int(cacheOff); i < cacheLen; i++ {
					if lc[i] != rc[i] {
						diff = i
						break
					}
				}
				t.Errorf("%s cache codes diverge live↔replay (first diff at byte %d of row region %d..%d)", tc.name, diff, cacheOff, cacheLen)
			}
			if !bytes.Equal(ls[scaleOff:], rs[scaleOff:]) {
				diff := -1
				for i := int(scaleOff); i < scaleLen; i++ {
					if ls[i] != rs[i] {
						diff = i
						break
					}
				}
				t.Errorf("%s scales diverge live↔replay (first diff at byte %d of region %d..%d)", tc.name, diff, scaleOff, scaleLen)
			}
			var zero [8]byte
			if bytes.Equal(rc[cacheOff:cacheOff+8], zero[:]) && !bytes.Equal(lc[cacheOff:cacheOff+8], zero[:]) {
				t.Errorf("%s replay wrote nothing at the row region (live did)", tc.name)
			}
		})
	}
}
