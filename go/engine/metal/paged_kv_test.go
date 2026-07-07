// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"fmt"
	"math"
	"slices"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
)

func TestPagedKVCacheAttentionMatchesContiguousReference(t *testing.T) {
	requireSDPAPagedKernel(t)

	const nHeads, nKVHeads, headDim = 4, 2, 64
	cache, err := NewPagedKVCache(nKVHeads, headDim, 0, 3)
	if err != nil {
		t.Fatalf("NewPagedKVCache: %v", err)
	}
	defer cache.Close()

	k0 := toBF16Bytes(syntheticFloat32(nKVHeads*2*headDim, 5))
	v0 := toBF16Bytes(syntheticFloat32(nKVHeads*2*headDim, 7))
	if _, err := cache.Update(k0, v0, 2); err != nil {
		t.Fatalf("first Update: %v", err)
	}
	k1 := toBF16Bytes(syntheticFloat32(nKVHeads*4*headDim, 11))
	v1 := toBF16Bytes(syntheticFloat32(nKVHeads*4*headDim, 13))
	state, err := cache.Update(k1, v1, 4)
	if err != nil {
		t.Fatalf("second Update: %v", err)
	}

	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 17))
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	got, err := cache.Attention(q, nHeads, scale)
	if err != nil {
		t.Fatalf("PagedKVCache.Attention: %v", err)
	}
	kFull := compactPagedKVStatePages(state.KeyPages, state.PageLens, nKVHeads, headDim)
	vFull := compactPagedKVStatePages(state.ValuePages, state.PageLens, nKVHeads, headDim)
	want, err := SDPA(q, kFull, vFull, 1, nHeads, nKVHeads, headDim, state.Length, scale)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("paged cache attention cosine = %.6f vs contiguous reference", cos)
	}
}

func TestPagedKVCacheStateBorrowsPinnedPages(t *testing.T) {
	requireNativeRuntime(t)

	const nKVHeads, headDim = 2, 16
	cache, err := NewPagedKVCache(nKVHeads, headDim, 0, 2)
	if err != nil {
		t.Fatalf("NewPagedKVCache: %v", err)
	}
	defer cache.Close()
	k := toBF16Bytes(syntheticFloat32(nKVHeads*2*headDim, 19))
	v := toBF16Bytes(syntheticFloat32(nKVHeads*2*headDim, 23))
	state, err := cache.Update(k, v, 2)
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if len(state.KeyPages) != 1 || len(state.ValuePages) != 1 || len(state.PageLens) != 1 {
		t.Fatalf("state pages = %d/%d lens=%d, want one page", len(state.KeyPages), len(state.ValuePages), len(state.PageLens))
	}
	if got, want := pagedKVBytePtr(state.KeyPages[0]), pagedKVBytePtr(cache.kPages[0].bytes); got != want {
		t.Fatalf("key page backing = %#x, want pinned backing %#x", got, want)
	}
	if got, want := pagedKVBytePtr(state.ValuePages[0]), pagedKVBytePtr(cache.vPages[0].bytes); got != want {
		t.Fatalf("value page backing = %#x, want pinned backing %#x", got, want)
	}
}

func TestPagedKVCacheSlidingWindowTrimsOldestTokens(t *testing.T) {
	requireNativeRuntime(t)

	const nKVHeads, headDim = 2, 8
	cache, err := NewPagedKVCache(nKVHeads, headDim, 4, 3)
	if err != nil {
		t.Fatalf("NewPagedKVCache: %v", err)
	}
	defer cache.Close()
	k := toBF16Bytes(syntheticFloat32(nKVHeads*5*headDim, 29))
	v := toBF16Bytes(syntheticFloat32(nKVHeads*5*headDim, 31))
	state, err := cache.Update(k, v, 5)
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if state.Length != 4 || cache.Len() != 4 || cache.Offset() != 5 {
		t.Fatalf("length/offset = state %d cache %d/%d, want 4/4/5", state.Length, cache.Len(), cache.Offset())
	}
	gotK := compactPagedKVStatePages(state.KeyPages, state.PageLens, nKVHeads, headDim)
	gotV := compactPagedKVStatePages(state.ValuePages, state.PageLens, nKVHeads, headDim)
	wantK := make([]byte, len(gotK))
	wantV := make([]byte, len(gotV))
	copyPagedKVTokens(wantK, 4, 0, k, 5, 1, 4, nKVHeads, headDim)
	copyPagedKVTokens(wantV, 4, 0, v, 5, 1, 4, nKVHeads, headDim)
	if !bytes.Equal(gotK, wantK) {
		t.Fatal("trimmed key pages did not keep the newest window")
	}
	if !bytes.Equal(gotV, wantV) {
		t.Fatal("trimmed value pages did not keep the newest window")
	}
}

func TestDevicePagedKVAttentionHalfMatchesContiguousStep(t *testing.T) {
	requireSDPAPagedKernel(t)

	const dModel, nHeads, nKVHeads, headDim, dFF = 64, 1, 1, 64, 128
	const maxLen, pageSize = 4, 2
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	layer := decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, 3)
	inputs := [][]byte{
		toBF16Bytes(syntheticFloat32(dModel, 101)),
		toBF16Bytes(syntheticFloat32(dModel, 103)),
		toBF16Bytes(syntheticFloat32(dModel, 107)),
	}

	kContig := make([]byte, maxLen*kvDim*bf16Size)
	vContig := make([]byte, maxLen*kvDim*bf16Size)
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, 0, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	proj := bf16Projector{
		wQ: bufView{buf: residentBytes(layer.WQ)}, wK: bufView{buf: residentBytes(layer.WK)}, wV: bufView{buf: residentBytes(layer.WV)}, wO: bufView{buf: residentBytes(layer.WO)},
		dModel: dModel, qDim: qDim, kvDim: kvDim,
	}
	attnNorm := bufView{buf: residentBytes(layer.AttnNormW)}
	outBuf := scratchBF16(dModel)
	sc := getAttnScratch(dModel, qDim, kvDim, nHeads, maxLen)
	defer putAttnScratch(sc)

	for pos, x := range inputs {
		want, err := AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kContig, vContig, dModel, nHeads, nKVHeads, headDim, maxLen, pos, base, scale, eps)
		if err != nil {
			t.Fatalf("AttentionStepKV pos %d: %v", pos, err)
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if enc, err = encAttnHalfKVPaged(enc, cb, nil, residentBytes(x), cache, scalarI32(int32(pos)), outBuf, 0, attnNorm, bufView{}, bufView{}, bufView{}, nil, *sc, proj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); err != nil {
			endEncodingFast(enc)
			t.Fatalf("encAttnHalfKVPaged pos %d: %v", pos, err)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		got := append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), dModel*bf16Size)...)
		eqBytes(t, "paged device attention pos", got, want)
	}
	if got := len(cache.kPages); got != 2 {
		t.Fatalf("device paged cache pages = %d, want 2", got)
	}
	if got := bufferLengthFast(cache.kPages[0]); got != uint(pageSize*kvDim*bf16Size) {
		t.Fatalf("device paged cache first page bytes = %d, want %d", got, pageSize*kvDim*bf16Size)
	}
}

func TestArchDecodeStateDevicePagedKVSlidingOwnerMatchesLinearRing(t *testing.T) {
	requireSDPAPagedKernel(t)

	const dModel, nHeads, nKVHeads, headDim, dFF = 64, 1, 1, 64, 128
	const maxLen, slideW, pageSize = 5, 2, 1
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{
		{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKVHeads},
	}
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, 127)}
	inputs := [][]byte{
		toBF16Bytes(syntheticFloat32(dModel, 1301)),
		toBF16Bytes(syntheticFloat32(dModel, 1303)),
		toBF16Bytes(syntheticFloat32(dModel, 1307)),
		toBF16Bytes(syntheticFloat32(dModel, 1319)),
	}

	var testErr error
	withAutoreleasePool(func() {
		linearLB, linearMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slideW, nil)
		if err != nil {
			testErr = err
			return
		}
		pagedLB, pagedMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slideW, nil)
		if err != nil {
			testErr = err
			return
		}
		linear := newArchDecodeState(specs, linearLB, linearMoE, dModel, nHeads, nKVHeads, headDim, dFF, slideW, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer linear.Close()
		paged := newArchDecodeState(specs, pagedLB, pagedMoE, dModel, nHeads, nKVHeads, headDim, dFF, slideW, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer paged.Close()
		if err := paged.initDevicePagedKV(pageSize); err != nil {
			testErr = err
			return
		}
		if len(paged.pagedKV) != len(specs) || paged.pagedKV[0] == nil {
			testErr = fmt.Errorf("sliding owner layer did not initialise device paged KV")
			return
		}
		for pos, input := range inputs {
			want, err := linear.stepToken(input, pos)
			if err != nil {
				testErr = fmt.Errorf("linear step pos %d: %w", pos, err)
				return
			}
			got, err := paged.stepToken(input, pos)
			if err != nil {
				testErr = fmt.Errorf("paged step pos %d: %w", pos, err)
				return
			}
			if !bytes.Equal(got, want) {
				if cos := cosineBF16(got, want); cos < 0.999 {
					testErr = fmt.Errorf("sliding paged state pos %d cosine = %.6f", pos, cos)
					return
				}
			}
		}
		cache := paged.pagedKV[0]
		if cache.length != slideW || cache.offset != len(inputs) {
			testErr = fmt.Errorf("sliding paged length/offset = %d/%d, want %d/%d", cache.length, cache.offset, slideW, len(inputs))
			return
		}
		if got := append([]int(nil), cache.pageLens...); !slices.Equal(got, []int{1, 1}) {
			testErr = fmt.Errorf("sliding paged page lens = %v, want [1 1]", got)
			return
		}
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func TestArchDecodeStateDevicePagedKVSlidingShareMatchesLinearRing(t *testing.T) {
	requireSDPAPagedKernel(t)

	const dModel, nHeads, nKVHeads, headDim, dFF = 64, 1, 1, 64, 128
	const maxLen, slideW, pageSize = 5, 2, 1
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{
		{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKVHeads},
		{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: headDim, KVHeads: nKVHeads},
	}
	layers := []DecodeLayerWeights{
		decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, 137),
		decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, 139),
	}
	inputs := [][]byte{
		toBF16Bytes(syntheticFloat32(dModel, 1409)),
		toBF16Bytes(syntheticFloat32(dModel, 1423)),
		toBF16Bytes(syntheticFloat32(dModel, 1427)),
		toBF16Bytes(syntheticFloat32(dModel, 1429)),
	}

	var testErr error
	withAutoreleasePool(func() {
		linearLB, linearMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slideW, nil)
		if err != nil {
			testErr = err
			return
		}
		pagedLB, pagedMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slideW, nil)
		if err != nil {
			testErr = err
			return
		}
		linear := newArchDecodeState(specs, linearLB, linearMoE, dModel, nHeads, nKVHeads, headDim, dFF, slideW, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer linear.Close()
		paged := newArchDecodeState(specs, pagedLB, pagedMoE, dModel, nHeads, nKVHeads, headDim, dFF, slideW, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer paged.Close()
		if err := paged.initDevicePagedKV(pageSize); err != nil {
			testErr = err
			return
		}
		if len(paged.pagedKV) != len(specs) || paged.pagedKV[0] == nil || paged.pagedKV[1] != nil {
			testErr = fmt.Errorf("sliding shared topology did not initialise owner-only device paged KV")
			return
		}
		for pos, input := range inputs {
			want, err := linear.stepToken(input, pos)
			if err != nil {
				testErr = fmt.Errorf("linear shared step pos %d: %w", pos, err)
				return
			}
			got, err := paged.stepToken(input, pos)
			if err != nil {
				testErr = fmt.Errorf("paged shared step pos %d: %w", pos, err)
				return
			}
			if !bytes.Equal(got, want) {
				if cos := cosineBF16(got, want); cos < 0.999 {
					testErr = fmt.Errorf("sliding shared paged state pos %d cosine = %.6f", pos, cos)
					return
				}
			}
		}
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func TestDevicePagedKVSharedAttentionMatchesLinearShared(t *testing.T) {
	requireSDPAPagedKernel(t)

	const dModel, nHeads, nKVHeads, headDim, dFF = 128, 2, 1, 64, 256
	const maxLen, pageSize, pos = 4, 2, 2
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	owner := decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, 11)
	shared := decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, 13)
	inputs := [][]byte{
		toBF16Bytes(syntheticFloat32(dModel, 211)),
		toBF16Bytes(syntheticFloat32(dModel, 223)),
		toBF16Bytes(syntheticFloat32(dModel, 227)),
	}

	kContig := make([]byte, maxLen*kvDim*bf16Size)
	vContig := make([]byte, maxLen*kvDim*bf16Size)
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, 0, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	ownerProj := bf16Projector{
		wQ: bufView{buf: residentBytes(owner.WQ)}, wK: bufView{buf: residentBytes(owner.WK)}, wV: bufView{buf: residentBytes(owner.WV)}, wO: bufView{buf: residentBytes(owner.WO)},
		dModel: dModel, qDim: qDim, kvDim: kvDim,
	}
	ownerNorm := bufView{buf: residentBytes(owner.AttnNormW)}
	ownerOut := scratchBF16(dModel)
	ownerScratch := getAttnScratch(dModel, qDim, kvDim, nHeads, maxLen)
	defer putAttnScratch(ownerScratch)
	for i, x := range inputs {
		if _, err := AttentionStepKV(x, owner.AttnNormW, owner.WQ, owner.WK, owner.WV, owner.WO, kContig, vContig, dModel, nHeads, nKVHeads, headDim, maxLen, i, base, scale, eps); err != nil {
			t.Fatalf("AttentionStepKV owner pos %d: %v", i, err)
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if enc, err = encAttnHalfKVPaged(enc, cb, nil, residentBytes(x), cache, scalarI32(int32(i)), ownerOut, 0, ownerNorm, bufView{}, bufView{}, bufView{}, nil, *ownerScratch, ownerProj, dModel, nHeads, nKVHeads, headDim, i, 0, headDim, base, scale, eps, nil); err != nil {
			endEncodingFast(enc)
			t.Fatalf("encAttnHalfKVPaged owner pos %d: %v", i, err)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	}

	sharedX := toBF16Bytes(syntheticFloat32(dModel, 229))
	sharedProj := bf16Projector{
		wQ: bufView{buf: residentBytes(shared.WQ)}, wO: bufView{buf: residentBytes(shared.WO)},
		dModel: dModel, qDim: qDim, kvDim: kvDim,
	}
	sharedNorm := bufView{buf: residentBytes(shared.AttnNormW)}
	sharedScratch := getAttnScratch(dModel, qDim, kvDim, nHeads, maxLen)
	defer putAttnScratch(sharedScratch)
	wantBuf := scratchBF16(dModel)
	gotBuf := scratchBF16(dModel)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encAttnHalfShared(enc, residentBytes(sharedX), residentBytes(kContig), residentBytes(vContig), scalarI32(int32(pos)), wantBuf, sharedNorm, bufView{}, bufView{}, *sharedScratch, sharedProj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encAttnHalfShared reference: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	want := append([]byte(nil), unsafe.Slice((*byte)(wantBuf.Contents()), dModel*bf16Size)...)

	cb = commandBufferFast(queue)
	enc = computeCommandEncoderFast(cb)
	if err := encAttnHalfSharedPaged(enc, residentBytes(sharedX), cache, scalarI32(int32(pos)), gotBuf, 0, sharedNorm, bufView{}, bufView{}, *sharedScratch, sharedProj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encAttnHalfSharedPaged: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := append([]byte(nil), unsafe.Slice((*byte)(gotBuf.Contents()), dModel*bf16Size)...)
	eqBytes(t, "paged shared attention", got, want)
}

func TestDevicePagedKVCacheLinearSnapshotRoundTrip(t *testing.T) {
	requireNativeRuntime(t)

	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 4, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	wantK := make([]byte, maxLen*kvBytes)
	wantV := make([]byte, maxLen*kvBytes)
	for pos := range 3 {
		kRow := toBF16Bytes(syntheticFloat32(nKVHeads*headDim, 503+pos))
		vRow := toBF16Bytes(syntheticFloat32(nKVHeads*headDim, 607+pos))
		copy(wantK[pos*kvBytes:(pos+1)*kvBytes], kRow)
		copy(wantV[pos*kvBytes:(pos+1)*kvBytes], vRow)
		kPage, vPage, rowOff, err := cache.slot(pos)
		if err != nil {
			t.Fatalf("slot %d: %v", pos, err)
		}
		copy(unsafe.Slice((*byte)(unsafe.Add(kPage.Contents(), uintptr(rowOff))), kvBytes), kRow)
		copy(unsafe.Slice((*byte)(unsafe.Add(vPage.Contents(), uintptr(rowOff))), kvBytes), vRow)
	}

	kBuf, vBuf, kPtr, vPtr, err := cache.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("linearSnapshot: %v", err)
	}
	if got, want := bufferLengthFast(kBuf), uint(maxLen*kvBytes); got != want {
		t.Fatalf("linear K snapshot bytes = %d, want %d", got, want)
	}
	if got, want := bufferLengthFast(vBuf), uint(maxLen*kvBytes); got != want {
		t.Fatalf("linear V snapshot bytes = %d, want %d", got, want)
	}
	kLinear := append([]byte(nil), unsafe.Slice(kPtr, maxLen*kvBytes)...)
	vLinear := append([]byte(nil), unsafe.Slice(vPtr, maxLen*kvBytes)...)
	eqBytes(t, "device page linear K snapshot", kLinear, wantK)
	eqBytes(t, "device page linear V snapshot", vLinear, wantV)

	restored, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache restored: %v", err)
	}
	defer restored.Close()
	if err := restored.loadLinearSnapshot(kLinear, vLinear, 3); err != nil {
		t.Fatalf("loadLinearSnapshot: %v", err)
	}
	_, _, rkPtr, rvPtr, err := restored.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("restored linearSnapshot: %v", err)
	}
	eqBytes(t, "restored device page K snapshot", unsafe.Slice(rkPtr, maxLen*kvBytes), wantK)
	eqBytes(t, "restored device page V snapshot", unsafe.Slice(rvPtr, maxLen*kvBytes), wantV)
	if got := len(restored.kPages); got != 2 {
		t.Fatalf("restored page count = %d, want 2", got)
	}
}

func TestDevicePagedKVCacheTruncateShrinksVisiblePages(t *testing.T) {
	requireNativeRuntime(t)

	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 5, 2
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	for pos := range maxLen {
		if _, _, _, err := cache.slot(pos); err != nil {
			t.Fatalf("slot %d: %v", pos, err)
		}
	}
	if got := append([]int(nil), cache.pageLens...); !slices.Equal(got, []int{2, 2, 1}) {
		t.Fatalf("initial page lens = %v, want [2 2 1]", got)
	}

	if err := cache.truncate(3); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	if got := cache.length; got != 3 {
		t.Fatalf("length after truncate = %d, want 3", got)
	}
	if got := cache.offset; got != 3 {
		t.Fatalf("offset after truncate = %d, want 3", got)
	}
	if got := append([]int(nil), cache.pageLens...); !slices.Equal(got, []int{2, 1, 0}) {
		t.Fatalf("page lens after truncate = %v, want [2 1 0]", got)
	}
}

func TestDevicePagedKVCacheTracksLinearSyncBoundary(t *testing.T) {
	requireNativeRuntime(t)

	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 5, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()

	kRows := toBF16Bytes(syntheticFloat32(3*nKVHeads*headDim, 701))
	vRows := toBF16Bytes(syntheticFloat32(3*nKVHeads*headDim, 709))
	if len(kRows) != 3*kvBytes || len(vRows) != 3*kvBytes {
		t.Fatalf("fixture rows = %d/%d, want %d", len(kRows), len(vRows), 3*kvBytes)
	}
	if err := cache.loadLinearSnapshot(kRows, vRows, 3); err != nil {
		t.Fatalf("loadLinearSnapshot: %v", err)
	}
	if got := cache.linearSynced; got != 3 {
		t.Fatalf("linear synced after load = %d, want 3", got)
	}
	if _, _, _, err := cache.slot(3); err != nil {
		t.Fatalf("slot append: %v", err)
	}
	if got := cache.linearSynced; got != 3 {
		t.Fatalf("linear synced after append slot = %d, want 3", got)
	}
	if err := cache.truncate(2); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	if got := cache.linearSynced; got != 2 {
		t.Fatalf("linear synced after truncate = %d, want 2", got)
	}
	if _, _, _, err := cache.slot(1); err != nil {
		t.Fatalf("slot overwrite: %v", err)
	}
	if got := cache.linearSynced; got != 1 {
		t.Fatalf("linear synced after overwrite slot = %d, want 1", got)
	}
}

func compactPagedKVStatePages(pages [][]byte, lens []int, nKVHeads, headDim int) []byte {
	total := 0
	for _, n := range lens {
		total += n
	}
	out := make([]byte, nKVHeads*total*headDim*bf16Size)
	headBytes := headDim * bf16Size
	for h := range nKVHeads {
		pos := 0
		for i, page := range pages {
			pageLen := lens[i]
			pageSpan := len(page) / (nKVHeads * headBytes)
			src := (h * pageSpan) * headBytes
			dst := (h*total + pos) * headBytes
			copy(out[dst:dst+pageLen*headBytes], page[src:src+pageLen*headBytes])
			pos += pageLen
		}
	}
	return out
}

func pagedKVBytePtr(b []byte) uintptr {
	if len(b) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&b[0]))
}
