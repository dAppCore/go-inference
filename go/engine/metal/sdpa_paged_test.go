// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func concatHeadMajorKVPages(pages [][]byte, nKVHeads, headDim int) ([]byte, int) {
	total := 0
	for _, page := range pages {
		total += len(page) / (nKVHeads * headDim * bf16Size)
	}
	out := make([]byte, nKVHeads*total*headDim*bf16Size)
	for h := 0; h < nKVHeads; h++ {
		pos := 0
		for _, page := range pages {
			pageLen := len(page) / (nKVHeads * headDim * bf16Size)
			src := (h * pageLen * headDim) * bf16Size
			dst := (h*total*headDim + pos*headDim) * bf16Size
			copy(out[dst:dst+pageLen*headDim*bf16Size], page[src:src+pageLen*headDim*bf16Size])
			pos += pageLen
		}
	}
	return out, total
}

func requireSDPAPagedKernel(t *testing.T) {
	t.Helper()
	requireNativeRuntime(t)
	if customLibrary == nil || customLibrary.GetID() == 0 {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}
	if fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p1_bf16"); fn == nil || fn.GetID() == 0 {
		t.Skip("custom paged SDPA pass-1 kernel not loaded - run `task build:kernels`")
	}
	if fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_p2_bf16"); fn == nil || fn.GetID() == 0 {
		t.Skip("custom paged SDPA pass-2 kernel not loaded - run `task build:kernels`")
	}
}

func TestSDPAPagedBF16MatchesContiguousReference(t *testing.T) {
	requireSDPAPagedKernel(t)

	const nHeads, nKVHeads, headDim = 4, 2, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 3))
	kPages := [][]byte{
		toBF16Bytes(syntheticFloat32(nKVHeads*3*headDim, 5)),
		toBF16Bytes(syntheticFloat32(nKVHeads*5*headDim, 7)),
	}
	vPages := [][]byte{
		toBF16Bytes(syntheticFloat32(nKVHeads*3*headDim, 11)),
		toBF16Bytes(syntheticFloat32(nKVHeads*5*headDim, 13)),
	}
	kFull, kvLen := concatHeadMajorKVPages(kPages, nKVHeads, headDim)
	vFull, _ := concatHeadMajorKVPages(vPages, nKVHeads, headDim)

	got, err := SDPAPagedBF16(q, kPages, vPages, nHeads, nKVHeads, headDim, scale)
	if err != nil {
		t.Fatalf("SDPAPagedBF16: %v", err)
	}
	want, err := SDPA(q, kFull, vFull, 1, nHeads, nKVHeads, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("paged SDPA cosine = %.6f vs contiguous reference", cos)
	}
}

func TestSDPAPagedBF16IntoUsesCallerBacking(t *testing.T) {
	requireSDPAPagedKernel(t)

	const nHeads, nKVHeads, headDim = 4, 2, 64
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 17))
	kPages := [][]byte{
		toBF16Bytes(syntheticFloat32(nKVHeads*2*headDim, 19)),
		toBF16Bytes(syntheticFloat32(nKVHeads*4*headDim, 23)),
	}
	vPages := [][]byte{
		toBF16Bytes(syntheticFloat32(nKVHeads*2*headDim, 29)),
		toBF16Bytes(syntheticFloat32(nKVHeads*4*headDim, 31)),
	}
	out := make([]byte, nHeads*headDim*bf16Size)
	got, err := SDPAPagedBF16Into(out, q, kPages, vPages, nHeads, nKVHeads, headDim, 0.125)
	if err != nil {
		t.Fatalf("SDPAPagedBF16Into: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("SDPAPagedBF16Into len = %d, want %d", len(got), len(out))
	}
	if len(got) > 0 && uintptr(unsafe.Pointer(&got[0])) != uintptr(unsafe.Pointer(&out[0])) {
		t.Fatal("SDPAPagedBF16Into did not return caller-owned output backing")
	}
	if bytes.Equal(out, make([]byte, len(out))) {
		t.Fatal("SDPAPagedBF16Into left caller output untouched")
	}
}

func TestEncSDPAPagedDecodeBuffersRespectVisiblePageLens(t *testing.T) {
	requireSDPAPagedKernel(t)

	const nHeads, nKVHeads, headDim = 4, 2, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 37))
	pageLens := []int{2, 3}
	pageSpans := []int{4, 4}
	kPages := [][]byte{
		toBF16Bytes(syntheticFloat32(nKVHeads*pageSpans[0]*headDim, 41)),
		toBF16Bytes(syntheticFloat32(nKVHeads*pageSpans[1]*headDim, 43)),
	}
	vPages := [][]byte{
		toBF16Bytes(syntheticFloat32(nKVHeads*pageSpans[0]*headDim, 47)),
		toBF16Bytes(syntheticFloat32(nKVHeads*pageSpans[1]*headDim, 53)),
	}

	kFull := compactPagedKVStatePages(kPages, pageLens, nKVHeads, headDim)
	vFull := compactPagedKVStatePages(vPages, pageLens, nKVHeads, headDim)
	want, err := SDPA(q, kFull, vFull, 1, nHeads, nKVHeads, headDim, pageLens[0]+pageLens[1], scale)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}

	qBuf := residentBytes(q)
	keyBufs := make([]metal.MTLBuffer, len(kPages))
	valueBufs := make([]metal.MTLBuffer, len(vPages))
	for i := range kPages {
		keyBufs[i] = residentBytes(kPages[i])
		valueBufs[i] = residentBytes(vPages[i])
	}
	outBuf := scratchBF16(nHeads * headDim)
	scratch, err := newSDPAPagedDecodeScratch(nHeads, headDim)
	if err != nil {
		t.Fatalf("newSDPAPagedDecodeScratch: %v", err)
	}

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPAPagedDecode(enc, qBuf, keyBufs, valueBufs, pageLens, pageSpans, outBuf, scratch, nHeads, nKVHeads, headDim, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPAPagedDecode: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	got := append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), nHeads*headDim*bf16Size)...)
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("buffer paged SDPA cosine = %.6f vs contiguous reference", cos)
	}
}

func TestEncSDPAPagedDecodeStridedAcceptsSeqMajorPages(t *testing.T) {
	requireSDPAPagedKernel(t)

	const nHeads, nKVHeads, headDim = 4, 2, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 59))
	pageLens := []int{2, 3}
	pageSpans := []int{4, 4}
	kPages := [][]byte{
		toBF16Bytes(syntheticFloat32(pageSpans[0]*nKVHeads*headDim, 61)),
		toBF16Bytes(syntheticFloat32(pageSpans[1]*nKVHeads*headDim, 67)),
	}
	vPages := [][]byte{
		toBF16Bytes(syntheticFloat32(pageSpans[0]*nKVHeads*headDim, 71)),
		toBF16Bytes(syntheticFloat32(pageSpans[1]*nKVHeads*headDim, 73)),
	}

	kFull := compactSeqMajorPagesToHeadMajor(kPages, pageLens, nKVHeads, headDim)
	vFull := compactSeqMajorPagesToHeadMajor(vPages, pageLens, nKVHeads, headDim)
	want, err := SDPA(q, kFull, vFull, 1, nHeads, nKVHeads, headDim, pageLens[0]+pageLens[1], scale)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}

	qBuf := residentBytes(q)
	keyBufs := make([]metal.MTLBuffer, len(kPages))
	valueBufs := make([]metal.MTLBuffer, len(vPages))
	keyHeadStrides := make([]int, len(kPages))
	keySeqStrides := make([]int, len(kPages))
	valueHeadStrides := make([]int, len(vPages))
	valueSeqStrides := make([]int, len(vPages))
	for i := range kPages {
		keyBufs[i] = residentBytes(kPages[i])
		valueBufs[i] = residentBytes(vPages[i])
		keyHeadStrides[i] = headDim
		keySeqStrides[i] = nKVHeads * headDim
		valueHeadStrides[i] = headDim
		valueSeqStrides[i] = nKVHeads * headDim
	}
	outBuf := scratchBF16(nHeads * headDim)
	scratch, err := newSDPAPagedDecodeScratch(nHeads, headDim)
	if err != nil {
		t.Fatalf("newSDPAPagedDecodeScratch: %v", err)
	}

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPAPagedDecodeStrided(enc, qBuf, keyBufs, valueBufs, pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides, outBuf, scratch, nHeads, nKVHeads, headDim, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPAPagedDecodeStrided: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	got := append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), nHeads*headDim*bf16Size)...)
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("strided buffer paged SDPA cosine = %.6f vs contiguous reference", cos)
	}
}

func compactSeqMajorPagesToHeadMajor(pages [][]byte, lens []int, nKVHeads, headDim int) []byte {
	total := 0
	for _, n := range lens {
		total += n
	}
	out := make([]byte, nKVHeads*total*headDim*bf16Size)
	headBytes := headDim * bf16Size
	for h := 0; h < nKVHeads; h++ {
		pos := 0
		for i, page := range pages {
			for t := 0; t < lens[i]; t++ {
				src := (t*nKVHeads + h) * headBytes
				dst := (h*total + pos + t) * headBytes
				copy(out[dst:dst+headBytes], page[src:src+headBytes])
			}
			pos += lens[i]
		}
	}
	return out
}
