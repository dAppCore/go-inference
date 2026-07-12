// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// per_layer_batch_slab_test.go gates the K-token PLE slab BUILDERS
// (perLayerInputsBatchIntoSlab + perLayerInputsBatchQuantIntoSlab, driving
// perLayerInputsBatchQuantEncode) — the one-command-buffer batched twins of the
// per-token PerLayerInputs chain. The reference is the already-trusted
// per-token perLayerProjBatched run for each token over the SAME gathered PLE
// rows, relaid out to layer-major on the host: the slab must match it within
// the bf16 tolerance the GEMM-vs-GEMV projection difference allows.

// pleGatherScaledRow reproduces lthn_ple_gather_rows_bf16 on the host: row id of
// the bf16 PLE table, each element scaled by embScale (= √pliDim) back to bf16.
func pleGatherScaledRow(table []byte, id, plDim int, embScale float32) []byte {
	out := make([]byte, plDim*bf16Size)
	base := id * plDim
	for c := range plDim {
		v := bf16ToF32(table[(base+c)*bf16Size], table[(base+c)*bf16Size+1]) * embScale
		h := f32ToBF16(v)
		out[2*c], out[2*c+1] = byte(h), byte(h>>8)
	}
	return out
}

// pleSlabPerTokenReference builds the expected layer-major slab: gather+scale
// each token's PLE row, run the trusted per-token projection chain, then apply
// the relayout permutation slab[(li·K+i)·pliDim+d] = tokenMajor_i[li·pliDim+d].
func pleSlabPerTokenReference(t *testing.T, projFn func(hidden, perLayer []byte) []byte, table []byte, ids []int32, embs [][]byte, numLayers, pliDim int, embScale float32) []byte {
	t.Helper()
	k := len(ids)
	plDim := numLayers * pliDim
	slab := make([]byte, k*plDim*bf16Size)
	for i, id := range ids {
		perLayer := pleGatherScaledRow(table, int(id), plDim, embScale)
		tokenMajor := projFn(embs[i], perLayer)
		for li := range numLayers {
			for d := range pliDim {
				src := (li*pliDim + d) * bf16Size
				dst := ((li*k+i)*pliDim + d) * bf16Size
				slab[dst], slab[dst+1] = tokenMajor[src], tokenMajor[src+1]
			}
		}
	}
	return slab
}

func pleSlabMaxDiff(got, want []byte) float64 {
	var maxDiff float64
	for i := 0; i+1 < len(want); i += bf16Size {
		g := float64(bf16ToF32(got[i], got[i+1]))
		w := float64(bf16ToF32(want[i], want[i+1]))
		if d := math.Abs(g - w); d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

func TestPerLayerInputsBatchIntoSlabMatchesPerTokenReference(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := pleGatherRowsPipeline(); err != nil {
		t.Skip("ple gather-rows kernel not loaded")
	}
	const numLayers, pliDim, dModel, vocabPLI = 2, 32, 256, 64
	const eps = float32(1e-5)
	const k = steelGEMMMinRows
	plDim := numLayers * pliDim
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1 / math.Sqrt(float64(dModel)))

	table := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 7))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	projView := copyView(projW)
	ids := make([]int32, k)
	embs := make([][]byte, k)
	for i := range ids {
		ids[i] = int32((i*11 + 5) % vocabPLI)
		embs[i] = toBF16Bytes(syntheticFloat32(dModel, i+1))
	}

	sc := &pleBatchScratch{}
	slab := make([]byte, k*plDim*bf16Size)
	engaged, err := perLayerInputsBatchIntoSlab(sc, table, projView, projNormW, ids, embs, slab, vocabPLI, numLayers, pliDim, dModel, eps)
	if err != nil {
		t.Fatalf("perLayerInputsBatchIntoSlab: %v", err)
	}
	if !engaged {
		t.Skip("batched bf16 slab path declined (steel/kernels unavailable)")
	}

	projFn := func(hidden, perLayer []byte) []byte {
		out, perr := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
		if perr != nil {
			t.Fatalf("per-token reference perLayerProjBatched: %v", perr)
		}
		return out
	}
	want := pleSlabPerTokenReference(t, projFn, table, ids, embs, numLayers, pliDim, embScale)
	if maxDiff := pleSlabMaxDiff(slab, want); maxDiff > 0.05 {
		t.Fatalf("batched slab vs per-token reference maxDiff = %.5f (> 0.05)", maxDiff)
	}
}

func TestPerLayerInputsBatchQuantIntoSlabMatchesPerTokenReference(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := pleGatherRowsQuantPipeline(); err != nil {
		t.Skip("ple gather-rows-quant kernel not loaded")
	}
	const numLayers, pliDim, dModel, vocabPLI = 2, 32, 256, 64
	const tableGS, tableBits = 64, 4
	const projGS, projBits = 64, 4
	const eps = float32(1e-5)
	const k = steelGEMMMinRows
	plDim := numLayers * pliDim
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1 / math.Sqrt(float64(dModel)))

	// quant PLE table [vocabPLI × plDim] and a quant projection [plDim × dModel];
	// the host reference dequantises both through the package's own unpacker.
	tPacked, tScales, tBiases := packAffineQuant(syntheticFloat32(vocabPLI*plDim, 9), vocabPLI, plDim, tableGS, tableBits)
	projQ := quantWeightFixture(t, plDim, dModel, projGS, projBits, 13)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	ids := make([]int32, k)
	embs := make([][]byte, k)
	for i := range ids {
		ids[i] = int32((i*11 + 5) % vocabPLI)
		embs[i] = toBF16Bytes(syntheticFloat32(dModel, i+1))
	}

	sc := &pleBatchScratch{}
	slab := make([]byte, k*plDim*bf16Size)
	engaged, err := perLayerInputsBatchQuantIntoSlab(sc,
		residentBytes(tPacked), residentBytes(tScales), residentBytes(tBiases), tableGS, tableBits,
		nil, 0, residentBytes(projQ.Packed), residentBytes(projQ.Scales), residentBytes(projQ.Biases), projGS, projBits,
		projNormW, ids, embs, slab, numLayers, pliDim, dModel, eps)
	if err != nil {
		t.Fatalf("perLayerInputsBatchQuantIntoSlab: %v", err)
	}
	if !engaged {
		t.Skip("batched quant slab path declined (qmm_t instantiation unavailable)")
	}

	// host reference: dequantise the PLE table to bf16, then reuse the quant
	// per-token projection (perLayerProjQuantBatched) over the gathered rows.
	tableBF16, derr := dequantizeAffineRowsF32(tPacked, tScales, tBiases, vocabPLI, plDim, tableGS, tableBits)
	if derr != nil {
		t.Fatalf("dequantize PLE table: %v", derr)
	}
	table := toBF16Bytes(tableBF16)
	projFn := func(hidden, perLayer []byte) []byte {
		out, perr := perLayerProjQuantBatched(projQ, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, projGS, projBits, eps)
		if perr != nil {
			t.Fatalf("per-token reference perLayerProjQuantBatched: %v", perr)
		}
		return out
	}
	want := pleSlabPerTokenReference(t, projFn, table, ids, embs, numLayers, pliDim, embScale)
	if maxDiff := pleSlabMaxDiff(slab, want); maxDiff > 0.08 {
		t.Fatalf("batched quant slab vs per-token reference maxDiff = %.5f (> 0.08)", maxDiff)
	}
}

func TestPerLayerInputsBatchIntoSlabDeclinesAndGuards(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel, vocabPLI = 2, 8, 16, 32
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	table := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 7))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	mkIDs := func(k int) ([]int32, [][]byte) {
		ids := make([]int32, k)
		embs := make([][]byte, k)
		for i := range ids {
			ids[i] = int32(i % vocabPLI)
			embs[i] = toBF16Bytes(syntheticFloat32(dModel, i+1))
		}
		return ids, embs
	}

	// Below the steel floor: no work, no error (caller keeps the per-token loop).
	ids, embs := mkIDs(steelGEMMMinRows - 1)
	slab := make([]byte, len(ids)*plDim*bf16Size)
	if engaged, err := perLayerInputsBatchIntoSlab(&pleBatchScratch{}, table, copyView(projW), projNormW, ids, embs, slab, vocabPLI, numLayers, pliDim, dModel, eps); engaged || err != nil {
		t.Fatalf("below-floor: want decline, got engaged=%v err=%v", engaged, err)
	}

	// nil projection view: decline.
	ids, embs = mkIDs(steelGEMMMinRows)
	slab = make([]byte, len(ids)*plDim*bf16Size)
	if engaged, err := perLayerInputsBatchIntoSlab(&pleBatchScratch{}, table, bufView{}, projNormW, ids, embs, slab, vocabPLI, numLayers, pliDim, dModel, eps); engaged || err != nil {
		t.Fatalf("nil projView: want decline, got engaged=%v err=%v", engaged, err)
	}

	// Malformed slab length: hard error via pleSlabOutLayers.
	if engaged, err := perLayerInputsBatchIntoSlab(&pleBatchScratch{}, table, copyView(projW), projNormW, ids, embs, make([]byte, 7), vocabPLI, numLayers, pliDim, dModel, eps); engaged || err == nil {
		t.Fatalf("bad slab length: want error, got engaged=%v err=%v", engaged, err)
	}

	// Hidden row of the wrong size: hard error.
	badEmbs := make([][]byte, len(ids))
	copy(badEmbs, embs)
	badEmbs[0] = badEmbs[0][:dModel*bf16Size-2]
	slab = make([]byte, len(ids)*plDim*bf16Size)
	if _, err := perLayerInputsBatchIntoSlab(&pleBatchScratch{}, table, copyView(projW), projNormW, ids, badEmbs, slab, vocabPLI, numLayers, pliDim, dModel, eps); err == nil {
		t.Fatal("bad hidden row size: want error, got nil")
	}
}

func TestPleSlabOutLayersBounds(t *testing.T) {
	// pleSlabOutLayers derives the slab's layer count from its byte length.
	const k, pliDim, numLayers = 4, 8, 3
	rowBytes := k * pliDim * bf16Size
	for _, tc := range []struct {
		name    string
		slabLen int
		want    int
		wantErr bool
	}{
		{"full", rowBytes * numLayers, numLayers, false},
		{"bounded", rowBytes * 1, 1, false},
		{"notMultiple", rowBytes + 3, 0, true},
		{"tooMany", rowBytes * (numLayers + 1), 0, true},
		{"zeroK", 0, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kk := k
			if tc.name == "zeroK" {
				kk = 0
			}
			got, err := pleSlabOutLayers(tc.slabLen, kk, pliDim, numLayers)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("%s: want error, got nOut=%d", tc.name, got)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("%s: nOut=%d err=%v, want %d", tc.name, got, err, tc.want)
			}
		})
	}
}
