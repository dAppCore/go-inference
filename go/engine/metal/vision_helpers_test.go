// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

func TestVisionGridForPatchCount(t *testing.T) {
	tests := []struct {
		patches, pool int
		wantH, wantW  int
	}{
		{patches: 0, pool: 2, wantH: 0, wantW: 0},
		{patches: 12, pool: 1, wantH: 3, wantW: 4},
		{patches: 16, pool: 2, wantH: 4, wantW: 4},
		{patches: 18, pool: 2, wantH: 1, wantW: 18},
	}
	for _, tt := range tests {
		gotH, gotW := visionGridForPatchCount(tt.patches, tt.pool)
		if gotH != tt.wantH || gotW != tt.wantW {
			t.Fatalf("visionGridForPatchCount(%d, %d) = (%d, %d), want (%d, %d)", tt.patches, tt.pool, gotH, gotW, tt.wantH, tt.wantW)
		}
	}
}

func TestVisionPoolerBranches(t *testing.T) {
	hidden := toBF16Bytes([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	got := bf16Floats(visionPooler(hidden, 2, 2, 2, 2, 2))
	want := []float32{8, 10}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("grid pool value %d = %v, want %v", i, got[i], want[i])
		}
	}

	group := bf16Floats(visionPooler(hidden, 1, 4, 2, 2, 1))
	wantGroup := []float32{4, 5}
	for i := range wantGroup {
		if group[i] != wantGroup[i] {
			t.Fatalf("group pool value %d = %v, want %v", i, group[i], wantGroup[i])
		}
	}

	pass := bf16Floats(visionPooler(toBF16Bytes([]float32{1, 2, 3, 4, 5, 6}), 3, 1, 2, 2, 1))
	wantPass := []float32{1, 2, 3, 4, 5, 6}
	for i := range wantPass {
		if pass[i] != wantPass[i] {
			t.Fatalf("pass pool value %d = %v, want %v", i, pass[i], wantPass[i])
		}
	}
}

func TestVisionPositionEmbeddingsSplitXYGood(t *testing.T) {
	const hidden, gridH, gridW, slots = 2, 2, 3, 3
	table := toBF16Bytes([]float32{
		// x table rows.
		10, 100,
		20, 200,
		30, 300,
		// y table rows.
		1, 2,
		3, 4,
		5, 6,
	})

	got, err := visionPositionEmbeddings(table, gridH*gridW, hidden, gridH, gridW, slots)
	if err != nil {
		t.Fatalf("visionPositionEmbeddings(split): %v", err)
	}
	values := bf16Floats(got)
	want := []float32{
		11, 102,
		21, 202,
		31, 302,
		13, 104,
		23, 204,
		33, 304,
	}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("split position value %d = %v, want %v", i, values[i], want[i])
		}
	}
}

func TestVisionPositionEmbeddingsSplitXYAllocationBudget(t *testing.T) {
	const hidden, gridH, gridW, slots = 64, 12, 10, 16
	table := toBF16Bytes(syntheticFloat32(2*slots*hidden, 71))
	got, err := visionPositionEmbeddings(table, gridH*gridW, hidden, gridH, gridW, slots)
	if err != nil {
		t.Fatalf("visionPositionEmbeddings(split warmup): %v", err)
	}
	if len(got) != gridH*gridW*hidden*bf16Size {
		t.Fatalf("split position embedding bytes = %d, want %d", len(got), gridH*gridW*hidden*bf16Size)
	}
	var embedErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, embedErr = visionPositionEmbeddings(table, gridH*gridW, hidden, gridH, gridW, slots)
	})
	if embedErr != nil {
		t.Fatalf("visionPositionEmbeddings(split): %v", embedErr)
	}
	if allocs > 1 {
		t.Fatalf("split position embedding allocations = %.0f, want <= 1", allocs)
	}
}

func BenchmarkVisionPositionEmbeddingsSplitXY(b *testing.B) {
	const hidden, gridH, gridW, slots = 768, 24, 18, 32
	table := toBF16Bytes(syntheticFloat32(2*slots*hidden, 73))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, err := visionPositionEmbeddings(table, gridH*gridW, hidden, gridH, gridW, slots)
		if err != nil {
			b.Fatalf("visionPositionEmbeddings(split): %v", err)
		}
		if len(got) != gridH*gridW*hidden*bf16Size {
			b.Fatalf("split position embedding bytes = %d, want %d", len(got), gridH*gridW*hidden*bf16Size)
		}
	}
}

func TestVisionPatchConvEmbedNHWCGood(t *testing.T) {
	const height, width, channels, hidden, patch = 4, 4, 1, 2, 2
	pixels := []float32{
		1.0, 0.5, 0.25, 0.75,
		0.0, 0.25, 1.0, 0.5,
		0.5, 1.0, 0.0, 0.25,
		0.75, 0.5, 0.25, 1.0,
	}
	conv := toBF16Bytes([]float32{
		// hidden row 0: sum the scaled 2x2 patch.
		1, 1,
		1, 1,
		// hidden row 1: read the top-left scaled pixel.
		1, 0,
		0, 0,
	})

	got, gridH, gridW, err := visionPatchConvEmbedNHWC(pixels, conv, height, width, channels, hidden, patch)
	if err != nil {
		t.Fatalf("visionPatchConvEmbedNHWC: %v", err)
	}
	if gridH != 2 || gridW != 2 {
		t.Fatalf("grid = %dx%d, want 2x2", gridH, gridW)
	}
	values := bf16Floats(got)
	want := []float32{
		-0.5, 1.0,
		1.0, -0.5,
		1.5, 0.0,
		-1.0, -1.0,
	}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("raw conv value %d = %v, want %v", i, values[i], want[i])
		}
	}
}

func TestVisionPatchEmbedNHWCAddsPositionEmbeddings(t *testing.T) {
	pixels := []float32{
		1.0, 0.5,
		0.0, 0.25,
	}
	weights := &VisionWeights{
		PatchConvWeight: toBF16Bytes([]float32{
			1, 1,
			1, 1,
			1, 0,
			0, 0,
		}),
		PositionEmbeddings: toBF16Bytes([]float32{1.0, 2.0}),
	}
	got, gridH, gridW, err := VisionPatchEmbedNHWC(pixels, 2, 2, weights, VisionConfig{
		Hidden: 2, PatchDim: 4, PatchSize: 2, NumChannels: 1, PositionEmbeddingSize: 1,
	})
	if err != nil {
		t.Fatalf("VisionPatchEmbedNHWC: %v", err)
	}
	if gridH != 1 || gridW != 1 {
		t.Fatalf("grid = %dx%d, want 1x1", gridH, gridW)
	}
	values := bf16Floats(got)
	want := []float32{0.5, 3.0}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("raw patch embedding value %d = %v, want %v", i, values[i], want[i])
		}
	}
}

func TestVisionPatchConvEmbedNHWCAllocationBudget(t *testing.T) {
	const height, width, channels, hidden, patch = 64, 64, 3, 64, 16
	pixels := syntheticFloat32(height*width*channels, 75)
	conv := toBF16Bytes(syntheticFloat32(hidden*patch*patch*channels, 77))
	got, gridH, gridW, err := visionPatchConvEmbedNHWC(pixels, conv, height, width, channels, hidden, patch)
	if err != nil {
		t.Fatalf("visionPatchConvEmbedNHWC warmup: %v", err)
	}
	if gridH != 4 || gridW != 4 || len(got) != gridH*gridW*hidden*bf16Size {
		t.Fatalf("raw conv output = grid %dx%d bytes %d", gridH, gridW, len(got))
	}
	var convErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, _, _, convErr = visionPatchConvEmbedNHWC(pixels, conv, height, width, channels, hidden, patch)
	})
	if convErr != nil {
		t.Fatalf("visionPatchConvEmbedNHWC: %v", convErr)
	}
	if allocs > 1 {
		t.Fatalf("raw conv patch embed allocations = %.0f, want <= 1", allocs)
	}
}

func BenchmarkVisionPatchConvEmbedNHWC(b *testing.B) {
	const height, width, channels, hidden, patch = 64, 64, 3, 64, 16
	pixels := syntheticFloat32(height*width*channels, 79)
	conv := toBF16Bytes(syntheticFloat32(hidden*patch*patch*channels, 81))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, gridH, gridW, err := visionPatchConvEmbedNHWC(pixels, conv, height, width, channels, hidden, patch)
		if err != nil {
			b.Fatalf("visionPatchConvEmbedNHWC: %v", err)
		}
		if gridH != 4 || gridW != 4 || len(got) != gridH*gridW*hidden*bf16Size {
			b.Fatalf("raw conv output = grid %dx%d bytes %d", gridH, gridW, len(got))
		}
	}
}

func TestVisionStandardize(t *testing.T) {
	pooled := toBF16Bytes([]float32{2, 4, 6, 8})
	if got := visionStandardize(pooled, nil, nil, 2); &got[0] != &pooled[0] {
		t.Fatal("visionStandardize without weights should return the original slice")
	}
	got := bf16Floats(visionStandardize(pooled, toBF16Bytes([]float32{1, 2}), toBF16Bytes([]float32{3, 4}), 2))
	want := []float32{3, 8, 15, 24}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("standardized value %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestVisionProjectorNoProjectionNormalisesRows(t *testing.T) {
	rows := toBF16Bytes([]float32{3, 4, 1, 2})
	got, err := visionProjector(rows, &VisionProjectorWeights{Eps: 0}, 2)
	if err != nil {
		t.Fatalf("visionProjector: %v", err)
	}
	values := bf16Floats(got)
	want := []float32{
		3 / 3.5355339,
		4 / 3.5355339,
		1 / 1.5811388,
		2 / 1.5811388,
	}
	for i := range want {
		if diff := values[i] - want[i]; diff < -0.01 || diff > 0.01 {
			t.Fatalf("projector value %d = %v, want about %v", i, values[i], want[i])
		}
	}
}

func TestVisionProjectorMLPBranch(t *testing.T) {
	requireNativeRuntime(t)
	rows := toBF16Bytes([]float32{3, 4})
	identity := toBF16Bytes([]float32{1, 0, 0, 1})
	got, err := visionProjector(rows, &VisionProjectorWeights{
		Linear1: VisionProjectorLinear{Weight: identity},
		Linear2: VisionProjectorLinear{Weight: identity},
		Eps:     0,
	}, 2)
	if err != nil {
		t.Fatalf("visionProjector MLP branch: %v", err)
	}
	values := bf16Floats(got)
	n0, n1 := float32(3/3.5355339), float32(4/3.5355339)
	want := []float32{geluTanhScalar(n0), geluTanhScalar(n1)}
	for i := range want {
		if diff := values[i] - want[i]; diff < -0.02 || diff > 0.02 {
			t.Fatalf("MLP projector value %d = %v, want about %v", i, values[i], want[i])
		}
	}
}

func TestVisionProjectorQuantizedRows(t *testing.T) {
	requireNativeRuntime(t)
	const inDim, outDim, groupSize, bits = 64, 2, 64, 4
	rows := f32ToBf16Slice(syntheticFloat32(inDim, 5))
	projector := VisionProjectorWeights{
		Projection: VisionProjectorLinear{
			Weight:    make([]byte, outDim*(inDim*bits/32)*4),
			Scales:    toBF16Bytes([]float32{1, 1}),
			Biases:    toBF16Bytes([]float32{0, 0}),
			OutDim:    outDim,
			InDim:     inDim,
			GroupSize: groupSize,
			Bits:      bits,
		},
		Eps: 1e-6,
	}
	got, err := visionProjector(rows, &projector, inDim)
	if err != nil {
		t.Fatalf("visionProjector(quant): %v", err)
	}
	if len(got) != outDim*bf16Size {
		t.Fatalf("quant projector bytes = %d, want %d", len(got), outDim*bf16Size)
	}
}

func TestVisionProjectorDenseBias(t *testing.T) {
	requireNativeRuntime(t)
	rows := toBF16Bytes([]float32{3, 4})
	projector := VisionProjectorWeights{
		Projection: VisionProjectorLinear{
			Weight: toBF16Bytes([]float32{
				0, 0,
				0, 0,
			}),
			Bias: toBF16Bytes([]float32{1, -2}),
		},
		Eps: 0,
	}
	got, err := visionProjector(rows, &projector, 2)
	if err != nil {
		t.Fatalf("visionProjector(bias): %v", err)
	}
	want := []float32{1, -2}
	values := bf16Floats(got)
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("projector bias value %d = %v, want %v", i, values[i], want[i])
		}
	}
}

func TestVisionMLPAddsLinearBiases(t *testing.T) {
	requireNativeRuntime(t)
	identity := toBF16Bytes([]float32{1, 0, 0, 1})
	zero := toBF16Bytes([]float32{0, 0, 0, 0})
	weights := &VisionLayerWeights{
		WGate: zero, BGate: toBF16Bytes([]float32{1, 2}),
		WUp: zero, BUp: toBF16Bytes([]float32{3, 4}),
		WDown: identity, BDown: toBF16Bytes([]float32{5, 6}),
	}
	got, err := visionMLP(toBF16Bytes([]float32{7, 8}), weights, 1, 2)
	if err != nil {
		t.Fatalf("visionMLP(bias): %v", err)
	}
	want := []float32{
		geluTanhScalar(1)*3 + 5,
		geluTanhScalar(2)*4 + 6,
	}
	values := bf16Floats(got)
	for i := range want {
		if diff := values[i] - want[i]; diff < -0.03 || diff > 0.03 {
			t.Fatalf("MLP bias value %d = %v, want about %v", i, values[i], want[i])
		}
	}
}

func TestVisionValidationGuards(t *testing.T) {
	requireNativeRuntime(t)

	pixels := toBF16Bytes(syntheticFloat32(2, 21))
	weight := toBF16Bytes(syntheticFloat32(4, 23))
	if _, err := VisionPatchEmbed(pixels[:len(pixels)-1], weight, nil, 1, 2, 2); err == nil {
		t.Fatal("VisionPatchEmbed(short pixels) error = nil")
	}
	if _, err := VisionPatchEmbed(pixels, weight[:len(weight)-1], nil, 1, 2, 2); err == nil {
		t.Fatal("VisionPatchEmbed(short weight) error = nil")
	}
	if _, err := VisionPatchEmbed(pixels, weight, toBF16Bytes([]float32{1}), 1, 2, 2); err == nil {
		t.Fatal("VisionPatchEmbed(short position embedding) error = nil")
	}

	if _, err := matRowsF32([]float32{1}, []float32{1, 2}, 1, 2, 2); err == nil {
		t.Fatal("matRowsF32(size mismatch) error = nil")
	}
	if got, err := matRowsF32(syntheticFloat32(4, 25), nil, 0, 2, 2); err != nil || len(got) != 0 {
		t.Fatalf("matRowsF32(zero rows) = len %d, err %v; want empty nil-error result", len(got), err)
	}

	q := toBF16Bytes(syntheticFloat32(4, 27))
	kv := toBF16Bytes(syntheticFloat32(2, 29))
	if _, err := VisionSDPA(q, kv, kv, 1, 2, 0, 2, 1); err == nil {
		t.Fatal("VisionSDPA(zero KV heads) error = nil")
	}
	if _, err := VisionSDPA(q[:len(q)-1], kv, kv, 1, 2, 1, 2, 1); err == nil {
		t.Fatal("VisionSDPA(short q) error = nil")
	}
	if _, err := VisionSDPA(q, kv[:len(kv)-1], kv, 1, 2, 1, 2, 1); err == nil {
		t.Fatal("VisionSDPA(short k) error = nil")
	}
	if _, err := VisionSDPA(q, kv, kv[:len(kv)-1], 1, 2, 1, 2, 1); err == nil {
		t.Fatal("VisionSDPA(short v) error = nil")
	}

	in := []float32{1, 2, 3, 4}
	noRoPE := vision2DRoPEHeadMajor(in, 1, 1, 4, 1, 1, 0)
	for i, want := range in {
		if noRoPE[i] != want {
			t.Fatalf("no-RoPE value %d = %v, want %v", i, noRoPE[i], want)
		}
	}
}

func TestVisionKernelFailureGuards(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, headDim, L = 4, 4, 1
	cfg := VisionConfig{
		Hidden: hidden, NumHeads: 1, NumKVHeads: 1, HeadDim: headDim,
		GridH: 1, GridW: 1, RMSNormEps: 1e-5,
	}
	weights := visionGuardLayerWeights(hidden, headDim, 6)
	x := toBF16Bytes(syntheticFloat32(L*hidden, 31))
	withWrongMainLibrary(t, func() {
		if _, err := VisionPatchEmbed(
			toBF16Bytes(syntheticFloat32(2, 33)),
			toBF16Bytes(syntheticFloat32(hidden*2, 35)),
			nil,
			L, 2, hidden,
		); err == nil {
			t.Fatal("VisionPatchEmbed(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := VisionSDPA(
			toBF16Bytes(syntheticFloat32(L*headDim, 37)),
			toBF16Bytes(syntheticFloat32(L*headDim, 39)),
			toBF16Bytes(syntheticFloat32(L*headDim, 41)),
			L, 1, 1, headDim, 1,
		); err == nil {
			t.Fatal("VisionSDPA(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := visionAttention(x, weights, cfg); err == nil {
			t.Fatal("visionAttention(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := visionMLP(x, weights, L, hidden); err == nil {
			t.Fatal("visionMLP(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := VisionEncoderLayer(x, weights, cfg); err == nil {
			t.Fatal("VisionEncoderLayer(wrong library) error = nil")
		}
	})
}

func visionGuardLayerWeights(hidden, headDim, ffDim int) *VisionLayerWeights {
	return &VisionLayerWeights{
		InputNorm:    toBF16Bytes(syntheticFloat32(hidden, 43)),
		PostAttnNorm: toBF16Bytes(syntheticFloat32(hidden, 45)),
		PreFFNorm:    toBF16Bytes(syntheticFloat32(hidden, 47)),
		PostFFNorm:   toBF16Bytes(syntheticFloat32(hidden, 49)),
		WQ:           toBF16Bytes(syntheticFloat32(headDim*hidden, 51)),
		WK:           toBF16Bytes(syntheticFloat32(headDim*hidden, 53)),
		WV:           toBF16Bytes(syntheticFloat32(headDim*hidden, 55)),
		WO:           toBF16Bytes(syntheticFloat32(hidden*headDim, 57)),
		QNorm:        toBF16Bytes(syntheticFloat32(headDim, 59)),
		KNorm:        toBF16Bytes(syntheticFloat32(headDim, 61)),
		WGate:        toBF16Bytes(syntheticFloat32(ffDim*hidden, 63)),
		WUp:          toBF16Bytes(syntheticFloat32(ffDim*hidden, 65)),
		WDown:        toBF16Bytes(syntheticFloat32(hidden*ffDim, 67)),
	}
}

func TestVisionTowerRejectsShortPositionEmbeddings(t *testing.T) {
	requireNativeRuntime(t)
	cfg := VisionConfig{Hidden: 2, PatchDim: 2, PoolKernel: 1}
	patches := toBF16Bytes([]float32{1, 2, 3, 4})
	w := &VisionWeights{
		PatchEmbedding:     toBF16Bytes([]float32{1, 0, 0, 1}),
		PositionEmbeddings: toBF16Bytes([]float32{1, 2}),
	}
	if _, err := VisionTower(patches, w, cfg); err == nil {
		t.Fatal("VisionTower(short position embeddings) error = nil")
	}
}

func TestVisionTowerRejectsNilWeights(t *testing.T) {
	requireNativeRuntime(t)
	cfg := VisionConfig{Hidden: 2, PatchDim: 2, PoolKernel: 1}
	patches := toBF16Bytes([]float32{1, 2})
	if _, err := VisionTower(patches, nil, cfg); err == nil {
		t.Fatal("VisionTower(nil weights) error = nil")
	}
}

// TestVisionImagePatchesGrid_Good pins the true-grid threading: a portrait
// image whose grid (6 rows × 3 cols) is NOT the most-square factorisation of
// its patch count returns the REAL dims — the seam that stops VisionTower
// re-deriving a transposed grid for every non-square image.
func TestVisionImagePatchesGrid_Good(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 48, 96)) // W=48 H=96 → grid 6×3 at patch 16
	for i := range img.Pix {
		img.Pix[i] = byte(i * 7)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode probe: %v", err)
	}
	cfg := &VisionImageFeatureConfig{PatchSize: 16, MaxSoftTokens: 1120, PoolingKernelSize: 3}
	patches, gridH, gridW, softTokens, err := VisionImagePatchesGrid(buf.Bytes(), cfg)
	if err != nil {
		t.Fatalf("VisionImagePatchesGrid: %v", err)
	}
	if gridH != 6 || gridW != 3 {
		t.Fatalf("grid = %dx%d, want 6x3 (the true portrait grid, not a transposed factorisation)", gridH, gridW)
	}
	if wantRows := gridH * gridW; len(patches) != wantRows*16*16*3*2 || softTokens != wantRows/9 {
		t.Fatalf("patches %d bytes, soft %d — want %d rows and %d soft tokens", len(patches), softTokens, wantRows, wantRows/9)
	}
}

// TestNativeTokenModel_ProjectImageFeaturesAt_Bad pins the At entry's guards:
// nil model and non-positive grid dims refuse before any tower work.
func TestNativeTokenModel_ProjectImageFeaturesAt_Bad(t *testing.T) {
	var nilModel *NativeTokenModel
	if _, err := nilModel.ProjectImageFeaturesAt(nil, 2, 3); err == nil {
		t.Fatal("nil model must refuse")
	}
	m := &NativeTokenModel{}
	if _, err := m.ProjectImageFeaturesAt([]byte{0, 0}, 0, 3); err == nil {
		t.Fatal("zero gridH must refuse")
	}
	if _, err := m.ProjectImageFeaturesAt([]byte{0, 0}, 2, -1); err == nil {
		t.Fatal("negative gridW must refuse")
	}
}
