// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPGemma4GGUFAffineSynthesis_Good_ReleasesSourceAllocations(t *testing.T) {
	driver := &fakeHIPDriver{available: true, maxLiveBytes: 464}
	tensors := make(map[string]hipTensor, 2)
	for _, name := range []string{"blk.0.attn_q.weight", "blk.0.attn_k.weight"} {
		payload := make([]byte, hipGGUFQ4KBlockBytes)
		pointer, err := driver.Malloc(uint64(len(payload)))
		core.RequireNoError(t, err)
		core.RequireNoError(t, driver.CopyHostToDevice(pointer, payload))
		tensors[name] = hipTensor{
			info: nativeTensorInfo{
				Name:       name,
				Dimensions: []uint64{hipGGUFQ4KBlockSize, 1},
				Type:       hipGGUFQ4KTensorType,
				TypeName:   "Q4_K",
				ByteSize:   uint64(len(payload)),
			},
			pointer: pointer,
		}
	}
	model := &hipLoadedModel{
		driver:      driver,
		modelInfo:   inference.ModelInfo{Architecture: "gemma4"},
		modelLabels: map[string]string{"gemma4_source_format": "gguf"},
		tensors:     tensors,
		hostTensors: map[string]nativeTensorInfo{},
	}

	core.RequireNoError(t, model.synthesizeGemma4GGUFAffineTensors())
	core.AssertEqual(t, uint64(320), driver.liveBytes)
	core.AssertEqual(t, 2, len(driver.frees))
	for _, name := range []string{"blk.0.attn_q.weight", "blk.0.attn_k.weight"} {
		if _, ok := model.tensors[name]; ok {
			t.Fatalf("source tensor %q remains resident after affine synthesis", name)
		}
	}
	for _, base := range []string{
		"language_model.model.layers.0.self_attn.q_proj",
		"language_model.model.layers.0.self_attn.k_proj",
	} {
		for _, suffix := range []string{".weight", ".scales", ".biases"} {
			if _, ok := model.tensors[base+suffix]; !ok {
				t.Fatalf("synthesized tensor %q is missing", base+suffix)
			}
		}
	}

	core.RequireNoError(t, model.Close())
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
}

func TestHIPGemma4GGUFNative12BGateUpSources_Good(t *testing.T) {
	t.Setenv(hipGemma4DenseQ4KEnv, "1")
	q4K := func(name string, rows int) hipTensor {
		return hipTensor{info: nativeTensorInfo{
			Name: name, Dimensions: []uint64{3840, uint64(rows)}, Type: hipGGUFQ4KTensorType, TypeName: "Q4_K",
		}}
	}
	sources := map[string]hipTensor{
		"blk.0.ffn_gate.weight": q4K("blk.0.ffn_gate.weight", 15360),
		"blk.0.ffn_up.weight":   q4K("blk.0.ffn_up.weight", 15360),
		"blk.0.ffn_down.weight": q4K("blk.0.ffn_down.weight", 3840),
		"blk.1.ffn_gate.weight": q4K("blk.1.ffn_gate.weight", 15360),
		"blk.2.ffn_gate.weight": {info: nativeTensorInfo{Name: "blk.2.ffn_gate.weight", Dimensions: []uint64{3840, 15360}, TypeName: "Q5_K"}},
		"blk.2.ffn_up.weight":   q4K("blk.2.ffn_up.weight", 15360),
	}

	selected := hipGemma4GGUFNative12BGateUpSourceNames(inference.ModelInfo{Architecture: "gemma4", HiddenSize: 3840}, sources)

	core.AssertEqual(t, 2, len(selected))
	core.AssertTrue(t, selected["blk.0.ffn_gate.weight"], "12B gate source must use native Q4_K")
	core.AssertTrue(t, selected["blk.0.ffn_up.weight"], "12B up source must use native Q4_K")
}

func TestHIPGemma4GGUFNative12BGateUpSources_Bad(t *testing.T) {
	t.Setenv(hipGemma4DenseQ4KEnv, "")
	q4K := func(name string) hipTensor {
		return hipTensor{info: nativeTensorInfo{
			Name: name, Dimensions: []uint64{3840, 15360}, Type: hipGGUFQ4KTensorType, TypeName: "Q4_K",
		}}
	}
	sources := map[string]hipTensor{
		"blk.0.ffn_gate.weight": q4K("blk.0.ffn_gate.weight"),
		"blk.0.ffn_up.weight":   q4K("blk.0.ffn_up.weight"),
	}

	selected := hipGemma4GGUFNative12BGateUpSourceNames(inference.ModelInfo{Architecture: "gemma4", HiddenSize: 3840}, sources)

	core.AssertEqual(t, 0, len(selected))
}

func TestHIPGemma4GGUFExpandedQ4KSynthesis_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	payload := make([]byte, 2*hipGGUFQ4KBlockBytes)
	pointer, err := driver.Malloc(uint64(len(payload)))
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(pointer, payload))
	source := hipTensor{info: nativeTensorInfo{
		Name: "blk.0.ffn_gate.weight", Dimensions: []uint64{256, 2}, Type: hipGGUFQ4KTensorType, TypeName: "Q4_K", ByteSize: uint64(len(payload)),
	}, pointer: pointer}
	model := &hipLoadedModel{driver: driver, tensors: map[string]hipTensor{source.info.Name: source}}

	core.RequireNoError(t, model.synthesizeGemma4GGUFExpandedQ4KTensor("language_model.model.layers.0.mlp.gate_proj", source))

	expanded, ok := model.tensors["language_model.model.layers.0.mlp.gate_proj.q4_k_expanded"]
	core.RequireTrue(t, ok)
	core.AssertEqual(t, []uint64{256, 2}, expanded.info.Dimensions)
	core.AssertEqual(t, "Q4_K_EXPANDED", expanded.info.TypeName)
	core.AssertEqual(t, uint64(2*hipGGUFQ4KExpandedBlockBytes), expanded.info.ByteSize)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameGGUFQ4KExpandMetadata, driver.launches[0].Name)

	core.RequireNoError(t, model.Close())
}

func TestHIPGemma4GGUFExpandedQ4KConfig_Good(t *testing.T) {
	const (
		rows = 15360
		cols = 3840
	)
	bytes := uint64(rows * (cols / hipGGUFQ4KBlockSize) * hipGGUFQ4KExpandedBlockBytes)
	prefix := "language_model.model.layers.0.mlp"
	model := &hipLoadedModel{tensors: map[string]hipTensor{
		prefix + ".gate_proj.q4_k_expanded": {info: nativeTensorInfo{Name: prefix + ".gate_proj.q4_k_expanded", Dimensions: []uint64{cols, rows}, Type: hipNativeTensorTypeQ4KExpanded, TypeName: "Q4_K_EXPANDED", ByteSize: bytes}, pointer: 11},
		prefix + ".up_proj.q4_k_expanded":   {info: nativeTensorInfo{Name: prefix + ".up_proj.q4_k_expanded", Dimensions: []uint64{cols, rows}, Type: hipNativeTensorTypeQ4KExpanded, TypeName: "Q4_K_EXPANDED", ByteSize: bytes}, pointer: 22},
	}}

	cfg, ok, err := model.loadedGemma4Q4NativeQ4KGateUpConfig(prefix)
	core.RequireNoError(t, err)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, nativeDevicePointer(11), cfg.GatePointer)
	core.AssertEqual(t, nativeDevicePointer(22), cfg.UpPointer)
	core.AssertEqual(t, rows, cfg.Rows)
	core.AssertEqual(t, cols, cfg.Cols)
	core.AssertEqual(t, bytes, cfg.GateBytes)
	core.AssertEqual(t, bytes, cfg.UpBytes)
}

func TestHIPGemma4GGUFQ4KAffineRepack_Good(t *testing.T) {
	block := make([]byte, hipGGUFQ4KBlockBytes)
	binary.LittleEndian.PutUint16(block[0:], rocmFloat32ToFloat16(0.5))
	binary.LittleEndian.PutUint16(block[2:], rocmFloat32ToFloat16(0.25))
	scales := block[4:16]
	for group := 0; group < 4; group++ {
		scales[group] = byte(group + 1)
		scales[group+4] = byte(group + 9)
	}
	for group := 4; group < 8; group++ {
		scales[group+4] = byte((group+9)<<4 | (group + 1))
	}
	qs := block[16:]
	for groupPair := 0; groupPair < 4; groupPair++ {
		lowGroup := groupPair * 2
		highGroup := lowGroup + 1
		for col := 0; col < 32; col++ {
			low := byte((lowGroup + col) & 0x0f)
			high := byte((highGroup + col) & 0x0f)
			qs[groupPair*32+col] = low | high<<4
		}
	}

	payload, err := hipRepackGGUFQ4KToAffine(nativeTensorInfo{
		Name:       "blk.0.attn_q.weight",
		Dimensions: []uint64{256, 1},
		Type:       12,
		TypeName:   "Q4_K",
		ByteSize:   uint64(len(block)),
	}, block)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 256, payload.Cols)
	core.AssertEqual(t, 32, payload.GroupSize)
	core.AssertEqual(t, 32, payload.PackedCols)
	core.AssertEqual(t, hipGGUFQ4KPackedWeightBytes, len(payload.Weights))
	core.AssertEqual(t, hipGGUFQ4KGroupsPerBlock*2, len(payload.Scales))
	core.AssertEqual(t, hipGGUFQ4KGroupsPerBlock*2, len(payload.Biases))

	group0Word := binary.LittleEndian.Uint32(payload.Weights[0:])
	group1Word := binary.LittleEndian.Uint32(payload.Weights[16:])
	core.AssertEqual(t, uint32(0x76543210), group0Word)
	core.AssertEqual(t, uint32(0x87654321), group1Word)
	assertFloat32Near(t, 0.5, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, -2.25, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))
	assertFloat32Near(t, 2.5, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[8:])))
	assertFloat32Near(t, -3.25, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[8:])))
}

func TestHIPGemma4GGUFQ4_0AffineRepack_Good(t *testing.T) {
	block := make([]byte, hipGGUFQ4_0BlockBytes)
	binary.LittleEndian.PutUint16(block[0:], rocmFloat32ToFloat16(0.25))
	for lane := 0; lane < hipGGUFQ4_0BlockSize/2; lane++ {
		block[2+lane] = byte(lane) | byte(15-lane)<<4
	}

	payload, err := hipRepackGGUFQ4_0ToAffine(nativeTensorInfo{
		Name:       "blk.0.attn_q.weight",
		Dimensions: []uint64{32, 1},
		Type:       hipGGUFQ4_0TensorType,
		TypeName:   "Q4_0",
		ByteSize:   uint64(len(block)),
	}, block)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 32, payload.Cols)
	core.AssertEqual(t, hipGGUFQ4_0GroupSize, payload.GroupSize)
	core.AssertEqual(t, hipGGUFQ4_0PackedWeightBytes/4, payload.PackedCols)
	core.AssertEqual(t, 4, payload.Bits)
	core.AssertEqual(t, uint32(0), hipTestPackedValue(payload.Weights, 0, 4))
	core.AssertEqual(t, uint32(7), hipTestPackedValue(payload.Weights, 7, 4))
	core.AssertEqual(t, uint32(8), hipTestPackedValue(payload.Weights, 8, 4))
	core.AssertEqual(t, uint32(15), hipTestPackedValue(payload.Weights, 15, 4))
	core.AssertEqual(t, uint32(15), hipTestPackedValue(payload.Weights, 16, 4))
	core.AssertEqual(t, uint32(8), hipTestPackedValue(payload.Weights, 23, 4))
	core.AssertEqual(t, uint32(7), hipTestPackedValue(payload.Weights, 24, 4))
	core.AssertEqual(t, uint32(0), hipTestPackedValue(payload.Weights, 31, 4))
	assertFloat32Near(t, 0.25, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, -2, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))
}

func TestHIPGemma4GGUFQ4_1AffineRepack_Good(t *testing.T) {
	block := make([]byte, hipGGUFQ4_1BlockBytes)
	binary.LittleEndian.PutUint16(block[0:], rocmFloat32ToFloat16(0.5))
	binary.LittleEndian.PutUint16(block[2:], rocmFloat32ToFloat16(-1.25))
	for lane := 0; lane < hipGGUFQ4_1BlockSize/2; lane++ {
		low := byte((lane + 3) & 0x0f)
		high := byte((lane + 11) & 0x0f)
		block[4+lane] = low | high<<4
	}

	payload, err := hipRepackGGUFQ4_1ToAffine(nativeTensorInfo{
		Name:       "blk.0.ffn_down.weight",
		Dimensions: []uint64{32, 1},
		Type:       hipGGUFQ4_1TensorType,
		TypeName:   "Q4_1",
		ByteSize:   uint64(len(block)),
	}, block)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 32, payload.Cols)
	core.AssertEqual(t, hipGGUFQ4_1GroupSize, payload.GroupSize)
	core.AssertEqual(t, hipGGUFQ4_1PackedWeightBytes/4, payload.PackedCols)
	core.AssertEqual(t, 4, payload.Bits)
	core.AssertEqual(t, uint32(3), hipTestPackedValue(payload.Weights, 0, 4))
	core.AssertEqual(t, uint32(10), hipTestPackedValue(payload.Weights, 7, 4))
	core.AssertEqual(t, uint32(11), hipTestPackedValue(payload.Weights, 8, 4))
	core.AssertEqual(t, uint32(2), hipTestPackedValue(payload.Weights, 15, 4))
	core.AssertEqual(t, uint32(11), hipTestPackedValue(payload.Weights, 16, 4))
	core.AssertEqual(t, uint32(2), hipTestPackedValue(payload.Weights, 23, 4))
	core.AssertEqual(t, uint32(3), hipTestPackedValue(payload.Weights, 24, 4))
	core.AssertEqual(t, uint32(10), hipTestPackedValue(payload.Weights, 31, 4))
	assertFloat32Near(t, 0.5, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, -1.25, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))
}

func TestHIPGemma4GGUFQ5KAffineRepack_Good(t *testing.T) {
	block := make([]byte, hipGGUFQ5KBlockBytes)
	binary.LittleEndian.PutUint16(block[0:], rocmFloat32ToFloat16(0.5))
	binary.LittleEndian.PutUint16(block[2:], rocmFloat32ToFloat16(0.25))
	scales := block[4:16]
	for group := 0; group < 4; group++ {
		scales[group] = byte(group + 1)
		scales[group+4] = byte(group + 9)
	}
	for group := 4; group < 8; group++ {
		scales[group+4] = byte((group+9)<<4 | (group + 1))
	}
	setQ5 := func(group, lane int, value byte) {
		t.Helper()
		il := group / 2
		highGroup := group%2 == 1
		ir := lane / 2
		sub := lane % 2
		qsIndex := 48 + 32*il + 2*ir + sub
		if highGroup {
			block[qsIndex] |= (value & 0x0f) << 4
		} else {
			block[qsIndex] |= value & 0x0f
		}
		if value&0x10 != 0 {
			mask := byte(1 << uint(2*il))
			if highGroup {
				mask <<= 1
			}
			block[16+2*ir+sub] |= mask
		}
	}
	for group := 0; group < hipGGUFQ5KGroupsPerBlock; group++ {
		for lane := 0; lane < hipGGUFQ5KGroupSize; lane++ {
			setQ5(group, lane, byte((group+lane)&0x1f))
		}
	}
	core.AssertEqual(t, uint8(0), hipGGUFQ5KQuant(block, 0, 0))
	core.AssertEqual(t, uint8(16), hipGGUFQ5KQuant(block, 0, 16))
	core.AssertEqual(t, uint8(17), hipGGUFQ5KQuant(block, 1, 16))

	payload, err := hipRepackGGUFQ5KToAffine(nativeTensorInfo{
		Name:       "per_layer_token_embd.weight",
		Dimensions: []uint64{256, 1},
		Type:       13,
		TypeName:   "Q5_K",
		ByteSize:   uint64(len(block)),
	}, block)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 256, payload.Cols)
	core.AssertEqual(t, 32, payload.GroupSize)
	core.AssertEqual(t, 48, payload.PackedCols)
	core.AssertEqual(t, hipGGUFQ5KPackedWeightBytes, len(payload.Weights))
	core.AssertEqual(t, uint32(0), hipTestPackedQ6Value(payload.Weights, 0))
	core.AssertEqual(t, uint32(15), hipTestPackedQ6Value(payload.Weights, 15))
	core.AssertEqual(t, uint32(16), hipTestPackedQ6Value(payload.Weights, 16))
	core.AssertEqual(t, uint32(20), hipTestPackedQ6Value(payload.Weights, 3*32+17))
	assertFloat32Near(t, 0.5, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, -2.25, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))
}

func TestHIPGemma4GGUFQ6KAffineRepack_Good(t *testing.T) {
	block := make([]byte, hipGGUFQ6KBlockBytes)
	binary.LittleEndian.PutUint16(block[208:], rocmFloat32ToFloat16(0.25))
	for group := 0; group < hipGGUFQ6KGroupsPerBlock; group++ {
		block[192+group] = byte(int8(group + 2))
	}
	negativeScale := int8(-3)
	block[192+5] = byte(negativeScale)
	setQ6 := func(group, lane int, value byte) {
		t.Helper()
		ip := group / 8
		localGroup := group % 8
		laneOffset := 0
		if localGroup%2 == 1 {
			laneOffset = 16
		}
		quartile := localGroup / 2
		qlIndex := ip*64 + laneOffset + lane
		if quartile == 1 || quartile == 3 {
			qlIndex += 32
		}
		if quartile >= 2 {
			block[qlIndex] |= (value & 0x0f) << 4
		} else {
			block[qlIndex] |= value & 0x0f
		}
		qhIndex := 128 + ip*32 + laneOffset + lane
		block[qhIndex] |= ((value >> 4) & 0x03) << uint(quartile*2)
	}
	for group := 0; group < hipGGUFQ6KGroupsPerBlock; group++ {
		for lane := 0; lane < hipGGUFQ6KGroupSize; lane++ {
			setQ6(group, lane, byte((32+group+lane)&0x3f))
		}
	}

	payload, err := hipRepackGGUFQ6KToAffine(nativeTensorInfo{
		Name:       "blk.0.attn_v.weight",
		Dimensions: []uint64{256, 1},
		Type:       14,
		TypeName:   "Q6_K",
		ByteSize:   uint64(len(block)),
	}, block)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 256, payload.Cols)
	core.AssertEqual(t, 16, payload.GroupSize)
	core.AssertEqual(t, 48, payload.PackedCols)
	core.AssertEqual(t, hipGGUFQ6KPackedWeightBytes, len(payload.Weights))
	core.AssertEqual(t, hipGGUFQ6KGroupsPerBlock*2, len(payload.Scales))
	core.AssertEqual(t, uint32(32), hipTestPackedQ6Value(payload.Weights, 0))
	core.AssertEqual(t, uint32(47), hipTestPackedQ6Value(payload.Weights, 15))
	core.AssertEqual(t, uint32(39), hipTestPackedQ6Value(payload.Weights, 5*16+2))
	assertFloat32Near(t, 0.5, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, -16, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))
	assertFloat32Near(t, -0.75, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[5*2:])))
	assertFloat32Near(t, 24, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[5*2:])))
}

func TestHIPGemma4GGUFQ8_0AffineRepack_Good(t *testing.T) {
	block := make([]byte, hipGGUFQ8_0BlockBytes)
	binary.LittleEndian.PutUint16(block[0:], rocmFloat32ToFloat16(0.5))
	for lane := 0; lane < hipGGUFQ8_0BlockSize; lane++ {
		block[2+lane] = byte(int8(lane - 16))
	}

	payload, err := hipRepackGGUFQ8_0ToAffine(nativeTensorInfo{
		Name:       "blk.0.attn_q.weight",
		Dimensions: []uint64{32, 1},
		Type:       hipGGUFQ8_0TensorType,
		TypeName:   "Q8_0",
		ByteSize:   uint64(len(block)),
	}, block)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 32, payload.Cols)
	core.AssertEqual(t, hipGGUFQ8_0GroupSize, payload.GroupSize)
	core.AssertEqual(t, 8, payload.PackedCols)
	core.AssertEqual(t, 8, payload.Bits)
	core.AssertEqual(t, uint32(112), hipTestPackedValue(payload.Weights, 0, 8))
	core.AssertEqual(t, uint32(127), hipTestPackedValue(payload.Weights, 15, 8))
	core.AssertEqual(t, uint32(128), hipTestPackedValue(payload.Weights, 16, 8))
	core.AssertEqual(t, uint32(143), hipTestPackedValue(payload.Weights, 31, 8))
	assertFloat32Near(t, 0.5, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, -64, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))

	weights := make([]uint32, len(payload.Weights)/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(payload.Weights[index*4:])
	}
	scales := []uint16{binary.LittleEndian.Uint16(payload.Scales)}
	biases := []uint16{binary.LittleEndian.Uint16(payload.Biases)}
	input := make([]float32, hipGGUFQ8_0BlockSize)
	want := float32(0)
	for lane := 0; lane < hipGGUFQ8_0BlockSize; lane++ {
		input[lane] = float32((lane % 7) - 3)
		want += input[lane] * float32(int8(block[2+lane])) * 0.5
	}
	got, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, 1, hipGGUFQ8_0BlockSize, hipGGUFQ8_0GroupSize, 8)
	core.RequireNoError(t, err)
	assertFloat32Near(t, want, got[0])
}

func TestHIPGemma4GGUFAliasPayload_Good(t *testing.T) {
	f32Data := make([]byte, 8)
	binary.LittleEndian.PutUint32(f32Data[0:], math.Float32bits(1))
	binary.LittleEndian.PutUint32(f32Data[4:], math.Float32bits(-2))
	payload, dimensions, tensorType, typeName, err := hipGGUFAliasPayload(nativeTensorInfo{
		Name:       "output_norm.weight",
		Dimensions: []uint64{2},
		Type:       0,
		TypeName:   "F32",
		ByteSize:   uint64(len(f32Data)),
	}, f32Data)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []uint64{2}, dimensions)
	core.AssertEqual(t, uint32(0), tensorType)
	core.AssertEqual(t, "F32", typeName)
	core.AssertEqual(t, f32Data, payload)

	f32Projection := make([]byte, 8)
	binary.LittleEndian.PutUint32(f32Projection[0:], math.Float32bits(3))
	binary.LittleEndian.PutUint32(f32Projection[4:], math.Float32bits(4))
	payload, dimensions, tensorType, typeName, err = hipGGUFAliasPayload(nativeTensorInfo{
		Name:       "per_layer_model_proj.weight",
		Dimensions: []uint64{1, 2},
		Type:       0,
		TypeName:   "F32",
		ByteSize:   uint64(len(f32Projection)),
	}, f32Projection)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []uint64{2, 1}, dimensions)
	core.AssertEqual(t, hipNativeTensorTypeBF16, tensorType)
	core.AssertEqual(t, "BF16", typeName)
	core.AssertEqual(t, 4, len(payload))
	assertFloat32Near(t, 3, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[0:])))
	assertFloat32Near(t, 4, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[2:])))

	bf16Data := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	payload, dimensions, tensorType, typeName, err = hipGGUFAliasPayload(nativeTensorInfo{
		Name:       "per_layer_model_proj.weight",
		Dimensions: []uint64{3, 2},
		Type:       30,
		TypeName:   "BF16",
		ByteSize:   uint64(len(bf16Data)),
	}, bf16Data)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []uint64{2, 3}, dimensions)
	core.AssertEqual(t, hipNativeTensorTypeBF16, tensorType)
	core.AssertEqual(t, "BF16", typeName)
	core.AssertEqual(t, bf16Data, payload)
}

func TestHIPGemma4GGUFAliasPayload_Good_F16Projection(t *testing.T) {
	f16Data := make([]byte, 4)
	binary.LittleEndian.PutUint16(f16Data[0:], 0x4200)
	binary.LittleEndian.PutUint16(f16Data[2:], 0xc400)
	payload, dimensions, tensorType, typeName, err := hipGGUFAliasPayload(nativeTensorInfo{
		Name:       "per_layer_model_proj.weight",
		Dimensions: []uint64{1, 2},
		Type:       hipNativeTensorTypeF16,
		TypeName:   "F16",
		ByteSize:   uint64(len(f16Data)),
	}, f16Data)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []uint64{2, 1}, dimensions)
	core.AssertEqual(t, hipNativeTensorTypeBF16, tensorType)
	core.AssertEqual(t, "BF16", typeName)
	core.AssertEqual(t, 4, len(payload))
	assertFloat32Near(t, 3, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[0:])))
	assertFloat32Near(t, -4, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[2:])))
}

func TestHIPGemma4GGUFAffineCanonicalName_Good_AssistantNextNProjection(t *testing.T) {
	base, ok := hipGemma4CanonicalAffineBaseForGGUFWeightName("nextn.pre_projection.weight")
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, "pre_projection", base)

	base, ok = hipGemma4CanonicalAffineBaseForGGUFWeightName("nextn.post_projection.weight")
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, "post_projection", base)
}

func TestHIPGemma4GGUFBF16CanonicalName_Good_AssistantLayerOutputScale(t *testing.T) {
	name, ok := hipGemma4CanonicalBF16NameForGGUFName("blk.3.layer_output_scale.weight")
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, "language_model.model.layers.3.layer_scalar", name)
}

func TestHIPGemma4GGUFBF16CanonicalName_Good_MoENorms(t *testing.T) {
	for source, want := range map[string]string{
		"blk.3.pre_ffw_norm_2.weight":  "language_model.model.layers.3.pre_feedforward_layernorm_2.weight",
		"blk.3.post_ffw_norm_1.weight": "language_model.model.layers.3.post_feedforward_layernorm_1.weight",
		"blk.3.post_ffw_norm_2.weight": "language_model.model.layers.3.post_feedforward_layernorm_2.weight",
	} {
		name, ok := hipGemma4CanonicalBF16NameForGGUFName(source)
		core.AssertEqual(t, true, ok)
		core.AssertEqual(t, want, name)
	}
}

func TestHIPGemma4GGUFF32AffineQuantize_Good(t *testing.T) {
	data := make([]byte, 32*4)
	for index := 0; index < 32; index++ {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(float32(index)))
	}
	payload, err := hipQuantizeGGUFF32ToAffine8(nativeTensorInfo{
		Name:       "blk.0.inp_gate.weight",
		Dimensions: []uint64{32, 1},
		Type:       0,
		TypeName:   "F32",
		ByteSize:   uint64(len(data)),
	}, data)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, payload.Rows)
	core.AssertEqual(t, 32, payload.Cols)
	core.AssertEqual(t, 32, payload.GroupSize)
	core.AssertEqual(t, 8, payload.Bits)
	core.AssertEqual(t, 8, payload.PackedCols)
	core.AssertEqual(t, uint32(0), hipTestPackedValue(payload.Weights, 0, 8))
	core.AssertEqual(t, uint32(255), hipTestPackedValue(payload.Weights, 31, 8))
	assertFloat32Near(t, 31.0/255.0, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Scales[0:])))
	assertFloat32Near(t, 0, hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload.Biases[0:])))
}

func hipTestPackedQ6Value(weights []byte, col int) uint32 {
	return hipTestPackedValue(weights, col, 6)
}

func hipTestPackedValue(weights []byte, col, bits int) uint32 {
	bitOffset := col * 6
	if bits != 6 {
		bitOffset = col * bits
	}
	wordIndex := bitOffset / 32
	shift := uint(bitOffset % 32)
	value := binary.LittleEndian.Uint32(weights[wordIndex*4:]) >> shift
	if shift+uint(bits) > 32 {
		value |= binary.LittleEndian.Uint32(weights[(wordIndex+1)*4:]) << (32 - shift)
	}
	return value & ((1 << bits) - 1)
}
