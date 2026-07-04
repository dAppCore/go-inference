// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type rocmLoRAFusePair struct {
	Name   string
	A      rocmFuseTensorRef
	B      rocmFuseTensorRef
	AShape []uint64
	BShape []uint64
}

const rocmLoRAFuseMLXAffineGroupSize = 64

type rocmLoRAFuseBaseMatch struct {
	Key         string
	Ref         rocmFuseTensorRef
	Quantized   bool
	ScaleKey    string
	Scale       rocmFuseTensorRef
	BiasKey     string
	Bias        rocmFuseTensorRef
	SidecarKeys []string
	Bits        int
	GroupSize   int
	DenseShape  []uint64
}

type rocmFuseTensorRef struct {
	Name      string
	Path      string
	DType     string
	Shape     []uint64
	DataStart int64
	ByteLen   uint64
}

type rocmFuseWriteTensor struct {
	Name  string
	DType string
	Shape []uint64
	Data  []byte
}

func FuseLoRAIntoModelPack(ctx context.Context, opts LoRAFuseOptions) (*LoRAFuseResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	basePath := strings.TrimSpace(opts.BasePath)
	adapterPath := strings.TrimSpace(opts.AdapterPath)
	outputPath := strings.TrimSpace(opts.OutputPath)
	if basePath == "" {
		return nil, core.NewError("rocm: source pack root is required")
	}
	if adapterPath == "" {
		return nil, core.NewError("rocm: LoRA adapter path is required")
	}
	if outputPath == "" {
		return nil, core.NewError("rocm: fused model output path is required")
	}
	if rocmLoRAFuseLooksLikeWeightFile(outputPath) {
		return nil, core.NewError("rocm: fused output path must be a model-pack directory")
	}

	baseRoot, sourceWeights, err := rocmLoRAFuseBaseWeights(basePath)
	if err != nil {
		return nil, err
	}
	if len(sourceWeights) == 0 {
		return nil, core.NewError("rocm: no base safetensors weight files available for LoRA fusion")
	}
	if sameFilesystemPath(baseRoot, outputPath) {
		return nil, core.NewError("rocm: fused output path must differ from source model path")
	}
	if err := rocmLoRAFuseEnsureEmptyWeightDestination(outputPath); err != nil {
		return nil, err
	}

	adapterWeightPath, err := rocmLoRAFuseAdapterWeights(adapterPath)
	if err != nil {
		return nil, err
	}
	adapterIndex, err := rocmReadFuseSafetensorsIndex(adapterWeightPath)
	if err != nil {
		return nil, core.E("rocm.LoRA.Fuse", "read adapter safetensors", err)
	}
	pairs, err := rocmLoRAFusePairs(adapterIndex)
	if err != nil {
		return nil, err
	}
	scale, err := rocmLoRAFuseScale(opts.Adapter)
	if err != nil {
		return nil, err
	}
	architecture := firstNonEmptyString(opts.Architecture, opts.Adapter.Labels["adapter_base_architecture"])

	baseIndexes := make([]map[string]rocmFuseTensorRef, 0, len(sourceWeights))
	baseIndexByCanonical := map[string]rocmFuseTensorRef{}
	for _, sourceWeight := range sourceWeights {
		index, err := rocmReadFuseSafetensorsIndex(sourceWeight)
		if err != nil {
			return nil, core.E("rocm.LoRA.Fuse", "read base safetensors "+filepath.Base(sourceWeight), err)
		}
		baseIndexes = append(baseIndexes, index)
		for name, ref := range index {
			baseIndexByCanonical[name] = ref
			if canonical, ok := ROCmCanonicalWeightName(architecture, name); ok && canonical != "" {
				baseIndexByCanonical[canonical] = ref
			}
		}
	}
	pairBaseMatches := make(map[string]rocmLoRAFuseBaseMatch, len(pairs))
	sidecarSkips := map[string]struct{}{}
	quantizedTargets := 0
	for name, pair := range pairs {
		baseKey := rocmLoRAFuseBaseWeightKey(name, architecture)
		baseMatch, ok, err := rocmLoRAFuseBaseMatchForKey(baseIndexByCanonical, baseKey)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, core.NewError("rocm: base weight not found for LoRA target: " + baseKey)
		}
		if err := rocmLoRAFuseValidatePair(baseMatch, pair); err != nil {
			return nil, err
		}
		pairBaseMatches[name] = baseMatch
		if baseMatch.Quantized {
			quantizedTargets++
			for _, sidecarKey := range baseMatch.SidecarKeys {
				if sidecarKey != "" {
					sidecarSkips[sidecarKey] = struct{}{}
				}
			}
		}
	}

	if err := os.MkdirAll(outputPath, 0o755); err != nil {
		return nil, err
	}
	if err := rocmLoRAFuseCopyModelPackMetadata(baseRoot, outputPath); err != nil {
		return nil, err
	}

	fusedKeys := make([]string, 0, len(pairs))
	weightFiles := make([]string, 0, len(sourceWeights))
	fusedPairs := map[string]struct{}{}
	multiShard := len(sourceWeights) > 1
	for i, sourceWeight := range sourceWeights {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		index := baseIndexes[i]
		tensors := make([]rocmFuseWriteTensor, 0, len(index))
		names := make([]string, 0, len(index))
		for name := range index {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			if _, skip := sidecarSkips[name]; skip {
				continue
			}
			ref := index[name]
			pairName, baseMatch, pair, ok := rocmLoRAFusePairForBaseKey(pairs, pairBaseMatches, name)
			if ok {
				data, err := rocmLoRAFuseMergedF32(baseMatch, pair, scale)
				if err != nil {
					return nil, err
				}
				tensors = append(tensors, rocmFuseWriteTensor{Name: name, DType: "F32", Shape: cloneUint64Slice(baseMatch.DenseShape), Data: data})
				fusedKeys = append(fusedKeys, name)
				fusedPairs[pairName] = struct{}{}
				continue
			}
			raw, err := rocmReadFuseTensorRaw(ref)
			if err != nil {
				return nil, err
			}
			tensors = append(tensors, rocmFuseWriteTensor{Name: name, DType: ref.DType, Shape: cloneUint64Slice(ref.Shape), Data: raw})
		}

		outputName := "model.safetensors"
		if multiShard {
			outputName = filepath.Base(sourceWeight)
		}
		weightPath := filepath.Join(outputPath, outputName)
		if err := rocmWriteFuseSafetensors(weightPath, tensors); err != nil {
			return nil, core.E("rocm.LoRA.Fuse", "write fused safetensors", err)
		}
		weightFiles = append(weightFiles, weightPath)
	}
	for name := range pairs {
		if _, ok := fusedPairs[name]; !ok {
			return nil, core.NewError("rocm: base weight not fused for LoRA target: " + rocmLoRAFuseBaseWeightKey(name, architecture))
		}
	}
	slices.Sort(fusedKeys)
	fusedLayers := rocmLoRAFuseLayerNames(fusedKeys)

	labels := cloneStringMap(opts.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels["backend"] = "rocm"
	labels["fuse_runtime"] = "dense_f32_cpu"
	labels["fuse_safetensors"] = "linked"
	labels["fuse_quantized_base"] = "not_present"
	if quantizedTargets > 0 {
		labels["fuse_quantized_base"] = "dequantized_dense"
		labels["fuse_dequantized_targets"] = fmt.Sprintf("%d", quantizedTargets)
		labels["fuse_quantized_modes"] = "mlx_affine_q4_q6_q8"
	}
	labels["fuse_weight_files"] = fmt.Sprintf("%d", len(weightFiles))
	labels["fuse_weight_count"] = fmt.Sprintf("%d", len(fusedKeys))
	labels["fuse_layer_count"] = fmt.Sprintf("%d", len(fusedLayers))

	provenancePath := filepath.Join(outputPath, LoRAFuseProvenanceFile)
	provenance := LoRAFuseProvenance{
		Version:         1,
		SourcePath:      baseRoot,
		OutputPath:      outputPath,
		WeightFiles:     rocmLoRAFuseOutputWeightFileNames(weightFiles),
		Adapter:         cloneAdapterIdentity(opts.Adapter),
		FusedWeightKeys: append([]string(nil), fusedKeys...),
		FusedLayers:     append([]string(nil), fusedLayers...),
		Labels:          cloneStringMap(labels),
	}
	if err := rocmWriteLoRAFuseProvenance(provenancePath, provenance); err != nil {
		return nil, err
	}

	return &LoRAFuseResult{
		OutputPath:      outputPath,
		WeightFiles:     weightFiles,
		ProvenancePath:  provenancePath,
		Adapter:         cloneAdapterIdentity(opts.Adapter),
		FusedWeights:    len(fusedKeys),
		FusedWeightKeys: fusedKeys,
		FusedLayers:     fusedLayers,
		Labels:          labels,
	}, nil
}

func rocmLoRAFuseLayerNames(fusedKeys []string) []string {
	seen := map[string]struct{}{}
	layers := make([]string, 0, len(fusedKeys))
	for _, key := range fusedKeys {
		layer := strings.TrimSuffix(key, ".weight")
		if strings.TrimSpace(layer) == "" {
			continue
		}
		if _, ok := seen[layer]; ok {
			continue
		}
		seen[layer] = struct{}{}
		layers = append(layers, layer)
	}
	slices.Sort(layers)
	return layers
}

func rocmLoRAFuseBaseWeights(basePath string) (string, []string, error) {
	info, err := os.Stat(basePath)
	if err != nil {
		return "", nil, err
	}
	baseRoot := basePath
	if !info.IsDir() {
		baseRoot = filepath.Dir(basePath)
	}
	weights := discoverROCmWeightFiles(basePath, info)
	safetensors := weights[:0]
	for _, weight := range weights {
		if strings.EqualFold(filepath.Ext(weight), ".safetensors") {
			safetensors = append(safetensors, weight)
		}
	}
	return baseRoot, safetensors, nil
}

func rocmLoRAFuseAdapterWeights(adapterPath string) (string, error) {
	info, err := os.Stat(adapterPath)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		if strings.EqualFold(filepath.Ext(adapterPath), ".safetensors") {
			return adapterPath, nil
		}
		return "", core.NewError("rocm: LoRA adapter file must be .safetensors")
	}
	candidate := filepath.Join(adapterPath, "adapter.safetensors")
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}
	matches, err := filepath.Glob(filepath.Join(adapterPath, "*.safetensors"))
	if err != nil {
		return "", err
	}
	slices.Sort(matches)
	if len(matches) == 0 {
		return "", core.NewError("rocm: no adapter safetensors found")
	}
	return matches[0], nil
}

func rocmLoRAFuseScale(adapter inference.AdapterIdentity) (float32, error) {
	if scale := firstPositiveFloatFromLabels(adapter.Labels, "adapter_scale", "lora_scale"); scale > 0 {
		return float32(scale), nil
	}
	if adapter.Rank > 0 && adapter.Alpha > 0 {
		return adapter.Alpha / float32(adapter.Rank), nil
	}
	if adapter.Rank <= 0 {
		return 0, core.NewError("rocm: LoRA adapter rank is required for fusion")
	}
	return 2, nil
}

func firstPositiveFloatFromLabels(labels map[string]string, keys ...string) float64 {
	if labels == nil {
		return 0
	}
	for _, key := range keys {
		value := strings.TrimSpace(labels[key])
		if value == "" {
			continue
		}
		if parsed, err := strconv.ParseFloat(value, 64); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 0
}

func rocmLoRAFusePairs(index map[string]rocmFuseTensorRef) (map[string]rocmLoRAFusePair, error) {
	pairs := map[string]rocmLoRAFusePair{}
	for name, ref := range index {
		pairName, suffix, ok := rocmLoRAFusePairName(name)
		if !ok {
			continue
		}
		pair := pairs[pairName]
		pair.Name = pairName
		switch suffix {
		case "a":
			pair.A = ref
			pair.AShape = cloneUint64Slice(ref.Shape)
		case "b":
			pair.B = ref
			pair.BShape = cloneUint64Slice(ref.Shape)
		}
		pairs[pairName] = pair
	}
	for name, pair := range pairs {
		if pair.A.Name == "" || pair.B.Name == "" {
			return nil, core.NewError("rocm: incomplete LoRA tensor pair: " + name)
		}
	}
	if len(pairs) == 0 {
		return nil, core.NewError("rocm: no LoRA tensor pairs found")
	}
	return pairs, nil
}

func rocmLoRAFusePairName(weightName string) (string, string, bool) {
	if strings.HasSuffix(weightName, ".weight") {
		head := len(weightName) - len(".lora_X.weight")
		if head < 0 || weightName[head:head+6] != ".lora_" {
			return "", "", false
		}
		switch weightName[head+6] {
		case 'a', 'A':
			return weightName[:head], "a", true
		case 'b', 'B':
			return weightName[:head], "b", true
		default:
			return "", "", false
		}
	}
	head := len(weightName) - len(".lora_X")
	if head < 0 || weightName[head:head+6] != ".lora_" {
		return "", "", false
	}
	switch weightName[head+6] {
	case 'a', 'A':
		return weightName[:head], "a", true
	case 'b', 'B':
		return weightName[:head], "b", true
	default:
		return "", "", false
	}
}

func rocmLoRAFuseBaseWeightKey(pairName string, architecture string) string {
	if canonical, ok := Gemma4LoRACanonicalTarget(architecture, pairName); ok {
		return canonical + ".weight"
	}
	return pairName + ".weight"
}

func rocmLoRAFuseBaseMatchForKey(index map[string]rocmFuseTensorRef, baseKey string) (rocmLoRAFuseBaseMatch, bool, error) {
	base, ok := index[baseKey]
	if !ok {
		return rocmLoRAFuseBaseMatch{}, false, nil
	}
	match := rocmLoRAFuseBaseMatch{
		Key:        base.Name,
		Ref:        base,
		DenseShape: cloneUint64Slice(base.Shape),
	}
	scale, bias, sidecars := rocmLoRAFuseBaseSidecars(index, base.Name, baseKey)
	if scale.Name == "" {
		return match, true, nil
	}
	bits, denseShape, err := rocmLoRAFuseInferMLXAffine(base, scale, rocmLoRAFuseMLXAffineGroupSize)
	if err != nil {
		return rocmLoRAFuseBaseMatch{}, false, err
	}
	match.Quantized = true
	match.ScaleKey = scale.Name
	match.Scale = scale
	match.BiasKey = bias.Name
	match.Bias = bias
	match.SidecarKeys = sidecars
	match.Bits = bits
	match.GroupSize = rocmLoRAFuseMLXAffineGroupSize
	match.DenseShape = denseShape
	return match, true, nil
}

func rocmLoRAFuseBaseSidecars(index map[string]rocmFuseTensorRef, actualKey, canonicalKey string) (rocmFuseTensorRef, rocmFuseTensorRef, []string) {
	prefixes := make([]string, 0, 2)
	if prefix, ok := rocmLoRAFuseBaseWeightPrefix(actualKey); ok {
		prefixes = append(prefixes, prefix)
	}
	if prefix, ok := rocmLoRAFuseBaseWeightPrefix(canonicalKey); ok && prefix != "" {
		duplicate := false
		for _, existing := range prefixes {
			if existing == prefix {
				duplicate = true
				break
			}
		}
		if !duplicate {
			prefixes = append(prefixes, prefix)
		}
	}

	var scale rocmFuseTensorRef
	var bias rocmFuseTensorRef
	sidecars := []string{}
	seen := map[string]struct{}{}
	for _, prefix := range prefixes {
		if ref, ok := index[prefix+".scales"]; ok {
			if scale.Name == "" {
				scale = ref
			}
			if _, exists := seen[ref.Name]; !exists {
				sidecars = append(sidecars, ref.Name)
				seen[ref.Name] = struct{}{}
			}
		}
		if ref, ok := index[prefix+".biases"]; ok {
			if bias.Name == "" {
				bias = ref
			}
			if _, exists := seen[ref.Name]; !exists {
				sidecars = append(sidecars, ref.Name)
				seen[ref.Name] = struct{}{}
			}
		}
	}
	return scale, bias, sidecars
}

func rocmLoRAFuseBaseWeightPrefix(key string) (string, bool) {
	if !strings.HasSuffix(key, ".weight") {
		return "", false
	}
	return strings.TrimSuffix(key, ".weight"), true
}

func rocmLoRAFuseInferMLXAffine(base, scale rocmFuseTensorRef, groupSize int) (int, []uint64, error) {
	if len(base.Shape) != 2 {
		return 0, nil, core.NewError("rocm: MLX affine LoRA fuse requires rank-2 base tensor: " + base.Name)
	}
	if groupSize <= 0 {
		return 0, nil, core.NewError("rocm: MLX affine LoRA fuse requires positive group size")
	}
	rows := base.Shape[0]
	packedCols := base.Shape[1]
	var scaleRows uint64
	var scaleGroups uint64
	switch len(scale.Shape) {
	case 1:
		if rows == 0 || scale.Shape[0]%rows != 0 {
			return 0, nil, core.NewError("rocm: MLX affine sidecar shape does not match base rows: " + scale.Name)
		}
		scaleRows = rows
		scaleGroups = scale.Shape[0] / rows
	case 2:
		scaleRows = scale.Shape[0]
		scaleGroups = scale.Shape[1]
	default:
		return 0, nil, core.NewError("rocm: MLX affine sidecars must be rank-1 or rank-2: " + scale.Name)
	}
	if rows == 0 || packedCols == 0 || scaleRows != rows || scaleGroups == 0 {
		return 0, nil, core.NewError("rocm: MLX affine base/sidecar dimensions must be positive and row-aligned")
	}
	numerator := packedCols * 32
	denominator := scaleGroups * uint64(groupSize)
	if denominator == 0 || numerator%denominator != 0 {
		return 0, nil, core.NewError("rocm: cannot infer MLX affine bit width from base and sidecar shapes")
	}
	bits64 := numerator / denominator
	if bits64 > uint64(int(^uint(0)>>1)) {
		return 0, nil, core.NewError("rocm: MLX affine bit width is out of int range")
	}
	bits := int(bits64)
	if !hipMLXAffineSupportedBits(bits) {
		return 0, nil, core.NewError("rocm: only q4, q6, and q8 MLX affine LoRA fuse targets are supported")
	}
	denseCols := scaleGroups * uint64(groupSize)
	if denseCols > uint64(int(^uint(0)>>1)) {
		return 0, nil, core.NewError("rocm: MLX affine logical column count is out of int range")
	}
	packedCheck, err := hipMLXAffinePackedCols(int(denseCols), bits)
	if err != nil {
		return 0, nil, err
	}
	if uint64(packedCheck) != packedCols {
		return 0, nil, core.NewError("rocm: MLX affine packed column shape does not match inferred logical shape")
	}
	return bits, []uint64{rows, denseCols}, nil
}

func rocmLoRAFuseValidatePair(base rocmLoRAFuseBaseMatch, pair rocmLoRAFusePair) error {
	baseType := strings.ToUpper(base.Ref.DType)
	if (!base.Quantized && baseType != "F32") || strings.ToUpper(pair.A.DType) != "F32" || strings.ToUpper(pair.B.DType) != "F32" {
		return core.NewError("rocm: dense LoRA fuse currently supports F32 adapter tensors and F32 or MLX affine base tensors")
	}
	if base.Quantized && baseType != "U32" {
		return core.NewError("rocm: quantized LoRA fuse requires a U32 MLX affine base tensor")
	}
	if len(base.DenseShape) != 2 || len(pair.A.Shape) != 2 || len(pair.B.Shape) != 2 {
		return core.NewError("rocm: dense LoRA fuse requires rank-2 base, A, and B tensors")
	}
	if base.Quantized && base.ScaleKey == "" {
		return core.NewError("rocm: quantized LoRA fuse requires MLX affine scale sidecar")
	}
	if base.Quantized && base.BiasKey != "" && !sameUint64Shape(base.Scale.Shape, base.Bias.Shape) {
		return core.NewError("rocm: MLX affine scale and bias sidecar shapes must match")
	}
	outRows, inCols := base.DenseShape[0], base.DenseShape[1]
	rank, aCols := pair.A.Shape[0], pair.A.Shape[1]
	bRows, bRank := pair.B.Shape[0], pair.B.Shape[1]
	if rank == 0 || outRows == 0 || inCols == 0 {
		return core.NewError("rocm: dense LoRA fuse tensor dimensions must be positive")
	}
	if aCols != inCols || bRows != outRows || bRank != rank {
		return core.NewError("rocm: LoRA tensor shapes do not match base weight")
	}
	return nil
}

func rocmLoRAFusePairForBaseKey(pairs map[string]rocmLoRAFusePair, pairBaseMatches map[string]rocmLoRAFuseBaseMatch, baseKey string) (string, rocmLoRAFuseBaseMatch, rocmLoRAFusePair, bool) {
	for pairName, match := range pairBaseMatches {
		if match.Key == baseKey {
			return pairName, match, pairs[pairName], true
		}
	}
	return "", rocmLoRAFuseBaseMatch{}, rocmLoRAFusePair{}, false
}

func rocmLoRAFuseMergedF32(base rocmLoRAFuseBaseMatch, pair rocmLoRAFusePair, scale float32) ([]byte, error) {
	baseValues, err := rocmReadFuseBaseTensorF32(base)
	if err != nil {
		return nil, err
	}
	aValues, err := rocmReadFuseTensorF32(pair.A)
	if err != nil {
		return nil, err
	}
	bValues, err := rocmReadFuseTensorF32(pair.B)
	if err != nil {
		return nil, err
	}
	rows, cols, rank := int(base.DenseShape[0]), int(base.DenseShape[1]), int(pair.A.Shape[0])
	out := make([]byte, len(baseValues)*4)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			sum := float32(0)
			for k := 0; k < rank; k++ {
				sum += bValues[row*rank+k] * aValues[k*cols+col]
			}
			value := baseValues[row*cols+col] + sum*scale
			binary.LittleEndian.PutUint32(out[(row*cols+col)*4:], math.Float32bits(value))
		}
	}
	return out, nil
}

func rocmReadFuseBaseTensorF32(base rocmLoRAFuseBaseMatch) ([]float32, error) {
	if !base.Quantized {
		return rocmReadFuseTensorF32(base.Ref)
	}
	return rocmReadFuseMLXAffineTensorF32(base)
}

func rocmReadFuseMLXAffineTensorF32(base rocmLoRAFuseBaseMatch) ([]float32, error) {
	weights, err := rocmReadFuseTensorU32(base.Ref)
	if err != nil {
		return nil, err
	}
	scales, err := rocmReadFuseTensorFloat32(base.Scale)
	if err != nil {
		return nil, err
	}
	var biases []float32
	if base.BiasKey != "" {
		biases, err = rocmReadFuseTensorFloat32(base.Bias)
		if err != nil {
			return nil, err
		}
	} else {
		biases = make([]float32, len(scales))
	}
	rows, cols := int(base.DenseShape[0]), int(base.DenseShape[1])
	packedPerRow := int(base.Ref.Shape[1])
	if base.GroupSize <= 0 || cols%base.GroupSize != 0 {
		return nil, core.NewError("rocm: MLX affine logical columns must divide group size")
	}
	groupsPerRow := cols / base.GroupSize
	groupCount := rows * groupsPerRow
	if len(scales) != groupCount || len(biases) != groupCount {
		return nil, core.NewError("rocm: MLX affine scale/bias length does not match inferred groups")
	}
	if len(weights) != rows*packedPerRow {
		return nil, core.NewError("rocm: MLX affine packed weight length does not match inferred shape")
	}
	out := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		rowWeights := weights[row*packedPerRow : (row+1)*packedPerRow]
		for col := 0; col < cols; col++ {
			quantized, err := hipMLXAffineUnpackValue(rowWeights, col, base.Bits)
			if err != nil {
				return nil, err
			}
			group := row*groupsPerRow + col/base.GroupSize
			out[row*cols+col] = float32(quantized)*scales[group] + biases[group]
		}
	}
	return out, nil
}

func rocmReadFuseSafetensorsIndex(path string) (map[string]rocmFuseTensorRef, error) {
	tensors, err := readROCmSafetensorsNativeTensors(path)
	if err != nil {
		return nil, err
	}
	index := make(map[string]rocmFuseTensorRef, len(tensors))
	for _, tensor := range tensors {
		index[tensor.Name] = rocmFuseTensorRef{
			Name:      tensor.Name,
			Path:      tensor.SourcePath,
			DType:     strings.ToUpper(tensor.TypeName),
			Shape:     cloneUint64Slice(tensor.Dimensions),
			DataStart: tensor.DataOffset + int64(tensor.Offset),
			ByteLen:   tensor.ByteSize,
		}
	}
	return index, nil
}

func rocmReadFuseTensorRaw(ref rocmFuseTensorRef) ([]byte, error) {
	file, err := os.Open(ref.Path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	raw := make([]byte, int(ref.ByteLen))
	n, err := file.ReadAt(raw, ref.DataStart)
	if err != nil && !(errors.Is(err, io.EOF) && n == len(raw)) {
		return nil, err
	}
	if n != len(raw) {
		return nil, core.NewError("rocm: safetensors tensor payload is truncated: " + ref.Name)
	}
	return raw, nil
}

func rocmReadFuseTensorF32(ref rocmFuseTensorRef) ([]float32, error) {
	if strings.ToUpper(ref.DType) != "F32" {
		return nil, core.NewError("rocm: dense LoRA fuse currently supports F32 safetensors tensors only")
	}
	raw, err := rocmReadFuseTensorRaw(ref)
	if err != nil {
		return nil, err
	}
	if len(raw)%4 != 0 {
		return nil, core.NewError("rocm: F32 safetensors payload length is invalid: " + ref.Name)
	}
	values := make([]float32, len(raw)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values, nil
}

func rocmReadFuseTensorU32(ref rocmFuseTensorRef) ([]uint32, error) {
	if strings.ToUpper(ref.DType) != "U32" {
		return nil, core.NewError("rocm: MLX affine LoRA fuse requires U32 safetensors tensor: " + ref.Name)
	}
	raw, err := rocmReadFuseTensorRaw(ref)
	if err != nil {
		return nil, err
	}
	if len(raw)%4 != 0 {
		return nil, core.NewError("rocm: U32 safetensors payload length is invalid: " + ref.Name)
	}
	values := make([]uint32, len(raw)/4)
	for i := range values {
		values[i] = binary.LittleEndian.Uint32(raw[i*4:])
	}
	return values, nil
}

func rocmReadFuseTensorFloat32(ref rocmFuseTensorRef) ([]float32, error) {
	switch strings.ToUpper(ref.DType) {
	case "F32":
		return rocmReadFuseTensorF32(ref)
	case "BF16":
		raw, err := rocmReadFuseTensorRaw(ref)
		if err != nil {
			return nil, err
		}
		if len(raw)%2 != 0 {
			return nil, core.NewError("rocm: BF16 safetensors payload length is invalid: " + ref.Name)
		}
		values := make([]float32, len(raw)/2)
		for i := range values {
			values[i] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return values, nil
	case "F16":
		raw, err := rocmReadFuseTensorRaw(ref)
		if err != nil {
			return nil, err
		}
		if len(raw)%2 != 0 {
			return nil, core.NewError("rocm: F16 safetensors payload length is invalid: " + ref.Name)
		}
		values := make([]float32, len(raw)/2)
		for i := range values {
			values[i] = hipFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return values, nil
	default:
		return nil, core.NewError("rocm: MLX affine sidecar dtype must be BF16, F16, or F32: " + ref.Name)
	}
}

func rocmWriteFuseSafetensors(path string, tensors []rocmFuseWriteTensor) error {
	if len(tensors) == 0 {
		return core.NewError("rocm: safetensors write requires at least one tensor")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	names := make([]string, 0, len(tensors))
	byName := make(map[string]rocmFuseWriteTensor, len(tensors))
	for _, tensor := range tensors {
		if strings.TrimSpace(tensor.Name) == "" {
			return core.NewError("rocm: safetensors tensor name is required")
		}
		if _, ok := byName[tensor.Name]; ok {
			return core.NewError("rocm: duplicate safetensors tensor: " + tensor.Name)
		}
		byName[tensor.Name] = tensor
		names = append(names, tensor.Name)
	}
	slices.Sort(names)

	header := make(map[string]rocmSafetensorsTensor, len(names))
	payloads := make([][]byte, 0, len(names))
	offset := uint64(0)
	for _, name := range names {
		tensor := byName[name]
		dtypeBytes, ok := rocmSafetensorsDTypeBytes(tensor.DType)
		if !ok {
			return core.NewError("rocm: unsupported safetensors dtype: " + tensor.DType)
		}
		shapeBytes, err := rocmSafetensorsShapeBytes(tensor.Shape, dtypeBytes)
		if err != nil {
			return err
		}
		if shapeBytes != uint64(len(tensor.Data)) {
			return core.NewError("rocm: safetensors tensor byte length does not match shape: " + name)
		}
		header[name] = rocmSafetensorsTensor{
			DType:       strings.ToUpper(tensor.DType),
			Shape:       cloneUint64Slice(tensor.Shape),
			DataOffsets: []uint64{offset, offset + uint64(len(tensor.Data))},
		}
		payloads = append(payloads, tensor.Data)
		offset += uint64(len(tensor.Data))
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	var headerLen [8]byte
	binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerBytes)))
	if _, err := file.Write(headerLen[:]); err != nil {
		return err
	}
	if _, err := file.Write(headerBytes); err != nil {
		return err
	}
	for _, payload := range payloads {
		if _, err := file.Write(payload); err != nil {
			return err
		}
	}
	return nil
}

func rocmLoRAFuseCopyModelPackMetadata(sourceRoot, outputRoot string) error {
	patterns := []string{"*.json", "*.model", "*.txt"}
	seen := map[string]struct{}{}
	for _, pattern := range patterns {
		matches, err := filepath.Glob(filepath.Join(sourceRoot, pattern))
		if err != nil {
			return err
		}
		slices.Sort(matches)
		for _, sourcePath := range matches {
			name := filepath.Base(sourcePath)
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			if rocmLoRAFuseSkipMetadataFile(name) {
				continue
			}
			if err := copyFile(sourcePath, filepath.Join(outputRoot, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

func rocmLoRAFuseSkipMetadataFile(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".safetensors.index.json") ||
		lower == LoRAFuseProvenanceFile
}

func copyFile(sourcePath, destPath string) error {
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}
	return os.WriteFile(destPath, data, 0o644)
}

func rocmWriteLoRAFuseProvenance(path string, provenance LoRAFuseProvenance) error {
	data, err := json.MarshalIndent(provenance, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func rocmLoRAFuseOutputWeightFileNames(paths []string) []string {
	names := make([]string, 0, len(paths))
	for _, path := range paths {
		names = append(names, filepath.Base(path))
	}
	return names
}

func rocmLoRAFuseEnsureEmptyWeightDestination(outputPath string) error {
	for _, pattern := range []string{"*.safetensors", "*.gguf"} {
		matches, err := filepath.Glob(filepath.Join(outputPath, pattern))
		if err != nil {
			return err
		}
		if len(matches) > 0 {
			return core.NewError("rocm: fused output path already contains model weights")
		}
	}
	return nil
}

func rocmLoRAFuseLooksLikeWeightFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".safetensors" || ext == ".gguf"
}

func sameFilesystemPath(a, b string) bool {
	if a == b {
		return true
	}
	absA, errA := filepath.Abs(a)
	absB, errB := filepath.Abs(b)
	return errA == nil && errB == nil && absA == absB
}

func cloneUint64Slice(values []uint64) []uint64 {
	if len(values) == 0 {
		return nil
	}
	return append([]uint64(nil), values...)
}

func sameUint64Shape(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
