// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const rocmModelSliceManifestVersion = "go-rocm.model-slice.v1"

var (
	errROCmModelSliceOutputPathRequired   = core.NewError("rocm: model slice output path is required")
	errROCmModelSliceSourcePathRequired   = core.NewError("rocm: model slice source path is required")
	errROCmModelSliceUnsupportedFormat    = core.NewError("rocm: model slice materialisation currently supports safetensors packs only")
	errROCmModelSliceNoSafetensorsWeights = core.NewError("rocm: model slice source has no safetensors weights")
	errROCmModelSliceNoTensorsSelected    = core.NewError("rocm: model slice selected no tensors")
)

type rocmModelSliceManifest struct {
	Version   string                   `json:"version"`
	Source    string                   `json:"source"`
	Output    string                   `json:"output"`
	Plan      inference.ModelSlicePlan `json:"plan"`
	Weight    string                   `json:"weight"`
	Tensors   []string                 `json:"tensors"`
	Labels    map[string]string        `json:"labels,omitempty"`
	WeightMap map[string]string        `json:"weight_map,omitempty"`
}

type rocmModelSliceTensorRef struct {
	Name      string
	Path      string
	DType     string
	Shape     []uint64
	DataStart int64
	ByteLen   uint64
}

// PlanModelSlice expands a portable model-slice preset through the shared
// go-inference split contract.
func PlanModelSlice(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	return (&rocmBackend{}).PlanModelSlice(ctx, req)
}

// SliceModel materialises a safetensors subset for split/reload tests.
func SliceModel(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	return (&rocmBackend{}).SliceModel(ctx, req)
}

func (b *rocmBackend) PlanModelSlice(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	plan, err := inference.PlanModelSlice(req)
	if err != nil {
		return nil, err
	}
	plan.Model = req.Model
	plan.Adapter = req.Adapter
	plan.SourcePath = req.Model.Path
	plan.OutputPath = req.OutputPath
	if plan.Labels == nil {
		plan.Labels = map[string]string{}
	}
	for key, value := range req.Labels {
		plan.Labels[key] = value
	}
	plan.Labels["backend"] = "rocm"
	plan.Labels["cli_contract"] = "reactive-inference-v1"
	plan.Labels["slice_runtime"] = "native_safetensors_subset"
	return &plan, nil
}

func (b *rocmBackend) SliceModel(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	plan, err := b.PlanModelSlice(ctx, req)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(req.OutputPath) == "" {
		return nil, errROCmModelSliceOutputPathRequired
	}
	if strings.TrimSpace(req.Model.Path) == "" {
		return nil, errROCmModelSliceSourcePathRequired
	}
	inspection, err := b.InspectModelPack(ctx, req.Model.Path)
	if err != nil {
		return nil, err
	}
	if inspection.Format != "safetensors" {
		return nil, errROCmModelSliceUnsupportedFormat
	}
	weightPaths, err := rocmSafetensorsWeightFiles(req.Model.Path)
	if err != nil {
		if strings.Contains(err.Error(), "at least one safetensors weight file") {
			return nil, errROCmModelSliceNoSafetensorsWeights
		}
		return nil, err
	}
	sourceRoot, err := rocmModelPackRoot(req.Model.Path)
	if err != nil {
		return nil, err
	}
	refs, names, sourceBytes, err := rocmSelectModelSliceTensorRefs(plan, weightPaths)
	if err != nil {
		return nil, err
	}
	if len(refs) == 0 {
		return nil, errROCmModelSliceNoTensorsSelected
	}
	if err := os.MkdirAll(req.OutputPath, 0o755); err != nil {
		return nil, err
	}
	if err := rocmCopyModelSliceMetadata(sourceRoot, req.OutputPath, plan); err != nil {
		return nil, err
	}
	writeTensors, selectedBytes, err := rocmReadModelSliceTensors(refs)
	if err != nil {
		return nil, err
	}
	if err := rocmWriteFuseSafetensors(filepath.Join(req.OutputPath, "model.safetensors"), writeTensors); err != nil {
		return nil, err
	}
	plan.OutputPath = req.OutputPath
	plan.SourcePath = req.Model.Path
	plan.Model = inspection.Model
	if plan.Model.Path == "" {
		plan.Model.Path = req.Model.Path
	}
	if plan.Labels == nil {
		plan.Labels = map[string]string{}
	}
	plan.Labels["tensor_count"] = strconv.Itoa(len(refs))
	plan.Labels["weight_file"] = "model.safetensors"
	plan.Labels["source_weight_files"] = strconv.Itoa(len(weightPaths))
	plan.Labels["selected_tensor_bytes"] = strconv.FormatInt(selectedBytes, 10)
	plan.Labels["source_tensor_bytes"] = strconv.FormatInt(sourceBytes, 10)
	if sourceBytes > 0 {
		plan.Labels["retained_tensor_ratio"] = strconv.FormatFloat(float64(selectedBytes)/float64(sourceBytes), 'f', 4, 64)
	}
	if err := rocmWriteModelSliceManifest(req.OutputPath, plan, names); err != nil {
		return nil, err
	}
	return plan, nil
}

func rocmSelectModelSliceTensorRefs(plan *inference.ModelSlicePlan, weightPaths []string) ([]rocmModelSliceTensorRef, []string, int64, error) {
	refs := []rocmModelSliceTensorRef{}
	names := []string{}
	var sourceBytes int64
	for _, weightPath := range weightPaths {
		tensors, err := readROCmSafetensorsNativeTensors(weightPath)
		if err != nil {
			return nil, nil, 0, err
		}
		for _, tensor := range tensors {
			sourceBytes += int64(tensor.ByteSize)
			if !rocmModelSliceIncludesTensor(plan, tensor.Name) {
				continue
			}
			refs = append(refs, rocmModelSliceTensorRef{
				Name:      tensor.Name,
				Path:      tensor.SourcePath,
				DType:     strings.ToUpper(tensor.TypeName),
				Shape:     cloneUint64Slice(tensor.Dimensions),
				DataStart: tensor.DataOffset + int64(tensor.Offset),
				ByteLen:   tensor.ByteSize,
			})
			names = append(names, tensor.Name)
		}
	}
	order := make([]int, len(refs))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		return strings.Compare(refs[a].Name, refs[b].Name)
	})
	sortedRefs := make([]rocmModelSliceTensorRef, len(refs))
	sortedNames := make([]string, len(names))
	for out, in := range order {
		sortedRefs[out] = refs[in]
		sortedNames[out] = names[in]
	}
	return sortedRefs, sortedNames, sourceBytes, nil
}

func rocmReadModelSliceTensors(refs []rocmModelSliceTensorRef) ([]rocmFuseWriteTensor, int64, error) {
	tensors := make([]rocmFuseWriteTensor, 0, len(refs))
	var selectedBytes int64
	for _, ref := range refs {
		raw, err := rocmReadModelSliceTensorRaw(ref)
		if err != nil {
			return nil, 0, err
		}
		tensors = append(tensors, rocmFuseWriteTensor{
			Name:  ref.Name,
			DType: ref.DType,
			Shape: cloneUint64Slice(ref.Shape),
			Data:  raw,
		})
		selectedBytes += int64(len(raw))
	}
	return tensors, selectedBytes, nil
}

func rocmReadModelSliceTensorRaw(ref rocmModelSliceTensorRef) ([]byte, error) {
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

func rocmModelSliceIncludesTensor(plan *inference.ModelSlicePlan, name string) bool {
	if plan == nil {
		return false
	}
	if plan.ExtractLevel == inference.ModelExtractLevelAll {
		return true
	}
	lower := strings.ToLower(name)
	switch {
	case plan.HasComponent(inference.ModelComponentAttention) && rocmModelSliceTensorIsAttention(lower):
		return true
	case plan.HasComponent(inference.ModelComponentFFN) && rocmModelSliceTensorIsFFN(lower):
		return true
	case plan.HasComponent(inference.ModelComponentNorms) && strings.Contains(lower, "norm"):
		return true
	case plan.HasComponent(inference.ModelComponentGate) && rocmModelSliceTensorIsGate(lower):
		return true
	case plan.HasComponent(inference.ModelComponentExperts) && rocmModelSliceTensorIsExpert(lower):
		return true
	case plan.HasComponent(inference.ModelComponentRouter) && rocmModelSliceTensorIsRouter(lower):
		return true
	case plan.HasComponent(inference.ModelComponentDownMeta) && (strings.Contains(lower, "down_meta") || strings.Contains(lower, "down_proj.meta")):
		return true
	case plan.HasComponent(inference.ModelComponentEmbeddings) && (strings.Contains(lower, "embed") || strings.Contains(lower, ".wte.")):
		return true
	case plan.HasComponent(inference.ModelComponentLMHead) && strings.HasPrefix(lower, "lm_head."):
		return true
	default:
		return false
	}
}

func rocmModelSliceTensorIsAttention(name string) bool {
	return strings.Contains(name, "self_attn") ||
		strings.Contains(name, "attention") ||
		strings.Contains(name, ".attn.") ||
		rocmModelSliceHasProjection(name, "q_proj") ||
		rocmModelSliceHasProjection(name, "k_proj") ||
		rocmModelSliceHasProjection(name, "v_proj") ||
		rocmModelSliceHasProjection(name, "o_proj") ||
		rocmModelSliceHasProjection(name, "out_proj")
}

func rocmModelSliceTensorIsFFN(name string) bool {
	return strings.Contains(name, ".mlp.") ||
		strings.Contains(name, "feed_forward") ||
		strings.Contains(name, "ffn") ||
		rocmModelSliceHasProjection(name, "up_proj") ||
		rocmModelSliceHasProjection(name, "down_proj")
}

func rocmModelSliceTensorIsGate(name string) bool {
	return strings.Contains(name, ".gate.") || rocmModelSliceHasProjection(name, "gate_proj")
}

func rocmModelSliceTensorIsRouter(name string) bool {
	return strings.Contains(name, "router") || strings.Contains(name, "gate_score") || strings.HasSuffix(name, ".gate.weight")
}

func rocmModelSliceTensorIsExpert(name string) bool {
	return strings.Contains(name, "experts") || strings.Contains(name, ".expert.")
}

func rocmModelSliceHasProjection(name, projection string) bool {
	return strings.Contains(name, "."+projection+".") || strings.HasSuffix(name, "."+projection+".weight")
}

func rocmCopyModelSliceMetadata(sourceRoot, outputRoot string, plan *inference.ModelSlicePlan) error {
	for _, name := range rocmModelSliceMetadataFiles(plan) {
		sourcePath := filepath.Join(sourceRoot, name)
		if _, err := os.Stat(sourcePath); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return err
		}
		if err := copyFile(sourcePath, filepath.Join(outputRoot, name)); err != nil {
			return err
		}
	}
	return nil
}

func rocmModelSliceMetadataFiles(plan *inference.ModelSlicePlan) []string {
	files := []string{"config.json"}
	if plan == nil {
		return files
	}
	if plan.HasComponent(inference.ModelComponentTokenizer) {
		files = append(files, "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json")
	}
	if plan.HasComponent(inference.ModelComponentLabels) {
		files = append(files, "label_map.json", "labels.json", "id2label.json")
	}
	return files
}

func rocmWriteModelSliceManifest(outputRoot string, plan *inference.ModelSlicePlan, tensors []string) error {
	manifest := rocmModelSliceManifest{
		Version:   rocmModelSliceManifestVersion,
		Source:    plan.SourcePath,
		Output:    plan.OutputPath,
		Plan:      *plan,
		Weight:    "model.safetensors",
		Tensors:   append([]string(nil), tensors...),
		Labels:    plan.Labels,
		WeightMap: map[string]string{"model.safetensors": "selected tensors"},
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outputRoot, "slice_manifest.json"), data, 0o644)
}
