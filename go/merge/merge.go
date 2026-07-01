// SPDX-Licence-Identifier: EUPL-1.2

// Package merge combines or compares local safetensors model packs without
// loading either model into an inference engine — pure safetensors byte and
// metadata manipulation, shared across go-mlx, go-rocm, and go-cpu so each
// engine does not re-implement pack-level merge/compare and layers its own
// engine-specific validation (e.g. an MLX ValidateModelPack) on top by
// producing a Source per participating pack.
//
// Packs combines N (linear) or exactly 2 (SLERP) compatible packs into a
// new safetensors pack plus a provenance record. ComparePacks diffs a base
// pack against a fine-tuned pack tensor-by-tensor (see compare.go).
//
// Every source weight file is read fully into memory (via modelmgmt.
// ReadSafetensors) rather than streamed — the right tradeoff for the pack
// sizes go-inference callers merge today (LoRA adapters, small-to-mid
// safetensors exports); a chunked/streaming variant for multi-GB sharded
// checkpoints is future work, not ported here.
//
//	result, err := merge.Packs(ctx, merge.Options{
//	    Sources: []merge.Source{
//	        {Root: "/models/base", Architecture: "gemma3", TokenizerPath: "/models/base/tokenizer.json", WeightFiles: []string{"/models/base/model.safetensors"}},
//	        {Root: "/models/tuned", Architecture: "gemma3", TokenizerPath: "/models/tuned/tokenizer.json", WeightFiles: []string{"/models/tuned/model.safetensors"}},
//	    },
//	    OutputPath: "/models/merged",
//	    Method:     merge.MethodLinear,
//	})
//	if err != nil { return err }
package merge

import (
	"context"

	core "dappco.re/go"
)

// Method names the tensor merge algorithm.
type Method string

const (
	MethodLinear Method = "linear"
	MethodSLERP  Method = "slerp"

	// ProvenanceFile is the filename Packs writes its Provenance record to,
	// inside OutputPath.
	ProvenanceFile = "model_merge_provenance.json"

	// outputWeightsFile is the filename Packs writes the merged tensors to,
	// inside OutputPath. Always a single file — merge output is never
	// re-sharded, regardless of how many shards a source contributed.
	outputWeightsFile = "model.safetensors"
)

// Constant validation errors hoisted to package vars — shared instances
// make errors.Is comparable for callers distinguishing failure modes
// without parsing message text, and avoid a fresh allocation per failure.
var (
	errSLERPLenMismatch        = core.NewError("merge: tensor length mismatch during SLERP merge")
	errSLERPNeedTwoTensors     = core.NewError("merge: SLERP tensor merge requires exactly two tensors")
	errLinearLenMismatch       = core.NewError("merge: tensor length mismatch during linear merge")
	errNoTensors               = core.NewError("merge: no tensors to merge")
	errOutputHasWeights        = core.NewError("merge: merged output path already contains model weights")
	errWeightsSourceCount      = core.NewError("merge: tensor merge weights do not match source count")
	errTokenizerMismatch       = core.NewError("merge: model merge tokenizer mismatch")
	errMergeTOutOfRange        = core.NewError("merge: model merge t must be between 0 and 1")
	errMergeWeightsSumZero     = core.NewError("merge: model merge source weights sum to zero")
	errMergeWeightNotFinite    = core.NewError("merge: model merge source weight must be finite")
	errMergeSourceRootRequired = core.NewError("merge: model merge source root is required")
	errMergeNeedTwoSources     = core.NewError("merge: model merge requires at least two sources")
	errMergeNeedsSafetensors   = core.NewError("merge: model merge requires one or more safetensors source weight files")
	errSLERPNeedTwoSources     = core.NewError("merge: SLERP model merge requires exactly two sources")
	errOutputSameAsSource      = core.NewError("merge: merged output path must differ from source model path")
	errOutputNotPackDir        = core.NewError("merge: merged output path must be a model-pack directory")
	errOutputPathRequired      = core.NewError("merge: merged model output path is required")
	errCoreResultFailed        = core.NewError("core result failed")
)

// Source identifies a local safetensors model pack participating in a merge
// or comparison. go-inference does not validate model-pack structure itself
// — each engine owns its own pack inspector/validator and maps the result
// onto a Source before calling Packs or ComparePacks.
type Source struct {
	// Root is the model pack's directory. Used to resolve/copy sibling
	// metadata (config.json, tokenizer files, chat templates) into the
	// merged output, and to detect an output path that collides with a
	// source.
	Root string `json:"root"`

	// Architecture is compared across sources unless
	// Options.AllowArchitectureMismatch is set.
	Architecture string `json:"architecture,omitempty"`

	// TokenizerPath is hashed and compared across sources unless
	// Options.AllowTokenizerMismatch is set.
	TokenizerPath string `json:"tokenizer_path,omitempty"`

	// WeightFiles lists the pack's safetensors shard paths — exactly one
	// entry for a single-file pack, more for a sharded export. Every entry
	// must end in ".safetensors".
	WeightFiles []string `json:"weight_files"`

	// Weight is this source's contribution to a linear merge (ignored by
	// SLERP, which always uses Options.T). Zero on every source means
	// "equal split" — see normalizedWeights.
	Weight float64 `json:"weight,omitempty"`
}

// Options configures a local model-pack tensor merge.
type Options struct {
	Sources                   []Source          `json:"sources"`
	OutputPath                string            `json:"output_path"`
	Method                    Method            `json:"method,omitempty"`
	T                         float64           `json:"t,omitempty"`
	AllowArchitectureMismatch bool              `json:"allow_architecture_mismatch,omitempty"`
	AllowTokenizerMismatch    bool              `json:"allow_tokenizer_mismatch,omitempty"`
	AllowTensorMismatch       bool              `json:"allow_tensor_mismatch,omitempty"`
	Labels                    map[string]string `json:"labels,omitempty"`
}

// Result reports the paths of the generated merged model pack and its
// per-tensor counts.
type Result struct {
	OutputPath     string   `json:"output_path"`
	WeightPath     string   `json:"weight_path"`
	ProvenancePath string   `json:"provenance_path"`
	Method         Method   `json:"method"`
	T              float64  `json:"t,omitempty"`
	Sources        []Source `json:"sources"`
	TensorCount    int      `json:"tensor_count"`
	MergedTensors  int      `json:"merged_tensors"`
	CopiedTensors  int      `json:"copied_tensors,omitempty"`
	SkippedTensors []string `json:"skipped_tensors,omitempty"`
}

// Provenance records how a merged pack was produced. Written alongside the
// merged weights as ProvenanceFile.
type Provenance struct {
	Version        int               `json:"version"`
	Method         Method            `json:"method"`
	T              float64           `json:"t,omitempty"`
	Sources        []Source          `json:"sources"`
	OutputWeight   string            `json:"output_weight"`
	MergedTensors  int               `json:"merged_tensors"`
	CopiedTensors  int               `json:"copied_tensors,omitempty"`
	SkippedTensors []string          `json:"skipped_tensors,omitempty"`
	Labels         map[string]string `json:"labels,omitempty"`
}

type prepared struct {
	Method  Method
	T       float64
	Sources []Source
	Output  string
}

// Packs merges compatible local safetensors model packs and writes a new
// pack (merged weights + copied metadata + provenance) to Options.OutputPath.
//
//	result, err := merge.Packs(ctx, opts)
//	if err != nil { return err }
//	core.Println(result.WeightPath)
func Packs(ctx context.Context, opts Options) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prep, err := prepare(ctx, opts)
	if err != nil {
		return nil, err
	}

	indexes, err := indexSources(prep.Sources)
	if err != nil {
		return nil, err
	}
	if err := validateTensorIndexes(indexes, opts.AllowTensorMismatch); err != nil {
		return nil, err
	}

	weightPath := core.PathJoin(prep.Output, outputWeightsFile)
	merged, copied, skipped, err := writeMergedSafetensors(ctx, weightPath, indexes, prep.Method, prep.T, prep.Sources, opts.AllowTensorMismatch)
	if err != nil {
		return nil, err
	}

	provenancePath := core.PathJoin(prep.Output, ProvenanceFile)
	if err := writeProvenance(provenancePath, Provenance{
		Version:        1,
		Method:         prep.Method,
		T:              prep.T,
		Sources:        prep.Sources,
		OutputWeight:   core.PathBase(weightPath),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
		Labels:         opts.Labels,
	}); err != nil {
		return nil, err
	}

	return &Result{
		OutputPath:     prep.Output,
		WeightPath:     weightPath,
		ProvenancePath: provenancePath,
		Method:         prep.Method,
		T:              prep.T,
		Sources:        prep.Sources,
		TensorCount:    len(indexes[0].Names),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
	}, nil
}

func prepare(ctx context.Context, opts Options) (prepared, error) {
	if err := ctx.Err(); err != nil {
		return prepared{}, err
	}
	if len(opts.Sources) < 2 {
		return prepared{}, errMergeNeedTwoSources
	}
	if opts.OutputPath == "" {
		return prepared{}, errOutputPathRequired
	}
	if hasSuffixFold(opts.OutputPath, ".safetensors") || hasSuffixFold(opts.OutputPath, ".gguf") {
		return prepared{}, errOutputNotPackDir
	}

	method := opts.Method
	if method == "" {
		method = MethodLinear
	}
	switch method {
	case MethodLinear, MethodSLERP:
	default:
		return prepared{}, core.NewError("merge: unsupported model merge method: " + string(method))
	}
	if method == MethodSLERP && len(opts.Sources) != 2 {
		return prepared{}, errSLERPNeedTwoSources
	}
	if opts.T < 0 || opts.T > 1 {
		return prepared{}, errMergeTOutOfRange
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if err := ensureEmptyDestination(output); err != nil {
		return prepared{}, err
	}

	sources := make([]Source, 0, len(opts.Sources))
	for _, source := range opts.Sources {
		if source.Root == "" {
			return prepared{}, errMergeSourceRootRequired
		}
		if len(source.WeightFiles) == 0 {
			return prepared{}, errMergeNeedsSafetensors
		}
		for _, weightFile := range source.WeightFiles {
			if !hasSuffixFold(weightFile, ".safetensors") {
				return prepared{}, errMergeNeedsSafetensors
			}
		}
		if samePathResolved(source.Root, output) {
			return prepared{}, errOutputSameAsSource
		}
		sources = append(sources, source)
	}

	if err := validatePackCompatibility(sources, opts); err != nil {
		return prepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return prepared{}, core.E("Packs", "create merged model directory", resultError(result))
	}
	if err := copyModelPackMetadata(sources[0].Root, output); err != nil {
		return prepared{}, err
	}

	return prepared{
		Method:  method,
		T:       opts.T,
		Sources: sources,
		Output:  output,
	}, nil
}

func ensureEmptyDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("Packs", "inspect output path", resultError(stat))
	}
	if len(core.PathGlob(core.PathJoin(output, "*.safetensors"))) > 0 {
		return errOutputHasWeights
	}
	if len(core.PathGlob(core.PathJoin(output, "*.gguf"))) > 0 {
		return errOutputHasWeights
	}
	return nil
}

func validatePackCompatibility(sources []Source, opts Options) error {
	base := sources[0]
	// Hash the base tokenizer once, lazily — only if a non-
	// AllowTokenizerMismatch comparison actually needs it.
	var baseHash string
	var baseHashErr error
	baseHashLoaded := opts.AllowTokenizerMismatch
	for i := 1; i < len(sources); i++ {
		source := sources[i]
		if !opts.AllowArchitectureMismatch && source.Architecture != base.Architecture {
			return core.NewError(core.Concat(
				"merge: model merge architecture mismatch: ",
				base.Architecture,
				" vs ",
				source.Architecture,
			))
		}
		if opts.AllowTokenizerMismatch {
			continue
		}
		if !baseHashLoaded {
			baseHash, baseHashErr = hashFile(base.TokenizerPath)
			baseHashLoaded = true
		}
		if baseHashErr != nil {
			return core.E("Packs", "hash base tokenizer", baseHashErr)
		}
		hash, err := hashFile(source.TokenizerPath)
		if err != nil {
			return core.E("Packs", "hash tokenizer", err)
		}
		if hash != baseHash {
			return errTokenizerMismatch
		}
	}
	return nil
}

func indexSources(sources []Source) ([]sourceIndex, error) {
	indexes := make([]sourceIndex, 0, len(sources))
	for _, source := range sources {
		index, err := indexWeightFiles(source.WeightFiles)
		if err != nil {
			return nil, err
		}
		indexes = append(indexes, index)
	}
	return indexes, nil
}

func validateTensorIndexes(indexes []sourceIndex, allowMismatch bool) error {
	base := indexes[0]
	for i := 1; i < len(indexes); i++ {
		index := indexes[i]
		for _, name := range base.Names {
			ref, ok := index.Tensors[name]
			if !ok {
				if allowMismatch {
					continue
				}
				return core.NewError("merge: model merge tensor missing from source: " + name)
			}
			baseRef := base.Tensors[name]
			if !sameIntSlice(baseRef.Shape, ref.Shape) {
				if allowMismatch {
					continue
				}
				return core.NewError("merge: model merge tensor shape mismatch: " + name)
			}
		}
		if allowMismatch {
			continue
		}
		for _, name := range index.Names {
			if _, ok := base.Tensors[name]; !ok {
				return core.NewError("merge: model merge extra tensor in source: " + name)
			}
		}
	}
	return nil
}

// hasSuffixFold reports whether s ends with suffix using ASCII case
// folding. suffix is required to be lowercase.
func hasSuffixFold(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	off := len(s) - len(suffix)
	for i := 0; i < len(suffix); i++ {
		c := s[off+i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != suffix[i] {
			return false
		}
	}
	return true
}

func sameIntSlice(a, b []int) bool {
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

func clampFloat64(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errCoreResultFailed
}

// equalFold is len-prefixed ASCII case-insensitive equality. Zero allocations.
func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// containsFold reports whether s contains substr using ASCII case folding.
// substr is required to be lowercase.
func containsFold(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(substr) > len(s) {
		return false
	}
	last := len(s) - len(substr)
outer:
	for i := 0; i <= last; i++ {
		for j := 0; j < len(substr); j++ {
			c := s[i+j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != substr[j] {
				continue outer
			}
		}
		return true
	}
	return false
}
