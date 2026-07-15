// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// CompareStatus classifies one tensor when comparing a base model pack
// against a fine-tuned pack.
type CompareStatus string

const (
	CompareStatusChanged        CompareStatus = "changed"
	CompareStatusUnchanged      CompareStatus = "unchanged"
	CompareStatusMissingInTuned CompareStatus = "missing_in_fine_tuned"
	CompareStatusExtraInTuned   CompareStatus = "extra_in_fine_tuned"
	CompareStatusShapeMismatch  CompareStatus = "shape_mismatch"
	CompareStatusDTypeMismatch  CompareStatus = "dtype_mismatch"
)

// CompareOptions configures a safetensors weight comparison.
type CompareOptions struct {
	Base             Source            `json:"base"`
	FineTuned        Source            `json:"fine_tuned"`
	IncludeUnchanged bool              `json:"include_unchanged,omitempty"`
	MaxTensorReports int               `json:"max_tensor_reports,omitempty"`
	Labels           map[string]string `json:"labels,omitempty"`
}

// TensorDelta reports per-tensor distance statistics between base and
// fine-tuned weights.
type TensorDelta struct {
	Name           string        `json:"name"`
	Status         CompareStatus `json:"status"`
	BaseDType      string        `json:"base_dtype,omitempty"`
	FineTunedDType string        `json:"fine_tuned_dtype,omitempty"`
	Shape          []int         `json:"shape,omitempty"`
	BaseShape      []int         `json:"base_shape,omitempty"`
	FineTunedShape []int         `json:"fine_tuned_shape,omitempty"`
	Elements       int           `json:"elements,omitempty"`
	MeanAbsDelta   float64       `json:"mean_abs_delta,omitempty"`
	RMSDelta       float64       `json:"rms_delta,omitempty"`
	MaxAbsDelta    float64       `json:"max_abs_delta,omitempty"`
	L2Delta        float64       `json:"l2_delta,omitempty"`
	Cosine         float64       `json:"cosine,omitempty"`
}

// CompareResult summarises base/fine-tuned tensor differences without
// loading either model through an inference engine.
type CompareResult struct {
	Base               Source            `json:"base"`
	FineTuned          Source            `json:"fine_tuned"`
	TensorCount        int               `json:"tensor_count"`
	ComparedTensors    int               `json:"compared_tensors"`
	ChangedTensors     int               `json:"changed_tensors"`
	UnchangedTensors   int               `json:"unchanged_tensors"`
	MissingInFineTuned int               `json:"missing_in_fine_tuned"`
	ExtraInFineTuned   int               `json:"extra_in_fine_tuned"`
	ShapeMismatches    int               `json:"shape_mismatches"`
	DTypeMismatches    int               `json:"dtype_mismatches"`
	ElementsCompared   int               `json:"elements_compared"`
	MeanAbsDelta       float64           `json:"mean_abs_delta,omitempty"`
	RMSDelta           float64           `json:"rms_delta,omitempty"`
	MaxAbsDelta        float64           `json:"max_abs_delta,omitempty"`
	Tensors            []TensorDelta     `json:"tensors,omitempty"`
	Labels             map[string]string `json:"labels,omitempty"`
}

// ComparePacks compares safetensors weights in a base model pack against a
// fine-tuned pack and returns aggregate plus per-tensor delta metrics.
//
//	result, err := merge.ComparePacks(ctx, merge.CompareOptions{Base: base, FineTuned: tuned})
//	if err != nil { return err }
//	core.Println(result.ChangedTensors)
func ComparePacks(ctx context.Context, opts CompareOptions) (*CompareResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateComparePack("base", opts.Base); err != nil {
		return nil, err
	}
	if err := validateComparePack("fine-tuned", opts.FineTuned); err != nil {
		return nil, err
	}
	baseIndex, err := indexWeightFiles(opts.Base.WeightFiles)
	if err != nil {
		return nil, core.E("ComparePacks", "index base weights", err)
	}
	tunedIndex, err := indexWeightFiles(opts.FineTuned.WeightFiles)
	if err != nil {
		return nil, core.E("ComparePacks", "index fine-tuned weights", err)
	}

	// Pre-size result.Tensors: it grows to at most len(baseIndex.Names)
	// entries (every base tensor either appears in tuned or not).
	expectedTensors := len(baseIndex.Names)
	if opts.MaxTensorReports > 0 && opts.MaxTensorReports < expectedTensors {
		expectedTensors = opts.MaxTensorReports
	}
	result := &CompareResult{
		Base:      opts.Base,
		FineTuned: opts.FineTuned,
		Labels:    cloneCompareLabels(opts.Labels),
		Tensors:   make([]TensorDelta, 0, expectedTensors),
	}

	acc := compareAccumulator{}
	for _, name := range baseIndex.Names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		baseEntry := baseIndex.Tensors[name]
		tunedEntry, ok := tunedIndex.Tensors[name]
		if !ok {
			result.MissingInFineTuned++
			appendTensorDelta(result, opts, TensorDelta{
				Name:      name,
				Status:    CompareStatusMissingInTuned,
				BaseDType: baseEntry.DType,
				BaseShape: cloneIntSlice(baseEntry.Shape),
				Elements:  shapeElements(baseEntry.Shape),
			})
			continue
		}
		delta, err := compareTensorEntries(name, baseEntry, tunedEntry)
		if err != nil {
			return nil, core.E("ComparePacks", "compare tensor "+name, err)
		}
		recordTensorDelta(result, &acc, opts, delta)
	}
	// Walk tunedIndex.Names once and consult baseIndex.Tensors to detect
	// extras.
	for _, name := range tunedIndex.Names {
		if _, ok := baseIndex.Tensors[name]; ok {
			continue
		}
		tunedEntry := tunedIndex.Tensors[name]
		result.ExtraInFineTuned++
		appendTensorDelta(result, opts, TensorDelta{
			Name:           name,
			Status:         CompareStatusExtraInTuned,
			FineTunedDType: tunedEntry.DType,
			FineTunedShape: cloneIntSlice(tunedEntry.Shape),
			Elements:       shapeElements(tunedEntry.Shape),
		})
	}
	result.TensorCount = result.ComparedTensors + result.MissingInFineTuned + result.ExtraInFineTuned + result.ShapeMismatches + result.DTypeMismatches
	if acc.elements > 0 {
		result.ElementsCompared = acc.elements
		result.MeanAbsDelta = acc.sumAbs / float64(acc.elements)
		result.RMSDelta = math.Sqrt(acc.sumSq / float64(acc.elements))
		result.MaxAbsDelta = acc.maxAbs
	}
	return result, nil
}

type compareAccumulator struct {
	elements int
	sumAbs   float64
	sumSq    float64
	maxAbs   float64
}

func validateComparePack(label string, source Source) error {
	if source.Root == "" {
		return core.NewError("merge: " + label + " model pack root is required")
	}
	if len(source.WeightFiles) == 0 {
		return core.NewError("merge: " + label + " model comparison requires weight files")
	}
	return nil
}

// compareTensorEntries decodes base and tuned to float32 (when shape and
// dtype agree) and computes per-tensor distance statistics.
func compareTensorEntries(name string, base, tuned tensorEntry) (TensorDelta, error) {
	shapeMatch := core.SliceEqual(base.Shape, tuned.Shape)
	baseShapeClone := cloneIntSlice(base.Shape)
	tunedShapeClone := cloneIntSlice(tuned.Shape)
	delta := TensorDelta{
		Name:           name,
		BaseDType:      base.DType,
		FineTunedDType: tuned.DType,
		BaseShape:      baseShapeClone,
		FineTunedShape: tunedShapeClone,
		Elements:       shapeElements(base.Shape),
	}
	if !shapeMatch {
		delta.Status = CompareStatusShapeMismatch
		return delta, nil
	}
	// Reuse the base-shape clone for Shape — same array, and TensorDelta
	// does not mutate either field.
	delta.Shape = baseShapeClone
	if base.DType != tuned.DType {
		delta.Status = CompareStatusDTypeMismatch
		return delta, nil
	}

	baseValues, err := safetensors.DecodeFloat32(base.DType, base.Raw, delta.Elements)
	if err != nil {
		return TensorDelta{}, err
	}
	tunedValues, err := safetensors.DecodeFloat32(tuned.DType, tuned.Raw, delta.Elements)
	if err != nil {
		return TensorDelta{}, err
	}

	var sumAbs, sumSq, maxAbs, dot, baseNorm, tunedNorm float64
	for i := range baseValues {
		baseValue := float64(baseValues[i])
		tunedValue := float64(tunedValues[i])
		diff := tunedValue - baseValue
		abs := diff
		if abs < 0 {
			abs = -abs
		}
		sumAbs += abs
		sumSq += diff * diff
		if abs > maxAbs {
			maxAbs = abs
		}
		dot += baseValue * tunedValue
		baseNorm += baseValue * baseValue
		tunedNorm += tunedValue * tunedValue
	}
	delta.MeanAbsDelta = sumAbs / float64(delta.Elements)
	delta.RMSDelta = math.Sqrt(sumSq / float64(delta.Elements))
	delta.MaxAbsDelta = maxAbs
	delta.L2Delta = math.Sqrt(sumSq)
	delta.Cosine = compareCosine(dot, baseNorm, tunedNorm)
	if maxAbs == 0 {
		delta.Status = CompareStatusUnchanged
	} else {
		delta.Status = CompareStatusChanged
	}
	return delta, nil
}

func recordTensorDelta(result *CompareResult, acc *compareAccumulator, opts CompareOptions, delta TensorDelta) {
	switch delta.Status {
	case CompareStatusChanged:
		result.ComparedTensors++
		result.ChangedTensors++
		acc.elements += delta.Elements
		acc.sumAbs += delta.MeanAbsDelta * float64(delta.Elements)
		acc.sumSq += delta.RMSDelta * delta.RMSDelta * float64(delta.Elements)
		if delta.MaxAbsDelta > acc.maxAbs {
			acc.maxAbs = delta.MaxAbsDelta
		}
	case CompareStatusUnchanged:
		result.ComparedTensors++
		result.UnchangedTensors++
		acc.elements += delta.Elements
	case CompareStatusShapeMismatch:
		result.ShapeMismatches++
	case CompareStatusDTypeMismatch:
		result.DTypeMismatches++
	}
	appendTensorDelta(result, opts, delta)
}

func appendTensorDelta(result *CompareResult, opts CompareOptions, delta TensorDelta) {
	if delta.Status == CompareStatusUnchanged && !opts.IncludeUnchanged {
		return
	}
	if opts.MaxTensorReports > 0 && len(result.Tensors) >= opts.MaxTensorReports {
		return
	}
	result.Tensors = append(result.Tensors, delta)
}

func compareCosine(dot, baseNorm, tunedNorm float64) float64 {
	switch {
	case baseNorm == 0 && tunedNorm == 0:
		return 1
	case baseNorm == 0 || tunedNorm == 0:
		return 0
	default:
		return clampFloat64(dot/(math.Sqrt(baseNorm)*math.Sqrt(tunedNorm)), -1, 1)
	}
}

func cloneCompareLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	return core.MapClone(labels)
}

func cloneIntSlice(values []int) []int {
	if len(values) == 0 {
		return nil
	}
	return core.SliceClone(values)
}
