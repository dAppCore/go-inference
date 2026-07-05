// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"sort"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// sourceIndex is the in-memory tensor set for one merge source, built by
// reading every WeightFiles entry via safetensors.ReadSafetensors and unioning
// the results. Unlike a chunked/offset-addressed index designed for multi-GB
// sharded checkpoints, sourceIndex holds each tensor's full raw bytes in
// memory — see the package doc for the tradeoff.
type sourceIndex struct {
	Names   []string
	Tensors map[string]tensorEntry
}

// tensorEntry is one tensor's dtype, shape, and raw (still-encoded) bytes.
type tensorEntry struct {
	DType string
	Shape []int
	Raw   []byte
}

// indexWeightFiles reads every safetensors file in paths and unions their
// tensors into one sourceIndex. A tensor name repeated across two files in
// the same source is an error — shards of one pack must not overlap.
func indexWeightFiles(paths []string) (sourceIndex, error) {
	index := sourceIndex{Tensors: make(map[string]tensorEntry)}
	for _, path := range paths {
		read := safetensors.ReadSafetensors(path)
		if !read.OK {
			return sourceIndex{}, core.E("Packs", "read safetensors "+path, resultError(read))
		}
		data := read.Value.(safetensors.SafetensorsData)
		for name, info := range data.Tensors {
			if _, exists := index.Tensors[name]; exists {
				return sourceIndex{}, core.NewError("merge: duplicate tensor across safetensors shards: " + name)
			}
			index.Tensors[name] = tensorEntry{
				DType: info.Dtype,
				Shape: info.Shape,
				Raw:   safetensors.GetTensorData(info, data.Data),
			}
			index.Names = append(index.Names, name)
		}
	}
	sort.Strings(index.Names)
	return index, nil
}

// shapeElements returns the element count a shape describes (the product
// of its dimensions; 1 for a scalar's empty shape).
func shapeElements(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

// writeMergedSafetensors merges every tensor named in indexes[0] (the base
// pack) across all sources and writes the result as a single safetensors
// file at path. Output dtype is always F32, matching go-mlx's merge
// convention: even a mismatch-tolerated "copied" tensor is decoded and
// re-encoded through F32 rather than passed through in its original dtype,
// so a merged pack never mixes dtypes.
func writeMergedSafetensors(ctx context.Context, path string, indexes []sourceIndex, method Method, t float64, sources []Source, allowMismatch bool) (merged int, copied int, skipped []string, err error) {
	linearWeights, err := normalizedWeights(sources)
	if err != nil {
		return 0, 0, nil, err
	}

	base := indexes[0]
	mergedInfo := make(map[string]safetensors.SafetensorsTensorInfo, len(base.Names))
	mergedData := make(map[string][]byte, len(base.Names))

	for _, name := range base.Names {
		if err := ctx.Err(); err != nil {
			return 0, 0, nil, err
		}
		entries, complete := gatherTensorEntries(indexes, name)
		baseEntry := entries[0]

		var outValues []float32
		switch {
		case complete:
			decoded, decodeErr := decodeAll(entries)
			if decodeErr != nil {
				return 0, 0, nil, decodeErr
			}
			outValues, err = mergeTensorValues(decoded, method, t, linearWeights)
			if err != nil {
				return 0, 0, nil, err
			}
			merged++
		case allowMismatch:
			outValues, err = safetensors.DecodeFloat32(baseEntry.DType, baseEntry.Raw, shapeElements(baseEntry.Shape))
			if err != nil {
				return 0, 0, nil, err
			}
			copied++
			skipped = append(skipped, name)
		default:
			return 0, 0, nil, core.NewError("merge: model merge tensor mismatch: " + name)
		}

		mergedInfo[name] = safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: baseEntry.Shape}
		mergedData[name] = safetensors.EncodeFloat32(outValues)
	}

	if result := safetensors.WriteSafetensors(path, mergedInfo, mergedData); !result.OK {
		return 0, 0, nil, core.E("Packs", "write merged safetensors", resultError(result))
	}
	return merged, copied, skipped, nil
}

// gatherTensorEntries collects name's tensor entry from every source index,
// in source order. complete is true only when every source has the tensor
// AND every shape matches the base (first) source's shape — entries[0] is
// always the base source's entry, since callers only ever look up a name
// drawn from indexes[0].Names.
func gatherTensorEntries(indexes []sourceIndex, name string) ([]tensorEntry, bool) {
	entries := make([]tensorEntry, 0, len(indexes))
	complete := true
	var shape []int
	for _, index := range indexes {
		entry, ok := index.Tensors[name]
		if !ok {
			complete = false
			continue
		}
		if shape == nil {
			shape = entry.Shape
		} else if !sameIntSlice(shape, entry.Shape) {
			complete = false
		}
		entries = append(entries, entry)
	}
	return entries, complete && len(entries) == len(indexes)
}

// decodeAll decodes every entry's raw bytes to float32 according to its own
// dtype and shape.
func decodeAll(entries []tensorEntry) ([][]float32, error) {
	values := make([][]float32, len(entries))
	for i, entry := range entries {
		decoded, err := safetensors.DecodeFloat32(entry.DType, entry.Raw, shapeElements(entry.Shape))
		if err != nil {
			return nil, err
		}
		values[i] = decoded
	}
	return values, nil
}

func mergeTensorValues(values [][]float32, method Method, t float64, weights []float64) ([]float32, error) {
	switch method {
	case MethodLinear:
		return linearMerge(values, weights)
	case MethodSLERP:
		return slerpMerge(values, t)
	default:
		return nil, core.NewError("merge: unsupported model merge method: " + string(method))
	}
}

// linearMerge computes the per-element weighted sum of values. Unlike
// go-mlx's reference (where this path is unreachable in production once the
// chunked writer takes over), this is the primary merge path here, so it
// defensively checks len(values) == len(weights) rather than trusting the
// caller.
func linearMerge(values [][]float32, weights []float64) ([]float32, error) {
	if len(values) == 0 {
		return nil, errNoTensors
	}
	if len(values) != len(weights) {
		return nil, errWeightsSourceCount
	}
	out := make([]float32, len(values[0]))
	for srcIdx, source := range values {
		if len(source) != len(out) {
			return nil, errLinearLenMismatch
		}
		// Cast the weight to float32 once outside the inner loop — linear
		// merge weights are normalised in [0,1], so float32 precision is
		// sufficient (matches the source tensor dtype anyway).
		weight32 := float32(weights[srcIdx])
		for i, value := range source {
			out[i] += value * weight32
		}
	}
	return out, nil
}

// slerpMerge spherically interpolates exactly two tensors at t. Falls back
// to a linear blend when either vector is zero or the two are nearly
// parallel/antiparallel (sin(theta) would be ~0, making the SLERP scale
// factors numerically unstable).
func slerpMerge(values [][]float32, t float64) ([]float32, error) {
	if len(values) != 2 {
		return nil, errSLERPNeedTwoTensors
	}
	a, b := values[0], values[1]
	if len(a) != len(b) {
		return nil, errSLERPLenMismatch
	}
	var dot, normA, normB float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return linearMerge(values, []float64{1 - t, t})
	}
	cosTheta := clampFloat64(dot/(math.Sqrt(normA)*math.Sqrt(normB)), -1, 1)
	if math.Abs(cosTheta) > 0.9995 {
		return linearMerge(values, []float64{1 - t, t})
	}
	theta := math.Acos(cosTheta)
	sinTheta := math.Sin(theta)
	scaleA := math.Sin((1-t)*theta) / sinTheta
	scaleB := math.Sin(t*theta) / sinTheta
	return linearMerge(values, []float64{scaleA, scaleB})
}

// normalizedWeights turns each source's raw Weight into a linear-merge
// coefficient set that sums to 1. When every source leaves Weight at its
// zero value, the split is equal across all sources.
func normalizedWeights(sources []Source) ([]float64, error) {
	weights := make([]float64, len(sources))
	var total float64
	var explicit bool
	for i, source := range sources {
		if math.IsNaN(source.Weight) || math.IsInf(source.Weight, 0) {
			return nil, errMergeWeightNotFinite
		}
		if source.Weight != 0 {
			explicit = true
		}
		weights[i] = source.Weight
		total += source.Weight
	}
	if !explicit {
		equal := 1 / float64(len(sources))
		for i := range weights {
			weights[i] = equal
		}
		return weights, nil
	}
	if total == 0 {
		return nil, errMergeWeightsSumZero
	}
	for i := range weights {
		weights[i] /= total
	}
	return weights, nil
}

// writeProvenance marshals provenance to JSON and writes it to path,
// sorting SkippedTensors for deterministic output.
func writeProvenance(path string, provenance Provenance) error {
	sorted := core.SliceClone(provenance.SkippedTensors)
	sort.Strings(sorted)
	provenance.SkippedTensors = sorted
	data := core.JSONMarshal(provenance)
	if !data.OK {
		return core.E("Packs", "marshal merge provenance", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("Packs", "write merge provenance", resultError(result))
	}
	return nil
}
