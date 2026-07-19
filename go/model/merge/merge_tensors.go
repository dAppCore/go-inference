// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"sort"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// sourceIndex is one merge source's tensor set, resolved to shard locations
// only (see tensorEntry) — never a whole shard's payload bytes, however many
// files the source spans.
type sourceIndex struct {
	Names   []string
	Tensors map[string]tensorEntry
}

// tensorEntry is one tensor's dtype, shape, and shard location. Ref carries
// the owning shard's file path plus byte offsets, so a ShardCache-bound
// reader can stream the tensor's payload on demand — no tensor payload is
// read at index-build time.
type tensorEntry struct {
	DType string
	Shape []int
	Ref   safetensors.TensorRef
}

// indexWeightFiles resolves every tensor name across paths (one merge
// source's shard set — a single safetensors file, or the N shards of one
// sharded checkpoint) to its owning shard file and byte offsets, via
// safetensors.IndexFiles: a header-only walk across every shard (no tensor
// payload read) that is the same shard-resolution primitive model/quant's
// snapshot converters (fp8, gptq, awq, mlxaffine, nf4, autoround) already
// use for their own sharded sources. Tensor payloads are streamed from
// their shard on demand by decodeAll/writeMergedSafetensors, through a
// ShardCache shared across the whole merge — so a multi-GB sharded source
// is never resident beyond the one tensor currently being merged. A tensor
// name repeated across two files in the same source is an error (surfaced
// by IndexFiles): shards of one pack must not overlap.
func indexWeightFiles(paths []string) (sourceIndex, error) {
	idx, err := safetensors.IndexFiles(paths)
	if err != nil {
		return sourceIndex{}, core.E("Packs", "read safetensors weight files", err)
	}
	index := sourceIndex{
		Names:   idx.Names,
		Tensors: make(map[string]tensorEntry, len(idx.Names)),
	}
	for _, name := range idx.Names {
		ref := idx.Tensors[name]
		index.Tensors[name] = tensorEntry{
			DType: ref.DType,
			Shape: intShapeFromRef(ref.Shape),
			Ref:   ref,
		}
	}
	return index, nil
}

// intShapeFromRef converts a TensorRef's []uint64 shape (the safetensors
// index's on-disk-dimension type) to the []int shape merge/compare's
// public result types (TensorDelta.Shape et al.) and shapeElements have
// always used.
func intShapeFromRef(shape []uint64) []int {
	out := make([]int, len(shape))
	for i, dim := range shape {
		out[i] = int(dim)
	}
	return out
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
// so a merged pack never mixes dtypes. Every source tensor payload is
// streamed from its shard on demand — through a ShardCache shared across
// this whole call — exactly when this loop reaches its name, so a multi-GB
// sharded source is never resident beyond the one tensor currently being
// merged. The merged OUTPUT tensor set is still assembled in mergedInfo/
// mergedData before the single safetensors.WriteSafetensors call below
// (merge output is always one file — see outputWeightsFile), so peak
// memory is bounded by one copy of the output pack, not the input shard
// set(s).
func writeMergedSafetensors(ctx context.Context, path string, indexes []sourceIndex, method Method, t float64, sources []Source, allowMismatch bool) (merged int, copied int, skipped []string, err error) {
	linearWeights, err := normalizedWeights(sources)
	if err != nil {
		return 0, 0, nil, err
	}

	cache := safetensors.NewShardCache()
	defer cache.Close()

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
			decoded, decodeErr := decodeAll(cache, entries)
			if decodeErr != nil {
				return 0, 0, nil, decodeErr
			}
			outValues, err = mergeTensorValues(decoded, method, t, linearWeights)
			if err != nil {
				return 0, 0, nil, err
			}
			merged++
		case allowMismatch:
			outValues, err = cache.ReadRefValues(baseEntry.Ref)
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
		return 0, 0, nil, core.E("Packs", "write merged safetensors", result.Err())
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
		} else if !core.SliceEqual(shape, entry.Shape) {
			complete = false
		}
		entries = append(entries, entry)
	}
	return entries, complete && len(entries) == len(indexes)
}

// decodeAll reads and decodes every entry's tensor payload to float32, from
// its own shard, over cache — the ShardCache shared for the whole merge
// call, so tensors sharing a shard file reuse one open handle instead of
// reopening it per tensor.
func decodeAll(cache *safetensors.ShardCache, entries []tensorEntry) ([][]float32, error) {
	values := make([][]float32, len(entries))
	for i, entry := range entries {
		decoded, err := cache.ReadRefValues(entry.Ref)
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
		return core.E("Packs", "marshal merge provenance", data.Err())
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("Packs", "write merge provenance", result.Err())
	}
	return nil
}
