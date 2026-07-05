// SPDX-Licence-Identifier: EUPL-1.2

// Package dataset loads engine-agnostic supervised fine-tuning and
// evaluation samples from JSONL training corpora, and provides small
// in-memory dataset adapters (a replayable slice and a function
// generator). It is the engine-agnostic half of what was previously
// go-mlx-only (go-mlx/go/dataset): JSONL ingestion plus shape
// normalisation carries no engine dependency, so it belongs here
// where every driver (go-mlx, go-rocm, go-cpu) can share it.
//
// Sample, Dataset, and Resetter are aliases onto the canonical
// inference.DatasetSample / DatasetStream / DatasetResetter contracts
// (see dataset.go at the module root) — callers do not need to import
// the root package directly, and nothing here grows a second,
// competing sample shape.
//
// # Chat-template rendering is deliberately not reproduced here
//
// Turning a chat message list into the exact prompt string a given
// model's own chat template produces is a model-family concern: in
// go-mlx that rendering lives in per-family packages
// (pkg/metal/model/{family}/chat) that register themselves into a
// template registry, and even go-mlx's own dataset package falls back
// to a plain, role-free join whenever no such family package happens
// to be imported in the same binary. This package always uses that
// same plain join to build Sample.Prompt for chat-shaped rows, and
// retains the normalised turns verbatim in Sample.Messages so an
// engine can apply its own real chat template downstream without
// re-parsing the source corpus.
//
//	d, err := dataset.LoadJSONL(reader)
//	for {
//	    sample, ok, err := d.Next()
//	    if err != nil { return err }
//	    if !ok { break }
//	    // sample.Messages holds the raw turns for chat-shaped rows —
//	    // apply the engine's own chat template before tokenising.
//	}
package dataset

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sample is one supervised fine-tuning or evaluation record — an alias onto
// the canonical inference.DatasetSample contract.
type Sample = inference.DatasetSample

// Dataset streams Samples — an alias onto the canonical
// inference.DatasetStream contract.
type Dataset = inference.DatasetStream

// Resetter marks datasets that can be replayed for multiple epochs — an
// alias onto the canonical inference.DatasetResetter contract.
type Resetter = inference.DatasetResetter

// Sentinel errors hoisted from the nil-guard call sites so they allocate
// exactly once at package init instead of one *Err per nil-receiver call.
// These are cold paths (only fire when a caller has passed a nil receiver)
// but the package contract is the same either way.
var (
	errFuncDatasetNil  = core.NewError("dataset: dataset func is nil")
	errSliceDatasetNil = core.NewError("dataset: slice dataset is nil")
)

// Func adapts a function into a Dataset.
type Func func() (Sample, bool, error)

// Next returns the next sample from the wrapped function.
//
//	dataset := dataset.Func(func() (dataset.Sample, bool, error) { ... })
func (fn Func) Next() (Sample, bool, error) {
	if fn == nil {
		return Sample{}, false, errFuncDatasetNil
	}
	return fn()
}

// SliceDataset is an in-memory replayable dataset.
type SliceDataset struct {
	samples []Sample
	index   int
}

// Compile-time proof that the concrete adapters satisfy the canonical
// inference.DatasetStream / DatasetResetter contracts via the Dataset /
// Resetter aliases.
var (
	_ Dataset  = (*SliceDataset)(nil)
	_ Resetter = (*SliceDataset)(nil)
	_ Dataset  = Func(nil)
)

// NewSliceDataset returns a replayable dataset backed by samples.
//
//	d := dataset.NewSliceDataset(samples)
func NewSliceDataset(samples []Sample) *SliceDataset {
	return &SliceDataset{samples: CloneSamples(samples)}
}

// Next returns the next sample.
func (d *SliceDataset) Next() (Sample, bool, error) {
	if d == nil {
		return Sample{}, false, errSliceDatasetNil
	}
	if d.index >= len(d.samples) {
		return Sample{}, false, nil
	}
	sample := d.samples[d.index]
	d.index++
	return sample, true, nil
}

// Reset rewinds the dataset.
func (d *SliceDataset) Reset() error {
	if d == nil {
		return errSliceDatasetNil
	}
	d.index = 0
	return nil
}

// CloneSample returns a defensive deep copy of sample, including Labels and
// Messages — mutating the clone's map or message slice never reaches the
// source.
//
//	copy := dataset.CloneSample(sample)
func CloneSample(sample Sample) Sample {
	sample.Labels = cloneStringMap(sample.Labels)
	sample.Messages = cloneMessages(sample.Messages)
	return sample
}

// CloneSamples returns a defensive deep copy of samples.
//
//	copies := dataset.CloneSamples(samples)
func CloneSamples(samples []Sample) []Sample {
	if len(samples) == 0 {
		return nil
	}
	out := make([]Sample, len(samples))
	for i, sample := range samples {
		out[i] = CloneSample(sample)
	}
	return out
}

func cloneStringMap(values map[string]string) map[string]string {
	// core.MapClone wraps maps.Clone which uses runtime internals to
	// pre-size the destination and bulk-copy entries, skipping the
	// per-key hash/insert ceremony of a range-copy loop. Returns nil for
	// an empty input (matching the prior nil-fast-path).
	if len(values) == 0 {
		return nil
	}
	return core.MapClone(values)
}

// cloneMessages returns a defensive copy of messages. inference.Message's
// Images field is a [][]byte, so this is a shallow clone of that field —
// dataset ingestion (LoadJSONL) never populates Images (JSONL rows are
// text-only), so no caller can currently observe the aliasing; deep-clone
// Images too if a future caller starts attaching them here.
func cloneMessages(messages []inference.Message) []inference.Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]inference.Message, len(messages))
	copy(out, messages)
	return out
}
