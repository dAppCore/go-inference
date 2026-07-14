// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// gpuCounterProfiler measures per-family GPU time on the live-encoder decode lane. Apple GPUs
// sample timestamp counters ONLY at encoder stage boundaries (no mid-encoder sampling), so the
// instrument works by SPLITTING the token's single encoder at family seams — each split opens a
// fresh encoder with a timestamp sample pair attached, and the per-encoder spans aggregate by
// label. Prod decodes run with the profiler nil: every seam collapses to the unsplit
// single-encoder fast path (see archDecodeState.profSeam).
type gpuCounterProfiler struct {
	buf    metal.MTLCounterSampleBuffer
	labels []string
	// countAt[i] is the dispatchCountForTest snapshot taken when labelled encoder i opened, so
	// encoder i's live-encoder dispatch count is countAt[i+1]-countAt[i] (the last uses the
	// final total). Populated only when pieceTimingOn armed dispatchCountForTest for the run;
	// see dispatchCountsByLabel. Counts the encSink funnel — exact for the MoE labels (the whole
	// block dispatches through encSink) and a lower bound for attn (which also uses the Object
	// funnel dispatchThreadsObject, which the test counter deliberately leaves untouched).
	countAt []int64
	max     int
}

// newGPUCounterProfiler builds a profiler with capacity for maxEncoders sampled encoders
// (two timestamps each). Returns an error when the device exposes no timestamp counter set.
func newGPUCounterProfiler(maxEncoders int) (*gpuCounterProfiler, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if maxEncoders <= 0 {
		return nil, core.NewError("native.newGPUCounterProfiler: capacity must be > 0")
	}
	var ts metal.MTLCounterSet
	for _, cs := range device.CounterSets() {
		set := metal.MTLCounterSetObjectFromID(cs.GetID())
		if set.Name() == "timestamp" {
			ts = set
			break
		}
	}
	if ts == nil {
		return nil, core.NewError("native.newGPUCounterProfiler: no timestamp counter set")
	}
	desc := metal.NewMTLCounterSampleBufferDescriptor()
	desc.SetCounterSet(ts)
	desc.SetSampleCount(uint(2 * maxEncoders))
	desc.SetStorageMode(metal.MTLStorageModeShared)
	buf, err := device.NewCounterSampleBufferWithDescriptorError(desc)
	if err != nil {
		return nil, core.E("native.newGPUCounterProfiler", "sample buffer", err)
	}
	return &gpuCounterProfiler{buf: buf, max: maxEncoders}, nil
}

// encoderFor opens a compute encoder on cb whose start/end stage-boundary timestamps land in
// the profiler's sample buffer under label. Past capacity it degrades to a plain (unsampled)
// encoder so the decode still completes; the resolved table then under-reports that stretch.
func (p *gpuCounterProfiler) encoderFor(cb metal.MTLCommandBufferObject, label string) metal.MTLComputeCommandEncoderObject {
	if len(p.labels) >= p.max {
		return computeCommandEncoderFast(cb)
	}
	i := len(p.labels)
	p.labels = append(p.labels, label)
	p.countAt = append(p.countAt, dispatchCountForTest)
	pd := metal.NewMTLComputePassDescriptor()
	att := pd.SampleBufferAttachments().ObjectAtIndexedSubscript(0)
	att.SetSampleBuffer(p.buf)
	att.SetStartOfEncoderSampleIndex(uint(2 * i))
	att.SetEndOfEncoderSampleIndex(uint(2*i + 1))
	enc := cb.ComputeCommandEncoderWithDescriptor(pd)
	return metal.MTLComputeCommandEncoderObjectFromID(enc.GetID())
}

// sampled reports how many sampled encoders have been opened since the last reset.
func (p *gpuCounterProfiler) sampled() int { return len(p.labels) }

// spans resolves the sampled timestamp pairs (after the command buffers have completed) and
// sums the per-encoder GPU spans by label. Values are GPU timestamp ticks — nanoseconds on
// Apple Silicon; treat ratios as the primary signal.
func (p *gpuCounterProfiler) spans() (map[string]uint64, error) {
	n := len(p.labels)
	if n == 0 {
		return map[string]uint64{}, nil
	}
	data := p.buf.ResolveCounterRange(foundation.NSRange{Location: 0, Length: uint(2 * n)})
	raw := data.GoBytes()
	if len(raw) < 16*n {
		return nil, core.NewError("native.gpuCounterProfiler.spans: short counter resolve")
	}
	out := make(map[string]uint64, 8)
	for i, label := range p.labels {
		start := binary.LittleEndian.Uint64(raw[16*i:])
		end := binary.LittleEndian.Uint64(raw[16*i+8:])
		if end > start {
			out[label] += end - start
		}
	}
	return out, nil
}

// dispatchCountsByLabel attributes live-encoder dispatches to each sampled encoder's label —
// the structural companion to spans (GPU time). Pass the final dispatchCountForTest so the
// last encoder's tail is attributed. Same-label encoders (the per-layer seams repeat every
// layer) accumulate. Meaningful only when the run armed pieceTimingOn; otherwise every
// countAt snapshot is 0 and the map is empty.
func (p *gpuCounterProfiler) dispatchCountsByLabel(finalCount int64) map[string]int64 {
	out := make(map[string]int64, 8)
	for i, label := range p.labels {
		next := finalCount
		if i+1 < len(p.countAt) {
			next = p.countAt[i+1]
		}
		if d := next - p.countAt[i]; d > 0 {
			out[label] += d
		}
	}
	return out
}

// reset clears the label cursor so the next sampled encoder reuses the buffer from index 0.
func (p *gpuCounterProfiler) reset() {
	p.labels = p.labels[:0]
	p.countAt = p.countAt[:0]
}

// profSeam ends enc and opens a new counter-sampled encoder labelled label when the GPU
// profiler is armed; with the profiler nil (every prod decode) it returns enc unchanged and
// the token keeps its single encoder.
func (s *archDecodeState) profSeam(cb metal.MTLCommandBufferObject, enc metal.MTLComputeCommandEncoderObject, label string) metal.MTLComputeCommandEncoderObject {
	if s.gpuProf == nil {
		return enc
	}
	endEncodingFast(enc)
	return s.gpuProf.encoderFor(cb, label)
}
