// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"sync"

	core "dappco.re/go"
)

const (
	hipGemma4Q4PrefillForwardBatchPoolMax      = 1024
	hipGemma4Q4PrefillForwardLayerBatchPoolMax = 1024
	hipGemma4Q4PrefillLayerBodyBatchPoolMax    = 4096
	hipGemma4Q4PrefillUBatchPoolMax            = 1024
)

var hipGemma4Q4PrefillForwardBatchPool = struct {
	sync.Mutex
	entries []*hipGemma4Q4PrefillForwardBatch
}{
	entries: make([]*hipGemma4Q4PrefillForwardBatch, 0, hipGemma4Q4PrefillForwardBatchPoolMax),
}

var hipGemma4Q4PrefillForwardLayerBatchPool = struct {
	sync.Mutex
	layers [][]hipGemma4Q4PrefillForwardLayerBatch
}{
	layers: make([][]hipGemma4Q4PrefillForwardLayerBatch, 0, hipGemma4Q4PrefillForwardLayerBatchPoolMax),
}

var hipGemma4Q4PrefillLayerBodyBatchPool = struct {
	sync.Mutex
	entries []*hipGemma4Q4PrefillLayerBodyBatch
}{
	entries: make([]*hipGemma4Q4PrefillLayerBodyBatch, 0, hipGemma4Q4PrefillLayerBodyBatchPoolMax),
}

var hipGemma4Q4PrefillUBatchPool = struct {
	sync.Mutex
	entries [][]hipGemma4Q4PrefillUBatch
}{
	entries: make([][]hipGemma4Q4PrefillUBatch, 0, hipGemma4Q4PrefillUBatchPoolMax),
}

const (
	hipPerLayerInputTransposeLaunchArgsVersion uint32 = 1
	hipPerLayerInputTransposeLaunchArgsBytes          = 56
)

type hipPerLayerInputTransposeLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	InputBytes    uint64
	OutputBytes   uint64
	Batch         int
	LayerCount    int
	InputSize     int
}

type hipGemma4Q4PrefillPlan struct {
	PromptTokens int
	StartPos     int
	UBatchTokens int
	OutputTokens int
	BatchCount   int
	InlineBatch  hipGemma4Q4PrefillUBatch
	Batches      []hipGemma4Q4PrefillUBatch
}

func (plan hipGemma4Q4PrefillPlan) NextPosition() int {
	return plan.StartPos + plan.PromptTokens
}

func (plan hipGemma4Q4PrefillPlan) LenBatches() int {
	if plan.BatchCount > 0 {
		return plan.BatchCount
	}
	return len(plan.Batches)
}

func (plan hipGemma4Q4PrefillPlan) Batch(index int) hipGemma4Q4PrefillUBatch {
	if len(plan.Batches) > 0 {
		return plan.Batches[index]
	}
	if index == 0 && plan.BatchCount == 1 {
		return plan.InlineBatch
	}
	return hipGemma4Q4PrefillUBatch{}
}

type hipGemma4Q4PrefillUBatch struct {
	Start    int
	End      int
	Position int
	Tokens   []int32
	// AttentionCaps contains one absolute visible-token count per query row.
	// Nil means the ordinary causal cap (Position+row+1).
	AttentionCaps []int32
	OutputRow     int
	OutputTokens  []bool
}

func (batch hipGemma4Q4PrefillUBatch) OutputToken(index int) bool {
	if batch.OutputRow >= 0 {
		return index == batch.OutputRow
	}
	return index >= 0 && index < len(batch.OutputTokens) && batch.OutputTokens[index]
}

type hipGemma4Q4PrefillQKVBatch struct {
	Query     *hipDeviceByteBuffer
	Key       *hipDeviceByteBuffer
	Value     *hipDeviceByteBuffer
	queryView hipDeviceByteBuffer
	keyView   hipDeviceByteBuffer
	valueView hipDeviceByteBuffer
}

func (batch *hipGemma4Q4PrefillQKVBatch) borrowQueryView(driver nativeHIPDriver, label string, source *hipDeviceByteBuffer) *hipDeviceByteBuffer {
	batch.queryView = hipBorrowDeviceByteBufferValue(driver, label, source.Pointer(), source.SizeBytes(), source.Count())
	return &batch.queryView
}

func (batch *hipGemma4Q4PrefillQKVBatch) borrowKeyView(driver nativeHIPDriver, label string, source *hipDeviceByteBuffer) *hipDeviceByteBuffer {
	batch.keyView = hipBorrowDeviceByteBufferValue(driver, label, source.Pointer(), source.SizeBytes(), source.Count())
	return &batch.keyView
}

func (batch *hipGemma4Q4PrefillQKVBatch) borrowValueView(driver nativeHIPDriver, label string, source *hipDeviceByteBuffer) *hipDeviceByteBuffer {
	batch.valueView = hipBorrowDeviceByteBufferValue(driver, label, source.Pointer(), source.SizeBytes(), source.Count())
	return &batch.valueView
}

func (batch *hipGemma4Q4PrefillQKVBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{batch.Value, batch.Key, batch.Query} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

type hipGemma4Q4PrefillRoPEQKBatch struct {
	Query     *hipDeviceByteBuffer
	Key       *hipDeviceByteBuffer
	queryView hipDeviceByteBuffer
	keyView   hipDeviceByteBuffer
}

func (batch *hipGemma4Q4PrefillRoPEQKBatch) borrowQueryView(driver nativeHIPDriver, label string, source *hipDeviceByteBuffer) *hipDeviceByteBuffer {
	batch.queryView = hipBorrowDeviceByteBufferValue(driver, label, source.Pointer(), source.SizeBytes(), source.Count())
	return &batch.queryView
}

func (batch *hipGemma4Q4PrefillRoPEQKBatch) borrowKeyView(driver nativeHIPDriver, label string, source *hipDeviceByteBuffer) *hipDeviceByteBuffer {
	batch.keyView = hipBorrowDeviceByteBufferValue(driver, label, source.Pointer(), source.SizeBytes(), source.Count())
	return &batch.keyView
}

func (batch *hipGemma4Q4PrefillRoPEQKBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{batch.Key, batch.Query} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

type hipGemma4Q4PrefillDeviceKVBatch struct {
	Cache           *rocmDeviceKVCache
	DescriptorTable *rocmDeviceKVDescriptorTable
	Launch          rocmDeviceKVLaunchDescriptor
	RetainWindow    int
}

func (batch *hipGemma4Q4PrefillDeviceKVBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	if err := batch.DescriptorTable.Close(); err != nil {
		lastErr = err
	}
	if err := batch.Cache.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

type hipGemma4Q4QueryRMSRoPEAttention struct {
	Enabled        bool
	Input          *hipDeviceByteBuffer
	NormConfig     hipRMSNormDeviceWeightConfig
	StartPosition  int
	Base           float32
	FrequencyDim   int
	RotaryCount    int
	FrequencyScale float32
}

type hipGemma4Q4PrefillLayerKVBatch struct {
	InputNorm             *hipDeviceByteBuffer
	QKV                   *hipGemma4Q4PrefillQKVBatch
	QK                    *hipGemma4Q4PrefillRoPEQKBatch
	Value                 *hipDeviceByteBuffer
	DeviceKV              *hipGemma4Q4PrefillDeviceKVBatch
	AttentionOverride     *hipDeviceByteBuffer
	SharedKey             *hipDeviceByteBuffer
	SharedVal             *hipDeviceByteBuffer
	VisibleTokenCaps      *hipDeviceTokenBuffer
	QueryRMSRoPEAttention hipGemma4Q4QueryRMSRoPEAttention
	inputNormView         hipDeviceByteBuffer
	valueView             hipDeviceByteBuffer
	qkvStorage            hipGemma4Q4PrefillQKVBatch
	qkStorage             hipGemma4Q4PrefillRoPEQKBatch
	deviceKVStorage       hipGemma4Q4PrefillDeviceKVBatch
}

func (batch *hipGemma4Q4PrefillLayerKVBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	if err := batch.DeviceKV.Close(); err != nil {
		lastErr = err
	}
	for _, buffer := range []*hipDeviceByteBuffer{batch.AttentionOverride, batch.Value, batch.InputNorm} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	if err := batch.QK.Close(); err != nil {
		lastErr = err
	}
	if err := batch.QKV.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

type hipGemma4Q4PrefillLayerBodyBatch struct {
	AttentionOutput         *hipDeviceByteBuffer
	AttentionProjection     *hipDeviceByteBuffer
	AttentionResidual       *hipDeviceByteBuffer
	PreFeedForward          *hipDeviceByteBuffer
	MLPOutput               *hipDeviceByteBuffer
	PostFeedForward         *hipDeviceByteBuffer
	PerLayerProjection      *hipDeviceByteBuffer
	FinalHidden             *hipDeviceByteBuffer
	NextInputNorm           *hipDeviceByteBuffer
	attentionOutputView     hipDeviceByteBuffer
	attentionProjectionView hipDeviceByteBuffer
	attentionResidualView   hipDeviceByteBuffer
	preFeedForwardView      hipDeviceByteBuffer
	mlpOutputView           hipDeviceByteBuffer
	postFeedForwardView     hipDeviceByteBuffer
	perLayerProjectionView  hipDeviceByteBuffer
	finalHiddenView         hipDeviceByteBuffer
	nextInputNormView       hipDeviceByteBuffer
	closed                  bool
	pooled                  bool
}

func hipBorrowGemma4Q4PrefillLayerBodyBatch() *hipGemma4Q4PrefillLayerBodyBatch {
	hipGemma4Q4PrefillLayerBodyBatchPool.Lock()
	count := len(hipGemma4Q4PrefillLayerBodyBatchPool.entries)
	if count > 0 {
		batch := hipGemma4Q4PrefillLayerBodyBatchPool.entries[count-1]
		hipGemma4Q4PrefillLayerBodyBatchPool.entries[count-1] = nil
		hipGemma4Q4PrefillLayerBodyBatchPool.entries = hipGemma4Q4PrefillLayerBodyBatchPool.entries[:count-1]
		hipGemma4Q4PrefillLayerBodyBatchPool.Unlock()
		*batch = hipGemma4Q4PrefillLayerBodyBatch{pooled: true}
		return batch
	}
	hipGemma4Q4PrefillLayerBodyBatchPool.Unlock()
	return &hipGemma4Q4PrefillLayerBodyBatch{pooled: true}
}

func hipReleaseGemma4Q4PrefillLayerBodyBatch(batch *hipGemma4Q4PrefillLayerBodyBatch) {
	if batch == nil {
		return
	}
	if !batch.pooled {
		*batch = hipGemma4Q4PrefillLayerBodyBatch{closed: true}
		return
	}
	*batch = hipGemma4Q4PrefillLayerBodyBatch{closed: true, pooled: true}
	hipGemma4Q4PrefillLayerBodyBatchPool.Lock()
	if len(hipGemma4Q4PrefillLayerBodyBatchPool.entries) < hipGemma4Q4PrefillLayerBodyBatchPoolMax {
		hipGemma4Q4PrefillLayerBodyBatchPool.entries = append(hipGemma4Q4PrefillLayerBodyBatchPool.entries, batch)
	}
	hipGemma4Q4PrefillLayerBodyBatchPool.Unlock()
}

func hipBorrowGemma4Q4PrefillForwardLayerBatches(layerCapacity int) []hipGemma4Q4PrefillForwardLayerBatch {
	if layerCapacity <= 0 {
		layerCapacity = 1
	}
	hipGemma4Q4PrefillForwardLayerBatchPool.Lock()
	for index := len(hipGemma4Q4PrefillForwardLayerBatchPool.layers) - 1; index >= 0; index-- {
		layers := hipGemma4Q4PrefillForwardLayerBatchPool.layers[index]
		hipGemma4Q4PrefillForwardLayerBatchPool.layers[index] = nil
		hipGemma4Q4PrefillForwardLayerBatchPool.layers = hipGemma4Q4PrefillForwardLayerBatchPool.layers[:index]
		if cap(layers) >= layerCapacity {
			hipGemma4Q4PrefillForwardLayerBatchPool.Unlock()
			return layers[:0]
		}
	}
	hipGemma4Q4PrefillForwardLayerBatchPool.Unlock()
	return make([]hipGemma4Q4PrefillForwardLayerBatch, 0, layerCapacity)
}

func hipReleaseGemma4Q4PrefillForwardLayerBatches(layers []hipGemma4Q4PrefillForwardLayerBatch) {
	if cap(layers) == 0 {
		return
	}
	clear(layers[:cap(layers)])
	hipGemma4Q4PrefillForwardLayerBatchPool.Lock()
	if len(hipGemma4Q4PrefillForwardLayerBatchPool.layers) < hipGemma4Q4PrefillForwardLayerBatchPoolMax {
		hipGemma4Q4PrefillForwardLayerBatchPool.layers = append(hipGemma4Q4PrefillForwardLayerBatchPool.layers, layers[:0])
	}
	hipGemma4Q4PrefillForwardLayerBatchPool.Unlock()
}

func hipBorrowGemma4Q4PrefillUBatches(batchCapacity int) []hipGemma4Q4PrefillUBatch {
	if batchCapacity <= 1 {
		return nil
	}
	hipGemma4Q4PrefillUBatchPool.Lock()
	for index := len(hipGemma4Q4PrefillUBatchPool.entries) - 1; index >= 0; index-- {
		batches := hipGemma4Q4PrefillUBatchPool.entries[index]
		hipGemma4Q4PrefillUBatchPool.entries[index] = nil
		hipGemma4Q4PrefillUBatchPool.entries = hipGemma4Q4PrefillUBatchPool.entries[:index]
		if cap(batches) >= batchCapacity {
			hipGemma4Q4PrefillUBatchPool.Unlock()
			return batches[:0]
		}
	}
	hipGemma4Q4PrefillUBatchPool.Unlock()
	return make([]hipGemma4Q4PrefillUBatch, 0, batchCapacity)
}

func hipReleaseGemma4Q4PrefillUBatches(batches []hipGemma4Q4PrefillUBatch) {
	if cap(batches) == 0 {
		return
	}
	clear(batches[:cap(batches)])
	hipGemma4Q4PrefillUBatchPool.Lock()
	if len(hipGemma4Q4PrefillUBatchPool.entries) < hipGemma4Q4PrefillUBatchPoolMax {
		hipGemma4Q4PrefillUBatchPool.entries = append(hipGemma4Q4PrefillUBatchPool.entries, batches[:0])
	}
	hipGemma4Q4PrefillUBatchPool.Unlock()
}

func hipGemma4Q4PrefillBatchCount(tokenCount, ubatchTokens int) int {
	if tokenCount <= 0 || ubatchTokens <= 0 {
		return 0
	}
	return (tokenCount + ubatchTokens - 1) / ubatchTokens
}

func hipPrewarmGemma4Q4PrefillForwardLayerBatchPool(layerCapacity, depth int) {
	if layerCapacity <= 0 || depth <= 0 {
		return
	}
	forwardBatches := make([]*hipGemma4Q4PrefillForwardBatch, 0, depth)
	batches := make([][]hipGemma4Q4PrefillForwardLayerBatch, 0, depth)
	for range depth {
		forwardBatches = append(forwardBatches, hipBorrowGemma4Q4PrefillForwardBatch(layerCapacity))
		batches = append(batches, hipBorrowGemma4Q4PrefillForwardLayerBatches(layerCapacity))
	}
	for _, batch := range forwardBatches {
		_ = batch.Close()
	}
	for _, layers := range batches {
		hipReleaseGemma4Q4PrefillForwardLayerBatches(layers)
	}
}

type hipGemma4Q4PrefillForwardLayerBatch struct {
	KV          *hipGemma4Q4PrefillLayerKVBatch
	Body        *hipGemma4Q4PrefillLayerBodyBatch
	kvStorage   hipGemma4Q4PrefillLayerKVBatch
	bodyStorage hipGemma4Q4PrefillLayerBodyBatch
}

func (batch *hipGemma4Q4PrefillForwardLayerBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	if err := batch.Body.Close(); err != nil {
		lastErr = err
	}
	if err := batch.KV.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

type hipGemma4Q4PrefillGreedyBatchOutput struct {
	Row    int
	Greedy hipGreedySampleResult
}

type hipGemma4Q4PrefillForwardBatch struct {
	Embedding     *hipDeviceByteBuffer
	Layers        []hipGemma4Q4PrefillForwardLayerBatch
	FinalHidden   *hipDeviceByteBuffer
	Greedy        []hipGemma4Q4PrefillGreedyBatchOutput
	embeddingView hipDeviceByteBuffer
	greedyStorage [1]hipGemma4Q4PrefillGreedyBatchOutput
	closed        bool
}

func hipBorrowGemma4Q4PrefillForwardBatch(layerCapacity int) *hipGemma4Q4PrefillForwardBatch {
	hipGemma4Q4PrefillForwardBatchPool.Lock()
	count := len(hipGemma4Q4PrefillForwardBatchPool.entries)
	if count > 0 {
		batch := hipGemma4Q4PrefillForwardBatchPool.entries[count-1]
		hipGemma4Q4PrefillForwardBatchPool.entries[count-1] = nil
		hipGemma4Q4PrefillForwardBatchPool.entries = hipGemma4Q4PrefillForwardBatchPool.entries[:count-1]
		hipGemma4Q4PrefillForwardBatchPool.Unlock()
		*batch = hipGemma4Q4PrefillForwardBatch{
			Layers: hipBorrowGemma4Q4PrefillForwardLayerBatches(layerCapacity),
		}
		batch.Greedy = batch.greedyStorage[:0]
		return batch
	}
	hipGemma4Q4PrefillForwardBatchPool.Unlock()
	batch := &hipGemma4Q4PrefillForwardBatch{
		Layers: hipBorrowGemma4Q4PrefillForwardLayerBatches(layerCapacity),
	}
	batch.Greedy = batch.greedyStorage[:0]
	return batch
}

func hipReleaseGemma4Q4PrefillForwardBatch(batch *hipGemma4Q4PrefillForwardBatch) {
	if batch == nil {
		return
	}
	*batch = hipGemma4Q4PrefillForwardBatch{closed: true}
	hipGemma4Q4PrefillForwardBatchPool.Lock()
	if len(hipGemma4Q4PrefillForwardBatchPool.entries) < hipGemma4Q4PrefillForwardBatchPoolMax {
		hipGemma4Q4PrefillForwardBatchPool.entries = append(hipGemma4Q4PrefillForwardBatchPool.entries, batch)
	}
	hipGemma4Q4PrefillForwardBatchPool.Unlock()
}

func (batch *hipGemma4Q4PrefillForwardBatch) Close() error {
	if batch == nil || batch.closed {
		return nil
	}
	var lastErr error
	for index := len(batch.Layers) - 1; index >= 0; index-- {
		if err := batch.Layers[index].Close(); err != nil {
			lastErr = err
		}
	}
	hipReleaseGemma4Q4PrefillForwardLayerBatches(batch.Layers)
	batch.Layers = nil
	if err := batch.Embedding.Close(); err != nil {
		lastErr = err
	}
	hipReleaseGemma4Q4PrefillForwardBatch(batch)
	return lastErr
}

func (batch *hipGemma4Q4PrefillLayerBodyBatch) Close() error {
	if batch == nil || batch.closed {
		return nil
	}
	pooled := batch.pooled
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{
		batch.NextInputNorm,
		batch.FinalHidden,
		batch.PerLayerProjection,
		batch.PostFeedForward,
		batch.MLPOutput,
		batch.PreFeedForward,
		batch.AttentionResidual,
		batch.AttentionProjection,
		batch.AttentionOutput,
	} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	if pooled {
		hipReleaseGemma4Q4PrefillLayerBodyBatch(batch)
	} else {
		*batch = hipGemma4Q4PrefillLayerBodyBatch{closed: true}
	}
	return lastErr
}

func (args hipPerLayerInputTransposeLaunchArgs) Binary() ([]byte, error) {
	return args.BinaryInto(nil)
}

func (args hipPerLayerInputTransposeLaunchArgs) BinaryInto(payload []byte) ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose pointers are required", nil)
	}
	batch, err := rocmDeviceKVPositiveUint32("per-layer input transpose batch", args.Batch)
	if err != nil {
		return nil, err
	}
	layerCount, err := rocmDeviceKVPositiveUint32("per-layer input transpose layer count", args.LayerCount)
	if err != nil {
		return nil, err
	}
	inputSize, err := rocmDeviceKVPositiveUint32("per-layer input transpose input size", args.InputSize)
	if err != nil {
		return nil, err
	}
	wantBytes := uint64(batch) * uint64(layerCount) * uint64(inputSize) * 4
	if args.InputBytes != wantBytes || args.OutputBytes != wantBytes {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose byte count mismatch", nil)
	}
	if cap(payload) < hipPerLayerInputTransposeLaunchArgsBytes {
		payload = hipBorrowLaunchPacket(hipPerLayerInputTransposeLaunchArgsBytes)
	} else {
		payload = payload[:hipPerLayerInputTransposeLaunchArgsBytes]
		clear(payload)
	}
	binary.LittleEndian.PutUint32(payload[0:], hipPerLayerInputTransposeLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[24:], args.InputBytes)
	binary.LittleEndian.PutUint64(payload[32:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[40:], batch)
	binary.LittleEndian.PutUint32(payload[44:], layerCount)
	binary.LittleEndian.PutUint32(payload[48:], inputSize)
	return payload, nil
}

func hipGemma4Q4PlanPromptPrefill(promptTokens []int32, startPos int, ubatchTokens int) (hipGemma4Q4PrefillPlan, error) {
	plan, _, err := hipGemma4Q4PlanPromptPrefillInto(promptTokens, startPos, ubatchTokens, nil)
	return plan, err
}

func hipGemma4Q4PlanPromptPrefillInto(promptTokens []int32, startPos int, ubatchTokens int, batches []hipGemma4Q4PrefillUBatch) (hipGemma4Q4PrefillPlan, []hipGemma4Q4PrefillUBatch, error) {
	if len(promptTokens) == 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill requires at least one token", nil)
	}
	if startPos < 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill start position must be non-negative", nil)
	}
	if ubatchTokens <= 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill ubatch size must be positive", nil)
	}
	batchCount := hipGemma4Q4PrefillBatchCount(len(promptTokens), ubatchTokens)
	plan := hipGemma4Q4PrefillPlan{
		PromptTokens: len(promptTokens),
		StartPos:     startPos,
		UBatchTokens: ubatchTokens,
		OutputTokens: 1,
		BatchCount:   batchCount,
	}
	if batchCount > 1 {
		if cap(batches) < batchCount {
			batches = make([]hipGemma4Q4PrefillUBatch, 0, batchCount)
		} else {
			batches = batches[:0]
		}
		plan.Batches = batches
	}
	for start := 0; start < len(promptTokens); start += ubatchTokens {
		end := start + ubatchTokens
		if end > len(promptTokens) {
			end = len(promptTokens)
		}
		tokens := promptTokens[start:end]
		outputRow := -1
		if end == len(promptTokens) {
			outputRow = len(tokens) - 1
		}
		batch := hipGemma4Q4PrefillUBatch{
			Start:     start,
			End:       end,
			Position:  startPos + start,
			Tokens:    tokens,
			OutputRow: outputRow,
		}
		if batchCount == 1 {
			plan.InlineBatch = batch
		} else {
			plan.Batches = append(plan.Batches, batch)
		}
	}
	return plan, plan.Batches, nil
}

func hipGemma4Q4PlanPromptPrefillSharedSuffixInto(promptTokens []int32, startPos, ubatchTokens, slidingWindow int, batches []hipGemma4Q4PrefillUBatch) (hipGemma4Q4PrefillPlan, []hipGemma4Q4PrefillUBatch, error) {
	if len(promptTokens) == 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill requires at least one token", nil)
	}
	if startPos < 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill start position must be non-negative", nil)
	}
	if ubatchTokens <= 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill ubatch size must be positive", nil)
	}
	if slidingWindow <= 0 {
		return hipGemma4Q4PlanPromptPrefillInto(promptTokens, startPos, ubatchTokens, batches)
	}
	plan := hipGemma4Q4PrefillPlan{
		PromptTokens: len(promptTokens),
		StartPos:     startPos,
		UBatchTokens: ubatchTokens,
		OutputTokens: 1,
	}
	first := hipGemma4Q4PrefillSharedSuffixChunkLen(startPos, len(promptTokens), ubatchTokens, slidingWindow)
	if first == len(promptTokens) {
		plan.BatchCount = 1
		plan.InlineBatch = hipGemma4Q4PrefillUBatch{
			End:       len(promptTokens),
			Position:  startPos,
			Tokens:    promptTokens,
			OutputRow: len(promptTokens) - 1,
		}
		return plan, batches[:0], nil
	}
	capacity := hipGemma4Q4PrefillBatchCount(len(promptTokens), ubatchTokens) + 2
	if cap(batches) < capacity {
		batches = make([]hipGemma4Q4PrefillUBatch, 0, capacity)
	} else {
		batches = batches[:0]
	}
	for start := 0; start < len(promptTokens); {
		count := hipGemma4Q4PrefillSharedSuffixChunkLen(startPos+start, len(promptTokens)-start, ubatchTokens, slidingWindow)
		if count <= 0 || count > len(promptTokens)-start {
			return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "shared-suffix prefill produced an invalid batch", nil)
		}
		end := start + count
		outputRow := -1
		if end == len(promptTokens) {
			outputRow = count - 1
		}
		batches = append(batches, hipGemma4Q4PrefillUBatch{
			Start:     start,
			End:       end,
			Position:  startPos + start,
			Tokens:    promptTokens[start:end],
			OutputRow: outputRow,
		})
		start = end
	}
	plan.BatchCount = len(batches)
	plan.Batches = batches
	return plan, batches, nil
}

func hipGemma4Q4PrefillSharedSuffixChunkLen(position, remaining, ubatchTokens, slidingWindow int) int {
	if remaining <= 0 || ubatchTokens <= 0 {
		return 0
	}
	if slidingWindow <= 0 {
		return min(remaining, ubatchTokens)
	}
	if remaining > ubatchTokens {
		if offset := position % slidingWindow; offset > 0 {
			aligned := slidingWindow - offset
			if aligned < remaining && aligned <= ubatchTokens {
				return aligned
			}
		}
		return ubatchTokens
	}
	boundary := (position + remaining) % slidingWindow
	if boundary == 0 {
		boundary = slidingWindow
	}
	if boundary < remaining {
		return remaining - boundary
	}
	return remaining
}

// hipGemma4Q4BidirectionalTokenSpans returns the [start,end) runs of image and
// video placeholder IDs. Adjacent runs of different media token types remain
// separate so an image cannot attend into the following video block.
func hipGemma4Q4BidirectionalTokenSpans(ids []int32, tokenIDs [2]int32) [][2]int {
	if tokenIDs[0] == 0 && tokenIDs[1] == 0 {
		return nil
	}
	isSpan := func(id int32) bool {
		return id != 0 && (id == tokenIDs[0] || id == tokenIDs[1])
	}
	var spans [][2]int
	start := -1
	var startToken int32
	for index, id := range ids {
		switch {
		case isSpan(id) && start < 0:
			start, startToken = index, id
		case start >= 0 && id != startToken:
			spans = append(spans, [2]int{start, index})
			if isSpan(id) {
				start, startToken = index, id
			} else {
				start = -1
			}
		}
	}
	if start >= 0 {
		spans = append(spans, [2]int{start, len(ids)})
	}
	return spans
}

// hipGemma4Q4PlanPromptPrefillBidirectionalInto cuts microbatches only between
// media spans and assigns per-row absolute attention caps. A media span larger
// than the configured microbatch grows that batch; splitting it would make its
// early rows unable to see K/V rows that have not landed yet.
func hipGemma4Q4PlanPromptPrefillBidirectionalInto(promptTokens []int32, startPos int, ubatchTokens int, spans [][2]int, batches []hipGemma4Q4PrefillUBatch) (hipGemma4Q4PrefillPlan, []hipGemma4Q4PrefillUBatch, error) {
	if len(spans) == 0 {
		return hipGemma4Q4PlanPromptPrefillInto(promptTokens, startPos, ubatchTokens, batches)
	}
	if len(promptTokens) == 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill requires at least one token", nil)
	}
	if startPos < 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill start position must be non-negative", nil)
	}
	if ubatchTokens <= 0 {
		return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "prompt prefill ubatch size must be positive", nil)
	}
	previousEnd := 0
	for _, span := range spans {
		if span[0] < previousEnd || span[0] < 0 || span[1] <= span[0] || span[1] > len(promptTokens) {
			return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "invalid bidirectional prompt span", nil)
		}
		previousEnd = span[1]
	}

	capacity := hipGemma4Q4PrefillBatchCount(len(promptTokens), ubatchTokens) + len(spans)
	if cap(batches) < capacity {
		batches = make([]hipGemma4Q4PrefillUBatch, 0, capacity)
	} else {
		batches = batches[:0]
	}
	for start := 0; start < len(promptTokens); {
		end := min(start+ubatchTokens, len(promptTokens))
		for adjusted := true; adjusted; {
			adjusted = false
			for _, span := range spans {
				if span[0] < end && span[1] > end {
					if span[0] > start {
						end = span[0]
					} else {
						end = span[1]
					}
					adjusted = true
					break
				}
			}
		}
		if end <= start || end > len(promptTokens) {
			return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "invalid bidirectional prefill batch", nil)
		}
		var caps []int32
		for _, span := range spans {
			if span[1] <= start || span[0] >= end {
				continue
			}
			if span[0] < start || span[1] > end {
				return hipGemma4Q4PrefillPlan{}, batches, core.E(hipGemma4Q4Layer0Operation, "bidirectional span crosses a prefill batch", nil)
			}
			if caps == nil {
				caps = make([]int32, end-start)
				for row := range caps {
					caps[row] = int32(startPos + start + row + 1)
				}
			}
			spanCap := int32(startPos + span[1])
			for row := span[0]; row < span[1]; row++ {
				caps[row-start] = spanCap
			}
		}
		outputRow := -1
		if end == len(promptTokens) {
			outputRow = end - start - 1
		}
		batches = append(batches, hipGemma4Q4PrefillUBatch{
			Start:         start,
			End:           end,
			Position:      startPos + start,
			Tokens:        promptTokens[start:end],
			AttentionCaps: caps,
			OutputRow:     outputRow,
		})
		start = end
	}
	plan := hipGemma4Q4PrefillPlan{
		PromptTokens: len(promptTokens),
		StartPos:     startPos,
		UBatchTokens: ubatchTokens,
		OutputTokens: 1,
		BatchCount:   len(batches),
		Batches:      batches,
	}
	return plan, batches, nil
}

func hipRunGemma4Q4PrefillEmbeddingBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokens []int32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Embedding.HiddenSize != cfg.HiddenSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding hidden size mismatch", nil)
	}
	if err := cfg.Embedding.validate(tokens); err != nil {
		return nil, err
	}
	tokenBuffer, err := hipUploadTokenIDs(driver, tokens)
	if err != nil {
		return nil, err
	}
	defer tokenBuffer.Close()
	return hipRunGemma4Q4PrefillEmbeddingBatchTokenBuffer(ctx, driver, cfg, tokens, tokenBuffer)
}

func hipRunGemma4Q4PrefillEmbeddingBatchTokenBuffer(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokens []int32, tokenBuffer *hipDeviceTokenBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Embedding.HiddenSize != cfg.HiddenSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding hidden size mismatch", nil)
	}
	if err := cfg.Embedding.validate(tokens); err != nil {
		return nil, err
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != len(tokens) || tokenBuffer.SizeBytes() != uint64(len(tokens)*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding token buffer shape mismatch", nil)
	}
	embedding, err := hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchBuffer(ctx, driver, cfg.Embedding, tokenBuffer)
	if err != nil {
		return nil, err
	}
	defer embedding.Close()
	scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, embedding, cfg.embeddingScale())
	if err != nil {
		return nil, err
	}
	return scaled, nil
}

func hipRunGemma4Q4PrefillEmbeddingBatchTokenBufferWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokens []int32, tokenBuffer *hipDeviceTokenBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillEmbeddingBatchTokenBufferWorkspaceView(ctx, driver, cfg, tokens, tokenBuffer, workspace, nil)
}

func hipRunGemma4Q4PrefillEmbeddingBatchTokenBufferWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokens []int32, tokenBuffer *hipDeviceTokenBuffer, workspace *hipAttentionHeadsChunkedWorkspace, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return hipRunGemma4Q4PrefillEmbeddingBatchTokenBuffer(ctx, driver, cfg, tokens, tokenBuffer)
	}
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Embedding.HiddenSize != cfg.HiddenSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding hidden size mismatch", nil)
	}
	if err := cfg.Embedding.validate(tokens); err != nil {
		return nil, err
	}
	if tokenBuffer == nil || tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != len(tokens) || tokenBuffer.SizeBytes() != uint64(len(tokens)*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding token buffer shape mismatch", nil)
	}
	count := len(tokens) * cfg.HiddenSize
	output, err := workspace.EnsureScaledEmbedding(driver, count)
	if err != nil {
		return nil, err
	}
	if err := hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchScaledOutput(ctx, driver, cfg.Embedding, tokenBuffer, output, cfg.embeddingScale()); err != nil {
		return nil, err
	}
	if view != nil {
		*view = hipBorrowDeviceByteBufferValue(driver, "prefill embedding workspace view", output.Pointer(), output.SizeBytes(), output.Count())
		return view, nil
	}
	return hipBorrowDeviceByteBuffer(driver, "prefill embedding workspace view", output.Pointer(), output.SizeBytes(), output.Count()), nil
}

func hipRunGemma4Q4PrefillEmbeddingBatchGreedyTokenWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokens []int32, greedyToken *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Embedding.HiddenSize != cfg.HiddenSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding hidden size mismatch", nil)
	}
	if len(tokens) != 1 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill greedy embedding requires one token", nil)
	}
	if err := cfg.Embedding.validate(tokens); err != nil {
		return nil, err
	}
	if greedyToken == nil || greedyToken.Pointer() == 0 || greedyToken.Count() != 1 || greedyToken.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill greedy embedding token buffer shape mismatch", nil)
	}
	count := cfg.HiddenSize
	if workspace != nil {
		output, err := workspace.EnsureScaledEmbedding(driver, count)
		if err != nil {
			return nil, err
		}
		if err := hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx, driver, cfg.Embedding, greedyToken, output, cfg.embeddingScale(), workspace); err != nil {
			return nil, err
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill embedding workspace view", output.Pointer(), output.SizeBytes(), output.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, "prefill embedding workspace view", output.Pointer(), output.SizeBytes(), output.Count()), nil
	}
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill greedy embedding output", uint64(count*4), count)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutput(ctx, driver, cfg.Embedding, greedyToken, output, cfg.embeddingScale()); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunPerLayerInputTransposeKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, batch, layerCount, inputSize int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose input buffer is required", nil)
	}
	if batch <= 0 || layerCount <= 0 || inputSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose shape must be positive", nil)
	}
	count := batch * layerCount * inputSize
	if input.Count() != count || input.SizeBytes() != uint64(count*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "Gemma4 q4 per-layer input transpose output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipPerLayerInputTransposeLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Batch:         batch,
		LayerCount:    layerCount,
		InputSize:     inputSize,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNamePerLayerInputTranspose, launchBytes, count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernelContext(ctx, driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, hidden *hipDeviceByteBuffer, epsilon float32) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	return hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspace(ctx, driver, cfg, tokens, hidden, epsilon, nil)
}

func hipRunPerLayerInputTransposeKernelOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, batch, layerCount, inputSize int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose input buffer is required", nil)
	}
	if batch <= 0 || layerCount <= 0 || inputSize <= 0 {
		return core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose shape must be positive", nil)
	}
	count := batch * layerCount * inputSize
	if input.Count() != count || input.SizeBytes() != uint64(count*4) {
		return core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose input shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != count || output.SizeBytes() != input.SizeBytes() {
		return core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose output shape mismatch", nil)
	}
	launchBytes, err := (hipPerLayerInputTransposeLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Batch:         batch,
		LayerCount:    layerCount,
		InputSize:     inputSize,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNamePerLayerInputTranspose, launchBytes, count)
	if err != nil {
		return err
	}
	return hipLaunchKernelContext(ctx, driver, config)
}

func hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, hidden *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	return hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceTokenBuffer(ctx, driver, cfg, tokens, hidden, epsilon, workspace, nil)
}

func hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceTokenBuffer(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, hidden *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, tokenBuffer *hipDeviceTokenBuffer) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	return hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceDeviceToken(ctx, driver, cfg, tokens, hidden, epsilon, workspace, tokenBuffer, nil)
}

func hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceGreedyToken(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, hidden *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, greedyToken *hipDeviceByteBuffer) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	return hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceDeviceToken(ctx, driver, cfg, tokens, hidden, epsilon, workspace, nil, greedyToken)
}

func hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceDeviceToken(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, hidden *hipDeviceByteBuffer, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, tokenBuffer *hipDeviceTokenBuffer, greedyToken *hipDeviceByteBuffer) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if len(cfg.Layers) == 0 || !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		return nil, nil
	}
	if hidden == nil || hidden.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden batch is required", nil)
	}
	if len(tokens) == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input tokens are required", nil)
	}
	perLayer := cfg.Layers[0].PerLayerInput
	if !perLayer.hasLayerApply() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input precompute requires per-layer gate/projection tensors", nil)
	}
	rows := perLayer.modelProjectionRows()
	cols := perLayer.modelProjectionCols()
	if perLayer.InputSize <= 0 || rows%perLayer.InputSize != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input rows must align with input size", nil)
	}
	layerCount := rows / perLayer.InputSize
	if layerCount < len(cfg.Layers) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "computed per-layer input count is smaller than forward layer count", nil)
	}
	if hidden.Count() != len(tokens)*cols || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden batch shape mismatch", nil)
	}
	if tokenBuffer != nil && greedyToken != nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input requires one token buffer format", nil)
	}
	if tokenBuffer != nil && (tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != len(tokens) || tokenBuffer.SizeBytes() != uint64(len(tokens)*4)) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input token buffer shape mismatch", nil)
	}
	if greedyToken != nil && (len(tokens) != 1 || greedyToken.Pointer() == 0 || greedyToken.Count() != 1 || greedyToken.SizeBytes() != hipMLXQ4ProjectionBestBytes) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input greedy token buffer shape mismatch", nil)
	}
	outputCount := rows * len(tokens)
	var err error
	var transposed *hipDeviceByteBuffer
	var perLayerEmbeddingScaled *hipDeviceByteBuffer
	if workspace != nil {
		transposed, err = workspace.EnsurePerLayerOutput(driver, outputCount)
		if err == nil {
			perLayerEmbeddingScaled = transposed
			if greedyToken != nil {
				err = perLayer.Embedding.validate(tokens)
				if err == nil {
					err = hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(ctx, driver, perLayer.Embedding, greedyToken, perLayerEmbeddingScaled, perLayer.embeddingScale(), workspace)
				}
			} else if tokenBuffer != nil {
				err = perLayer.Embedding.validate(tokens)
				if err == nil {
					err = hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchScaledOutput(ctx, driver, perLayer.Embedding, tokenBuffer, perLayerEmbeddingScaled, perLayer.embeddingScale())
				}
			} else {
				err = hipRunEmbeddingLookupKernelWithDeviceTableBufferScaledOutput(ctx, driver, tokens, perLayer.Embedding, perLayerEmbeddingScaled, perLayer.embeddingScale())
			}
		}
	} else {
		var perLayerEmbedding *hipDeviceByteBuffer
		if greedyToken != nil {
			perLayerEmbeddingScaled, err = hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "Gemma4 q4 per-layer greedy embedding scaled", uint64(outputCount*4), outputCount)
			if err == nil {
				err = perLayer.Embedding.validate(tokens)
			}
			if err == nil {
				err = hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutput(ctx, driver, perLayer.Embedding, greedyToken, perLayerEmbeddingScaled, perLayer.embeddingScale())
			}
		} else if tokenBuffer != nil {
			if err = perLayer.Embedding.validate(tokens); err == nil {
				perLayerEmbedding, err = hipRunEmbeddingLookupKernelWithDeviceTableTokenBatchBuffer(ctx, driver, perLayer.Embedding, tokenBuffer)
			}
		} else {
			perLayerEmbedding, err = hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx, driver, tokens, perLayer.Embedding)
		}
		if err != nil {
			return nil, err
		}
		if perLayerEmbedding != nil {
			defer perLayerEmbedding.Close()
			perLayerEmbeddingScaled, err = hipRunVectorScaleDeviceKernel(ctx, driver, perLayerEmbedding, perLayer.embeddingScale())
		}
	}
	if err != nil {
		if workspace == nil && perLayerEmbeddingScaled != nil {
			_ = perLayerEmbeddingScaled.Close()
		}
		return nil, err
	}
	if workspace == nil {
		defer perLayerEmbeddingScaled.Close()
	}
	var projected *hipDeviceByteBuffer
	if workspace != nil {
		projected, err = workspace.EnsurePerLayerProjected(driver, outputCount)
		if err == nil {
			if perLayer.modelProjectionQuantized() {
				err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, hidden, perLayer.ModelProjectionQ4, len(tokens), projected)
			} else {
				err = hipRunProjectionBatchKernelWithDeviceInputWeightEncodingOutput(
					ctx,
					driver,
					hidden,
					perLayer.ModelProjection.WeightPointer,
					perLayer.ModelProjection.WeightBytes,
					rows,
					cols,
					hipProjectionWeightEncodingBF16,
					len(tokens),
					projected,
				)
			}
		}
	} else if perLayer.modelProjectionQuantized() {
		projected, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, hidden, perLayer.ModelProjectionQ4, len(tokens))
	} else {
		projected, err = hipRunProjectionBatchKernelWithDeviceInputWeightEncoding(
			ctx,
			driver,
			hidden,
			perLayer.ModelProjection.WeightPointer,
			perLayer.ModelProjection.WeightBytes,
			rows,
			cols,
			hipProjectionWeightEncodingBF16,
			len(tokens),
		)
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer projected.Close()
	}
	var projectedScaled *hipDeviceByteBuffer
	if workspace != nil {
		projectedScaled = projected
		err = hipRunVectorScaleDeviceKernelOutput(ctx, driver, projected, perLayer.modelProjectionScale(), projectedScaled)
	} else {
		projectedScaled, err = hipRunVectorScaleDeviceKernel(ctx, driver, projected, perLayer.modelProjectionScale())
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer projectedScaled.Close()
	}
	normCfg := perLayer.ProjectionNorm
	normCfg.Epsilon = epsilon
	normCfg.Count = perLayer.InputSize
	var projectedNorm *hipDeviceByteBuffer
	if workspace != nil {
		projectedNorm = projectedScaled
		err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, projectedScaled, normCfg, layerCount*len(tokens), projectedNorm)
	} else {
		projectedNorm, err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, projectedScaled, normCfg, layerCount*len(tokens))
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer projectedNorm.Close()
	}
	var scaled *hipDeviceByteBuffer
	if workspace != nil {
		scaled = projectedNorm
		err = hipRunVectorAddScaledDeviceKernelOutput(ctx, driver, projectedNorm, perLayerEmbeddingScaled, hipGemma4Q4PerLayerCombineScale, scaled)
	} else {
		scaled, err = hipRunVectorAddScaledDeviceKernel(ctx, driver, projectedNorm, perLayerEmbeddingScaled, hipGemma4Q4PerLayerCombineScale)
	}
	if err != nil {
		return nil, err
	}
	if workspace == nil {
		defer scaled.Close()
	}
	if workspace != nil {
		err = hipRunPerLayerInputTransposeKernelOutput(ctx, driver, scaled, len(tokens), layerCount, perLayer.InputSize, transposed)
	} else {
		transposed, err = hipRunPerLayerInputTransposeKernel(ctx, driver, scaled, len(tokens), layerCount, perLayer.InputSize)
	}
	if err != nil {
		return nil, err
	}
	if workspace != nil {
		return workspace.BorrowPerLayerInputDeviceSetBatch(driver, layerCount, len(tokens)*perLayer.InputSize, transposed, "per-layer input batch slice")
	}
	outputs := &hipGemma4Q4PerLayerInputDeviceSet{
		driver:           driver,
		layerCount:       layerCount,
		layerStrideBytes: uint64(len(tokens) * perLayer.InputSize * 4),
		layerValueCount:  len(tokens) * perLayer.InputSize,
		viewLabel:        "per-layer input batch slice",
		borrowedBacking:  false,
		Backing:          []*hipDeviceByteBuffer{transposed},
	}
	success := false
	defer func() {
		if !success {
			_ = outputs.Close()
		}
	}()
	success = true
	return outputs, nil
}

func hipRunGemma4Q4PrefillInputNormBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillInputNormBatchWorkspace(ctx, driver, cfg, input, tokenCount, nil)
}

func hipRunGemma4Q4PrefillInputNormBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillInputNormBatchWorkspaceView(ctx, driver, cfg, input, tokenCount, workspace, nil)
}

func hipRunGemma4Q4PrefillInputNormBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm hidden size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm input buffer shape mismatch", nil)
	}
	if err := hipValidateGemma4Q4NormConfig("Gemma4Q4PrefillInputNorm", cfg.InputNorm, cfg.HiddenSize); err != nil {
		return nil, err
	}
	if workspace != nil {
		output, err := workspace.EnsurePrefillInputNormOutput(driver, input.Count())
		if err != nil {
			return nil, err
		}
		if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, cfg.InputNorm, tokenCount, output); err != nil {
			return nil, err
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill input norm workspace view", output.Pointer(), output.SizeBytes(), output.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, "prefill input norm workspace view", output.Pointer(), output.SizeBytes(), output.Count()), nil
	}
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, cfg.InputNorm, tokenCount)
}

func hipBorrowGemma4Q4PrefillPrecomputedInputNorm(driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, inputNorm *hipDeviceByteBuffer, tokenCount int, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill precomputed input-norm token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill precomputed input-norm hidden size must be positive", nil)
	}
	if inputNorm == nil || inputNorm.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill precomputed input-norm buffer is required", nil)
	}
	wantCount := tokenCount * cfg.HiddenSize
	if inputNorm.Count() != wantCount || inputNorm.SizeBytes() != uint64(wantCount*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill precomputed input-norm buffer shape mismatch", nil)
	}
	if view != nil {
		*view = hipBorrowDeviceByteBufferValue(driver, "prefill precomputed input norm view", inputNorm.Pointer(), inputNorm.SizeBytes(), inputNorm.Count())
		return view, nil
	}
	return hipBorrowDeviceByteBuffer(driver, "prefill precomputed input norm view", inputNorm.Pointer(), inputNorm.SizeBytes(), inputNorm.Count()), nil
}

func hipRunGemma4Q4PrefillQKVProjectionBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) (*hipGemma4Q4PrefillQKVBatch, error) {
	return hipRunGemma4Q4PrefillQKVProjectionBatchWorkspace(ctx, driver, cfg, input, tokenCount, nil)
}

func hipRunGemma4Q4PrefillQKVProjectionBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillQKVBatch, error) {
	return hipRunGemma4Q4PrefillQKVProjectionBatchWorkspaceTransient(ctx, driver, cfg, input, tokenCount, workspace, false)
}

func hipRunGemma4Q4PrefillQKVProjectionBatchWorkspaceTransient(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV bool) (*hipGemma4Q4PrefillQKVBatch, error) {
	return hipRunGemma4Q4PrefillQKVProjectionBatchWorkspaceTransientInto(ctx, driver, cfg, input, tokenCount, workspace, borrowRawKV, nil)
}

func hipGemma4Q4PrefillQKVProjectionBatchTripleCompatible(cfg hipGemma4Q4Layer0Config) bool {
	return !cfg.AttentionKEqV &&
		cfg.QueryProjection.WeightEncoding == 0 &&
		cfg.KeyProjection.WeightEncoding == 0 &&
		cfg.ValueProjection.WeightEncoding == 0 &&
		cfg.QueryProjection.Cols == cfg.KeyProjection.Cols &&
		cfg.QueryProjection.Cols == cfg.ValueProjection.Cols &&
		cfg.QueryProjection.GroupSize == cfg.KeyProjection.GroupSize &&
		cfg.QueryProjection.GroupSize == cfg.ValueProjection.GroupSize &&
		cfg.QueryProjection.quantBits() == cfg.KeyProjection.quantBits() &&
		cfg.QueryProjection.quantBits() == cfg.ValueProjection.quantBits()
}

func hipGemma4Q4PrefillQKVProjectionBatchPairCompatible(cfg hipGemma4Q4Layer0Config) bool {
	return cfg.AttentionKEqV &&
		cfg.QueryProjection.WeightEncoding == 0 &&
		cfg.KeyProjection.WeightEncoding == 0 &&
		cfg.QueryProjection.Cols == cfg.KeyProjection.Cols &&
		cfg.QueryProjection.GroupSize == cfg.KeyProjection.GroupSize &&
		cfg.QueryProjection.quantBits() == cfg.KeyProjection.quantBits()
}

func hipRunGemma4Q4PrefillQKVProjectionBatchWorkspaceTransientInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV bool, out *hipGemma4Q4PrefillQKVBatch) (*hipGemma4Q4PrefillQKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV hidden size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV input buffer shape mismatch", nil)
	}
	if out == nil {
		out = &hipGemma4Q4PrefillQKVBatch{}
	} else {
		*out = hipGemma4Q4PrefillQKVBatch{}
	}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	if workspace != nil && tokenCount == 1 && hipGemma4Q4PrefillQKVProjectionBatchTripleCompatible(cfg) {
		totalRows := cfg.QueryProjection.Rows + cfg.KeyProjection.Rows + cfg.ValueProjection.Rows
		output, workspaceErr := workspace.EnsureProjectionOutput(driver, totalRows)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		query, key, value, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx, driver, input, cfg.QueryProjection, cfg.KeyProjection, cfg.ValueProjection, output, workspace)
		if err != nil {
			return nil, err
		}
		out.queryView = query
		out.keyView = key
		out.valueView = value
		out.Query = &out.queryView
		out.Key = &out.keyView
		out.Value = &out.valueView
		success = true
		return out, nil
	}
	if workspace != nil && tokenCount == 1 && hipGemma4Q4PrefillQKVProjectionBatchPairCompatible(cfg) {
		totalRows := cfg.QueryProjection.Rows + cfg.KeyProjection.Rows
		output, workspaceErr := workspace.EnsureProjectionOutput(driver, totalRows)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		query, key, pairErr := hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace(ctx, driver, input, cfg.QueryProjection, cfg.KeyProjection, output, workspace)
		if pairErr != nil {
			return nil, pairErr
		}
		out.queryView = query
		out.keyView = key
		out.valueView = key
		out.valueView.borrowed = true
		out.Query = &out.queryView
		out.Key = &out.keyView
		out.Value = &out.valueView
		success = true
		return out, nil
	}
	if workspace != nil {
		queryOutput, workspaceErr := workspace.EnsureProjectionOutput(driver, tokenCount*cfg.QueryProjection.Rows)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if tokenCount == 1 {
			err = hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg.QueryProjection, queryOutput, workspace)
		} else {
			err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, input, cfg.QueryProjection, tokenCount, queryOutput)
		}
		if err != nil {
			return nil, err
		}
		out.Query = out.borrowQueryView(driver, "prefill query projection workspace view", queryOutput)
	} else if tokenCount == 1 {
		out.Query, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg.QueryProjection)
	} else {
		out.Query, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg.QueryProjection, tokenCount)
	}
	if err != nil {
		return nil, err
	}
	if workspace != nil && borrowRawKV {
		keyOutput, workspaceErr := workspace.EnsureKVProjectionOutput(driver, tokenCount*cfg.KeyProjection.Rows, 0)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if tokenCount == 1 {
			err = hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg.KeyProjection, keyOutput, workspace)
		} else {
			err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, input, cfg.KeyProjection, tokenCount, keyOutput)
		}
		if err != nil {
			return nil, err
		}
		out.Key = out.borrowKeyView(driver, "prefill key projection workspace view", keyOutput)
	} else if tokenCount == 1 {
		out.Key, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg.KeyProjection)
	} else {
		out.Key, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg.KeyProjection, tokenCount)
	}
	if err != nil {
		return nil, err
	}
	if cfg.AttentionKEqV {
		out.valueView = *out.Key
		out.valueView.borrowed = true
		out.Value = &out.valueView
	} else if workspace != nil && borrowRawKV {
		valueOutput, workspaceErr := workspace.EnsureKVProjectionOutput(driver, tokenCount*cfg.ValueProjection.Rows, 1)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if tokenCount == 1 {
			err = hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg.ValueProjection, valueOutput, workspace)
		} else {
			err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, input, cfg.ValueProjection, tokenCount, valueOutput)
		}
		if err != nil {
			return nil, err
		}
		out.Value = out.borrowValueView(driver, "prefill value projection workspace view", valueOutput)
	} else if tokenCount == 1 {
		out.Value, err = hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg.ValueProjection)
	} else {
		out.Value, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg.ValueProjection, tokenCount)
		if err != nil {
			return nil, err
		}
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillQKNormRoPEBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, startPosition int, epsilon float32) (*hipGemma4Q4PrefillRoPEQKBatch, error) {
	return hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspace(ctx, driver, cfg, qkv, tokenCount, startPosition, epsilon, nil)
}

func hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, startPosition int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillRoPEQKBatch, error) {
	return hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspaceTransient(ctx, driver, cfg, qkv, tokenCount, startPosition, epsilon, workspace, false)
}

func hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspaceTransient(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, startPosition int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV bool) (*hipGemma4Q4PrefillRoPEQKBatch, error) {
	return hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspaceTransientInto(ctx, driver, cfg, qkv, tokenCount, startPosition, epsilon, workspace, borrowRawKV, nil)
}

func hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspaceTransientInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, startPosition int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV bool, out *hipGemma4Q4PrefillRoPEQKBatch) (*hipGemma4Q4PrefillRoPEQKBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE token count must be positive", nil)
	}
	if startPosition < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE start position must be non-negative", nil)
	}
	keyHeads := firstPositiveInt(cfg.KeyHeads, 1)
	kvDim := cfg.keyValueDim()
	if cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 || cfg.QueryHeads <= 0 || keyHeads <= 0 || kvDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE layer geometry mismatch", nil)
	}
	if cfg.RoPEBase <= 0 || math.IsNaN(float64(cfg.RoPEBase)) || math.IsInf(float64(cfg.RoPEBase), 0) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE base must be positive and finite", nil)
	}
	if cfg.RoPERotaryDim <= 0 || cfg.RoPERotaryDim > cfg.HeadDim || cfg.RoPERotaryDim%2 != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE rotary dimension mismatch", nil)
	}
	if cfg.effectiveRoPEFrequencyScale() <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE frequency scale must be positive and finite", nil)
	}
	if qkv == nil || qkv.Query == nil || qkv.Query.Pointer() == 0 || qkv.Key == nil || qkv.Key.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE QKV buffers are required", nil)
	}
	queryRows := cfg.QueryHeads * cfg.HeadDim
	keyRows := kvDim
	if qkv.Query.Count() != tokenCount*queryRows || qkv.Query.SizeBytes() != uint64(qkv.Query.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE query buffer shape mismatch", nil)
	}
	if qkv.Key.Count() != tokenCount*keyRows || qkv.Key.SizeBytes() != uint64(qkv.Key.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE key buffer shape mismatch", nil)
	}
	if err := hipValidateGemma4Q4NormConfig("Gemma4Q4PrefillQueryNorm", cfg.QueryNorm, cfg.HeadDim); err != nil {
		return nil, err
	}
	if err := hipValidateGemma4Q4NormConfig("Gemma4Q4PrefillKeyNorm", cfg.KeyNorm, cfg.HeadDim); err != nil {
		return nil, err
	}
	queryNormCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, epsilon, cfg.HeadDim)
	keyNormCfg := hipGemma4Q4RoPENormConfig(cfg.KeyNorm, epsilon, cfg.HeadDim)
	ropeFrequencyDim, ropeRotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	ropeFrequencyScale := cfg.effectiveRoPEFrequencyScale()
	if out == nil {
		out = &hipGemma4Q4PrefillRoPEQKBatch{}
	} else {
		*out = hipGemma4Q4PrefillRoPEQKBatch{}
	}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	if workspace != nil && borrowRawKV && tokenCount == 1 {
		queryOutput, workspaceErr := workspace.EnsureRMSRoPEOutput(driver, qkv.Query.Count())
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		keyOutput, workspaceErr := workspace.EnsureKeyRMSRoPEOutput(driver, qkv.Key.Count())
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		err = hipRunRMSNormRoPEHeadsPairKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, qkv.Query, qkv.Key, queryNormCfg, keyNormCfg, cfg.QueryHeads, keyHeads, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, queryOutput, keyOutput, workspace)
		if err != nil {
			return nil, err
		}
		out.Query = out.borrowQueryView(driver, "prefill query rope workspace view", queryOutput)
		out.Key = out.borrowKeyView(driver, "prefill key rope workspace view", keyOutput)
		success = true
		return out, nil
	}
	if workspace != nil {
		queryOutput, workspaceErr := workspace.EnsureRMSRoPEOutput(driver, qkv.Query.Count())
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if tokenCount == 1 {
			err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, qkv.Query, queryNormCfg, cfg.QueryHeads, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, queryOutput, workspace)
		} else {
			err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx, driver, qkv.Query, queryNormCfg, cfg.QueryHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, queryOutput)
		}
		if err != nil {
			return nil, err
		}
		out.Query = out.borrowQueryView(driver, "prefill query rope workspace view", queryOutput)
	} else if tokenCount == 1 {
		out.Query, err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, qkv.Query, queryNormCfg, cfg.QueryHeads, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale)
	} else {
		out.Query, err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, qkv.Query, queryNormCfg, cfg.QueryHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale)
	}
	if err != nil {
		return nil, err
	}
	if workspace != nil && borrowRawKV {
		keyOutput, workspaceErr := workspace.EnsureKeyRMSRoPEOutput(driver, qkv.Key.Count())
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if tokenCount == 1 {
			err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(ctx, driver, qkv.Key, keyNormCfg, keyHeads, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, keyOutput, workspace)
		} else {
			err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx, driver, qkv.Key, keyNormCfg, keyHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale, keyOutput)
		}
		if err != nil {
			return nil, err
		}
		out.Key = out.borrowKeyView(driver, "prefill key rope workspace view", keyOutput)
	} else if tokenCount == 1 {
		out.Key, err = hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, qkv.Key, keyNormCfg, keyHeads, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale)
	} else {
		out.Key, err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, qkv.Key, keyNormCfg, keyHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, ropeFrequencyScale)
	}
	if err != nil {
		return nil, err
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillValueNormBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, epsilon float32) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillValueNormBatchWorkspace(ctx, driver, cfg, qkv, tokenCount, epsilon, nil, false)
}

func hipRunGemma4Q4PrefillValueNormBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, borrowedOutput bool) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillValueNormBatchWorkspaceView(ctx, driver, cfg, qkv, tokenCount, epsilon, workspace, borrowedOutput, nil)
}

func hipRunGemma4Q4PrefillValueNormBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, borrowedOutput bool, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm token count must be positive", nil)
	}
	kvDim := cfg.keyValueDim()
	keyHeads := firstPositiveInt(cfg.KeyHeads, 1)
	if cfg.HeadDim <= 0 || kvDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm head dim must be positive", nil)
	}
	if qkv == nil || qkv.Value == nil || qkv.Value.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm value buffer is required", nil)
	}
	if qkv.Value.Count() != tokenCount*kvDim || qkv.Value.SizeBytes() != uint64(qkv.Value.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm value buffer shape mismatch", nil)
	}
	normCfg := hipRMSNormDeviceWeightConfig{
		Count:          cfg.HeadDim,
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	if workspace != nil && borrowedOutput {
		output, err := workspace.EnsureRMSNoScaleOutput(driver, qkv.Value.Count())
		if err != nil {
			return nil, err
		}
		if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, qkv.Value, normCfg, tokenCount*keyHeads, output); err != nil {
			return nil, err
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill value norm workspace view", output.Pointer(), output.SizeBytes(), output.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, "prefill value norm workspace view", output.Pointer(), output.SizeBytes(), output.Count()), nil
	}
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, qkv.Value, normCfg, tokenCount*keyHeads)
}

func hipRunGemma4Q4PrefillDeviceKVBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	return hipRunGemma4Q4PrefillDeviceKVBatchWithPrior(ctx, driver, cfg, nil, qk, value, tokenCount, mode)
}

func hipRunGemma4Q4PrefillDeviceKVBatchWithPrior(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, prior *rocmDeviceKVCache, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	return hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptor(ctx, driver, cfg, prior, nil, qk, value, tokenCount, mode)
}

func hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptor(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	return hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorInto(ctx, driver, cfg, prior, priorDescriptorTable, qk, value, tokenCount, mode, nil)
}

func hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string, out *hipGemma4Q4PrefillDeviceKVBatch) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	return hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorIntoWithEngineConfig(ctx, driver, cfg, prior, priorDescriptorTable, qk, value, tokenCount, mode, out, defaultHIPGemma4Q4EngineConfig())
}

func hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorIntoWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string, out *hipGemma4Q4PrefillDeviceKVBatch, engineConfig hipGemma4Q4EngineConfig) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV token count must be positive", nil)
	}
	kvDim := cfg.keyValueDim()
	if cfg.HeadDim <= 0 || kvDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV head dim must be positive", nil)
	}
	if qk == nil || qk.Key == nil || qk.Key.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV key buffer is required", nil)
	}
	if value == nil || value.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV value buffer is required", nil)
	}
	if qk.Key.Count() != tokenCount*kvDim || qk.Key.SizeBytes() != uint64(qk.Key.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV key buffer shape mismatch", nil)
	}
	if value.Count() != tokenCount*kvDim || value.SizeBytes() != uint64(value.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV value buffer shape mismatch", nil)
	}
	if out == nil {
		out = &hipGemma4Q4PrefillDeviceKVBatch{}
	} else {
		*out = hipGemma4Q4PrefillDeviceKVBatch{}
	}
	var cache *rocmDeviceKVCache
	var err error
	if prior != nil {
		if prior.closed {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV cache is closed", nil)
		}
		if prior.TokenCount() <= 0 {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV cache is empty", nil)
		}
		if mode != "" && prior.mode != "" && prior.mode != mode {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV mode mismatch", nil)
		}
		window := 0
		if cfg.SlidingWindow > 0 {
			window = cfg.SlidingWindow + tokenCount
		}
		cache, err = prior.withAppendedDeviceRowsWindowWithEngineConfig(ctx, qk.Key, value, kvDim, kvDim, tokenCount, window, engineConfig)
	} else {
		cache, err = newROCmDeviceKVCacheFromDeviceRowsWithEngineConfig(ctx, driver, firstNonEmptyString(mode, rocmKVCacheModeFP16), engineConfig.deviceKVBlockSizeForSlidingWindow(cfg.SlidingWindow), qk.Key, value, kvDim, kvDim, tokenCount, 0, engineConfig)
	}
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = cache.Close()
		}
	}()
	var table *rocmDeviceKVDescriptorTable
	if prior != nil && priorDescriptorTable != nil {
		table, err = cache.KernelDescriptorTableFromAppendedToken(ctx, prior, priorDescriptorTable)
	}
	if table == nil && err == nil {
		label := "prefill_new_device_kv"
		if prior != nil {
			label = "prefill_append_rows"
		}
		table, err = cache.kernelDescriptorTableLabeled("rocm.KVCache.DeviceDescriptor", label)
	}
	if err != nil {
		return nil, err
	}
	defer func() {
		if !success {
			_ = table.Close()
		}
	}()
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		return nil, err
	}
	success = true
	out.Cache = cache
	out.DescriptorTable = table
	out.Launch = launch
	out.RetainWindow = cfg.SlidingWindow
	return out, nil
}

func hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorRawValueNormIntoWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, qk *hipGemma4Q4PrefillRoPEQKBatch, rawValue *hipDeviceByteBuffer, tokenCount int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace, out *hipGemma4Q4PrefillDeviceKVBatch, engineConfig hipGemma4Q4EngineConfig) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount != 1 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm device KV requires one token", nil)
	}
	kvDim := cfg.keyValueDim()
	keyHeads := firstPositiveInt(cfg.KeyHeads, 1)
	if cfg.HeadDim <= 0 || kvDim <= 0 || keyHeads <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm device KV head dim must be positive", nil)
	}
	if qk == nil || qk.Key == nil || qk.Key.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm device KV key buffer is required", nil)
	}
	if rawValue == nil || rawValue.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm device KV value buffer is required", nil)
	}
	if qk.Key.Count() != kvDim || qk.Key.SizeBytes() != uint64(qk.Key.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm device KV key buffer shape mismatch", nil)
	}
	if rawValue.Count() != kvDim || rawValue.SizeBytes() != uint64(rawValue.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm device KV value buffer shape mismatch", nil)
	}
	if prior == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm prior device KV cache is required", nil)
	}
	if prior.closed {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm prior device KV cache is closed", nil)
	}
	if prior.TokenCount() <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm prior device KV cache is empty", nil)
	}
	if mode != "" && prior.mode != "" && prior.mode != mode {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill fused value norm prior device KV mode mismatch", nil)
	}
	if out == nil {
		out = &hipGemma4Q4PrefillDeviceKVBatch{}
	} else {
		*out = hipGemma4Q4PrefillDeviceKVBatch{}
	}
	window := 0
	if cfg.SlidingWindow > 0 {
		window = cfg.SlidingWindow + tokenCount
	}
	var cache *rocmDeviceKVCache
	var table *rocmDeviceKVDescriptorTable
	var err error
	if priorDescriptorTable != nil {
		var fused bool
		cache, table, fused, err = prior.withAppendedDeviceRowsValueNormDescriptorGrowLastPageWithWorkspaceAndEngineConfig(ctx, qk.Key, rawValue, priorDescriptorTable, kvDim, kvDim, tokenCount, cfg.HeadDim, keyHeads, epsilon, window, workspace, engineConfig)
		if err != nil {
			return nil, err
		}
		if !fused {
			cache = nil
			table = nil
		}
	}
	if cache == nil {
		cache, err = prior.withAppendedDeviceRowsValueNormWindowWithWorkspaceAndEngineConfig(ctx, qk.Key, rawValue, kvDim, kvDim, tokenCount, cfg.HeadDim, keyHeads, epsilon, window, workspace, engineConfig)
		if err != nil {
			return nil, err
		}
	}
	success := false
	defer func() {
		if !success {
			_ = cache.Close()
		}
	}()
	if table == nil && priorDescriptorTable != nil {
		table, err = cache.KernelDescriptorTableFromAppendedToken(ctx, prior, priorDescriptorTable)
	}
	if table == nil && err == nil {
		table, err = cache.kernelDescriptorTableLabeled("rocm.KVCache.DeviceDescriptor", "prefill_append_fused_value_norm")
	}
	if err != nil {
		return nil, err
	}
	defer func() {
		if !success {
			_ = table.Close()
		}
	}()
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		return nil, err
	}
	success = true
	out.Cache = cache
	out.DescriptorTable = table
	out.Launch = launch
	out.RetainWindow = cfg.SlidingWindow
	return out, nil
}

func hipRunGemma4Q4PrefillLayerKVBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, startPosition int, epsilon float32, mode string) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPrior(ctx, driver, cfg, input, nil, tokenCount, startPosition, epsilon, mode)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPrior(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, tokenCount int, startPosition int, epsilon float32, mode string) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPriorWorkspace(ctx, driver, cfg, input, prior, tokenCount, startPosition, epsilon, mode, nil)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPriorWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, tokenCount int, startPosition int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPriorWorkspaceTransient(ctx, driver, cfg, input, prior, tokenCount, startPosition, epsilon, mode, workspace, false, false)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPriorWorkspaceTransient(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, tokenCount int, startPosition int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV, borrowRetainedKV bool) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransient(ctx, driver, cfg, input, prior, nil, tokenCount, startPosition, epsilon, mode, workspace, borrowRawKV, borrowRetainedKV)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransient(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, tokenCount int, startPosition int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV, borrowRetainedKV bool) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientInto(ctx, driver, cfg, input, prior, priorDescriptorTable, tokenCount, startPosition, epsilon, mode, workspace, borrowRawKV, borrowRetainedKV, nil)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, tokenCount int, startPosition int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV, borrowRetainedKV bool, out *hipGemma4Q4PrefillLayerKVBatch) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientIntoWithEngineConfig(ctx, driver, cfg, input, prior, priorDescriptorTable, tokenCount, startPosition, epsilon, mode, workspace, borrowRawKV, borrowRetainedKV, out, defaultHIPGemma4Q4EngineConfig())
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientIntoWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, tokenCount int, startPosition int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV, borrowRetainedKV bool, out *hipGemma4Q4PrefillLayerKVBatch, engineConfig hipGemma4Q4EngineConfig) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientPrecomputedInputNormIntoWithEngineConfig(ctx, driver, cfg, input, nil, prior, priorDescriptorTable, tokenCount, startPosition, epsilon, mode, workspace, borrowRawKV, borrowRetainedKV, out, engineConfig)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientPrecomputedInputNormIntoWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input, precomputedInputNorm *hipDeviceByteBuffer, prior *rocmDeviceKVCache, priorDescriptorTable *rocmDeviceKVDescriptorTable, tokenCount int, startPosition int, epsilon float32, mode string, workspace *hipAttentionHeadsChunkedWorkspace, borrowRawKV, borrowRetainedKV bool, out *hipGemma4Q4PrefillLayerKVBatch, engineConfig hipGemma4Q4EngineConfig) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer KV token count must be positive", nil)
	}
	if startPosition < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer KV start position must be non-negative", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer KV input buffer is required", nil)
	}
	if prior != nil {
		priorTokens := prior.TokenCount()
		if priorTokens != startPosition && (cfg.SlidingWindow <= 0 || priorTokens <= 0 || priorTokens > startPosition) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV token count must match start position or a retained sliding window", nil)
		}
	}
	if out == nil {
		out = &hipGemma4Q4PrefillLayerKVBatch{}
	} else {
		*out = hipGemma4Q4PrefillLayerKVBatch{}
	}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	if precomputedInputNorm != nil {
		out.InputNorm, err = hipBorrowGemma4Q4PrefillPrecomputedInputNorm(driver, cfg, precomputedInputNorm, tokenCount, &out.inputNormView)
	} else {
		out.InputNorm, err = hipRunGemma4Q4PrefillInputNormBatchWorkspaceView(ctx, driver, cfg, input, tokenCount, workspace, &out.inputNormView)
	}
	if err != nil {
		return nil, err
	}
	out.QKV = &out.qkvStorage
	_, err = hipRunGemma4Q4PrefillQKVProjectionBatchWorkspaceTransientInto(ctx, driver, cfg, out.InputNorm, tokenCount, workspace, borrowRawKV, out.QKV)
	if err != nil {
		return nil, err
	}
	out.QK = &out.qkStorage
	_, err = hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspaceTransientInto(ctx, driver, cfg, out.QKV, tokenCount, startPosition, epsilon, workspace, borrowRetainedKV, out.QK)
	if err != nil {
		return nil, err
	}
	if workspace != nil && borrowRetainedKV && prior != nil && tokenCount == 1 && out.QKV != nil && out.QKV.Value != nil {
		out.DeviceKV = &out.deviceKVStorage
		_, err = hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorRawValueNormIntoWithEngineConfig(ctx, driver, cfg, prior, priorDescriptorTable, out.QK, out.QKV.Value, tokenCount, epsilon, mode, workspace, out.DeviceKV, engineConfig)
		if err != nil {
			return nil, err
		}
		success = true
		return out, nil
	}
	borrowValueNorm := workspace != nil && borrowRetainedKV
	out.Value, err = hipRunGemma4Q4PrefillValueNormBatchWorkspaceView(ctx, driver, cfg, out.QKV, tokenCount, epsilon, workspace, borrowValueNorm, &out.valueView)
	if err != nil {
		return nil, err
	}
	out.DeviceKV = &out.deviceKVStorage
	_, err = hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorIntoWithEngineConfig(ctx, driver, cfg, prior, priorDescriptorTable, out.QK, out.Value, tokenCount, mode, out.DeviceKV, engineConfig)
	if err != nil {
		return nil, err
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKV(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, sharedSource *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, startPosition int, epsilon float32) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspace(ctx, driver, cfg, input, sharedSource, tokenCount, startPosition, epsilon, nil)
}

func hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, sharedSource *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, startPosition int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspaceInto(ctx, driver, cfg, input, sharedSource, tokenCount, startPosition, epsilon, workspace, nil)
}

func hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspaceInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, sharedSource *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, startPosition int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, out *hipGemma4Q4PrefillLayerKVBatch) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspacePrecomputedInputNormInto(ctx, driver, cfg, input, nil, sharedSource, tokenCount, startPosition, epsilon, workspace, out)
}

func hipBorrowGemma4Q4PrefillSharedDeviceKVLayerInto(sharedSource, out *hipGemma4Q4PrefillLayerKVBatch) error {
	if sharedSource == nil || sharedSource.DeviceKV == nil || sharedSource.DeviceKV.Cache == nil || sharedSource.DeviceKV.DescriptorTable == nil {
		return core.E(hipGemma4Q4Layer0Operation, "prefill shared layer source device KV is required", nil)
	}
	if out == nil {
		return core.E(hipGemma4Q4Layer0Operation, "prefill shared layer output is required", nil)
	}
	*out = hipGemma4Q4PrefillLayerKVBatch{}
	cache, err := sharedSource.DeviceKV.Cache.borrowedAlias()
	if err != nil {
		return err
	}
	table, err := sharedSource.DeviceKV.DescriptorTable.borrowedAlias()
	if err != nil {
		_ = cache.Close()
		return err
	}
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		_ = cache.Close()
		return err
	}
	out.DeviceKV = &out.deviceKVStorage
	out.DeviceKV.Cache = cache
	out.DeviceKV.DescriptorTable = table
	out.DeviceKV.Launch = launch
	out.DeviceKV.RetainWindow = sharedSource.DeviceKV.RetainWindow
	return nil
}

func hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspacePrecomputedInputNormInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input, precomputedInputNorm *hipDeviceByteBuffer, sharedSource *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, startPosition int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, out *hipGemma4Q4PrefillLayerKVBatch) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer query token count must be positive", nil)
	}
	if startPosition < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer query start position must be non-negative", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer query input buffer is required", nil)
	}
	if sharedSource == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer source KV is required", nil)
	}
	shared := sharedSource.DeviceKV
	if shared == nil || shared.Cache == nil || shared.DescriptorTable == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer device KV is required", nil)
	}
	if out == nil {
		out = &hipGemma4Q4PrefillLayerKVBatch{}
	} else {
		*out = hipGemma4Q4PrefillLayerKVBatch{}
	}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	if precomputedInputNorm != nil {
		out.InputNorm, err = hipBorrowGemma4Q4PrefillPrecomputedInputNorm(driver, cfg, precomputedInputNorm, tokenCount, &out.inputNormView)
	} else {
		out.InputNorm, err = hipRunGemma4Q4PrefillInputNormBatchWorkspaceView(ctx, driver, cfg, input, tokenCount, workspace, &out.inputNormView)
	}
	if err != nil {
		return nil, err
	}
	out.QKV = &out.qkvStorage
	var query *hipDeviceByteBuffer
	if workspace != nil {
		queryOutput, workspaceErr := workspace.EnsureProjectionOutput(driver, tokenCount*cfg.QueryProjection.Rows)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if tokenCount == 1 {
			err = hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, out.InputNorm, cfg.QueryProjection, queryOutput, workspace)
		} else {
			err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, out.InputNorm, cfg.QueryProjection, tokenCount, queryOutput)
		}
		if err != nil {
			return nil, err
		}
		query = out.QKV.borrowQueryView(driver, "prefill shared query projection workspace view", queryOutput)
	} else {
		query, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, out.InputNorm, cfg.QueryProjection, tokenCount)
		if err != nil {
			return nil, err
		}
	}
	out.QKV.Query = query
	queryNormCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, epsilon, cfg.HeadDim)
	ropeFrequencyDim, ropeRotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	out.QK = &out.qkStorage
	if workspace != nil && tokenCount == 1 {
		out.QueryRMSRoPEAttention = hipGemma4Q4QueryRMSRoPEAttention{
			Enabled:        true,
			Input:          query,
			NormConfig:     queryNormCfg,
			StartPosition:  startPosition,
			Base:           cfg.RoPEBase,
			FrequencyDim:   ropeFrequencyDim,
			RotaryCount:    ropeRotaryCount,
			FrequencyScale: cfg.effectiveRoPEFrequencyScale(),
		}
	} else if workspace != nil {
		queryOutput, workspaceErr := workspace.EnsureRMSRoPEOutput(driver, query.Count())
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx, driver, query, queryNormCfg, cfg.QueryHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, cfg.effectiveRoPEFrequencyScale(), queryOutput)
		if err != nil {
			return nil, err
		}
		out.QK.Query = out.QK.borrowQueryView(driver, "prefill shared query rope workspace view", queryOutput)
	} else {
		out.QK.Query, err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScale(ctx, driver, query, queryNormCfg, cfg.QueryHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount, cfg.effectiveRoPEFrequencyScale())
		if err != nil {
			return nil, err
		}
	}
	if sharedSource.QK != nil && sharedSource.QK.Key != nil && sharedSource.Value != nil &&
		!sharedSource.QK.Key.borrowed && !sharedSource.Value.borrowed {
		out.SharedKey = sharedSource.QK.Key
		out.SharedVal = sharedSource.Value
	}
	cache, err := shared.Cache.borrowedAlias()
	if err != nil {
		return nil, err
	}
	table, err := shared.DescriptorTable.borrowedAlias()
	if err != nil {
		_ = cache.Close()
		return nil, err
	}
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		_ = cache.Close()
		return nil, err
	}
	out.DeviceKV = &out.deviceKVStorage
	out.DeviceKV.Cache = cache
	out.DeviceKV.DescriptorTable = table
	out.DeviceKV.Launch = launch
	out.DeviceKV.RetainWindow = shared.RetainWindow
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillAttentionBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillAttentionBatchWorkspace(ctx, driver, cfg, layer, tokenCount, queryStartToken, nil)
}

func hipRunGemma4Q4PrefillAttentionBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillAttentionBatchWorkspaceView(ctx, driver, cfg, layer, tokenCount, queryStartToken, workspace, nil)
}

func hipRunGemma4Q4PrefillAttentionBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int, workspace *hipAttentionHeadsChunkedWorkspace, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention token count must be positive", nil)
	}
	if queryStartToken < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention query start token must be non-negative", nil)
	}
	if cfg.HeadDim <= 0 || cfg.QueryHeads <= 0 || firstPositiveInt(cfg.KeyHeads, 1) <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention layer geometry mismatch", nil)
	}
	if layer == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention layer is required", nil)
	}
	queryCount := tokenCount * cfg.QueryHeads * cfg.HeadDim
	if layer.AttentionOverride != nil {
		override := layer.AttentionOverride
		if override.Pointer() == 0 || override.Count() != queryCount || override.SizeBytes() != uint64(queryCount*4) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention override buffer shape mismatch", nil)
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill attention override view", override.Pointer(), override.SizeBytes(), override.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, "prefill attention override view", override.Pointer(), override.SizeBytes(), override.Count()), nil
	}
	if layer.DeviceKV == nil || layer.DeviceKV.Cache == nil || layer.DeviceKV.DescriptorTable == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention Q/K/V device buffers are required", nil)
	}
	query := (*hipDeviceByteBuffer)(nil)
	fusedQueryAttention := layer.QueryRMSRoPEAttention.Enabled
	if fusedQueryAttention {
		query = layer.QueryRMSRoPEAttention.Input
	} else if layer.QK != nil {
		query = layer.QK.Query
	}
	if query == nil || query.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention query buffer is required", nil)
	}
	if query.Count() != queryCount || query.SizeBytes() != uint64(queryCount*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention query buffer shape mismatch", nil)
	}
	if uint64(queryStartToken)+uint64(tokenCount) > uint64(layer.DeviceKV.Cache.TokenCount()) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention causal window exceeds device KV token count", nil)
	}
	var output *hipDeviceByteBuffer
	var err error
	if workspace != nil {
		workspaceOutput, workspaceErr := workspace.EnsureBatchAttentionOutput(driver, queryCount)
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill attention batch workspace view", workspaceOutput.Pointer(), workspaceOutput.SizeBytes(), workspaceOutput.Count())
			output = view
		} else {
			output = hipBorrowDeviceByteBuffer(driver, "prefill attention batch workspace view", workspaceOutput.Pointer(), workspaceOutput.SizeBytes(), workspaceOutput.Count())
		}
	} else {
		output, err = hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "Gemma4 q4 prefill attention batch output", uint64(queryCount*4), queryCount)
		if err != nil {
			return nil, err
		}
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	attentionReq := hipAttentionHeadsBatchCausalDeviceRequest{
		Dim:              cfg.HeadDim,
		TokenCount:       layer.DeviceKV.Cache.TokenCount(),
		HeadCount:        cfg.QueryHeads,
		KeyHeads:         firstPositiveInt(cfg.KeyHeads, 1),
		QueryCount:       tokenCount,
		QueryStartToken:  queryStartToken,
		WindowSize:       cfg.SlidingWindow,
		Scale:            hipGemma4Q4AttentionScale(cfg.HeadDim),
		VisibleTokenCaps: layer.VisibleTokenCaps,
	}
	contiguousKey := layer.QK.Key
	contiguousValue := layer.Value
	if contiguousKey == nil || contiguousValue == nil {
		contiguousKey = layer.SharedKey
		contiguousValue = layer.SharedVal
	}
	if queryStartToken == 0 && layer.DeviceKV.Cache.TokenCount() == tokenCount && contiguousKey != nil && contiguousValue != nil {
		attentionReq.Key = contiguousKey
		attentionReq.Value = contiguousValue
	} else {
		attentionReq.DeviceKV = layer.DeviceKV.Cache
		attentionReq.DescriptorTable = layer.DeviceKV.DescriptorTable
	}
	if fusedQueryAttention && (attentionReq.VisibleTokenCaps != nil || hipAttentionHeadsBatchChunkedEligible(attentionReq, workspace)) {
		queryOutput, workspaceErr := workspace.EnsureRMSRoPEOutput(driver, query.Count())
		if workspaceErr != nil {
			return nil, workspaceErr
		}
		fused := layer.QueryRMSRoPEAttention
		err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfigFrequencyScaleOutput(ctx, driver, query, fused.NormConfig, cfg.QueryHeads, tokenCount, fused.StartPosition, fused.Base, fused.FrequencyDim, fused.RotaryCount, fused.FrequencyScale, queryOutput)
		if err != nil {
			return nil, err
		}
		query = queryOutput
		fusedQueryAttention = false
	}
	queryChunkTokens := hipGemma4Q4PrefillAttentionQueryChunkTokens()
	if workspace != nil && queryChunkTokens > 0 && tokenCount > queryChunkTokens {
		queryRowCount := cfg.QueryHeads * cfg.HeadDim
		queryRowBytes := uint64(queryRowCount * 4)
		for tokenOffset := 0; tokenOffset < tokenCount; tokenOffset += queryChunkTokens {
			chunkTokens := queryChunkTokens
			if remaining := tokenCount - tokenOffset; remaining < chunkTokens {
				chunkTokens = remaining
			}
			chunkCount := chunkTokens * queryRowCount
			chunkBytes := uint64(chunkTokens) * queryRowBytes
			chunkQuery := hipBorrowDeviceByteBufferValue(driver, "prefill attention query chunk view", query.Pointer()+nativeDevicePointer(uint64(tokenOffset)*queryRowBytes), chunkBytes, chunkCount)
			chunkOutput := hipBorrowDeviceByteBufferValue(driver, "prefill attention output chunk view", output.Pointer()+nativeDevicePointer(uint64(tokenOffset)*queryRowBytes), chunkBytes, chunkCount)
			chunkReq := attentionReq
			chunkReq.QueryCount = chunkTokens
			chunkReq.QueryStartToken = queryStartToken + tokenOffset
			var chunkCaps hipDeviceTokenBuffer
			if attentionReq.VisibleTokenCaps != nil {
				chunkCaps = hipDeviceTokenBuffer{
					driver:    driver,
					pointer:   attentionReq.VisibleTokenCaps.Pointer() + nativeDevicePointer(tokenOffset*4),
					count:     chunkTokens,
					sizeBytes: uint64(chunkTokens * 4),
					borrowed:  true,
				}
				chunkReq.VisibleTokenCaps = &chunkCaps
			}
			if err := hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, chunkReq, &chunkQuery, &chunkOutput, workspace); err != nil {
				return nil, err
			}
		}
	} else {
		if fusedQueryAttention {
			err = hipRunAttentionHeadsBatchCausalQueryRMSRoPEOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, attentionReq, query, layer.QueryRMSRoPEAttention.NormConfig, layer.QueryRMSRoPEAttention.StartPosition, layer.QueryRMSRoPEAttention.Base, layer.QueryRMSRoPEAttention.FrequencyDim, layer.QueryRMSRoPEAttention.RotaryCount, layer.QueryRMSRoPEAttention.FrequencyScale, output, workspace)
		} else {
			err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, attentionReq, query, output, workspace)
		}
		if err != nil {
			return nil, err
		}
	}
	success = true
	return output, nil
}

func hipRunGemma4Q4PrefillResidualAddNormBatch(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillResidualAddNormBatchWorkspace(ctx, driver, input, residual, residualCfg, normCfg, tokenCount, outputScale, nil)
}

func hipRunGemma4Q4PrefillResidualAddNormBatchWorkspace(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillResidualAddNormBatchWorkspaceView(ctx, driver, input, residual, residualCfg, normCfg, tokenCount, outputScale, workspace, nil, nil)
}

func hipRunGemma4Q4PrefillResidualAddNormBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace, residualView, normView *hipDeviceByteBuffer) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, nil, err
	}
	if tokenCount <= 0 {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm token count must be positive", nil)
	}
	if residualCfg.Count <= 0 || residualCfg.Count != normCfg.Count {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm dimensions must be positive and equal", nil)
	}
	if input == nil || input.Pointer() == 0 || residual == nil || residual.Pointer() == 0 {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm input buffers are required", nil)
	}
	wantCount := tokenCount * residualCfg.Count
	if input.Count() != wantCount || residual.Count() != wantCount ||
		input.SizeBytes() != uint64(wantCount*4) || residual.SizeBytes() != uint64(wantCount*4) {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm buffer shape mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm output scale must be finite", nil)
	}
	if workspace != nil && tokenCount <= 2 {
		residualOutput, err := workspace.EnsureRMSResidualOutput(driver, wantCount)
		if err != nil {
			return nil, nil, err
		}
		normOutput, err := workspace.EnsureRMSNormOutput(driver, wantCount)
		if err != nil {
			return nil, nil, err
		}
		if tokenCount == 1 {
			err = hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, residual, residualCfg, normCfg, residualOutput, normOutput, outputScale)
		} else {
			err = hipRunGemma4Q4PrefillResidualAddNormSmallBatchOutput(ctx, driver, input, residual, residualCfg, normCfg, tokenCount, outputScale, residualOutput, normOutput)
		}
		if err != nil {
			return nil, nil, err
		}
		if residualView != nil && normView != nil {
			*residualView = hipBorrowDeviceByteBufferValue(driver, "prefill residual-add workspace view", residualOutput.Pointer(), residualOutput.SizeBytes(), residualOutput.Count())
			*normView = hipBorrowDeviceByteBufferValue(driver, "prefill residual-add norm workspace view", normOutput.Pointer(), normOutput.SizeBytes(), normOutput.Count())
			return residualView, normView, nil
		}
		residualOutputView := hipBorrowDeviceByteBuffer(driver, "prefill residual-add workspace view", residualOutput.Pointer(), residualOutput.SizeBytes(), residualOutput.Count())
		normOutputView := hipBorrowDeviceByteBuffer(driver, "prefill residual-add norm workspace view", normOutput.Pointer(), normOutput.SizeBytes(), normOutput.Count())
		return residualOutputView, normOutputView, nil
	}
	if tokenCount == 1 {
		return hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx, driver, input, residual, residualCfg, normCfg, outputScale)
	}
	if tokenCount == 2 {
		return hipRunGemma4Q4PrefillResidualAddNormSmallBatch(ctx, driver, input, residual, residualCfg, normCfg, tokenCount, outputScale)
	}
	var normalizedInput *hipDeviceByteBuffer
	normalizedInputBorrowed := false
	var err error
	if workspace != nil {
		normalizedInput, err = workspace.EnsureRMSNormOutput(driver, wantCount)
		if err == nil {
			err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, residualCfg, tokenCount, normalizedInput)
		}
		normalizedInputBorrowed = err == nil
	} else {
		normalizedInput, err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, residualCfg, tokenCount)
	}
	if err != nil {
		return nil, nil, err
	}
	if !normalizedInputBorrowed {
		defer normalizedInput.Close()
	}
	if workspace != nil && outputScale == 1 {
		residualOutput, err := workspace.EnsureRMSResidualOutput(driver, wantCount)
		if err != nil {
			return nil, nil, err
		}
		if err := hipRunVectorAddDeviceKernelOutput(ctx, driver, normalizedInput, residual, residualOutput); err != nil {
			return nil, nil, err
		}
		normOutput, err := workspace.EnsureRMSNormOutput(driver, wantCount)
		if err != nil {
			return nil, nil, err
		}
		if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, residualOutput, normCfg, tokenCount, normOutput); err != nil {
			return nil, nil, err
		}
		if residualView != nil && normView != nil {
			*residualView = hipBorrowDeviceByteBufferValue(driver, "prefill residual-add workspace view", residualOutput.Pointer(), residualOutput.SizeBytes(), residualOutput.Count())
			*normView = hipBorrowDeviceByteBufferValue(driver, "prefill residual-add norm workspace view", normOutput.Pointer(), normOutput.SizeBytes(), normOutput.Count())
			return residualView, normView, nil
		}
		residualOutputView := hipBorrowDeviceByteBuffer(driver, "prefill residual-add workspace view", residualOutput.Pointer(), residualOutput.SizeBytes(), residualOutput.Count())
		normOutputView := hipBorrowDeviceByteBuffer(driver, "prefill residual-add norm workspace view", normOutput.Pointer(), normOutput.SizeBytes(), normOutput.Count())
		return residualOutputView, normOutputView, nil
	}
	residualOutput, err := hipRunVectorAddDeviceKernel(ctx, driver, normalizedInput, residual)
	if err != nil {
		return nil, nil, err
	}
	if outputScale != 1 {
		scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, residualOutput, outputScale)
		if err != nil {
			_ = residualOutput.Close()
			return nil, nil, err
		}
		_ = residualOutput.Close()
		residualOutput = scaled
	}
	success := false
	defer func() {
		if !success {
			_ = residualOutput.Close()
		}
	}()
	normOutput, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, residualOutput, normCfg, tokenCount)
	if err != nil {
		return nil, nil, err
	}
	success = true
	return residualOutput, normOutput, nil
}

func hipRunGemma4Q4PrefillResidualAddBatch(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillResidualAddBatchWorkspace(ctx, driver, input, residual, cfg, tokenCount, outputScale, nil)
}

func hipRunGemma4Q4PrefillResidualAddBatchWorkspace(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutput(ctx, driver, input, residual, cfg, tokenCount, outputScale, workspace, nil, "")
}

func hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutput(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace, output *hipDeviceByteBuffer, label string) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutputView(ctx, driver, input, residual, cfg, tokenCount, outputScale, workspace, output, label, nil)
}

func hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutputView(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, workspace *hipAttentionHeadsChunkedWorkspace, output *hipDeviceByteBuffer, label string, view *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add token count must be positive", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add dimension must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 || residual == nil || residual.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add input buffers are required", nil)
	}
	wantCount := tokenCount * cfg.Count
	if input.Count() != wantCount || residual.Count() != wantCount ||
		input.SizeBytes() != uint64(wantCount*4) || residual.SizeBytes() != uint64(wantCount*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add buffer shape mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add output scale must be finite", nil)
	}
	if output != nil && tokenCount <= 2 {
		if tokenCount == 1 {
			if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, residual, cfg, output, outputScale); err != nil {
				return nil, err
			}
		} else {
			if err := hipRunGemma4Q4PrefillResidualAddSmallBatchOutput(ctx, driver, input, residual, cfg, tokenCount, outputScale, output); err != nil {
				return nil, err
			}
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, label, output.Pointer(), output.SizeBytes(), output.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, label, output.Pointer(), output.SizeBytes(), output.Count()), nil
	}
	if tokenCount == 1 {
		return hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx, driver, input, residual, cfg, outputScale)
	}
	if tokenCount == 2 {
		return hipRunGemma4Q4PrefillResidualAddSmallBatch(ctx, driver, input, residual, cfg, tokenCount, outputScale)
	}
	var normalizedInput *hipDeviceByteBuffer
	normalizedInputBorrowed := false
	var err error
	if workspace != nil {
		normalizedInput, err = workspace.EnsureRMSNormOutput(driver, wantCount)
		if err == nil {
			err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, input, cfg, tokenCount, normalizedInput)
		}
		normalizedInputBorrowed = err == nil
	} else {
		normalizedInput, err = hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, cfg, tokenCount)
	}
	if err != nil {
		return nil, err
	}
	if !normalizedInputBorrowed {
		defer normalizedInput.Close()
	}
	if workspace != nil {
		residualOutput, err := workspace.EnsureRMSResidualOutput(driver, wantCount)
		if err != nil {
			return nil, err
		}
		if err := hipRunVectorAddDeviceKernelOutput(ctx, driver, normalizedInput, residual, residualOutput); err != nil {
			return nil, err
		}
		if outputScale != 1 {
			scaleOutput := output
			scaleLabel := label
			if scaleOutput == nil {
				scaleOutput, err = workspace.EnsureRMSNormOutput(driver, wantCount)
				if err != nil {
					return nil, err
				}
				scaleLabel = "prefill residual-add scaled workspace view"
			}
			if err := hipRunVectorScaleDeviceKernelOutput(ctx, driver, residualOutput, outputScale, scaleOutput); err != nil {
				return nil, err
			}
			if view != nil {
				*view = hipBorrowDeviceByteBufferValue(driver, scaleLabel, scaleOutput.Pointer(), scaleOutput.SizeBytes(), scaleOutput.Count())
				return view, nil
			}
			return hipBorrowDeviceByteBuffer(driver, scaleLabel, scaleOutput.Pointer(), scaleOutput.SizeBytes(), scaleOutput.Count()), nil
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill residual-add workspace view", residualOutput.Pointer(), residualOutput.SizeBytes(), residualOutput.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, "prefill residual-add workspace view", residualOutput.Pointer(), residualOutput.SizeBytes(), residualOutput.Count()), nil
	}
	residualOutput, err := hipRunVectorAddDeviceKernel(ctx, driver, normalizedInput, residual)
	if err != nil {
		return nil, err
	}
	if outputScale == 1 {
		return residualOutput, nil
	}
	scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, residualOutput, outputScale)
	if err != nil {
		_ = residualOutput.Close()
		return nil, err
	}
	_ = residualOutput.Close()
	return scaled, nil
}

func hipRunGemma4Q4PrefillResidualAddNormSmallBatch(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	residualOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddNormLaunch", "prefill residual-add output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, nil, err
	}
	normOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddNormLaunch", "prefill residual-add norm output", input.SizeBytes(), input.Count())
	if err != nil {
		_ = residualOutput.Close()
		return nil, nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = normOutput.Close()
			_ = residualOutput.Close()
		}
	}()
	if err := hipRunGemma4Q4PrefillResidualAddNormSmallBatchOutput(ctx, driver, input, residual, residualCfg, normCfg, tokenCount, outputScale, residualOutput, normOutput); err != nil {
		return nil, nil, err
	}
	success = true
	return residualOutput, normOutput, nil
}

func hipRunGemma4Q4PrefillResidualAddNormSmallBatchOutput(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, residualOutput, normOutput *hipDeviceByteBuffer) error {
	wantCount := tokenCount * residualCfg.Count
	if residualOutput == nil || residualOutput.Pointer() == 0 || residualOutput.Count() != wantCount || residualOutput.SizeBytes() != uint64(wantCount*4) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill residual-add output buffer shape mismatch", nil)
	}
	if normOutput == nil || normOutput.Pointer() == 0 || normOutput.Count() != wantCount || normOutput.SizeBytes() != uint64(wantCount*4) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill residual-add norm output buffer shape mismatch", nil)
	}
	rowBytes := uint64(residualCfg.Count * 4)
	for token := 0; token < tokenCount; token++ {
		offset := nativeDevicePointer(token * residualCfg.Count * 4)
		rowInput := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add-norm input row", input.Pointer()+offset, rowBytes, residualCfg.Count)
		rowResidual := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add-norm residual row", residual.Pointer()+offset, rowBytes, residualCfg.Count)
		rowResidualOutput := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add-norm residual output row", residualOutput.Pointer()+offset, rowBytes, residualCfg.Count)
		rowNormOutput := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add-norm norm output row", normOutput.Pointer()+offset, rowBytes, residualCfg.Count)
		if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutput(ctx, driver, &rowInput, &rowResidual, residualCfg, normCfg, &rowResidualOutput, &rowNormOutput, outputScale); err != nil {
			return err
		}
	}
	return nil
}

func hipRunGemma4Q4PrefillResidualAddSmallBatch(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32) (*hipDeviceByteBuffer, error) {
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "prefill residual-add output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunGemma4Q4PrefillResidualAddSmallBatchOutput(ctx, driver, input, residual, cfg, tokenCount, outputScale, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunGemma4Q4PrefillResidualAddSmallBatchOutput(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32, output *hipDeviceByteBuffer) error {
	wantCount := tokenCount * cfg.Count
	if output == nil || output.Pointer() == 0 || output.Count() != wantCount || output.SizeBytes() != uint64(wantCount*4) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill residual-add output buffer shape mismatch", nil)
	}
	rowBytes := uint64(cfg.Count * 4)
	for token := 0; token < tokenCount; token++ {
		offset := nativeDevicePointer(token * cfg.Count * 4)
		rowInput := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add input row", input.Pointer()+offset, rowBytes, cfg.Count)
		rowResidual := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add residual row", residual.Pointer()+offset, rowBytes, cfg.Count)
		rowOutput := hipBorrowDeviceByteBufferValue(driver, "prefill residual-add output row", output.Pointer()+offset, rowBytes, cfg.Count)
		if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutput(ctx, driver, &rowInput, &rowResidual, cfg, &rowOutput, outputScale); err != nil {
			return err
		}
	}
	return nil
}

func hipRunGemma4Q4PrefillMLPBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillMLPBatchWorkspace(ctx, driver, cfg, input, tokenCount, nil)
}

func hipRunGemma4Q4PrefillMLPBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillMLPBatchWorkspaceView(ctx, driver, cfg, input, tokenCount, workspace, nil, false)
}

func hipRunGemma4Q4PrefillMLPBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, view *hipDeviceByteBuffer, forceBatchedProjection bool) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP hidden size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP input buffer shape mismatch", nil)
	}
	var err error
	var activated *hipDeviceByteBuffer
	closeActivated := false
	if workspace != nil && tokenCount == 1 && hipMLXQ4GELUTanhMLPPersistentRouteEnabled && hipMLXQ4GELUTanhMLPPersistentCompatible(input, cfg.GateProjection, cfg.UpProjection, cfg.DownProjection) {
		activated, err = workspace.EnsureActivationOutput(driver, cfg.GateProjection.Rows)
		if err != nil {
			return nil, err
		}
		output, err := workspace.EnsureProjectionOutput(driver, cfg.DownProjection.Rows)
		if err != nil {
			return nil, err
		}
		barrier, err := workspace.EnsureGELUTanhMLPBarrier(driver)
		if err != nil {
			return nil, err
		}
		if err := hipRunMLXQ4GELUTanhMLPPersistentKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg.GateProjection, cfg.UpProjection, cfg.DownProjection, activated, output, barrier, workspace); err != nil {
			return nil, err
		}
		if view != nil {
			*view = hipBorrowDeviceByteBufferValue(driver, "prefill MLP projection workspace view", output.Pointer(), output.SizeBytes(), output.Count())
			return view, nil
		}
		return hipBorrowDeviceByteBuffer(driver, "prefill MLP projection workspace view", output.Pointer(), output.SizeBytes(), output.Count()), nil
	}
	if workspace != nil {
		activated, err = workspace.EnsureActivationOutput(driver, tokenCount*cfg.GateProjection.Rows)
		if err == nil {
			if tokenCount == 1 {
				err = hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg.GateProjection, cfg.UpProjection, activated, workspace)
			} else {
				err = hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInputOutput(ctx, driver, input, cfg.GateProjection, cfg.UpProjection, tokenCount, activated)
			}
		}
	} else if tokenCount == 1 {
		activated, err = hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(ctx, driver, input, cfg.GateProjection, cfg.UpProjection)
		closeActivated = true
	} else {
		activated, err = hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(ctx, driver, input, cfg.GateProjection, cfg.UpProjection, tokenCount)
		closeActivated = true
	}
	if err != nil {
		return nil, err
	}
	if closeActivated {
		defer activated.Close()
	}
	return hipRunGemma4Q4PrefillProjectionBatchWorkspaceView(ctx, driver, activated, cfg.DownProjection, tokenCount, workspace, "prefill MLP projection workspace view", view, forceBatchedProjection)
}

func hipRunGemma4Q4PrefillProjectionBatchWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, label string) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillProjectionBatchWorkspaceView(ctx, driver, input, cfg, tokenCount, workspace, label, nil, false)
}

func hipRunGemma4Q4PrefillProjectionBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, label string, view *hipDeviceByteBuffer, forceBatchedProjection bool) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		if tokenCount == 1 && !forceBatchedProjection {
			return hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg)
		}
		return hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg, tokenCount)
	}
	outputCount := tokenCount * cfg.Rows
	output, err := workspace.EnsureProjectionOutput(driver, outputCount)
	if err != nil {
		return nil, err
	}
	if tokenCount == 1 && !forceBatchedProjection {
		err = hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(ctx, driver, input, cfg, output, workspace)
	} else {
		err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInputOutput(ctx, driver, input, cfg, tokenCount, output)
	}
	if err != nil {
		return nil, err
	}
	if view != nil {
		*view = hipBorrowDeviceByteBufferValue(driver, label, output.Pointer(), output.SizeBytes(), output.Count())
		return view, nil
	}
	return hipBorrowDeviceByteBuffer(driver, label, output.Pointer(), output.SizeBytes(), output.Count()), nil
}

func hipRunGemma4Q4PrefillLayerBodyBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int, epsilon float32) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(ctx, driver, cfg, input, layer, nil, tokenCount, queryStartToken, epsilon)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx, driver, cfg, input, layer, perLayerInput, tokenCount, queryStartToken, epsilon, workspace, nil, nil, nil, false)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputWorkspaceInto(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, out *hipGemma4Q4PrefillLayerBodyBatch) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx, driver, cfg, input, layer, perLayerInput, tokenCount, queryStartToken, epsilon, workspace, out, nil, nil, false)
}

func hipValidateGemma4Q4PrefillForwardBatch(cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, priorLayerKV []*rocmDeviceKVCache, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, outputRow int) error {
	if len(tokens) == 0 {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward token span is required", nil)
	}
	if startPosition < 0 {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward start position must be non-negative", nil)
	}
	if err := cfg.validate(); err != nil {
		return err
	}
	if startPosition == 0 && len(priorLayerKV) != 0 {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward prior layer KV requires nonzero start position", nil)
	}
	if startPosition > 0 && len(priorLayerKV) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward prior layer KV count mismatch", nil)
	}
	if len(priorLayerKV) != 0 && len(priorLayerKV) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward prior layer KV count mismatch", nil)
	}
	for index, prior := range priorLayerKV {
		if prior == nil {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d prior device KV is required", index), nil)
		}
		if prior.closed {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d prior device KV is closed", index), nil)
		}
		priorTokens := prior.TokenCount()
		if priorTokens != startPosition && (cfg.Layers[index].SlidingWindow <= 0 || priorTokens <= 0 || priorTokens > startPosition) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d prior device KV token count must match start position or a retained sliding window", index), nil)
		}
	}
	if len(outputRows) != 0 && len(outputRows) != len(tokens) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward output mask length mismatch", nil)
	}
	if outputRow >= len(tokens) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward output row is outside token span", nil)
	}
	if len(perLayerInputs) != 0 && len(perLayerInputs) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward per-layer input count mismatch", nil)
	}
	generatePerLayerInputs := len(perLayerInputs) == 0 && len(cfg.Layers) > 0 && cfg.Layers[0].PerLayerInput.hasGlobalPrecompute()
	for index, layer := range cfg.Layers {
		if !layer.PerLayerInput.hasLayerApply() {
			continue
		}
		if generatePerLayerInputs {
			continue
		}
		if len(perLayerInputs) == 0 || perLayerInputs[index] == nil {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d per-layer input batch is required", index), nil)
		}
		input := perLayerInputs[index]
		wantCount := len(tokens) * layer.PerLayerInput.InputSize
		if input.Pointer() == 0 || input.Count() != wantCount || input.SizeBytes() != uint64(wantCount*4) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d per-layer input batch shape mismatch", index), nil)
		}
	}
	return nil
}

func hipGemma4Q4CanUseBatchedGeneratePrefill(cfg hipGemma4Q4ForwardConfig) bool {
	if len(cfg.Layers) == 0 {
		return false
	}
	for _, layer := range cfg.Layers {
		if layer.MoE != nil {
			return false
		}
		if layer.PerLayerInput.hasLayerApply() && !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
			return false
		}
	}
	return true
}

func hipGemma4Q4PrefillSharedSuffixStart(sharedSources []int) int {
	start := len(sharedSources)
	for start > 0 && sharedSources[start-1] != start-1 {
		start--
	}
	if start == 0 || start == len(sharedSources) {
		return -1
	}
	for index := 0; index < start; index++ {
		if sharedSources[index] != index {
			return -1
		}
	}
	for index := start; index < len(sharedSources); index++ {
		source := sharedSources[index]
		if source < 0 || source >= start {
			return -1
		}
	}
	return start
}

func hipGemma4Q4PrefillSharedSuffixWindow(cfg hipGemma4Q4ForwardConfig, suffixStart int) int {
	if suffixStart <= 0 || suffixStart > len(cfg.Layers) {
		return 0
	}
	window := 0
	for _, layer := range cfg.Layers[:suffixStart] {
		if layer.SlidingWindow > 0 && (window == 0 || layer.SlidingWindow < window) {
			window = layer.SlidingWindow
		}
	}
	return window
}

func hipRunGemma4Q4PrefillForwardBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, driver, cfg, tokens, startPosition, epsilon, mode, nil, perLayerInputs, outputRows, best)
}

func hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorWorkspace(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, perLayerInputs, outputRows, best, nil)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspace(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, nil, perLayerInputs, outputRows, best, workspace)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRow(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, priorLayerDescriptorTables, perLayerInputs, outputRows, -1, best, workspace)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRow(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, outputRow int, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, priorLayerDescriptorTables, perLayerInputs, outputRows, outputRow, best, workspace, defaultHIPGemma4Q4EngineConfig())
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, outputRow int, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowDeviceTokenWithEngineConfig(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, priorLayerDescriptorTables, perLayerInputs, outputRows, outputRow, best, workspace, engineConfig, nil, nil, nil, nil)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowInitialHiddenWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, outputRow int, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig, initialHidden *hipDeviceByteBuffer, visibleTokenCaps *hipDeviceTokenBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowDeviceTokenWithEngineConfig(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, priorLayerDescriptorTables, perLayerInputs, outputRows, outputRow, best, workspace, engineConfig, nil, nil, initialHidden, visibleTokenCaps)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowGreedyTokenWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, outputRow int, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig, greedyToken *hipDeviceByteBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	var routeMetrics *hipDecodeRouteMetrics
	if len(tokens) == 1 {
		routeMetrics = hipBeginDecodeRouteMetrics()
		defer hipFinishDecodeRouteMetrics(routeMetrics)
	}
	return hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowDeviceTokenWithEngineConfig(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, priorLayerDescriptorTables, perLayerInputs, outputRows, outputRow, best, workspace, engineConfig, nil, greedyToken, nil, nil)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowDeviceTokenWithEngineConfig(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, priorLayerDescriptorTables []*rocmDeviceKVDescriptorTable, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, outputRow int, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace, engineConfig hipGemma4Q4EngineConfig, tokenBuffer *hipDeviceTokenBuffer, greedyToken, initialHidden *hipDeviceByteBuffer, visibleTokenCaps *hipDeviceTokenBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if err := hipValidateGemma4Q4PrefillForwardBatch(cfg, tokens, startPosition, priorLayerKV, perLayerInputs, outputRows, outputRow); err != nil {
		return nil, err
	}
	out := hipBorrowGemma4Q4PrefillForwardBatch(len(cfg.Layers))
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	if tokenBuffer != nil && greedyToken != nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward requires one token buffer format", nil)
	}
	if initialHidden != nil {
		wantCount := len(tokens) * cfg.Layers[0].HiddenSize
		if greedyToken != nil || initialHidden.Pointer() == 0 || initialHidden.Count() != wantCount || initialHidden.SizeBytes() != uint64(wantCount*4) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward initial hidden shape mismatch", nil)
		}
	}
	if visibleTokenCaps != nil && (visibleTokenCaps.Pointer() == 0 || visibleTokenCaps.Count() != len(tokens) || visibleTokenCaps.SizeBytes() != uint64(len(tokens)*4)) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill visible-token cap shape mismatch", nil)
	}
	if greedyToken != nil && (len(tokens) != 1 || greedyToken.Pointer() == 0 || greedyToken.Count() != 1 || greedyToken.SizeBytes() != hipMLXQ4ProjectionBestBytes) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward greedy token buffer shape mismatch", nil)
	}
	if tokenBuffer != nil {
		if tokenBuffer.Pointer() == 0 || tokenBuffer.Count() != len(tokens) || tokenBuffer.SizeBytes() != uint64(len(tokens)*4) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward token buffer shape mismatch", nil)
		}
	} else if greedyToken == nil {
		if workspace != nil {
			tokenBuffer, err = workspace.EnsurePrefillTokenBuffer(driver, tokens)
		} else {
			tokenBuffer, err = hipUploadTokenIDs(driver, tokens)
		}
		if err != nil {
			return nil, err
		}
		defer tokenBuffer.Close()
	}
	var hidden *hipDeviceByteBuffer
	if initialHidden != nil {
		out.embeddingView = hipBorrowDeviceByteBufferValue(driver, "prefill initial hidden view", initialHidden.Pointer(), initialHidden.SizeBytes(), initialHidden.Count())
		hidden = &out.embeddingView
	} else if greedyToken != nil {
		hidden, err = hipRunGemma4Q4PrefillEmbeddingBatchGreedyTokenWorkspaceView(ctx, driver, cfg.Layers[0], tokens, greedyToken, workspace, &out.embeddingView)
	} else {
		hidden, err = hipRunGemma4Q4PrefillEmbeddingBatchTokenBufferWorkspaceView(ctx, driver, cfg.Layers[0], tokens, tokenBuffer, workspace, &out.embeddingView)
	}
	if err != nil {
		return nil, err
	}
	out.Embedding = hidden
	var generatedPerLayerInputs *hipGemma4Q4PerLayerInputDeviceSet
	if len(perLayerInputs) == 0 && cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		if greedyToken != nil {
			generatedPerLayerInputs, err = hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceGreedyToken(ctx, driver, cfg, tokens, hidden, epsilon, workspace, greedyToken)
		} else {
			generatedPerLayerInputs, err = hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceTokenBuffer(ctx, driver, cfg, tokens, hidden, epsilon, workspace, tokenBuffer)
		}
		if err != nil {
			return nil, err
		}
		defer generatedPerLayerInputs.Close()
	}
	tokenCount := len(tokens)
	sharedSources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	executionLayerCount := len(cfg.Layers)
	if engineConfig.prefillLayerLimit > 0 {
		if engineConfig.prefillLayerLimit != hipGemma4Q4PrefillSharedSuffixStart(sharedSources) ||
			len(outputRows) > 0 || outputRow >= 0 || greedyToken != nil || initialHidden != nil || visibleTokenCaps != nil {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer limit requires a non-final causal shared-suffix chunk", nil)
		}
		executionLayerCount = engineConfig.prefillLayerLimit
	}
	var precomputedInputNorm *hipDeviceByteBuffer
	for index, layerCfg := range cfg.Layers[:executionLayerCount] {
		if metrics := hipActiveDecodeRouteMetrics(); metrics != nil {
			metrics.setLayer(index, layerCfg.LayerType)
		}
		out.Layers = append(out.Layers, hipGemma4Q4PrefillForwardLayerBatch{})
		layerBatch := &out.Layers[index]
		layerBatch.KV = &layerBatch.kvStorage
		layerInput := hidden
		prior := (*rocmDeviceKVCache)(nil)
		if len(priorLayerKV) > index {
			prior = priorLayerKV[index]
		}
		priorDescriptorTable := (*rocmDeviceKVDescriptorTable)(nil)
		if len(priorLayerDescriptorTables) > index {
			priorDescriptorTable = priorLayerDescriptorTables[index]
		}
		var layerKV *hipGemma4Q4PrefillLayerKVBatch
		if len(sharedSources) > index && sharedSources[index] != index {
			source := sharedSources[index]
			if source < 0 || source >= index || out.Layers[source].KV == nil || out.Layers[source].KV.DeviceKV == nil {
				return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared KV source layer is unavailable", nil)
			}
			layerKV, err = hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspacePrecomputedInputNormInto(ctx, driver, layerCfg, layerInput, precomputedInputNorm, out.Layers[source].KV, tokenCount, startPosition, epsilon, workspace, layerBatch.KV)
			if err != nil {
				return nil, err
			}
		} else {
			borrowRawKV := workspace != nil
			borrowRetainedKV := workspace != nil
			layerKV, err = hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientPrecomputedInputNormIntoWithEngineConfig(ctx, driver, layerCfg, layerInput, precomputedInputNorm, prior, priorDescriptorTable, tokenCount, startPosition, epsilon, mode, workspace, borrowRawKV, borrowRetainedKV, layerBatch.KV, engineConfig)
			if err != nil {
				return nil, err
			}
		}
		layerKV.VisibleTokenCaps = visibleTokenCaps
		precomputedInputNorm = nil
		layerBatch.KV = layerKV
		perLayerInput := (*hipDeviceByteBuffer)(nil)
		if generatedPerLayerInputs != nil {
			perLayerInput = generatedPerLayerInputs.Layer(index)
		} else if len(perLayerInputs) > index {
			perLayerInput = perLayerInputs[index]
		}
		queryStartToken := layerKV.DeviceKV.Cache.TokenCount() - tokenCount
		if queryStartToken < 0 {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer device KV token count is smaller than query batch", nil)
		}
		layerBatch.Body = &layerBatch.bodyStorage
		nextInputNormCfg := (*hipRMSNormDeviceWeightConfig)(nil)
		nextInputNormOutput := (*hipDeviceByteBuffer)(nil)
		if workspace != nil && tokenCount == 1 && index+1 < len(cfg.Layers) {
			nextCfg := cfg.Layers[index+1].InputNorm
			if nextCfg.Count == 0 {
				nextCfg.Count = cfg.Layers[index+1].HiddenSize
			}
			nextCfg.Epsilon = epsilon
			nextInputNormOutput, err = workspace.EnsureNextInputOutput(driver, tokenCount*nextCfg.Count, layerCfg.Layer)
			if err != nil {
				return nil, err
			}
			nextInputNormCfg = &nextCfg
		}
		body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx, driver, layerCfg, layerInput, layerKV, perLayerInput, tokenCount, queryStartToken, epsilon, workspace, layerBatch.Body, nextInputNormCfg, nextInputNormOutput, engineConfig.ForceBatchedProjection)
		if err != nil {
			return nil, err
		}
		hidden = body.FinalHidden
		precomputedInputNorm = body.NextInputNorm
		out.FinalHidden = hidden
	}
	for index := executionLayerCount; index < len(cfg.Layers); index++ {
		source := sharedSources[index]
		if source < 0 || source >= executionLayerCount || out.Layers[source].KV == nil {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill skipped shared KV source layer is unavailable", nil)
		}
		out.Layers = append(out.Layers, hipGemma4Q4PrefillForwardLayerBatch{})
		layerBatch := &out.Layers[index]
		layerBatch.KV = &layerBatch.kvStorage
		if err := hipBorrowGemma4Q4PrefillSharedDeviceKVLayerInto(out.Layers[source].KV, layerBatch.KV); err != nil {
			return nil, err
		}
	}
	if len(outputRows) > 0 {
		last := cfg.Layers[len(cfg.Layers)-1]
		for row, selected := range outputRows {
			if !selected {
				continue
			}
			greedy, err := hipRunGemma4Q4PrefillFinalGreedyTokenForRowWorkspace(ctx, driver, last, out.FinalHidden, tokenCount, row, epsilon, best, workspace)
			if err != nil {
				return nil, err
			}
			out.Greedy = append(out.Greedy, hipGemma4Q4PrefillGreedyBatchOutput{
				Row:    row,
				Greedy: greedy,
			})
		}
	} else if outputRow >= 0 {
		last := cfg.Layers[len(cfg.Layers)-1]
		greedy, err := hipRunGemma4Q4PrefillFinalGreedyTokenForRowWorkspace(ctx, driver, last, out.FinalHidden, tokenCount, outputRow, epsilon, best, workspace)
		if err != nil {
			return nil, err
		}
		out.Greedy = append(out.Greedy, hipGemma4Q4PrefillGreedyBatchOutput{
			Row:    outputRow,
			Greedy: greedy,
		})
	}
	success = true
	return out, nil
}

func hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward *hipGemma4Q4PrefillForwardBatch, mode string) (*hipGemma4Q4DeviceDecodeState, error) {
	if forward == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward output is required", nil)
	}
	var sharedSourceScratch [128]int
	sharedSources := hipGemma4Q4PrefillForwardSharedSourceLayers(forward, sharedSourceScratch[:0])
	state := hipNewGemma4Q4DeviceDecodeState(firstNonEmptyString(mode, rocmKVCacheModeFP16), len(forward.Layers))
	state.appendLayers = len(forward.Layers)
	success := false
	defer func() {
		if !success {
			_ = state.Close()
		}
	}()
	for index := range forward.Layers {
		deviceKV := (*hipGemma4Q4PrefillDeviceKVBatch)(nil)
		if forward.Layers[index].KV != nil {
			deviceKV = forward.Layers[index].KV.DeviceKV
		}
		if deviceKV == nil || deviceKV.Cache == nil || deviceKV.DescriptorTable == nil {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward layer device KV is required", nil)
		}
		if deviceKV.Cache.borrowed {
			shared, err := hipGemma4Q4PrefillSharedDecodeLayerState(state, sharedSources[index])
			if err != nil {
				return nil, err
			}
			state.layers = append(state.layers, shared)
			continue
		}
		if err := hipGemma4Q4PrefillFinalizeRetainWindow(deviceKV); err != nil {
			return nil, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("finalize prefill layer %d retained KV", index), err)
		}
		state.layers = append(state.layers, hipGemma4Q4DeviceLayerKVState{
			cache:           deviceKV.Cache,
			descriptorTable: deviceKV.DescriptorTable,
			launch:          deviceKV.Launch,
		})
		deviceKV.Cache = nil
		deviceKV.DescriptorTable = nil
		deviceKV.Launch = rocmDeviceKVLaunchDescriptor{}
	}
	success = true
	return state, nil
}

func hipGemma4Q4PrefillLayerHasSharedDependents(sharedSources []int, layerIndex int) bool {
	for index, source := range sharedSources {
		if index != layerIndex && source == layerIndex {
			return true
		}
	}
	return false
}

func hipGemma4Q4PrefillForwardSharedSourceLayers(forward *hipGemma4Q4PrefillForwardBatch, sources []int) []int {
	if forward == nil {
		return sources[:0]
	}
	if cap(sources) < len(forward.Layers) {
		sources = make([]int, len(forward.Layers))
	} else {
		sources = sources[:len(forward.Layers)]
	}
	for index := range sources {
		sources[index] = index
	}
	for index := range forward.Layers {
		deviceKV := (*hipGemma4Q4PrefillDeviceKVBatch)(nil)
		if forward.Layers[index].KV != nil {
			deviceKV = forward.Layers[index].KV.DeviceKV
		}
		if deviceKV == nil || deviceKV.Cache == nil || !deviceKV.Cache.borrowed {
			continue
		}
		sources[index] = -1
		for sourceIndex := index - 1; sourceIndex >= 0; sourceIndex-- {
			sourceKV := (*hipGemma4Q4PrefillDeviceKVBatch)(nil)
			if forward.Layers[sourceIndex].KV != nil {
				sourceKV = forward.Layers[sourceIndex].KV.DeviceKV
			}
			if sourceKV == nil || sourceKV.Cache == nil || !deviceKV.Cache.sharesPagesFrom(sourceKV.Cache) {
				continue
			}
			sources[index] = sourceIndex
			break
		}
	}
	return sources
}

func hipGemma4Q4PrefillFinalizeRetainWindow(deviceKV *hipGemma4Q4PrefillDeviceKVBatch) error {
	if deviceKV == nil || deviceKV.Cache == nil || deviceKV.RetainWindow <= 0 || deviceKV.Cache.TokenCount() <= deviceKV.RetainWindow {
		return nil
	}
	beforeTokens := deviceKV.Cache.TokenCount()
	deviceKV.Cache = deviceKV.Cache.trimDeviceTokenWindowForAppend(deviceKV.RetainWindow)
	if deviceKV.Cache.TokenCount() == beforeTokens {
		return nil
	}
	if err := deviceKV.DescriptorTable.Close(); err != nil {
		return err
	}
	table, err := deviceKV.Cache.KernelDescriptorTable()
	if err != nil {
		return err
	}
	launch, err := deviceKV.Cache.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		return err
	}
	deviceKV.DescriptorTable = table
	deviceKV.Launch = launch
	return nil
}

func hipGemma4Q4PrefillSharedDecodeLayerState(state *hipGemma4Q4DeviceDecodeState, sourceIndex int) (hipGemma4Q4DeviceLayerKVState, error) {
	if state == nil || sourceIndex < 0 || sourceIndex >= len(state.layers) {
		return hipGemma4Q4DeviceLayerKVState{}, core.E(hipGemma4Q4Layer0Operation, "prefill shared KV source state is required", nil)
	}
	source := &state.layers[sourceIndex]
	if source.cache == nil || source.descriptorTable == nil {
		return hipGemma4Q4DeviceLayerKVState{}, core.E(hipGemma4Q4Layer0Operation, "prefill shared KV source layer is unavailable", nil)
	}
	return hipGemma4Q4DeviceLayerKVState{
		cache:                   source.cache,
		descriptorTable:         source.descriptorTable,
		launch:                  source.launch,
		borrowedCache:           true,
		borrowedDescriptorTable: true,
	}, nil
}

func hipRunGemma4Q4PrefillPerLayerInputProjectionBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input, perLayerInput *hipDeviceByteBuffer, tokenCount int) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillPerLayerInputProjectionBatchWorkspace(ctx, driver, cfg, input, perLayerInput, tokenCount, nil)
}

func hipRunGemma4Q4PrefillPerLayerInputProjectionBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input, perLayerInput *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillPerLayerInputProjectionBatchWorkspaceView(ctx, driver, cfg, input, perLayerInput, tokenCount, workspace, nil, false)
}

func hipRunGemma4Q4PrefillPerLayerInputProjectionBatchWorkspaceView(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input, perLayerInput *hipDeviceByteBuffer, tokenCount int, workspace *hipAttentionHeadsChunkedWorkspace, view *hipDeviceByteBuffer, forceBatchedProjection bool) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validatePerLayerInput(); err != nil {
		return nil, err
	}
	if !cfg.PerLayerInput.hasLayerApply() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input tensors are not configured", nil)
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.PerLayerInput.InputSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input geometry mismatch", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden buffer shape mismatch", nil)
	}
	if perLayerInput == nil || perLayerInput.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input multiplier buffer is required", nil)
	}
	if perLayerInput.Count() != tokenCount*cfg.PerLayerInput.InputSize ||
		perLayerInput.SizeBytes() != uint64(perLayerInput.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input multiplier buffer shape mismatch", nil)
	}
	var activated *hipDeviceByteBuffer
	activatedBorrowed := false
	var err error
	if workspace != nil {
		activated, err = workspace.EnsureActivationOutput(driver, tokenCount*cfg.PerLayerInput.InputGate.Rows)
		if err == nil {
			if tokenCount == 1 {
				err = hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(ctx, driver, input, perLayerInput, cfg.PerLayerInput.InputGate, activated, workspace)
			} else {
				err = hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplierOutput(ctx, driver, input, perLayerInput, cfg.PerLayerInput.InputGate, tokenCount, activated)
			}
		}
		activatedBorrowed = err == nil
	} else if tokenCount == 1 {
		activated, err = hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplier(ctx, driver, input, perLayerInput, cfg.PerLayerInput.InputGate)
	} else {
		activated, err = hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(ctx, driver, input, perLayerInput, cfg.PerLayerInput.InputGate, tokenCount)
	}
	if err != nil {
		return nil, err
	}
	if !activatedBorrowed {
		defer activated.Close()
	}
	return hipRunGemma4Q4PrefillProjectionBatchWorkspaceView(ctx, driver, activated, cfg.PerLayerInput.Projection, tokenCount, workspace, "prefill per-layer projection workspace view", view, forceBatchedProjection)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx, driver, cfg, input, layer, perLayerInput, tokenCount, queryStartToken, epsilon, nil, nil, nil, nil, false)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace, out *hipGemma4Q4PrefillLayerBodyBatch, nextInputNormCfg *hipRMSNormDeviceWeightConfig, nextInputNormOutput *hipDeviceByteBuffer, forceBatchedProjection bool) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body token count must be positive", nil)
	}
	if queryStartToken < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body query start token must be non-negative", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.QueryHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body geometry mismatch", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body residual input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body residual input buffer shape mismatch", nil)
	}
	if perLayerInput != nil {
		if err := cfg.validatePerLayerInput(); err != nil {
			return nil, err
		}
		if !cfg.PerLayerInput.hasLayerApply() {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input tensors are not configured", nil)
		}
		if perLayerInput.Pointer() == 0 ||
			perLayerInput.Count() != tokenCount*cfg.PerLayerInput.InputSize ||
			perLayerInput.SizeBytes() != uint64(perLayerInput.Count()*4) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input multiplier buffer shape mismatch", nil)
		}
	}
	if nextInputNormCfg != nil {
		if tokenCount != 1 {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill next input-norm carry requires one token", nil)
		}
		if workspace == nil {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill next input-norm carry requires workspace", nil)
		}
		nextCfg := *nextInputNormCfg
		if nextCfg.Count != cfg.HiddenSize {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill next input-norm count must match hidden size", nil)
		}
		if err := hipValidateRMSNormDeviceWeightConfig("Gemma4Q4PrefillNextInputNorm", nextCfg); err != nil {
			return nil, err
		}
		if nextInputNormOutput == nil || nextInputNormOutput.Pointer() == 0 ||
			nextInputNormOutput.Count() != nextCfg.Count ||
			nextInputNormOutput.SizeBytes() != uint64(nextCfg.Count*4) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill next input-norm output buffer shape mismatch", nil)
		}
	}
	if layer == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body KV setup is required", nil)
	}
	if out == nil {
		out = hipBorrowGemma4Q4PrefillLayerBodyBatch()
	} else {
		*out = hipGemma4Q4PrefillLayerBodyBatch{}
	}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	out.AttentionOutput, err = hipRunGemma4Q4PrefillAttentionBatchWorkspaceView(ctx, driver, cfg, layer, tokenCount, queryStartToken, workspace, &out.attentionOutputView)
	if err != nil {
		return nil, err
	}
	out.AttentionProjection, err = hipRunGemma4Q4PrefillProjectionBatchWorkspaceView(ctx, driver, out.AttentionOutput, cfg.OutputProjection, tokenCount, workspace, "prefill attention projection workspace view", &out.attentionProjectionView, forceBatchedProjection)
	if err != nil {
		return nil, err
	}
	postAttentionNormCfg := cfg.PostAttentionNorm
	postAttentionNormCfg.Epsilon = epsilon
	postAttentionNormCfg.Count = cfg.HiddenSize
	preFeedForwardNormCfg := cfg.PreFeedForwardNorm
	preFeedForwardNormCfg.Epsilon = epsilon
	preFeedForwardNormCfg.Count = cfg.HiddenSize
	out.AttentionResidual, out.PreFeedForward, err = hipRunGemma4Q4PrefillResidualAddNormBatchWorkspaceView(ctx, driver, out.AttentionProjection, input, postAttentionNormCfg, preFeedForwardNormCfg, tokenCount, 1, workspace, &out.attentionResidualView, &out.preFeedForwardView)
	if err != nil {
		return nil, err
	}
	if cfg.MoE != nil {
		out.MLPOutput, err = hipRunGemma4MoEDeviceMLPBatchWithWorkspace(ctx, driver, out.AttentionResidual, out.PreFeedForward, cfg, epsilon, tokenCount, workspace)
	} else {
		out.MLPOutput, err = hipRunGemma4Q4PrefillMLPBatchWorkspaceView(ctx, driver, cfg, out.PreFeedForward, tokenCount, workspace, &out.mlpOutputView, forceBatchedProjection)
	}
	if err != nil {
		return nil, err
	}
	postFeedForwardNormCfg := cfg.PostFeedForwardNorm
	postFeedForwardNormCfg.Epsilon = epsilon
	postFeedForwardNormCfg.Count = cfg.HiddenSize
	postFeedForwardScale := float32(1)
	if perLayerInput == nil {
		postFeedForwardScale = cfg.effectiveLayerScalar()
	}
	var postFeedForwardOutput *hipDeviceByteBuffer
	postFeedForwardLabel := ""
	if workspace != nil {
		if perLayerInput == nil {
			postFeedForwardOutput, err = workspace.EnsureFinalHiddenOutput(driver, tokenCount*cfg.HiddenSize, cfg.Layer)
			postFeedForwardLabel = "prefill final hidden workspace view"
		} else {
			postFeedForwardOutput, err = workspace.EnsureIntermediateOutput(driver, tokenCount*cfg.HiddenSize)
			postFeedForwardLabel = "prefill post-feedforward workspace view"
		}
		if err != nil {
			return nil, err
		}
	}
	if perLayerInput == nil {
		if nextInputNormCfg != nil {
			err = hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, out.MLPOutput, out.AttentionResidual, postFeedForwardNormCfg, *nextInputNormCfg, postFeedForwardOutput, nextInputNormOutput, postFeedForwardScale, workspace)
			if err != nil {
				return nil, err
			}
			out.postFeedForwardView = hipBorrowDeviceByteBufferValue(driver, postFeedForwardLabel, postFeedForwardOutput.Pointer(), postFeedForwardOutput.SizeBytes(), postFeedForwardOutput.Count())
			out.PostFeedForward = &out.postFeedForwardView
			out.FinalHidden = out.PostFeedForward
			out.nextInputNormView = hipBorrowDeviceByteBufferValue(driver, "prefill next input norm workspace view", nextInputNormOutput.Pointer(), nextInputNormOutput.SizeBytes(), nextInputNormOutput.Count())
			out.NextInputNorm = &out.nextInputNormView
		} else {
			out.PostFeedForward, err = hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutputView(ctx, driver, out.MLPOutput, out.AttentionResidual, postFeedForwardNormCfg, tokenCount, postFeedForwardScale, workspace, postFeedForwardOutput, postFeedForwardLabel, &out.postFeedForwardView)
			if err != nil {
				return nil, err
			}
			out.FinalHidden = out.PostFeedForward
		}
	} else {
		out.PostFeedForward, err = hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutputView(ctx, driver, out.MLPOutput, out.AttentionResidual, postFeedForwardNormCfg, tokenCount, postFeedForwardScale, workspace, postFeedForwardOutput, postFeedForwardLabel, &out.postFeedForwardView)
		if err != nil {
			return nil, err
		}
		out.PerLayerProjection, err = hipRunGemma4Q4PrefillPerLayerInputProjectionBatchWorkspaceView(ctx, driver, cfg, out.PostFeedForward, perLayerInput, tokenCount, workspace, &out.perLayerProjectionView, forceBatchedProjection)
		if err != nil {
			return nil, err
		}
		perLayerNormCfg := cfg.PerLayerInput.PostInputNorm
		perLayerNormCfg.Epsilon = epsilon
		perLayerNormCfg.Count = cfg.HiddenSize
		finalHiddenOutput := (*hipDeviceByteBuffer)(nil)
		finalHiddenLabel := ""
		if workspace != nil {
			finalHiddenOutput, err = workspace.EnsureFinalHiddenOutput(driver, tokenCount*cfg.HiddenSize, cfg.Layer)
			if err != nil {
				return nil, err
			}
			finalHiddenLabel = "prefill final hidden workspace view"
		}
		if nextInputNormCfg != nil {
			err = hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(ctx, driver, out.PerLayerProjection, out.PostFeedForward, perLayerNormCfg, *nextInputNormCfg, finalHiddenOutput, nextInputNormOutput, cfg.effectiveLayerScalar(), workspace)
			if err != nil {
				return nil, err
			}
			out.finalHiddenView = hipBorrowDeviceByteBufferValue(driver, finalHiddenLabel, finalHiddenOutput.Pointer(), finalHiddenOutput.SizeBytes(), finalHiddenOutput.Count())
			out.FinalHidden = &out.finalHiddenView
			out.nextInputNormView = hipBorrowDeviceByteBufferValue(driver, "prefill next input norm workspace view", nextInputNormOutput.Pointer(), nextInputNormOutput.SizeBytes(), nextInputNormOutput.Count())
			out.NextInputNorm = &out.nextInputNormView
		} else {
			out.FinalHidden, err = hipRunGemma4Q4PrefillResidualAddBatchWorkspaceOutputView(ctx, driver, out.PerLayerProjection, out.PostFeedForward, perLayerNormCfg, tokenCount, cfg.effectiveLayerScalar(), workspace, finalHiddenOutput, finalHiddenLabel, &out.finalHiddenView)
			if err != nil {
				return nil, err
			}
		}
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillFinalGreedyForRow(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer) (hipGreedySampleResult, error) {
	return hipRunGemma4Q4PrefillFinalGreedyForRowSuppress(ctx, driver, cfg, hidden, tokenCount, row, epsilon, best, nil)
}

func hipRunGemma4Q4PrefillFinalGreedyForRowSuppress(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, suppressTokens []int32) (hipGreedySampleResult, error) {
	return hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx, driver, cfg, hidden, tokenCount, row, epsilon, best, suppressTokens, nil)
}

func hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, error) {
	return hipRunGemma4Q4PrefillFinalGreedyForRowWorkspace(ctx, driver, cfg, hidden, tokenCount, row, epsilon, best, suppressTokens, workspace, false)
}

func hipRunGemma4Q4PrefillFinalGreedyBatchSuppressWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, epsilon float32, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) ([]hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy batch token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy batch hidden size must be positive", nil)
	}
	if hidden == nil || hidden.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy batch hidden is required", nil)
	}
	if hidden.Count() != tokenCount*cfg.HiddenSize || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy batch hidden shape mismatch", nil)
	}
	finalNormCfg := cfg.FinalNorm
	finalNormCfg.Epsilon = epsilon
	finalNormCfg.Count = cfg.HiddenSize
	if err := hipValidateGemma4Q4NormConfig("prefill_final_norm", finalNormCfg, cfg.HiddenSize); err != nil {
		return nil, err
	}
	if cfg.LMHeadProjection.Rows != cfg.VocabSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy batch LM head shape mismatch", nil)
	}
	if err := cfg.LMHeadProjection.validateInputCount(cfg.HiddenSize); err != nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy batch LM head config", err)
	}
	finalNormCount := tokenCount * cfg.HiddenSize
	var finalNorm *hipDeviceByteBuffer
	ownsFinalNorm := false
	var err error
	if workspace != nil {
		finalNorm, err = workspace.EnsureActivationOutput(driver, finalNormCount)
	} else {
		finalNorm, err = hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy batch norm output", uint64(finalNormCount*4), finalNormCount)
		ownsFinalNorm = err == nil
	}
	if err != nil {
		return nil, err
	}
	if ownsFinalNorm {
		defer finalNorm.Close()
	}
	if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutput(ctx, driver, hidden, finalNormCfg, tokenCount, finalNorm); err != nil {
		return nil, err
	}
	var best *hipDeviceByteBuffer
	if workspace != nil {
		best, err = workspace.BorrowProjectionGreedyBestBatch(driver, tokenCount)
		if err != nil {
			return nil, err
		}
	}
	return hipRunMLXQ4ProjectionSoftcapGreedyBatchKernelWithDeviceInputBufferSuppress(ctx, driver, finalNorm, cfg.LMHeadProjection, cfg.FinalLogitSoftcap, tokenCount, best, suppressTokens, workspace)
}

func hipRunGemma4Q4PrefillFinalGreedyTokenForRowWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, error) {
	return hipRunGemma4Q4PrefillFinalGreedyForRowWorkspace(ctx, driver, cfg, hidden, tokenCount, row, epsilon, best, nil, workspace, true)
}

func hipRunGemma4Q4PrefillFinalGreedyDeviceForRowWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy token count must be positive", nil)
	}
	if row < 0 || row >= tokenCount {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy row is outside token batch", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden size must be positive", nil)
	}
	if hidden == nil || hidden.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden batch is required", nil)
	}
	if hidden.Count() != tokenCount*cfg.HiddenSize || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden batch shape mismatch", nil)
	}
	finalNormCfg := cfg.FinalNorm
	finalNormCfg.Epsilon = epsilon
	finalNormCfg.Count = cfg.HiddenSize
	if err := hipValidateGemma4Q4NormConfig("prefill_final_norm", finalNormCfg, cfg.HiddenSize); err != nil {
		return nil, err
	}
	if cfg.LMHeadProjection.Rows != cfg.VocabSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy LM head shape mismatch", nil)
	}
	if err := cfg.LMHeadProjection.validateInputCount(cfg.HiddenSize); err != nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy LM head config", err)
	}
	rowOffset := nativeDevicePointer(row * cfg.HiddenSize * 4)
	hiddenRow := hipBorrowDeviceByteBufferValue(driver, "Gemma4 q4 prefill selected final hidden row", hidden.Pointer()+rowOffset, uint64(cfg.HiddenSize*4), cfg.HiddenSize)
	var err error
	var finalNorm *hipDeviceByteBuffer
	ownsFinalNorm := false
	if workspace != nil {
		finalNorm, err = workspace.EnsureRMSNormOutput(driver, cfg.HiddenSize)
		if err != nil {
			return nil, err
		}
		if err := hipRunRMSNormDeviceToDeviceKernel(ctx, driver, hiddenRow.Pointer(), hiddenRow.SizeBytes(), finalNorm.Pointer(), finalNorm.SizeBytes(), finalNormCfg); err != nil {
			return nil, err
		}
	} else {
		finalNorm, err = hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, &hiddenRow, finalNormCfg)
		if err != nil {
			return nil, err
		}
		ownsFinalNorm = true
	}
	if ownsFinalNorm {
		defer finalNorm.Close()
	}
	return hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressDevice(ctx, driver, finalNorm, cfg.LMHeadProjection, cfg.FinalLogitSoftcap, best, suppressTokens, workspace)
}

func hipRunGemma4Q4PrefillFinalGreedyForRowWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace, tokenOnly bool) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if tokenCount <= 0 {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy token count must be positive", nil)
	}
	if row < 0 || row >= tokenCount {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy row is outside token batch", nil)
	}
	if cfg.HiddenSize <= 0 {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden size must be positive", nil)
	}
	if hidden == nil || hidden.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden batch is required", nil)
	}
	if hidden.Count() != tokenCount*cfg.HiddenSize || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden batch shape mismatch", nil)
	}
	finalNormCfg := cfg.FinalNorm
	finalNormCfg.Epsilon = epsilon
	finalNormCfg.Count = cfg.HiddenSize
	if err := hipValidateGemma4Q4NormConfig("prefill_final_norm", finalNormCfg, cfg.HiddenSize); err != nil {
		return hipGreedySampleResult{}, err
	}
	if cfg.LMHeadProjection.Rows != cfg.VocabSize {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy LM head shape mismatch", nil)
	}
	if err := cfg.LMHeadProjection.validateInputCount(cfg.HiddenSize); err != nil {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy LM head config", err)
	}
	rowOffset := nativeDevicePointer(row * cfg.HiddenSize * 4)
	hiddenRow := hipBorrowDeviceByteBufferValue(driver, "Gemma4 q4 prefill selected final hidden row", hidden.Pointer()+rowOffset, uint64(cfg.HiddenSize*4), cfg.HiddenSize)
	var err error
	var finalNorm *hipDeviceByteBuffer
	ownsFinalNorm := false
	if workspace != nil {
		finalNorm, err = workspace.EnsureRMSNormOutput(driver, cfg.HiddenSize)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		if err := hipRunRMSNormDeviceToDeviceKernel(ctx, driver, hiddenRow.Pointer(), hiddenRow.SizeBytes(), finalNorm.Pointer(), finalNorm.SizeBytes(), finalNormCfg); err != nil {
			return hipGreedySampleResult{}, err
		}
	} else {
		finalNorm, err = hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, &hiddenRow, finalNormCfg)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		ownsFinalNorm = true
	}
	if ownsFinalNorm {
		defer finalNorm.Close()
	}
	if tokenOnly && len(suppressTokens) == 0 && best != nil {
		tokenID, err := hipRunMLXQ4ProjectionSoftcapGreedyTokenKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, finalNorm, cfg.LMHeadProjection, cfg.FinalLogitSoftcap, best, nil, true)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		return hipGreedySampleResult{TokenID: tokenID}, nil
	}
	var suppress *hipDeviceTokenBuffer
	if len(suppressTokens) > 0 {
		if workspace != nil {
			suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		} else {
			suppress, err = hipUploadTokenIDs(driver, suppressTokens)
		}
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		if workspace == nil {
			defer suppress.Close()
		}
	}
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBufferInitialized(ctx, driver, finalNorm, cfg.LMHeadProjection, cfg.FinalLogitSoftcap, best, suppress, true)
}
