// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"math"
	"strconv"
	"sync"

	core "dappco.re/go"
)

// teacherScratchPool / teacherProbPool / studentScratchPool recycle the
// three vocab-sized float64 scratch buffers consumed by the per-token
// log-softmax + prob accumulators in BatchLoss. Vocab is essentially
// process-invariant (tokenizer-fixed), so pool entries warm to the
// correct capacity after the first call and every subsequent BatchLoss
// invocation lifts pre-sized buffers off the pool instead of paying
// three vocab-sized makes per call. For a 32k vocab that's 3 × 256KB =
// 768KB saved per call.
//
// Three separate pools rather than one wrapper struct — the buffers are
// independent (no shared lifecycle), and a wrapper struct would just add
// a pointer indirection per access on the hot per-token loop without
// saving any pool churn.
var (
	teacherScratchPool sync.Pool
	teacherProbPool    sync.Pool
	studentScratchPool sync.Pool
)

// getFloat64Scratch returns a *[]float64 from the pool sized to hold at
// least vocab elements. The pointer wrapper is stable across grow —
// callers pass the same *[]float64 to the matching pool.Put when done,
// which preserves any grown cap (no second wrapper alloc per call). Pool
// entries pre-sized to the running vocab amortise to zero per-call alloc
// cost across an entire distillation run.
func getFloat64Scratch(pool *sync.Pool, vocab int) *[]float64 {
	if v := pool.Get(); v != nil {
		ptr := v.(*[]float64)
		if cap(*ptr) < vocab {
			*ptr = make([]float64, vocab)
		} else {
			*ptr = (*ptr)[:vocab]
		}
		return ptr
	}
	buf := make([]float64, vocab)
	return &buf
}

// putScratchBuffers returns the three log-softmax scratch pointers to
// their respective pools. Grouped helper so the multiple error-return
// paths in BatchLoss stay one-liners instead of three lines per
// terminus.
func putScratchBuffers(teacherPtr, teacherProbPtr, studentPtr *[]float64) {
	if teacherPtr != nil {
		teacherScratchPool.Put(teacherPtr)
	}
	if teacherProbPtr != nil {
		teacherProbPool.Put(teacherProbPtr)
	}
	if studentPtr != nil {
		studentScratchPool.Put(studentPtr)
	}
}

// NormalizeConfig fills Config defaults: BatchSize floors to 1, Epochs
// floors to 1, Temperature defaults to 1 (or NaN-poisons an invalid
// value so BatchLoss rejects it explicitly rather than silently using a
// nonsensical scale), and Loss defaults to KL. A driver's own training
// loop should call this once per run, exactly as BatchLoss does
// internally for every call.
func NormalizeConfig(cfg Config) Config {
	cfg.Batch = normalizeBatchConfig(cfg.Batch)
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 1
	}
	if cfg.Temperature < 0 || math.IsNaN(cfg.Temperature) || math.IsInf(cfg.Temperature, 0) {
		cfg.Temperature = math.NaN()
	}
	if cfg.Loss == "" {
		cfg.Loss = LossKL
	}
	return cfg
}

func normalizeBatchConfig(cfg BatchConfig) BatchConfig {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	return cfg
}

// BatchLoss computes KL and soft cross-entropy over masked tokens between
// teacher and student logits. mask marks which token positions
// contribute to the loss (nil/empty mask means every position counts).
//
//	loss, err := distill.BatchLoss(teacherLogits, studentLogits, mask, cfg)
func BatchLoss(teacher, student Logits, mask [][]float32, cfg Config) (Loss, error) {
	cfg = NormalizeConfig(cfg)
	switch cfg.Loss {
	case LossKL, LossSoftCrossEntropy:
	default:
		return Loss{}, core.NewError("mlx: unsupported distillation loss kind: " + string(cfg.Loss))
	}
	if err := validateLogitShapes(teacher, student); err != nil {
		return Loss{}, err
	}
	// Validate temperature once at the call boundary — the per-token inner
	// loop invokes logSoftmax{,AndProb}TemperatureInto thousands of times,
	// and the helpers' per-call `temperature <= 0 || NaN || Inf` check is
	// the same gate every iteration. Hoist + pass the pre-computed invTemp
	// so the helpers skip both the per-call validation and the per-call
	// reciprocal division.
	if cfg.Temperature <= 0 || math.IsNaN(cfg.Temperature) || math.IsInf(cfg.Temperature, 0) {
		return Loss{}, errTempInvalid
	}
	invTemp := 1.0 / cfg.Temperature
	var softCE float64
	var entropy float64
	var tokens int
	// Scratch buffers reused across every masked token — vocab size is
	// constant (shape-checked above), so three pre-allocated float64 slices
	// replace per-token allocations inside logSoftmaxInvTempInto +
	// logSoftmaxAndProbInvTempInto. For a 32k vocab and 1000 tokens
	// this skips ~2000 256KB allocations per call.
	// teacherProbScratch holds prob(x) = exp(log_prob(x)) computed once
	// inside the log-softmax loop — the inner accumulator below would
	// otherwise call math.Exp per element to recover it.
	//
	// The buffers themselves are pooled across BatchLoss calls — vocab is
	// process-invariant (tokenizer-fixed), so pool entries hold the right
	// cap from the first call onwards and BatchLoss itself amortises down
	// to zero per-call alloc cost (3 × vocab × 8 B saved per call, e.g.
	// ~768 KB for 32k vocab). Avoiding `defer` here is deliberate — a
	// deferred Put closure heap-allocates the defer record on every call,
	// which would re-introduce the alloc the pool is trying to eliminate.
	// Pool puts run on the explicit return paths below (one per terminal
	// branch).
	var teacherScratch, teacherProbScratch, studentScratch []float64
	var teacherScratchPtr, teacherProbPtr, studentScratchPtr *[]float64
	// Hoist mask-empty once — an empty mask means "all tokens included",
	// so per-cell calls were wasted when the mask is absent or zero-length.
	// maskRows is non-nil only when we need per-row inspection.
	var maskRows [][]float32
	if len(mask) > 0 {
		maskRows = mask
	}
	for i := range teacher {
		// Per-row mask access — fetch maskRow once, then per-column the
		// check is a single len + element compare with no extra branches.
		// Hoist tRow + sRow once per i: the inner loop previously paid for
		// three teacher[i] / two student[i] slice-header loads per token
		// the compiler can't fold because mask/teacher/student aliasing
		// can't be proven away through the function call boundary.
		tRow := teacher[i]
		sRow := student[i]
		upper := len(tRow)
		var maskRow []float32
		if maskRows != nil {
			if i >= len(maskRows) {
				continue
			}
			maskRow = maskRows[i]
			if maskRow == nil {
				continue
			}
			// Cap the inner loop at len(maskRow) — j values past the
			// mask length all hit the original `j >= len(maskRow)`
			// guard and were skipped anyway. Bounding upper eliminates
			// the per-j length check inside the loop.
			if len(maskRow) < upper {
				upper = len(maskRow)
			}
		}
		// Split mask-present vs mask-absent paths — the per-j `if maskRow
		// != nil && maskRow[j] <= 0` check fires every iteration even when
		// the entire batch was called without a mask, which is the common
		// pre-tokenized teacher-forcing path. Mask-absent branch drops the
		// per-token branch + bounds-check entirely.
		if maskRow == nil {
			for j := 0; j < upper; j++ {
				tCell := tRow[j]
				sCell := sRow[j]
				vocab := len(tCell)
				if cap(teacherScratch) < vocab {
					// First-call cap grow (pool warm-up) or vocab-growth
					// across the per-cell variation case. Lift the pool
					// pointer once and grow in place — subsequent cap
					// trips inside this call grow the existing pointer
					// without re-Get'ing a fresh wrapper.
					if teacherScratchPtr == nil {
						teacherScratchPtr = getFloat64Scratch(&teacherScratchPool, vocab)
						teacherProbPtr = getFloat64Scratch(&teacherProbPool, vocab)
						studentScratchPtr = getFloat64Scratch(&studentScratchPool, vocab)
					} else {
						*teacherScratchPtr = make([]float64, vocab)
						*teacherProbPtr = make([]float64, vocab)
						*studentScratchPtr = make([]float64, vocab)
					}
					teacherScratch = *teacherScratchPtr
					teacherProbScratch = *teacherProbPtr
					studentScratch = *studentScratchPtr
				}
				teacherScratch = teacherScratch[:vocab]
				teacherProbScratch = teacherProbScratch[:vocab]
				studentScratch = studentScratch[:vocab]
				if err := logSoftmaxAndProbInvTempInto(tCell, invTemp, teacherScratch, teacherProbScratch); err != nil {
					putScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
					return Loss{}, err
				}
				if err := logSoftmaxInvTempInto(sCell, invTemp, studentScratch); err != nil {
					putScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
					return Loss{}, err
				}
				// Teacher probabilities are already in teacherProbScratch —
				// the inner loop skips the per-element math.Exp the original
				// form paid to recover prob from log-prob. For 32k vocab this
				// saves ~32k math.Exp calls per masked token. Subtracting
				// directly (softCE -= prob*X) folds the negation into the
				// accumulator update so no per-iteration temporary is
				// needed.
				for k, teacherProb := range teacherProbScratch {
					softCE -= teacherProb * studentScratch[k]
					entropy -= teacherProb * teacherScratch[k]
				}
				tokens++
			}
			continue
		}
		for j := 0; j < upper; j++ {
			if maskRow[j] <= 0 {
				continue
			}
			tCell := tRow[j]
			sCell := sRow[j]
			vocab := len(tCell)
			if cap(teacherScratch) < vocab {
				if teacherScratchPtr == nil {
					teacherScratchPtr = getFloat64Scratch(&teacherScratchPool, vocab)
					teacherProbPtr = getFloat64Scratch(&teacherProbPool, vocab)
					studentScratchPtr = getFloat64Scratch(&studentScratchPool, vocab)
				} else {
					*teacherScratchPtr = make([]float64, vocab)
					*teacherProbPtr = make([]float64, vocab)
					*studentScratchPtr = make([]float64, vocab)
				}
				teacherScratch = *teacherScratchPtr
				teacherProbScratch = *teacherProbPtr
				studentScratch = *studentScratchPtr
			}
			teacherScratch = teacherScratch[:vocab]
			teacherProbScratch = teacherProbScratch[:vocab]
			studentScratch = studentScratch[:vocab]
			if err := logSoftmaxAndProbInvTempInto(tCell, invTemp, teacherScratch, teacherProbScratch); err != nil {
				putScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
				return Loss{}, err
			}
			if err := logSoftmaxInvTempInto(sCell, invTemp, studentScratch); err != nil {
				putScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
				return Loss{}, err
			}
			for k, teacherProb := range teacherProbScratch {
				softCE -= teacherProb * studentScratch[k]
				entropy -= teacherProb * teacherScratch[k]
			}
			tokens++
		}
	}
	putScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
	if tokens == 0 {
		return Loss{}, errNoMaskedTokens
	}
	softCE /= float64(tokens)
	entropy /= float64(tokens)
	kl := softCE - entropy
	if kl < 0 && math.Abs(kl) < 1e-12 {
		kl = 0
	}
	if kl < 0 || math.IsNaN(kl) || math.IsInf(kl, 0) {
		return Loss{}, errKLNotFinite
	}
	lossValue := kl
	if cfg.Loss == LossSoftCrossEntropy {
		lossValue = softCE
	}
	return Loss{
		Value:            lossValue,
		KL:               kl,
		SoftCrossEntropy: softCE,
		TeacherEntropy:   entropy,
		Tokens:           tokens,
		Temperature:      cfg.Temperature,
		Kind:             cfg.Loss,
	}, nil
}

// BatchCacheKey returns a stable hash for teacher-logit cache lookup,
// computed directly from a batch's token IDs, shifted targets, and loss
// mask — the three plain arrays every driver's tokenizer batch reduces
// to. Adapted from go-mlx's DistillBatchCacheKey(SFTBatch), which read
// these identical three fields off its driver-specific SFTBatch wrapper
// (batch.Batch.Tokens, batch.Targets, batch.Batch.LossMask); taking them
// directly here keeps the hash engine-agnostic while leaving the
// byte-for-byte hashing untouched.
//
// The key bytes are a hand-rolled, byte-identical re-emission of what
// core.JSONMarshal (encoding/json) would produce for the equivalent
// {tokens, targets, mask} payload — appendBatchCacheKeyJSON walks the
// three numeric slice fields directly into a pre-sized buffer with no
// reflection. The cache KEY bytes MUST stay unchanged — the teacher-logit
// cache is keyed on this hash, so any drift would silently invalidate
// every cached entry.
func BatchCacheKey(tokens, targets [][]int, mask [][]float32) string {
	// encoding/json errors (data.OK == false) on a NaN/Inf mask value;
	// the emitter signals the same via ok == false so both paths take the
	// identical Sprintf fallback below — keeping the key byte-identical
	// whether or not the float path is well-defined.
	if data, ok := appendBatchCacheKeyJSON(nil, tokens, targets, mask); ok {
		return core.SHA256Hex(data)
	}
	payload := struct {
		Tokens  [][]int     `json:"tokens"`
		Targets [][]int     `json:"targets"`
		Mask    [][]float32 `json:"mask"`
	}{Tokens: tokens, Targets: targets, Mask: mask}
	return core.SHA256HexString(core.Sprintf("%+v", payload))
}

// appendBatchCacheKeyJSON emits the BatchCacheKey payload as JSON
// byte-identical to core.JSONMarshal (plain encoding/json with HTML
// escaping on — irrelevant here, the keys are literal ASCII and the
// values carry no strings). Fields emit in {tokens, targets, mask} order
// (encoding/json walks struct fields in declaration order, so there is
// nothing to sort). Returns ok == false on the first NaN/Inf float,
// exactly where json.Marshal would error, so the caller falls back to
// the same Sprintf-based key.
func appendBatchCacheKeyJSON(dst []byte, tokens, targets [][]int, mask [][]float32) ([]byte, bool) {
	// Two-pass exact sizing: a length pass computes the precise emitted
	// byte count so the single make below is sized to the content, never
	// over-provisioned — json.Marshal returned a content-sized slice, so
	// a loose estimate would regress B/op even while it cuts allocs. The
	// length pass also surfaces NaN/Inf before any allocation, so the
	// caller's Sprintf fallback pays nothing here. The emitted bytes are
	// identical either way; only the buffer capacity changes.
	n, ok := batchCacheKeyJSONLen(tokens, targets, mask)
	if !ok {
		return nil, false
	}
	out := make([]byte, 0, len(dst)+n)
	out = append(out, dst...)
	out = append(out, '{', '"', 't', 'o', 'k', 'e', 'n', 's', '"', ':')
	out = appendIntMatrix(out, tokens)
	out = append(out, ',', '"', 't', 'a', 'r', 'g', 'e', 't', 's', '"', ':')
	out = appendIntMatrix(out, targets)
	out = append(out, ',', '"', 'm', 'a', 's', 'k', '"', ':')
	out, _ = appendFloat32Matrix(out, mask) // NaN/Inf already ruled out by the length pass.
	out = append(out, '}')
	return out, true
}

// batchCacheKeyJSONLen returns the exact number of bytes
// appendBatchCacheKeyJSON will emit for the same inputs, or ok == false
// if a NaN/Inf float would make json.Marshal (and therefore the emitter)
// fail. It mirrors the emit walk byte-for-byte: 30 bytes of fixed
// {"tokens":,"targets":,"mask":} scaffold plus the measured length of
// each matrix. Pairing an exact size pass with one make keeps the build
// at a single buffer allocation without over-provisioning.
func batchCacheKeyJSONLen(tokens, targets [][]int, mask [][]float32) (int, bool) {
	// {"tokens": (10) + ,"targets": (11) + ,"mask": (8) + } (1) = 30. The
	// per-matrix [] brackets are counted by the matrix-length helpers, not
	// here.
	const scaffold = 30
	n := scaffold
	n += intMatrixJSONLen(tokens)
	n += intMatrixJSONLen(targets)
	maskLen, ok := float32MatrixJSONLen(mask)
	if !ok {
		return 0, false
	}
	return n + maskLen, true
}

// intMatrixJSONLen returns the emitted byte length of appendIntMatrix.
func intMatrixJSONLen(rows [][]int) int {
	if rows == nil {
		return 4 // null
	}
	n := 2 + max(0, len(rows)-1) // [] plus inter-row commas
	for _, row := range rows {
		n += intArrayJSONLen(row)
	}
	return n
}

// intArrayJSONLen returns the emitted byte length of appendIntArray.
func intArrayJSONLen(values []int) int {
	if values == nil {
		return 4 // null
	}
	n := 2 + max(0, len(values)-1) // [] plus inter-value commas
	for _, v := range values {
		n += int64DecimalLen(int64(v))
	}
	return n
}

// int64DecimalLen returns the number of bytes appendCacheKeyInt64 emits
// for v — the base-10 digit count plus one for a minus sign. uint64(-v)
// handles math.MinInt64 without overflow, matching the emitter.
func int64DecimalLen(v int64) int {
	if v == 0 {
		return 1
	}
	n := 0
	var uv uint64
	if v < 0 {
		n = 1 // '-'
		uv = uint64(-v)
	} else {
		uv = uint64(v)
	}
	for uv > 0 {
		n++
		uv /= 10
	}
	return n
}

// float32MatrixJSONLen returns the emitted byte length of
// appendFloat32Matrix, or ok == false on the first NaN/Inf.
func float32MatrixJSONLen(rows [][]float32) (int, bool) {
	if rows == nil {
		return 4, true // null
	}
	n := 2 + max(0, len(rows)-1) // [] plus inter-row commas
	for _, row := range rows {
		rowLen, ok := float32ArrayJSONLen(row)
		if !ok {
			return 0, false
		}
		n += rowLen
	}
	return n, true
}

// float32ArrayJSONLen returns the emitted byte length of
// appendFloat32Array, or ok == false on the first NaN/Inf.
func float32ArrayJSONLen(values []float32) (int, bool) {
	if values == nil {
		return 4, true // null
	}
	n := 2 + max(0, len(values)-1) // [] plus inter-value commas
	for _, v := range values {
		vLen, ok := float32JSONLen(v)
		if !ok {
			return 0, false
		}
		n += vLen
	}
	return n, true
}

// float32JSONLen returns the number of bytes appendCacheKeyFloat32 emits
// for f, or ok == false for NaN/Inf. It formats into a stack scratch
// buffer with the identical format/cleanup logic and measures the result
// — the only way to know a shortest-round-trip float's width is to format
// it, and the scratch buffer keeps that allocation-free.
func float32JSONLen(f float32) (int, bool) {
	f64 := float64(f)
	if math.IsInf(f64, 0) || math.IsNaN(f64) {
		return 0, false
	}
	abs := math.Abs(f64)
	format := byte('f')
	if abs != 0 {
		if float32(abs) < 1e-6 || float32(abs) >= 1e21 {
			format = 'e'
		}
	}
	var buf [32]byte
	b := strconv.AppendFloat(buf[:0], f64, format, -1, 32)
	n := len(b)
	if format == 'e' {
		if n >= 4 && b[n-4] == 'e' && b[n-3] == '-' && b[n-2] == '0' {
			n-- // the e-09 -> e-9 cleanup drops one byte
		}
	}
	return n, true
}

// appendIntMatrix emits a JSON array-of-arrays of ints, or null for a nil
// outer slice (encoding/json marshals a nil slice as null, a non-nil
// empty slice as []). The nil/empty distinction is preserved at the inner
// level too: [][]int{nil} -> [null], [][]int{{}} -> [[]].
func appendIntMatrix(dst []byte, rows [][]int) []byte {
	if rows == nil {
		return append(dst, 'n', 'u', 'l', 'l')
	}
	dst = append(dst, '[')
	for i, row := range rows {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendIntArray(dst, row)
	}
	return append(dst, ']')
}

// appendIntArray emits a JSON array of ints, or null for a nil slice.
func appendIntArray(dst []byte, values []int) []byte {
	if values == nil {
		return append(dst, 'n', 'u', 'l', 'l')
	}
	dst = append(dst, '[')
	for i, v := range values {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendCacheKeyInt64(dst, int64(v))
	}
	return append(dst, ']')
}

// appendCacheKeyInt64 emits v in base-10 with no leading zeros, matching
// encoding/json / strconv.FormatInt. The digits land in a fixed stack
// buffer so no heap allocation occurs regardless of magnitude; uint64(-v)
// handles math.MinInt64 without overflow (its two's-complement is the
// correct unsigned magnitude).
func appendCacheKeyInt64(dst []byte, v int64) []byte {
	if v == 0 {
		return append(dst, '0')
	}
	var buf [20]byte
	i := len(buf)
	neg := v < 0
	var uv uint64
	if neg {
		uv = uint64(-v)
	} else {
		uv = uint64(v)
	}
	for uv > 0 {
		i--
		buf[i] = byte('0' + uv%10)
		uv /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return append(dst, buf[i:]...)
}

// appendFloat32Matrix emits a JSON array-of-arrays of float32s, or null
// for a nil outer slice. ok is false if any value is NaN or +/-Inf —
// encoding/json errors on those, so the caller must fall back. Nil/empty
// is preserved at both levels, as for the int matrix.
func appendFloat32Matrix(dst []byte, rows [][]float32) ([]byte, bool) {
	if rows == nil {
		return append(dst, 'n', 'u', 'l', 'l'), true
	}
	dst = append(dst, '[')
	for i, row := range rows {
		if i > 0 {
			dst = append(dst, ',')
		}
		var ok bool
		dst, ok = appendFloat32Array(dst, row)
		if !ok {
			return nil, false
		}
	}
	return append(dst, ']'), true
}

// appendFloat32Array emits a JSON array of float32s, or null for a nil
// slice. ok is false on the first NaN/Inf.
func appendFloat32Array(dst []byte, values []float32) ([]byte, bool) {
	if values == nil {
		return append(dst, 'n', 'u', 'l', 'l'), true
	}
	dst = append(dst, '[')
	for i, v := range values {
		if i > 0 {
			dst = append(dst, ',')
		}
		var ok bool
		dst, ok = appendCacheKeyFloat32(dst, v)
		if !ok {
			return nil, false
		}
	}
	return append(dst, ']'), true
}

// appendCacheKeyFloat32 emits a float32 byte-identical to encoding/json's
// float32 encoder (see encoding/json/encode.go floatEncoder.encode):
// widen to float64, pick 'f' unless the float32-precision magnitude is
// < 1e-6 or >= 1e21 (then 'e'), strconv.AppendFloat with bitSize 32 for
// the shortest float32 round-trip, then the e-09 -> e-9 exponent cleanup.
// ok is false for NaN/Inf (json.Marshal errors on those). strconv writes
// into a fixed stack buffer, so no heap allocation per value.
func appendCacheKeyFloat32(dst []byte, f float32) ([]byte, bool) {
	f64 := float64(f)
	if math.IsInf(f64, 0) || math.IsNaN(f64) {
		return dst, false
	}
	abs := math.Abs(f64)
	format := byte('f')
	// Must use float32 comparisons for the underlying float32 value to
	// get the precise cutoffs right (matches encode.go's comment).
	if abs != 0 {
		if float32(abs) < 1e-6 || float32(abs) >= 1e21 {
			format = 'e'
		}
	}
	var buf [32]byte
	b := strconv.AppendFloat(buf[:0], f64, format, -1, 32)
	if format == 'e' {
		// clean up e-09 to e-9
		n := len(b)
		if n >= 4 && b[n-4] == 'e' && b[n-3] == '-' && b[n-2] == '0' {
			b[n-2] = b[n-1]
			b = b[:n-1]
		}
	}
	return append(dst, b...), true
}

func validateLogitShapes(teacher, student Logits) error {
	if len(teacher) == 0 {
		return errTeacherLogitsEmpty
	}
	if len(teacher) != len(student) {
		return errLogitBatch
	}
	for i := range teacher {
		// Hoist the per-row [][]float32 slice headers once so the inner
		// loop re-indexing pays one pointer load instead of two double-
		// indexes per token.
		tRow := teacher[i]
		sRow := student[i]
		if len(tRow) != len(sRow) {
			return errLogitSeq
		}
		for j := range tRow {
			tVocab := len(tRow[j])
			if tVocab == 0 {
				return errLogitEmptyVocab
			}
			if tVocab != len(sRow[j]) {
				return errLogitVocab
			}
		}
	}
	return nil
}

// logSoftmaxAndProbInvTempInto writes both log_prob and prob for
// each logit, given pre-computed invTemp (1/temperature). logOut[i] =
// log(softmax(logits/temp))[i] and probOut[i] = exp(logOut[i]). The
// BatchLoss inner loop needs both teacher log-probs (for the entropy
// term) and teacher probs (as the weight on the softCE / entropy
// accumulators). Capturing prob during the renormalize pass here skips
// a per-element math.Exp inside the accumulator loop entirely. The
// invTemp + buffer-shape preconditions are caller-owned (validated once
// in BatchLoss), so the per-token call pays no validation overhead.
func logSoftmaxAndProbInvTempInto(logits []float32, invTemp float64, logOut, probOut []float64) error {
	maxLogit := math.Inf(-1)
	for i, logit := range logits {
		value := float64(logit) * invTemp
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errLogitNotFinite
		}
		logOut[i] = value
		if value > maxLogit {
			maxLogit = value
		}
	}
	// Compute exp(value - maxLogit) and accumulate the partition fn.
	// Store the unnormalised exp in probOut so we don't need to
	// recompute math.Exp during the normalise pass below.
	var sumExp float64
	for i, value := range logOut {
		e := math.Exp(value - maxLogit)
		probOut[i] = e
		sumExp += e
	}
	logDenom := maxLogit + math.Log(sumExp)
	invSum := 1.0 / sumExp
	for i, value := range logOut {
		logOut[i] = value - logDenom
		probOut[i] *= invSum
	}
	return nil
}

// logSoftmaxInvTempInto writes len(logits) log-softmax values into out,
// given pre-computed invTemp (1/temperature). out must be pre-sized to
// len(logits); callers in the distillation hot loop reuse the same
// scratch buffer across every masked token to skip per-token allocation
// of vocab-sized float64 slices. invTemp + buffer-shape preconditions
// are caller-owned (validated once in BatchLoss), so the per-token call
// pays no validation overhead.
func logSoftmaxInvTempInto(logits []float32, invTemp float64, out []float64) error {
	maxLogit := math.Inf(-1)
	for i, logit := range logits {
		value := float64(logit) * invTemp
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errLogitNotFinite
		}
		out[i] = value
		if value > maxLogit {
			maxLogit = value
		}
	}
	var sumExp float64
	for _, value := range out {
		sumExp += math.Exp(value - maxLogit)
	}
	logDenom := maxLogit + math.Log(sumExp)
	for i, value := range out {
		out[i] = value - logDenom
	}
	return nil
}

func cloneLogits(logits Logits) Logits {
	if len(logits) == 0 {
		return nil
	}
	// Three-flat-buffer clone — first count rows + cells across the
	// batch, then allocate THREE flat buffers (the outer Logits, one
	// shared [][]float32 for the middle row-slice-headers, one shared
	// []float32 for all cell data). Each per-batch middle slice + per-cell
	// []float32 are carved as 3-index slice views into the shared
	// backings instead of paying their own malloc.
	//
	// For a 4×128×32000 teacher tensor:
	//   pre:   513 allocs (1 outer + 4 middle + 4×128 inner)
	//   2-pass:  6 allocs (1 outer + 4 middle + 1 flat cell buffer)
	//   3-pass:  3 allocs (1 outer + 1 flat middle + 1 flat cell)
	//
	// The flat-backing form also gives the resulting clone better cache
	// locality (sequential float32 + sequential slice-header stride)
	// versus the per-cell-alloc form where each row could land on a
	// distinct page.
	var totalRows, totalCells int
	for i := range logits {
		row := logits[i]
		totalRows += len(row)
		for j := range row {
			totalCells += len(row[j])
		}
	}
	out := make(Logits, len(logits))
	if totalRows == 0 {
		return out
	}
	rowBacking := make([][]float32, totalRows)
	flat := make([]float32, totalCells)
	rowCursor := 0
	cellCursor := 0
	for i := range logits {
		row := logits[i]
		rowsHere := len(row)
		rowEnd := rowCursor + rowsHere
		outRow := rowBacking[rowCursor:rowEnd:rowEnd]
		for j := range row {
			src := row[j]
			next := cellCursor + len(src)
			dst := flat[cellCursor:next:next]
			copy(dst, src)
			outRow[j] = dst
			cellCursor = next
		}
		out[i] = outRow
		rowCursor = rowEnd
	}
	return out
}
