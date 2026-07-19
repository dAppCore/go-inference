// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// train_trainer.go promotes the real-session SFT proof (TestRealSessionHeadLoRASFT) into an exported,
// reusable trainer: a LoRA adapter on the output head, trained over the engine's OWN forward. The base
// stays frozen and is run by the real ArchSession (ForwardCaptureHiddens), so the trainer is
// architecture-agnostic on the base — it drives a real gemma-4 E2B/E4B (PLE) session exactly, because
// the whole PLE forward is the engine's, not a host re-derivation. Only the head LoRA (A [rank,dModel],
// B [vocab,rank]) and its two AdamW states live here; each Step captures the frozen final hidden, adds
// the LoRA delta to the frozen head logits, and steps A/B down the cross-entropy of the sequence's own
// next token. This is the LoRATrainer half of the engine.Trainer seam (go/engine/trainer.go); the shape
// mirrors go-mlx pkg/metal.LoRAAdapter (Step / StepAccumulated / Loss / Save), with the no-cgo gradient
// kernels (train_lora.go / train_optim.go) standing in for mlx autodiff.
//
// Scope note (the honest boundary, #31 → #40 → #42): the HEAD LoRA trains on EVERY shape — the
// seam is exact over any real base via the frozen capture. Per-LAYER projection targets
// (TargetKeys naming q_proj/v_proj/…) train through the FD-gated real-arch layer reference
// (train_real_layer.go + train_real_shared.go + train_trainer_layers.go) — but ONLY on shapes
// whose every layer feature that reference covers (validatePerLayerLoRAShape): dense attention
// layers with QK-norm / sandwich norms / value-norm / K==V / PLE / sliding+partial+proportional
// rope / layer scalars / KV-cache sharing (the gemma4 E2B/E4B tail — the OWNER-ROUTED backward:
// every consumer's dK/dV accumulates back through the owner's k_proj/v_proj and hidden, so any of
// the 7 targets trains on any layer; consumers simply carry no k/v of their own). Any other
// feature — MoE, recurrent mixers, projection biases, logit soft-caps, non-rotary positions —
// REFUSES per-layer targets at open, naming the blocking feature, exactly the #31 shape: never
// silently train less (or wronger) than asked.

// loraTargetHead is the one adapter target this trainer trains: the output head. It is the tensor-name
// prefix LoRATrainer.Save writes (lm_head.lora_a / lm_head.lora_b), the name the load path reads back
// (lora_apply.go), and the only inference.LoRAConfig.TargetKeys entry validateHeadLoRATargets accepts.
const loraTargetHead = "lm_head"

// validateHeadLoRATargets refuses a LoRA configuration this head-only trainer will not honour — the
// honesty gate of #31. Empty TargetKeys means the engine's default target (the head) and passes; an
// explicit ["lm_head"] passes (it names exactly what is trained); any other key is a per-layer
// projection request (q_proj, v_proj, down_proj, …) whose backward is not wired into this trainer, so
// it is refused LOUDLY here rather than silently training less than asked. The error names both the
// requested keys and the supported target.
func validateHeadLoRATargets(cfg inference.LoRAConfig) error {
	for _, key := range cfg.TargetKeys {
		if key != loraTargetHead {
			return core.NewError(core.Concat(
				"native.NewLoRATrainer: LoRA TargetKeys ", core.Sprintf("%v", cfg.TargetKeys),
				" request a per-layer projection adapter (", core.Sprintf("%q", key),
				"); this trainer trains the HEAD adapter only (", loraTargetHead,
				") — the backward through the engine's per-layer projection matmuls is not wired (#31). ",
				"Leave TargetKeys empty (or [", loraTargetHead, "]) to train the head."))
		}
	}
	return nil
}

// adapterConfigJSON is the go-mlx on-disk adapter_config.json: rank/alpha, the number of decoder layers
// the adapter spans (0 for a head-only adapter), and the target projection names. Written by
// LoRATrainer.Save and read back by the load path (lora_apply.go).
type adapterConfigJSON struct {
	Rank       int      `json:"rank"`
	Alpha      float32  `json:"alpha"`
	NumLayers  int      `json:"num_layers"`
	LoRALayers []string `json:"lora_layers"`
}

// LoRATrainer is a retained head-LoRA SFT session over a loaded native model. It wraps a fresh frozen
// base decode session, the trainable head LoRA factors, and their AdamW state — the exported promotion
// of the real-session SFT proof. Single-goroutine (the ArchSession contract).
type LoRATrainer struct {
	sess      *ArchSession
	finalNorm []float32 // [dModel] bf16→f32, the frozen final RMSNorm weight
	softCap   float32   // final-logit soft-cap (arch.SoftCap; 0 = none) — the trainer's head forward matches serving (train_softcap.go)
	lmHead    []float32 // [vocab,dModel] bf16→f32, the frozen output head
	a, b      []float32 // trainable (head mode): A [rank,dModel], B [vocab,rank] (B starts at zero → adapter is a no-op)
	optA      *AdamW
	optB      *AdamW
	dModel    int
	vocab     int
	rank      int
	alpha     float32
	scaling   float32 // alpha/rank
	eps       float32

	// per-layer projection mode (#40 stage 3) — set when TargetKeys name layer projections and the
	// shape passed validatePerLayerLoRAShape; the head factors above are then unused (nil).
	perLayer  bool
	layers    []*RealTrainLayerF32 // frozen per-layer templates (T + PLEInput filled per sequence)
	adapters  []*layerLoRAAdapter  // the trainable (layer, projection) adapters + optimiser state
	shareFrom []int                // per-layer KV-cache owner map (arch.Layer[i].KVShareFrom; == i for owners)
	hasPLE    bool                 // the session computes per-layer inputs (gemma4 E2B/E4B)
}

// NewLoRATrainer opens a head-LoRA trainer over tm: it takes a fresh frozen base session, widens the
// final-norm + head weights to f32, and initialises the LoRA factors (A small-random, B zero, so the
// adapter starts as the identity). cfg.LoRA supplies rank/alpha (defaults 8/16); cfg.LearningRate the
// AdamW step (default 0.05). A cfg.LoRA.TargetKeys naming per-layer projections is refused loudly —
// this trainer trains the head only (validateHeadLoRATargets, #31). The trainer OWNS the base
// session — Close releases it.
func NewLoRATrainer(tm *NativeTokenModel, cfg inference.TrainingConfig) (*LoRATrainer, error) {
	// The honesty gates first (#31/#40): a config this trainer will not honour is refused before
	// any resource is touched. TargetKeys naming per-layer projections open the PER-LAYER path —
	// but ONLY for shapes whose every layer feature has a green FD gate (validatePerLayerLoRAShape);
	// any other shape keeps the refusal, naming the blocking feature, so a per-layer request can
	// never silently train head-only OR train through an unmodelled feature.
	perLayer, err := loraTargetMode(cfg.LoRA)
	if err != nil {
		return nil, err
	}
	if perLayer {
		if err := validatePerLayerLoRAShape(tm); err != nil {
			return nil, core.NewError(core.Concat(
				"native.NewLoRATrainer: LoRA TargetKeys ", core.Sprintf("%v", cfg.LoRA.TargetKeys),
				" request a per-layer projection adapter; ", err.Error()))
		}
	} else if err := validateHeadLoRATargets(cfg.LoRA); err != nil {
		return nil, err
	}
	if tm == nil || tm.bf16 == nil {
		return nil, core.NewError("native.NewLoRATrainer: trainer needs a loaded bf16 model")
	}
	if len(tm.bf16.FinalNorm) == 0 || len(tm.bf16.LMHead) == 0 {
		return nil, core.NewError("native.NewLoRATrainer: model is missing the final norm or head weight")
	}
	dModel, vocab, eps := tm.arch.Hidden, tm.arch.Vocab, tm.arch.Eps
	if dModel <= 0 || vocab <= 0 {
		return nil, core.NewError("native.NewLoRATrainer: model reports a zero hidden or vocab size")
	}
	rank := cfg.LoRA.Rank
	if rank <= 0 {
		rank = 8
	}
	alpha := cfg.LoRA.Alpha
	if alpha == 0 {
		alpha = 16
	}
	lr := float32(cfg.LearningRate)
	if lr <= 0 {
		lr = 0.05
	}

	stepper, err := tm.OpenSession()
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		if closer, closeOK := stepper.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return nil, core.NewError("native.NewLoRATrainer: token model does not open an ArchSession")
	}

	t := &LoRATrainer{
		sess:      sess,
		finalNorm: bf16ToF32Slice(tm.bf16.FinalNorm),
		softCap:   tm.arch.SoftCap,
		lmHead:    bf16ToF32Slice(tm.bf16.LMHead),
		dModel:    dModel,
		vocab:     vocab,
		rank:      rank,
		alpha:     alpha,
		scaling:   alpha / float32(rank),
		eps:       eps,
	}
	if perLayer {
		shareFrom := make([]int, len(tm.arch.Layer))
		for li := range tm.arch.Layer {
			shareFrom[li] = tm.arch.Layer[li].KVShareFrom
		}
		layers, lerr := buildRealLayerTemplates(tm.bf16, tm.arch)
		if lerr == nil {
			t.adapters, lerr = buildLayerAdapters(layers, cfg.LoRA.TargetKeys, rank, lr)
		}
		if lerr != nil {
			_ = sess.Close()
			return nil, lerr
		}
		t.perLayer = true
		t.layers = layers
		t.shareFrom = shareFrom
		t.hasPLE = sess.perLayerInput != nil
		return t, nil
	}
	t.a = initLoRAFactorA(rank*dModel, dModel)
	t.b = make([]float32, vocab*rank)
	t.optA = NewAdamW(rank*dModel, lr, 0)
	t.optB = NewAdamW(vocab*rank, lr, 0)
	return t, nil
}

// initLoRAFactorA fills the LoRA A factor with small deterministic pseudo-random values (Kaiming-style
// stddev 1/√dModel), so training is reproducible and starts from a non-degenerate A while B=0 keeps the
// adapter an initial no-op. Deterministic LCG — no RNG dependency, no per-run drift.
func initLoRAFactorA(n, dModel int) []float32 {
	a := make([]float32, n)
	stddev := float32(1.0 / math.Sqrt(float64(dModel)))
	state := uint32(0x9E3779B9)
	for i := range a {
		state = state*1664525 + 1013904223
		u := float32(state) / float32(1<<32) // [0,1)
		a[i] = (u*2 - 1) * stddev            // [-stddev, stddev)
	}
	return a
}

// forwardFrozen runs the engine's real forward over ids and returns the post-final-norm hidden
// (normed [T,dModel]) and the frozen base head logits (baseLogits [T,vocab]) — the frozen half of every
// step. The base weights never change, so this is the model's own (PLE-correct) forward; the LoRA delta
// is added on top of baseLogits. The head LoRA needs only the FINAL hidden, so the forward rides the
// batched-dense capture (ForwardCaptureFinalHidden — 17.8× over the serial walk at T=128 on E2B bf16,
// which was 88% of the SFT step's wall); the serial per-layer capture stays the full-stack backward's
// forward.
func (t *LoRATrainer) forwardFrozen(ids []int32) (normed, baseLogits []float32, rows int, err error) {
	lastHidden, err := t.sess.ForwardCaptureFinalHidden(ids)
	if err != nil {
		return nil, nil, 0, err
	}
	tokens := len(ids)
	hLast := bf16ToF32Slice(lastHidden) // [T,dModel]
	normed = rmsNormForwardF32(hLast, t.finalNorm, tokens, t.dModel, t.eps)
	baseLogits, err = MatMulF32NT(normed, t.lmHead, tokens, t.dModel, t.vocab)
	if err != nil {
		return nil, nil, 0, err
	}
	return normed, baseLogits, tokens, nil
}

// lossMaskRows resolves a batch's LossMask for ONE sequence into the prediction-row indices that
// contribute to the loss. The mask is per TOKEN over the full sequence (len(ids) entries): entry t marks
// whether token t contributes AS A TARGET, so prediction row p (hidden p predicting token p+1) trains
// iff mask[p+1] > 0 — the same target-token semantics as engine/hip's hipTrainerIncludesTarget. An
// unset mask (no Values) returns masked=false: every row trains, the pre-mask behaviour. A set mask
// that does not cover the sequence (missing sample row, or a row whose length is not the token count)
// is a malformed batch and is refused LOUDLY — never silently treated as all-masked. Values are a
// binary restriction (positive = contributes); fractional loss WEIGHTING is not implemented.
func lossMaskRows(mask inference.LossMask, sample, tokens int) (rows []int, masked bool, err error) {
	if len(mask.Values) == 0 {
		return nil, false, nil
	}
	if sample >= len(mask.Values) {
		return nil, true, core.NewError(core.Concat(
			"native.LoRATrainer: batch.LossMask has ", core.Sprintf("%d", len(mask.Values)),
			" rows but sequence ", core.Sprintf("%d", sample),
			" needs one — a set mask must carry a row per sequence (or clear the mask)"))
	}
	row := mask.Values[sample]
	if len(row) != tokens {
		return nil, true, core.NewError(core.Concat(
			"native.LoRATrainer: batch.LossMask row ", core.Sprintf("%d", sample),
			" has ", core.Sprintf("%d", len(row)), " entries but the sequence has ",
			core.Sprintf("%d", tokens), " tokens — the mask is per token"))
	}
	rows = make([]int, 0, tokens-1)
	for p := 0; p < tokens-1; p++ {
		if row[p+1] > 0 {
			rows = append(rows, p)
		}
	}
	return rows, true, nil
}

// gatherRowsF32 gathers the given rows of a row-major [*, width] tensor into a packed
// [len(rows), width] tensor — the loss-mask row selection. Rows must be in range.
func gatherRowsF32(src []float32, rows []int, width int) []float32 {
	out := make([]float32, len(rows)*width)
	for i, r := range rows {
		copy(out[i*width:(i+1)*width], src[r*width:(r+1)*width])
	}
	return out
}

// forwardFrozenRows is forwardFrozen restricted to the given prediction rows — the loss-masked frozen
// half. The engine forward still runs over the WHOLE sequence (a masked token is still an input every
// later position attends), but the final norm + frozen head matmul run only on the contributing rows,
// so a response-masked batch pays the [rows × vocab] head cost for its response rows only. Row-for-row
// identical to forwardFrozen's rows (both the norm and the head matmul are row-independent).
func (t *LoRATrainer) forwardFrozenRows(ids []int32, rows []int) (normedRows, baseLogits []float32, err error) {
	lastHidden, err := t.sess.ForwardCaptureFinalHidden(ids)
	if err != nil {
		return nil, nil, err
	}
	hRows := gatherRowsF32(bf16ToF32Slice(lastHidden), rows, t.dModel)
	normedRows = rmsNormForwardF32(hRows, t.finalNorm, len(rows), t.dModel, t.eps)
	baseLogits, err = MatMulF32NT(normedRows, t.lmHead, len(rows), t.dModel, t.vocab)
	if err != nil {
		return nil, nil, err
	}
	return normedRows, baseLogits, nil
}

// seqGrads runs one sequence's head-LoRA forward+backward under the current A/B and returns its loss,
// the gradients of A and B, and the number of prediction rows that contributed. Targets are the
// sequence's own next token (causal SFT): hidden t predicts token t+1, so the trainable rows are
// 0..T-2 — restricted to the response rows when the batch carries a LossMask (lossMaskRows): a masked
// position contributes zero loss and zero gradient, and the per-sequence mean divides by the UNMASKED
// row count. A fully-masked sequence returns rows=0 with no error (and skips the forward entirely) —
// the caller drops it from the batch mean. No optimiser step (the caller accumulates then steps).
func (t *LoRATrainer) seqGrads(ids []int32, mask inference.LossMask, sample int) (loss float32, dA, dB []float32, rows int, err error) {
	if len(ids) < 2 {
		return 0, nil, nil, 0, core.NewError("native.LoRATrainer: a training sequence needs at least 2 tokens")
	}
	maskRows, masked, err := lossMaskRows(mask, sample, len(ids))
	if err != nil {
		return 0, nil, nil, 0, err
	}
	if masked && len(maskRows) == 0 {
		return 0, nil, nil, 0, nil // every position masked: nothing to train, nothing to forward
	}

	var normedPred, baseLogits []float32
	var targets []int32
	if masked {
		normedPred, baseLogits, err = t.forwardFrozenRows(ids, maskRows)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		rows = len(maskRows)
		targets = make([]int32, rows)
		for i, p := range maskRows {
			targets[i] = ids[p+1]
		}
	} else {
		var normed []float32
		var tokens int
		normed, baseLogits, tokens, err = t.forwardFrozen(ids)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		rows = tokens - 1
		normedPred = normed[:rows*t.dModel]
		baseLogits = baseLogits[:rows*t.vocab]
		targets = make([]int32, rows)
		for i := range rows {
			targets[i] = ids[i+1]
		}
	}
	xA, delta, err := LoRAForwardF32(normedPred, t.a, t.b, rows, t.dModel, t.vocab, t.rank, t.scaling)
	if err != nil {
		return 0, nil, nil, 0, err
	}
	logits := make([]float32, rows*t.vocab)
	for i := range logits {
		logits[i] = baseLogits[i] + delta[i]
	}
	softcapForwardF32(logits, t.softCap) // match serving's capped head (no-op when 0)
	loss, dLogits, err := CrossEntropyBackwardF32Auto(logits, targets, rows, t.vocab)
	if err != nil {
		return 0, nil, nil, 0, err
	}
	softcapBackwardScaleF32(dLogits, logits, t.softCap)
	dA, dB, _, err = LoRABackwardF32(dLogits, normedPred, t.a, t.b, xA, rows, t.dModel, t.vocab, t.rank, t.scaling)
	if err != nil {
		return 0, nil, nil, 0, err
	}
	return loss, dA, dB, rows, nil
}

// accumulate sums the per-sequence loss and A/B gradients across every sequence in batch (no step),
// honouring batch.LossMask: a fully-masked sequence contributes nothing and is not counted in n, so
// the batch mean divides by the sequences that actually trained.
func (t *LoRATrainer) accumulate(batch inference.Batch) (lossSum float64, sumDA, sumDB []float32, n int, err error) {
	sumDA = make([]float32, len(t.a))
	sumDB = make([]float32, len(t.b))
	for si, ids := range batch.TokenIDs {
		loss, dA, dB, rows, e := t.seqGrads(ids, batch.LossMask, si)
		if e != nil {
			return 0, nil, nil, 0, e
		}
		if rows == 0 {
			continue // fully-masked sequence: zero loss, zero gradient, not counted
		}
		for i := range sumDA {
			sumDA[i] += dA[i]
		}
		for i := range sumDB {
			sumDB[i] += dB[i]
		}
		lossSum += float64(loss)
		n++
	}
	return lossSum, sumDA, sumDB, n, nil
}

// applyMeanStep scales the summed gradients by 1/count and applies one AdamW update to A and B.
func (t *LoRATrainer) applyMeanStep(sumDA, sumDB []float32, count int) error {
	inv := float32(1.0 / float64(count))
	for i := range sumDA {
		sumDA[i] *= inv
	}
	for i := range sumDB {
		sumDB[i] *= inv
	}
	if err := t.optA.Step(t.a, sumDA); err != nil {
		return err
	}
	return t.optB.Step(t.b, sumDB)
}

// Step runs one SFT gradient step over batch (one AdamW update from the batch-mean gradient) and returns
// the mean cross-entropy loss. batch.LossMask, when set, restricts the loss to response positions —
// masked positions contribute zero loss and zero gradient, and every mean divides by the UNMASKED
// counts (the engine.Trainer contract); a batch whose mask leaves nothing trainable is refused rather
// than divided by zero. Implements engine.Trainer.
func (t *LoRATrainer) Step(batch inference.Batch) (float64, error) {
	if len(batch.TokenIDs) == 0 {
		return 0, core.NewError("native.LoRATrainer.Step: empty batch")
	}
	if t.perLayer {
		lossSum, sumDA, sumDB, n, err := t.perLayerAccumulate(batch)
		if err != nil {
			return 0, err
		}
		if n == 0 {
			return 0, core.NewError("native.LoRATrainer.Step: batch produced no trainable sequences (a set LossMask masks every position)")
		}
		if err := t.perLayerApplyMeanStep(sumDA, sumDB, n); err != nil {
			return 0, err
		}
		return lossSum / float64(n), nil
	}
	lossSum, sumDA, sumDB, n, err := t.accumulate(batch)
	if err != nil {
		return 0, err
	}
	if n == 0 {
		return 0, core.NewError("native.LoRATrainer.Step: batch produced no trainable sequences (a set LossMask masks every position)")
	}
	if err := t.applyMeanStep(sumDA, sumDB, n); err != nil {
		return 0, err
	}
	return lossSum / float64(n), nil
}

// StepAccumulated accumulates the gradients of every micro-batch and applies ONE AdamW update from their
// combined mean. Returns the mean loss across all sequences. Implements engine.Trainer.
func (t *LoRATrainer) StepAccumulated(batches []inference.Batch) (float64, error) {
	if len(batches) == 0 {
		return 0, core.NewError("native.LoRATrainer.StepAccumulated: no batches")
	}
	if t.perLayer {
		totalDA := make([][]float32, len(t.adapters))
		totalDB := make([][]float32, len(t.adapters))
		for ai, ad := range t.adapters {
			totalDA[ai] = make([]float32, len(ad.a))
			totalDB[ai] = make([]float32, len(ad.b))
		}
		var lossSum float64
		total := 0
		for _, batch := range batches {
			ls, sumDA, sumDB, n, err := t.perLayerAccumulate(batch)
			if err != nil {
				return 0, err
			}
			for ai := range totalDA {
				for i := range totalDA[ai] {
					totalDA[ai][i] += sumDA[ai][i]
				}
				for i := range totalDB[ai] {
					totalDB[ai][i] += sumDB[ai][i]
				}
			}
			lossSum += ls
			total += n
		}
		if total == 0 {
			return 0, core.NewError("native.LoRATrainer.StepAccumulated: batches produced no trainable sequences (a set LossMask masks every position)")
		}
		if err := t.perLayerApplyMeanStep(totalDA, totalDB, total); err != nil {
			return 0, err
		}
		return lossSum / float64(total), nil
	}
	totalDA := make([]float32, len(t.a))
	totalDB := make([]float32, len(t.b))
	var lossSum float64
	total := 0
	for _, batch := range batches {
		ls, sumDA, sumDB, n, err := t.accumulate(batch)
		if err != nil {
			return 0, err
		}
		for i := range totalDA {
			totalDA[i] += sumDA[i]
		}
		for i := range totalDB {
			totalDB[i] += sumDB[i]
		}
		lossSum += ls
		total += n
	}
	if total == 0 {
		return 0, core.NewError("native.LoRATrainer.StepAccumulated: batches produced no trainable sequences (a set LossMask masks every position)")
	}
	if err := t.applyMeanStep(totalDA, totalDB, total); err != nil {
		return 0, err
	}
	return lossSum / float64(total), nil
}

// Loss is the forward-only mean cross-entropy over batch under the current adapter weights: no
// gradients, no optimiser update — the validation lane, honouring batch.LossMask exactly as Step does
// (masked positions contribute nothing; the mean divides by the unmasked counts). Implements
// engine.Trainer.
func (t *LoRATrainer) Loss(batch inference.Batch) (float64, error) {
	if len(batch.TokenIDs) == 0 {
		return 0, core.NewError("native.LoRATrainer.Loss: empty batch")
	}
	if t.perLayer {
		return t.perLayerLoss(batch)
	}
	var lossSum float64
	n := 0
	for si, ids := range batch.TokenIDs {
		if len(ids) < 2 {
			return 0, core.NewError("native.LoRATrainer.Loss: a sequence needs at least 2 tokens")
		}
		maskRows, masked, err := lossMaskRows(batch.LossMask, si, len(ids))
		if err != nil {
			return 0, err
		}
		if masked && len(maskRows) == 0 {
			continue // fully-masked sequence: contributes nothing to the validation mean
		}
		var normedPred, baseLogits []float32
		var targets []int32
		var rows int
		if masked {
			normedPred, baseLogits, err = t.forwardFrozenRows(ids, maskRows)
			if err != nil {
				return 0, err
			}
			rows = len(maskRows)
			targets = make([]int32, rows)
			for i, p := range maskRows {
				targets[i] = ids[p+1]
			}
		} else {
			var normed []float32
			var tokens int
			normed, baseLogits, tokens, err = t.forwardFrozen(ids)
			if err != nil {
				return 0, err
			}
			rows = tokens - 1
			normedPred = normed[:rows*t.dModel]
			baseLogits = baseLogits[:rows*t.vocab]
			targets = make([]int32, rows)
			for i := range rows {
				targets[i] = ids[i+1]
			}
		}
		_, delta, err := LoRAForwardF32(normedPred, t.a, t.b, rows, t.dModel, t.vocab, t.rank, t.scaling)
		if err != nil {
			return 0, err
		}
		logits := make([]float32, rows*t.vocab)
		for i := range logits {
			logits[i] = baseLogits[i] + delta[i]
		}
		softcapForwardF32(logits, t.softCap)
		loss, _, err := CrossEntropyBackwardF32Auto(logits, targets, rows, t.vocab)
		if err != nil {
			return 0, err
		}
		lossSum += float64(loss)
		n++
	}
	if n == 0 {
		return 0, core.NewError("native.LoRATrainer.Loss: batch produced no scoreable sequences (a set LossMask masks every position)")
	}
	return lossSum / float64(n), nil
}

// Save writes the trained head LoRA as a reloadable adapter package — adapter.safetensors (the A/B
// factors as F32) + adapter_config.json — in the go-mlx on-disk format. The head is named "lm_head"
// (lm_head.lora_a / lm_head.lora_b); the native load path honours it via AdapterPath (lora_apply.go).
// Implements engine.Trainer.
func (t *LoRATrainer) Save(path string) error {
	if t.perLayer {
		return t.perLayerSave(path)
	}
	if path == "" {
		return core.NewError("native.LoRATrainer.Save: path is required")
	}
	if res := core.MkdirAll(path, core.FileMode(0o755)); !res.OK {
		return core.E("native.LoRATrainer.Save", "ensure adapter dir", resultErr(res))
	}
	tensors := map[string]safetensors.Tensor{
		"lm_head.lora_a": {Dtype: "F32", Shape: []int{t.rank, t.dModel}, Data: safetensors.EncodeFloat32(t.a)},
		"lm_head.lora_b": {Dtype: "F32", Shape: []int{t.vocab, t.rank}, Data: safetensors.EncodeFloat32(t.b)},
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		return core.E("native.LoRATrainer.Save", "encode adapter safetensors", err)
	}
	if werr := coreio.Local.Write(core.PathJoin(path, "adapter.safetensors"), string(blob)); werr != nil {
		return core.E("native.LoRATrainer.Save", "write adapter.safetensors", werr)
	}
	cfg := adapterConfigJSON{Rank: t.rank, Alpha: t.alpha, NumLayers: 0, LoRALayers: []string{loraTargetHead}}
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		return core.E("native.LoRATrainer.Save", "marshal adapter_config.json", nil)
	}
	if werr := coreio.Local.Write(core.PathJoin(path, "adapter_config.json"), string(cj.Bytes())); werr != nil {
		return core.E("native.LoRATrainer.Save", "write adapter_config.json", werr)
	}
	return nil
}

// Close releases the retained frozen base session. Implements engine.Trainer.
func (t *LoRATrainer) Close() error {
	if t == nil || t.sess == nil {
		return nil
	}
	err := t.sess.Close()
	t.sess = nil
	return err
}

// resultErr unwraps a core.Result's error value (nil when the Result carried no error).
func resultErr(res core.Result) error {
	if err, ok := res.Value.(error); ok {
		return err
	}
	return nil
}
