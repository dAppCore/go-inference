// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/internal/enginegate"
)

// TestAssistantQuantTargetDraftSelfConsistency is the quant-lane extension of the
// cross-engine MTP parity instrument (pkg/metal/model/gemma4). It was built as a
// fails-by-design reproducer for the quant-target 0%-acceptance bug and now guards
// the fix: an ICB (quant) session's stateLayerViews() re-materialised the drafter's
// K/V views from the session's unused, EMPTY paged cache on every extraction after
// the first, zeroing the target Key the drafter cross-attends (see probe C and the
// dedicated regression in assistant_quant_kv_test.go).
// This is a NATIVE-ONLY discriminator (metal cannot even load the 4-bit E2B target):
// the SAME drafter is attached to a bf16 session and a 4-bit session of the SAME
// model, the same prompt is prefilled, and every drafter-facing input is fingerprinted
// side by side. The two targets are the same nominal weights, so every probe should
// agree within quantisation noise — the one that doesn't is the defect:
//
//	probe A — embedID: the target token embedding fed to the draft concat
//	          (quant dequant+scale vs bf16 lookup+scale).
//	probe B — the boundary seed hidden (retention convention on quant sessions).
//	probe C — the per-layer-type target K/V slabs the drafter cross-attends
//	          (stateLayerViews extraction from the quant KV cache).
//	stage D — the draft block, then SELF-verify: the quant target judging its own
//	          drafter's proposals. The bf16 session accepts these; near-zero HERE
//	          with healthy probes = the verify row mapping.
func TestAssistantQuantTargetDraftSelfConsistency(t *testing.T) {
	quantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	bfDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-bf16")
	drafterDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")

	const draftTokens = 4
	// A deterministic prompt whose opening tokens the bf16 and 4-bit targets agree on.
	// Stage D is a SINGLE draft block at the prompt boundary: greedy decode forks from
	// the drafter's proposal at any near-tie, so a prompt whose very first token is a
	// quantisation near-tie (e.g. "Name the planets…" → bf16 opens "The", the 4-bit
	// target opens "Here") makes the quant target reject the whole block for reasons
	// unrelated to the drafter-facing inputs under test. This factual-recall prompt has
	// no such first-token fork, so a healthy quant drafter is accepted just like bf16.
	prompt := "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"

	load := func(dir string) (*ArchSession, *AssistantPair, []int32) {
		t.Helper()
		sess, err := LoadDir(dir, 640)
		if err != nil {
			t.Fatalf("LoadDir(%s): %v", dir, err)
		}
		t.Cleanup(func() { sess.Close() })
		pair, err := LoadAssistantPairDirs(dir, drafterDir)
		if err != nil {
			t.Fatalf("LoadAssistantPairDirs(%s): %v", dir, err)
		}
		t.Cleanup(func() { pair.Close() })
		ids := pair.Assistant.Tok.Encode(prompt)
		if len(ids) < 4 {
			t.Fatalf("prompt tokenised to %d ids", len(ids))
		}
		if err := sess.prepareAssistantPrompt(ids); err != nil {
			t.Fatalf("prepareAssistantPrompt(%s): %v", dir, err)
		}
		return sess, pair, ids
	}

	sessBF, pairBF, ids := load(bfDir)
	sessQ, pairQ, idsQ := load(quantDir)
	if len(ids) != len(idsQ) {
		t.Fatalf("tokenisations differ: bf16 %d vs quant %d ids", len(ids), len(idsQ))
	}
	lastToken := ids[len(ids)-1]

	// ---- probe A: the target token embedding the draft concat consumes ----
	embBF, err := sessBF.embedID(lastToken)
	if err != nil {
		t.Fatalf("probe A: bf16 embedID: %v", err)
	}
	embQ, err := sessQ.embedID(lastToken)
	if err != nil {
		t.Fatalf("probe A: quant embedID: %v", err)
	}
	a := quantParityFloats(t, embBF)
	b := quantParityFloats(t, embQ)
	logCompare(t, "probe A embedID", a, b)

	// ---- probe B: the boundary seed hidden ----
	seedBF, err := sessBF.boundaryNormedHiddenInto(nil)
	if err != nil {
		t.Fatalf("probe B: bf16 seed: %v", err)
	}
	seedQ, err := sessQ.boundaryNormedHiddenInto(nil)
	if err != nil {
		t.Fatalf("probe B: quant seed: %v", err)
	}
	logCompare(t, "probe B seed hidden", quantParityFloats(t, seedBF), quantParityFloats(t, seedQ))

	// ---- probe C: the per-layer-type target K/V slabs ----
	kvBF, err := pairBF.TargetKVByLayerTypeFromSession(sessBF)
	if err != nil {
		t.Fatalf("probe C: bf16 target KV: %v", err)
	}
	kvQ, err := pairQ.TargetKVByLayerTypeFromSession(sessQ)
	if err != nil {
		t.Fatalf("probe C: quant target KV: %v", err)
	}
	for _, layerType := range []string{"full_attention", "sliding_attention"} {
		sb, okB := kvBF.Get(layerType)
		sq, okQ := kvQ.Get(layerType)
		if !okB || !okQ {
			t.Fatalf("probe C: %s stream missing (bf16 %v quant %v)", layerType, okB, okQ)
		}
		t.Logf("probe C %s: bf16 len=%d off=%d kvh=%d hd=%d | quant len=%d off=%d kvh=%d hd=%d",
			layerType, sb.Length, sb.Offset, sb.KVHeads, sb.HeadDim, sq.Length, sq.Offset, sq.KVHeads, sq.HeadDim)
		logCompare(t, "probe C "+layerType+" K", quantParityFloats(t, sb.Key), quantParityFloats(t, sq.Key))
		logCompare(t, "probe C "+layerType+" V", quantParityFloats(t, sb.Value), quantParityFloats(t, sq.Value))
	}

	// ---- stage D: draft + SELF-verify on each target ----
	for _, side := range []struct {
		name string
		sess *ArchSession
		pair *AssistantPair
	}{{"bf16", sessBF, pairBF}, {"quant", sessQ, pairQ}} {
		block, err := side.pair.DraftBlockFromSession(side.sess, lastToken, draftTokens)
		if err != nil {
			t.Fatalf("stage D: %s draft block: %v", side.name, err)
		}
		vr, err := side.pair.VerifyDraftBlockFromSession(side.sess, block.Tokens)
		if err != nil {
			t.Fatalf("stage D: %s verify: %v", side.name, err)
		}
		t.Logf("stage D %s: drafted=%v targetSays=%v accepted=%d/%d",
			side.name, block.Tokens, vr.TargetTokens, vr.AcceptedCount, len(block.Tokens))
		if side.name == "quant" && vr.AcceptedCount == 0 {
			t.Errorf("stage D FAIL: the quant target accepts NONE of its own drafter's proposals — cross-check the probes above for the diverging input")
		}
	}

	// ---- stage E: scratchless single steps with CROSS-FED inputs ----
	// The session-scratch draft path produced garbage on quant; these steps run the
	// same drafter through the allocation-fresh path, mixing and matching each
	// session's (embed, seed, KV) to pinpoint the poisoned ingredient — or, if all
	// combinations draft sensibly, convict the session-scratch plumbing itself.
	scratchless := func(label string, emb, seed []byte, kvs AssistantTargetKVByType) {
		t.Helper()
		projected, err := pairQ.Assistant.DraftInputProjection(emb, seed)
		if err != nil {
			t.Fatalf("stage E %s: projection: %v", label, err)
		}
		step, err := pairQ.draftStepFromProjectedWithSuppress(projected, kvs, nil)
		if err != nil {
			t.Fatalf("stage E %s: draft step: %v", label, err)
		}
		t.Logf("stage E %s: first draft token = %d", label, step.Token)
	}
	scratchless("quant emb+seed+kv (all quant)", embQ, seedQ, kvQ)
	scratchless("bf16 emb+seed, quant kv", embBF, seedBF, kvQ)
	scratchless("quant emb+seed, bf16 kv", embQ, seedQ, kvBF)
	scratchless("all bf16 (control)", embBF, seedBF, kvBF)
	// the split: which HALF of the quant pair poisons the draft?
	scratchless("quant emb, bf16 seed+kv", embQ, seedBF, kvBF)
	scratchless("bf16 emb, quant seed, bf16 kv", embBF, seedQ, kvBF)
}

// quantParityFloats widens a bf16 byte slab for probing.
func quantParityFloats(t *testing.T, b []byte) []float32 {
	t.Helper()
	if len(b)%2 != 0 {
		t.Fatalf("odd bf16 slab length %d", len(b))
	}
	out := make([]float32, len(b)/2)
	for i := range out {
		bits := uint32(b[2*i]) | uint32(b[2*i+1])<<8
		out[i] = math.Float32frombits(bits << 16)
	}
	return out
}

// logCompare fingerprints two vectors that should agree within quantisation noise:
// rms of each, max abs difference and where. Lengths may legitimately differ only
// if a probe is broken — that IS the finding.
func logCompare(t *testing.T, label string, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Errorf("%s: LENGTH mismatch bf16=%d quant=%d", label, len(a), len(b))
		return
	}
	var maxAbs float64
	maxIdx := -1
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > maxAbs {
			maxAbs, maxIdx = d, i
		}
	}
	rmsA, rmsB := rmsF32(a), rmsF32(b)
	t.Logf("%s: rms bf16=%.4f quant=%.4f maxAbs=%.4f @%d (bf16=%.4f quant=%.4f)",
		label, rmsA, rmsB, maxAbs, maxIdx, a[maxIdx], b[maxIdx])
}

func rmsF32(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(x)))
}
