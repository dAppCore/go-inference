// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"

	"dappco.re/go/inference/internal/enginegate"
)

// TestAssistantPairTargetKVByLayerTypeFromSessionRepeatExtract guards the quant-lane
// MTP root cause: stateLayerViews() on an ICB (quant) session must NOT re-materialise
// the drafter-facing K/V views from the session's unused, empty paged cache on repeat
// extraction — doing so zeroed the target Key and collapsed speculative acceptance to
// 0%. Back-to-back extractions with no forward in between must return the same live,
// non-zero K/V on both the ICB (quant) and paged (bf16) session shapes.
func TestAssistantPairTargetKVByLayerTypeFromSessionRepeatExtract(t *testing.T) {
	for _, tc := range []struct{ name, dir string }{
		{"quant", "mlx-community/gemma-4-e2b-it-4bit"},
		{"bf16", "mlx-community/gemma-4-E2B-it-bf16"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := enginegate.HFModelPath(t, tc.dir)
			drafterDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
			prompt := "<|turn>user\nName the planets of the solar system in order.<turn|>\n<|turn>model\n"
			sess, err := LoadDir(dir, 640)
			if err != nil {
				t.Fatalf("LoadDir: %v", err)
			}
			t.Cleanup(func() { sess.Close() })
			pair, err := LoadAssistantPairDirs(dir, drafterDir)
			if err != nil {
				t.Fatalf("pair: %v", err)
			}
			t.Cleanup(func() { pair.Close() })
			ids := pair.Assistant.Tok.Encode(prompt)
			if err := sess.prepareAssistantPrompt(ids); err != nil {
				t.Fatalf("prepare: %v", err)
			}
			rms := func(b []byte) float64 { return rmsF32(quantParityFloats(t, b)) }
			kvK := func() float64 {
				kv, err := pair.TargetKVByLayerTypeFromSession(sess)
				if err != nil {
					t.Fatalf("kv: %v", err)
				}
				fa, _ := kv.Get("full_attention")
				return rms(fa.Key)
			}
			icb := sess.state.icb != nil
			paged := sess.state.hasDevicePagedKV()
			t.Logf("[%s] icb=%v pagedKV=%v", tc.name, icb, paged)
			e1, e2, e3 := kvK(), kvK(), kvK()
			t.Logf("[%s] extract#1=%.4f extract#2=%.4f extract#3=%.4f (back-to-back, no forward)",
				tc.name, e1, e2, e3)
			if e1 == 0 {
				t.Fatalf("[%s] first extraction returned an all-zero Key — prefill never reached the drafter-facing views", tc.name)
			}
			if e2 != e1 || e3 != e1 {
				t.Errorf("[%s] repeat extraction changed the Key rms (%.6f, %.6f, %.6f) — a stale-snapshot refresh is overwriting live K/V", tc.name, e1, e2, e3)
			}
		})
	}
}
