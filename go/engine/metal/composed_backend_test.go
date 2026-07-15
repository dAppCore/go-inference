// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/mamba2"
	"dappco.re/go/inference/model/qwen3"
	"dappco.re/go/inference/model/rwkv7"
)

const composedPrefillABDefaultDir = "/Users/snider/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-OptiQ-4bit/snapshots/9affd71fc70de2bb08a666ac2d08a3fff5c858e0"

// TestComposedLongPromptPrefillAB is the real-checkpoint receipt: batched prefill with the normal
// composed hook family versus the exact binding state selected by LTHN_OPROJ_FUSE=0. It reports
// prefill throughput for both arms and requires greedy continuation token identity. The checkpoint may
// be overridden for another small composed model with LTHN_COMPOSED_AB_MODEL.
func TestComposedLongPromptPrefillAB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — composed GPU prefill receipt")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — composed GPU prefill receipt: %v", err)
	}
	dir := os.Getenv("LTHN_COMPOSED_AB_MODEL")
	if dir == "" {
		dir = composedPrefillABDefaultDir
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("composed A/B checkpoint absent (%s)", dir)
	}
	tm, err := LoadTokenModelDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(%s): %v", dir, err)
	}

	const promptLen, maxNew = 256, 8
	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32(16 + (i*37)%2048)
	}
	prefill := func(label string) (float64, ComposedHookCounts) {
		sm, ok := tm.(model.SessionModel)
		if !ok {
			t.Fatalf("loaded model is %T, want model.SessionModel", tm)
		}
		sess, err := sm.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession(%s): %v", label, err)
		}
		embs := make([][]byte, len(prompt))
		for i, id := range prompt {
			embs[i], err = tm.Embed(id)
			if err != nil {
				t.Fatalf("Embed(%s,%d): %v", label, id, err)
			}
		}
		bp, ok := sess.(model.BatchPrefillStepper)
		if !ok {
			t.Fatalf("session %T lacks BatchPrefillStepper", sess)
		}
		receipts := EnableComposedHookReceipts()
		start := time.Now()
		if _, err := bp.PrefillBatch(embs); err != nil {
			receipts.Close()
			t.Fatalf("PrefillBatch(%s): %v", label, err)
		}
		elapsed := time.Since(start)
		counts := receipts.Snapshot()
		receipts.Close()
		return float64(len(prompt)) / elapsed.Seconds(), counts
	}

	onTokS, onCounts := prefill("hooks-on")
	onTokens, err := model.Generate(tm, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate(hooks-on): %v", err)
	}

	// This is the runtime equivalent of init under LTHN_OPROJ_FUSE=0: the base projection-tail hook
	// and every extension which depends on it are unbound; lower-level projection/QKV/MLP hooks remain.
	savedTail, savedAttn, savedGD := composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice
	savedMamba, savedRWKV, savedHead := composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice = nil, nil, nil
	composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice = nil, nil, nil
	defer func() {
		composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice = savedTail, savedAttn, savedGD
		composed.ResidualNormMLPProjMamba2InputDevice, composed.ResidualNormMLPProjRWKV7InputDevice, composed.ResidualNormMLPProjHeadDevice = savedMamba, savedRWKV, savedHead
	}()
	offTokS, offCounts := prefill("OPROJ_FUSE=0")
	offTokens, err := model.Generate(tm, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate(OPROJ_FUSE=0): %v", err)
	}
	if !slices.Equal(onTokens, offTokens) {
		t.Fatalf("greedy output tokens differ: hooks-on=%v OPROJ_FUSE=0=%v", onTokens, offTokens)
	}
	// A PACKED checkpoint serves every projection through the quant matvec seam — the f32 fold ladder is
	// bypassed by design (it takes f32 weights; quant fused tails are a later slice), so disabling the fold
	// hooks is a no-op and both arms are identical. The engagement assertion then targets the quant seam;
	// for a dense checkpoint the f32 folds must engage as before.
	if onCounts.QuantProjection.Prefill == 0 {
		if onCounts.AttentionInput.Prefill+onCounts.GatedDeltaFold.Prefill == 0 || onCounts.Head.Prefill == 0 {
			t.Fatalf("hooks-on prefill did not engage composed folds: %+v", onCounts)
		}
	}
	if offCounts.ProjectionTail.Prefill+offCounts.AttentionInput.Prefill+offCounts.GatedDeltaFold.Prefill+offCounts.Head.Prefill != 0 {
		t.Fatalf("OPROJ_FUSE=0 arm engaged a disabled fold: %+v", offCounts)
	}
	t.Logf("composed long-prompt A/B (%d prompt tokens): hooks-on %.2f tok/s counts=%+v; OPROJ_FUSE=0 %.2f tok/s counts=%+v; output tokens identical=%v",
		promptLen, onTokS, onCounts, offTokS, offCounts, onTokens)
}

// TestComposedPrefillBatchDeviceEngagement receipts the L>1 composed path itself, rather than calling
// ComposedSession.Forward directly. L=16, D=128 and FF=512 put the dense tail exactly at the 1<<20 device
// floor. For both mixer families the predecessor's proj+tail+next-input fold and the last layer's
// proj+tail+head fold must engage once. Those folds deliberately supersede the standalone projection and
// MLP hooks, so their zero counters are part of the contract rather than evidence that prefill stayed on
// the host. The same-binary fallback keeps the lower-level device hooks bound and disables only the fold
// family; bf16 output parity therefore isolates the orchestration change.
func TestComposedPrefillBatchDeviceEngagement(t *testing.T) {
	const L, D, FF, vocab = 16, 128, 512, 128

	newMLP := func(seed int) *composed.MLP {
		return &composed.MLP{Gate: cbSyn(FF*D, seed), Up: cbSyn(FF*D, seed+1), Down: cbSyn(D*FF, seed+2), FF: FF}
	}
	attnLayer := func(seed int) composed.Layer {
		const heads, hd = 4, 32
		return composed.Layer{
			InputNorm: cbSyn(D, seed), PostAttnNorm: cbSyn(D, seed+1), MLP: newMLP(seed + 2),
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(D*D, seed+5), KProj: cbSyn(D*D, seed+6), VProj: cbSyn(D*D, seed+7),
				OProj: cbSyn(D*D, seed+8), QNorm: cbSyn(hd, seed+9), KNorm: cbSyn(hd, seed+10),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd, RopeTheta: 1e6, NormEps: 1e-6}),
		}
	}
	gdLayer := func(seed int) composed.Layer {
		cfg := qwen3.GatedDeltaConfig{KeyHeads: 4, ValueHeads: 4, HeadDim: 32, ConvKernel: 4, Eps: 1e-5}
		convDim, vDim := cfg.ConvDim(), cfg.VDim()
		return composed.Layer{
			InputNorm: cbSyn(D, seed), PostAttnNorm: cbSyn(D, seed+1), MLP: newMLP(seed + 2),
			Mixer: composed.NewGatedDeltaMixer(&qwen3.GatedDeltaWeights{
				InProjQKV: cbSyn(convDim*D, seed+5), ConvWeight: cbSyn(convDim*cfg.ConvKernel, seed+6), ConvBias: cbSyn(convDim, seed+7),
				InProjA: cbSyn(cfg.ValueHeads*D, seed+8), ALog: cbSyn(cfg.ValueHeads, seed+9), DtBias: cbSyn(cfg.ValueHeads, seed+10),
				InProjB: cbSyn(cfg.ValueHeads*D, seed+11), InProjZ: cbSyn(vDim*D, seed+12), Norm: cbSyn(cfg.HeadDim, seed+13), OutProj: cbSyn(D*vDim, seed+14),
			}, cfg),
		}
	}

	for _, tc := range []struct {
		name       string
		layers     []composed.Layer
		wantAttnIn uint64
		wantGDIn   uint64
	}{
		{name: "attention", layers: []composed.Layer{attnLayer(20), attnLayer(40)}, wantAttnIn: 1},
		{name: "gated_delta", layers: []composed.Layer{attnLayer(60), gdLayer(80)}, wantGDIn: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &composed.ComposedModel{Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6, Layers: tc.layers}
			embs := make([][]byte, L)
			tm := composed.NewTokenModel(m)
			for i := range embs {
				var err error
				embs[i], err = tm.Embed(int32(i % vocab))
				if err != nil {
					t.Fatalf("Embed(%d): %v", i, err)
				}
			}

			projCalls, mlpCalls, projTailCalls, inputCalls, headCalls := 0, 0, 0, 0, 0
			savedProj, savedMLP := composed.ProjMatMulInto, composed.MLPDevice
			savedTail, savedAttn, savedGD, savedHead := composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjHeadDevice
			defer func() {
				composed.ProjMatMulInto, composed.MLPDevice = savedProj, savedMLP
				composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjHeadDevice = savedTail, savedAttn, savedGD, savedHead
			}()
			composed.ProjMatMulInto = func(out, x, w []float32, M, K, N int) ([]float32, error) {
				projCalls++
				return MatMulF32NTInto(out, x, w, M, K, N)
			}
			composed.MLPDevice = func(g, u, d, x []float32, L, D, FF int) ([]float32, error) {
				mlpCalls++
				return ComposedMLPDevice(g, u, d, x, L, D, FF)
			}
			composed.ResidualNormMLPProjDevice = func(mh, pw, h, nw, g, u, d []float32, L, D, mc, FF int, eps float32) ([]float32, error) {
				projTailCalls++
				y, err := ResidualNormMLPProjDevice(mh, pw, h, nw, g, u, d, L, D, mc, FF, eps)
				if err != nil {
					t.Logf("plain proj tail: %v", err)
				}
				return y, err
			}
			composed.ResidualNormMLPProjAttnInputDevice = func(mh, pw, h, nw, g, u, d []float32, L, D, mc, FF int, eps float32, nn, qw, kw, vw []float32, qc, kvc int) ([]float32, []float32, []float32, []float32, error) {
				inputCalls++
				y, q, k, v, err := ResidualNormMLPProjAttnInputDevice(mh, pw, h, nw, g, u, d, L, D, mc, FF, eps, nn, qw, kw, vw, qc, kvc)
				if err != nil {
					t.Logf("attention input fold: %v", err)
				}
				return y, q, k, v, err
			}
			composed.ResidualNormMLPProjGatedDeltaInputDevice = func(mh, pw, h, nw, g, u, d []float32, L, D, mc, FF int, eps float32, nn, qkvw, zw, aw, bw []float32, cd, vd, vh int) ([]float32, []float32, []float32, []float32, []float32, error) {
				inputCalls++
				y, qkv, z, a, b, err := ResidualNormMLPProjGatedDeltaInputDevice(mh, pw, h, nw, g, u, d, L, D, mc, FF, eps, nn, qkvw, zw, aw, bw, cd, vd, vh)
				if err != nil {
					t.Logf("gated-delta input fold: %v", err)
				}
				return y, qkv, z, a, b, err
			}
			composed.ResidualNormMLPProjHeadDevice = func(mh, pw, h, nw, g, u, d []float32, L, D, mc, FF int, eps float32, nf, hw []float32, v int) ([]float32, []float32, error) {
				headCalls++
				y, logits, err := ResidualNormMLPProjHeadDevice(mh, pw, h, nw, g, u, d, L, D, mc, FF, eps, nf, hw, v)
				if err != nil {
					t.Logf("head fold: %v", err)
				}
				return y, logits, err
			}
			receipts := EnableComposedHookReceipts()

			stepper, err := tm.OpenSession()
			if err != nil {
				t.Fatalf("OpenSession: %v", err)
			}
			bp := stepper.(interface {
				PrefillBatch([][]byte) ([]byte, error)
			})
			got, err := bp.PrefillBatch(embs)
			if err != nil {
				t.Fatalf("PrefillBatch device: %v", err)
			}
			if inputCalls != 1 || headCalls != 1 {
				t.Fatalf("fold hook calls: input=%d head=%d, want 1,1 (fallback plain-proj-tail=%d projection=%d mlp=%d)", inputCalls, headCalls, projTailCalls, projCalls, mlpCalls)
			}
			if projTailCalls == 0 && (projCalls != 0 || mlpCalls != 0) {
				t.Fatalf("successful folds reached superseded hooks: projection=%d mlp=%d, want 0,0", projCalls, mlpCalls)
			}
			prefill := receipts.Snapshot()
			if prefill.AttentionInput.Prefill != tc.wantAttnIn || prefill.GatedDeltaFold.Prefill != tc.wantGDIn || prefill.Head.Prefill != 1 {
				t.Fatalf("prefill hook receipt = attn-input:%d gated-delta-input:%d head:%d, want %d,%d,1; full=%+v",
					prefill.AttentionInput.Prefill, prefill.GatedDeltaFold.Prefill, prefill.Head.Prefill, tc.wantAttnIn, tc.wantGDIn, prefill)
			}
			if prefill.AttentionInput.Decode != 0 || prefill.GatedDeltaFold.Decode != 0 || prefill.Head.Decode != 0 {
				t.Fatalf("prefill contaminated decode counters: %+v", prefill)
			}

			// The same session's next single-token step is deliberately below the
			// 1<<20 device floor (1*128*512), so it must leave the decode buckets at
			// zero. Wider decode parity tests cover the engaged L=1 path.
			if _, err := stepper.Step(embs[0]); err != nil {
				t.Fatalf("decode Step after prefill: %v", err)
			}
			decode := receipts.Snapshot()
			if decode.AttentionInput.Decode != 0 || decode.GatedDeltaFold.Decode != 0 || decode.Head.Decode != 0 {
				t.Fatalf("sub-floor decode unexpectedly engaged hooks: %+v", decode)
			}
			t.Logf("hook receipt prefill=%+v decode=%+v", prefill, decode)
			receipts.Close()

			composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjHeadDevice = nil, nil, nil
			stepper, err = tm.OpenSession()
			if err != nil {
				t.Fatalf("OpenSession fallback: %v", err)
			}
			want, err := stepper.(interface {
				PrefillBatch([][]byte) ([]byte, error)
			}).PrefillBatch(embs)
			if err != nil {
				t.Fatalf("PrefillBatch fallback: %v", err)
			}
			if len(got) != len(want) {
				t.Fatalf("bf16 last hidden length: folded=%d fallback=%d", len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bf16 last hidden byte %d: folded=%d fallback=%d", i, got[i], want[i])
				}
			}
		})
	}
}

func cbSyn(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*seed+11)%29-14) * 0.03
	}
	return v
}

// TestComposedMixerInputFusesDeviceVsFallback batches the Mamba-2 and RWKV-7 next-layer cases. Each
// counter proves forwardEmb selected the new pending-resume fuse; disabling only that hook supplies the
// same-device fallback reference, isolating the command-buffer fold from unrelated device/host drift.
func TestComposedMixerInputFusesDeviceVsFallback(t *testing.T) {
	const D, FF, heads, hd = 512, 2048, 4, 128
	first := composed.Layer{
		InputNorm: cbSyn(D, 10), PostAttnNorm: cbSyn(D, 11),
		MLP: &composed.MLP{Gate: cbSyn(FF*D, 12), Up: cbSyn(FF*D, 13), Down: cbSyn(D*FF, 14), FF: FF},
		Mixer: composed.NewAttnMixer(&composed.AttnWeights{
			QProj: cbSyn(heads*hd*D, 15), KProj: cbSyn(heads*hd*D, 16), VProj: cbSyn(heads*hd*D, 17),
			OProj: cbSyn(D*heads*hd, 18), QNorm: cbSyn(hd, 19), KNorm: cbSyn(hd, 20),
		}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
	}
	t.Run("mamba2", func(t *testing.T) {
		if composed.ResidualNormMLPProjMamba2InputDevice == nil {
			t.Fatal("native init did not wire Mamba-2 input fuse")
		}
		cfg := mamba2.BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
		dInner := cfg.NumHeads * cfg.HeadDim
		convDim := dInner + 2*cfg.NumGroups*cfg.StateDim
		projDim := 2*dInner + 2*cfg.NumGroups*cfg.StateDim + cfg.NumHeads
		w := &mamba2.BlockWeights{InProj: cbSyn(projDim*D, 30), ConvWeight: cbSyn(convDim*cfg.ConvKernel, 31), ConvBias: cbSyn(convDim, 32), ALog: cbSyn(cfg.NumHeads, 33), D: cbSyn(cfg.NumHeads, 34), DtBias: cbSyn(cfg.NumHeads, 35), Norm: cbSyn(dInner, 36), OutProj: cbSyn(D*dInner, 37)}
		second := composed.Layer{InputNorm: cbSyn(D, 38), PostAttnNorm: cbSyn(D, 39), MLP: &composed.MLP{Gate: cbSyn(FF*D, 40), Up: cbSyn(FF*D, 41), Down: cbSyn(D*FF, 42), FF: FF}, Mixer: composed.NewMamba2Mixer(w, cfg)}
		m := &composed.ComposedModel{Embed: cbSyn(64*D, 43), D: D, Vocab: 64, Eps: 1e-6, Layers: []composed.Layer{first, second}}
		calls := 0
		saved := composed.ResidualNormMLPProjMamba2InputDevice
		composed.ResidualNormMLPProjMamba2InputDevice = func(mh, pw, h, nw, g, u, d []float32, L, D, mc, FF int, eps float32, nn, iw []float32, pd int) ([]float32, []float32, error) {
			calls++
			return ResidualNormMLPProjMamba2InputDevice(mh, pw, h, nw, g, u, d, L, D, mc, FF, eps, nn, iw, pd)
		}
		got, err := composed.NewSession(m).Forward([]int32{1, 2, 3})
		composed.ResidualNormMLPProjMamba2InputDevice = nil
		want, werr := composed.NewSession(m).Forward([]int32{1, 2, 3})
		composed.ResidualNormMLPProjMamba2InputDevice = saved
		checkMixerInputFuse(t, calls, got, want, err, werr)
	})
	t.Run("rwkv7", func(t *testing.T) {
		if composed.ResidualNormMLPProjRWKV7InputDevice == nil {
			t.Fatal("native init did not wire RWKV-7 input fuse")
		}
		cfg := rwkv7.BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
		hk, hv := cfg.NumHeads*cfg.KeyDim, cfg.NumHeads*cfg.ValueDim
		w := &rwkv7.BlockWeights{RProj: cbSyn(hk*D, 50), WProj: cbSyn(hk*D, 51), KProj: cbSyn(hk*D, 52), VProj: cbSyn(hv*D, 53), AProj: cbSyn(hk*D, 54), BProj: cbSyn(hk*D, 55), OutProj: cbSyn(D*hv, 56)}
		second := composed.Layer{InputNorm: cbSyn(D, 57), PostAttnNorm: cbSyn(D, 58), MLP: &composed.MLP{Gate: cbSyn(FF*D, 59), Up: cbSyn(FF*D, 60), Down: cbSyn(D*FF, 61), FF: FF}, Mixer: composed.NewRWKV7Mixer(w, cfg)}
		m := &composed.ComposedModel{Embed: cbSyn(64*D, 62), D: D, Vocab: 64, Eps: 1e-6, Layers: []composed.Layer{first, second}}
		calls := 0
		saved := composed.ResidualNormMLPProjRWKV7InputDevice
		composed.ResidualNormMLPProjRWKV7InputDevice = func(mh, pw, h, nw, g, u, d []float32, L, D, mc, FF int, eps float32, nn, rw, ww, kw, vw, aw, bw []float32, hk, hv int) ([]float32, []float32, []float32, []float32, []float32, []float32, []float32, error) {
			calls++
			return ResidualNormMLPProjRWKV7InputDevice(mh, pw, h, nw, g, u, d, L, D, mc, FF, eps, nn, rw, ww, kw, vw, aw, bw, hk, hv)
		}
		got, err := composed.NewSession(m).Forward([]int32{1, 2, 3})
		composed.ResidualNormMLPProjRWKV7InputDevice = nil
		want, werr := composed.NewSession(m).Forward([]int32{1, 2, 3})
		composed.ResidualNormMLPProjRWKV7InputDevice = saved
		checkMixerInputFuse(t, calls, got, want, err, werr)
	})
}

func checkMixerInputFuse(t *testing.T, calls int, got, want []float32, err, wantErr error) {
	t.Helper()
	if err != nil || wantErr != nil {
		t.Fatalf("forward errors: fused=%v fallback=%v", err, wantErr)
	}
	if calls == 0 {
		t.Fatal("input-fuse hook never engaged")
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-2*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("hidden[%d]: fused %v fallback %v", i, got[i], want[i])
		}
	}
}

// TestComposedDeviceVsHost runs a one-layer composed forward (attention mixer + MLP) with native's
// device-GEMM hook (init-wired) and confirms the logits match a host run (hook nil'd) within f32
// tolerance — the device path is the projection swap only. D/FF sit above composed.deviceMinWork so
// the hook genuinely engages on the MLP and head matmuls.
func TestComposedDeviceVsHost(t *testing.T) {
	if composed.ProjMatMulInto == nil {
		t.Fatal("native init did not wire composed.ProjMatMulInto")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	dev, err := composed.NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("device forward: %v", err)
	}
	saved, savedMLP := composed.ProjMatMulInto, composed.MLPDevice
	composed.ProjMatMulInto, composed.MLPDevice = nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ProjMatMulInto, composed.MLPDevice = saved, savedMLP
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (device GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed forward: device-GEMM projections match host within f32 tol over %d logits", len(dev))
}

// TestComposedAttnQKVFuseDeviceVsHost exercises the fused attention projection (q/k/v in ONE command
// buffer) at a shape whose q projection crosses composed.deviceMinWork, so AttnQKVDevice genuinely
// engages (k/v are sub-floor free riders), and confirms the logits match a pure-host run within f32
// tolerance. A call-counter around the hook asserts the fuse actually fired. heads·hd = 8·128 = 1024
// with 3 prefill tokens ⇒ q's L·D·qCols = 3·512·1024 = 1,572,864 ≥ 1<<20 opens the fuse gate; the
// KV heads (2·128 = 256) stay sub-floor, riding the fused CB for free.
func TestComposedAttnQKVFuseDeviceVsHost(t *testing.T) {
	if composed.AttnQKVDevice == nil {
		t.Fatal("native init did not wire composed.AttnQKVDevice")
	}
	const D, FF, vocab, heads, kvheads, hd = 512, 2048, 4096, 8, 2, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(kvheads*hd*D, 9),
				VProj: cbSyn(kvheads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: kvheads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	savedFuse := composed.AttnQKVDevice
	composed.AttnQKVDevice = func(h, qW, kW, vW []float32, L, D, qCols, kvCols int) ([]float32, []float32, []float32, error) {
		calls++
		return ComposedAttnQKVDevice(h, qW, kW, vW, L, D, qCols, kvCols)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.AttnQKVDevice = savedFuse
	if err != nil {
		t.Fatalf("device (fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("fused q/k/v hook never engaged — shape below device floor?")
	}

	savedFuse2, savedProj, savedMLP := composed.AttnQKVDevice, composed.ProjMatMulInto, composed.MLPDevice
	composed.AttnQKVDevice, composed.ProjMatMulInto, composed.MLPDevice = nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.AttnQKVDevice, composed.ProjMatMulInto, composed.MLPDevice = savedFuse2, savedProj, savedMLP
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (fused q/k/v GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed attn q/k/v fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedResidualNormMLPFuseDeviceVsHost exercises the fused FFN-tail primitive (mixer residual add +
// post-attn RMSNorm + SwiGLU MLP + MLP residual add, all in ONE command buffer) at a shape whose MLP
// crosses composed.deviceMinWork so the tail hook genuinely engages, and confirms the logits match a
// pure-host run within f32 tolerance. A call-counter around the hook asserts the fuse actually fired.
// L·D·FF = 3·512·2048 = 3,145,728 ≥ 1<<20 opens the fuse gate.
func TestComposedResidualNormMLPFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	// The proj-fused tail (ResidualNormMLPProjDevice) supersedes the plain tail for a projMixer when both are
	// wired; disable it here so this test exercises the plain ResidualNormMLPDevice path it targets.
	savedProjTail := composed.ResidualNormMLPProjDevice
	composed.ResidualNormMLPProjDevice = nil
	defer func() { composed.ResidualNormMLPProjDevice = savedProjTail }()

	calls := 0
	savedTail := composed.ResidualNormMLPDevice
	composed.ResidualNormMLPDevice = func(h, mixOut, normW, gate, up, down []float32, L, D, FF int, eps float32) ([]float32, error) {
		calls++
		return ResidualNormMLPDevice(h, mixOut, normW, gate, up, down, L, D, FF, eps)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPDevice = savedTail
	if err != nil {
		t.Fatalf("device (fused tail) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("fused FFN-tail hook never engaged — shape below device floor?")
	}

	savedTail2, savedProj, savedMLP, savedFuse := composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = savedTail2, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (fused FFN-tail diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed FFN-tail fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedResidualNormMLPProjFuseDeviceVsHost exercises the projection-fused FFN-tail primitive: the
// attention mixer's o_proj folded onto the front of the tail CB (o_proj + mixer residual + post-attn RMSNorm
// + SwiGLU MLP + MLP residual, all in ONE command buffer), and confirms the logits match a pure-host run
// within f32 tolerance. A call-counter around the hook asserts the proj-fused path actually fired — the tail
// gate L·D·FF = 3·512·2048 = 3,145,728 ≥ 1<<20 opens it; the o_proj (mixCols = heads·hd = 512) rides free.
func TestComposedResidualNormMLPProjFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	savedProjTail := composed.ResidualNormMLPProjDevice
	composed.ResidualNormMLPProjDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error) {
		calls++
		return ResidualNormMLPProjDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps)
	}
	// A 1-layer model's only layer is also the LAST layer, so the head-fuse (which also folds onto the
	// back of this same tail, taking priority when it applies) would otherwise intercept before this
	// wrapped hook ever fires — nil it out to isolate the plain proj-fused tail this test targets.
	savedHeadFuse := composed.ResidualNormMLPProjHeadDevice
	composed.ResidualNormMLPProjHeadDevice = nil
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice = savedProjTail
	composed.ResidualNormMLPProjHeadDevice = savedHeadFuse
	if err != nil {
		t.Fatalf("device (proj-fused tail) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("proj-fused FFN-tail hook never engaged — shape below device floor?")
	}

	savedProjTail2, savedTail, savedProj, savedMLP, savedFuse := composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = savedProjTail2, savedTail, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (proj-fused FFN-tail diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed o_proj+FFN-tail fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedGatedDeltaProjFuseDeviceVsHost exercises the projection-fused FFN-tail for a GATED-DELTA mixer
// layer: out_proj folded onto the front of the tail CB (out_proj + mixer residual + post-attn RMSNorm +
// SwiGLU MLP + MLP residual, all in ONE command buffer), and confirms the logits match a pure-host run
// within f32 tolerance. A call-counter around the hook asserts the proj-fused path fired. The tail gate
// L·D·FF = 3·768·2048 = 4,718,592 ≥ 1<<20 opens it; the out_proj (mixCols = vDim = 512) rides free.
func TestComposedGatedDeltaProjFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjDevice")
	}
	const D, FF, vocab = 768, 2048, 4096
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 8, ValueHeads: 8, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	qDim, vDim := cfg.KeyHeads*cfg.HeadDim, cfg.ValueHeads*cfg.HeadDim
	convDim := 2*qDim + vDim
	gdw := &qwen3.GatedDeltaWeights{
		InProjQKV:  cbSyn(convDim*D, 11),
		ConvWeight: cbSyn(convDim*cfg.ConvKernel, 12),
		ConvBias:   cbSyn(convDim, 13),
		InProjA:    cbSyn(cfg.ValueHeads*D, 14),
		ALog:       cbSyn(cfg.ValueHeads, 15),
		DtBias:     cbSyn(cfg.ValueHeads, 16),
		InProjB:    cbSyn(cfg.ValueHeads*D, 17),
		InProjZ:    cbSyn(vDim*D, 18),
		Norm:       cbSyn(cfg.HeadDim, 19),
		OutProj:    cbSyn(D*vDim, 20),
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer:        composed.NewGatedDeltaMixer(gdw, cfg),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	savedProjTail := composed.ResidualNormMLPProjDevice
	composed.ResidualNormMLPProjDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error) {
		calls++
		return ResidualNormMLPProjDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps)
	}
	// A 1-layer model's only layer is also the LAST layer, so the head-fuse (which also folds onto the
	// back of this same tail, taking priority when it applies) would otherwise intercept before this
	// wrapped hook ever fires — nil it out to isolate the plain proj-fused tail this test targets.
	savedHeadFuse := composed.ResidualNormMLPProjHeadDevice
	composed.ResidualNormMLPProjHeadDevice = nil
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice = savedProjTail
	composed.ResidualNormMLPProjHeadDevice = savedHeadFuse
	if err != nil {
		t.Fatalf("device (proj-fused tail) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("gated-delta proj-fused FFN-tail hook never engaged — shape below device floor?")
	}

	savedProjTail2, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedInput := composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, qwen3.GatedDeltaInputDevice
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, qwen3.GatedDeltaInputDevice = nil, nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, qwen3.GatedDeltaInputDevice = savedProjTail2, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedInput
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (gated-delta proj-fused FFN-tail diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed gated-delta out_proj+FFN-tail fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedResidualNormMLPProjAttnInputFuseDeviceVsHost exercises the input-side mirror of the o_proj
// fuse: layer 0's proj-fused tail command buffer additionally folds in layer 1's (a full-attention mixer)
// input RMSNorm + q/k/v projections — the symmetric collapse to TestComposedResidualNormMLPProjFuseDeviceVsHost.
// A call-counter around the hook asserts the input-fuse actually fired; the full 2-layer forward's hiddens
// must still match a pure-host run within f32 tolerance. Both layers share D=512, FF=2048 (L·D·FF =
// 3,145,728 ≥ 1<<20 opens layer 0's proj-fused tail; the input-fuse rides free whatever layer 1's own q/k/v
// shape).
func TestComposedResidualNormMLPProjAttnInputFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjAttnInputDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjAttnInputDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	newAttnLayer := func(seed int) composed.Layer {
		return composed.Layer{
			InputNorm:    cbSyn(D, seed),
			PostAttnNorm: cbSyn(D, seed+1),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, seed+2), Up: cbSyn(FF*D, seed+3), Down: cbSyn(D*FF, seed+4), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, seed+5), KProj: cbSyn(heads*hd*D, seed+6),
				VProj: cbSyn(heads*hd*D, seed+7), OProj: cbSyn(D*heads*hd, seed+8),
				QNorm: cbSyn(hd, seed+9), KNorm: cbSyn(hd, seed+10),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{newAttnLayer(20), newAttnLayer(40)},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	saved := composed.ResidualNormMLPProjAttnInputDevice
	composed.ResidualNormMLPProjAttnInputDevice = func(
		mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
		nextNormW, nextQW, nextKW, nextVW []float32, nextQCols, nextKVCols int,
	) ([]float32, []float32, []float32, []float32, error) {
		calls++
		return ResidualNormMLPProjAttnInputDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, nextNormW, nextQW, nextKW, nextVW, nextQCols, nextKVCols)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjAttnInputDevice = saved
	if err != nil {
		t.Fatalf("device (input-fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("attn input-fuse hook never engaged — layer 1 not seen as a full-attention next mixer?")
	}

	savedInput, savedProjTail, savedTail, savedProj, savedMLP, savedFuse :=
		composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice =
		nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice =
		savedInput, savedProjTail, savedTail, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("hidden[%d]: device %v host %v (attn input-fuse diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed proj-tail+attn-input fuse: %d fused-CB call(s); device matches host within f32 tol over %d values", calls, len(dev))
}

// TestComposedResidualNormMLPProjGatedDeltaInputFuseDeviceVsHost exercises the input-side mirror of the
// o_proj fuse for a GATED-DELTA next layer: layer 0's (full-attention) proj-fused tail command buffer
// additionally folds in layer 1's (a gated-delta mixer) input RMSNorm + in_proj_qkv/z/a/b. A call-counter
// around the hook asserts the input-fuse actually fired; the full 2-layer forward's hiddens must still
// match a pure-host run within f32 tolerance.
func TestComposedResidualNormMLPProjGatedDeltaInputFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjGatedDeltaInputDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjGatedDeltaInputDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	attnLayer := composed.Layer{
		InputNorm:    cbSyn(D, 20),
		PostAttnNorm: cbSyn(D, 21),
		MLP:          &composed.MLP{Gate: cbSyn(FF*D, 22), Up: cbSyn(FF*D, 23), Down: cbSyn(D*FF, 24), FF: FF},
		Mixer: composed.NewAttnMixer(&composed.AttnWeights{
			QProj: cbSyn(heads*hd*D, 25), KProj: cbSyn(heads*hd*D, 26),
			VProj: cbSyn(heads*hd*D, 27), OProj: cbSyn(D*heads*hd, 28),
			QNorm: cbSyn(hd, 29), KNorm: cbSyn(hd, 30),
		}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
	}
	gdCfg := qwen3.GatedDeltaConfig{KeyHeads: 8, ValueHeads: 8, HeadDim: 32, ConvKernel: 4, Eps: 1e-5}
	qDim, vDim := gdCfg.KeyHeads*gdCfg.HeadDim, gdCfg.ValueHeads*gdCfg.HeadDim
	convDim := 2*qDim + vDim
	gdw := &qwen3.GatedDeltaWeights{
		InProjQKV:  cbSyn(convDim*D, 40),
		ConvWeight: cbSyn(convDim*gdCfg.ConvKernel, 41),
		ConvBias:   cbSyn(convDim, 42),
		InProjA:    cbSyn(gdCfg.ValueHeads*D, 43),
		ALog:       cbSyn(gdCfg.ValueHeads, 44),
		DtBias:     cbSyn(gdCfg.ValueHeads, 45),
		InProjB:    cbSyn(gdCfg.ValueHeads*D, 46),
		InProjZ:    cbSyn(vDim*D, 47),
		Norm:       cbSyn(gdCfg.HeadDim, 48),
		OutProj:    cbSyn(D*vDim, 49),
	}
	gdLayer := composed.Layer{
		InputNorm:    cbSyn(D, 50),
		PostAttnNorm: cbSyn(D, 51),
		MLP:          &composed.MLP{Gate: cbSyn(FF*D, 52), Up: cbSyn(FF*D, 53), Down: cbSyn(D*FF, 54), FF: FF},
		Mixer:        composed.NewGatedDeltaMixer(gdw, gdCfg),
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{attnLayer, gdLayer},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	saved := composed.ResidualNormMLPProjGatedDeltaInputDevice
	composed.ResidualNormMLPProjGatedDeltaInputDevice = func(
		mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
		nextNormW, nextQKVW, nextZW, nextAW, nextBW []float32, nextConvDim, nextVDim, nextVH int,
	) ([]float32, []float32, []float32, []float32, []float32, error) {
		calls++
		return ResidualNormMLPProjGatedDeltaInputDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, nextNormW, nextQKVW, nextZW, nextAW, nextBW, nextConvDim, nextVDim, nextVH)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjGatedDeltaInputDevice = saved
	if err != nil {
		t.Fatalf("device (input-fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("gated-delta input-fuse hook never engaged — layer 1 not seen as a gated-delta next mixer?")
	}

	savedInput, savedProjTail, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedFuse, savedGDIn :=
		composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice
	composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice =
		nil, nil, nil, nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice =
		savedInput, savedProjTail, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedFuse, savedGDIn
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("hidden[%d]: device %v host %v (gated-delta input-fuse diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed proj-tail+gated-delta-input fuse: %d fused-CB call(s); device matches host within f32 tol over %d values", calls, len(dev))
}

// TestComposedResidualNormMLPProjHeadFuseDeviceVsHost exercises the terminal (OUTPUT-side) fuse: the LAST
// layer's proj-fused tail command buffer additionally folds in the model's own final RMSNorm + LM head
// GEMM — the mirror of TestComposedResidualNormMLPProjAttnInputFuseDeviceVsHost at the OTHER end of the
// stack (that one folds the NEXT layer's input on; here there is no next layer, so the model's own head
// goes on instead). A call-counter around the hook asserts the head-fuse actually fired; the fused logits
// must match composed.HeadLogitsHost's pure-host reference within f32 tolerance. L·D·FF = 3·512·2048 =
// 3,145,728 ≥ 1<<20 opens the tail's fuse gate; the head GEMM (1·D·Vocab) rides on top regardless of its
// own size.
func TestComposedResidualNormMLPProjHeadFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjHeadDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjHeadDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	saved := composed.ResidualNormMLPProjHeadDevice
	composed.ResidualNormMLPProjHeadDevice = func(
		mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
		normF, head []float32, Vocab int,
	) ([]float32, []float32, error) {
		calls++
		return ResidualNormMLPProjHeadDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, normF, head, Vocab)
	}
	sess := composed.NewSession(m)
	dev, err := sess.Forward(tokens)
	devLogits := sess.PendingHeadLogits()
	composed.ResidualNormMLPProjHeadDevice = saved
	if err != nil {
		t.Fatalf("device (head-fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("head-fuse hook never engaged — last layer not seen as terminal?")
	}
	if devLogits == nil {
		t.Fatal("head-fuse fired but PendingHeadLogits is nil")
	}
	if len(devLogits) != vocab {
		t.Fatalf("logits length: got %d want %d", len(devLogits), vocab)
	}

	// Host reference: every device hook nil'd, then the model's own headLogits over the LAST row — the
	// exact computation the fused path replaces.
	savedHead, savedProjTail, savedTail, savedProj, savedMLP, savedFuse :=
		composed.ResidualNormMLPProjHeadDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPProjHeadDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice =
		nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjHeadDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice =
		savedHead, savedProjTail, savedTail, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("hidden[%d]: device %v host %v (head-fused tail diverged)", i, dev[i], host[i])
		}
	}
	hostLogits := composed.HeadLogitsHost(m, host[(len(tokens)-1)*D:])
	if len(hostLogits) != vocab {
		t.Fatalf("host logits length: got %d want %d", len(hostLogits), vocab)
	}
	for i := range devLogits {
		if math.Abs(float64(devLogits[i]-hostLogits[i])) > 1e-2*(1+math.Abs(float64(hostLogits[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (head-fuse GEMM/RMSNorm diverged)", i, devLogits[i], hostLogits[i])
		}
	}
	t.Logf("composed proj-tail+head fuse: %d fused-CB call(s); device matches host within f32 tol over %d hiddens + %d logits", calls, len(dev), len(devLogits))
}
