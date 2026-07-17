// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
)

// TestComposedChainICBReplayParity is the CB-recording acceptance (#18 queue item 4) on the
// real chainable checkpoint: a greedy decode whose tokens 2..N ride the RECORDED ICB replay
// must produce exactly the token stream the re-encode chain produces, and the replay must
// actually have engaged (a recording that silently never replays would pass parity for the
// wrong reason). Skips without the metallib or the cached checkpoint.
func TestComposedChainICBReplayParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable: %v", err)
	}
	dir := os.Getenv("LTHN_COMPOSED_AB_MODEL")
	if dir == "" {
		dir = composedPrefillABDefaultDir
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("composed chain checkpoint absent (%s)", dir)
	}
	if composed.ComposedChainRecordBegin == nil || composed.ComposedChainReplayDevice == nil {
		t.Fatal("native init did not wire the chain recording hooks")
	}

	// maxNew crosses the KV cache's first capacity boundary (ensureCap floor 128), so the
	// recording is invalidated mid-decode, the token re-encodes live, and a fresh recording
	// takes over — the model-side position lockstep after replays is exactly what that
	// transition exercises (the desync class the serve sequence hit).
	prompt := []int32{16, 53, 90, 127, 4, 991}
	const maxNew = 140

	decode := func() []int32 {
		tm, _, err := model.LoadComposedDir(dir)
		if err != nil {
			t.Fatalf("LoadComposedDir: %v", err)
		}
		sm := tm.(model.SessionModel)
		gen, err := model.GenerateSampledWithStopTokensTransformEach(sm, model.NewSampler(0), model.SampleParams{}, prompt, maxNew, nil, nil, nil)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		return gen
	}

	// Arm A: recording live, with an engagement counter around the replay hook.
	replays := 0
	savedReplay := composed.ComposedChainReplayDevice
	composed.ComposedChainReplayDevice = func(rec any, h []float32) ([]float32, []float32, bool, error) {
		y, logits, ok, err := savedReplay(rec, h)
		if ok && err == nil {
			replays++
		}
		return y, logits, ok, err
	}
	withICB := decode()
	composed.ComposedChainReplayDevice = savedReplay

	// Arm B: recording hooks off — every token re-encodes (the pre-ICB chain).
	savedBegin, savedEnd, savedRelease := composed.ComposedChainRecordBegin, composed.ComposedChainRecordEnd, composed.ComposedChainRecordingRelease
	composed.ComposedChainRecordBegin, composed.ComposedChainRecordEnd, composed.ComposedChainRecordingRelease = nil, nil, nil
	reencode := decode()
	composed.ComposedChainRecordBegin, composed.ComposedChainRecordEnd, composed.ComposedChainRecordingRelease = savedBegin, savedEnd, savedRelease

	if len(withICB) == 0 || len(withICB) != len(reencode) {
		t.Fatalf("token counts differ: icb %d vs re-encode %d", len(withICB), len(reencode))
	}
	for i := range withICB {
		if withICB[i] != reencode[i] {
			t.Fatalf("token %d: icb %d vs re-encode %d (streams: %v vs %v)", i, withICB[i], reencode[i], withICB, reencode)
		}
	}
	if replays < maxNew-8 { // records twice (start + post-invalidation) — everything else replays
		t.Fatalf("replay engaged %d times, want ≥ %d — the recording is not serving the decode", replays, maxNew-8)
	}
	t.Logf("ICB replay parity: %d tokens identical, %d replays engaged", len(withICB), replays)
}
