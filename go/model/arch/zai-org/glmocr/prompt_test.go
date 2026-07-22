// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import "testing"

// BuildPrompt's token-level correctness (the exact special-token/image-span layout) needs the
// REAL tokenizer's BPE vocabulary to verify byte-for-byte — see live_test.go's
// TestLive_RealCheckpoint_BuildPrompt_Good, which checks it against testdata/e2e_golden.json's
// captured input_ids/mm_token_type_ids. This file covers BuildPrompt's nil/argument-validation
// paths (hermetic, no tokenizer needed) plus PositionIDs' full behaviour (pure index arithmetic,
// no tokenizer dependency at all).

func TestBuildPrompt_Bad(t *testing.T) {
	if _, _, err := BuildPrompt(nil, &Config{}, "hi", 4); err == nil {
		t.Fatal("BuildPrompt accepted a nil tokenizer")
	}
}

func TestBuildPrompt_Ugly(t *testing.T) {
	cfg := &Config{ImageTokenID: 100}
	if _, _, err := BuildPrompt(nil, cfg, "hi", 0); err == nil {
		t.Fatal("BuildPrompt accepted numImageTokens=0")
	}
}

func TestPositionIDs_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).RopeIndex
	mmType := g.MMTokenTypeIDs[0]
	gridT, gridH, gridW := g.ImageGridTHW[0][0], g.ImageGridTHW[0][1], g.ImageGridTHW[0][2]

	tPos, hPos, wPos, err := PositionIDs(mmType, gridT, gridH, gridW, g.SpatialMergeSize)
	if err != nil {
		t.Fatalf("PositionIDs: %v", err)
	}
	wantT, wantH, wantW := g.PositionIDs[0][0], g.PositionIDs[1][0], g.PositionIDs[2][0]
	for i := range wantT {
		if tPos[i] != wantT[i] || hPos[i] != wantH[i] || wPos[i] != wantW[i] {
			t.Fatalf("PositionIDs[%d] = (%d,%d,%d), want (%d,%d,%d)", i, tPos[i], hPos[i], wPos[i], wantT[i], wantH[i], wantW[i])
		}
	}
}

func TestPositionIDs_Bad(t *testing.T) {
	// two separate image spans is a named boundary this lane refuses, not silently mishandles
	mmType := []int32{0, 1, 1, 1, 1, 0, 1, 1, 1, 1}
	if _, _, _, err := PositionIDs(mmType, 1, 4, 4, 2); err == nil {
		t.Fatal("PositionIDs accepted two separate image spans")
	}
}

func TestPositionIDs_Ugly(t *testing.T) {
	// an image span whose length doesn't match grid_thw/merge must refuse, not silently
	// misalign positions
	mmType := []int32{0, 1, 1, 1} // 3 image tokens, but a 4x4/merge2 grid implies 4
	if _, _, _, err := PositionIDs(mmType, 1, 4, 4, 2); err == nil {
		t.Fatal("PositionIDs accepted an image span of the wrong length")
	}
}

func TestPositionIDs_PureText_Good(t *testing.T) {
	// no image span at all: every axis is the plain 0..N-1 counter
	mmType := []int32{0, 0, 0, 0}
	tPos, hPos, wPos, err := PositionIDs(mmType, 0, 0, 0, 2)
	if err != nil {
		t.Fatalf("PositionIDs: %v", err)
	}
	for i := range mmType {
		if tPos[i] != i || hPos[i] != i || wPos[i] != i {
			t.Fatalf("PositionIDs pure-text[%d] = (%d,%d,%d), want (%d,%d,%d)", i, tPos[i], hPos[i], wPos[i], i, i, i)
		}
	}
}
