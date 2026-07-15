// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

func TestAddLearnedPositions_Golden(t *testing.T) {
	h := []float32{1, 2, 3, 4}
	if err := AddLearnedPositions(h, []float32{10, 20, 30, 40, 50, 60}, 2, 2, 1); err != nil {
		t.Fatal(err)
	}
	want := []float32{31, 42, 53, 64}
	for i := range want {
		if h[i] != want[i] {
			t.Fatalf("hidden[%d]=%v want %v", i, h[i], want[i])
		}
	}
}

func TestAddLearnedPositions_Bad(t *testing.T) {
	if AddLearnedPositions([]float32{1}, nil, 1, 1, 0) == nil {
		t.Fatal("short table accepted")
	}
}

func TestExpandMultiQueryKV_Golden(t *testing.T) {
	got, err := ExpandMultiQueryKV([]float32{1, 2, 3, 4}, 2, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestExpandMultiQueryKV_Bad(t *testing.T) {
	if _, err := ExpandMultiQueryKV([]float32{1}, 1, 2, 2); err == nil {
		t.Fatal("bad shape accepted")
	}
}
