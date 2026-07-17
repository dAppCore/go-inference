// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"
)

// TestGenerateSampledResumeEach_Good proves the stateful resume contract on ONE open
// stepper: two resume calls (the second fed the first's returned SessionResume, which
// carries the picked-but-unstepped final token as PendingID) produce exactly the token
// stream one longer call produces — no gap, no duplicate, the pending token entering the
// session on the second call.
func TestGenerateSampledResumeEach_Good(t *testing.T) {
	m := sessionCounterModel{counterModel: counterModel{vocab: 32, dModel: 4}, opened: new(int), closed: new(int)}

	prefill := func() (DecodeStepper, []byte) {
		sess, err := m.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession: %v", err)
		}
		var hidden []byte
		for _, id := range []int32{3} {
			emb, err := m.Embed(id)
			if err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if hidden, err = sess.Step(emb); err != nil {
				t.Fatalf("Step: %v", err)
			}
		}
		return sess, hidden
	}

	// reference: one 6-token decode from the prefilled state.
	sess, hidden := prefill()
	want, _, err := GenerateSampledResumeEach(m, sess, SessionResume{Hidden: hidden, PendingID: -1}, NewSampler(0), SampleParams{}, 6, nil, nil, nil)
	if err != nil {
		t.Fatalf("reference resume: %v", err)
	}
	if wantIDs := []int32{4, 5, 6, 7, 8, 9}; !idsEqual(want, wantIDs) {
		t.Fatalf("reference = %v, want %v", want, wantIDs)
	}

	// split: 3 tokens, then 3 more resuming with the returned SessionResume.
	sess, hidden = prefill()
	first, r, err := GenerateSampledResumeEach(m, sess, SessionResume{Hidden: hidden, PendingID: -1}, NewSampler(0), SampleParams{}, 3, nil, nil, nil)
	if err != nil {
		t.Fatalf("first resume: %v", err)
	}
	if r.PendingID != first[len(first)-1] {
		t.Fatalf("PendingID = %d, want the unstepped final pick %d", r.PendingID, first[len(first)-1])
	}
	second, r2, err := GenerateSampledResumeEach(m, sess, r, NewSampler(0), SampleParams{}, 3, nil, nil, nil)
	if err != nil {
		t.Fatalf("second resume: %v", err)
	}
	got := append(append([]int32(nil), first...), second...)
	if !idsEqual(got, want) {
		t.Fatalf("split resume = %v, want the unbroken stream %v", got, want)
	}
	if r2.PendingID != second[len(second)-1] {
		t.Fatalf("second PendingID = %d, want %d", r2.PendingID, second[len(second)-1])
	}
}

// TestGenerateSampledResumeEach_Bad locks the argument guards: nil sampler, nil session,
// a missing resume hidden (session never prefilled) and a non-positive budget all error
// rather than decode from undefined state.
func TestGenerateSampledResumeEach_Bad(t *testing.T) {
	m := sessionCounterModel{counterModel: counterModel{vocab: 8, dModel: 4}, opened: new(int), closed: new(int)}
	sess, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	hidden := []byte{0, 0}
	if _, _, err := GenerateSampledResumeEach(m, sess, SessionResume{Hidden: hidden, PendingID: -1}, nil, SampleParams{}, 1, nil, nil, nil); err == nil {
		t.Fatal("nil sampler must error")
	}
	if _, _, err := GenerateSampledResumeEach(m, nil, SessionResume{Hidden: hidden, PendingID: -1}, NewSampler(0), SampleParams{}, 1, nil, nil, nil); err == nil {
		t.Fatal("nil session must error")
	}
	if _, _, err := GenerateSampledResumeEach(m, sess, SessionResume{PendingID: -1}, NewSampler(0), SampleParams{}, 1, nil, nil, nil); err == nil {
		t.Fatal("missing resume hidden must error")
	}
	if _, _, err := GenerateSampledResumeEach(m, sess, SessionResume{Hidden: hidden, PendingID: -1}, NewSampler(0), SampleParams{}, 0, nil, nil, nil); err == nil {
		t.Fatal("maxNew 0 must error")
	}
}
