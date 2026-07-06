// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import "testing"

// TestSource_known_Good checks that every declared feedback source constant
// is recognised.
func TestSource_known_Good(t *testing.T) {
	for _, s := range []Source{SourceHuman, SourceEvaluator, SourceHeuristic} {
		if !s.known() {
			t.Errorf("known(): %q should be recognised", s)
		}
	}
}

// TestSource_known_Bad checks that an arbitrary, undeclared source is not
// recognised — RecordFeedback relies on this to reject bad input.
func TestSource_known_Bad(t *testing.T) {
	for _, s := range []Source{Source("robot"), Source("ROBOT"), Source("Human")} {
		if s.known() {
			t.Errorf("known(): %q should not be recognised", s)
		}
	}
}

// TestSource_known_Ugly checks the empty source: it is not itself "known" —
// RecordFeedback defaults an empty source to SourceEvaluator before storing,
// rather than treating an unset source as already valid.
func TestSource_known_Ugly(t *testing.T) {
	if Source("").known() {
		t.Error("the empty source should not be known — callers must default it first")
	}
}
