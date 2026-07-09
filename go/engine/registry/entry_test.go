// SPDX-Licence-Identifier: EUPL-1.2

package registry

import "testing"

// fullEntry returns an Entry satisfying every Filter field, so tests can
// selectively strip one capability/status to see it individually rejected.
func fullEntry() Entry {
	return Entry{
		ID: "full",
		Capabilities: Capabilities{
			Tools:     true,
			Vision:    true,
			Grammar:   true,
			Streaming: true,
		},
		Status: StatusReady,
	}
}

func TestEntry_Filter_Matches_Good(t *testing.T) {
	f := Filter{Tools: true, Vision: true, Grammar: true, Streaming: true, ReadyOnly: true}
	if !f.matches(fullEntry()) {
		t.Fatalf("full filter should match a fully-capable, ready entry")
	}
}

func TestEntry_Filter_Matches_Bad(t *testing.T) {
	// Each required field independently rejects an entry that lacks it.
	cases := []struct {
		name string
		f    Filter
		e    Entry
	}{
		{"tools", Filter{Tools: true}, func() Entry { e := fullEntry(); e.Capabilities.Tools = false; return e }()},
		{"vision", Filter{Vision: true}, func() Entry { e := fullEntry(); e.Capabilities.Vision = false; return e }()},
		{"grammar", Filter{Grammar: true}, func() Entry { e := fullEntry(); e.Capabilities.Grammar = false; return e }()},
		{"streaming", Filter{Streaming: true}, func() Entry { e := fullEntry(); e.Capabilities.Streaming = false; return e }()},
		{"ready", Filter{ReadyOnly: true}, func() Entry { e := fullEntry(); e.Status = StatusDraft; return e }()},
	}
	for _, c := range cases {
		if c.f.matches(c.e) {
			t.Errorf("%s: entry lacking the required field should not match", c.name)
		}
	}
}

func TestEntry_Filter_Matches_Ugly(t *testing.T) {
	// The zero Filter matches everything, regardless of capability or status
	// — an unset requirement narrows nothing.
	zero := Filter{}
	if !zero.matches(fullEntry()) {
		t.Fatalf("zero filter should match a fully-capable entry")
	}
	bare := Entry{ID: "bare"} // no capabilities set, zero-value status
	if !zero.matches(bare) {
		t.Fatalf("zero filter should match even a bare entry")
	}
}
