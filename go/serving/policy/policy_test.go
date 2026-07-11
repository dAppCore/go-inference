// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"testing"

	core "dappco.re/go"
)

// TestPolicy_Compile_Good pins a valid policy compiling: the rule count, the
// per-kind fields, and the redact/pattern defaults.
func TestPolicy_Compile_Good(t *testing.T) {
	pol, err := Compile([]byte(`{"rules":[
		{"match":"term","value":"PROJECT-X","action":"redact","replacement":"[gone]"},
		{"match":"pattern","value":"rc[0-9]+","action":"refuse","message":"no unreleased builds"},
		{"match":"term","value":"client","action":"redact"}
	]}`))
	if err != nil {
		t.Fatalf("Compile of a valid policy failed: %v", err)
	}
	if pol.Len() != 3 {
		t.Fatalf("rule count = %d, want 3", pol.Len())
	}
	if pol.rules[0].Replacement != "[gone]" {
		t.Fatalf("explicit replacement = %q, want [gone]", pol.rules[0].Replacement)
	}
	if pol.rules[1].Action != ActionRefuse || pol.rules[1].Message != "no unreleased builds" {
		t.Fatalf("refuse rule = %+v, want the configured message", pol.rules[1])
	}
	if pol.rules[2].Replacement != DefaultReplacement {
		t.Fatalf("defaulted replacement = %q, want %q", pol.rules[2].Replacement, DefaultReplacement)
	}
	if pol.rules[1].window != DefaultWindow {
		t.Fatalf("defaulted pattern window = %d, want %d", pol.rules[1].window, DefaultWindow)
	}
}

// TestPolicy_Compile_Empty pins the empty-but-valid policy: no rules is a no-op
// layer, not a failure (an empty FILE, by contrast, is malformed JSON).
func TestPolicy_Compile_Empty(t *testing.T) {
	pol, err := Compile([]byte(`{"rules":[]}`))
	if err != nil {
		t.Fatalf("empty rule set should compile, got %v", err)
	}
	if pol.Len() != 0 {
		t.Fatalf("rule count = %d, want 0", pol.Len())
	}
	if pol.HoldBack() != 1 {
		t.Fatalf("empty policy hold-back = %d, want 1 (no look-ahead)", pol.HoldBack())
	}
}

// TestPolicy_HoldBack pins the streaming hold-back bound as max(longest term,
// largest pattern window).
func TestPolicy_HoldBack(t *testing.T) {
	t.Run("term dominates", func(t *testing.T) {
		pol, err := Compile([]byte(`{"rules":[{"match":"term","value":"abcdefgh","action":"redact"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if pol.HoldBack() != 8 {
			t.Fatalf("hold-back = %d, want 8 (len of the term)", pol.HoldBack())
		}
	})
	t.Run("window dominates", func(t *testing.T) {
		pol, err := Compile([]byte(`{"rules":[
			{"match":"term","value":"ab","action":"redact"},
			{"match":"pattern","value":"x[0-9]+","action":"redact","window":40}
		]}`))
		if err != nil {
			t.Fatal(err)
		}
		if pol.HoldBack() != 40 {
			t.Fatalf("hold-back = %d, want 40 (pattern window)", pol.HoldBack())
		}
	})
}

// TestPolicy_Compile_Bad pins every load-time rejection: a misconfigured policy
// fails at load with a clear message, never at serve time.
func TestPolicy_Compile_Bad(t *testing.T) {
	cases := []struct {
		name string
		json string
		want string
	}{
		{"malformed json", `{`, "parse policy JSON"},
		{"unknown match", `{"rules":[{"match":"glob","value":"x","action":"redact"}]}`, "unknown match"},
		{"unknown action", `{"rules":[{"match":"term","value":"x","action":"delete"}]}`, "unknown action"},
		{"empty term", `{"rules":[{"match":"term","value":"","action":"redact"}]}`, "term rule requires a non-empty value"},
		{"empty pattern", `{"rules":[{"match":"pattern","value":"","action":"redact"}]}`, "pattern rule requires a non-empty value"},
		{"bad regexp", `{"rules":[{"match":"pattern","value":"[","action":"redact"}]}`, "compile pattern"},
		{"empty-matching pattern", `{"rules":[{"match":"pattern","value":"a*","action":"redact"}]}`, "matches the empty string"},
		{"refuse without message", `{"rules":[{"match":"term","value":"x","action":"refuse"}]}`, "refuse rule requires a non-empty message"},
		{"redact carrying message", `{"rules":[{"match":"term","value":"x","action":"redact","message":"no"}]}`, "must not carry a refuse message"},
		{"refuse carrying replacement", `{"rules":[{"match":"term","value":"x","action":"refuse","message":"m","replacement":"r"}]}`, "must not carry a redact replacement"},
		{"term carrying window", `{"rules":[{"match":"term","value":"x","action":"redact","window":10}]}`, "window is valid only for pattern rules"},
		{"file window too large", `{"window":99999999,"rules":[]}`, "policy window"},
		{"rule window too large", `{"rules":[{"match":"pattern","value":"x","action":"redact","window":99999999}]}`, "pattern window"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pol, err := Compile([]byte(tc.json))
			if err == nil {
				t.Fatalf("expected a load error, got a policy with %d rule(s)", pol.Len())
			}
			if !core.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %q, want it to contain %q", err.Error(), tc.want)
			}
		})
	}
}

// TestPolicy_Load_Good pins loading from a file on disk.
func TestPolicy_Load_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "policy.json")
	if r := core.WriteFile(path, []byte(`{"rules":[{"match":"term","value":"secret","action":"redact"}]}`), 0o644); !r.OK {
		t.Fatalf("write policy fixture: %s", r.Error())
	}
	pol, err := Load(path)
	if err != nil {
		t.Fatalf("Load of a valid file failed: %v", err)
	}
	if pol.Len() != 1 {
		t.Fatalf("rule count = %d, want 1", pol.Len())
	}
}

// TestPolicy_Load_Bad pins the fail-at-boot contract: an unreadable file is a
// load error naming the path.
func TestPolicy_Load_Bad(t *testing.T) {
	_, err := Load(core.PathJoin(t.TempDir(), "does-not-exist.json"))
	if err == nil {
		t.Fatal("Load of a missing file must error")
	}
	if !core.Contains(err.Error(), "read policy file") {
		t.Fatalf("error = %q, want it to mention the read failure", err.Error())
	}
}
