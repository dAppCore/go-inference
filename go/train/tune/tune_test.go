// SPDX-Licence-Identifier: EUPL-1.2

package tune

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestTune_ParseDraftBlocks_Good pins the parse of the --depths surface: the
// empty default, spaces around entries, and skipped empty commas.
func TestTune_ParseDraftBlocks_Good(t *testing.T) {
	cases := []struct {
		in   string
		want []int
	}{
		{"", []int{4, 5, 6}},          // empty defaults to 4,5,6
		{"4,5,6", []int{4, 5, 6}},     // plain
		{" 3, 4 ,8 ", []int{3, 4, 8}}, // whitespace trimmed per entry
		{"5,,6", []int{5, 6}},         // empty entries skipped
		{"2", []int{2}},               // lower bound
		{"8", []int{8}},               // upper bound
	}
	for _, c := range cases {
		got, err := parseDraftBlocks(c.in)
		if err != nil {
			t.Fatalf("parseDraftBlocks(%q) = error %v, want %v", c.in, err, c.want)
		}
		if len(got) != len(c.want) {
			t.Fatalf("parseDraftBlocks(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("parseDraftBlocks(%q) = %v, want %v", c.in, got, c.want)
			}
		}
	}
}

// TestTune_ParseDraftBlocks_Bad pins the rejections: out-of-range blocks
// (a block of 1 has no proposals to verify; >8 is out of MTP range), a
// non-numeric entry, and a value that yields no blocks at all.
func TestTune_ParseDraftBlocks_Bad(t *testing.T) {
	for _, in := range []string{"1", "9", "abc", ", ,"} {
		if got, err := parseDraftBlocks(in); err == nil {
			t.Fatalf("parseDraftBlocks(%q) = %v, nil error, want rejection", in, got)
		}
	}
}

// TestTune_DraftFlag_Good pins the --draft default: blank (or whitespace) means
// "auto" (the reactive ladder), an explicit path passes through untouched.
func TestTune_DraftFlag_Good(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", "auto"},
		{"   ", "auto"},
		{"/models/draft", "/models/draft"},
	}
	for _, c := range cases {
		if got := draftFlag(c.in); got != c.want {
			t.Fatalf("draftFlag(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTune_ValidWorkload_Good pins the standard-set membership check: a workload
// from DefaultTuningWorkloads is accepted, an unknown one is rejected.
func TestTune_ValidWorkload_Good(t *testing.T) {
	if !validWorkload(inference.TuningWorkloadChat) {
		t.Fatalf("validWorkload(%q) = false, want true", inference.TuningWorkloadChat)
	}
	if validWorkload(inference.TuningWorkload("not-a-workload")) {
		t.Fatal("validWorkload(\"not-a-workload\") = true, want false")
	}
}

// TestTune_StandardTuningProfileDir_Good pins the profile-dir default shape:
// <HOME>/Lethean/lem/tuning, the path serve reads profiles from.
func TestTune_StandardTuningProfileDir_Good(t *testing.T) {
	dir := standardTuningProfileDir()
	want := core.PathJoin("Lethean", "lem", "tuning")
	if !core.HasSuffix(dir, want) {
		t.Fatalf("standardTuningProfileDir() = %q, want a path ending in %q", dir, want)
	}
}

// TestTune_RunTune_Bad pins the input-validation arms: a missing --model, an
// unsupported workload, and an out-of-range --depths each error before any
// drafter detection runs.
func TestTune_RunTune_Bad(t *testing.T) {
	if err := RunTune(context.Background(), Config{}); err == nil {
		t.Fatal("RunTune(empty model) = nil, want --model required error")
	}
	if err := RunTune(context.Background(), Config{ModelPath: "/tmp/x", Workload: "bogus"}); err == nil {
		t.Fatal("RunTune(bogus workload) = nil, want unsupported-workload error")
	}
	if err := RunTune(context.Background(), Config{ModelPath: "/tmp/x", Depths: "99"}); err == nil {
		t.Fatal("RunTune(out-of-range depths) = nil, want parse-depths error")
	}
}

// TestTune_RunTune_Ugly pins the no-drafter arm: a valid request whose model
// directory carries no detectable MTP drafter is reported as nothing-to-tune
// rather than silently succeeding.
func TestTune_RunTune_Ugly(t *testing.T) {
	// An empty directory is not a Gemma 4 family config, so the ladder stands
	// down and RunTune reports no drafter.
	err := RunTune(context.Background(), Config{ModelPath: t.TempDir()})
	if err == nil {
		t.Fatal("RunTune(no drafter) = nil, want no-MTP-drafter error")
	}
	if !core.Contains(err.Error(), "no MTP drafter") {
		t.Fatalf("RunTune(no drafter) error = %v, want a no-MTP-drafter message", err)
	}
}

// TestTune_RunTune_Good drives the happy path: a Gemma 4 target with an
// assistant/ drafter beside it resolves an ACTIVE drafter, so RunTune reports
// the tune plan and — honestly — that the MTP sweep is blocked on the
// speculative-pair engine seam, returning nil without writing a faked profile.
func TestTune_RunTune_Good(t *testing.T) {
	modelDir := t.TempDir()
	writeFixture(t, core.PathJoin(modelDir, "config.json"), `{"model_type":"gemma4"}`)
	assistant := core.PathJoin(modelDir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeFixture(t, core.PathJoin(assistant, "config.json"), `{"model_type":"gemma4"}`)
	writeFixture(t, core.PathJoin(assistant, "model.safetensors"), "weights")

	var out bytes.Buffer
	err := RunTune(context.Background(), Config{
		ModelPath: modelDir,
		Depths:    "4,5",
		Out:       &out,
	})
	if err != nil {
		t.Fatalf("RunTune(active drafter) = %v, want nil", err)
	}
	report := out.String()
	for _, want := range []string{
		"tune: target " + modelDir,
		"tune: drafter " + assistant,
		"no registered go-inference engine exposes one yet",
	} {
		if !core.Contains(report, want) {
			t.Fatalf("RunTune report missing %q\n--- got ---\n%s", want, report)
		}
	}
}

func writeFixture(t *testing.T, path, content string) {
	t.Helper()
	if r := core.WriteFile(path, []byte(content), 0o644); !r.OK {
		t.Fatalf("write fixture %s: %v", path, r.Value)
	}
}
