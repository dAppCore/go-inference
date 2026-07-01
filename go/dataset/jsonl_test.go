// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// --- LoadJSONL ---

// Good: LoadJSONL recognises every supported training shape (text,
// prompt/response, alpaca, openai_messages, sharegpt, reasoning) and stamps
// the correct provenance Format on each. Chat-shape rows render Prompt as a
// plain role-free join (see package doc) and retain the raw turns in
// Messages. Reset replays from the top.
func TestJsonl_LoadJSONL_Good(t *testing.T) {
	input := core.Join("\n",
		`{"text":"plain corpus row"}`,
		`{"prompt":"p","response":"r"}`,
		`{"instruction":"summarise","input":"lem notes","output":"short answer"}`,
		`{"messages":[{"role":"system","content":"steady"},{"role":"user","content":"ping"},{"role":"assistant","content":"pong"}]}`,
		`{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"there"}]}`,
		`{"problem":"2+2","thinking":"add the pair","solution":"4"}`,
	)
	ds, err := LoadJSONL(strings.NewReader(input))
	if err != nil {
		t.Fatalf("LoadJSONL() error = %v", err)
	}
	samples := collectDatasetSamples(t, ds)
	if len(samples) != 6 {
		t.Fatalf("samples len = %d, want 6", len(samples))
	}
	if samples[0].Text != "plain corpus row" || samples[0].Format != "text" {
		t.Fatalf("text sample = %+v", samples[0])
	}
	if samples[1].Prompt != "p" || samples[1].Response != "r" || samples[1].Format != "prompt_response" {
		t.Fatalf("prompt/response sample = %+v", samples[1])
	}
	if !core.Contains(samples[2].Prompt, "summarise") || !core.Contains(samples[2].Prompt, "lem notes") || samples[2].Response != "short answer" || samples[2].Format != "alpaca" {
		t.Fatalf("alpaca sample = %+v", samples[2])
	}
	if samples[3].Prompt != "steady\nping\n" || samples[3].Response != "pong" || samples[3].Format != "openai_messages" || len(samples[3].Messages) != 3 {
		t.Fatalf("openai messages sample = %+v", samples[3])
	}
	if samples[4].Prompt != "hi\n" || samples[4].Response != "there" || samples[4].Format != "sharegpt" || len(samples[4].Messages) != 2 {
		t.Fatalf("sharegpt sample = %+v", samples[4])
	}
	// Deliberate divergence from go-mlx: Reasoning is retained in its own
	// field rather than concatenated into Response (see dataset.go's
	// DatasetSample.Reasoning field).
	if samples[5].Prompt != "2+2" || samples[5].Reasoning != "add the pair" || samples[5].Response != "4" || samples[5].Format != "reasoning" {
		t.Fatalf("reasoning sample = %+v", samples[5])
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	again, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() after Reset error = %v", err)
	}
	if !ok || again.Text != "plain corpus row" {
		t.Fatalf("Next() after Reset = %+v ok=%v", again, ok)
	}
}

// Bad: LoadJSONL rejects malformed input. A nil reader trips the hoisted
// sentinel before the decoder is touched; a syntactically invalid record
// surfaces a parse error naming the record number.
func TestJsonl_LoadJSONL_Bad(t *testing.T) {
	t.Run("nil reader", func(t *testing.T) {
		if _, err := LoadJSONL(nil); err == nil {
			t.Fatal("LoadJSONL(nil) expected error, got nil")
		}
	})
	t.Run("invalid json", func(t *testing.T) {
		_, err := LoadJSONL(strings.NewReader("{not-json}\n"))
		if err == nil {
			t.Fatal("expected invalid JSONL error")
		}
	})
}

// Ugly: LoadJSONL on the awkward-but-valid edges. Blank/whitespace-only
// input yields an empty dataset (no error). Partial-shape rows sweep the
// prompt-only, response-only, alpaca half-row and reasoning half-row
// branches, dropping wholly-unrecognised rows. A messages row with an
// empty (or whitespace-only, post-normalise-empty) leading message
// exercises both empty-skip short-circuits while surviving turns still
// normalise.
func TestJsonl_LoadJSONL_Ugly(t *testing.T) {
	t.Run("empty input", func(t *testing.T) {
		ds, err := LoadJSONL(strings.NewReader("\n  \n\n"))
		if err != nil {
			t.Fatalf("LoadJSONL(blank) error = %v", err)
		}
		if got := collectDatasetSamples(t, ds); len(got) != 0 {
			t.Fatalf("LoadJSONL(blank) samples = %d, want 0", len(got))
		}
	})
	t.Run("partial shapes", func(t *testing.T) {
		input := core.Join("\n",
			`{"response":"bare completion"}`,             // response-only -> prompt_response
			`{"completion":"via completion key"}`,        // completion alias
			`{"instruction":"do the thing"}`,             // alpaca, no input/output
			`{"input":"only input text","output":"out"}`, // alpaca, instruction empty
			`{"thinking":"reason only"}`,                 // not a recognised shape (no solution) -> skipped
			`{"solution":"42"}`,                          // reasoning, solution-only, no thinking
			`{"problem":"why","thinking":"because"}`,     // reasoning, problem + thinking, no solution
			`{"unknown":"field"}`,                        // wholly unrecognised -> skipped
		)
		ds, err := LoadJSONL(strings.NewReader(input))
		if err != nil {
			t.Fatalf("LoadJSONL(partial) error = %v", err)
		}
		samples := collectDatasetSamples(t, ds)
		// 6 recognised rows; the thinking-only and unknown rows are dropped.
		if len(samples) != 6 {
			t.Fatalf("partial-shape samples = %d, want 6: %+v", len(samples), samples)
		}
		if samples[0].Response != "bare completion" || samples[0].Format != "prompt_response" {
			t.Fatalf("response-only sample = %+v", samples[0])
		}
		if samples[1].Response != "via completion key" {
			t.Fatalf("completion-alias sample = %+v", samples[1])
		}
		// instruction-only: prompt is the instruction, no input appended.
		if samples[2].Prompt != "do the thing" || samples[2].Format != "alpaca" {
			t.Fatalf("instruction-only sample = %+v", samples[2])
		}
		// input-only (empty instruction): prompt is just the input.
		if samples[3].Prompt != "only input text" || samples[3].Response != "out" {
			t.Fatalf("input-only alpaca sample = %+v", samples[3])
		}
		// solution-only reasoning: response is the bare solution, no
		// Reasoning (thinking was never supplied).
		if samples[4].Response != "42" || samples[4].Reasoning != "" || samples[4].Format != "reasoning" {
			t.Fatalf("solution-only reasoning sample = %+v", samples[4])
		}
		// problem + thinking, no solution: Reasoning carries the thinking
		// text; Response stays empty (there is no solution/answer field to
		// fall back to).
		if samples[5].Prompt != "why" || samples[5].Reasoning != "because" || samples[5].Response != "" || samples[5].Format != "reasoning" {
			t.Fatalf("problem+thinking reasoning sample = %+v", samples[5])
		}
	})
	t.Run("openai empty message skipped", func(t *testing.T) {
		input := `{"messages":[{"role":"","content":""},{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}`
		ds, err := LoadJSONL(strings.NewReader(input))
		if err != nil {
			t.Fatalf("LoadJSONL(empty-msg) error = %v", err)
		}
		samples := collectDatasetSamples(t, ds)
		if len(samples) != 1 {
			t.Fatalf("empty-msg samples = %d, want 1", len(samples))
		}
		if samples[0].Response != "a" || samples[0].Format != "openai_messages" {
			t.Fatalf("empty-msg sample = %+v", samples[0])
		}
	})
	t.Run("openai whitespace role and content post-normalise skip", func(t *testing.T) {
		// The raw fields are non-empty so the leading short-circuit
		// (Role=="" && Content=="") does not fire; NormaliseRole lowers
		// the trimmed role to "" and Trim collapses the content to "", so
		// the *post-normalisation* skip drops the turn instead.
		input := `{"messages":[{"role":"   ","content":"  "},{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}`
		ds, err := LoadJSONL(strings.NewReader(input))
		if err != nil {
			t.Fatalf("LoadJSONL(whitespace-role) error = %v", err)
		}
		samples := collectDatasetSamples(t, ds)
		if len(samples) != 1 {
			t.Fatalf("whitespace-role samples = %d, want 1 (skipped turn must not yield a sample)", len(samples))
		}
		if samples[0].Response != "a" || samples[0].Format != "openai_messages" {
			t.Fatalf("whitespace-role surviving sample = %+v", samples[0])
		}
	})
}

// --- NewJSONL ---

// Good: NewJSONL clones the supplied samples (including each Labels map) so
// later mutation of the source slice cannot reach the dataset's stored
// records.
func TestJsonl_NewJSONL_Good(t *testing.T) {
	samples := []Sample{{Text: "a", Labels: map[string]string{"k": "v"}}}
	ds := NewJSONL(samples)
	samples[0].Text = "mutated"
	samples[0].Labels["k"] = "changed"

	got, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || got.Text != "a" || got.Labels["k"] != "v" {
		t.Fatalf("Next() = %+v ok=%v, want cloned original", got, ok)
	}
}

// Bad: NewJSONL is total (no error channel), so the degenerate input is a
// nil slice — the constructed dataset must be usable and immediately
// exhausted, never nil and never a panic on first Next.
func TestJsonl_NewJSONL_Bad(t *testing.T) {
	ds := NewJSONL(nil)
	if ds == nil {
		t.Fatal("NewJSONL(nil) = nil, want a usable empty dataset")
	}
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("NewJSONL(nil).Next() = ok %v err %v, want false,nil", ok, err)
	}
}

// Ugly: NewJSONL over an empty (zero-length, non-nil) slice yields an
// immediately-exhausted dataset, and Samples() on it returns nil rather
// than a zero-length non-nil slice.
func TestJsonl_NewJSONL_Ugly(t *testing.T) {
	ds := NewJSONL([]Sample{})
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("NewJSONL(empty).Next() = ok %v err %v, want false,nil", ok, err)
	}
	if got := ds.Samples(); got != nil {
		t.Fatalf("NewJSONL(empty).Samples() = %v, want nil", got)
	}
}

// --- JSONLDataset.Next ---

// Good: sequential Next over a JSONLDataset yields each stored record in
// order and exhausts cleanly. Each returned sample is a defensive clone, so
// mutating the returned Labels does not corrupt the dataset's copy for the
// next pass.
func TestJsonl_JSONLDataset_Next_Good(t *testing.T) {
	ds := NewJSONL([]Sample{
		{Text: "a", Labels: map[string]string{"k": "v"}},
		{Prompt: "p", Response: "r"},
	})

	first, ok, err := ds.Next()
	if err != nil || !ok {
		t.Fatalf("Next()[0] ok=%v err=%v", ok, err)
	}
	if first.Text != "a" {
		t.Fatalf("Next()[0] = %+v, want Text 'a'", first)
	}
	first.Labels["k"] = "tampered"

	second, ok, err := ds.Next()
	if err != nil || !ok || second.Prompt != "p" || second.Response != "r" {
		t.Fatalf("Next()[1] = %+v ok=%v err=%v", second, ok, err)
	}
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("Next() at end = ok %v err %v, want false,nil", ok, err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	again, _, _ := ds.Next()
	if again.Labels["k"] != "v" {
		t.Fatalf("returned-clone mutation leaked: Labels[k] = %q, want 'v'", again.Labels["k"])
	}
}

// Bad: Next on a nil *JSONLDataset returns the sentinel error rather than
// panicking.
func TestJsonl_JSONLDataset_Next_Bad(t *testing.T) {
	var ds *JSONLDataset
	if _, _, err := ds.Next(); err == nil {
		t.Fatal("expected nil Next error")
	}
}

// Ugly: Next past the end of a JSONLDataset is idempotent — once exhausted
// it keeps returning (zero, false, nil) with no advance and no leaked
// fields.
func TestJsonl_JSONLDataset_Next_Ugly(t *testing.T) {
	ds := NewJSONL([]Sample{{Text: "only"}})
	if _, ok, _ := ds.Next(); !ok {
		t.Fatal("first Next() ok = false, want the single record")
	}
	for i := 0; i < 3; i++ {
		got, ok, err := ds.Next()
		if ok || err != nil {
			t.Fatalf("Next() past end #%d = ok %v err %v, want false,nil", i, ok, err)
		}
		if got.Text != "" {
			t.Fatalf("Next() past end #%d leaked Text %q", i, got.Text)
		}
	}
}

// --- JSONLDataset.Reset ---

// Good: Reset rewinds a JSONLDataset so a second epoch replays the same
// records from the top.
func TestJsonl_JSONLDataset_Reset_Good(t *testing.T) {
	ds := NewJSONL([]Sample{{Text: "row0"}, {Text: "row1"}})
	if _, _, err := ds.Next(); err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	again, ok, err := ds.Next()
	if err != nil || !ok || again.Text != "row0" {
		t.Fatalf("after Reset Next() = %+v ok=%v err=%v, want row0", again, ok, err)
	}
}

// Bad: Reset on a nil *JSONLDataset returns the sentinel error rather than
// panicking.
func TestJsonl_JSONLDataset_Reset_Bad(t *testing.T) {
	var ds *JSONLDataset
	if err := ds.Reset(); err == nil {
		t.Fatal("expected nil Reset error")
	}
}

// Ugly: Reset is safe at the boundaries — before any Next (no-op) and
// twice in a row both leave the cursor at the top with no error.
func TestJsonl_JSONLDataset_Reset_Ugly(t *testing.T) {
	ds := NewJSONL([]Sample{{Text: "head"}, {Text: "tail"}})
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() before first Next error = %v", err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("second consecutive Reset() error = %v", err)
	}
	got, ok, err := ds.Next()
	if err != nil || !ok || got.Text != "head" {
		t.Fatalf("after boundary Resets, Next() = %+v ok=%v err=%v, want head", got, ok, err)
	}
}

// --- JSONLDataset.Samples ---

// Good: Samples returns the full record set as a defensive copy — every
// call hands back an independent slice with its own Labels maps.
func TestJsonl_JSONLDataset_Samples_Good(t *testing.T) {
	ds := NewJSONL([]Sample{
		{Text: "a", Labels: map[string]string{"format": "text"}},
		{Prompt: "p", Response: "r"},
	})
	got := ds.Samples()
	if len(got) != 2 {
		t.Fatalf("Samples() len = %d, want 2", len(got))
	}
	if got[0].Text != "a" || got[1].Prompt != "p" {
		t.Fatalf("Samples() content = %+v", got)
	}
}

// Bad: Samples on a nil *JSONLDataset returns nil rather than panicking.
func TestJsonl_JSONLDataset_Samples_Bad(t *testing.T) {
	var ds *JSONLDataset
	if got := ds.Samples(); got != nil {
		t.Fatalf("nil JSONLDataset.Samples() = %v, want nil", got)
	}
}

// Ugly: successive Samples() calls do not alias each other or the
// dataset's backing — mutating one returned slice (text and Labels) leaves
// a later call untouched.
func TestJsonl_JSONLDataset_Samples_Ugly(t *testing.T) {
	ds := NewJSONL([]Sample{{Text: "a", Labels: map[string]string{"format": "text"}}})
	samples := ds.Samples()
	samples[0].Text = "changed"
	samples[0].Labels["format"] = "changed"
	again := ds.Samples()
	if again[0].Text != "a" || again[0].Labels["format"] != "text" {
		t.Fatalf("Samples() aliased storage: %+v", again)
	}
}

// --- MessagesToSample ---

// Good: MessagesToSample takes the trailing assistant turn as the
// (trimmed) response, joins the preceding turns plainly into Prompt, and
// retains the full normalised turn list in Messages.
func TestJsonl_MessagesToSample_Good(t *testing.T) {
	messages := []inference.Message{
		{Role: "system", Content: "be exact"},
		{Role: "user", Content: "Write one line."},
		{Role: "assistant", Content: " one line "},
	}

	sample, ok, err := MessagesToSample(messages, "openai_messages")
	if err != nil {
		t.Fatalf("MessagesToSample() error = %v", err)
	}
	if !ok {
		t.Fatal("MessagesToSample() ok = false, want sample")
	}
	wantPrompt := "be exact\nWrite one line.\n"
	if sample.Prompt != wantPrompt {
		t.Fatalf("Prompt = %q, want plain join %q", sample.Prompt, wantPrompt)
	}
	if sample.Response != "one line" {
		t.Fatalf("Response = %q, want trimmed assistant response", sample.Response)
	}
	if sample.Format != "openai_messages" {
		t.Fatalf("format = %q, want openai_messages", sample.Format)
	}
	if len(sample.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3 (raw turns retained)", len(sample.Messages))
	}
}

// Bad: MessagesToSample with no messages returns (zero, false, nil) — the
// empty-slice guard, for both a nil slice and an explicitly empty one.
func TestJsonl_MessagesToSample_Bad(t *testing.T) {
	if _, ok, err := MessagesToSample(nil, "openai_messages"); ok || err != nil {
		t.Fatalf("MessagesToSample(nil) = ok %v err %v, want false,nil", ok, err)
	}
	if _, ok, err := MessagesToSample([]inference.Message{}, "openai_messages"); ok || err != nil {
		t.Fatalf("MessagesToSample(empty) = ok %v err %v, want false,nil", ok, err)
	}
}

// Ugly: a conversation that ends without an assistant turn drops into the
// no-assistant Text-fallback branch and returns a Text sample (the plain
// join of every turn) rather than a Prompt/Response pair.
func TestJsonl_MessagesToSample_Ugly(t *testing.T) {
	messages := []inference.Message{
		{Role: "system", Content: "steady"},
		{Role: "user", Content: "ping"},
	}

	sample, ok, err := MessagesToSample(messages, "openai_messages")
	if err != nil {
		t.Fatalf("MessagesToSample() error = %v", err)
	}
	if !ok {
		t.Fatal("MessagesToSample() ok = false, want true")
	}
	if sample.Response != "" || sample.Prompt != "" {
		t.Fatalf("no-assistant sample should be Text-only, got Prompt=%q Response=%q", sample.Prompt, sample.Response)
	}
	wantText := "steady\nping\n"
	if sample.Text != wantText {
		t.Fatalf("Text = %q, want plain join %q", sample.Text, wantText)
	}
	if sample.Format != "openai_messages" {
		t.Fatalf("format = %q, want openai_messages", sample.Format)
	}
	if len(sample.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2 (raw turns retained)", len(sample.Messages))
	}
}

// --- NormaliseRole ---

// Good: NormaliseRole passes canonical roles through unchanged and maps
// every documented HF / ShareGPT / Llama / Gemma alias to its canonical
// form, case- and whitespace-insensitively.
func TestJsonl_NormaliseRole_Good(t *testing.T) {
	cases := map[string]string{
		"user":        "user",
		"assistant":   "assistant",
		"system":      "system",
		"human":       "user",
		"gpt":         "assistant",
		"bot":         "assistant",
		"model":       "assistant",
		"developer":   "system",
		" ASSISTANT ": "assistant",
		" Human ":     "user",
	}
	for in, want := range cases {
		if got := NormaliseRole(in); got != want {
			t.Fatalf("NormaliseRole(%q) = %q, want %q", in, got, want)
		}
	}
}

// Bad: empty input returns empty — no invented role for the degenerate
// case.
func TestJsonl_NormaliseRole_Bad(t *testing.T) {
	if got := NormaliseRole(""); got != "" {
		t.Fatalf("NormaliseRole(\"\") = %q, want empty", got)
	}
}

// Ugly: an unrecognised role is lower-cased and returned rather than
// dropped or erroring; whitespace-only input collapses to empty.
func TestJsonl_NormaliseRole_Ugly(t *testing.T) {
	if got := NormaliseRole("ROBOT"); got != "robot" {
		t.Fatalf("NormaliseRole(unknown) = %q, want lower-cased passthrough 'robot'", got)
	}
	if got := NormaliseRole("   "); got != "" {
		t.Fatalf("NormaliseRole(whitespace) = %q, want empty", got)
	}
}

// --- appendMessagesFromOpenAI (nil-buffer contract) ---

// Ugly: appendMessagesFromOpenAI honours the documented "pass nil when no
// reuse is available" contract — with a nil buffer it allocates a fresh
// slice via claimMessageBuf's nil branch rather than reusing a backing
// array. LoadJSONL always passes &msgBuf, so this contract is only
// reachable by an external-style (no-reuse) call; assert the returned
// slice end-to-end (normalised roles, trimmed content, empties dropped).
func TestJsonl_AppendMessagesFromOpenAI_NilBuffer(t *testing.T) {
	records := []messageRecord{
		{Role: "system", Content: "steady"},
		{Role: "", Content: ""},          // raw short-circuit skip
		{Role: "human", Content: " hi "}, // human -> user, content trimmed
		{Role: "gpt", Content: "yo"},     // gpt -> assistant
	}
	out := appendMessagesFromOpenAI(nil, records)
	want := []inference.Message{
		{Role: "system", Content: "steady"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "yo"},
	}
	if len(out) != len(want) {
		t.Fatalf("appendMessagesFromOpenAI(nil) len = %d, want %d: %+v", len(out), len(want), out)
	}
	for i := range want {
		if out[i].Role != want[i].Role || out[i].Content != want[i].Content {
			t.Fatalf("appendMessagesFromOpenAI(nil)[%d] = %+v, want %+v", i, out[i], want[i])
		}
	}
}

func collectDatasetSamples(t *testing.T, ds Dataset) []Sample {
	t.Helper()
	var samples []Sample
	for {
		sample, ok, err := ds.Next()
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if !ok {
			return samples
		}
		samples = append(samples, sample)
	}
}
