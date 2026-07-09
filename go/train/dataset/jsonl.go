// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"bufio"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sentinel errors hoisted from the nil-guard call sites so they
// allocate exactly once at package init instead of one *Err per
// nil-receiver call. These are cold paths but the package contract
// is the same either way.
var (
	errReaderNil       = core.NewError("dataset: reader is nil")
	errJSONLDatasetNil = core.NewError("dataset: JSONL dataset is nil")
)

// JSONLDataset is a replayable in-memory dataset loaded from JSONL records.
type JSONLDataset struct {
	samples []Sample
	index   int
}

// Compile-time proof that JSONLDataset satisfies the canonical
// inference.DatasetStream / DatasetResetter contracts via the Dataset /
// Resetter aliases.
var (
	_ Dataset  = (*JSONLDataset)(nil)
	_ Resetter = (*JSONLDataset)(nil)
)

type jsonRecord struct {
	Text          string           `json:"text"`
	Prompt        string           `json:"prompt"`
	Response      string           `json:"response"`
	Completion    string           `json:"completion"`
	Instruction   string           `json:"instruction"`
	Input         string           `json:"input"`
	Output        string           `json:"output"`
	Problem       string           `json:"problem"`
	Question      string           `json:"question"`
	Thinking      string           `json:"thinking"`
	Reasoning     string           `json:"reasoning"`
	Solution      string           `json:"solution"`
	Answer        string           `json:"answer"`
	Messages      []messageRecord  `json:"messages"`
	Conversations []shareGPTRecord `json:"conversations"`
}

type messageRecord struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type shareGPTRecord struct {
	From  string `json:"from"`
	Value string `json:"value"`
}

// LoadJSONL reads JSONL into a replayable Dataset, auto-detecting each row's
// shape: a bare "text" field, OpenAI-style "messages", ShareGPT-style
// "conversations", a "prompt"/"response" pair, Alpaca
// "instruction"/"input"/"output", or a "problem"/"solution" reasoning pair.
// Chat-shaped rows (messages/conversations) keep their normalised turns in
// Sample.Messages rather than pre-rendering a model-specific chat-template
// prompt — see the package doc.
//
//	d, err := dataset.LoadJSONL(reader)
func LoadJSONL(reader core.Reader) (*JSONLDataset, error) {
	if reader == nil {
		return nil, errReaderNil
	}
	// One streaming decoder for the whole file — core.JSONUnmarshal would
	// allocate a fresh decodeState (~5 allocs per call) per row, whereas
	// JSONNewDecoder's Decoder reuses its internal scratch buffers across
	// Decode() calls. Decoder handles inter-record whitespace (including
	// empty lines) on its own. core.JSONNewDecoder wraps encoding/json so
	// this package never imports it directly (core/json.go's guardrail).
	dec := core.JSONNewDecoder(bufio.NewReaderSize(reader, 64*1024))

	// Pre-size the samples buffer — corpora of any meaningful size run
	// through several growslice rounds otherwise (nil → 1 → 2 → 4 → 8 →
	// ...). Starting at 64 covers the first ~6 doublings and is small
	// enough to be no waste on tiny inputs. Larger corpora still grow
	// naturally past this initial capacity.
	samples := make([]Sample, 0, 64)
	// Hoist the record buffer out of the loop, and zero each string
	// field by hand before every Decode call — json.Decode leaves keys
	// absent from the current row untouched, so a hoisted record would
	// otherwise carry the previous row's values forward. The slice
	// fields (Messages, Conversations) are reset to length 0 in-place so
	// the backing array is kept across rows of the same shape. msgBuf
	// reuses the []inference.Message backing across openai/sharegpt
	// rows — nothing downstream retains the slice past the row that
	// built it, so reuse is safe.
	var record jsonRecord
	var msgBuf []inference.Message
	// recordNo numbers non-empty input records — empty/whitespace-only
	// lines do not bump it. Error messages name "record N" for that
	// reason.
	recordNo := 0
	for dec.More() {
		recordNo++
		record.Text = ""
		record.Prompt = ""
		record.Response = ""
		record.Completion = ""
		record.Instruction = ""
		record.Input = ""
		record.Output = ""
		record.Problem = ""
		record.Question = ""
		record.Thinking = ""
		record.Reasoning = ""
		record.Solution = ""
		record.Answer = ""
		record.Messages = record.Messages[:0]
		record.Conversations = record.Conversations[:0]
		if err := dec.Decode(&record); err != nil {
			return nil, core.Errorf("dataset: parse JSONL record %d: %w", recordNo, err)
		}
		sample, ok, err := record.toSample(&msgBuf)
		if err != nil {
			return nil, core.Errorf("dataset: normalise JSONL record %d: %w", recordNo, err)
		}
		if ok {
			samples = append(samples, sample)
		}
	}
	// samples was built locally — every entry's Labels/Messages were
	// constructed fresh by labelled()/MessagesToSample. The slice is
	// owned by the dataset, so a defensive CloneSamples pass here would
	// be pure duplication. Hand off the freshly built slice directly.
	return &JSONLDataset{samples: samples}, nil
}

// NewJSONL returns a replayable dataset from already-normalised samples.
//
//	d := dataset.NewJSONL(samples)
func NewJSONL(samples []Sample) *JSONLDataset {
	return &JSONLDataset{samples: CloneSamples(samples)}
}

// Next returns the next normalised sample.
func (d *JSONLDataset) Next() (Sample, bool, error) {
	if d == nil {
		return Sample{}, false, errJSONLDatasetNil
	}
	if d.index >= len(d.samples) {
		return Sample{}, false, nil
	}
	sample := CloneSample(d.samples[d.index])
	d.index++
	return sample, true, nil
}

// Reset rewinds the replayable dataset.
func (d *JSONLDataset) Reset() error {
	if d == nil {
		return errJSONLDatasetNil
	}
	d.index = 0
	return nil
}

// Samples returns a defensive copy of all normalized samples.
//
//	samples := d.Samples()
func (d *JSONLDataset) Samples() []Sample {
	if d == nil {
		return nil
	}
	return CloneSamples(d.samples)
}

// toSample normalises a parsed jsonRecord. msgBuf is an optional pointer to
// a reusable []inference.Message backing array for the openai/sharegpt
// branches — pass nil when no reuse is available. The helpers write back
// through *msgBuf so a grown backing array is captured for the next row.
//
// Pointer receiver — jsonRecord is 14 fields; a value receiver would copy
// the whole struct into the callee's frame on every row. The pointer is
// read-only inside the method (r.* is never mutated), so call-site
// semantics are identical to a value receiver.
func (r *jsonRecord) toSample(msgBuf *[]inference.Message) (Sample, bool, error) {
	if text := core.Trim(r.Text); text != "" {
		return labelled(Sample{Text: text}, "text"), true, nil
	}
	if len(r.Messages) > 0 {
		return MessagesToSample(appendMessagesFromOpenAI(msgBuf, r.Messages), "openai_messages")
	}
	if len(r.Conversations) > 0 {
		return MessagesToSample(appendMessagesFromShareGPT(msgBuf, r.Conversations), "sharegpt")
	}
	// Trim each candidate once per row. Branch order matches frequency:
	// prompt-response, alpaca, reasoning.
	if prompt := core.Trim(r.Prompt); prompt != "" {
		return labelled(Sample{
			Prompt:   prompt,
			Response: firstNonEmpty(r.Response, r.Completion),
		}, "prompt_response"), true, nil
	}
	if response := firstNonEmpty(r.Response, r.Completion); response != "" {
		return labelled(Sample{
			Response: response,
		}, "prompt_response"), true, nil
	}
	if output := core.Trim(r.Output); core.Trim(r.Instruction) != "" || output != "" {
		return labelled(Sample{
			Prompt:   formatInstructionPrompt(r.Instruction, r.Input),
			Response: output,
		}, "alpaca"), true, nil
	}
	if problem := firstNonEmpty(r.Problem, r.Question); problem != "" {
		return labelled(Sample{
			Prompt:    problem,
			Reasoning: firstNonEmpty(r.Thinking, r.Reasoning),
			Response:  firstNonEmpty(r.Solution, r.Answer),
		}, "reasoning"), true, nil
	}
	if solution := firstNonEmpty(r.Solution, r.Answer); solution != "" {
		return labelled(Sample{
			Reasoning: firstNonEmpty(r.Thinking, r.Reasoning),
			Response:  solution,
		}, "reasoning"), true, nil
	}
	return Sample{}, false, nil
}

// appendMessagesFromOpenAI fills *buf with normalised messages from
// records, writing back through buf so a grown backing array is captured
// for the next call. When buf is nil (no reuse available) the slice is
// allocated fresh; otherwise the existing backing is reset in place if cap
// is sufficient. Pass a reusable buffer (typical: one per LoadJSONL call)
// to avoid a per-row slice allocation.
func appendMessagesFromOpenAI(buf *[]inference.Message, records []messageRecord) []inference.Message {
	out := claimMessageBuf(buf, len(records))
	for _, record := range records {
		// Short-circuit empty rows before the Trim/NormaliseRole work —
		// JSON unmarshal leaves missing fields as "" so this is a hot
		// skip for sparse messages.
		if record.Role == "" && record.Content == "" {
			continue
		}
		role := NormaliseRole(record.Role)
		content := core.Trim(record.Content)
		if role == "" && content == "" {
			continue
		}
		out = append(out, inference.Message{Role: role, Content: content})
	}
	if buf != nil {
		*buf = out
	}
	return out
}

// appendMessagesFromShareGPT mirrors appendMessagesFromOpenAI for the
// ShareGPT-shape record (from/value rather than role/content).
func appendMessagesFromShareGPT(buf *[]inference.Message, records []shareGPTRecord) []inference.Message {
	out := claimMessageBuf(buf, len(records))
	for _, record := range records {
		if record.From == "" && record.Value == "" {
			continue
		}
		role := NormaliseRole(record.From)
		content := core.Trim(record.Value)
		if role == "" && content == "" {
			continue
		}
		out = append(out, inference.Message{Role: role, Content: content})
	}
	if buf != nil {
		*buf = out
	}
	return out
}

// claimMessageBuf returns an empty slice with at least n capacity, reusing
// *buf's backing array when possible.
func claimMessageBuf(buf *[]inference.Message, n int) []inference.Message {
	if buf == nil {
		return make([]inference.Message, 0, n)
	}
	if cap(*buf) < n {
		return make([]inference.Message, 0, n)
	}
	return (*buf)[:0]
}

// MessagesToSample converts a message list into a normalised Sample. When a
// trailing assistant turn is present, it becomes Response (trimmed) and the
// preceding turns are joined into Prompt; otherwise there is no reply to
// hold out as a training target, so the whole list is joined into Text
// instead. Either way the normalised turns are also retained verbatim in
// Sample.Messages, so a caller can apply a real, model-specific chat
// template downstream without re-parsing the source corpus — see the
// package doc for why that template rendering does not happen here.
//
//	sample, ok, err := dataset.MessagesToSample(messages, "sharegpt")
func MessagesToSample(messages []inference.Message, format string) (Sample, bool, error) {
	if len(messages) == 0 {
		return Sample{}, false, nil
	}
	// The internal LoadJSONL path feeds MessagesToSample already-
	// normalised Role values (appendMessagesFromOpenAI/ShareGPT both run
	// NormaliseRole before assembling the slice), so most scans hit the
	// direct-compare fast path with zero NormaliseRole call overhead.
	// NormaliseRole stays as the fallback for external callers passing
	// un-normalised roles ("gpt", "bot", "MODEL") so the public contract
	// is unchanged.
	assistantIdx := -1
	for i, message := range slices.Backward(messages) {
		role := message.Role
		if role == "assistant" || NormaliseRole(role) == "assistant" {
			assistantIdx = i
			break
		}
	}
	if assistantIdx < 0 {
		sample := Sample{
			Text:     joinPlain(messages),
			Messages: cloneMessages(messages),
		}
		return labelled(sample, format), true, nil
	}
	response := core.Trim(messages[assistantIdx].Content)
	sample := Sample{
		Prompt:   joinPlain(messages[:assistantIdx]),
		Response: response,
		Messages: cloneMessages(messages),
	}
	return labelled(sample, format), true, nil
}

// joinPlain renders messages as a role-free transcript: each message's
// non-empty Content on its own line. This mirrors the neutral fallback
// go-mlx's own chat-template formatter degrades to whenever no
// model-family template happens to be registered in the same binary — see
// the package doc.
func joinPlain(messages []inference.Message) string {
	if len(messages) == 0 {
		return ""
	}
	n := 0
	for _, msg := range messages {
		n += len(msg.Content) + len("\n")
	}
	var sb core.Builder
	sb.Grow(n)
	for _, msg := range messages {
		if msg.Content == "" {
			continue
		}
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	return sb.String()
}

func labelled(sample Sample, format string) Sample {
	sample.Format = format
	return sample
}

func formatInstructionPrompt(instruction, input string) string {
	instruction = core.Trim(instruction)
	input = core.Trim(input)
	if instruction == "" {
		return input
	}
	if input == "" {
		return instruction
	}
	return instruction + "\n\n" + input
}

// firstNonEmpty returns the first of (a, b) with a non-empty trimmed form,
// already trimmed. All callers pass exactly two strings, so the fixed-arity
// form skips variadic-slice materialisation.
func firstNonEmpty(a, b string) string {
	if trimmed := core.Trim(a); trimmed != "" {
		return trimmed
	}
	return core.Trim(b)
}

// NormaliseRole canonicalises the free-form role strings found in JSONL
// training corpora (OpenAI, ShareGPT, Alpaca-family exports) across the HF /
// ShareGPT / Llama / Gemma naming variations. Empty input returns empty.
//
//	dataset.NormaliseRole("gpt")   // "assistant"
//	dataset.NormaliseRole("human") // "user"
func NormaliseRole(role string) string {
	// Canonical fast path — messages built by appendMessagesFromOpenAI/
	// ShareGPT already carry canonical role names on a second pass (e.g.
	// via MessagesToSample's fallback check), so this hits without a
	// Lower/Trim/switch table walk.
	switch role {
	case "user", "assistant", "system":
		return role
	}
	return normaliseRoleSlow(role)
}

func normaliseRoleSlow(role string) string {
	// Trim is alloc-free (it returns a sub-slice of role). Match known
	// aliases case-insensitively on the trimmed form via EqualFold —
	// every known-alias branch returns a compile-time literal, so
	// lowering the input first would allocate a string purely to drive
	// the switch and then discard it. Only the unknown-role fallthrough
	// actually returns the canonicalised input, so that is the one
	// branch that pays for core.Lower.
	trimmed := core.Trim(role)
	switch {
	case core.EqualFold(trimmed, "human"), core.EqualFold(trimmed, "user"):
		return "user"
	case core.EqualFold(trimmed, "gpt"), core.EqualFold(trimmed, "bot"),
		core.EqualFold(trimmed, "assistant"), core.EqualFold(trimmed, "model"):
		return "assistant"
	case core.EqualFold(trimmed, "system"), core.EqualFold(trimmed, "developer"):
		return "system"
	default:
		return core.Lower(trimmed)
	}
}
