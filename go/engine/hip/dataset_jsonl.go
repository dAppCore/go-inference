// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"bufio"
	"encoding/json"
	"io"
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

var (
	errROCmDatasetReaderNil = core.NewError("rocm: dataset reader is nil")
	errROCmDatasetNil       = core.NewError("rocm: JSONL dataset is nil")
)

// JSONLDataset is a replayable in-memory DatasetStream loaded from JSONL.
type JSONLDataset struct {
	samples []inference.DatasetSample
	index   int
}

type rocmJSONLRecord struct {
	Text                  string               `json:"text"`
	Prompt                string               `json:"prompt"`
	Response              string               `json:"response"`
	Completion            string               `json:"completion"`
	Instruction           string               `json:"instruction"`
	Input                 string               `json:"input"`
	Output                string               `json:"output"`
	Problem               string               `json:"problem"`
	Question              string               `json:"question"`
	Thinking              string               `json:"thinking"`
	Reasoning             string               `json:"reasoning"`
	Solution              string               `json:"solution"`
	Answer                string               `json:"answer"`
	Messages              []rocmMessageRecord  `json:"messages"`
	Conversations         []rocmShareGPTRecord `json:"conversations"`
	Labels                map[string]string    `json:"labels"`
	TargetTokenID         any                  `json:"target_token_id"`
	StudentLogits         any                  `json:"student_logits"`
	TeacherLogits         any                  `json:"teacher_logits"`
	Reward                any                  `json:"reward"`
	Rewards               any                  `json:"rewards"`
	Advantage             any                  `json:"advantage"`
	Advantages            any                  `json:"advantages"`
	Logprob               any                  `json:"logprob"`
	Logprobs              any                  `json:"logprobs"`
	PolicyLogprob         any                  `json:"policy_logprob"`
	PolicyLogprobs        any                  `json:"policy_logprobs"`
	CurrentLogprob        any                  `json:"current_logprob"`
	CurrentLogprobs       any                  `json:"current_logprobs"`
	CurrentPolicyLogprob  any                  `json:"current_policy_logprob"`
	CurrentPolicyLogprobs any                  `json:"current_policy_logprobs"`
	OldLogprob            any                  `json:"old_logprob"`
	OldLogprobs           any                  `json:"old_logprobs"`
	OldPolicyLogprob      any                  `json:"old_policy_logprob"`
	OldPolicyLogprobs     any                  `json:"old_policy_logprobs"`
	ReferenceLogprob      any                  `json:"reference_logprob"`
	ReferenceLogprobs     any                  `json:"reference_logprobs"`
	RefLogprob            any                  `json:"ref_logprob"`
	RefLogprobs           any                  `json:"ref_logprobs"`
	PolicyClipRange       any                  `json:"policy_clip_range"`
	ClipRange             any                  `json:"clip_range"`
	ClipEpsilon           any                  `json:"clip_epsilon"`
	GRPOClipRange         any                  `json:"grpo_clip_range"`
	PolicyWeight          any                  `json:"policy_weight"`
	PolicyWeights         any                  `json:"policy_weights"`
	LossWeight            any                  `json:"loss_weight"`
	LossWeights           any                  `json:"loss_weights"`
	PolicyMask            any                  `json:"policy_mask"`
	PolicyMasks           any                  `json:"policy_masks"`
	LossMask              any                  `json:"loss_mask"`
	LossMasks             any                  `json:"loss_masks"`
	ResponseMask          any                  `json:"response_mask"`
	ResponseMasks         any                  `json:"response_masks"`
	ActionMask            any                  `json:"action_mask"`
	ActionMasks           any                  `json:"action_masks"`
	TokenMask             any                  `json:"token_mask"`
	TokenMasks            any                  `json:"token_masks"`
	GroupID               any                  `json:"group_id"`
	PromptID              any                  `json:"prompt_id"`
	QueryID               any                  `json:"query_id"`
	RolloutID             any                  `json:"rollout_id"`
	SampleID              any                  `json:"sample_id"`
	TrajectoryID          any                  `json:"trajectory_id"`
	TurnID                any                  `json:"turn_id"`
	CompletionID          any                  `json:"completion_id"`
	EpisodeID             any                  `json:"episode_id"`
	Meta                  map[string]string    `json:"meta"`
	Raw                   map[string]any       `json:"-"`
}

type rocmMessageRecord struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type rocmShareGPTRecord struct {
	From  string `json:"from"`
	Value string `json:"value"`
}

// LoadJSONLDataset reads JSONL rows into a replayable go-inference dataset.
func LoadJSONLDataset(reader io.Reader) (*JSONLDataset, error) {
	if reader == nil {
		return nil, errROCmDatasetReaderNil
	}
	decoder := json.NewDecoder(bufio.NewReaderSize(reader, 64*1024))
	samples := make([]inference.DatasetSample, 0, 64)
	var record rocmJSONLRecord
	var messageBuf []inference.Message
	recordNo := 0
	for {
		resetROCmJSONLRecord(&record)
		if err := decoder.Decode(&record); err != nil {
			if err == io.EOF {
				break
			}
			return nil, core.Errorf("rocm: parse JSONL record %d: %w", recordNo+1, err)
		}
		recordNo++
		sample, ok := record.toDatasetSample(&messageBuf)
		if ok {
			samples = append(samples, sample)
		}
	}
	return &JSONLDataset{samples: samples}, nil
}

// NewJSONLDataset returns a replayable dataset from already-normalised samples.
func NewJSONLDataset(samples []inference.DatasetSample) *JSONLDataset {
	return &JSONLDataset{samples: cloneDatasetSamples(samples)}
}

// Next returns the next sample.
func (dataset *JSONLDataset) Next() (inference.DatasetSample, bool, error) {
	if dataset == nil {
		return inference.DatasetSample{}, false, errROCmDatasetNil
	}
	if dataset.index >= len(dataset.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := cloneDatasetSample(dataset.samples[dataset.index])
	dataset.index++
	return sample, true, nil
}

// Reset rewinds the dataset.
func (dataset *JSONLDataset) Reset() error {
	if dataset == nil {
		return errROCmDatasetNil
	}
	dataset.index = 0
	return nil
}

// Remaining returns the number of samples left in the current replay pass.
func (dataset *JSONLDataset) Remaining() int {
	if dataset == nil || dataset.index >= len(dataset.samples) {
		return 0
	}
	return len(dataset.samples) - dataset.index
}

// Samples returns a defensive copy of all samples.
func (dataset *JSONLDataset) Samples() []inference.DatasetSample {
	if dataset == nil {
		return nil
	}
	return cloneDatasetSamples(dataset.samples)
}

func resetROCmJSONLRecord(record *rocmJSONLRecord) {
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
	record.Labels = nil
	record.TargetTokenID = nil
	record.StudentLogits = nil
	record.TeacherLogits = nil
	record.Reward = nil
	record.Rewards = nil
	record.Advantage = nil
	record.Advantages = nil
	record.Logprob = nil
	record.Logprobs = nil
	record.PolicyLogprob = nil
	record.PolicyLogprobs = nil
	record.CurrentLogprob = nil
	record.CurrentLogprobs = nil
	record.CurrentPolicyLogprob = nil
	record.CurrentPolicyLogprobs = nil
	record.OldLogprob = nil
	record.OldLogprobs = nil
	record.OldPolicyLogprob = nil
	record.OldPolicyLogprobs = nil
	record.ReferenceLogprob = nil
	record.ReferenceLogprobs = nil
	record.RefLogprob = nil
	record.RefLogprobs = nil
	record.PolicyClipRange = nil
	record.ClipRange = nil
	record.ClipEpsilon = nil
	record.GRPOClipRange = nil
	record.PolicyWeight = nil
	record.PolicyWeights = nil
	record.LossWeight = nil
	record.LossWeights = nil
	record.PolicyMask = nil
	record.PolicyMasks = nil
	record.LossMask = nil
	record.LossMasks = nil
	record.ResponseMask = nil
	record.ResponseMasks = nil
	record.ActionMask = nil
	record.ActionMasks = nil
	record.TokenMask = nil
	record.TokenMasks = nil
	record.GroupID = nil
	record.PromptID = nil
	record.QueryID = nil
	record.RolloutID = nil
	record.SampleID = nil
	record.TrajectoryID = nil
	record.TurnID = nil
	record.CompletionID = nil
	record.EpisodeID = nil
	record.Meta = nil
	record.Raw = nil
}

func (record *rocmJSONLRecord) toDatasetSample(messageBuf *[]inference.Message) (inference.DatasetSample, bool) {
	labels := recordLabels(record)
	if text := core.Trim(record.Text); text != "" {
		return labelDatasetSample(inference.DatasetSample{Text: text, Labels: labels}, "text"), true
	}
	if len(record.Messages) > 0 {
		messages := appendROCmMessages(messageBuf, record.Messages)
		return messagesDatasetSample(messages, labels, "openai_messages")
	}
	if len(record.Conversations) > 0 {
		messages := appendROCmShareGPTMessages(messageBuf, record.Conversations)
		return messagesDatasetSample(messages, labels, "sharegpt")
	}
	if prompt := core.Trim(record.Prompt); prompt != "" {
		return labelDatasetSample(inference.DatasetSample{
			Prompt:   prompt,
			Response: datasetFirstNonEmptyString(record.Response, record.Completion),
			Labels:   labels,
		}, "prompt_response"), true
	}
	if response := datasetFirstNonEmptyString(record.Response, record.Completion); response != "" {
		return labelDatasetSample(inference.DatasetSample{Response: response, Labels: labels}, "prompt_response"), true
	}
	if output := core.Trim(record.Output); core.Trim(record.Instruction) != "" || output != "" {
		return labelDatasetSample(inference.DatasetSample{
			Prompt:   formatInstructionPrompt(record.Instruction, record.Input),
			Response: output,
			Labels:   labels,
		}, "alpaca"), true
	}
	if problem := datasetFirstNonEmptyString(record.Problem, record.Question); problem != "" {
		return labelDatasetSample(inference.DatasetSample{
			Prompt:   problem,
			Response: formatReasoningResponse(datasetFirstNonEmptyString(record.Thinking, record.Reasoning), datasetFirstNonEmptyString(record.Solution, record.Answer)),
			Labels:   labels,
		}, "reasoning"), true
	}
	if solution := datasetFirstNonEmptyString(record.Solution, record.Answer); solution != "" {
		return labelDatasetSample(inference.DatasetSample{
			Response: formatReasoningResponse(datasetFirstNonEmptyString(record.Thinking, record.Reasoning), solution),
			Labels:   labels,
		}, "reasoning"), true
	}
	if len(labels) > 0 {
		return labelDatasetSample(inference.DatasetSample{Labels: labels}, "labels"), true
	}
	return inference.DatasetSample{}, false
}

func recordLabels(record *rocmJSONLRecord) map[string]string {
	labels := cloneStringMap(record.Meta)
	if len(record.Labels) > 0 {
		if labels == nil {
			labels = make(map[string]string, len(record.Labels)+3)
		}
		for key, value := range record.Labels {
			if trimmedKey := core.Trim(key); trimmedKey != "" {
				labels[trimmedKey] = value
			}
		}
	}
	labels = addAnyLabel(labels, "target_token_id", record.TargetTokenID)
	labels = addAnyLabel(labels, "student_logits", record.StudentLogits)
	labels = addAnyLabel(labels, "teacher_logits", record.TeacherLogits)
	labels = addAnyLabel(labels, "reward", record.Reward)
	labels = addAnyLabel(labels, "rewards", record.Rewards)
	labels = addAnyLabel(labels, "advantage", record.Advantage)
	labels = addAnyLabel(labels, "advantages", record.Advantages)
	labels = addFirst4AnyLabel(labels, "logprob", record.Logprob, record.PolicyLogprob, record.CurrentLogprob, record.CurrentPolicyLogprob)
	labels = addFirst4AnyLabel(labels, "logprobs", record.Logprobs, record.PolicyLogprobs, record.CurrentLogprobs, record.CurrentPolicyLogprobs)
	labels = addFirst2AnyLabel(labels, "old_logprob", record.OldLogprob, record.OldPolicyLogprob)
	labels = addFirst2AnyLabel(labels, "old_logprobs", record.OldLogprobs, record.OldPolicyLogprobs)
	labels = addFirst2AnyLabel(labels, "reference_logprob", record.ReferenceLogprob, record.RefLogprob)
	labels = addFirst2AnyLabel(labels, "reference_logprobs", record.ReferenceLogprobs, record.RefLogprobs)
	labels = addFirst4AnyLabel(labels, "policy_clip_range", record.PolicyClipRange, record.ClipRange, record.ClipEpsilon, record.GRPOClipRange)
	labels = addFirst8AnyLabel(labels, "policy_weight", record.PolicyWeight, record.LossWeight, record.PolicyMask, record.LossMask, record.ResponseMask, record.ActionMask, record.TokenMask, nil)
	labels = addFirst8AnyLabel(labels, "policy_weights", record.PolicyWeights, record.LossWeights, record.PolicyMasks, record.LossMasks, record.ResponseMasks, record.ActionMasks, record.TokenMasks, nil)
	labels = addAnyLabel(labels, "group_id", record.GroupID)
	labels = addAnyLabel(labels, "prompt_id", record.PromptID)
	labels = addAnyLabel(labels, "query_id", record.QueryID)
	labels = addAnyLabel(labels, "rollout_id", record.RolloutID)
	labels = addAnyLabel(labels, "sample_id", record.SampleID)
	labels = addAnyLabel(labels, "trajectory_id", record.TrajectoryID)
	labels = addAnyLabel(labels, "turn_id", record.TurnID)
	labels = addAnyLabel(labels, "completion_id", record.CompletionID)
	labels = addAnyLabel(labels, "episode_id", record.EpisodeID)
	return labels
}

func addAnyLabel(labels map[string]string, key string, value any) map[string]string {
	out, _ := addAnyLabelOK(labels, key, value)
	return out
}

func addAnyLabelOK(labels map[string]string, key string, value any) (map[string]string, bool) {
	text, ok := anyLabelString(value)
	if !ok {
		return labels, false
	}
	if labels == nil {
		labels = make(map[string]string, 4)
	}
	labels[key] = text
	return labels, true
}

func addFirst2AnyLabel(labels map[string]string, key string, a, b any) map[string]string {
	if out, ok := addAnyLabelOK(labels, key, a); ok {
		return out
	}
	return addAnyLabel(labels, key, b)
}

func addFirst4AnyLabel(labels map[string]string, key string, a, b, c, d any) map[string]string {
	if out, ok := addAnyLabelOK(labels, key, a); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, b); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, c); ok {
		return out
	}
	return addAnyLabel(labels, key, d)
}

func addFirst8AnyLabel(labels map[string]string, key string, a, b, c, d, e, f, g, h any) map[string]string {
	if out, ok := addAnyLabelOK(labels, key, a); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, b); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, c); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, d); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, e); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, f); ok {
		return out
	}
	if out, ok := addAnyLabelOK(labels, key, g); ok {
		return out
	}
	return addAnyLabel(labels, key, h)
}

func anyLabelString(value any) (string, bool) {
	switch typed := value.(type) {
	case nil:
		return "", false
	case string:
		text := core.Trim(typed)
		return text, text != ""
	case float64:
		return strconv.FormatFloat(typed, 'f', -1, 64), true
	case bool:
		return strconv.FormatBool(typed), true
	case []any:
		if len(typed) == 0 {
			return "", false
		}
		builder := core.NewBuilder()
		for i, item := range typed {
			text, ok := anyLabelString(item)
			if !ok {
				return "", false
			}
			if i > 0 {
				builder.WriteString(",")
			}
			builder.WriteString(text)
		}
		return builder.String(), true
	default:
		return core.Sprintf("%v", typed), true
	}
}

func appendROCmMessages(buf *[]inference.Message, records []rocmMessageRecord) []inference.Message {
	out := claimROCmMessageBuf(buf, len(records))
	for _, record := range records {
		if record.Role == "" && record.Content == "" {
			continue
		}
		role := normalizeDatasetRole(record.Role)
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

func appendROCmShareGPTMessages(buf *[]inference.Message, records []rocmShareGPTRecord) []inference.Message {
	out := claimROCmMessageBuf(buf, len(records))
	for _, record := range records {
		if record.From == "" && record.Value == "" {
			continue
		}
		role := normalizeDatasetRole(record.From)
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

func claimROCmMessageBuf(buf *[]inference.Message, n int) []inference.Message {
	if buf == nil || cap(*buf) < n {
		return make([]inference.Message, 0, n)
	}
	return (*buf)[:0]
}

func messagesDatasetSample(messages []inference.Message, labels map[string]string, format string) (inference.DatasetSample, bool) {
	if len(messages) == 0 {
		return inference.DatasetSample{}, false
	}
	assistantIdx := -1
	for i, message := range slices.Backward(messages) {
		if normalizeDatasetRole(message.Role) == "assistant" {
			assistantIdx = i
			break
		}
	}
	if assistantIdx < 0 {
		return labelDatasetSample(inference.DatasetSample{
			Messages: cloneMessages(messages),
			Labels:   labels,
		}, format), true
	}
	return labelDatasetSample(inference.DatasetSample{
		Messages: cloneMessages(messages[:assistantIdx]),
		Response: core.Trim(messages[assistantIdx].Content),
		Labels:   labels,
	}, format), true
}

func labelDatasetSample(sample inference.DatasetSample, format string) inference.DatasetSample {
	if sample.Labels == nil {
		sample.Labels = make(map[string]string, 1)
	}
	sample.Labels["format"] = format
	return sample
}

func normalizeDatasetRole(role string) string {
	switch strings.ToLower(core.Trim(role)) {
	case "human", "user":
		return "user"
	case "gpt", "bot", "model", "assistant":
		return "assistant"
	case "system":
		return "system"
	default:
		return core.Trim(role)
	}
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

func formatReasoningResponse(thinking, solution string) string {
	thinking = core.Trim(thinking)
	solution = core.Trim(solution)
	if thinking == "" {
		return solution
	}
	if solution == "" {
		return thinking
	}
	return thinking + "\n\n" + solution
}

func datasetFirstNonEmptyString(a, b string) string {
	if trimmed := core.Trim(a); trimmed != "" {
		return trimmed
	}
	return core.Trim(b)
}

func cloneDatasetSamples(samples []inference.DatasetSample) []inference.DatasetSample {
	if len(samples) == 0 {
		return nil
	}
	out := make([]inference.DatasetSample, len(samples))
	for i, sample := range samples {
		out[i] = cloneDatasetSample(sample)
	}
	return out
}

func cloneDatasetSample(sample inference.DatasetSample) inference.DatasetSample {
	sample.Messages = cloneMessages(sample.Messages)
	sample.Labels = cloneStringMap(sample.Labels)
	return sample
}

func cloneMessages(messages []inference.Message) []inference.Message {
	if len(messages) == 0 {
		return nil
	}
	return append([]inference.Message(nil), messages...)
}
