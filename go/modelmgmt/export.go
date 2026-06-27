package modelmgmt

import (
	"bufio"
	"math/rand"

	"dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/score"
	coreio "dappco.re/go/io"
)

// TrainingExample is a single training example in chat JSONL format. Messages
// use inference.Message (the shared chat-message type) rather than a parallel
// ChatMessage struct.
type TrainingExample struct {
	Messages []inference.Message `json:"messages"`
}

// ValidatePercentages checks that train+valid+test percentages sum to 100
// and that none are negative.
//
//	r := modelmgmt.ValidatePercentages(80, 10, 10)
//	if !r.OK { return r }
func ValidatePercentages(trainPct, validPct, testPct int) core.Result {
	if trainPct < 0 || validPct < 0 || testPct < 0 {
		return core.Fail(core.E("modelmgmt.ValidatePercentages", core.Sprintf("percentages must be non-negative: train=%d, valid=%d, test=%d", trainPct, validPct, testPct), nil))
	}
	sum := trainPct + validPct + testPct
	if sum != 100 {
		return core.Fail(core.E("modelmgmt.ValidatePercentages", core.Sprintf("percentages must sum to 100, got %d (train=%d + valid=%d + test=%d)", sum, trainPct, validPct, testPct), nil))
	}
	return core.Ok(nil)
}

// FilterResponses removes responses with empty content, "ERROR:" prefix,
// or response length < 50 characters.
func FilterResponses(responses []score.Response) []score.Response {
	var filtered []score.Response
	for _, r := range responses {
		if r.Response == "" {
			continue
		}
		if core.HasPrefix(r.Response, "ERROR:") {
			continue
		}
		if len(r.Response) < 50 {
			continue
		}
		filtered = append(filtered, r)
	}
	return filtered
}

// SplitData shuffles responses with a deterministic seed and splits them
// into train, valid, and test sets by the given percentages.
func SplitData(responses []score.Response, trainPct, validPct, testPct int, seed int64) (train, valid, test []score.Response) {
	shuffled := make([]score.Response, len(responses))
	copy(shuffled, responses)

	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	n := len(shuffled)
	trainN := n * trainPct / 100
	validN := n * validPct / 100
	_ = testPct

	train = shuffled[:trainN]
	valid = shuffled[trainN : trainN+validN]
	test = shuffled[trainN+validN:]

	return train, valid, test
}

// WriteTrainingJSONL writes responses in chat JSONL format suitable for
// MLX LoRA fine-tuning.
//
//	r := modelmgmt.WriteTrainingJSONL("/data/train.jsonl", responses)
//	if !r.OK { return r }
func WriteTrainingJSONL(path string, responses []score.Response) core.Result {
	f, err := coreio.Local.Create(path)
	if err != nil {
		return core.Fail(core.E("modelmgmt.WriteTrainingJSONL", core.Sprintf("create %s", path), err))
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	for _, r := range responses {
		example := TrainingExample{
			Messages: []inference.Message{
				{Role: "user", Content: r.Prompt},
				{Role: "assistant", Content: r.Response},
			},
		}

		if _, err := w.WriteString(core.JSONMarshalString(example)); err != nil {
			return core.Fail(core.E("modelmgmt.WriteTrainingJSONL", "write line", err))
		}
		if _, err := w.WriteString("\n"); err != nil {
			return core.Fail(core.E("modelmgmt.WriteTrainingJSONL", "write newline", err))
		}
	}

	return core.Ok(nil)
}
