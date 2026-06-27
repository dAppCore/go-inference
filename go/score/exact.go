package score

import (
	"math"
	"regexp"
	"strconv"

	"dappco.re/go"
)

// Pre-compiled regex patterns for GSM8K answer extraction.
var (
	// hashAnswer matches the #### delimiter pattern used in GSM8K.
	hashAnswer = regexp.MustCompile(`####\s*([\d,.\-]+)`)

	// lastNumber matches the last number in a response.
	lastNumber = regexp.MustCompile(`(?:^|[\s=])(-?[\d,]+(?:\.\d+)?)`)
)

// ScoreExact returns 1.0 when the response matches the correct answer after
// GSM8K-style numeric extraction, 0.0 otherwise. Strings are compared after
// trimming whitespace; numbers are compared within an epsilon of 0.01.
//
//	score := ml.ScoreExact("The answer is 42", "42")   // 1.0
//	score := ml.ScoreExact("I don't know", "42")       // 0.0
func ScoreExact(response, correctAnswer string) float64 {
	std := scoreGSM8K(response, correctAnswer)
	if std.Correct != nil && *std.Correct {
		return 1.0
	}
	// Fall back to plain string equality when numeric extraction fails.
	if core.Trim(response) != "" && core.Trim(response) == core.Trim(correctAnswer) {
		return 1.0
	}
	return 0.0
}

// scoreGSM8K extracts a numeric answer from a model response and compares
// it to the correct answer using exact match (within epsilon of 0.01).
func scoreGSM8K(response, correctAnswer string) *StandardScores {
	correct := false

	// Empty or error response.
	if response == "" || isErrorResponse(response) {
		return &StandardScores{
			Correct:   &correct,
			Extracted: "",
			Expected:  correctAnswer,
		}
	}

	// Try #### delimiter first.
	var extracted string
	if m := hashAnswer.FindStringSubmatch(response); len(m) > 1 {
		extracted = m[1]
	} else {
		// Find the last number in the response.
		matches := lastNumber.FindAllStringSubmatch(response, -1)
		if len(matches) > 0 {
			extracted = matches[len(matches)-1][1]
		}
	}

	// No number found.
	if extracted == "" {
		return &StandardScores{
			Correct:   &correct,
			Extracted: "",
			Expected:  correctAnswer,
		}
	}

	// Clean commas and parse both numbers.
	cleanExtracted := core.Trim(core.Replace(extracted, ",", ""))
	cleanExpected := core.Trim(core.Replace(correctAnswer, ",", ""))

	extVal, errExt := strconv.ParseFloat(cleanExtracted, 64)
	expVal, errExp := strconv.ParseFloat(cleanExpected, 64)

	if errExt != nil || errExp != nil {
		return &StandardScores{
			Correct:   &correct,
			Extracted: extracted,
			Expected:  correctAnswer,
		}
	}

	correct = math.Abs(expVal-extVal) <= 0.01

	return &StandardScores{
		Correct:   &correct,
		Extracted: extracted,
		Expected:  correctAnswer,
	}
}
