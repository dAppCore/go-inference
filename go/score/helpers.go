// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// ReadScorerOutput reads a scorer output JSON file and returns the parsed
// *ScorerOutput in the Result value.
//
//	r := ReadScorerOutput("scores.json")
//	if !r.OK { return r }
//	out := r.Value.(*ScorerOutput)
func ReadScorerOutput(path string) core.Result {
	data, err := coreio.Local.Read(path)
	if err != nil {
		return core.Fail(core.E("score.ReadScorerOutput", core.Sprintf("read %s", path), err))
	}

	var output ScorerOutput
	if r := core.JSONUnmarshalString(data, &output); !r.OK {
		return core.Fail(core.E("score.ReadScorerOutput", core.Sprintf("unmarshal %s", path), r.Value.(error)))
	}

	return core.Ok(&output)
}

// WriteScores writes a scorer output struct to a JSON file (indented).
//
//	r := WriteScores("scores.json", out)
//	if !r.OK { return r }
func WriteScores(path string, output *ScorerOutput) core.Result {
	r := core.JSONMarshalIndent(output, "", "  ")
	if !r.OK {
		return core.Fail(core.E("score.WriteScores", "marshal scores", r.Value.(error)))
	}

	if err := coreio.Local.Write(path, string(r.Value.([]byte))); err != nil {
		return core.Fail(core.E("score.WriteScores", core.Sprintf("write %s", path), err))
	}

	return core.Ok(nil)
}

// isErrorResponse reports whether the response should be treated as an error
// prefix regardless of case or leading whitespace.
func isErrorResponse(s string) bool {
	return core.HasPrefix(core.Lower(core.Trim(s)), "error")
}

// strVal extracts a string value from a row map, returning "" when absent or
// not a string.
func strVal(row map[string]any, key string) string {
	v, ok := row[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}

// toInt coerces a numeric any (int64/int32/float64) to int, returning 0 for
// anything else.
func toInt(v any) int {
	switch n := v.(type) {
	case int64:
		return int(n)
	case int32:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}
