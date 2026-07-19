// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// LQLStatementKind names the LarQL-inspired model query operation.
type LQLStatementKind string

const (
	LQLStatementUse      LQLStatementKind = "use"
	LQLStatementDescribe LQLStatementKind = "describe"
	LQLStatementWalk     LQLStatementKind = "walk"
	LQLStatementSelect   LQLStatementKind = "select"
	LQLStatementInfer    LQLStatementKind = "infer"
	LQLStatementTrace    LQLStatementKind = "trace"
	LQLStatementDiff     LQLStatementKind = "diff"
	LQLStatementCompile  LQLStatementKind = "compile"
	LQLStatementExtract  LQLStatementKind = "extract"
)

// LQLStatement is a parsed, backend-neutral model query. It intentionally keeps
// the raw query text so richer backends can handle clauses this lightweight
// parser does not yet understand.
type LQLStatement struct {
	Kind      LQLStatementKind        `json:"kind"`
	Operation LQLStatementKind        `json:"operation,omitempty"`
	Raw       string                  `json:"raw,omitempty"`
	Target    string                  `json:"target,omitempty"`
	Prompt    string                  `json:"prompt,omitempty"`
	Base      string                  `json:"base,omitempty"`
	Tuned     string                  `json:"tuned,omitempty"`
	Patch     string                  `json:"patch,omitempty"`
	Limit     int                     `json:"limit,omitempty"`
	Model     inference.ModelIdentity `json:"model"`
	Labels    map[string]string       `json:"labels,omitempty"`
}

// LQLResult is the generic result shape for research backends that execute
// model-structure queries.
type LQLResult struct {
	Statement LQLStatement            `json:"statement"`
	Model     inference.ModelIdentity `json:"model"`
	Rows      []map[string]any        `json:"rows,omitempty"`
	Text      string                  `json:"text,omitempty"`
	Labels    map[string]string       `json:"labels,omitempty"`
}

// LQLExecutor is implemented by research backends that can execute LQL over a
// vindex, direct weights, or a split inference runtime.
type LQLExecutor interface {
	ExecuteLQL(context.Context, LQLStatement) (LQLResult, error)
}

// ParseLQL parses one LarQL-inspired query statement. The first supported
// subset covers the research path needed for base/fine-tune comparison and
// walk/trace smoke tests.
func ParseLQL(query string) (LQLStatement, error) {
	raw := core.Trim(query)
	if raw == "" {
		return LQLStatement{}, core.NewError("ml: LQL statement is empty")
	}
	tokens, err := lexLQL(raw)
	if err != nil {
		return LQLStatement{}, err
	}
	if len(tokens) == 0 {
		return LQLStatement{}, core.NewError("ml: LQL statement is empty")
	}
	kind := LQLStatementKind(core.Lower(tokens[0]))
	stmt := LQLStatement{Kind: kind, Raw: raw}
	switch kind {
	case LQLStatementUse:
		if len(tokens) < 2 {
			return LQLStatement{}, core.NewError("ml: USE requires a model or vindex target")
		}
		stmt.Target = tokens[1]
	case LQLStatementDescribe:
		if len(tokens) < 2 {
			return LQLStatement{}, core.NewError("ml: DESCRIBE requires a target")
		}
		stmt.Target = tokens[1]
	case LQLStatementWalk:
		if len(tokens) >= 2 && !lqlTokenIsKeyword(tokens[1]) {
			stmt.Prompt = tokens[1]
			stmt.Target = tokens[1]
		}
		stmt.Limit = lqlLimit(tokens)
	case LQLStatementInfer:
		if len(tokens) >= 2 {
			stmt.Prompt = tokens[1]
			stmt.Target = tokens[1]
		}
		stmt.Limit = lqlLimit(tokens)
	case LQLStatementTrace:
		if len(tokens) < 2 {
			return LQLStatement{}, core.NewError("ml: TRACE requires an operation")
		}
		stmt.Operation = LQLStatementKind(core.Lower(tokens[1]))
		if len(tokens) >= 3 {
			stmt.Prompt = tokens[2]
			stmt.Target = tokens[2]
		}
		stmt.Limit = lqlLimit(tokens)
	case LQLStatementDiff:
		if err := parseLQLDiff(&stmt, tokens); err != nil {
			return LQLStatement{}, err
		}
	case LQLStatementSelect, LQLStatementCompile, LQLStatementExtract:
		stmt.Target = lqlRest(tokens, 1)
		stmt.Limit = lqlLimit(tokens)
	default:
		return LQLStatement{}, core.Errorf("ml: unsupported LQL statement %q", tokens[0])
	}
	return stmt, nil
}

// ParseLQLScript splits a small statement batch on semicolons outside quoted
// strings, ignoring whole-line # and -- comments.
func ParseLQLScript(script string) ([]LQLStatement, error) {
	parts, err := splitLQLScript(script)
	if err != nil {
		return nil, err
	}
	statements := make([]LQLStatement, 0, len(parts))
	for _, part := range parts {
		trimmed := core.Trim(part)
		if trimmed == "" {
			continue
		}
		stmt, err := ParseLQL(trimmed)
		if err != nil {
			return nil, err
		}
		statements = append(statements, stmt)
	}
	return statements, nil
}

func parseLQLDiff(stmt *LQLStatement, tokens []string) error {
	if len(tokens) < 3 {
		return core.NewError("ml: DIFF requires base and tuned model targets")
	}
	for i := 1; i < len(tokens); i++ {
		key := core.Lower(tokens[i])
		switch key {
		case "base":
			if i+1 < len(tokens) {
				stmt.Base = tokens[i+1]
				i++
			}
		case "tuned", "finetune", "fine-tune", "target":
			if i+1 < len(tokens) {
				stmt.Tuned = tokens[i+1]
				i++
			}
		case "with", "against", "to":
			if stmt.Base == "" && i > 1 {
				stmt.Base = tokens[i-1]
			}
			if i+1 < len(tokens) {
				stmt.Tuned = tokens[i+1]
				i++
			}
		case "patch":
			if i+1 < len(tokens) {
				stmt.Patch = tokens[i+1]
				i++
			}
		}
	}
	if stmt.Base == "" && len(tokens) > 1 {
		stmt.Base = tokens[1]
	}
	if stmt.Tuned == "" && len(tokens) > 2 {
		stmt.Tuned = tokens[2]
	}
	stmt.Target = stmt.Tuned
	stmt.Limit = lqlLimit(tokens)
	if stmt.Base == "" || stmt.Tuned == "" {
		return core.NewError("ml: DIFF requires base and tuned model targets")
	}
	return nil
}

func lexLQL(input string) ([]string, error) {
	tokens := []string{}
	token := core.NewBuilder()
	var quote byte
	escaped := false
	for i := 0; i < len(input); i++ {
		ch := input[i]
		if quote != 0 {
			switch {
			case escaped:
				token.WriteByte(ch)
				escaped = false
			case ch == '\\':
				escaped = true
			case ch == quote:
				tokens = append(tokens, token.String())
				token.Reset()
				quote = 0
			default:
				token.WriteByte(ch)
			}
			continue
		}
		switch {
		case ch == '"' || ch == '\'':
			if token.Len() > 0 {
				tokens = append(tokens, token.String())
				token.Reset()
			}
			quote = ch
		case lqlIsSpace(ch) || ch == ';':
			if token.Len() > 0 {
				tokens = append(tokens, token.String())
				token.Reset()
			}
		default:
			token.WriteByte(ch)
		}
	}
	if quote != 0 {
		return nil, core.NewError("ml: unterminated quoted LQL string")
	}
	if token.Len() > 0 {
		tokens = append(tokens, token.String())
	}
	return tokens, nil
}

func splitLQLScript(script string) ([]string, error) {
	lines := core.Split(script, "\n")
	cleaned := core.NewBuilder()
	for _, line := range lines {
		trimmed := core.Trim(line)
		if core.HasPrefix(trimmed, "#") || core.HasPrefix(trimmed, "--") {
			continue
		}
		cleaned.WriteString(line)
		cleaned.WriteByte('\n')
	}
	input := cleaned.String()
	parts := []string{}
	part := core.NewBuilder()
	var quote byte
	escaped := false
	for i := 0; i < len(input); i++ {
		ch := input[i]
		if quote != 0 {
			part.WriteByte(ch)
			switch {
			case escaped:
				escaped = false
			case ch == '\\':
				escaped = true
			case ch == quote:
				quote = 0
			}
			continue
		}
		switch ch {
		case '"', '\'':
			quote = ch
			part.WriteByte(ch)
		case ';':
			parts = append(parts, part.String())
			part.Reset()
		default:
			part.WriteByte(ch)
		}
	}
	if quote != 0 {
		return nil, core.NewError("ml: unterminated quoted LQL string")
	}
	if core.Trim(part.String()) != "" {
		parts = append(parts, part.String())
	}
	return parts, nil
}

func lqlLimit(tokens []string) int {
	for i := 0; i+1 < len(tokens); i++ {
		if core.Lower(tokens[i]) != "limit" {
			continue
		}
		result := core.Atoi(tokens[i+1])
		if result.OK {
			return result.Int()
		}
	}
	return 0
}

func lqlRest(tokens []string, start int) string {
	if start >= len(tokens) {
		return ""
	}
	return core.Join(" ", tokens[start:]...)
}

func lqlTokenIsKeyword(token string) bool {
	switch core.Lower(token) {
	case "from", "where", "limit", "with", "against", "to", "base", "tuned", "target", "patch":
		return true
	default:
		return false
	}
}

func lqlIsSpace(ch byte) bool {
	return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'
}
