// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	core "dappco.re/go"
)

const (
	statusStart = "<<<LEM_STATUS>>>"
	statusEnd   = "<<<END_LEM_STATUS>>>"
)

// ParseFinalStatus validates the single terminal status envelope in a response.
func ParseFinalStatus(text string) core.Result {
	if core.Count(text, statusStart) != 1 || core.Count(text, statusEnd) != 1 {
		return core.Fail(core.NewError("agent provider response requires exactly one complete status envelope"))
	}
	start := core.Index(text, statusStart)
	end := core.Index(text, statusEnd)
	if start < 0 || end < 0 || end <= start+len(statusStart) {
		return core.Fail(core.NewError("agent provider status envelope markers are out of order"))
	}
	if core.Trim(text[end+len(statusEnd):]) != "" {
		return core.Fail(core.NewError("agent provider status envelope must end the response"))
	}
	payload := text[start+len(statusStart) : end]
	var status FinalStatus
	decoded := core.JSONUnmarshalString(payload, &status)
	if !decoded.OK {
		return core.Fail(core.E("provider.ParseFinalStatus", "invalid status envelope JSON", decoded.Err()))
	}
	status.Status = core.Lower(core.Trim(status.Status))
	status.Summary = core.Trim(status.Summary)
	status.Question = core.Trim(status.Question)
	status.Reason = core.Trim(status.Reason)
	switch status.Status {
	case "completed":
		status.Question = ""
		status.Reason = ""
	case "waiting":
		if status.Question == "" {
			return core.Fail(core.NewError("agent provider waiting status requires a question"))
		}
		status.Summary = ""
		status.Reason = ""
	case "failed":
		if status.Reason == "" {
			return core.Fail(core.NewError("agent provider failed status requires a reason"))
		}
		status.Summary = ""
		status.Question = ""
	default:
		return core.Fail(core.Errorf("agent provider status %q is not supported", status.Status))
	}
	return core.Ok(status)
}

func parseProviderLine(provider, stream, line string) []Output {
	if line == "" {
		return nil
	}
	detail := ""
	structured := core.HasPrefix(line, "{") || core.HasPrefix(line, "[")
	if structured {
		detail = line
	}
	if isRateLimit(line) {
		return []Output{{
			Kind:       "rate_limit",
			Text:       line,
			DetailJSON: detail,
			RetryAfter: retryDuration(line),
		}}
	}
	if stream == "stderr" {
		return []Output{{Kind: "stderr", Text: line, DetailJSON: detail}}
	}
	if !structured {
		return []Output{{Kind: "text", Text: line}}
	}

	var event map[string]any
	decoded := core.JSONUnmarshalString(line, &event)
	if !decoded.OK || event == nil {
		return []Output{{Kind: "raw", Text: line, DetailJSON: detail}}
	}
	outputs := providerOutputs(provider, event, detail)
	if len(outputs) == 0 {
		return []Output{{Kind: "raw", Text: line, DetailJSON: detail}}
	}
	return outputs
}

func providerOutputs(provider string, event map[string]any, detail string) []Output {
	eventType, _ := event["type"].(string)
	outputs := make([]Output, 0, 2)
	text := ""
	switch provider {
	case "codex":
		if item, ok := event["item"].(map[string]any); ok {
			text = mapText(item)
		}
		if text == "" {
			text = mapText(event)
		}
	case "claude":
		if message, ok := event["message"].(map[string]any); ok {
			text = mapText(message)
		}
		if text == "" {
			text = mapText(event)
		}
	case "opencode":
		if part, ok := event["part"].(map[string]any); ok {
			text = mapText(part)
		}
		if text == "" {
			text = mapText(event)
		}
	}
	if text != "" {
		outputs = append(outputs, Output{Kind: "text", Text: text, DetailJSON: detail})
	}
	if usage, exists := findMapValue(event, "usage"); exists {
		usageJSON := core.JSONMarshalString(usage)
		outputs = append(outputs, Output{Kind: "usage", DetailJSON: detail, UsageJSON: usageJSON})
	}
	if len(outputs) == 0 && eventType != "" {
		outputs = append(outputs, Output{Kind: "progress", Text: eventType, DetailJSON: detail})
	}
	return outputs
}

func mapText(value map[string]any) string {
	for _, key := range []string{"text", "result", "content", "message", "event", "delta"} {
		if candidate, exists := value[key]; exists {
			if text := nestedText(candidate); text != "" {
				return text
			}
		}
	}
	return ""
}

func nestedText(value any) string {
	switch typed := value.(type) {
	case string:
		return core.Trim(typed)
	case map[string]any:
		return mapText(typed)
	case []any:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			if text := nestedText(item); text != "" {
				parts = append(parts, text)
			}
		}
		return core.Join("\n", parts...)
	default:
		return ""
	}
}

func findMapValue(value any, key string) (any, bool) {
	switch typed := value.(type) {
	case map[string]any:
		if found, exists := typed[key]; exists {
			return found, true
		}
		for _, child := range typed {
			if found, exists := findMapValue(child, key); exists {
				return found, true
			}
		}
	case []any:
		for _, child := range typed {
			if found, exists := findMapValue(child, key); exists {
				return found, true
			}
		}
	}
	return nil, false
}

func isRateLimit(line string) bool {
	lower := core.Lower(line)
	return core.Contains(lower, "rate limit") ||
		core.Contains(lower, "too many requests") ||
		core.Contains(lower, "429")
}

func retryDuration(line string) string {
	for _, marker := range []string{"try again in ", "retry after ", "retry in "} {
		lower := core.Lower(line)
		index := core.Index(lower, marker)
		if index < 0 {
			continue
		}
		remainder := line[index+len(marker):]
		fields := core.Fields(remainder)
		if len(fields) == 0 {
			continue
		}
		if candidate := validDuration(fields[0]); candidate != "" {
			return candidate
		}
	}

	var event map[string]any
	if decoded := core.JSONUnmarshalString(line, &event); !decoded.OK {
		return ""
	}
	for _, key := range []string{"retry_after", "retryAfter"} {
		if value, exists := findMapValue(event, key); exists {
			if candidate, ok := value.(string); ok {
				if duration := validDuration(candidate); duration != "" {
					return duration
				}
			}
		}
	}
	return ""
}

func validDuration(value string) string {
	candidate := core.TrimCutset(core.Trim(value), ".,;:!?()[]{}\"'")
	parsed := core.ParseDuration(candidate)
	if !parsed.OK {
		return ""
	}
	if parsed.Duration() <= 0 {
		return ""
	}
	return candidate
}
