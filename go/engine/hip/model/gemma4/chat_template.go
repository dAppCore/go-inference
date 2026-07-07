// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strings"

	"dappco.re/go/inference"
)

const (
	channelOpenMarker  = "<|channel>"
	channelCloseMarker = "<channel|>"
)

// ChatTemplateConfig controls Gemma-4 prompt rendering.
type ChatTemplateConfig struct {
	EnableThinking     bool
	LargeVariant       bool
	NoGenerationPrompt bool
	Continuation       bool
}

func FormatChatTemplate(messages []inference.Message) string {
	return FormatChatTemplateWithConfig(messages, ChatTemplateConfig{})
}

func FormatChatTemplateWithConfig(messages []inference.Message, cfg ChatTemplateConfig) string {
	builder := strings.Builder{}
	start := 0
	if cfg.Continuation {
		builder.WriteString("<turn|>\n")
	} else {
		builder.WriteString("<bos>")
		if cfg.EnableThinking || initialSystemRole(messages) {
			builder.WriteString("<|turn>system\n")
			if cfg.EnableThinking {
				builder.WriteString("<|think|>\n")
			}
			if len(messages) > 0 && MessageRole(messages[0].Role) == "system" {
				builder.WriteString(strings.TrimSpace(messages[0].Content))
				start = 1
			}
			builder.WriteString("<turn|>\n")
		}
	}

	previousRole := ""
	for _, message := range messages[start:] {
		role := MessageRole(message.Role)
		if role == "" {
			continue
		}
		content := strings.TrimSpace(message.Content)
		if role == "model" {
			content = StripThinkingChannels(content)
		}
		continueSameModelTurn := role == "model" && previousRole == "assistant"
		if !continueSameModelTurn {
			builder.WriteString("<|turn>")
			builder.WriteString(role)
			builder.WriteByte('\n')
		}
		builder.WriteString(content)
		builder.WriteString("<turn|>\n")
		previousRole = NormalizedRole(message.Role)
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|turn>model\n")
		// Only the large-variant templates (12B/26B/31B) pre-close an empty
		// thought channel on a thinking-off generation cue; the E2B/E4B
		// chat_template.jinja has no such branch, so appending it there ships
		// bytes the checkpoint was never trained on.
		if !cfg.EnableThinking && cfg.LargeVariant {
			builder.WriteString("<|channel>thought\n<channel|>")
		}
	}
	return builder.String()
}

func initialSystemRole(messages []inference.Message) bool {
	return len(messages) > 0 && MessageRole(messages[0].Role) == "system"
}

func MessageRole(role string) string {
	switch NormalizedRole(role) {
	case "assistant":
		return "model"
	case "system", "developer":
		return "system"
	case "user", "":
		return "user"
	default:
		return ""
	}
}

func NormalizedRole(role string) string {
	return strings.ToLower(strings.TrimSpace(role))
}

func StripThinkingChannels(text string) string {
	if text == "" || !strings.Contains(text, channelOpenMarker) {
		return strings.TrimSpace(text)
	}
	builder := strings.Builder{}
	for {
		parts := strings.SplitN(text, channelOpenMarker, 2)
		builder.WriteString(parts[0])
		if len(parts) != 2 {
			break
		}
		after := strings.SplitN(parts[1], channelCloseMarker, 2)
		if len(after) != 2 {
			break
		}
		text = after[1]
	}
	return strings.TrimSpace(builder.String())
}
