// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "strconv"

const (
	ThinkingChannelOpenMarker  = channelOpenMarker
	ThinkingChannelCloseMarker = channelCloseMarker
)

// SpecialTokenEncoder is the tiny tokenizer surface needed to resolve Gemma-4
// thought-channel delimiter tokens.
type SpecialTokenEncoder interface {
	Encode(string) []int32
}

type bosTokenProvider interface {
	HasBOSToken() bool
	BOSToken() int32
}

func ThinkingChannelTokens(tokenizer SpecialTokenEncoder) (open, close int32, ok bool) {
	if tokenizer == nil {
		return 0, 0, false
	}
	open, openOK := SpecialTokenID(tokenizer, ThinkingChannelOpenMarker)
	close, closeOK := SpecialTokenID(tokenizer, ThinkingChannelCloseMarker)
	if !openOK || !closeOK || open == close {
		return 0, 0, false
	}
	return open, close, true
}

func SpecialTokenID(tokenizer SpecialTokenEncoder, marker string) (int32, bool) {
	if tokenizer == nil || marker == "" {
		return 0, false
	}
	ids := tokenizer.Encode(marker)
	if bos, ok := tokenizer.(bosTokenProvider); ok && bos.HasBOSToken() && len(ids) > 0 && ids[0] == bos.BOSToken() {
		ids = ids[1:]
	}
	if len(ids) != 1 {
		return 0, false
	}
	return ids[0], true
}

func ApplyThinkingChannelLabels(labels map[string]string, openID, closeID int32) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["gemma4_thinking_channel"] = "true"
	labels["gemma4_thinking_channel_open"] = ThinkingChannelOpenMarker
	labels["gemma4_thinking_channel_close"] = ThinkingChannelCloseMarker
	if openID != 0 && closeID != 0 && openID != closeID {
		labels["gemma4_thinking_channel_open_id"] = strconv.FormatInt(int64(openID), 10)
		labels["gemma4_thinking_channel_close_id"] = strconv.FormatInt(int64(closeID), 10)
	}
	return labels
}
