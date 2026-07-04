// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "strconv"

// AttentionWindowPolicy describes Gemma-4 attention-mask/window decisions that
// are independent of any concrete GPU array implementation.
type AttentionWindowPolicy struct {
	SlidingWindow              int
	DenseSlidingPrefillMask    bool
	CachedOffsetCausalMask     bool
	FixedSingleTokenCausalMask bool
	OffsetCausalAttention      bool
	SlidingContextTrim         bool
	VerifyProposalLimit        int
}

// AttentionWindowRange is a half-open visible key interval for one query.
type AttentionWindowRange struct {
	QueryPosition int
	KeyStart      int
	KeyEnd        int
}

func AttentionWindowPolicyOf(cfg TextConfig) AttentionWindowPolicy {
	window := positiveInt(cfg.SlidingWindow)
	if window <= 0 {
		return AttentionWindowPolicy{}
	}
	return AttentionWindowPolicy{
		SlidingWindow:              window,
		DenseSlidingPrefillMask:    true,
		CachedOffsetCausalMask:     true,
		FixedSingleTokenCausalMask: true,
		OffsetCausalAttention:      true,
		SlidingContextTrim:         true,
		VerifyProposalLimit:        MaxSpeculativeVerifyProposals(window, window),
	}
}

// CachedAttentionWindows maps each query token to the visible key range used by
// Gemma-4's cached sliding causal mask. Ranges are absolute key positions.
func CachedAttentionWindows(queryLen, keyLen, offset, keyStart, window int) []AttentionWindowRange {
	if queryLen <= 0 || keyLen <= 0 {
		return nil
	}
	ranges := make([]AttentionWindowRange, queryLen)
	keyEnd := keyStart + keyLen
	for query := range ranges {
		queryPos := offset + query
		start := keyStart
		if window > 0 {
			windowStart := queryPos - window + 1
			if windowStart > start {
				start = windowStart
			}
		}
		end := queryPos + 1
		if end > keyEnd {
			end = keyEnd
		}
		if end < start {
			end = start
		}
		ranges[query] = AttentionWindowRange{
			QueryPosition: queryPos,
			KeyStart:      start,
			KeyEnd:        end,
		}
	}
	return ranges
}

// CachedAttentionAllowed returns a row-major query-by-key visibility mask. True
// means the query may attend to that key.
func CachedAttentionAllowed(queryLen, keyLen, offset, keyStart, window int) []bool {
	ranges := CachedAttentionWindows(queryLen, keyLen, offset, keyStart, window)
	if len(ranges) == 0 {
		return nil
	}
	allowed := make([]bool, queryLen*keyLen)
	for query, visible := range ranges {
		for key := 0; key < keyLen; key++ {
			keyPos := keyStart + key
			allowed[query*keyLen+key] = keyPos >= visible.KeyStart && keyPos < visible.KeyEnd
		}
	}
	return allowed
}

func CanUseOffsetCausalAttention(queryLen, keyLen, window int) bool {
	if queryLen <= 1 || keyLen <= 0 {
		return false
	}
	if window <= 0 {
		return true
	}
	return queryLen <= window && keyLen <= window+queryLen-1
}

func SlidingCausalContextLen(queryLen, keyLen, window int) int {
	if queryLen <= 1 || keyLen <= 0 || window <= 0 || queryLen > window {
		return positiveInt(keyLen)
	}
	needed := window + queryLen - 1
	if needed >= keyLen {
		return keyLen
	}
	return needed
}

func FixedSingleTokenCausalWindow(capacity, offset int) (AttentionWindowRange, bool) {
	if capacity <= 0 || offset < 0 || offset+1 > capacity {
		return AttentionWindowRange{}, false
	}
	return AttentionWindowRange{
		QueryPosition: offset,
		KeyStart:      0,
		KeyEnd:        offset + 1,
	}, true
}

func FixedSingleTokenCausalAllowed(capacity, offset int) []bool {
	window, ok := FixedSingleTokenCausalWindow(capacity, offset)
	if !ok {
		return nil
	}
	allowed := make([]bool, capacity)
	for key := 0; key < capacity; key++ {
		allowed[key] = key >= window.KeyStart && key < window.KeyEnd
	}
	return allowed
}

func MaxSpeculativeVerifyProposals(draftTokens, slidingWindow int) int {
	if draftTokens <= 0 {
		return 0
	}
	if slidingWindow > 1 && draftTokens > slidingWindow-1 {
		return slidingWindow - 1
	}
	return draftTokens
}

func ApplyAttentionWindowPolicyLabels(labels map[string]string, policy AttentionWindowPolicy) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if policy.SlidingWindow <= 0 {
		return labels
	}
	window := strconv.Itoa(policy.SlidingWindow)
	labels["attention_window_policy"] = "sliding_causal"
	labels["gemma4_attention_window_policy"] = "sliding_causal"
	labels["attention_window_tokens"] = window
	labels["gemma4_attention_window_tokens"] = window
	setBoolLabel(labels, "attention_mask_dense_sliding_prefill", policy.DenseSlidingPrefillMask)
	setBoolLabel(labels, "gemma4_attention_mask_dense_sliding_prefill", policy.DenseSlidingPrefillMask)
	setBoolLabel(labels, "attention_mask_cached_offset_causal", policy.CachedOffsetCausalMask)
	setBoolLabel(labels, "gemma4_attention_mask_cached_offset_causal", policy.CachedOffsetCausalMask)
	setBoolLabel(labels, "attention_mask_fixed_single_token", policy.FixedSingleTokenCausalMask)
	setBoolLabel(labels, "gemma4_attention_mask_fixed_single_token", policy.FixedSingleTokenCausalMask)
	setBoolLabel(labels, "attention_offset_causal_fast_path", policy.OffsetCausalAttention)
	setBoolLabel(labels, "gemma4_attention_offset_causal_fast_path", policy.OffsetCausalAttention)
	setBoolLabel(labels, "attention_sliding_context_trim", policy.SlidingContextTrim)
	setBoolLabel(labels, "gemma4_attention_sliding_context_trim", policy.SlidingContextTrim)
	if policy.VerifyProposalLimit > 0 {
		value := strconv.Itoa(policy.VerifyProposalLimit)
		labels["speculative_verify_proposal_window_limit"] = value
		labels["gemma4_speculative_verify_proposal_window_limit"] = value
	}
	return labels
}

func setBoolLabel(labels map[string]string, key string, value bool) {
	if value {
		labels[key] = "true"
	}
}
