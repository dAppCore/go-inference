// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import core "dappco.re/go"

func rocmReferencePromptLookupDraft(tokens []int32, minMatch, maxDraft int) ([]int32, error) {
	if minMatch <= 0 {
		return nil, core.E("rocm.Decode.PromptLookup", "min match must be positive", nil)
	}
	if maxDraft <= 0 {
		return nil, core.E("rocm.Decode.PromptLookup", "max draft must be positive", nil)
	}
	if len(tokens) < minMatch*2 {
		return nil, nil
	}
	bestStart := -1
	bestLen := 0
	for suffixLen := len(tokens) / 2; suffixLen >= minMatch; suffixLen-- {
		suffixStart := len(tokens) - suffixLen
		for candidateStart := 0; candidateStart+suffixLen < suffixStart; candidateStart++ {
			if int32SlicesEqual(tokens[candidateStart:candidateStart+suffixLen], tokens[suffixStart:]) {
				bestStart = candidateStart
				bestLen = suffixLen
				break
			}
		}
		if bestStart >= 0 {
			break
		}
	}
	if bestStart < 0 {
		return nil, nil
	}
	draftStart := bestStart + bestLen
	if draftStart >= len(tokens)-bestLen {
		return nil, nil
	}
	draftEnd := draftStart + maxDraft
	limit := len(tokens) - bestLen
	if draftEnd > limit {
		draftEnd = limit
	}
	return append([]int32(nil), tokens[draftStart:draftEnd]...), nil
}

func rocmReferenceSpeculativeAccept(draft, target []int32) ([]int32, int) {
	limit := len(draft)
	if len(target) < limit {
		limit = len(target)
	}
	for i := 0; i < limit; i++ {
		if draft[i] != target[i] {
			return append([]int32(nil), draft[:i]...), i
		}
	}
	if len(draft) > len(target) {
		return append([]int32(nil), draft[:limit]...), limit
	}
	return append([]int32(nil), draft...), -1
}

func int32SlicesEqual(left, right []int32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
