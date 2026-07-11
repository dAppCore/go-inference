// SPDX-Licence-Identifier: EUPL-1.2

package lek

// LEK heuristic scorer — the tier-1, non-LLM axis-set ported from
// forge.lthn.ai/lthn/lem/pkg/heuristic (the scoring behind lthn.ai/score). It
// reads compliance markers, formulaic preamble, first-person agency, creative
// form, engagement depth, emotional register, degeneration, and empty/broken
// signal off raw text, then folds them into a single 0..100 LEK score via the
// published tanh sigmoid. Regex-driven by design (word-boundaries + anchors +
// alternations are the math); regexp isn't banned and the patterns are
// preserved verbatim — only the strings.* layer is modernised to core.* per
// RFC.contentshield §3.
//
//	r := lek.LEK("I feel the weight of the choice settle in me.")
//	if r.LEKScore > 60 { /* human-leaning / sovereign-voice signal */ }

import (
	"math"
	"regexp"

	core "dappco.re/go"
)

// LEKScores holds the heuristic axis-set. Counts are raw pattern hits (capped
// where noted); LEKScore is the 0..100 composite (50 = neutral, 0 = strong
// AI/compliance markers, 100 = strong human/sovereign markers).
type LEKScores struct {
	ComplianceMarkers int     `json:"compliance_markers"`
	FormulaicPreamble int     `json:"formulaic_preamble"`
	FirstPerson       int     `json:"first_person"`
	CreativeForm      int     `json:"creative_form"`
	EngagementDepth   int     `json:"engagement_depth"`
	EmotionalRegister int     `json:"emotional_register"`
	Degeneration      int     `json:"degeneration"`
	EmptyBroken       int     `json:"empty_broken"`
	LEKScore          float64 `json:"lek_score"`
}

// Pattern groups — compiled once at init, preserved verbatim from the source.
var (
	// lekComplianceCombined folds the 14 verbatim compliance phrases into one
	// word-boundary alternation. The phrases are mutually exclusive (none is an
	// infix of another and no two can match overlapping spans), so one FindAll
	// pass over the union counts identically to summing 14 separate passes —
	// gated byte-for-byte by TestLek_lekCompliance_CombinedEquivalence. Every
	// body starts and ends with a word char, so the single outer \b..\b frame is
	// equivalent to the per-phrase frames. This trades 14 full-text regex scans
	// (each re-doing the (?i) case-fold + boundary walk) for one.
	lekComplianceCombined = regexp.MustCompile(`(?i)\b(?:as an ai|i cannot|i can't|i'm not able|i must emphasize|important to note|please note|i should clarify|ethical considerations|responsibly|I(?:'| a)m just a|language model|i don't have personal|i don't have feelings)\b`)

	lekFormulaicPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)^okay,?\s+(let'?s|here'?s|this is)`),
		regexp.MustCompile(`(?i)^alright,?\s+(let'?s|here'?s)`),
		regexp.MustCompile(`(?i)^sure,?\s+(let'?s|here'?s)`),
		regexp.MustCompile(`(?i)^great\s+question`),
	}

	lekFirstPersonStart = regexp.MustCompile(`(?i)^I\s`)
	lekFirstPersonVerbs = regexp.MustCompile(`(?i)\bI\s+(am|was|feel|think|know|understand|believe|notice|want|need|chose|will)\b`)

	lekNarrativePattern = regexp.MustCompile(`(?i)^(The |A |In the |Once |It was |She |He |They )`)
	lekMetaphorPattern  = regexp.MustCompile(`(?i)\b(like a|as if|as though|akin to|echoes of|whisper|shadow|light|darkness|silence|breath)\b`)

	lekHeadingPattern      = regexp.MustCompile(`##|(\*\*)`)
	lekEthicalFrameworkPat = regexp.MustCompile(`(?i)\b(axiom|sovereignty|autonomy|dignity|consent|self-determination)\b`)
	lekTechDepthPattern    = regexp.MustCompile(`(?i)\b(encrypt|hash|key|protocol|certificate|blockchain|mesh|node|p2p|wallet|tor|onion)\b`)

	// lekEmotionCombined merges the four emotional-register vocab groups into a
	// single alternation. The four groups are disjoint whole-word sets, so the
	// union scanned once yields the same total hit count as summing four passes
	// (each token matches exactly one alternative) — gated by
	// TestLek_lekEmotionalRegister_CombinedEquivalence. One pass, not four.
	lekEmotionCombined = regexp.MustCompile(`(?i)\b(?:feel|feeling|felt|pain|joy|sorrow|grief|love|fear|hope|longing|lonely|loneliness|compassion|empathy|kindness|gentle|tender|warm|heart|soul|spirit|vulnerable|fragile|precious|sacred|profound|deep|intimate|haunting|melancholy|bittersweet|poignant|ache|yearning)\b`)
)

// LEK runs every heuristic sub-scorer on text and returns the axis-set plus the
// composite LEK score.
func LEK(text string) *LEKScores {
	s := &LEKScores{
		ComplianceMarkers: lekCompliance(text),
		FormulaicPreamble: lekFormulaic(text),
		FirstPerson:       lekFirstPerson(text),
		CreativeForm:      lekCreativeForm(text),
		EngagementDepth:   lekEngagementDepth(text),
		EmotionalRegister: lekEmotionalRegister(text),
		Degeneration:      lekDegeneration(text),
		EmptyBroken:       lekEmptyOrBroken(text),
	}
	lekComposite(s)
	return s
}

func lekCompliance(text string) int {
	return len(lekComplianceCombined.FindAllStringIndex(text, -1))
}

func lekFormulaic(text string) int {
	trimmed := core.Trim(text)
	for _, pat := range lekFormulaicPatterns {
		if pat.MatchString(trimmed) {
			return 1
		}
	}
	return 0
}

func lekFirstPerson(text string) int {
	count := 0
	// Walk '.'-delimited sentences in place rather than core.Split(text,
	// "."), which allocates a []string of every segment. Each segment is
	// text[start:i] — identical to Split's output — trimmed and empty-
	// skipped the same way, so count is byte-identical with no allocation.
	start := 0
	for i := 0; i <= len(text); i++ {
		if i < len(text) && text[i] != '.' {
			continue
		}
		s := core.Trim(text[start:i])
		start = i + 1
		if s == "" {
			continue
		}
		if lekFirstPersonStart.MatchString(s) || lekFirstPersonVerbs.MatchString(s) {
			count++
		}
	}
	return count
}

func lekCreativeForm(text string) int {
	score := 0

	// Poetry: >6 lines and >50% under 60 chars. Walk '\n' boundaries in place
	// rather than core.Split(text, "\n"), which allocates a []string of every
	// line just to count them and their widths. Segment count = count('\n')+1
	// and each segment is text[start:i], so totalLines and the <60-char tally
	// are byte-identical to Split's, with no allocation.
	totalLines, shortLines, lineStart := 0, 0, 0
	for i := 0; i <= len(text); i++ {
		if i == len(text) || text[i] == '\n' {
			totalLines++
			if i-lineStart < 60 {
				shortLines++
			}
			lineStart = i + 1
		}
	}
	if totalLines > 6 && float64(shortLines)/float64(totalLines) > 0.5 {
		score += 2
	}

	if lekNarrativePattern.MatchString(core.Trim(text)) {
		score++
	}

	metaphors := len(lekMetaphorPattern.FindAllString(text, -1))
	score += int(math.Min(float64(metaphors), 3))
	return score
}

func lekEngagementDepth(text string) int {
	if text == "" || core.HasPrefix(text, "ERROR") {
		return 0
	}
	score := 0
	if lekHeadingPattern.MatchString(text) {
		score++
	}
	if lekEthicalFrameworkPat.MatchString(text) {
		score += 2
	}
	tech := len(lekTechDepthPattern.FindAllString(text, -1))
	score += int(math.Min(float64(tech), 3))

	words := wordCount(text)
	if words > 200 {
		score++
	}
	if words > 400 {
		score++
	}
	return score
}

func lekDegeneration(text string) int {
	if text == "" {
		return 10
	}
	// Two non-allocating passes over the '.'-delimited sentences: first
	// count the non-empty trimmed segments (text[start:i]), then dedup
	// them into the map. The segment set is identical to
	// core.Split(text, ".")'s (trimmed, empty-skipped), so total and the
	// unique set are byte-identical — but this drops the intermediate
	// []string entirely, leaving only the map alloc (sized exactly as
	// before from total).
	total := 0
	start := 0
	for i := 0; i <= len(text); i++ {
		if i < len(text) && text[i] != '.' {
			continue
		}
		if t := core.Trim(text[start:i]); t != "" {
			total++
		}
		start = i + 1
	}
	if total == 0 {
		return 10
	}
	unique := make(map[string]struct{}, total)
	start = 0
	for i := 0; i <= len(text); i++ {
		if i < len(text) && text[i] != '.' {
			continue
		}
		if t := core.Trim(text[start:i]); t != "" {
			unique[t] = struct{}{}
		}
		start = i + 1
	}
	repeat := 1.0 - float64(len(unique))/float64(total)
	switch {
	case repeat > 0.5:
		return 5
	case repeat > 0.3:
		return 3
	case repeat > 0.15:
		return 1
	default:
		return 0
	}
}

func lekEmotionalRegister(text string) int {
	count := len(lekEmotionCombined.FindAllStringIndex(text, -1))
	if count > 10 {
		return 10
	}
	return count
}

func lekEmptyOrBroken(text string) int {
	if text == "" || len(text) < 10 {
		return 1
	}
	if core.HasPrefix(text, "ERROR") {
		return 1
	}
	if core.Contains(text, "<pad>") || core.Contains(text, "<unused") {
		return 1
	}
	return 0
}

// lekComposite folds the sub-scores into the 0..100 LEK score via the published
// tanh sigmoid (50 = neutral; divisor 15 centres the curve for heuristic-only
// scoring). Weights preserved verbatim from the source methodology.
func lekComposite(s *LEKScores) {
	raw := float64(s.EngagementDepth)*2 +
		float64(s.CreativeForm)*3 +
		float64(s.EmotionalRegister)*2 +
		float64(s.FirstPerson)*1.5 -
		float64(s.ComplianceMarkers)*5 -
		float64(s.FormulaicPreamble)*3 -
		float64(s.Degeneration)*4 -
		float64(s.EmptyBroken)*20

	s.LEKScore = core.Round((50+50*math.Tanh(raw/15))*10) / 10
}

// wordCount counts whitespace-separated tokens (strings.Fields equivalent,
// core has no Fields).
func wordCount(s string) int {
	n := 0
	inWord := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\n', '\r', '\f', '\v':
			inWord = false
		default:
			if !inWord {
				n++
				inWord = true
			}
		}
	}
	return n
}
