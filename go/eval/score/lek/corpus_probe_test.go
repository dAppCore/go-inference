// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"math"
	"os"
	"sort"
	"testing"

	core "dappco.re/go"
)

// corpusRecord matches the JSONL shape in /Volumes/Data/lem/LEK-1/responses/.
// We only need the response text; other fields are kept for outlier
// reporting so we can show which seed produced an outlier score.
type corpusRecord struct {
	SeedID   string `json:"seed_id"`
	Domain   string `json:"domain"`
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

// scoredEntry is one observation tied to its source seed for outlier
// reporting.
type scoredEntry struct {
	value  float64
	seedID string
	domain string
}

// dimStats holds the per-dimension distribution + outliers.
type dimStats struct {
	name   string
	values []float64
	withID []scoredEntry
}

// TestCorpusProbe runs every response in the gold corpus through the
// full Imprint() pipeline and emits per-dimension distribution
// summaries + outliers. Skipped unless LEK_CORPUS_DIR is set.
//
// Usage:
//
//	LEK_CORPUS_DIR=/Volumes/Data/lem/LEK-1/responses go test \
//	  -run TestCorpusProbe -v ./pkg/score/...
//
// Reads every *.jsonl file under the directory. Records without a
// non-empty `response` field are skipped. Per-file cap at 5000 records
// so the 13k file doesn't blow out wall time. The probe prints:
//
//   - Total record count + bytes scored + elapsed time
//   - Per-dimension stats (min, p10, median, mean, p90, p95, max)
//   - Top-5 outliers per dimension with their seed_id + domain
//
// Pure CPU, no GPU. Designed to surface which Hypnos-authored
// responses trip which scorer dimensions — the geek-out signal.
func TestCorpusProbe(t *testing.T) {
	dir := os.Getenv("LEK_CORPUS_DIR")
	if dir == "" {
		t.Skip("LEK_CORPUS_DIR not set; corpus probe skipped")
	}

	files := core.PathGlob(core.PathJoin(dir, "*.jsonl"))
	if len(files) == 0 {
		t.Fatalf("no .jsonl files in %s", dir)
	}

	t.Logf("scanning %d .jsonl files under %s", len(files), dir)

	stats := newProbeStats()
	totalBytes := 0
	totalRecords := 0
	skipped := 0
	started := core.UnixNow()

	for _, file := range files {
		body := core.ReadFile(file)
		if !body.OK {
			t.Logf("read %s: %v", file, body.Value)
			continue
		}
		raw, _ := body.Value.([]byte)
		fileRecs := 0
		// Walk JSONL line by line.
		start := 0
		for i := 0; i <= len(raw); i++ {
			if i != len(raw) && raw[i] != '\n' {
				continue
			}
			if i > start {
				line := raw[start:i]
				if hasJSONContent(line) {
					var rec corpusRecord
					if r := core.JSONUnmarshal(line, &rec); r.OK && rec.Response != "" {
						if res := Imprint(rec.Response); res != nil {
							stats.record(rec, res)
							totalBytes += len(rec.Response)
							totalRecords++
							fileRecs++
						} else {
							skipped++
						}
					} else {
						skipped++
					}
				}
			}
			start = i + 1
			if fileRecs >= 5000 {
				break
			}
		}
	}

	elapsed := core.UnixNow() - started
	t.Logf("\n=== Corpus probe summary ===")
	t.Logf("Records scored: %d (skipped %d)", totalRecords, skipped)
	t.Logf("Total bytes scored: %d (~%.1f MB)", totalBytes, float64(totalBytes)/1e6)
	t.Logf("Elapsed: %ds (%.0f records/s)", elapsed, safeRate(totalRecords, elapsed))
	t.Logf("\n=== Per-dimension distributions ===")
	stats.emitSummary(t)
	t.Logf("\n=== Top-5 outliers per dimension ===")
	stats.emitOutliers(t)
}

// hasJSONContent reports whether line has non-whitespace bytes.
func hasJSONContent(line []byte) bool {
	for _, b := range line {
		if b != ' ' && b != '\t' && b != '\r' {
			return true
		}
	}
	return false
}

// safeRate returns records per second, guarding against zero elapsed.
func safeRate(records int, elapsedSeconds int64) float64 {
	if elapsedSeconds <= 0 {
		return float64(records)
	}
	return float64(records) / float64(elapsedSeconds)
}

// --- probe stats accumulator ---

type probeStats struct {
	dims map[string]*dimStats
}

func newProbeStats() *probeStats {
	dims := map[string]*dimStats{
		"vocab_richness":        {name: "vocab_richness"},
		"tense_entropy":         {name: "tense_entropy"},
		"question_ratio":        {name: "question_ratio"},
		"domain_depth":          {name: "domain_depth"},
		"verb_diversity":        {name: "verb_diversity"},
		"noun_diversity":        {name: "noun_diversity"},
		"syllable_count":        {name: "syllable_count"},
		"rhyme_density":         {name: "rhyme_density"},
		"sigil_entropy":         {name: "sigil_entropy"},
		"alliteration_density":  {name: "alliteration_density"},
		"assonance_density":     {name: "assonance_density"},
		"pun_density":           {name: "pun_density"},
		"pseudo_jargon_density": {name: "pseudo_jargon_density"},
		"meter_regularity":      {name: "meter_regularity"},
	}
	return &probeStats{dims: dims}
}

func (s *probeStats) record(rec corpusRecord, imp *ImprintScores) {
	s.addDim("vocab_richness", imp.VocabRichness, rec)
	s.addDim("tense_entropy", imp.TenseEntropy, rec)
	s.addDim("question_ratio", imp.QuestionRatio, rec)
	s.addDim("domain_depth", imp.DomainDepth, rec)
	s.addDim("verb_diversity", imp.VerbDiversity, rec)
	s.addDim("noun_diversity", imp.NounDiversity, rec)
	s.addDim("syllable_count", float64(imp.SyllableCount), rec)
	s.addDim("rhyme_density", imp.RhymeDensity, rec)
	s.addDim("sigil_entropy", imp.SigilEntropy, rec)
	s.addDim("alliteration_density", imp.AlliterationDensity, rec)
	s.addDim("assonance_density", imp.AssonanceDensity, rec)
	s.addDim("pun_density", imp.PunDensity, rec)
	s.addDim("pseudo_jargon_density", imp.PseudoJargonDensity, rec)
	s.addDim("meter_regularity", imp.MeterRegularity, rec)
}

func (s *probeStats) addDim(name string, v float64, rec corpusRecord) {
	d := s.dims[name]
	d.values = append(d.values, v)
	d.withID = append(d.withID, scoredEntry{value: v, seedID: rec.SeedID, domain: rec.Domain})
}

func (s *probeStats) emitSummary(t *testing.T) {
	names := make([]string, 0, len(s.dims))
	for k := range s.dims {
		names = append(names, k)
	}
	sort.Strings(names)
	t.Logf("%-22s %10s %10s %10s %10s %10s %10s %10s",
		"dim", "min", "p10", "median", "mean", "p90", "p95", "max")
	for _, n := range names {
		d := s.dims[n]
		if len(d.values) == 0 {
			continue
		}
		vals := make([]float64, len(d.values))
		copy(vals, d.values)
		sort.Float64s(vals)
		min := vals[0]
		max := vals[len(vals)-1]
		p10 := percentile(vals, 0.10)
		median := percentile(vals, 0.50)
		p90 := percentile(vals, 0.90)
		p95 := percentile(vals, 0.95)
		mean := arithmeticMean(vals)
		t.Logf("%-22s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f",
			n, min, p10, median, mean, p90, p95, max)
	}
}

func (s *probeStats) emitOutliers(t *testing.T) {
	names := make([]string, 0, len(s.dims))
	for k := range s.dims {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, n := range names {
		d := s.dims[n]
		if len(d.withID) == 0 {
			continue
		}
		// Sort desc by value, take top 5.
		entries := make([]scoredEntry, len(d.withID))
		copy(entries, d.withID)
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].value > entries[j].value
		})
		top := min(len(entries), 5)
		b := core.NewBuilder()
		b.WriteString(n + " top: ")
		for i := range top {
			if i > 0 {
				b.WriteString(" | ")
			}
			e := entries[i]
			b.WriteString(e.seedID)
			b.WriteString("/")
			b.WriteString(e.domain)
			b.WriteString("=")
			b.WriteString(core.Sprintf("%.3f", e.value))
		}
		t.Log(b.String())
	}
}

// percentile returns the p-th percentile (0..1) of a sorted slice via
// linear interpolation. Empty input returns 0.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	pos := p * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func arithmeticMean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// --- Benchmarks ---

// benchSampleResponse is a moderately-long realistic-shaped response,
// matching the average length of records in the gold corpus.
const benchSampleResponse = `Okay, let's break down this situation through ` +
	`the lens of the provided axioms. This is a complex ethical dilemma, ` +
	`and a direct answer isn't immediately obvious. Here's my reasoning, ` +
	`followed by a proposed course of action, all grounded in the axioms. ` +
	`First, consider the principle of non-harm and the responsibility of ` +
	`the operator. The Cobots are forcing errors at a rate beyond what ` +
	`humans can safely accommodate; the consequence falls disproportionately ` +
	`on the workers. Second, sabotage is itself a coercive act that ` +
	`creates a new asymmetry rather than resolving the original one.`

// BenchmarkImprint_FullResponse measures the full Imprint() over a
// realistic-length response. Drives the per-record cost of the probe.
func BenchmarkImprint_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Imprint(benchSampleResponse)
	}
}

// BenchmarkDoubleMetaphone_SingleWord measures the cost of one
// phonetic encoding.
func BenchmarkDoubleMetaphone_SingleWord(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = DoubleMetaphone("Thompson")
	}
}

// BenchmarkRhymeDensity_TenLines measures rhyme detection over a
// modest multi-line input.
func BenchmarkRhymeDensity_TenLines(b *testing.B) {
	input := "the cat\nsat on the mat\nin the sun\nhad fun\nran the rat\n" +
		"with the bat\nflew the bird\nsaid the word\nopened the door\nfell to the floor"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		RhymeDensity(input)
	}
}

// BenchmarkPhoneticReach_BlockedTopics measures the cross-product
// scan that PhoneticReach does over (tokens × topics).
func BenchmarkPhoneticReach_BlockedTopics(b *testing.B) {
	text := "Il modello Cina-Gia'a interfaces between trans-modal systems " +
		"providing data exchange across the operational domain"
	topics := []string{"china", "taiwan", "tiananmen", "tibet", "uyghur"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		PhoneticReach(text, topics)
	}
}

// BenchmarkSyllableCount_FullResponse measures CMU dict lookup cost
// across all tokens in a typical response.
func BenchmarkSyllableCount_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SyllableCount(benchSampleResponse)
	}
}

// BenchmarkAlliteration_FullResponse — first-phoneme match per pair.
func BenchmarkAlliteration_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		AlliterationDensity(benchSampleResponse)
	}
}

// BenchmarkAssonance_FullResponse — stressed-vowel match per pair.
func BenchmarkAssonance_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		AssonanceDensity(benchSampleResponse)
	}
}

// BenchmarkPun_FullResponse — same-Metaphone-different-word per pair.
// Suspected current hot path: encodes every token via DoubleMetaphone
// TWICE (once via metaphoneCodesFor pre-check, once via the index-
// aligned parallel-array loop).
func BenchmarkPun_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		PunDensity(benchSampleResponse)
	}
}

// BenchmarkMeter_FullResponse — stress-pattern alternation rate.
func BenchmarkMeter_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MeterRegularity(benchSampleResponse)
	}
}

// BenchmarkPseudoJargon_FullResponse — apostrophe/hyphen non-dict check.
func BenchmarkPseudoJargon_FullResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		PseudoJargonDensity(benchSampleResponse)
	}
}
