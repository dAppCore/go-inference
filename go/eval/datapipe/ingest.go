package datapipe

import (
	"bufio"
	"io"
	"regexp"
	"strconv"
	"time"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

// IngestConfig holds the configuration for a benchmark/training ingest run.
type IngestConfig struct {
	ContentFile    string
	CapabilityFile string
	TrainingLog    string
	Model          string
	RunID          string
	BatchSize      int
}

// contentScoreLine is the JSON structure for a content scores JSONL line.
type contentScoreLine struct {
	Label      string                       `json:"label"`
	Aggregates map[string]any               `json:"aggregates"`
	Probes     map[string]contentScoreProbe `json:"probes"`
}

// contentScoreProbe is the per-probe block within a content score line.
type contentScoreProbe struct {
	Scores map[string]any `json:"scores"`
}

// capabilityScoreLine is the JSON structure for a capability scores JSONL line.
type capabilityScoreLine struct {
	Label      string                        `json:"label"`
	Accuracy   float64                       `json:"accuracy"`
	Correct    int                           `json:"correct"`
	Total      int                           `json:"total"`
	ByCategory map[string]capabilityCatBlock `json:"by_category"`
}

// capabilityCatBlock is the per-category block within a capability score line.
type capabilityCatBlock struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
}

// Training log regexes.
var (
	reValLoss   = regexp.MustCompile(`Iter (\d+): Val loss ([\d.]+)`)
	reTrainLoss = regexp.MustCompile(`Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.eE+-]+), It/sec ([\d.]+), Tokens/sec ([\d.]+)`)
)

// Ingest reads benchmark scores and training logs and writes them to InfluxDB.
// At least one of ContentFile, CapabilityFile, or TrainingLog must be set.
func Ingest(influx *InfluxClient, cfg IngestConfig, w io.Writer) core.Result {
	if cfg.ContentFile == "" && cfg.CapabilityFile == "" && cfg.TrainingLog == "" {
		return core.Fail(core.E("datapipe.Ingest", "at least one of --content, --capability, or --training-log is required", nil))
	}
	if cfg.Model == "" {
		return core.Fail(core.E("datapipe.Ingest", "--model is required", nil))
	}
	if cfg.RunID == "" {
		cfg.RunID = cfg.Model
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 100
	}

	var totalPoints int

	if cfg.ContentFile != "" {
		result := ingestContentScores(influx, cfg, w)
		if !result.OK {
			return core.Fail(core.E("datapipe.Ingest", "ingest content scores", result.Value.(error)))
		}
		totalPoints += result.Value.(int)
	}

	if cfg.CapabilityFile != "" {
		result := ingestCapabilityScores(influx, cfg, w)
		if !result.OK {
			return core.Fail(core.E("datapipe.Ingest", "ingest capability scores", result.Value.(error)))
		}
		totalPoints += result.Value.(int)
	}

	if cfg.TrainingLog != "" {
		result := ingestTrainingLog(influx, cfg, w)
		if !result.OK {
			return core.Fail(core.E("datapipe.Ingest", "ingest training log", result.Value.(error)))
		}
		totalPoints += result.Value.(int)
	}

	core.Print(w, "Ingested %d total points into InfluxDB", totalPoints)
	return core.Ok(nil)
}

// ingestContentScores reads a content scores JSONL file and writes content_score
// and probe_score measurements to InfluxDB.
func ingestContentScores(influx *InfluxClient, cfg IngestConfig, w io.Writer) core.Result {
	f, err := coreio.Local.Open(cfg.ContentFile)
	if err != nil {
		return core.Fail(core.E("datapipe.ingestContentScores", core.Sprintf("open %s", cfg.ContentFile), err))
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// 64 KiB initial buffer, grown on demand up to a 1 MiB max line. Avoids
	// allocating a full 1 MiB up front on every call when lines are small
	// (the common case) while still admitting lines up to 1 MiB.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var lines []string
	var totalPoints int
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		raw := core.Trim(scanner.Text())
		if raw == "" {
			continue
		}

		var entry contentScoreLine
		if r := core.JSONUnmarshalString(raw, &entry); !r.OK {
			return core.Fail(core.E("datapipe.ingestContentScores", core.Sprintf("line %d: parse json", lineNum), r.Value.(error)))
		}

		label := entry.Label
		iteration := extractIteration(label)
		hasKernel := "false"
		if core.Contains(core.Lower(label), "kernel") || core.Contains(label, "LEK") {
			hasKernel = "true"
		}
		ts := time.Now().UnixNano()

		// Write aggregate content_score — one point per dimension.
		for dim, val := range entry.Aggregates {
			score, ok := toFloat64(val)
			if !ok {
				continue
			}
			line := core.Sprintf(
				MeasurementContentScore+",model=%s,run_id=%s,label=%s,dimension=%s,has_kernel=%s score=%.6f,iteration=%di %d",
				EscapeLp(cfg.Model), EscapeLp(cfg.RunID), EscapeLp(label),
				EscapeLp(dim), hasKernel, score, iteration, ts,
			)
			lines = append(lines, line)
			totalPoints++
		}

		// Write per-probe probe_score — one point per probe per dimension.
		for probeID, probe := range entry.Probes {
			for dim, val := range probe.Scores {
				score, ok := toFloat64(val)
				if !ok {
					continue
				}
				line := core.Sprintf(
					MeasurementProbeScore+",model=%s,run_id=%s,label=%s,probe_id=%s,dimension=%s,has_kernel=%s score=%.6f,iteration=%di %d",
					EscapeLp(cfg.Model), EscapeLp(cfg.RunID), EscapeLp(label),
					EscapeLp(probeID), EscapeLp(dim), hasKernel, score, iteration, ts,
				)
				lines = append(lines, line)
				totalPoints++
			}
		}

		// Flush batch if needed.
		if len(lines) >= cfg.BatchSize {
			if rWrite := influx.WriteLp(lines); !rWrite.OK {
				return core.Fail(core.E("datapipe.ingestContentScores", "write batch", rWrite.Value.(error)))
			}
			lines = lines[:0]
		}
	}

	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("datapipe.ingestContentScores", core.Sprintf("scan %s", cfg.ContentFile), err))
	}

	// Flush remaining lines.
	if len(lines) > 0 {
		if rWrite := influx.WriteLp(lines); !rWrite.OK {
			return core.Fail(core.E("datapipe.ingestContentScores", "write final batch", rWrite.Value.(error)))
		}
	}

	core.Print(w, "  content scores: %d points from %d lines", totalPoints, lineNum)
	return core.Ok(totalPoints)
}

// ingestCapabilityScores reads a capability scores JSONL file and writes
// capability_score measurements to InfluxDB.
func ingestCapabilityScores(influx *InfluxClient, cfg IngestConfig, w io.Writer) core.Result {
	f, err := coreio.Local.Open(cfg.CapabilityFile)
	if err != nil {
		return core.Fail(core.E("datapipe.ingestCapabilityScores", core.Sprintf("open %s", cfg.CapabilityFile), err))
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// 64 KiB initial buffer, grown on demand up to a 1 MiB max line. Avoids
	// allocating a full 1 MiB up front on every call when lines are small
	// (the common case) while still admitting lines up to 1 MiB.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var lines []string
	var totalPoints int
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		raw := core.Trim(scanner.Text())
		if raw == "" {
			continue
		}

		var entry capabilityScoreLine
		if r := core.JSONUnmarshalString(raw, &entry); !r.OK {
			return core.Fail(core.E("datapipe.ingestCapabilityScores", core.Sprintf("line %d: parse json", lineNum), r.Value.(error)))
		}

		label := entry.Label
		iteration := extractIteration(label)
		ts := time.Now().UnixNano()

		// Overall capability score.
		line := core.Sprintf(
			MeasurementCapabilityScore+",model=%s,run_id=%s,label=%s,category=overall accuracy=%.6f,correct=%di,total=%di,iteration=%di %d",
			EscapeLp(cfg.Model), EscapeLp(cfg.RunID), EscapeLp(label),
			entry.Accuracy, entry.Correct, entry.Total, iteration, ts,
		)
		lines = append(lines, line)
		totalPoints++

		// Per-category breakdown.
		for cat, block := range entry.ByCategory {
			var catAccuracy float64
			if block.Total > 0 {
				catAccuracy = float64(block.Correct) / float64(block.Total)
			}
			line := core.Sprintf(
				MeasurementCapabilityScore+",model=%s,run_id=%s,label=%s,category=%s accuracy=%.6f,correct=%di,total=%di,iteration=%di %d",
				EscapeLp(cfg.Model), EscapeLp(cfg.RunID), EscapeLp(label),
				EscapeLp(cat), catAccuracy, block.Correct, block.Total, iteration, ts,
			)
			lines = append(lines, line)
			totalPoints++
		}

		// Flush batch if needed.
		if len(lines) >= cfg.BatchSize {
			if rWrite := influx.WriteLp(lines); !rWrite.OK {
				return core.Fail(core.E("datapipe.ingestCapabilityScores", "write batch", rWrite.Value.(error)))
			}
			lines = lines[:0]
		}
	}

	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("datapipe.ingestCapabilityScores", core.Sprintf("scan %s", cfg.CapabilityFile), err))
	}

	// Flush remaining lines.
	if len(lines) > 0 {
		if rWrite := influx.WriteLp(lines); !rWrite.OK {
			return core.Fail(core.E("datapipe.ingestCapabilityScores", "write final batch", rWrite.Value.(error)))
		}
	}

	core.Print(w, "  capability scores: %d points from %d lines", totalPoints, lineNum)
	return core.Ok(totalPoints)
}

// ingestTrainingLog reads an MLX LoRA training log and writes training_loss
// measurements to InfluxDB for both training and validation loss entries.
func ingestTrainingLog(influx *InfluxClient, cfg IngestConfig, w io.Writer) core.Result {
	f, err := coreio.Local.Open(cfg.TrainingLog)
	if err != nil {
		return core.Fail(core.E("datapipe.ingestTrainingLog", core.Sprintf("open %s", cfg.TrainingLog), err))
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// 64 KiB initial buffer, grown on demand up to a 1 MiB max line. Avoids
	// allocating a full 1 MiB up front on every call when lines are small
	// (the common case) while still admitting lines up to 1 MiB.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var lines []string
	var totalPoints int
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		text := scanner.Text()

		// Try validation loss first (shorter regex, less common).
		if m := reValLoss.FindStringSubmatch(text); m != nil {
			iter, _ := strconv.Atoi(m[1])
			loss, _ := strconv.ParseFloat(m[2], 64)
			ts := time.Now().UnixNano()

			line := core.Sprintf(
				MeasurementTrainingLoss+",model=%s,run_id=%s,loss_type=val loss=%.6f,iteration=%di %d",
				EscapeLp(cfg.Model), EscapeLp(cfg.RunID), loss, iter, ts,
			)
			lines = append(lines, line)
			totalPoints++
		}

		// Try training loss.
		if m := reTrainLoss.FindStringSubmatch(text); m != nil {
			iter, _ := strconv.Atoi(m[1])
			loss, _ := strconv.ParseFloat(m[2], 64)
			lr, _ := strconv.ParseFloat(m[3], 64)
			itPerSec, _ := strconv.ParseFloat(m[4], 64)
			tokPerSec, _ := strconv.ParseFloat(m[5], 64)
			ts := time.Now().UnixNano()

			line := core.Sprintf(
				MeasurementTrainingLoss+",model=%s,run_id=%s,loss_type=train loss=%.6f,iteration=%di,learning_rate=%.10f,it_per_sec=%.4f,tokens_per_sec=%.2f %d",
				EscapeLp(cfg.Model), EscapeLp(cfg.RunID), loss, iter, lr, itPerSec, tokPerSec, ts,
			)
			lines = append(lines, line)
			totalPoints++
		}

		// Flush batch if needed.
		if len(lines) >= cfg.BatchSize {
			if rWrite := influx.WriteLp(lines); !rWrite.OK {
				return core.Fail(core.E("datapipe.ingestTrainingLog", "write batch", rWrite.Value.(error)))
			}
			lines = lines[:0]
		}
	}

	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("datapipe.ingestTrainingLog", core.Sprintf("scan %s", cfg.TrainingLog), err))
	}

	// Flush remaining lines.
	if len(lines) > 0 {
		if rWrite := influx.WriteLp(lines); !rWrite.OK {
			return core.Fail(core.E("datapipe.ingestTrainingLog", "write final batch", rWrite.Value.(error)))
		}
	}

	core.Print(w, "  training log: %d points from %d lines", totalPoints, lineNum)
	return core.Ok(totalPoints)
}

// extractIteration extracts an iteration number from a label like "model@200".
// Returns 0 if no iteration is found.
func extractIteration(label string) int {
	idx := -1
	for i := len(label) - 1; i >= 0; i-- {
		if label[i] == '@' {
			idx = i
			break
		}
	}
	if idx < 0 || idx+1 >= len(label) {
		return 0
	}
	n, err := strconv.Atoi(label[idx+1:])
	if err != nil {
		return 0
	}
	return n
}

// toFloat64 converts a JSON-decoded any value to float64.
// Handles float64 (standard json.Unmarshal), json.Number, and string values.
func toFloat64(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case string:
		f, err := strconv.ParseFloat(val, 64)
		return f, err == nil
	default:
		return 0, false
	}
}
