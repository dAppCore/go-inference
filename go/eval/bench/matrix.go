// SPDX-Licence-Identifier: EUPL-1.2

package bench

import (
	"context"
	"io"
	"time"

	core "dappco.re/go"
)

// matrix.go is the multi-model layer over the single-model harness: a bench MATRIX is a list of
// (model, lanes) runs driven from one JSON config, so the benchmark grid is DATA — locked in for the
// model family that matters (bench/gemma4.json at the repo root) while remaining shiftable by editing
// the config, never the code. The same verb + config reproduce the grid on any box the lem binary
// builds for (the metal Mac, the hip AMD box), which is what makes the numbers comparable over time.
// Driver-neutral like the rest of this package: the CLI composition root supplies the loader; this
// file owns the loop, the timing discipline (warmup excluded, greedy, tokens/wall from the driver's
// own metrics) and the report.

// MatrixLanePlain and MatrixLaneMTP are the two lanes a run can measure: plain autoregressive decode,
// and the MTP speculative pair (which needs a drafter ref).
const (
	MatrixLanePlain = "plain"
	MatrixLaneMTP   = "mtp"
)

// MatrixRun is one model entry in the bench matrix.
type MatrixRun struct {
	// Name labels the row; empty defaults to the model ref's base name.
	Name string `json:"name,omitempty"`
	// Model is the target checkpoint: an absolute snapshot path, or a Hugging Face cache repo ref
	// ("org/name" or "org--name") resolved against the local HF cache — the portable spelling, so
	// one config runs on every box sharing the cache convention.
	Model string `json:"model"`
	// Draft is the MTP drafter ref for the mtp lane, same forms as Model. Empty = plain only.
	// EXPLICIT by design: drafter auto-detection ambiguity is what broke the shell sweep this verb
	// replaces, and the matrix receipts showed drafter/base quant matching is load-bearing.
	Draft string `json:"draft,omitempty"`
	// DraftBlock overrides the MTP draft block; 0 = engine default.
	DraftBlock int `json:"draft_block,omitempty"`
	// Lanes selects what to measure; empty defaults to plain, plus mtp when Draft is set.
	Lanes []string `json:"lanes,omitempty"`
	// Tokens overrides the matrix-level token budget for this run; 0 inherits.
	Tokens int `json:"tokens,omitempty"`
}

// MatrixConfig is the bench.json shape: the shared prompt/budgets plus the runs array.
type MatrixConfig struct {
	// Prompt is the generation prompt; the default is a short neutral instruction (the tg-N
	// method measures decode off a small prompt, prefill excluded).
	Prompt string `json:"prompt,omitempty"`
	// Tokens is the per-lane generation budget (the N of tg-N); default 512.
	Tokens int `json:"tokens,omitempty"`
	// Warmup is the untimed warmup generation's token budget; default 16, 0 keeps the default
	// (warmup is never skipped — first generations pay one-off pipeline/cache costs).
	Warmup int         `json:"warmup,omitempty"`
	Runs   []MatrixRun `json:"runs"`
}

// matrixDefaults fills the zero-value budgets.
func (c *MatrixConfig) matrixDefaults() {
	if c.Prompt == "" {
		// The house tg-N prompt: maximally predictable output, so the MTP lane's acceptance reads
		// at its ceiling — comparable with the historical reference tables and the cross-engine
		// numbers. Acceptance is WORKLOAD-DEPENDENT (this prompt ≈ the upper bound; creative text
		// measured 8-20% on the same pair) — shift the prompt in the config to bench a workload.
		c.Prompt = "Write the integers from 1 to 800 separated by single spaces. Output only the numbers, nothing else."
	}
	if c.Tokens <= 0 {
		c.Tokens = 512
	}
	if c.Warmup <= 0 {
		c.Warmup = 16
	}
}

// LoadMatrixConfig parses and validates a bench.json. Every run must name a model; lane names must
// be plain/mtp; an mtp lane requires a draft ref. Defaults are applied (prompt, 512 tokens, warmup
// 16, per-run lane derivation).
func LoadMatrixConfig(data []byte) (MatrixConfig, error) {
	var cfg MatrixConfig
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return cfg, core.NewError("bench.LoadMatrixConfig: parse: " + r.Error())
	}
	cfg.matrixDefaults()
	if len(cfg.Runs) == 0 {
		return cfg, core.NewError("bench.LoadMatrixConfig: config has no runs")
	}
	for i := range cfg.Runs {
		run := &cfg.Runs[i]
		if core.Trim(run.Model) == "" {
			return cfg, core.NewError(core.Sprintf("bench.LoadMatrixConfig: run %d has no model", i))
		}
		if run.Name == "" {
			run.Name = core.PathBase(core.Trim(run.Model))
		}
		if len(run.Lanes) == 0 {
			run.Lanes = []string{MatrixLanePlain}
			if run.Draft != "" {
				run.Lanes = append(run.Lanes, MatrixLaneMTP)
			}
		}
		for _, lane := range run.Lanes {
			switch lane {
			case MatrixLanePlain:
			case MatrixLaneMTP:
				if run.Draft == "" {
					return cfg, core.NewError(core.Sprintf("bench.LoadMatrixConfig: run %q asks for the mtp lane without a draft", run.Name))
				}
			default:
				return cfg, core.NewError(core.Sprintf("bench.LoadMatrixConfig: run %q has unknown lane %q (plain|mtp)", run.Name, lane))
			}
		}
	}
	return cfg, nil
}

// hfCacheRoot is the local Hugging Face hub cache: $HF_HOME/hub when set, else ~/.cache/huggingface/hub.
func hfCacheRoot() string {
	if home := core.Getenv("HF_HOME"); home != "" {
		return core.PathJoin(home, "hub")
	}
	return core.PathJoin(core.Getenv("HOME"), ".cache", "huggingface", "hub")
}

// ResolveModelRef turns a matrix model/draft ref into a checkpoint directory: an existing path is
// used as-is; otherwise the ref is treated as an HF cache repo name ("org/name" or "org--name") and
// resolved to its first snapshot directory. The error names both spellings tried, so a missing
// snapshot reads as "pull it", not a mystery.
func ResolveModelRef(ref string) (string, error) {
	ref = core.Trim(ref)
	if ref == "" {
		return "", core.NewError("bench.ResolveModelRef: empty ref")
	}
	if core.Stat(ref).OK {
		return ref, nil
	}
	repo := core.Replace(ref, "/", "--")
	base := core.PathJoin(hfCacheRoot(), "models--"+repo, "snapshots")
	if snaps := core.PathGlob(core.PathJoin(base, "*")); len(snaps) > 0 {
		return snaps[0], nil
	}
	return "", core.NewError("bench.ResolveModelRef: " + ref + " is neither a path nor a cached HF repo (looked for " + base + "/*)")
}

// MatrixModel is the loaded-model surface one matrix lane drives — the minimal slice of a driver's
// text model the timing discipline needs. Drain runs ONE greedy generation of maxTokens and discards
// the text; Metrics is valid after a Drain until the next one starts.
type MatrixModel interface {
	Drain(ctx context.Context, prompt string, maxTokens int) error
	Metrics() GenerationMetrics
	Close() error
}

// MatrixSpeculative is the optional speculative-counters probe a loader's mtp-lane model may
// expose; ok=false when the last generation ran plain.
type MatrixSpeculative interface {
	SpeculativeSummary() (MatrixSpec, bool)
}

// MatrixSpec is the mtp lane's acceptance summary.
type MatrixSpec struct {
	ProposedTokens int     `json:"proposed_tokens"`
	AcceptedTokens int     `json:"accepted_tokens"`
	AcceptanceRate float64 `json:"acceptance_rate"`
}

// MatrixLoad loads one run's model for one lane: draftPath is empty on the plain lane. The loader is
// the CLI composition root's concern (it owns the engine imports); the matrix owns everything else.
type MatrixLoad func(ctx context.Context, modelPath, draftPath string, draftBlock int) (MatrixModel, error)

// MatrixRow is one measured (run, lane) result. Err carries a per-row failure (missing snapshot,
// load error) without aborting the rest of the grid — a partial grid with honest holes beats none.
type MatrixRow struct {
	Name                string        `json:"name"`
	Lane                string        `json:"lane"`
	Model               string        `json:"model"`
	Draft               string        `json:"draft,omitempty"`
	GeneratedTokens     int           `json:"generated_tokens,omitempty"`
	DecodeTokensPerSec  float64       `json:"decode_tokens_per_sec,omitempty"`
	PrefillTokensPerSec float64       `json:"prefill_tokens_per_sec,omitempty"`
	Wall                time.Duration `json:"wall,omitempty"`
	PeakMemoryBytes     uint64        `json:"peak_memory_bytes,omitempty"`
	Spec                *MatrixSpec   `json:"speculative,omitempty"`
	Err                 string        `json:"error,omitempty"`
}

// MatrixReport is the whole grid: the resolved config plus every row, JSON-serialisable for the
// comparisons-over-time file the verb exists to feed.
type MatrixReport struct {
	Version int          `json:"version"`
	Config  MatrixConfig `json:"config"`
	Rows    []MatrixRow  `json:"rows"`
}

// RunMatrix executes the grid: for each run × lane it resolves the refs, loads through the supplied
// loader, runs one untimed warmup then one timed greedy generation, and records the driver-reported
// decode rate. Rows stream to out as they complete (a long grid shows progress); per-row failures
// are recorded and the grid continues. The context cancels between rows.
func RunMatrix(ctx context.Context, cfg MatrixConfig, load MatrixLoad, out io.Writer) (MatrixReport, error) {
	if load == nil {
		return MatrixReport{}, core.NewError("bench.RunMatrix: nil loader")
	}
	cfg.matrixDefaults()
	report := MatrixReport{Version: ReportVersion, Config: cfg}
	if out != nil {
		core.WriteString(out, core.Sprintf("%-22s %-6s %12s %12s %8s %9s  %s\n",
			"name", "lane", "decode tok/s", "prefill t/s", "tokens", "wall", "note"))
	}
	for _, run := range cfg.Runs {
		tokens := run.Tokens
		if tokens <= 0 {
			tokens = cfg.Tokens
		}
		lanes := run.Lanes
		if len(lanes) == 0 {
			// A directly-constructed config gets the same derivation LoadMatrixConfig applies:
			// plain always, plus the mtp lane when the run declares a draft.
			lanes = []string{MatrixLanePlain}
			if run.Draft != "" {
				lanes = append(lanes, MatrixLaneMTP)
			}
		}
		for _, lane := range lanes {
			if err := ctx.Err(); err != nil {
				return report, err
			}
			name := run.Name
			if name == "" {
				name = core.PathBase(core.Trim(run.Model)) // positional runs bypass LoadMatrixConfig's defaulting
			}
			row := MatrixRow{Name: name, Lane: lane, Model: run.Model, Draft: run.Draft}
			row = runMatrixLane(ctx, row, run, lane, tokens, cfg, load)
			report.Rows = append(report.Rows, row)
			if out != nil {
				core.WriteString(out, formatMatrixRow(row))
			}
		}
	}
	return report, nil
}

// runMatrixLane measures one (run, lane): resolve → load → warmup → timed drain → metrics.
func runMatrixLane(ctx context.Context, row MatrixRow, run MatrixRun, lane string, tokens int, cfg MatrixConfig, load MatrixLoad) MatrixRow {
	modelPath, err := ResolveModelRef(run.Model)
	if err != nil {
		row.Err = err.Error()
		return row
	}
	draftPath := ""
	if lane == MatrixLaneMTP {
		if draftPath, err = ResolveModelRef(run.Draft); err != nil {
			row.Err = err.Error()
			return row
		}
	}
	m, err := load(ctx, modelPath, draftPath, run.DraftBlock)
	if err != nil {
		row.Err = err.Error()
		return row
	}
	defer func() { _ = m.Close() }()
	if err := m.Drain(ctx, "Warm up.", cfg.Warmup); err != nil {
		row.Err = "warmup: " + err.Error()
		return row
	}
	start := time.Now()
	if err := m.Drain(ctx, cfg.Prompt, tokens); err != nil {
		row.Err = err.Error()
		return row
	}
	row.Wall = time.Since(start)
	mt := m.Metrics()
	row.GeneratedTokens = mt.GeneratedTokens
	row.DecodeTokensPerSec = mt.DecodeTokensPerSec
	row.PrefillTokensPerSec = mt.PrefillTokensPerSec
	row.PeakMemoryBytes = mt.PeakMemoryBytes
	if sp, ok := m.(MatrixSpeculative); ok {
		if spec, engaged := sp.SpeculativeSummary(); engaged {
			row.Spec = &spec
		}
	}
	return row
}

// formatMatrixRow renders one streamed table row.
func formatMatrixRow(r MatrixRow) string {
	if r.Err != "" {
		return core.Sprintf("%-22s %-6s %12s %12s %8s %9s  ERROR %s\n", r.Name, r.Lane, "-", "-", "-", "-", r.Err)
	}
	note := ""
	if r.Spec != nil {
		note = core.Sprintf("accept %.0f%% (%d/%d)", r.Spec.AcceptanceRate*100, r.Spec.AcceptedTokens, r.Spec.ProposedTokens)
	}
	return core.Sprintf("%-22s %-6s %12.1f %12.1f %8d %9s  %s\n",
		r.Name, r.Lane, r.DecodeTokensPerSec, r.PrefillTokensPerSec, r.GeneratedTokens, r.Wall.Round(time.Millisecond), note)
}
