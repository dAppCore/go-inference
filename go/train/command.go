// SPDX-Licence-Identifier: EUPL-1.2

// command.go: the cmd-facing SSD/SFT business logic, rescued out of go-mlx's
// cmd/mlx {ssd,sft} commands so it lives in a go-inference library rather than
// dying with go-mlx's cmd/ — the same move dappco.re/go/inference/generate made
// for the generate verb. cmd/lem {ssd,sft} are thin flag-parsing over
// RunSSDCommand / RunSFTCommand.
//
// These load a model (through the registered backend), a JSONL dataset, and —
// for SFT — the model's tokeniser, then drive RunSSDModel / RunSFTModel and
// print a summary. No engine type appears; the blank-imported backend in the
// cmd binary supplies the concrete engine.

package train

import (
	"context"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/train/dataset"
	coreio "dappco.re/go/io"
)

// SSDCommandConfig is the declarative `lem ssd` request mirroring go-mlx's ssd
// flag surface. RunSSDCommand turns it into a load + self-distillation sampling
// run that stops at the scored trace.
type SSDCommandConfig struct {
	ModelPath     string
	DataPath      string
	KernelPath    string
	CheckpointDir string
	ContextLen    int

	SampleMaxTokens   int
	SampleTemp        float64
	SampleTopK        int
	SampleTopP        float64
	SampleMinP        float64
	RepetitionPenalty float64
	FilterShortest    float64
	ScoreSamples      bool

	Out io.Writer // summary
	Log io.Writer // notices
}

// RunSSDCommand loads the frozen base and samples it over the prompt set,
// capturing (and, when a scorer is available, scoring) each self-output at
// birth. It is the ssd business logic ported out of go-mlx's cmd/mlx.
func RunSSDCommand(ctx context.Context, cfg SSDCommandConfig) error {
	if core.Trim(cfg.ModelPath) == "" || core.Trim(cfg.DataPath) == "" {
		return core.NewError("ssd: --model and --data are required")
	}
	ds, err := loadJSONLDataset(cfg.DataPath)
	if err != nil {
		return core.E("train.RunSSDCommand", "prompt data", err)
	}
	var kernel string
	if cfg.KernelPath != "" {
		text, rerr := coreio.Local.Read(cfg.KernelPath)
		if rerr != nil {
			return core.E("train.RunSSDCommand", "kernel unreadable", rerr)
		}
		kernel = text // verbatim — no normalisation (#97)
	}

	printNote(cfg.Log, "ssd: loading %s", cfg.ModelPath)
	model, err := loadTextModel(cfg.ModelPath, cfg.ContextLen)
	if err != nil {
		return core.E("train.RunSSDCommand", "model load", err)
	}
	defer model.Close()

	// The lem-scorer (go-mlx mlx/pkg/score, phonetics/cmudict) has no
	// go-inference home yet, so sampling-phase scoring is unavailable here —
	// honest note, then proceed with capture-only (the capture IS the
	// deliverable; scoring later over the captured trace is archaeology).
	if cfg.ScoreSamples {
		printNote(cfg.Log, "ssd: --score-samples requested but no scorer is wired into go-inference yet; capturing the trace without birth-scores (score the captures later)")
	}

	ssdCfg := SSDConfig{
		SampleMaxTokens:       cfg.SampleMaxTokens,
		SampleTemperature:     float32(cfg.SampleTemp),
		SampleTopK:            cfg.SampleTopK,
		SampleTopP:            float32(cfg.SampleTopP),
		SampleMinP:            float32(cfg.SampleMinP),
		RepetitionPenalty:     float32(cfg.RepetitionPenalty),
		FilterShortestPercent: float32(cfg.FilterShortest),
		CheckpointDir:         cfg.CheckpointDir,
		KernelPrefix:          kernel,
		// ScoreSamples stays false — no Score hook is available (see the note
		// above); RunSSD would no-op the cascade regardless.
	}

	result, err := RunSSDModel(ctx, model, ds, ssdCfg, nil)
	if err != nil {
		return core.E("train.RunSSDCommand", "self-distillation", err)
	}

	core.Print(cfg.Out, "self-samples %d  sample-temp %.2f  kernel %v", len(result.Samples), result.SampleTemperature, result.KernelApplied)
	if result.CaptureSidecar != "" {
		core.Print(cfg.Out, "ssd trace %s  (the lab picks steps from this)", result.CaptureSidecar)
	}
	core.Print(cfg.Out, "next: refine the trace in the lab, then  lem sft --data <artifact> --model %s", cfg.ModelPath)
	return nil
}

// SFTCommandConfig is the declarative `lem sft` request mirroring go-mlx's sft
// flag surface. RunSFTCommand turns it into a load + LoRA SFT training run.
type SFTCommandConfig struct {
	ModelPath       string
	DataPath        string
	ValidPath       string
	EvalPromptsPath string
	CheckpointDir   string
	SavePath        string
	ResumePath      string
	ContextLen      int

	Rank            int
	Alpha           float64
	LearningRate    float64
	Epochs          int
	BatchSize       int
	GradAccum       int
	MaxSeqLen       int
	Packing         bool
	Merge           bool
	EvalEvery       int
	EvalMaxTokens   int
	EvalProbes      int
	EvalTemp        float64
	CheckpointEvery int
	ScoreCascade    bool
	ScoreWindow     int

	Out io.Writer // summary
	Log io.Writer // notices
}

// RunSFTCommand loads the model + its tokeniser + the training set and runs the
// LoRA SFT loop through the engine trainer seam. It is the sft business logic
// ported out of go-mlx's cmd/mlx.
func RunSFTCommand(ctx context.Context, cfg SFTCommandConfig) error {
	if core.Trim(cfg.ModelPath) == "" || core.Trim(cfg.DataPath) == "" {
		return core.NewError("sft: --model and --data are required")
	}

	// Eval probes: an explicit file wins; otherwise the first user turns of the
	// validation set (fixed across the run — the cascade compares like with
	// like).
	prompts, err := sftEvalProbes(cfg)
	if err != nil {
		return core.E("train.RunSFTCommand", "eval probes", err)
	}

	ds, err := loadJSONLDataset(cfg.DataPath)
	if err != nil {
		return core.E("train.RunSFTCommand", "training data", err)
	}

	printNote(cfg.Log, "sft: loading %s", cfg.ModelPath)
	model, err := loadTextModel(cfg.ModelPath, cfg.ContextLen)
	if err != nil {
		return core.E("train.RunSFTCommand", "model load", err)
	}
	defer model.Close()

	tok, terr := tokenizer.LoadTokenizer(core.PathJoin(cfg.ModelPath, "tokenizer.json"))
	if terr != nil {
		return core.E("train.RunSFTCommand", "tokenizer load", terr)
	}

	save := cfg.SavePath
	if save == "" && cfg.CheckpointDir != "" {
		save = core.PathJoin(cfg.CheckpointDir, "adapter")
	}

	sftCfg := SFTConfig{
		Config: Config{
			BatchSize:                 cfg.BatchSize,
			GradientAccumulationSteps: cfg.GradAccum,
			Epochs:                    cfg.Epochs,
			LearningRate:              cfg.LearningRate,
			MaxSeqLen:                 cfg.MaxSeqLen,
			SequencePacking:           cfg.Packing,
			CheckpointDir:             cfg.CheckpointDir,
			CheckpointEvery:           cfg.CheckpointEvery,
			EvalEvery:                 cfg.EvalEvery,
			EvalPrompts:               prompts,
			EvalMaxTokens:             cfg.EvalMaxTokens,
			EvalTemperature:           float32(cfg.EvalTemp),
			ResumePath:                cfg.ResumePath,
		},
		LoRA:        inference.LoRAConfig{Rank: cfg.Rank, Alpha: float32(cfg.Alpha)},
		SavePath:    save,
		Merge:       cfg.Merge,
		ScoreWindow: cfg.ScoreWindow,
		// ScoreCascade needs a scorer hook; none is wired into go-inference yet
		// (see RunSSDCommand's note), so the cascade stays off. Capture is on.
	}
	if cfg.ScoreCascade {
		printNote(cfg.Log, "sft: --score-cascade requested but no scorer is wired into go-inference yet; eval generations are captured but not scored")
	}
	if cfg.Packing {
		printNote(cfg.Log, "sft: --packing has no effect on the head-LoRA trainer (it trains one sequence per row); ignored")
	}
	if cfg.Merge {
		printNote(cfg.Log, "sft: --merge is not supported by the head-LoRA trainer; the adapter is saved separately and applied via --adapter at load")
	}

	result, err := RunSFTModel(ctx, model, tok.Encode, ds, sftCfg)
	if err != nil {
		return core.E("train.RunSFTCommand", "training", err)
	}

	core.Print(cfg.Out, "steps %d  epochs %d  samples %d  last-loss %.4f", result.Steps, result.Epochs, result.Samples, result.LastLoss)
	if result.AdapterPath != "" {
		core.Print(cfg.Out, "adapter %s", result.AdapterPath)
	}
	for _, cp := range result.Checkpoints {
		core.Print(cfg.Out, "checkpoint %s", cp)
	}
	return nil
}

// sftEvalProbes resolves the fixed eval probe set from an explicit file or the
// validation set's first user turns.
func sftEvalProbes(cfg SFTCommandConfig) ([]string, error) {
	switch {
	case cfg.EvalPromptsPath != "":
		text, err := coreio.Local.Read(cfg.EvalPromptsPath)
		if err != nil {
			return nil, err
		}
		var prompts []string
		for _, line := range core.Split(text, "\n") {
			if trimmed := core.Trim(line); trimmed != "" {
				prompts = append(prompts, trimmed)
			}
		}
		return prompts, nil
	case cfg.ValidPath != "":
		return sftProbesFromValid(cfg.ValidPath, cfg.EvalProbes)
	default:
		return nil, nil
	}
}

// sftProbesFromValid derives the fixed eval probe set: the first n distinct user
// turns of the validation JSONL. Ported from go-mlx's cmd/mlx sftProbesFromValid.
func sftProbesFromValid(path string, n int) ([]string, error) {
	if n <= 0 {
		n = 4
	}
	text, err := coreio.Local.Read(path)
	if err != nil {
		return nil, err
	}
	var prompts []string
	for _, line := range core.Split(text, "\n") {
		if len(prompts) >= n {
			break
		}
		line = core.Trim(line)
		if line == "" {
			continue
		}
		var row struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if r := core.JSONUnmarshalString(line, &row); !r.OK {
			continue
		}
		for _, msg := range row.Messages {
			if msg.Role == "user" && core.Trim(msg.Content) != "" {
				prompts = append(prompts, msg.Content)
				break
			}
		}
	}
	if len(prompts) == 0 {
		return nil, core.NewError("mlx: no user turns found in validation set")
	}
	return prompts, nil
}

// loadJSONLDataset opens a JSONL file and parses it into a replayable dataset.
func loadJSONLDataset(path string) (*dataset.JSONLDataset, error) {
	file, err := coreio.Local.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	return dataset.LoadJSONL(file)
}

// loadTextModel loads path through the registered backend as an
// inference.TextModel, applying a context-length override when set.
func loadTextModel(path string, contextLen int) (inference.TextModel, error) {
	var opts []inference.LoadOption
	if contextLen > 0 {
		opts = append(opts, inference.WithContextLen(contextLen))
	}
	result := inference.LoadModel(path, opts...)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return nil, err
		}
		return nil, core.NewError("train: backend failed to load model")
	}
	model, ok := result.Value.(inference.TextModel)
	if !ok || model == nil {
		return nil, core.NewError("train: backend returned a non-TextModel value")
	}
	return model, nil
}

// printNote writes a notice to w (nil silences it).
func printNote(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	core.Print(w, format, args...)
}
