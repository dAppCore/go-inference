// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/generate"
	native "dappco.re/go/inference/engine/metal"
)

// stringListFlag is a repeatable string flag: each occurrence appends, so
// -image a.png -image b.png collects both paths. The zero value is an empty
// list, which the usage dump renders with no default (matching the other
// no-default flags).
type stringListFlag []string

func (s *stringListFlag) String() string {
	if s == nil {
		return ""
	}
	return core.Join(",", []string(*s)...)
}

func (s *stringListFlag) Set(value string) error {
	*s = append(*s, value)
	return nil
}

// runGenerateCommand parses the generate flags and hands them to
// generate.RunGenerate. Thin: flag parsing + one library call + exit mapping.
// All generate business logic (load, warm, timed decode + tok/s, the durable
// -state turn loop, reactive drafter resolution) lives in
// dappco.re/go/inference/generate.
func runGenerateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("generate"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	prompt := fs.String("prompt", "Write a detailed Go function that reverses a singly linked list, with inline comments on every step, then explain the pointer dance.", "user prompt")
	promptFile := fs.String("prompt-file", "", "read the user prompt from a file (long-context runs exceed argv limits); overrides -prompt")
	maxTokens := fs.Int("max-tokens", 128, "tokens to generate")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside a Gemma 4 target (assistant/ pair layout, MTP/ gguf), a path forces it, '' disables")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block (verify forward = carried lead + block-1 proposals); 0 = engine default 4")
	temp := fs.Float64("temp", 1.0, "sampling temperature (0 = greedy/argmax — fastest, fair vs llama-bench)")
	think := fs.Bool("think", false, "enable the thinking channel (off keeps the decode rate clean)")
	contextLen := fs.Int("context", 0, "context length override (0 = model default)")
	kvCacheMode := fs.String("kv-cache", "", "KV cache mode override (engine-reported; the no-cgo metal engine runs its built-in cache only — other values are noted and ignored)")
	pipeline := fs.Bool("pipeline", true, "one-ahead pipelined decode (the engine default; false forces the chained serial loop, for A/B traces)")
	kvStorage := fs.String("kv-storage", "", "KV snapshot encoding for -state sleeps (native, q8, float32; empty = native) — inert without -state")
	tracePhases := fs.Bool("trace", false, "print the per-token decode time budget — GPU wait vs host-serial work")
	nativeBackend := fs.Bool("native", false, "generate via the no-cgo native token-loop contract (the default go-inference metal engine already is)")
	stateName := fs.String("state", "", "conversation state name: wake it from the store if present, generate, sleep it back — the no-prompt-replay turn loop")
	stateStore := fs.String("state-store", "", "state store file (default ~/Lethean/lem/state/agent.kv)")
	rawState := fs.Bool("raw", false, "with -state: skip chat-framing and run the raw completion-loop turn (no template) — ignored without -state")
	var images stringListFlag
	fs.Var(&images, "image", "image input for a vision model: a local PNG/JPEG path or a base64 data: URL (repeatable) — gated on the model's vision capability, same as serve")
	var audio stringListFlag
	var videoFrames stringListFlag
	fs.Var(&audio, "audio", "audio input for an audio model: a local WAV path (16-bit PCM mono 16 kHz) or a base64 data: URL (repeatable) — gated on the model's audio capability")
	fs.Var(&videoFrames, "video-frame", "one sampled video frame in time order (repeatable): a local PNG/JPEG path or a base64 data: URL — frames become timestamped vision blocks 1s apart")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s generate [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Load a model and generate from a prompt with no HTTP serve in the path,\n")
		core.WriteString(stderr, "reporting decode-only tok/s (prefill excluded) for like-for-like benching.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s generate ~/models/gemma-4-e2b-it-4bit\n", name))
		core.WriteString(stderr, "    # one-shot generate + decode tok/s\n")
		core.WriteString(stderr, core.Sprintf("  %s generate -state chat1 -prompt \"Hello, who are you?\" ~/models/gemma-4-e2b-it-4bit\n", name))
		core.WriteString(stderr, "    # a durable conversation turn (wake -> generate -> sleep)\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s generate: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	promptText := *prompt
	if *promptFile != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s generate: read -prompt-file %s: %s", cliName(), *promptFile, read.Error())
			return 1
		}
		bytes, ok := read.Value.([]byte)
		if !ok || len(bytes) == 0 {
			core.Print(stderr, "%s generate: -prompt-file %s is empty", cliName(), *promptFile)
			return 1
		}
		promptText = string(bytes)
	}

	native.SetPipelinedGPUDecode(*pipeline) // engine-level: -pipeline=false forces the chained serial loop
	err := generate.RunGenerate(ctx, generate.Config{
		ModelPath:  fs.Arg(0),
		Prompt:     promptText,
		MaxTokens:  *maxTokens,
		Temp:       *temp,
		Think:      *think,
		ContextLen: *contextLen,
		DraftPath:  *draftPath,
		DraftBlock: *draftBlock,
		// Inject the metal engine's speculative loader so a detected drafter arms
		// the MTP lane instead of degrading to plain — the composition root is the
		// one place that may import the engine (keeps decode/generate neutral).
		SpeculativeLoader: native.LoadSpeculativePair,
		KVCacheMode:       *kvCacheMode,
		KVStorage:         *kvStorage,
		Pipeline:          *pipeline,
		Native:            *nativeBackend,
		Trace:             *tracePhases,
		StateName:         *stateName,
		StateStore:        *stateStore,
		Raw:               *rawState,
		ImageSources:      images,
		AudioSources:      audio,
		VideoFrameSources: videoFrames,
		Out:               stdout,
		Log:               stderr,
	})
	if err != nil {
		core.Print(stderr, "%s generate: %v", cliName(), err)
		return 1
	}
	return 0
}
