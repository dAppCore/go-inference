// SPDX-License-Identifier: EUPL-1.2

package mcp

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/capability"
	"dappco.re/go/inference/datapipe"
	"dappco.re/go/inference/mlservice"
	"dappco.re/go/inference/score"
	"dappco.re/go/inference/serving"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MLSubsystem exposes ML inference and scoring tools via MCP.
// Usage example: subsystem := mcp.NewMLSubsystem(service)
type MLSubsystem struct {
	service *mlservice.Service
	logger  *core.Log
}

// NewMLSubsystem creates an MCP subsystem for ML tools.
// Usage example: server.AddSubsystem(mcp.NewMLSubsystem(service))
func NewMLSubsystem(svc *mlservice.Service) *MLSubsystem {
	return &MLSubsystem{
		service: svc,
		logger:  core.Default(),
	}
}

// Name returns the subsystem identifier exposed to the MCP server.
func (m *MLSubsystem) Name() string { return "ml" }

// RegisterTools adds ML tools to the MCP server.
// Usage example: subsystem.RegisterTools(server)
func (m *MLSubsystem) RegisterTools(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{
		Name:        "ml_generate",
		Description: "Generate text via a configured ML inference backend.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input MLGenerateInput) (*mcp.CallToolResult, MLGenerateOutput, error) {
		result := m.mlGenerate(ctx, req, input)
		if !result.OK {
			return nil, MLGenerateOutput{}, result.Value.(error)
		}
		return nil, result.Value.(MLGenerateOutput), nil
	})

	mcp.AddTool(server, &mcp.Tool{
		Name:        "ml_score",
		Description: "Score a prompt/response pair using heuristic and LLM judge suites.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input MLScoreInput) (*mcp.CallToolResult, MLScoreOutput, error) {
		result := m.mlScore(ctx, req, input)
		if !result.OK {
			return nil, MLScoreOutput{}, result.Value.(error)
		}
		return nil, result.Value.(MLScoreOutput), nil
	})

	mcp.AddTool(server, &mcp.Tool{
		Name:        "ml_probe",
		Description: "Run capability probes against an inference backend.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input MLProbeInput) (*mcp.CallToolResult, MLProbeOutput, error) {
		result := m.mlProbe(ctx, req, input)
		if !result.OK {
			return nil, MLProbeOutput{}, result.Value.(error)
		}
		return nil, result.Value.(MLProbeOutput), nil
	})

	mcp.AddTool(server, &mcp.Tool{
		Name:        "ml_status",
		Description: "Show training and generation progress from InfluxDB.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input MLStatusInput) (*mcp.CallToolResult, MLStatusOutput, error) {
		result := m.mlStatus(ctx, req, input)
		if !result.OK {
			return nil, MLStatusOutput{}, result.Value.(error)
		}
		return nil, result.Value.(MLStatusOutput), nil
	})

	mcp.AddTool(server, &mcp.Tool{
		Name:        "ml_backends",
		Description: "List available inference backends and their status.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input MLBackendsInput) (*mcp.CallToolResult, MLBackendsOutput, error) {
		result := m.mlBackends(ctx, req, input)
		if !result.OK {
			return nil, MLBackendsOutput{}, result.Value.(error)
		}
		return nil, result.Value.(MLBackendsOutput), nil
	})
}

// Shutdown implements mcp.SubsystemWithShutdown.
func (m *MLSubsystem) Shutdown(_ context.Context) core.Result { return core.Ok(nil) }

// --- Input/Output types ---

// MLGenerateInput contains parameters for text generation.
type MLGenerateInput struct {
	Prompt      string  `json:"prompt"`                // The prompt to generate from
	Backend     string  `json:"backend,omitempty"`     // Backend name (default: service default)
	Model       string  `json:"model,omitempty"`       // Model override
	Temperature float64 `json:"temperature,omitempty"` // Sampling temperature
	MaxTokens   int     `json:"max_tokens,omitempty"`  // Maximum tokens to generate
}

// MLGenerateOutput contains the generation result.
type MLGenerateOutput struct {
	Response string `json:"response"`
	Backend  string `json:"backend"`
	Model    string `json:"model,omitempty"`
}

// MLScoreInput contains parameters for scoring a response.
type MLScoreInput struct {
	Prompt   string `json:"prompt"`           // The original prompt
	Response string `json:"response"`         // The model response to score
	Suites   string `json:"suites,omitempty"` // Comma-separated suites (default: heuristic)
}

// MLScoreOutput contains the scoring result.
type MLScoreOutput struct {
	Heuristic *score.HeuristicScores `json:"heuristic,omitempty"`
	Semantic  *score.SemanticScores  `json:"semantic,omitempty"`
	Content   *score.ContentScores   `json:"content,omitempty"`
}

// MLProbeInput contains parameters for running probes.
type MLProbeInput struct {
	Backend    string `json:"backend,omitempty"`    // Backend name
	Categories string `json:"categories,omitempty"` // Comma-separated categories to run
}

// MLProbeOutput contains probe results.
type MLProbeOutput struct {
	Total   int                 `json:"total"`
	Results []MLProbeResultItem `json:"results"`
}

// MLProbeResultItem is a single probe result.
type MLProbeResultItem struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Response string `json:"response"`
}

// MLStatusInput contains parameters for the status query.
type MLStatusInput struct {
	InfluxURL string `json:"influx_url,omitempty"` // InfluxDB URL override
	InfluxDB  string `json:"influx_db,omitempty"`  // InfluxDB database override
}

// MLStatusOutput contains pipeline status.
type MLStatusOutput struct {
	Status string `json:"status"`
}

// MLBackendsInput is empty — lists all backends.
type MLBackendsInput struct{}

// MLBackendsOutput lists available backends.
type MLBackendsOutput struct {
	Backends []MLBackendInfo `json:"backends"`
	Default  string          `json:"default"`
}

// MLBackendInfo describes a single backend.
type MLBackendInfo struct {
	Name         string   `json:"name"`
	Available    bool     `json:"available"`
	Capabilities []string `json:"capabilities,omitempty"`
	Native       bool     `json:"native,omitempty"`
}

// --- Tool handlers ---

// mlGenerate delegates to go-mlservice.Service.Generate, which internally uses
// InferenceAdapter to route generation through an inference.TextModel.
func (m *MLSubsystem) mlGenerate(ctx context.Context, _ *mcp.CallToolRequest, input MLGenerateInput) core.Result {
	if m.logger != nil {
		m.logger.Info("MCP tool execution", "tool", "ml_generate", "backend", input.Backend, "user", core.Username())
	}

	if input.Prompt == "" {
		return core.Fail(core.E("mcp.MLSubsystem.mlGenerate", "prompt cannot be empty", nil))
	}

	opts := serving.GenOpts{
		Temperature: input.Temperature,
		MaxTokens:   input.MaxTokens,
		Model:       input.Model,
	}

	generateResult := m.service.Generate(ctx, input.Backend, input.Prompt, opts)
	if !generateResult.OK {
		return core.Fail(core.E("mcp.MLSubsystem.mlGenerate", "generate", generateResult.Value.(error)))
	}
	result := generateResult.Value.(serving.Result)

	return core.Ok(MLGenerateOutput{
		Response: result.Text,
		Backend:  input.Backend,
		Model:    input.Model,
	})
}

func (m *MLSubsystem) mlScore(ctx context.Context, _ *mcp.CallToolRequest, input MLScoreInput) core.Result {
	if m.logger != nil {
		m.logger.Info("MCP tool execution", "tool", "ml_score", "suites", input.Suites, "user", core.Username())
	}

	if input.Prompt == "" || input.Response == "" {
		return core.Fail(core.E("mcp.MLSubsystem.mlScore", "prompt and response cannot be empty", nil))
	}

	suites := input.Suites
	if suites == "" {
		suites = "heuristic"
	}

	output := MLScoreOutput{}

	for _, suite := range core.Split(suites, ",") {
		suite = core.Trim(suite)
		switch suite {
		case "heuristic":
			output.Heuristic = score.ScoreHeuristic(input.Response)
		case "semantic":
			judge := m.service.Judge()
			if judge == nil {
				return core.Fail(core.E("mcp.MLSubsystem.mlScore", "semantic scoring requires a judge backend", nil))
			}
			scoreResult := judge.ScoreSemantic(ctx, input.Prompt, input.Response)
			if !scoreResult.OK {
				return core.Fail(core.E("mcp.MLSubsystem.mlScore", "semantic score", scoreResult.Value.(error)))
			}
			output.Semantic = scoreResult.Value.(*score.SemanticScores)
		case "content":
			return core.Fail(core.E("mcp.MLSubsystem.mlScore", "content scoring requires a ContentProbe — use ml_probe instead", nil))
		}
	}

	return core.Ok(output)
}

// mlProbe runs capability probes by generating responses via go-mlservice.Service.
func (m *MLSubsystem) mlProbe(ctx context.Context, _ *mcp.CallToolRequest, input MLProbeInput) core.Result {
	if m.logger != nil {
		m.logger.Info("MCP tool execution", "tool", "ml_probe", "backend", input.Backend, "user", core.Username())
	}

	probes := capability.CapabilityProbes
	if input.Categories != "" {
		cats := make(map[string]bool)
		for _, c := range core.Split(input.Categories, ",") {
			cats[core.Trim(c)] = true
		}
		var filtered []capability.Probe
		for _, p := range probes {
			if cats[p.Category] {
				filtered = append(filtered, p)
			}
		}
		probes = filtered
	}

	var results []MLProbeResultItem
	for _, probe := range probes {
		result := m.service.Generate(ctx, input.Backend, probe.Prompt, serving.GenOpts{Temperature: 0.7, MaxTokens: 2048})
		respText := ""
		if result.OK {
			respText = result.Value.(serving.Result).Text
		} else {
			respText = core.Sprintf("error: %s", result.Error())
		}
		results = append(results, MLProbeResultItem{
			ID:       probe.ID,
			Category: probe.Category,
			Response: respText,
		})
	}

	return core.Ok(MLProbeOutput{
		Total:   len(results),
		Results: results,
	})
}

func (m *MLSubsystem) mlStatus(ctx context.Context, _ *mcp.CallToolRequest, input MLStatusInput) core.Result {
	if m.logger != nil {
		m.logger.Info("MCP tool execution", "tool", "ml_status", "user", core.Username())
	}

	url := input.InfluxURL
	db := input.InfluxDB
	if url == "" {
		url = "http://localhost:8086"
	}
	if db == "" {
		db = "lem"
	}

	influx := datapipe.NewInfluxClient(url, db)
	buf := core.NewBuilder()
	if result := mlservice.PrintStatus(influx, buf); !result.OK {
		return core.Fail(core.E("mcp.MLSubsystem.mlStatus", "status", result.Value.(error)))
	}

	return core.Ok(MLStatusOutput{Status: buf.String()})
}

// mlBackends enumerates registered backends via the go-inference registry.
func (m *MLSubsystem) mlBackends(ctx context.Context, _ *mcp.CallToolRequest, input MLBackendsInput) core.Result {
	if m.logger != nil {
		m.logger.Info("MCP tool execution", "tool", "ml_backends", "user", core.Username())
	}

	names := inference.List()
	backends := make([]MLBackendInfo, 0, len(names))
	for _, name := range names {
		b, ok := inference.Get(name)
		info := MLBackendInfo{Name: name, Available: ok && b.Available()}
		if ok {
			report, _ := inference.CapabilitiesOf(b)
			info.Capabilities = capabilityIDStrings(report.SupportedCapabilityIDs())
			info.Native = report.Runtime.NativeRuntime
		}
		backends = append(backends, info)
	}

	defaultName := ""
	if result := inference.Default(); result.OK {
		if backend, ok := result.Value.(inference.Backend); ok && backend != nil {
			defaultName = backend.Name()
		}
	}

	return core.Ok(MLBackendsOutput{
		Backends: backends,
		Default:  defaultName,
	})
}

func capabilityIDStrings(ids []inference.CapabilityID) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = string(id)
	}
	return out
}
