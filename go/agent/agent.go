// SPDX-License-Identifier: EUPL-1.2

package agent

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/datapipe"
)

// Agent orchestrates model evaluation and fleet training. It wraps the
// underlying RunAgentLoop / DiscoverCheckpoints / SSH transport functions
// with a typed facade so callers never touch package-level state.
//
//	agent := agent.NewAgent(&agent.AgentConfig{
//	    M3Host:     "homelab.lthn.sh",
//	    M3User:     "snider",
//	    APIURL:     "http://127.0.0.1:11434",
//	    JudgeURL:   "http://127.0.0.1:11434",
//	    JudgeModel: "qwen3:8b",
//	    WorkDir:    "/tmp/scoring",
//	})
//	agent.Execute(ctx)
type Agent struct {
	cfg    *AgentConfig
	influx *datapipe.InfluxClient
}

// NewAgent creates a scoring/training Agent bound to the supplied config.
// The config is stored by pointer, so any later mutation by the caller is
// respected by subsequent method calls.
func NewAgent(cfg *AgentConfig) *Agent {
	return &Agent{cfg: cfg}
}

// Config returns the agent's underlying AgentConfig so callers can read or
// mutate fields (useful for tests).
func (a *Agent) Config() *AgentConfig { return a.cfg }

// Execute runs the full scoring/training loop (discovers checkpoints,
// scores them, pushes results to InfluxDB). Blocks until ctx is cancelled
// or cfg.OneShot is set. Optional override config replaces the agent's
// stored config for this call only — useful for single-shot variants.
//
//	agent.Execute(ctx)
//	agent.Execute(ctx, overrideConfig)   // one-shot config override
func (a *Agent) Execute(ctx context.Context, override ...*AgentConfig) {
	cfg := a.cfg
	if len(override) > 0 && override[0] != nil {
		cfg = override[0]
	}
	RunAgentLoop(cfg)
	_ = ctx
}

// Evaluate scores a single checkpoint (adapter) and returns the probe
// results without the full discovery loop. The target argument accepts
// either a Checkpoint (direct struct) or a model path string that is
// resolved via DiscoverCheckpoints. Spec §8.
//
//	r := agent.Evaluate(ctx, agent.Checkpoint{...})
//	if !r.OK { return r }
//	r = agent.Evaluate(ctx, "/models/adapter-42")
//	if !r.OK { return r }
func (a *Agent) Evaluate(ctx context.Context, target any) core.Result {
	if a == nil || a.cfg == nil {
		return core.Fail(core.E("agent.Agent.Evaluate", "agent config not set", nil))
	}

	r := a.resolveCheckpointTarget(ctx, target)
	if !r.OK {
		return r
	}
	cp := r.Value.(Checkpoint)

	return ProcessOne(a.cfg, a.influxClient(), cp)
}

// resolveCheckpointTarget normalises Evaluate() targets into a concrete
// Checkpoint. String targets are first resolved via DiscoverCheckpoints and
// then fall back to path-based metadata extraction when no exact match is
// found.
func (a *Agent) resolveCheckpointTarget(ctx context.Context, target any) core.Result {
	switch v := target.(type) {
	case Checkpoint:
		return core.Ok(v)
	case *Checkpoint:
		if v == nil {
			return core.Fail(core.E("agent.Agent.Evaluate", "nil checkpoint", nil))
		}
		return core.Ok(*v)
	case string:
		return a.resolveCheckpointPath(ctx, v)
	default:
		return core.Fail(core.E("agent.Agent.Evaluate", core.Sprintf("unsupported target type %T", target), nil))
	}
}

// resolveCheckpointPath tries to match a string target against the discovered
// checkpoint list before falling back to a path-derived checkpoint shape.
func (a *Agent) resolveCheckpointPath(ctx context.Context, target string) core.Result {
	target = core.Trim(target)
	if target == "" {
		return core.Fail(core.E("agent.Agent.Evaluate", "empty checkpoint path", nil))
	}

	if a != nil && a.cfg != nil {
		r := a.DiscoverCheckpoints(ctx)
		if r.OK {
			checkpoints := r.Value.([]Checkpoint)
			if cp, ok := matchCheckpointTarget(checkpoints, target); ok {
				return core.Ok(cp)
			}
		}
	}

	remoteDir := target
	filename := "adapters.safetensors"
	if core.HasSuffix(target, ".safetensors") {
		remoteDir = core.PathDir(target)
		filename = core.PathBase(target)
	}

	dirname := remoteDir
	if a != nil && a.cfg != nil && a.cfg.M3AdapterBase != "" {
		if rel, ok := cutPrefix(remoteDir, core.Concat(a.cfg.M3AdapterBase, "/")); ok && rel != "" {
			dirname = rel
		}
	}
	if dirname == "" {
		dirname = core.PathBase(remoteDir)
	}
	if dirname == "" {
		dirname = target
	}
	if core.HasPrefix(dirname, "/") {
		dirname = core.PathBase(dirname)
	}

	modelTag, labelPrefix, stem := AdapterMeta(dirname)
	label := labelPrefix
	if label == "" {
		label = dirname
	}
	runID := stem
	if runID == "" {
		runID = core.Replace(dirname, "/", "-")
	}
	if runID == "" {
		runID = dirname
	}

	return core.Ok(Checkpoint{
		RemoteDir: remoteDir,
		Filename:  filename,
		Dirname:   dirname,
		ModelTag:  modelTag,
		Label:     label,
		RunID:     core.Sprintf("%s-capability-auto", runID),
	})
}

// matchCheckpointTarget returns the first checkpoint that matches the target
// path exactly, or uniquely by basename when exact matching is unavailable.
func matchCheckpointTarget(checkpoints []Checkpoint, target string) (Checkpoint, bool) {
	var baseMatches []Checkpoint
	targetBase := core.PathBase(target)

	for _, cp := range checkpoints {
		if target == cp.RemoteDir || target == cp.Dirname {
			return cp, true
		}
		if target == core.Sprintf("%s/%s", cp.RemoteDir, cp.Filename) {
			return cp, true
		}
		if targetBase != "" && targetBase == core.PathBase(cp.RemoteDir) {
			baseMatches = append(baseMatches, cp)
			continue
		}
		if targetBase != "" && targetBase == core.PathBase(cp.Dirname) {
			baseMatches = append(baseMatches, cp)
		}
	}

	if len(baseMatches) == 1 {
		return baseMatches[0], true
	}
	return Checkpoint{}, false
}

// ExecuteRemote runs a shell command on the remote training host. The
// first positional arg MUST be the command; if host and port are passed
// in between ctx and command, a one-shot SSHTransport is built from them
// (useful when the caller does not want to rebuild the whole AgentConfig).
// Spec §8.
//
//	r := agent.ExecuteRemote(ctx, "ls /models")
//	if !r.OK { return r }
//	out := r.Value.(string)
//	r = agent.ExecuteRemote(ctx, "host.example", "2222", "uptime")
//	if !r.OK { return r }
func (a *Agent) ExecuteRemote(ctx context.Context, args ...string) core.Result {
	switch len(args) {
	case 0:
		return core.Fail(core.E("agent.Agent.ExecuteRemote", "no command supplied", nil))
	case 1:
		return a.cfg.transport().Run(ctx, args[0])
	case 3:
		host, port, command := args[0], args[1], args[2]
		keyPath := ""
		user := ""
		if a.cfg != nil {
			keyPath = a.cfg.M3SSHKey
			user = a.cfg.M3User
		}
		transport := NewSSHTransport(host, user, keyPath, WithPort(port))
		return transport.Run(ctx, command)
	default:
		return core.Fail(core.E("agent.Agent.ExecuteRemote",
			core.Sprintf("expected 1 arg (command) or 3 args (host,port,command); got %d", len(args)), nil))
	}
}

// CollectMetrics pushes queued probe/capability results to InfluxDB. Call
// this after Evaluate() has populated the internal buffer, or use it on a
// timer for long-running workflows. When influxURL is supplied, it
// replaces the agent's configured URL for this call only.
//
//	r := agent.CollectMetrics(ctx)
//	if !r.OK { return r }
func (a *Agent) CollectMetrics(ctx context.Context, influxURL ...string) core.Result {
	influx := a.influxClient()
	if len(influxURL) > 0 && influxURL[0] != "" {
		influx = datapipe.NewInfluxClient(influxURL[0], a.cfg.InfluxDB)
	}
	ReplayInfluxBuffer(a.cfg.WorkDir, influx)
	_ = ctx
	return core.Ok(nil)
}

// DiscoverCheckpoints lists all adapter checkpoints on the remote host.
//
//	r := agent.DiscoverCheckpoints(ctx)
//	if !r.OK { return r }
//	cps := r.Value.([]agent.Checkpoint)
func (a *Agent) DiscoverCheckpoints(ctx context.Context) core.Result {
	_ = ctx
	return DiscoverCheckpoints(a.cfg)
}

// Influx returns the shared datapipe.InfluxClient, constructing it lazily from the
// agent config on first access.
func (a *Agent) Influx() *datapipe.InfluxClient {
	return a.influxClient()
}

func (a *Agent) influxClient() *datapipe.InfluxClient {
	if a.influx != nil {
		return a.influx
	}
	url := a.cfg.InfluxURL
	db := a.cfg.InfluxDB
	if url == "" {
		url = core.Env("INFLUX_URL")
	}
	if db == "" {
		db = core.Env("INFLUX_DB")
	}
	a.influx = datapipe.NewInfluxClient(url, db)
	return a.influx
}
