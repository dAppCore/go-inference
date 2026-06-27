package agent

import (
	"context"
	"net/http"
	"runtime"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
)

// WorkerConfig holds the worker's runtime configuration.
type WorkerConfig struct {
	APIBase      string
	WorkerID     string
	Name         string
	APIKey       string
	GPUType      string
	VRAMGb       int
	Languages    []string
	Models       []string
	InferURL     string
	TaskType     string
	BatchSize    int
	PollInterval time.Duration
	OneShot      bool
	DryRun       bool
}

// APITask represents a task from the LEM API.
type APITask struct {
	ID         int    `json:"id"`
	TaskType   string `json:"task_type"`
	Status     string `json:"status"`
	Language   string `json:"language"`
	Domain     string `json:"domain"`
	ModelName  string `json:"model_name"`
	PromptID   string `json:"prompt_id"`
	PromptText string `json:"prompt_text"`
	Config     *struct {
		Temperature float64 `json:"temperature,omitempty"`
		MaxTokens   int     `json:"max_tokens,omitempty"`
	} `json:"config"`
	Priority int `json:"priority"`
}

// RunWorkerLoop is the main worker loop that polls for tasks and processes them.
func RunWorkerLoop(cfg *WorkerConfig) {
	core.Print(nil, "LEM Worker starting")
	core.Print(nil, "  ID:       %s", cfg.WorkerID)
	core.Print(nil, "  Name:     %s", cfg.Name)
	core.Print(nil, "  API:      %s", cfg.APIBase)
	core.Print(nil, "  Infer:    %s", cfg.InferURL)
	core.Print(nil, "  GPU:      %s (%d GB)", cfg.GPUType, cfg.VRAMGb)
	core.Print(nil, "  Langs:    %v", cfg.Languages)
	core.Print(nil, "  Models:   %v", cfg.Models)
	core.Print(nil, "  Batch:    %d", cfg.BatchSize)
	core.Print(nil, "  Dry-run:  %v", cfg.DryRun)

	registerResult := workerRegister(cfg)
	if !registerResult.OK {
		core.Print(nil, "Registration failed: %v", registerResult.Value.(error))
	}
	core.Print(nil, "Registered with LEM API")

	for {
		processed := workerPoll(cfg)

		if cfg.OneShot {
			core.Print(nil, "One-shot mode: processed %d tasks, exiting", processed)
			return
		}

		if processed == 0 {
			core.Print(nil, "No tasks available, sleeping %v", cfg.PollInterval)
			time.Sleep(cfg.PollInterval)
		}

		workerHeartbeat(cfg)
	}
}

func workerRegister(cfg *WorkerConfig) core.Result {
	body := map[string]any{
		"worker_id":   cfg.WorkerID,
		"name":        cfg.Name,
		"version":     "0.1.0",
		"platform_os": runtime.GOOS,
		"arch":        runtime.GOARCH,
	}
	if cfg.GPUType != "" {
		body["gpu_type"] = cfg.GPUType
	}
	if cfg.VRAMGb > 0 {
		body["vram_gb"] = cfg.VRAMGb
	}
	if len(cfg.Languages) > 0 {
		body["languages"] = cfg.Languages
	}
	if len(cfg.Models) > 0 {
		body["supported_models"] = cfg.Models
	}

	postResult := apiPost(cfg, "/api/lem/workers/register", body)
	if !postResult.OK {
		return postResult
	}
	return core.Ok(nil)
}

func workerHeartbeat(cfg *WorkerConfig) {
	body := map[string]any{
		"worker_id": cfg.WorkerID,
	}
	apiPost(cfg, "/api/lem/workers/heartbeat", body)
}

func workerPoll(cfg *WorkerConfig) int {
	url := core.Sprintf("/api/lem/tasks/next?worker_id=%s&limit=%d", cfg.WorkerID, cfg.BatchSize)
	if cfg.TaskType != "" {
		url += "&type=" + cfg.TaskType
	}

	respResult := apiGet(cfg, url)
	if !respResult.OK {
		core.Print(nil, "Error fetching tasks: %v", respResult.Value.(error))
		return 0
	}
	resp := respResult.Value.([]byte)

	var result struct {
		Tasks []APITask `json:"tasks"`
		Count int       `json:"count"`
	}
	if r := core.JSONUnmarshal(resp, &result); !r.OK {
		core.Print(nil, "Error parsing tasks: %v", r.Value)
		return 0
	}

	if result.Count == 0 {
		return 0
	}

	core.Print(nil, "Got %d tasks", result.Count)
	processed := 0

	for _, task := range result.Tasks {
		taskResult := workerProcessTask(cfg, task)
		if !taskResult.OK {
			core.Print(nil, "Task %d failed: %v", task.ID, taskResult.Value.(error))
			apiDelete(cfg, core.Sprintf("/api/lem/tasks/%d/claim", task.ID), map[string]any{
				"worker_id": cfg.WorkerID,
			})
			continue
		}
		processed++
	}

	return processed
}

func workerProcessTask(cfg *WorkerConfig, task APITask) core.Result {
	core.Print(nil, "Processing task %d: %s [%s/%s] %d chars prompt",
		task.ID, task.TaskType, task.Language, task.Domain, len(task.PromptText))

	claimResult := apiPost(cfg, core.Sprintf("/api/lem/tasks/%d/claim", task.ID), map[string]any{
		"worker_id": cfg.WorkerID,
	})
	if !claimResult.OK {
		return core.Fail(core.E("agent.workerProcessTask", "claim", claimResult.Value.(error)))
	}

	apiPatch(cfg, core.Sprintf("/api/lem/tasks/%d/status", task.ID), map[string]any{
		"worker_id": cfg.WorkerID,
		"status":    "in_progress",
	})

	if cfg.DryRun {
		core.Print(nil, "  [DRY-RUN] Would generate response for: %.80s...", task.PromptText)
		return core.Ok(nil)
	}

	start := time.Now()
	inferResult := workerInfer(cfg, task)
	genTime := time.Since(start)

	if !inferResult.OK {
		apiPatch(cfg, core.Sprintf("/api/lem/tasks/%d/status", task.ID), map[string]any{
			"worker_id": cfg.WorkerID,
			"status":    "abandoned",
		})
		return core.Fail(core.E("agent.workerProcessTask", "inference", inferResult.Value.(error)))
	}
	response := inferResult.Value.(string)

	modelUsed := task.ModelName
	if modelUsed == "" {
		modelUsed = "default"
	}

	postResult := apiPost(cfg, core.Sprintf("/api/lem/tasks/%d/result", task.ID), map[string]any{
		"worker_id":     cfg.WorkerID,
		"response_text": response,
		"model_used":    modelUsed,
		"gen_time_ms":   int(genTime.Milliseconds()),
	})
	if !postResult.OK {
		return core.Fail(core.E("agent.workerProcessTask", "submit result", postResult.Value.(error)))
	}

	core.Print(nil, "  Completed: %d chars in %v", len(response), genTime.Round(time.Millisecond))
	return core.Ok(nil)
}

func workerInfer(cfg *WorkerConfig, task APITask) core.Result {
	temp := 0.7
	maxTokens := 2048
	if task.Config != nil {
		if task.Config.Temperature > 0 {
			temp = task.Config.Temperature
		}
		if task.Config.MaxTokens > 0 {
			maxTokens = task.Config.MaxTokens
		}
	}

	// Use the shared serving.HTTPBackend (OpenAI-compatible /v1/chat/completions
	// client) instead of a bespoke request — one OpenAI client across the stack.
	backend := serving.NewHTTPBackend(cfg.InferURL, task.ModelName,
		serving.WithHTTPClient(&http.Client{Timeout: 5 * time.Minute}))

	r := backend.Generate(context.Background(), task.PromptText,
		serving.GenOpts{Temperature: temp, MaxTokens: maxTokens, Model: task.ModelName})
	if !r.OK {
		return core.Fail(core.E("agent.workerInfer", "inference request", r.Value.(error)))
	}

	content := r.Value.(serving.Result).Text
	if len(content) < 10 {
		return core.Fail(core.E("agent.workerInfer", core.Sprintf("response too short: %d chars", len(content)), nil))
	}

	return core.Ok(content)
}

// HTTP helpers for the LEM API.

func apiGet(cfg *WorkerConfig, path string) core.Result {
	req, err := http.NewRequest("GET", cfg.APIBase+path, nil)
	if err != nil {
		return core.Fail(core.E("agent.apiGet", "create request", err))
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("agent.apiGet", "send request", err))
	}
	defer resp.Body.Close()

	rBody := readAll(resp.Body)
	if !rBody.OK {
		return core.Fail(rBody.Value.(error))
	}
	body := rBody.Value.([]byte)

	if resp.StatusCode >= 400 {
		return core.Fail(core.E("agent.apiGet", core.Sprintf("HTTP %d: %s", resp.StatusCode, truncStr(string(body), 200)), nil))
	}

	return core.Ok(body)
}

func apiPost(cfg *WorkerConfig, path string, data map[string]any) core.Result {
	return apiRequest(cfg, "POST", path, data)
}

func apiPatch(cfg *WorkerConfig, path string, data map[string]any) core.Result {
	return apiRequest(cfg, "PATCH", path, data)
}

func apiDelete(cfg *WorkerConfig, path string, data map[string]any) core.Result {
	return apiRequest(cfg, "DELETE", path, data)
}

func apiRequest(cfg *WorkerConfig, method, path string, data map[string]any) core.Result {
	jsonData := []byte(core.JSONMarshalString(data))

	req, err := http.NewRequest(method, cfg.APIBase+path, core.NewBuffer(jsonData))
	if err != nil {
		return core.Fail(core.E("agent.apiRequest", "create request", err))
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("agent.apiRequest", "send request", err))
	}
	defer resp.Body.Close()

	rBody := readAll(resp.Body)
	if !rBody.OK {
		return core.Fail(rBody.Value.(error))
	}
	body := rBody.Value.([]byte)

	if resp.StatusCode >= 400 {
		return core.Fail(core.E("agent.apiRequest", core.Sprintf("HTTP %d: %s", resp.StatusCode, truncStr(string(body), 200)), nil))
	}

	return core.Ok(body)
}

// MachineID returns the machine ID from /etc/machine-id or hostname fallback.
func MachineID() string {
	if data, err := coreio.Local.Read("/etc/machine-id"); err == nil {
		id := core.Trim(data)
		if len(id) > 0 {
			return id
		}
	}
	rHost := hostname()
	if !rHost.OK {
		return ""
	}
	return rHost.Value.(string)
}

// Hostname returns the system hostname.
func Hostname() string {
	rHost := hostname()
	if !rHost.OK {
		return ""
	}
	return rHost.Value.(string)
}

// ReadKeyFile reads the LEM API key from ~/.config/lem/api_key.
func ReadKeyFile() string {
	rHome := userHomeDir()
	if !rHome.OK {
		return ""
	}
	home := rHome.Value.(string)
	path := core.Path(home, ".config", "lem", "api_key")
	data, err := coreio.Local.Read(path)
	if err != nil {
		return ""
	}
	return core.Trim(data)
}

// SplitComma splits a comma-separated string into trimmed parts.
func SplitComma(s string) []string {
	var result []string
	for _, part := range core.Split(s, ",") {
		trimmed := core.Trim(part)
		if len(trimmed) > 0 {
			result = append(result, trimmed)
		}
	}
	return result
}

func truncStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
