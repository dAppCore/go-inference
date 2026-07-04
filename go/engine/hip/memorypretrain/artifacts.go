// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"

	core "dappco.re/go"
)

// MemoryPretrainingArtifactConfig controls the native offline build for
// hierarchical-memory pretraining artifacts.
type MemoryPretrainingArtifactConfig struct {
	CorpusPath          string               `json:"corpus_path,omitempty"`
	RouterPath          string               `json:"router_path,omitempty"`
	FFNMemoryPath       string               `json:"ffn_memory_path,omitempty"`
	Build               BuildConfig          `json:"build,omitempty"`
	FFNMemory           FFNMemoryConfig      `json:"ffn_memory,omitempty"`
	ClusterIDInputPath  string               `json:"cluster_id_input_path,omitempty"`
	ClusterIDOutputPath string               `json:"cluster_id_output_path,omitempty"`
	ClusterIDJSONL      ClusterIDJSONLConfig `json:"cluster_id_jsonl,omitempty"`
}

// MemoryPretrainingArtifacts contains the in-memory artifacts built by the
// native offline pipeline and its summary report.
type MemoryPretrainingArtifacts struct {
	Router    *Bank                            `json:"-"`
	FFNMemory *FFNMemoryBank                   `json:"-"`
	Report    *MemoryPretrainingArtifactReport `json:"report,omitempty"`
}

// MemoryPretrainingArtifactReport summarises one offline artifact build.
type MemoryPretrainingArtifactReport struct {
	CorpusPath      string                `json:"corpus_path,omitempty"`
	RouterPath      string                `json:"router_path,omitempty"`
	FFNMemoryPath   string                `json:"ffn_memory_path,omitempty"`
	CorpusRecords   int                   `json:"corpus_records"`
	RouterNodes     int                   `json:"router_nodes"`
	FFNMemoryLayers int                   `json:"ffn_memory_layers"`
	ClusterIDInput  string                `json:"cluster_id_input,omitempty"`
	ClusterIDOutput string                `json:"cluster_id_output,omitempty"`
	ClusterIDReport *ClusterIDJSONLReport `json:"cluster_id_report,omitempty"`
}

// BuildMemoryPretrainingArtifactsFromFiles loads a corpus JSONL file, then runs
// the native offline artifact builder.
func BuildMemoryPretrainingArtifactsFromFiles(ctx context.Context, embedder Embedder, cfg MemoryPretrainingArtifactConfig) (*MemoryPretrainingArtifacts, error) {
	if cfg.CorpusPath == "" {
		return nil, core.NewError("memorypretrain: corpus path is required")
	}
	records, err := LoadCorpusRecordsJSONLFile(cfg.CorpusPath)
	if err != nil {
		return nil, err
	}
	artifacts, err := BuildMemoryPretrainingArtifacts(ctx, embedder, records, cfg)
	if err != nil {
		return nil, err
	}
	if artifacts.Report != nil {
		artifacts.Report.CorpusPath = cfg.CorpusPath
	}
	return artifacts, nil
}

// LoadCorpusRecordsJSONLFile reads corpus records from a JSONL file.
func LoadCorpusRecordsJSONLFile(path string) ([]CorpusRecord, error) {
	if path == "" {
		return nil, core.NewError("memorypretrain: corpus path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, memoryPretrainResultError(read)
	}
	return LoadCorpusRecordsJSONL(core.AsString(read.Value.([]byte)))
}

// LoadCorpusRecordsJSONL parses corpus records from JSONL. Each row accepts
// id, text, and an optional string-valued meta object.
func LoadCorpusRecordsJSONL(raw string) ([]CorpusRecord, error) {
	if core.Trim(raw) == "" {
		return nil, core.NewError("memorypretrain: corpus JSONL input is empty")
	}
	lines := core.Split(raw, "\n")
	records := make([]CorpusRecord, 0, len(lines))
	for index, line := range lines {
		line = core.Trim(line)
		if line == "" {
			continue
		}
		var row map[string]any
		if result := core.JSONUnmarshalString(line, &row); !result.OK {
			return nil, core.Errorf("memorypretrain: parse corpus JSONL record %d: %w", index+1, result.Value.(error))
		}
		text := stringField(row, "text")
		if text == "" {
			return nil, core.Errorf("memorypretrain: corpus JSONL record %d has no text", index+1)
		}
		records = append(records, CorpusRecord{
			ID:   stringField(row, "id"),
			Text: text,
			Meta: corpusRecordMeta(row["meta"]),
		})
	}
	if len(records) == 0 {
		return nil, core.NewError("memorypretrain: corpus JSONL input produced no rows")
	}
	return records, nil
}

// BuildMemoryPretrainingArtifacts embeds corpus records, builds the
// hierarchical router, allocates the matching FFN memory table, persists
// requested artifacts, and optionally writes a cluster-ID enriched JSONL file.
func BuildMemoryPretrainingArtifacts(ctx context.Context, embedder Embedder, records []CorpusRecord, cfg MemoryPretrainingArtifactConfig) (*MemoryPretrainingArtifacts, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if embedder == nil {
		return nil, core.NewError("memorypretrain: embedder is required")
	}
	if len(records) == 0 {
		return nil, core.NewError("memorypretrain: corpus records are required")
	}
	if cfg.FFNMemory.HiddenSize <= 0 {
		return nil, core.NewError("memorypretrain: FFN memory hidden size is required")
	}
	if cfg.FFNMemory.Layers <= 0 {
		return nil, core.NewError("memorypretrain: FFN memory layers are required")
	}
	if cfg.ClusterIDInputPath != "" && cfg.ClusterIDOutputPath == "" {
		return nil, core.NewError("memorypretrain: cluster-ID output path is required")
	}
	router, err := BuildBankFromCorpus(ctx, embedder, records, cfg.Build)
	if err != nil {
		return nil, err
	}
	ffnCfg := cfg.FFNMemory
	if len(ffnCfg.NumClusters) == 0 {
		ffnCfg.NumClusters = routerClusterCounts(router)
	}
	ffnMemory, err := NewFFNMemoryBank(ffnCfg)
	if err != nil {
		return nil, err
	}
	report := &MemoryPretrainingArtifactReport{
		CorpusPath:      cfg.CorpusPath,
		RouterPath:      cfg.RouterPath,
		FFNMemoryPath:   cfg.FFNMemoryPath,
		CorpusRecords:   len(records),
		RouterNodes:     len(router.Nodes),
		FFNMemoryLayers: len(ffnMemory.Layers),
		ClusterIDInput:  cfg.ClusterIDInputPath,
		ClusterIDOutput: cfg.ClusterIDOutputPath,
	}
	if cfg.RouterPath != "" {
		if err := SaveBank(cfg.RouterPath, router); err != nil {
			return nil, err
		}
	}
	if cfg.FFNMemoryPath != "" {
		if err := SaveFFNMemoryBank(cfg.FFNMemoryPath, ffnMemory); err != nil {
			return nil, err
		}
	}
	if cfg.ClusterIDInputPath != "" {
		clusterCfg := cfg.ClusterIDJSONL
		if len(clusterCfg.ClusterCounts) == 0 {
			clusterCfg.ClusterCounts = ffnMemory.ClusterCounts()
		}
		clusterReport, err := AddClusterIDsToJSONLFile(ctx, cfg.ClusterIDInputPath, cfg.ClusterIDOutputPath, embedder, router, clusterCfg)
		if err != nil {
			return nil, err
		}
		report.ClusterIDReport = &clusterReport
	}
	return &MemoryPretrainingArtifacts{
		Router:    router,
		FFNMemory: ffnMemory,
		Report:    report,
	}, nil
}

func corpusRecordMeta(value any) map[string]string {
	raw, ok := value.(map[string]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	meta := make(map[string]string, len(raw))
	for key, value := range raw {
		if text, ok := value.(string); ok {
			meta[key] = text
		}
	}
	if len(meta) == 0 {
		return nil
	}
	return meta
}

func routerClusterCounts(bank *Bank) []int {
	if bank == nil {
		return nil
	}
	cfg := normaliseBuildConfig(bank.Config)
	counts := make([]int, cfg.MaxDepth)
	count := 1
	for level := 0; level < cfg.MaxDepth; level++ {
		count *= cfg.BranchingFactor
		counts[level] = count
	}
	return counts
}
