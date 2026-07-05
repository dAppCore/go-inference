// SPDX-Licence-Identifier: EUPL-1.2

package state

import core "dappco.re/go"

type ProjectSeedMode string

const (
	ProjectSeedStateCheckpoint ProjectSeedMode = "state_checkpoint"
	ProjectSeedReuseCurrent    ProjectSeedMode = "reuse_current"
	ProjectSeedSummaryWindow   ProjectSeedMode = "summary_window"
	ProjectSeedHybrid          ProjectSeedMode = "hybrid"
)

type ProjectSeedOptions struct {
	BaseURI   string            `json:"base_uri,omitempty"`
	ProjectID string            `json:"project_id,omitempty"`
	EntryURI  string            `json:"entry_uri,omitempty"`
	BundleURI string            `json:"bundle_uri,omitempty"`
	IndexURI  string            `json:"index_uri,omitempty"`
	Title     string            `json:"title,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type ProjectSeed struct {
	BaseURI   string            `json:"base_uri,omitempty"`
	ProjectID string            `json:"project_id,omitempty"`
	EntryURI  string            `json:"entry_uri,omitempty"`
	BundleURI string            `json:"bundle_uri,omitempty"`
	IndexURI  string            `json:"index_uri,omitempty"`
	Title     string            `json:"title,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type ProjectSeedWakeOptions struct {
	Store     any               `json:"-"`
	Model     ModelIdentity     `json:"model,omitempty"`
	Tokenizer TokenizerIdentity `json:"tokenizer,omitempty"`
	Adapter   AdapterIdentity   `json:"adapter,omitempty"`
	Runtime   RuntimeIdentity   `json:"runtime,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

type ProjectSeedContinuationOptions struct {
	Mode      ProjectSeedMode   `json:"mode,omitempty"`
	Store     any               `json:"-"`
	EntryURI  string            `json:"entry_uri,omitempty"`
	BundleURI string            `json:"bundle_uri,omitempty"`
	IndexURI  string            `json:"index_uri,omitempty"`
	Title     string            `json:"title,omitempty"`
	Parent    WakeResult        `json:"parent,omitempty"`
	Model     ModelIdentity     `json:"model,omitempty"`
	Tokenizer TokenizerIdentity `json:"tokenizer,omitempty"`
	Adapter   AdapterIdentity   `json:"adapter,omitempty"`
	Runtime   RuntimeIdentity   `json:"runtime,omitempty"`
	BlockSize int               `json:"block_size,omitempty"`
	Encoding  string            `json:"encoding,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type ProjectSeedContinuationPlan struct {
	Mode             ProjectSeedMode `json:"mode,omitempty"`
	Sleep            SleepRequest    `json:"sleep,omitempty"`
	PersistState     bool            `json:"persist_state,omitempty"`
	NeedsSummary     bool            `json:"needs_summary,omitempty"`
	ReuseCurrentSeed bool            `json:"reuse_current_seed,omitempty"`
}

func NewProjectSeed(opts ProjectSeedOptions) ProjectSeed {
	seed := ProjectSeed{
		BaseURI:   cleanURI(opts.BaseURI),
		ProjectID: cleanURI(opts.ProjectID),
		EntryURI:  cleanURI(opts.EntryURI),
		BundleURI: cleanURI(opts.BundleURI),
		IndexURI:  cleanURI(opts.IndexURI),
		Title:     core.Trim(opts.Title),
		Labels:    cloneStringMap(opts.Labels),
		Metadata:  cloneStringMap(opts.Metadata),
	}
	if seed.BaseURI == "" {
		seed.BaseURI = "state://projects"
	}
	if seed.ProjectID == "" {
		seed.ProjectID = "default"
	}
	if seed.EntryURI == "" {
		seed.EntryURI = joinURI(seed.BaseURI, seed.ProjectID, "seed")
	}
	if seed.BundleURI == "" {
		seed.BundleURI = seed.EntryURI + "/bundle"
	}
	if seed.IndexURI == "" {
		seed.IndexURI = seed.EntryURI + "/index"
	}
	if seed.Title == "" {
		seed.Title = seed.ProjectID + " project seed"
	}
	return seed
}

func (s ProjectSeed) WakeRequest(opts ProjectSeedWakeOptions) WakeRequest {
	labels := mergeStringMaps(s.Labels, opts.Labels)
	setProjectLabel(labels, s.ProjectID)
	return WakeRequest{
		Store:     opts.Store,
		IndexURI:  s.IndexURI,
		EntryURI:  s.EntryURI,
		Model:     opts.Model,
		Tokenizer: opts.Tokenizer,
		Adapter:   opts.Adapter,
		Runtime:   opts.Runtime,
		Labels:    labels,
	}
}

func (s ProjectSeed) PlanContinuation(opts ProjectSeedContinuationOptions) ProjectSeedContinuationPlan {
	mode := opts.Mode
	if mode == "" {
		mode = ProjectSeedStateCheckpoint
	}
	plan := ProjectSeedContinuationPlan{Mode: mode}
	switch mode {
	case ProjectSeedReuseCurrent:
		plan.ReuseCurrentSeed = true
		return plan
	case ProjectSeedSummaryWindow:
		plan.NeedsSummary = true
		return plan
	case ProjectSeedHybrid:
		plan.PersistState = true
		plan.NeedsSummary = true
	default:
		plan.Mode = ProjectSeedStateCheckpoint
		plan.PersistState = true
	}
	plan.Sleep = s.sleepRequest(opts)
	return plan
}

func (s ProjectSeed) sleepRequest(opts ProjectSeedContinuationOptions) SleepRequest {
	entryURI := cleanURI(opts.EntryURI)
	if entryURI == "" {
		entryURI = joinURI(s.BaseURI, s.ProjectID, "checkpoints", "latest")
	}
	bundleURI := cleanURI(opts.BundleURI)
	if bundleURI == "" {
		bundleURI = entryURI + "/bundle"
	}
	indexURI := cleanURI(opts.IndexURI)
	if indexURI == "" {
		indexURI = entryURI + "/index"
	}
	metadata := mergeStringMaps(s.Metadata, opts.Metadata)
	setProjectLabel(metadata, s.ProjectID)
	labels := mergeStringMaps(s.Labels, opts.Labels)
	setProjectLabel(labels, s.ProjectID)
	parent := opts.Parent.Entry
	return SleepRequest{
		Store:             opts.Store,
		EntryURI:          entryURI,
		BundleURI:         bundleURI,
		IndexURI:          indexURI,
		ParentEntryURI:    core.FirstNonBlank(parent.URI, s.EntryURI),
		ParentBundleURI:   core.FirstNonBlank(parent.BundleURI, s.BundleURI),
		ParentIndexURI:    core.FirstNonBlank(parent.IndexURI, s.IndexURI),
		Title:             core.FirstNonBlank(core.Trim(opts.Title), s.Title),
		Model:             opts.Model,
		Tokenizer:         opts.Tokenizer,
		Adapter:           opts.Adapter,
		Runtime:           opts.Runtime,
		ReuseParentPrefix: true,
		BlockSize:         opts.BlockSize,
		Encoding:          opts.Encoding,
		Labels:            labels,
		Metadata:          metadata,
	}
}

type WakeCompatibilityReport struct {
	Compatible      bool     `json:"compatible"`
	SummaryRequired bool     `json:"summary_required,omitempty"`
	Reasons         []string `json:"reasons,omitempty"`
	Warnings        []string `json:"warnings,omitempty"`
}

func CheckWakeCompatibility(bundle Bundle, req WakeRequest) WakeCompatibilityReport {
	if req.SkipCompatibilityCheck {
		return WakeCompatibilityReport{
			Compatible: true,
			Warnings:   []string{"compatibility_check_skipped"},
		}
	}
	report := WakeCompatibilityReport{Compatible: true}
	compareModelIdentity(&report, bundle, req.Model)
	compareTokenizerIdentity(&report, bundle.Tokenizer, req.Tokenizer)
	compareAdapterIdentity(&report, bundle.Adapter, req.Adapter)
	compareRuntimeIdentity(&report, bundle.Runtime, req.Runtime)
	report.Compatible = len(report.Reasons) == 0
	report.SummaryRequired = !report.Compatible
	return report
}

func compareModelIdentity(report *WakeCompatibilityReport, bundle Bundle, req ModelIdentity) {
	model := bundle.Model
	if model.Hash != "" && req.Hash != "" && model.Hash != req.Hash {
		report.Reasons = append(report.Reasons, "model_hash_mismatch")
	}
	if model.Architecture != "" && req.Architecture != "" && model.Architecture != req.Architecture {
		report.Reasons = append(report.Reasons, "model_architecture_mismatch")
	}
	if model.NumLayers > 0 && req.NumLayers > 0 && model.NumLayers != req.NumLayers {
		report.Reasons = append(report.Reasons, "model_layer_mismatch")
	}
	if model.QuantBits > 0 && req.QuantBits > 0 && model.QuantBits != req.QuantBits {
		report.Reasons = append(report.Reasons, "model_quantisation_mismatch")
	}
	prefixTokens := bundle.PromptTokens + bundle.GeneratedTokens
	if prefixTokens <= 0 {
		prefixTokens = bundle.PromptTokens
	}
	if req.ContextLength > 0 && prefixTokens > 0 && req.ContextLength < prefixTokens {
		report.Reasons = append(report.Reasons, "context_length_too_small")
	}
}

func compareTokenizerIdentity(report *WakeCompatibilityReport, bundle, req TokenizerIdentity) {
	if bundle.Hash != "" && req.Hash != "" && bundle.Hash != req.Hash {
		report.Reasons = append(report.Reasons, "tokenizer_hash_mismatch")
	}
	if bundle.ChatTemplate != "" && req.ChatTemplate != "" && bundle.ChatTemplate != req.ChatTemplate {
		report.Reasons = append(report.Reasons, "chat_template_mismatch")
	}
}

func compareAdapterIdentity(report *WakeCompatibilityReport, bundle, req AdapterIdentity) {
	bundleActive := adapterIdentityActive(bundle)
	reqActive := adapterIdentityActive(req)
	switch {
	case bundleActive && !reqActive:
		report.Reasons = append(report.Reasons, "adapter_missing")
	case !bundleActive && reqActive:
		report.Reasons = append(report.Reasons, "adapter_unexpected")
	case bundle.Hash != "" && req.Hash != "" && bundle.Hash != req.Hash:
		report.Reasons = append(report.Reasons, "adapter_hash_mismatch")
	case bundle.Path != "" && req.Path != "" && bundle.Path != req.Path:
		report.Reasons = append(report.Reasons, "adapter_path_mismatch")
	case bundle.Rank > 0 && req.Rank > 0 && bundle.Rank != req.Rank:
		report.Reasons = append(report.Reasons, "adapter_rank_mismatch")
	}
}

func compareRuntimeIdentity(report *WakeCompatibilityReport, bundle, req RuntimeIdentity) {
	if bundle.Backend != "" && req.Backend != "" && bundle.Backend != req.Backend {
		report.Warnings = append(report.Warnings, "runtime_backend_changed")
	}
	if bundle.CacheMode != "" && req.CacheMode != "" && bundle.CacheMode != req.CacheMode {
		report.Warnings = append(report.Warnings, "runtime_cache_mode_changed")
	}
}

func adapterIdentityActive(adapter AdapterIdentity) bool {
	return adapter.Hash != "" || adapter.Path != "" || adapter.Format != "" || adapter.Rank != 0 || adapter.Alpha != 0 || len(adapter.TargetKeys) > 0 || adapter.BaseModelHash != ""
}

func cleanURI(value string) string {
	value = core.Trim(value)
	value = core.TrimPrefix(value, "/")
	return core.TrimSuffix(value, "/")
}

func joinURI(base string, parts ...string) string {
	// Walk parts twice — first to sum the exact final length, second to
	// append into a pre-sized []byte buffer. cleanURI is alloc-free
	// (string substring views), so the second walk is purely arithmetic
	// + byte copies. The previous shape used core.NewBuilder() (heap
	// pointer alloc) plus the Builder's internal buffer grow (second
	// heap alloc); collapsing to a direct []byte buffer + core.AsString
	// drops one heap alloc per call. The cleaned []string slot from the
	// previous shape was stack-resident, so eliding it costs nothing.
	cleanBase := cleanURI(base)
	total := len(cleanBase)
	for _, part := range parts {
		p := cleanURI(part)
		if p == "" {
			continue
		}
		if total > 0 {
			total++ // separator
		}
		total += len(p)
	}
	if total == 0 {
		return ""
	}
	buf := make([]byte, 0, total)
	if cleanBase != "" {
		buf = append(buf, cleanBase...)
	}
	for _, part := range parts {
		p := cleanURI(part)
		if p == "" {
			continue
		}
		if len(buf) > 0 {
			buf = append(buf, '/')
		}
		buf = append(buf, p...)
	}
	return core.AsString(buf)
}

func setProjectLabel(labels map[string]string, projectID string) {
	if labels == nil || projectID == "" {
		return
	}
	if labels["project_id"] == "" {
		labels["project_id"] = projectID
	}
}

func mergeStringMaps(left, right map[string]string) map[string]string {
	if len(left) == 0 && len(right) == 0 {
		return nil
	}
	out := make(map[string]string, len(left)+len(right)+1)
	for key, value := range left {
		out[key] = value
	}
	for key, value := range right {
		out[key] = value
	}
	return out
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
