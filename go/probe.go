// SPDX-Licence-Identifier: EUPL-1.2

package inference

// ProbeEventKind names the observable event being emitted by a backend.
type ProbeEventKind string

// ProbePhase marks where an event occurred in the model lifecycle.
type ProbePhase string

const (
	ProbeEventToken          ProbeEventKind = "token"
	ProbeEventLogits         ProbeEventKind = "logits"
	ProbeEventEntropy        ProbeEventKind = "entropy"
	ProbeEventSelectedHeads  ProbeEventKind = "selected_heads"
	ProbeEventLayerCoherence ProbeEventKind = "layer_coherence"
	ProbeEventRouterDecision ProbeEventKind = "router_decision"
	ProbeEventResidual       ProbeEventKind = "residual"
	ProbeEventCachePressure  ProbeEventKind = "cache_pressure"
	ProbeEventMemoryPressure ProbeEventKind = "memory_pressure"
	ProbeEventTraining       ProbeEventKind = "training"

	ProbePhasePrefill  ProbePhase = "prefill"
	ProbePhaseDecode   ProbePhase = "decode"
	ProbePhaseTraining ProbePhase = "training"
)

// ProbeEvent is the typed envelope for model-state observation.
type ProbeEvent struct {
	Kind           ProbeEventKind        `json:"kind,omitempty"`
	Phase          ProbePhase            `json:"phase,omitempty"`
	Step           int                   `json:"step,omitempty"`
	Labels         map[string]string     `json:"labels,omitempty"`
	Token          *ProbeToken           `json:"token,omitempty"`
	Logits         *ProbeLogits          `json:"logits,omitempty"`
	Entropy        *ProbeEntropy         `json:"entropy,omitempty"`
	SelectedHeads  *ProbeHeadSelection   `json:"selected_heads,omitempty"`
	LayerCoherence *ProbeLayerCoherence  `json:"layer_coherence,omitempty"`
	RouterDecision *ProbeRouterDecision  `json:"router_decision,omitempty"`
	Residual       *ProbeResidualSummary `json:"residual,omitempty"`
	Cache          *ProbeCachePressure   `json:"cache,omitempty"`
	Memory         *ProbeMemoryPressure  `json:"memory,omitempty"`
	Training       *ProbeTraining        `json:"training,omitempty"`
}

// ProbeToken records token-level stream state.
type ProbeToken struct {
	ID              int32  `json:"id,omitempty"`
	Text            string `json:"text,omitempty"`
	PromptTokens    int    `json:"prompt_tokens,omitempty"`
	GeneratedTokens int    `json:"generated_tokens,omitempty"`
}

// ProbeLogit is one sampled or selected logit entry.
type ProbeLogit struct {
	ID    int32   `json:"id,omitempty"`
	Text  string  `json:"text,omitempty"`
	Value float32 `json:"value,omitempty"`
}

// ProbeLogits summarises logits without requiring full-vocabulary transfer.
type ProbeLogits struct {
	VocabularySize int          `json:"vocabulary_size,omitempty"`
	Top            []ProbeLogit `json:"top,omitempty"`
	Min            float32      `json:"min,omitempty"`
	Max            float32      `json:"max,omitempty"`
	Mean           float32      `json:"mean,omitempty"`
}

// ProbeEntropy records a scalar entropy measurement.
type ProbeEntropy struct {
	Value float64 `json:"value,omitempty"`
	Unit  string  `json:"unit,omitempty"`
}

// ProbeHeadSelection records selected heads for attention probing.
type ProbeHeadSelection struct {
	Layer int   `json:"layer,omitempty"`
	Heads []int `json:"heads,omitempty"`
}

// ProbeLayerCoherence carries layer-level alignment and spectral summaries.
type ProbeLayerCoherence struct {
	Layer          int     `json:"layer,omitempty"`
	KVCoupling     float64 `json:"kv_coupling,omitempty"`
	MeanCoherence  float64 `json:"mean_coherence,omitempty"`
	PhaseLock      float64 `json:"phase_lock,omitempty"`
	SpectralStable float64 `json:"spectral_stable,omitempty"`
}

// ProbeRouterDecision records sparse expert routing decisions.
type ProbeRouterDecision struct {
	Layer       int       `json:"layer,omitempty"`
	ExpertIDs   []int     `json:"expert_ids,omitempty"`
	ExpertProbs []float32 `json:"expert_probs,omitempty"`
}

// ProbeResidualSummary records compact residual stream statistics.
type ProbeResidualSummary struct {
	Layer int     `json:"layer,omitempty"`
	Mean  float64 `json:"mean,omitempty"`
	RMS   float64 `json:"rms,omitempty"`
	Norm  float64 `json:"norm,omitempty"`
}

// ProbeCachePressure records prompt/cache utilisation without exposing tensors.
type ProbeCachePressure struct {
	PromptTokens    int     `json:"prompt_tokens,omitempty"`
	GeneratedTokens int     `json:"generated_tokens,omitempty"`
	CachedTokens    int     `json:"cached_tokens,omitempty"`
	CacheMode       string  `json:"cache_mode,omitempty"`
	HitRate         float64 `json:"hit_rate,omitempty"`
}

// ProbeMemoryPressure records active, peak, and limit memory counters.
type ProbeMemoryPressure struct {
	ActiveBytes uint64 `json:"active_bytes,omitempty"`
	PeakBytes   uint64 `json:"peak_bytes,omitempty"`
	LimitBytes  uint64 `json:"limit_bytes,omitempty"`
}

// ProbeTraining records live training metrics.
type ProbeTraining struct {
	Epoch        int     `json:"epoch,omitempty"`
	Step         int     `json:"step,omitempty"`
	Loss         float64 `json:"loss,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
}

// ProbeSink receives typed probe events from model backends.
type ProbeSink interface {
	EmitProbe(event ProbeEvent)
}

// ProbeSinkFunc adapts a function to ProbeSink.
type ProbeSinkFunc func(ProbeEvent)

// EmitProbe emits an event when the function is non-nil.
func (f ProbeSinkFunc) EmitProbe(event ProbeEvent) {
	if f != nil {
		f(event)
	}
}

// ProbeBus fans probe events out to zero or more sinks.
type ProbeBus struct {
	sinks []ProbeSink
}

// NewProbeBus creates a probe fan-out bus.
func NewProbeBus(sinks ...ProbeSink) *ProbeBus {
	bus := &ProbeBus{}
	for _, sink := range sinks {
		bus.Add(sink)
	}
	return bus
}

// Add attaches a sink to the bus. Nil receivers and nil sinks are ignored.
func (b *ProbeBus) Add(sink ProbeSink) {
	if b == nil || sink == nil {
		return
	}
	b.sinks = append(b.sinks, sink)
}

// EmitProbe emits an event to every registered sink.
func (b *ProbeBus) EmitProbe(event ProbeEvent) {
	if b == nil {
		return
	}
	for _, sink := range b.sinks {
		if sink == nil {
			continue
		}
		sink.EmitProbe(event)
	}
}
