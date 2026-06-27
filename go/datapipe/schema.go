// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

// InfluxDB measurement names — the schema the ingest/seed paths write and the
// LQL/agent read sides consume. They live with the data pipeline because the
// pipeline owns the storage schema; the agent orchestrator references them as
// datapipe.Measurement* once it is lifted.
const (
	MeasurementCapabilityScore = "capability_score"
	MeasurementCapabilityJudge = "capability_judge"
	MeasurementContentScore    = "content_score"
	MeasurementProbeScore      = "probe_score"
	MeasurementTrainingLoss    = "training_loss"
)

// DuckDB table names — the relational mirror of the measurement schema above.
const (
	TableCheckpointScores = "checkpoint_scores"
	TableProbeResults     = "probe_results"
)
