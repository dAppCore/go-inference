// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"math"

	core "dappco.re/go"
)

// FFNMemoryConfig describes the extra hierarchical memory parameters attached
// to each feed-forward layer.
type FFNMemoryConfig struct {
	HiddenSize         int      `json:"hidden_size"`
	Layers             int      `json:"layers"`
	MemoryLevels       []string `json:"memory_levels,omitempty"`
	FFNMemoryTokens    []int    `json:"ffn_memory_tokens,omitempty"`
	NumClusters        []int    `json:"num_clusters,omitempty"`
	LinearRampMemories bool     `json:"linear_ramp_memories,omitempty"`
	AddedGenericSize   int      `json:"added_generic_size,omitempty"`
	ZeroInitialiseW3   bool     `json:"zero_initialise_w3,omitempty"`
}

// FFNMemoryBank stores per-layer hierarchical FFN memory tensors. Each level
// uses W1/W2/W3 flattened as [cluster][hidden][tokens],
// [cluster][hidden][tokens], and [cluster][tokens][hidden].
type FFNMemoryBank struct {
	HiddenSize int              `json:"hidden_size"`
	Config     FFNMemoryConfig  `json:"config"`
	Layers     []FFNMemoryLayer `json:"layers,omitempty"`
}

// FFNMemoryLayer stores all memory hierarchy levels for one transformer layer.
type FFNMemoryLayer struct {
	Layer  int                    `json:"layer"`
	Levels []FFNMemoryLevelWeight `json:"levels,omitempty"`
}

// FFNMemoryLevelWeight stores one level's clustered memory weights.
type FFNMemoryLevelWeight struct {
	Name             string    `json:"name"`
	NumClusters      int       `json:"num_clusters"`
	AddedGenericSize int       `json:"added_generic_size"`
	MemoryTokens     int       `json:"memory_tokens"`
	W1               []float32 `json:"w1,omitempty"`
	W2               []float32 `json:"w2,omitempty"`
	W3               []float32 `json:"w3,omitempty"`
}

// FFNMemoryStats describes one memory application to an FFN output.
type FFNMemoryStats struct {
	Layer         int  `json:"layer"`
	LevelsApplied int  `json:"levels_applied"`
	MemoryTokens  int  `json:"memory_tokens"`
	Applied       bool `json:"applied"`
}

// NewFFNMemoryBank allocates a native hierarchical FFN memory table. W1 and W2
// receive deterministic small initial values and W3 starts at zero, so adding
// newly-created memories initially preserves the anchor model output.
func NewFFNMemoryBank(cfg FFNMemoryConfig) (*FFNMemoryBank, error) {
	cfg = normaliseFFNMemoryConfig(cfg)
	if err := validateFFNMemoryConfig(cfg); err != nil {
		return nil, err
	}
	bank := &FFNMemoryBank{
		HiddenSize: cfg.HiddenSize,
		Config:     cfg,
		Layers:     make([]FFNMemoryLayer, cfg.Layers),
	}
	for layerID := range bank.Layers {
		layer := &bank.Layers[layerID]
		layer.Layer = layerID
		layer.Levels = make([]FFNMemoryLevelWeight, len(cfg.MemoryLevels))
		for levelID := range cfg.MemoryLevels {
			tokens := cfg.FFNMemoryTokens[levelID]
			if cfg.LinearRampMemories {
				tokens = int(math.Floor(2 * float64(tokens) * float64(layerID+1) / float64(cfg.Layers)))
				if tokens < 1 {
					tokens = 1
				}
			}
			clusters := cfg.NumClusters[levelID]
			totalClusters := clusters + cfg.AddedGenericSize
			level := &layer.Levels[levelID]
			level.Name = cfg.MemoryLevels[levelID]
			level.NumClusters = clusters
			level.AddedGenericSize = cfg.AddedGenericSize
			level.MemoryTokens = tokens
			level.W1 = make([]float32, totalClusters*cfg.HiddenSize*tokens)
			level.W2 = make([]float32, totalClusters*cfg.HiddenSize*tokens)
			level.W3 = make([]float32, totalClusters*tokens*cfg.HiddenSize)
			initialiseFFNMemoryInputWeights(level.W1, cfg.HiddenSize, layerID, levelID, 1)
			initialiseFFNMemoryInputWeights(level.W2, cfg.HiddenSize, layerID, levelID, 17)
		}
	}
	return bank, nil
}

// AddToFFNOutput computes the memory contribution from mlpInput and adds it to
// ffnOutput, matching the upstream hook shape where memory augments the MLP
// output rather than replacing it.
func (bank *FFNMemoryBank) AddToFFNOutput(dst []float32, ffnOutput []float32, mlpInput []float32, layerID int, clusterIDs []int) ([]float32, FFNMemoryStats, error) {
	if bank == nil {
		return nil, FFNMemoryStats{}, core.NewError("memorypretrain: FFN memory bank is nil")
	}
	if len(ffnOutput) != bank.HiddenSize {
		return nil, FFNMemoryStats{}, core.Errorf("memorypretrain: FFN output dimension %d does not match hidden size %d", len(ffnOutput), bank.HiddenSize)
	}
	if len(mlpInput) != bank.HiddenSize {
		return nil, FFNMemoryStats{}, core.Errorf("memorypretrain: MLP input dimension %d does not match hidden size %d", len(mlpInput), bank.HiddenSize)
	}
	if layerID < 0 || layerID >= len(bank.Layers) {
		return nil, FFNMemoryStats{}, core.Errorf("memorypretrain: FFN memory layer %d is out of range", layerID)
	}
	layer := &bank.Layers[layerID]
	if len(clusterIDs) != len(layer.Levels) {
		return nil, FFNMemoryStats{}, core.Errorf("memorypretrain: cluster ID count %d does not match memory levels %d", len(clusterIDs), len(layer.Levels))
	}
	out := resetFloat32(dst, len(ffnOutput))
	copy(out, ffnOutput)
	stats := FFNMemoryStats{Layer: layerID}
	for levelID := range layer.Levels {
		level := &layer.Levels[levelID]
		clusterID := clusterIDs[levelID]
		if err := validateFFNMemoryLevel(level, bank.HiddenSize, clusterID); err != nil {
			return nil, stats, err
		}
		applyFFNMemoryLevel(out, mlpInput, level, clusterID)
		stats.LevelsApplied++
		stats.MemoryTokens += level.MemoryTokens
	}
	stats.Applied = true
	return out, stats, nil
}

// ClusterCounts returns the selectable memory count per hierarchy level,
// including the generic-memory slot added after learned clusters.
func (bank *FFNMemoryBank) ClusterCounts() []int {
	if bank == nil || len(bank.Layers) == 0 {
		return nil
	}
	counts := make([]int, len(bank.Layers[0].Levels))
	for i, level := range bank.Layers[0].Levels {
		counts[i] = level.NumClusters + level.AddedGenericSize
	}
	return counts
}

// GenericClusterIDs returns the bank's generic-memory cluster IDs.
func (bank *FFNMemoryBank) GenericClusterIDs() ([]int, error) {
	return GenericClusterIDs(bank.ClusterCounts())
}

// AddGenericToFFNOutput applies the upstream generic-memory fallback: the final
// cluster slot at each hierarchy level.
func (bank *FFNMemoryBank) AddGenericToFFNOutput(dst []float32, ffnOutput []float32, mlpInput []float32, layerID int) ([]float32, []int, FFNMemoryStats, error) {
	clusterIDs, err := bank.GenericClusterIDs()
	if err != nil {
		return nil, nil, FFNMemoryStats{}, err
	}
	out, stats, err := bank.AddToFFNOutput(dst, ffnOutput, mlpInput, layerID, clusterIDs)
	if err != nil {
		return nil, clusterIDs, stats, err
	}
	return out, clusterIDs, stats, nil
}

// AddRoutedToFFNOutput routes query through the offline clustering bank and
// applies the selected hierarchical memories to the FFN output.
func (bank *FFNMemoryBank) AddRoutedToFFNOutput(dst []float32, ffnOutput []float32, mlpInput []float32, router *Bank, query []float32, layerID int) ([]float32, []int, FFNMemoryStats, error) {
	if router == nil {
		return nil, nil, FFNMemoryStats{}, core.NewError("memorypretrain: memory router bank is nil")
	}
	clusterIDs, err := router.ClusterIDs(query)
	if err != nil {
		return nil, nil, FFNMemoryStats{}, err
	}
	clusterIDs, err = padClusterIDsWithGenericFallback(clusterIDs, bank.ClusterCounts())
	if err != nil {
		return nil, nil, FFNMemoryStats{}, err
	}
	out, stats, err := bank.AddToFFNOutput(dst, ffnOutput, mlpInput, layerID, clusterIDs)
	if err != nil {
		return nil, clusterIDs, stats, err
	}
	return out, clusterIDs, stats, nil
}

func padClusterIDsWithGenericFallback(clusterIDs []int, clusterCounts []int) ([]int, error) {
	if len(clusterCounts) == 0 {
		return append([]int(nil), clusterIDs...), nil
	}
	if len(clusterIDs) > len(clusterCounts) {
		return nil, core.Errorf("memorypretrain: cluster ID count %d exceeds memory levels %d", len(clusterIDs), len(clusterCounts))
	}
	out := make([]int, len(clusterCounts))
	for i := range clusterCounts {
		if clusterCounts[i] <= 0 {
			return nil, core.Errorf("memorypretrain: memory level %d cluster count must be positive", i)
		}
		out[i] = clusterCounts[i] - 1
	}
	for i, id := range clusterIDs {
		if id < 0 || id >= clusterCounts[i] {
			return nil, core.Errorf("memorypretrain: cluster ID %d is out of range for memory level %d with %d clusters", id, i, clusterCounts[i])
		}
		out[i] = id
	}
	return out, nil
}

func normaliseFFNMemoryConfig(cfg FFNMemoryConfig) FFNMemoryConfig {
	if len(cfg.MemoryLevels) == 0 {
		cfg.MemoryLevels = []string{"1", "2", "3", "4"}
	}
	if len(cfg.FFNMemoryTokens) == 0 {
		cfg.FFNMemoryTokens = []int{8, 16, 32, 64}
	}
	if len(cfg.NumClusters) == 0 {
		cfg.NumClusters = []int{256, 128, 64, 32}
	}
	if cfg.AddedGenericSize <= 0 {
		cfg.AddedGenericSize = 1
	}
	cfg.ZeroInitialiseW3 = true
	return cfg
}

func validateFFNMemoryConfig(cfg FFNMemoryConfig) error {
	if cfg.HiddenSize <= 0 {
		return core.NewError("memorypretrain: FFN memory hidden size must be positive")
	}
	if cfg.Layers <= 0 {
		return core.NewError("memorypretrain: FFN memory layers must be positive")
	}
	if len(cfg.MemoryLevels) != len(cfg.FFNMemoryTokens) || len(cfg.MemoryLevels) != len(cfg.NumClusters) {
		return core.NewError("memorypretrain: FFN memory level, token, and cluster counts must match")
	}
	for i := range cfg.MemoryLevels {
		if cfg.MemoryLevels[i] == "" {
			return core.Errorf("memorypretrain: FFN memory level %d name is required", i)
		}
		if cfg.FFNMemoryTokens[i] <= 0 {
			return core.Errorf("memorypretrain: FFN memory level %d token count must be positive", i)
		}
		if cfg.NumClusters[i] <= 0 {
			return core.Errorf("memorypretrain: FFN memory level %d cluster count must be positive", i)
		}
	}
	return nil
}

func validateFFNMemoryLevel(level *FFNMemoryLevelWeight, hiddenSize int, clusterID int) error {
	totalClusters := level.NumClusters + level.AddedGenericSize
	if clusterID < 0 || clusterID >= totalClusters {
		return core.Errorf("memorypretrain: FFN memory cluster %d is out of range for level %s", clusterID, level.Name)
	}
	w12Len := totalClusters * hiddenSize * level.MemoryTokens
	if len(level.W1) != w12Len {
		return core.Errorf("memorypretrain: FFN memory level %s W1 length %d does not match %d", level.Name, len(level.W1), w12Len)
	}
	if len(level.W2) != w12Len {
		return core.Errorf("memorypretrain: FFN memory level %s W2 length %d does not match %d", level.Name, len(level.W2), w12Len)
	}
	w3Len := totalClusters * level.MemoryTokens * hiddenSize
	if len(level.W3) != w3Len {
		return core.Errorf("memorypretrain: FFN memory level %s W3 length %d does not match %d", level.Name, len(level.W3), w3Len)
	}
	return nil
}

func applyFFNMemoryLevel(out []float32, mlpInput []float32, level *FFNMemoryLevelWeight, clusterID int) {
	for token := 0; token < level.MemoryTokens; token++ {
		gate := dotFFNMemoryW12(mlpInput, level, clusterID, token, level.W1)
		value := dotFFNMemoryW12(mlpInput, level, clusterID, token, level.W2)
		activated := silu(gate) * value
		for hidden := range out {
			out[hidden] += activated * level.W3[indexFFNMemoryW3(level, clusterID, token, hidden)]
		}
	}
}

func dotFFNMemoryW12(input []float32, level *FFNMemoryLevelWeight, clusterID int, token int, weights []float32) float32 {
	var sum float32
	for hidden, value := range input {
		sum += value * weights[indexFFNMemoryW12(level, clusterID, hidden, token)]
	}
	return sum
}

func indexFFNMemoryW12(level *FFNMemoryLevelWeight, clusterID int, hidden int, token int) int {
	return clusterID*levelHiddenStride(level) + hidden*level.MemoryTokens + token
}

func indexFFNMemoryW3(level *FFNMemoryLevelWeight, clusterID int, token int, hidden int) int {
	return (clusterID*level.MemoryTokens+token)*levelHiddenSize(level) + hidden
}

func levelHiddenStride(level *FFNMemoryLevelWeight) int {
	if level.MemoryTokens == 0 {
		return 0
	}
	totalClusters := level.NumClusters + level.AddedGenericSize
	return len(level.W1) / totalClusters
}

func levelHiddenSize(level *FFNMemoryLevelWeight) int {
	if level.MemoryTokens == 0 {
		return 0
	}
	totalClusters := level.NumClusters + level.AddedGenericSize
	return len(level.W3) / totalClusters / level.MemoryTokens
}

func silu(value float32) float32 {
	return value / (1 + float32(math.Exp(float64(-value))))
}

func initialiseFFNMemoryInputWeights(weights []float32, hiddenSize int, layerID int, levelID int, salt int) {
	if hiddenSize <= 0 {
		return
	}
	std := float32(1 / math.Sqrt(float64(hiddenSize)))
	for i := range weights {
		weights[i] = deterministicInitialWeight(i+salt, layerID, levelID) * std
	}
}

func deterministicInitialWeight(index int, layerID int, levelID int) float32 {
	value := uint64(index+1) * 0x9e3779b97f4a7c15
	value ^= uint64(layerID+1) * 0xbf58476d1ce4e5b9
	value ^= uint64(levelID+1) * 0x94d049bb133111eb
	value ^= value >> 30
	value *= 0xbf58476d1ce4e5b9
	value ^= value >> 27
	value *= 0x94d049bb133111eb
	value ^= value >> 31
	unit := float64(value&((1<<53)-1)) / float64(1<<53)
	centred := float32(2*unit - 1)
	if centred > 0.99 {
		return 0.99
	}
	if centred < -0.99 {
		return -0.99
	}
	return centred
}
