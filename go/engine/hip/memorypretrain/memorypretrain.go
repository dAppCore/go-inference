// SPDX-Licence-Identifier: EUPL-1.2

// Package memorypretrain contains the native hierarchical-memory pretraining
// primitives used by small local models to retrieve context-dependent memory
// blocks for feed-forward injection.
package memorypretrain

import (
	"context"
	"maps"
	"math"
	"slices"

	core "dappco.re/go"
)

const (
	defaultBranchingFactor = 8
	defaultMaxDepth        = 3
	defaultMinClusterSize  = 8
	defaultKMeansIters     = 16
)

// Block is one embedded corpus chunk available to the memory bank.
type Block struct {
	ID        string            `json:"id,omitempty"`
	Text      string            `json:"text,omitempty"`
	Embedding []float32         `json:"embedding,omitempty"`
	Meta      map[string]string `json:"meta,omitempty"`
}

// CorpusRecord is one text block to embed before building a memory bank.
type CorpusRecord struct {
	ID   string            `json:"id,omitempty"`
	Text string            `json:"text,omitempty"`
	Meta map[string]string `json:"meta,omitempty"`
}

// Embedder embeds corpus records with the small anchor model used by the
// hierarchical-memory pretraining pipeline.
type Embedder interface {
	Embed(context.Context, string) ([]float32, error)
}

// EmbedFunc adapts a function into an Embedder.
type EmbedFunc func(context.Context, string) ([]float32, error)

// Embed calls fn(ctx, text).
func (fn EmbedFunc) Embed(ctx context.Context, text string) ([]float32, error) {
	if fn == nil {
		return nil, core.NewError("memorypretrain: embed function is nil")
	}
	return fn(ctx, text)
}

// BuildConfig controls deterministic hierarchical KMeans construction.
type BuildConfig struct {
	BranchingFactor int `json:"branching_factor"`
	MaxDepth        int `json:"max_depth"`
	MinClusterSize  int `json:"min_cluster_size"`
	KMeansIters     int `json:"kmeans_iters"`
}

// Node is one centroid in the hierarchical memory tree.
type Node struct {
	ID       int       `json:"id"`
	Parent   int       `json:"parent,omitempty"`
	Depth    int       `json:"depth"`
	Centroid []float32 `json:"centroid,omitempty"`
	Children []int     `json:"children,omitempty"`
	BlockIDs []int     `json:"block_ids,omitempty"`
}

// Bank is a compact retrieval structure built from embedded blocks.
type Bank struct {
	Dimension int         `json:"dimension"`
	Blocks    []Block     `json:"blocks,omitempty"`
	Nodes     []Node      `json:"nodes,omitempty"`
	Root      int         `json:"root"`
	Config    BuildConfig `json:"config"`
}

// Retrieval is one block returned for a query vector.
type Retrieval struct {
	BlockIndex int     `json:"block_index"`
	BlockID    string  `json:"block_id,omitempty"`
	Score      float32 `json:"score"`
	Text       string  `json:"text,omitempty"`
}

// ClusterAssignment is one routed cluster ID for a hierarchy level.
type ClusterAssignment struct {
	Level          int `json:"level"`
	NodeID         int `json:"node_id"`
	ParentNodeID   int `json:"parent_node_id"`
	LocalClusterID int `json:"local_cluster_id"`
	ClusterID      int `json:"cluster_id"`
}

// InjectionConfig controls additive memory injection into a feed-forward
// activation. Scale is applied after score normalisation; 0 defaults to 1.
type InjectionConfig struct {
	TopK               int     `json:"top_k"`
	Scale              float32 `json:"scale,omitempty"`
	PositiveScoresOnly bool    `json:"positive_scores_only,omitempty"`
}

// InjectionStats describes one additive memory injection.
type InjectionStats struct {
	Retrieved int     `json:"retrieved"`
	WeightSum float32 `json:"weight_sum"`
	Scale     float32 `json:"scale"`
	Applied   bool    `json:"applied"`
}

// BuildBank builds a deterministic hierarchical KMeans memory bank.
func BuildBank(blocks []Block, cfg BuildConfig) (*Bank, error) {
	cfg = normaliseBuildConfig(cfg)
	if len(blocks) == 0 {
		return nil, core.NewError("memorypretrain: blocks are required")
	}
	dim, err := validateBlocks(blocks)
	if err != nil {
		return nil, err
	}
	copied := cloneBlocks(blocks)
	bank := &Bank{
		Dimension: dim,
		Blocks:    copied,
		Root:      0,
		Config:    cfg,
	}
	all := make([]int, len(copied))
	for i := range all {
		all[i] = i
	}
	bank.buildNode(-1, 0, all)
	return bank, nil
}

// BuildBankFromCorpus embeds records with embedder and builds a hierarchical
// memory bank from the resulting embedded blocks.
func BuildBankFromCorpus(ctx context.Context, embedder Embedder, records []CorpusRecord, cfg BuildConfig) (*Bank, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if embedder == nil {
		return nil, core.NewError("memorypretrain: embedder is nil")
	}
	if len(records) == 0 {
		return nil, core.NewError("memorypretrain: corpus records are required")
	}
	blocks := make([]Block, len(records))
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		embedding, err := embedder.Embed(ctx, record.Text)
		if err != nil {
			return nil, core.Errorf("memorypretrain: embed record %d: %v", i, err)
		}
		blocks[i] = Block{
			ID:        record.ID,
			Text:      record.Text,
			Embedding: embedding,
			Meta:      record.Meta,
		}
	}
	return BuildBank(blocks, cfg)
}

// Retrieve returns the top-k nearest blocks from the routed leaf cluster.
func (bank *Bank) Retrieve(query []float32, k int) ([]Retrieval, error) {
	return bank.RetrieveInto(nil, query, k)
}

// ClusterIDs returns upstream-compatible hierarchical cluster IDs for query.
func (bank *Bank) ClusterIDs(query []float32) ([]int, error) {
	assignments, err := bank.ClusterAssignments(query)
	if err != nil {
		return nil, err
	}
	ids := make([]int, len(assignments))
	for i, assignment := range assignments {
		ids[i] = assignment.ClusterID
	}
	return ids, nil
}

// ClusterAssignments routes query through the hierarchy and records one
// assignment per reached level. ClusterID uses parent*branching+local indexing,
// matching the learned hierarchical KMeans retriever format.
func (bank *Bank) ClusterAssignments(query []float32) ([]ClusterAssignment, error) {
	if bank == nil {
		return nil, core.NewError("memorypretrain: bank is nil")
	}
	if len(query) != bank.Dimension {
		return nil, core.Errorf("memorypretrain: query dimension %d does not match bank dimension %d", len(query), bank.Dimension)
	}
	if len(bank.Nodes) == 0 || bank.Root < 0 || bank.Root >= len(bank.Nodes) {
		return nil, core.NewError("memorypretrain: bank has no root node")
	}
	cfg := normaliseBuildConfig(bank.Config)
	assignments := make([]ClusterAssignment, 0, cfg.MaxDepth)
	parentID := bank.Root
	parentClusterID := 0
	for {
		parent := &bank.Nodes[parentID]
		if len(parent.Children) == 0 {
			break
		}
		nodeID := bank.nearestNode(query, parent.Children)
		localID := localClusterID(parent.Children, nodeID)
		clusterID := parentClusterID*cfg.BranchingFactor + localID
		assignments = append(assignments, ClusterAssignment{
			Level:          bank.Nodes[nodeID].Depth,
			NodeID:         nodeID,
			ParentNodeID:   parentID,
			LocalClusterID: localID,
			ClusterID:      clusterID,
		})
		parentID = nodeID
		parentClusterID = clusterID
	}
	return assignments, nil
}

// GenericClusterIDs returns the upstream generic-memory fallback: the last
// cluster index at each memory level.
func GenericClusterIDs(numClusters []int) ([]int, error) {
	if len(numClusters) == 0 {
		return nil, core.NewError("memorypretrain: memory cluster counts are required")
	}
	ids := make([]int, len(numClusters))
	for i, count := range numClusters {
		if count <= 0 {
			return nil, core.Errorf("memorypretrain: memory level %d cluster count must be positive", i)
		}
		ids[i] = count - 1
	}
	return ids, nil
}

// RetrieveInto appends the top-k nearest blocks to dst after resetting it.
func (bank *Bank) RetrieveInto(dst []Retrieval, query []float32, k int) ([]Retrieval, error) {
	if bank == nil {
		return nil, core.NewError("memorypretrain: bank is nil")
	}
	if len(query) != bank.Dimension {
		return nil, core.Errorf("memorypretrain: query dimension %d does not match bank dimension %d", len(query), bank.Dimension)
	}
	if k <= 0 {
		return nil, core.NewError("memorypretrain: retrieval k must be positive")
	}
	if len(bank.Nodes) == 0 || bank.Root < 0 || bank.Root >= len(bank.Nodes) {
		return nil, core.NewError("memorypretrain: bank has no root node")
	}
	nodeID := bank.Root
	for {
		node := &bank.Nodes[nodeID]
		if len(node.Children) == 0 {
			break
		}
		nodeID = bank.nearestNode(query, node.Children)
	}
	blockIDs := bank.Nodes[nodeID].BlockIDs
	if len(blockIDs) == 0 {
		return dst[:0], nil
	}
	scored := dst[:0]
	for _, blockIndex := range blockIDs {
		block := bank.Blocks[blockIndex]
		scored = append(scored, Retrieval{
			BlockIndex: blockIndex,
			BlockID:    block.ID,
			Score:      cosine(query, block.Embedding),
			Text:       block.Text,
		})
	}
	slices.SortFunc(scored, func(a, b Retrieval) int {
		if a.Score == b.Score {
			if a.BlockIndex < b.BlockIndex {
				return -1
			}
			if a.BlockIndex > b.BlockIndex {
				return 1
			}
			return 0
		}
		if a.Score > b.Score {
			return -1
		}
		return 1
	})
	if k > len(scored) {
		k = len(scored)
	}
	return scored[:k], nil
}

// InjectAdditive retrieves memory blocks for query and adds their weighted
// embedding into hidden, returning the activation in dst. The memory bank
// embedding dimension must match hidden; model-specific projection layers can
// sit around this primitive when the anchor model uses a different width.
func (bank *Bank) InjectAdditive(dst []float32, hidden []float32, query []float32, scratch []Retrieval, cfg InjectionConfig) ([]float32, []Retrieval, InjectionStats, error) {
	if len(hidden) != bankDimension(bank) {
		return nil, scratch[:0], InjectionStats{}, core.Errorf("memorypretrain: hidden dimension %d does not match bank dimension %d", len(hidden), bankDimension(bank))
	}
	cfg = normaliseInjectionConfig(cfg)
	retrievals, err := bank.RetrieveInto(scratch, query, cfg.TopK)
	if err != nil {
		return nil, retrievals, InjectionStats{}, err
	}
	out := resetFloat32(dst, len(hidden))
	copy(out, hidden)
	stats := InjectionStats{Retrieved: len(retrievals), Scale: cfg.Scale}
	if len(retrievals) == 0 {
		return out, retrievals, stats, nil
	}
	for _, retrieval := range retrievals {
		weight := retrieval.Score
		if cfg.PositiveScoresOnly && weight < 0 {
			weight = 0
		}
		stats.WeightSum += weight
	}
	if stats.WeightSum == 0 {
		uniform := cfg.Scale / float32(len(retrievals))
		for _, retrieval := range retrievals {
			block := bank.Blocks[retrieval.BlockIndex]
			addScaledInto(out, block.Embedding, uniform)
		}
		stats.WeightSum = 1
		stats.Applied = true
		return out, retrievals, stats, nil
	}
	invWeightSum := cfg.Scale / stats.WeightSum
	for _, retrieval := range retrievals {
		weight := retrieval.Score
		if cfg.PositiveScoresOnly && weight < 0 {
			weight = 0
		}
		if weight == 0 {
			continue
		}
		block := bank.Blocks[retrieval.BlockIndex]
		addScaledInto(out, block.Embedding, weight*invWeightSum)
	}
	stats.Applied = true
	return out, retrievals, stats, nil
}

func (bank *Bank) buildNode(parent int, depth int, blockIDs []int) int {
	id := len(bank.Nodes)
	node := Node{
		ID:       id,
		Parent:   parent,
		Depth:    depth,
		Centroid: centroidForBlocks(bank.Blocks, blockIDs, bank.Dimension),
		BlockIDs: append([]int(nil), blockIDs...),
	}
	bank.Nodes = append(bank.Nodes, node)
	if depth >= bank.Config.MaxDepth || len(blockIDs) <= bank.Config.MinClusterSize {
		return id
	}
	clusters := bank.kmeans(blockIDs)
	if len(clusters) <= 1 {
		return id
	}
	children := make([]int, 0, len(clusters))
	for _, cluster := range clusters {
		if len(cluster) == 0 {
			continue
		}
		children = append(children, bank.buildNode(id, depth+1, cluster))
	}
	bank.Nodes[id].Children = children
	if len(children) > 0 {
		bank.Nodes[id].BlockIDs = nil
	}
	return id
}

func (bank *Bank) kmeans(blockIDs []int) [][]int {
	k := min(bank.Config.BranchingFactor, len(blockIDs))
	centroids := initialCentroids(bank.Blocks, blockIDs, k)
	assignments := make([]int, len(blockIDs))
	for i := range assignments {
		assignments[i] = -1
	}
	for range bank.Config.KMeansIters {
		changed := false
		for i, blockID := range blockIDs {
			next := nearestVector(bank.Blocks[blockID].Embedding, centroids)
			if assignments[i] != next {
				assignments[i] = next
				changed = true
			}
		}
		nextCentroids := make([][]float32, len(centroids))
		counts := make([]int, len(centroids))
		for i := range nextCentroids {
			nextCentroids[i] = make([]float32, bank.Dimension)
		}
		for i, blockID := range blockIDs {
			cluster := assignments[i]
			counts[cluster]++
			addInto(nextCentroids[cluster], bank.Blocks[blockID].Embedding)
		}
		for i := range nextCentroids {
			if counts[i] == 0 {
				copy(nextCentroids[i], centroids[i])
				continue
			}
			scaleInto(nextCentroids[i], 1/float32(counts[i]))
		}
		centroids = nextCentroids
		if !changed {
			break
		}
	}
	clusters := make([][]int, len(centroids))
	for i, blockID := range blockIDs {
		cluster := assignments[i]
		clusters[cluster] = append(clusters[cluster], blockID)
	}
	out := clusters[:0]
	for _, cluster := range clusters {
		if len(cluster) > 0 {
			out = append(out, cluster)
		}
	}
	return out
}

func (bank *Bank) nearestNode(query []float32, nodeIDs []int) int {
	bestID := nodeIDs[0]
	bestScore := cosine(query, bank.Nodes[bestID].Centroid)
	for _, nodeID := range nodeIDs[1:] {
		score := cosine(query, bank.Nodes[nodeID].Centroid)
		if score > bestScore || score == bestScore && nodeID < bestID {
			bestID = nodeID
			bestScore = score
		}
	}
	return bestID
}

func localClusterID(nodeIDs []int, nodeID int) int {
	for i, candidate := range nodeIDs {
		if candidate == nodeID {
			return i
		}
	}
	return -1
}

func normaliseBuildConfig(cfg BuildConfig) BuildConfig {
	if cfg.BranchingFactor <= 0 {
		cfg.BranchingFactor = defaultBranchingFactor
	}
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = defaultMaxDepth
	}
	if cfg.MinClusterSize <= 0 {
		cfg.MinClusterSize = defaultMinClusterSize
	}
	if cfg.KMeansIters <= 0 {
		cfg.KMeansIters = defaultKMeansIters
	}
	return cfg
}

func normaliseInjectionConfig(cfg InjectionConfig) InjectionConfig {
	if cfg.TopK <= 0 {
		cfg.TopK = 4
	}
	if cfg.Scale == 0 {
		cfg.Scale = 1
	}
	return cfg
}

func bankDimension(bank *Bank) int {
	if bank == nil {
		return 0
	}
	return bank.Dimension
}

func validateBlocks(blocks []Block) (int, error) {
	dim := len(blocks[0].Embedding)
	if dim == 0 {
		return 0, core.NewError("memorypretrain: block embedding is required")
	}
	for i, block := range blocks {
		if len(block.Embedding) != dim {
			return 0, core.Errorf("memorypretrain: block %d dimension %d does not match %d", i, len(block.Embedding), dim)
		}
		for _, value := range block.Embedding {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return 0, core.Errorf("memorypretrain: block %d contains non-finite embedding value", i)
			}
		}
	}
	return dim, nil
}

func cloneBlocks(blocks []Block) []Block {
	out := make([]Block, len(blocks))
	for i, block := range blocks {
		out[i] = Block{
			ID:        block.ID,
			Text:      block.Text,
			Embedding: append([]float32(nil), block.Embedding...),
			Meta:      cloneMap(block.Meta),
		}
	}
	return out
}

func cloneMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	maps.Copy(out, values)
	return out
}

func centroidForBlocks(blocks []Block, blockIDs []int, dim int) []float32 {
	centroid := make([]float32, dim)
	if len(blockIDs) == 0 {
		return centroid
	}
	for _, blockID := range blockIDs {
		addInto(centroid, blocks[blockID].Embedding)
	}
	scaleInto(centroid, 1/float32(len(blockIDs)))
	return centroid
}

func initialCentroids(blocks []Block, blockIDs []int, k int) [][]float32 {
	centroids := make([][]float32, 0, k)
	centroids = append(centroids, append([]float32(nil), blocks[blockIDs[0]].Embedding...))
	for len(centroids) < k {
		bestBlock := blockIDs[0]
		bestDistance := float32(-1)
		for _, blockID := range blockIDs {
			minDistance := float32(math.MaxFloat32)
			for _, centroid := range centroids {
				distance := squaredDistance(blocks[blockID].Embedding, centroid)
				if distance < minDistance {
					minDistance = distance
				}
			}
			if minDistance > bestDistance || minDistance == bestDistance && blockID < bestBlock {
				bestBlock = blockID
				bestDistance = minDistance
			}
		}
		centroids = append(centroids, append([]float32(nil), blocks[bestBlock].Embedding...))
	}
	return centroids
}

func nearestVector(vector []float32, candidates [][]float32) int {
	best := 0
	bestScore := cosine(vector, candidates[0])
	for i := 1; i < len(candidates); i++ {
		score := cosine(vector, candidates[i])
		if score > bestScore {
			best = i
			bestScore = score
		}
	}
	return best
}

func addInto(dst []float32, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func addScaledInto(dst []float32, src []float32, scale float32) {
	for i := range dst {
		dst[i] += src[i] * scale
	}
}

func resetFloat32(dst []float32, n int) []float32 {
	if cap(dst) < n {
		return make([]float32, n)
	}
	return dst[:n]
}

func scaleInto(values []float32, scale float32) {
	for i := range values {
		values[i] *= scale
	}
}

func cosine(a []float32, b []float32) float32 {
	var dot float64
	var aNorm float64
	var bNorm float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		aNorm += av * av
		bNorm += bv * bv
	}
	if aNorm == 0 || bNorm == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(aNorm) * math.Sqrt(bNorm)))
}

func squaredDistance(a []float32, b []float32) float32 {
	var sum float32
	for i := range a {
		delta := a[i] - b[i]
		sum += delta * delta
	}
	return sum
}
