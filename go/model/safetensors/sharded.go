// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Standard HF checkpoint file names within a model directory.
const (
	indexName  = "model.safetensors.index.json"
	singleName = "model.safetensors"
)

// isShardName reports whether name is a sharded-checkpoint weight file in either spelling HF
// snapshots ship: the standard model-NNNNN-of-NNNNN.safetensors, or the
// model.safetensors-NNNNN-of-NNNNN.safetensors variant (Qwen3.5 snapshots name their shards this
// way). Both ordinal runs must be pure digits, any width; the bare singleName carries no ordinal
// suffix and never matches, so the single-file and shard-scan routes stay disjoint.
func isShardName(name string) bool {
	body, ok := core.CutSuffix(name, ".safetensors")
	if !ok {
		return false
	}
	tail, ok := core.CutPrefix(body, "model")
	if !ok {
		return false
	}
	tail, _ = core.CutPrefix(tail, ".safetensors") // the model.safetensors-NNNNN-of-NNNNN spelling
	tail, ok = core.CutPrefix(tail, "-")
	if !ok {
		return false
	}
	lo, hi, found := core.Cut(tail, "-of-")
	return found && allDigits(lo) && allDigits(hi)
}

// allDigits reports whether s is one or more ASCII digits — the shard-ordinal alphabet.
func allDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// scanShards lists dir's shard-named weight files for an INDEX-LESS sharded checkpoint, sorted
// so the merge order is deterministic. Restricted to isShardName matches — a bare *.safetensors
// glob would sweep stray tensor files (an adapter.safetensors) into the model map.
func scanShards(dir string) []string {
	var shards []string
	for _, p := range core.PathGlob(core.PathJoin(dir, "*.safetensors")) {
		if isShardName(core.PathBase(p)) {
			shards = append(shards, p)
		}
	}
	core.SliceSort(shards)
	return shards
}

// shardIndex is the subset of model.safetensors.index.json LoadDir consumes: weight_map names
// each tensor to the shard file holding it. (The metadata/total_size block is informational
// and ignored.)
type shardIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

// LoadDir loads a gemma4 checkpoint directory, handling the layouts HF emits: a SHARDED
// checkpoint (model.safetensors.index.json + model-NNNNN-of-NNNNN.safetensors shards), a
// SINGLE model.safetensors, or an INDEX-LESS set of shard-named files in either spelling
// (isShardName), merged in sorted order. It returns the merged name→Tensor map — the same shape Parse/Load
// give for one blob — so the gemma4 assembler is identical however the weights were split.
// Each shard is read+parsed ONCE (cached by file name), not once per tensor; this is the thin
// I/O layer over Parse the single-blob LoadGemma4BF16 doc flags. Loading a real multi-GB
// checkpoint is a deliberate, memory-heavy step — each shard's bytes stay resident, sub-sliced
// by its tensors (no copy), so the whole model is in memory once merged.
func LoadDir(dir string) (map[string]Tensor, error) {
	idxPath := core.PathJoin(dir, indexName)
	if coreio.Local.IsFile(idxPath) {
		idxStr, err := coreio.Local.Read(idxPath)
		if err != nil {
			return nil, core.E("safetensors.LoadDir", "read "+idxPath, err)
		}
		var idx shardIndex
		if r := core.JSONUnmarshalString(idxStr, &idx); !r.OK { // zero-copy: idxStr is already a string
			return nil, core.NewError("safetensors.LoadDir: " + indexName + " parse failed")
		}
		if len(idx.WeightMap) == 0 {
			return nil, core.NewError("safetensors.LoadDir: " + indexName + " has an empty weight_map")
		}
		shards := make(map[string]map[string]Tensor) // each shard parsed once, keyed by file name
		out := make(map[string]Tensor, len(idx.WeightMap))
		for name, shard := range idx.WeightMap {
			parsed, ok := shards[shard]
			if !ok {
				p, err := Load(core.PathJoin(dir, shard))
				if err != nil {
					return nil, core.E("safetensors.LoadDir", "load shard "+shard, err)
				}
				shards[shard] = p
				parsed = p
			}
			t, ok := parsed[name]
			if !ok {
				return nil, core.NewError("safetensors.LoadDir: index maps " + name + " to " + shard + " but that shard lacks it")
			}
			out[name] = t
		}
		return out, nil
	}
	singlePath := core.PathJoin(dir, singleName)
	if coreio.Local.IsFile(singlePath) {
		return Load(singlePath)
	}
	if shardPaths := scanShards(dir); len(shardPaths) > 0 {
		out := make(map[string]Tensor)
		for _, p := range shardPaths {
			parsed, err := Load(p)
			if err != nil {
				return nil, core.E("safetensors.LoadDir", "load shard "+core.PathBase(p), err)
			}
			for name, t := range parsed {
				if _, dup := out[name]; dup {
					// Refuse, don't pick: two shards claiming one tensor is a malformed checkpoint,
					// and a silent winner is a coherent-but-wrong hazard downstream.
					return nil, core.NewError("safetensors.LoadDir: tensor " + name + " appears in more than one shard of " + dir)
				}
				out[name] = t
			}
		}
		return out, nil
	}
	return nil, core.NewError("safetensors.LoadDir: no " + indexName + ", " + singleName + " or model shard files (model-NNNNN-of-NNNNN.safetensors) found in " + dir)
}

// DirMapping is a memory-mapped checkpoint directory: Shards holds every mmap'd shard (each a
// *Mapping), and Tensors is the merged name→Tensor view across them — the same shape LoadDir
// returns, but every Tensor.Data views its shard's page-aligned mmap (zero-copy). Close unmaps
// all shards; it MUST outlive every Tensor view and every GPU buffer taken over a shard's Data.
type DirMapping struct {
	Shards  []*Mapping
	Tensors map[string]Tensor
	// widened records the heap Data ranges of tensors WidenF16TensorsToBF16 converted from F16 to
	// BF16 — they are no longer shard mmap views, so the zero-copy binder consults IsWidened to bind
	// them resident rather than failing its wrong-mapping guard. nil until a widening pass runs.
	widened []widenedRange
	// owned records the heap Data ranges of load-time SYNTHESISED tensors AdoptOwnedTensors swept
	// up (packExperts packs, the b1→b2 repack — see owned.go): registered legitimate off-shard
	// buffers the binder binds resident, evicted with the owning session. nil until adoption runs.
	owned []widenedRange
}

// widenedRange is the [start,end) heap-address span of one F16→BF16-widened tensor's Data.
type widenedRange struct{ start, end uintptr }

// LoadDirMmap is LoadDir's zero-copy sibling: it memory-maps each shard (page-aligned) instead
// of reading it into the heap, so the whole checkpoint is VIEWED, not copied — the no-cgo Go
// counterpart to mlx-c's mmap loader. Handles the same layouts as LoadDir: sharded (index +
// shards), single (model.safetensors), and index-less shard-named files in either spelling
// (isShardName). Pair with DirMapping.Close to unmap.
//
//	dm, err := safetensors.LoadDirMmap(dir)
//	defer dm.Close()
//	// dm.Tensors[name].Data views a page-aligned shard mmap — bind one no-copy GPU buffer per
//	// shard (dm.Shards[i].Data) and address each tensor at its byte offset into that shard.
func LoadDirMmap(dir string) (*DirMapping, error) {
	closeAll := func(ms []*Mapping) {
		for _, m := range ms {
			_ = m.Close()
		}
	}
	idxPath := core.PathJoin(dir, indexName)
	if coreio.Local.IsFile(idxPath) {
		idxStr, err := coreio.Local.Read(idxPath)
		if err != nil {
			return nil, core.E("safetensors.LoadDirMmap", "read "+idxPath, err)
		}
		var idx shardIndex
		if r := core.JSONUnmarshalString(idxStr, &idx); !r.OK { // zero-copy: idxStr is already a string
			return nil, core.NewError("safetensors.LoadDirMmap: " + indexName + " parse failed")
		}
		if len(idx.WeightMap) == 0 {
			return nil, core.NewError("safetensors.LoadDirMmap: " + indexName + " has an empty weight_map")
		}
		byShard := make(map[string]*Mapping) // each shard mapped once, by file name
		var shards []*Mapping
		out := make(map[string]Tensor, len(idx.WeightMap))
		for name, shard := range idx.WeightMap {
			m, ok := byShard[shard]
			if !ok {
				mm, err := LoadMmap(core.PathJoin(dir, shard))
				if err != nil {
					closeAll(shards)
					return nil, core.E("safetensors.LoadDirMmap", "map shard "+shard, err)
				}
				byShard[shard] = mm
				shards = append(shards, mm)
				m = mm
			}
			t, ok := m.Tensors[name]
			if !ok {
				closeAll(shards)
				return nil, core.NewError("safetensors.LoadDirMmap: index maps " + name + " to " + shard + " but that shard lacks it")
			}
			out[name] = t
		}
		return &DirMapping{Shards: shards, Tensors: out}, nil
	}
	singlePath := core.PathJoin(dir, singleName)
	if coreio.Local.IsFile(singlePath) {
		m, err := LoadMmap(singlePath)
		if err != nil {
			return nil, err
		}
		return &DirMapping{Shards: []*Mapping{m}, Tensors: m.Tensors}, nil
	}
	if shardPaths := scanShards(dir); len(shardPaths) > 0 {
		var shards []*Mapping
		out := make(map[string]Tensor)
		for _, p := range shardPaths {
			m, err := LoadMmap(p)
			if err != nil {
				closeAll(shards)
				return nil, core.E("safetensors.LoadDirMmap", "map shard "+core.PathBase(p), err)
			}
			shards = append(shards, m)
			for name, t := range m.Tensors {
				if _, dup := out[name]; dup {
					// Refuse, don't pick — mirrors LoadDir's duplicate-tensor stance.
					closeAll(shards)
					return nil, core.NewError("safetensors.LoadDirMmap: tensor " + name + " appears in more than one shard of " + dir)
				}
				out[name] = t
			}
		}
		return &DirMapping{Shards: shards, Tensors: out}, nil
	}
	return nil, core.NewError("safetensors.LoadDirMmap: no " + indexName + ", " + singleName + " or model shard files (model-NNNNN-of-NNNNN.safetensors) found in " + dir)
}

// Close unmaps every shard. Safe on a nil mapping; call exactly once after all views + GPU
// buffers over the shards are done.
func (d *DirMapping) Close() error {
	if d == nil {
		return nil
	}
	var firstErr error
	for _, m := range d.Shards {
		if err := m.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	d.Shards = nil
	d.Tensors = nil
	// Drop the owned-tensor registrations: the ranges' backing slices die with the model, and a
	// stale range must never match a later allocation at a recycled address (see owned.go).
	d.owned = nil
	return firstErr
}
