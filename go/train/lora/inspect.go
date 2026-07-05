// SPDX-Licence-Identifier: EUPL-1.2

// Adapter inspection: reads adapter_config.json and hashes an adapter's
// weight files into a reproducible AdapterInfo identity, independent of
// which engine eventually loads the adapter onto a base model.

package lora

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

// errAdapterPathRequired is the sentinel returned by Inspect when the
// caller passes an empty adapter path. Hoisted to a package var so the
// guard does not allocate on every Inspect call.
var errAdapterPathRequired = core.NewError("lora: adapter path is required")

// errResultFailed is the fallback sentinel returned by resultError when
// a core.Result reports !OK but its Value is not an error.
var errResultFailed = core.NewError("core result failed")

// AdapterInfo is the reproducible identity for an inspected LoRA adapter:
// its adapter_config.json metadata plus a content hash over the config and
// weight files. Complements AdapterRef — AdapterRef.ID() is a cheap
// name+path hash for registry bookkeeping; AdapterInfo.Hash is a content
// hash requiring a filesystem read, used to detect a changed adapter body
// under an unchanged path.
type AdapterInfo struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

// IsEmpty reports whether the adapter info has no meaningful fields set.
func (info AdapterInfo) IsEmpty() bool {
	return info.Name == "" && info.Path == "" && info.Hash == "" && info.Rank == 0 && info.Alpha == 0 && info.Scale == 0 && len(info.TargetKeys) == 0
}

// Identity projects an inspected AdapterInfo into the portable,
// engine-agnostic state.AdapterIdentity consumed by capability.AdapterModel,
// ai/differential_loader, and state.CheckWakeCompatibility — the shapes any
// engine (go-mlx, go-rocm, go-cpu) needs once it has inspected a LoRA
// adapter with Inspect/InspectAdapter but must report or compare it in the
// shared identity schema rather than this package's own AdapterInfo.
//
// Like Manifest.Identity() in model/pack, this is an identity projection,
// not a lossless mirror: Name is a registry/display label rather than part
// of content identity, and Scale is fully derived from Rank and Alpha (see
// NormalizeAdapterConfig) — state.AdapterIdentity does not duplicate a
// derivable field. Both are intentionally dropped.
//
//	id := info.Identity()
func (info AdapterInfo) Identity() state.AdapterIdentity {
	return state.AdapterIdentity{
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		TargetKeys: core.SliceClone(info.TargetKeys),
	}
}

// InspectAdapter reads adapter_config.json and hashes adapter files.
//
//	info, err := lora.InspectAdapter("/path/to/adapter")
func InspectAdapter(path string) (AdapterInfo, error) {
	return Inspect(path, path)
}

// Inspect reads adapter_config.json at path and records identityPath as the
// user-facing path (which may differ from path when the adapter was staged
// from a Medium).
//
//	info, err := lora.Inspect(stagedPath, originalPath)
func Inspect(path string, identityPath string) (AdapterInfo, error) {
	if path == "" {
		return AdapterInfo{}, errAdapterPathRequired
	}
	// HasSuffix is called by both adapterConfigPath and hashAdapter on the
	// same path argument; compute it once and pass the result through the
	// internal variants so the SIMD scan only runs once per Inspect.
	isSafetensors := core.HasSuffix(path, ".safetensors")
	configPath := adapterConfigPathPrecomputed(path, isSafetensors)
	read := core.ReadFile(configPath)
	if !read.OK {
		return AdapterInfo{}, core.E("lora.Inspect", "read adapter_config.json", resultError(read))
	}
	// Cache the type assertion: read.Value is consumed once by the JSON
	// unmarshal and once by hashAdapter — both expect []byte. The
	// compiler treats each .([]byte) as an independent type-assert call,
	// so caching saves the second assertion and its associated iface-table
	// probe on every successful Inspect.
	configBytes := read.Value.([]byte)
	cfg, err := ParseAdapterConfig(configBytes)
	if err != nil {
		return AdapterInfo{}, core.E("lora.Inspect", "parse adapter_config.json", err)
	}
	info := AdapterInfo{
		Name:       core.PathBase(identityPath),
		Path:       identityPath,
		Rank:       cfg.Rank,
		Alpha:      cfg.Alpha,
		Scale:      cfg.Scale,
		TargetKeys: cfg.TargetKeys,
	}
	info.Hash = hashAdapterPrecomputed(path, configBytes, isSafetensors)
	return info, nil
}

func adapterConfigPath(path string) string {
	return adapterConfigPathPrecomputed(path, core.HasSuffix(path, ".safetensors"))
}

// adapterConfigSuffix carries the leading separator inline so the
// concat-path can drop it cheaply when the input already ends in '/'
// (matching filepath.Join's separator-collapse semantics).
const adapterConfigSuffix = "/adapter_config.json"

// joinDirChildPattern concatenates a directory path with a relative
// child segment, collapsing the duplicate separator when dir already
// ends in '/'. Skips the filepath.Clean trip core.PathJoin takes; the
// adapter directory paths fed in here are already canonical (PathAbs +
// MkdirAll output, or caller-supplied non-empty roots validated
// upstream), so the only normalisation needed is the trailing-slash
// collapse rule. An empty dir falls back to a bare child segment to
// preserve PathJoin's "empty root = relative result" semantics.
func joinDirChildPattern(dir, child string) string {
	if dir == "" {
		return child
	}
	if dir[len(dir)-1] == '/' {
		return dir + child
	}
	return dir + "/" + child
}

// adapterConfigPathPrecomputed is the precomputed-suffix variant of
// adapterConfigPath; the Inspect hot path computes the .safetensors
// suffix check once and threads the result through this helper.
//
// Builds the joined path with a direct concat instead of routing through
// core.PathJoin (filepath.Join → filepath.Clean): filepath.Clean always
// allocates an internal lazybuf even when the inputs are already canonical,
// roughly doubling the cost of producing the result string. Both Inspect
// callers feed an already-cleaned adapter path, so the only normalisation
// we need is the "collapse a duplicate '/'" rule that filepath.Join uses
// when joining a path that already ends in '/'.
func adapterConfigPathPrecomputed(path string, isSafetensors bool) string {
	base := path
	if isSafetensors {
		// PathDir returns a substring of path (no alloc); strip the
		// trailing weight-file segment so the join targets the parent dir.
		base = core.PathDir(path)
	}
	// Trailing-slash collapse: when base ends in '/', skip the leading
	// '/' from adapterConfigSuffix to avoid producing "//adapter_config".
	if len(base) > 0 && base[len(base)-1] == '/' {
		return base + adapterConfigSuffix[1:]
	}
	return base + adapterConfigSuffix
}

func hashAdapter(path string, config []byte) string {
	return hashAdapterPrecomputed(path, config, core.HasSuffix(path, ".safetensors"))
}

// hashAdapterPrecomputed is the precomputed-suffix variant of
// hashAdapter; the Inspect hot path computes the .safetensors suffix
// check once and threads the result through this helper to avoid the
// second SIMD scan.
func hashAdapterPrecomputed(path string, config []byte, isSafetensors bool) string {
	// Resolve weight paths first so we know the worst-case parts capacity
	// (config hash + one per weight file). The directory branch always
	// allocates a fresh slice from PathGlob; the file branch can skip the
	// throwaway 1-elem slice the previous code allocated unconditionally.
	var paths []string
	if isSafetensors {
		paths = []string{path}
	} else {
		// joinDirChildPattern skips the filepath.Clean trip core.PathJoin
		// would take — filepath.Glob handles trailing-slash / double-slash
		// patterns identically, so the only normalisation needed is the
		// "empty root = relative result" guard joinDirChildPattern already
		// provides. Shaves the lazybuf alloc filepath.Clean unconditionally
		// makes from the pattern build.
		paths = core.PathGlob(joinDirChildPattern(path, "*.safetensors"))
	}
	slices.Sort(paths)
	// Hash each input on the stack ([32]byte from core.SHA256), then
	// hex-encode straight into a single pre-sized buffer separated by
	// '\n'. A parts []string + one fresh hex string per input plus a
	// Join result string would cost (N+3) allocs for N weight files; the
	// single-buffer form drops that to ONE buffer alloc + the final outer
	// HexEncode, regardless of file count. SHA-256 still dominates timing
	// on real weights; allocs shed are the per-call constant cost.
	configSum := core.SHA256(config)
	// One hex digest is 64 bytes; the joiner adds one '\n' between
	// each consecutive pair. Worst case = config + all weight files
	// successfully read, so size for that ceiling and slice down once
	// the read loop finishes.
	totalCount := 1 + len(paths)
	buf := make([]byte, totalCount*64+(totalCount-1))
	hex.Encode(buf[:64], configSum[:])
	written := 64
	// Stream each weight file through a single reset-per-file SHA-256
	// accumulator instead of reading the whole shard into a heap buffer
	// via core.ReadFile + core.SHA256. core.ReadFile materialises the
	// entire weight file (MBs on a real adapter) into one allocation that
	// scales linearly with shard size — the dominant B/op cost on every
	// model load that attaches a LoRA. core.Copy (io.Copy) feeds the file
	// through the hasher in fixed 32KiB chunks, so B/op goes flat against
	// shard size. SHA-256 is chunk-invariant, so the digest (and the
	// final adapter identity hash) is bit-identical to the
	// read-whole-then-hash form. The hasher is allocated lazily on the
	// first weight file so the config-only path keeps its zero-shard
	// alloc profile.
	//
	// Small shards (<= streamHashMinBytes) stay on the in-memory path:
	// io.Copy allocates its own 32KiB scratch buffer, so for a file at
	// or below that size streaming would cost MORE bytes than reading it
	// whole. The gate keeps small-adapter inspection strictly at or below
	// its prior B/op while large/real shards win.
	var hasher hashWriter
	for _, weightPath := range paths {
		weightSum, ok := hashWeightFile(weightPath, &hasher)
		if !ok {
			continue
		}
		buf[written] = '\n'
		hex.Encode(buf[written+1:written+65], weightSum[:])
		written += 65
	}
	finalSum := core.SHA256(buf[:written])
	return core.HexEncode(finalSum[:])
}

// streamHashMinBytes is the shard-size gate above which hashWeightFile
// streams the file through the SHA-256 accumulator rather than reading
// it whole. Set above io.Copy's internal 32KiB scratch-buffer size so a
// file small enough that streaming would allocate more than a whole-file
// read stays on the cheap in-memory path.
const streamHashMinBytes = 128 * 1024

// hashWriter lazily holds the reusable SHA-256 accumulator used by the
// streaming branch of hashWeightFile. Allocated on first streamed file
// so the config-only / small-shard paths never construct one.
type hashWriter struct {
	h interface {
		Write([]byte) (int, error)
		Sum([]byte) []byte
		Reset()
	}
}

// hashWeightFile returns the SHA-256 digest of the weight file at path.
// Files larger than streamHashMinBytes are streamed through the shared
// reusable accumulator in hasher (chunked, flat B/op); smaller files are
// read whole (cheaper than a 32KiB copy buffer at that size). The bool
// is false when the file could not be read, matching the previous
// core.ReadFile !OK skip so the caller does not advance its write
// cursor for an unreadable shard.
func hashWeightFile(path string, hasher *hashWriter) ([32]byte, bool) {
	if stat := core.Stat(path); stat.OK {
		if info, ok := stat.Value.(interface{ Size() int64 }); ok && info.Size() > streamHashMinBytes {
			return streamHashWeightFile(path, hasher)
		}
	}
	read := core.ReadFile(path)
	if !read.OK {
		return [32]byte{}, false
	}
	return core.SHA256(read.Value.([]byte)), true
}

// streamHashWeightFile hashes the file at path by copying it through the
// reusable accumulator in hasher (reset per call). The reader is closed
// on every return path — including the copy-failure path — so a
// multi-shard adapter never holds more than one descriptor open.
func streamHashWeightFile(path string, hasher *hashWriter) ([32]byte, bool) {
	opened := core.Open(path)
	if !opened.OK {
		return [32]byte{}, false
	}
	reader := opened.Value.(core.ReadCloser)
	if hasher.h == nil {
		hasher.h = sha256.New()
	} else {
		hasher.h.Reset()
	}
	copied := core.Copy(hasher.h, reader)
	core.CloseStream(reader)
	if !copied.OK {
		return [32]byte{}, false
	}
	var sum [32]byte
	copy(sum[:], hasher.h.Sum(nil))
	return sum, true
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errResultFailed
}
