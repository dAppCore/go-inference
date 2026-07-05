// SPDX-Licence-Identifier: EUPL-1.2

// Tests for inspect.go — AdapterInfo metadata/hash extraction.

package lora

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestAdapterInfo_IsEmpty_Good(t *testing.T) {
	// Good: the zero-value AdapterInfo (no inference adapter attached) is
	// the canonical "empty" case IsEmpty exists to recognise.
	var info AdapterInfo
	if !info.IsEmpty() {
		t.Fatalf("AdapterInfo{}.IsEmpty() = false, want true for zero value")
	}
}

func TestAdapterInfo_IsEmpty_Bad(t *testing.T) {
	// Bad (for the empty predicate): a fully populated adapter identity is
	// emphatically not empty.
	info := AdapterInfo{
		Name:       "my-lora",
		Path:       "/models/my-lora",
		Hash:       "deadbeef",
		Rank:       16,
		Alpha:      32,
		Scale:      2,
		TargetKeys: []string{"self_attn.q_proj"},
	}
	if info.IsEmpty() {
		t.Fatalf("populated AdapterInfo.IsEmpty() = true, want false: %+v", info)
	}
}

func TestAdapterInfo_IsEmpty_Ugly(t *testing.T) {
	// Ugly: prove that setting ANY single field on an otherwise-zero
	// AdapterInfo flips IsEmpty to false. This is the assertion that earns
	// its keep — it catches a field being dropped from the AND chain. One
	// mutator per field IsEmpty inspects.
	cases := []struct {
		name string
		set  func(*AdapterInfo)
	}{
		{"Name", func(a *AdapterInfo) { a.Name = "x" }},
		{"Path", func(a *AdapterInfo) { a.Path = "/x" }},
		{"Hash", func(a *AdapterInfo) { a.Hash = "abc" }},
		{"Rank", func(a *AdapterInfo) { a.Rank = 1 }},
		{"Alpha", func(a *AdapterInfo) { a.Alpha = 0.5 }},
		{"Scale", func(a *AdapterInfo) { a.Scale = 0.5 }},
		{"TargetKeys", func(a *AdapterInfo) { a.TargetKeys = []string{"q_proj"} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var info AdapterInfo
			tc.set(&info)
			if info.IsEmpty() {
				t.Fatalf("AdapterInfo with only %s set reported IsEmpty() = true, want false: %+v", tc.name, info)
			}
		})
	}
}

func TestAdapterInfo_Identity_Good(t *testing.T) {
	// Good: Path/Hash/Rank/Alpha/TargetKeys carry straight across; Name and
	// Scale are the deliberate identity-projection exclusions (see the
	// Identity doc comment) and must not leak into the result.
	info := AdapterInfo{
		Name:       "my-lora",
		Path:       "/models/my-lora",
		Hash:       "deadbeef",
		Rank:       16,
		Alpha:      32,
		Scale:      2,
		TargetKeys: []string{"self_attn.q_proj", "self_attn.v_proj"},
	}
	want := state.AdapterIdentity{
		Path:       "/models/my-lora",
		Hash:       "deadbeef",
		Rank:       16,
		Alpha:      32,
		TargetKeys: []string{"self_attn.q_proj", "self_attn.v_proj"},
	}
	got := info.Identity()
	if got.Path != want.Path || got.Hash != want.Hash || got.Rank != want.Rank || got.Alpha != want.Alpha {
		t.Fatalf("AdapterInfo.Identity() = %+v, want %+v", got, want)
	}
	if !equalStringSlices(got.TargetKeys, want.TargetKeys) {
		t.Fatalf("AdapterInfo.Identity().TargetKeys = %v, want %v", got.TargetKeys, want.TargetKeys)
	}
	if got.Format != "" || got.BaseModelHash != "" || len(got.Labels) != 0 {
		t.Fatalf("AdapterInfo.Identity() populated a field Identity never sets: %+v", got)
	}
}

func TestAdapterInfo_Identity_Bad(t *testing.T) {
	// Bad (degenerate input): the zero-value AdapterInfo projects to the
	// zero-value AdapterIdentity — no field invents content from nothing.
	// AdapterIdentity holds a slice and a map, so it is compared field by
	// field rather than with == / !=.
	var info AdapterInfo
	got := info.Identity()
	if got.Path != "" || got.Hash != "" || got.Rank != 0 || got.Alpha != 0 ||
		got.Format != "" || got.BaseModelHash != "" || len(got.TargetKeys) != 0 || len(got.Labels) != 0 {
		t.Fatalf("AdapterInfo{}.Identity() = %+v, want zero value", got)
	}
}

func TestAdapterInfo_Identity_Ugly(t *testing.T) {
	// Ugly: the returned TargetKeys is a defensive clone — mutating it must
	// not reach back into the source AdapterInfo's backing array.
	info := AdapterInfo{TargetKeys: []string{"q_proj"}}
	got := info.Identity()
	got.TargetKeys[0] = "mutated"
	if info.TargetKeys[0] != "q_proj" {
		t.Fatalf("AdapterInfo.Identity().TargetKeys aliases the source slice: source = %v", info.TargetKeys)
	}
}

func TestInspectAdapter_LargeShardStreamingHash_Good(t *testing.T) {
	// Drives InspectAdapter through the large-shard streaming hash path:
	// streamHashWeightFile only fires for weight files larger than
	// streamHashMinBytes (128 KiB), so a synthetic shard above that gate
	// exercises the streaming accumulator that the small stub fixtures
	// never reach. Asserts the hash is deterministic for identical content
	// and changes when a single byte changes — proving the streamed bytes
	// actually feed the digest.
	const shardSize = streamHashMinBytes + 64*1024 // ~192 KiB, well over the 128 KiB gate

	makeAdapter := func(t *testing.T, fillByte byte) AdapterInfo {
		t.Helper()
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter_config: %s", result.Error())
		}
		weights := make([]byte, shardSize)
		for i := range weights {
			weights[i] = fillByte
		}
		if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), weights, 0o600); !result.OK {
			t.Fatalf("WriteFile large shard: %s", result.Error())
		}
		info, err := InspectAdapter(dir)
		if err != nil {
			t.Fatalf("InspectAdapter(large shard) error = %v", err)
		}
		if info.Hash == "" {
			t.Fatalf("InspectAdapter(large shard) produced empty hash: %+v", info)
		}
		return info
	}

	first := makeAdapter(t, 0xAB)
	repeat := makeAdapter(t, 0xAB)
	if first.Hash != repeat.Hash {
		t.Fatalf("streaming hash not deterministic: %q != %q", first.Hash, repeat.Hash)
	}

	different := makeAdapter(t, 0xCD)
	if first.Hash == different.Hash {
		t.Fatalf("streaming hash collided across distinct shard content: %q", first.Hash)
	}
}

func TestInspectAdapter_ReadsMetadataAndHashes_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"rank":16,"alpha":32,"lora_layers":["self_attn.q_proj","self_attn.v_proj"]}`)

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter() error = %v", err)
	}
	if info.Name != core.PathBase(dir) || info.Path != dir {
		t.Fatalf("adapter identity = %+v, want name/path", info)
	}
	if info.Rank != 16 || info.Alpha != 32 || info.Hash == "" {
		t.Fatalf("adapter metadata = %+v, want rank/alpha/hash", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"self_attn.q_proj", "self_attn.v_proj"}) {
		t.Fatalf("adapter targets = %v, want q/v", info.TargetKeys)
	}
}

func TestInspectAdapter_MissingConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("stub"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	_, err := InspectAdapter(dir)
	if err == nil {
		t.Fatal("expected missing adapter_config.json error")
	}
}

func TestInspectAdapter_SafetensorsPath_Ugly(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"r":4,"lora_alpha":8,"target_modules":["q_proj"]}`)
	path := core.PathJoin(dir, "adapter.safetensors")

	info, err := InspectAdapter(path)
	if err != nil {
		t.Fatalf("InspectAdapter(.safetensors) error = %v", err)
	}
	if info.Path != path || info.Name != "adapter.safetensors" || info.Rank != 4 || info.Alpha != 8 {
		t.Fatalf("adapter info = %+v, want safetensors path metadata", info)
	}
}

func TestInspectAdapter_UsesSharedConfigPrecedence_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{
		"rank": 4,
		"scale": 2,
		"target_keys": ["explicit"],
		"target_modules": ["peft"],
		"lora_layers": ["mlx-lm"]
	}`)

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter() error = %v", err)
	}
	if info.Rank != 4 || info.Alpha != 8 || info.Scale != 2 {
		t.Fatalf("adapter metadata = %+v, want scale-derived alpha", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"explicit"}) {
		t.Fatalf("adapter targets = %v, want shared explicit target_keys precedence", info.TargetKeys)
	}
}

func TestInspectAdapter_PreservesMissingRank_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"target_modules":["q_proj"]}`)

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter() error = %v", err)
	}
	if info.Rank != 0 || info.Alpha != 0 || info.Scale != 0 {
		t.Fatalf("adapter metadata = %+v, want missing rank/alpha/scale preserved", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("adapter targets = %v, want target_modules alias", info.TargetKeys)
	}
}

func TestInspectAdapter_MalformedConfig_Bad(t *testing.T) {
	// Bad: adapter_config.json exists and reads OK, but the bytes are not
	// valid JSON. ParseAdapterConfig fails, so Inspect must surface a parse
	// error (the wrap in Inspect) rather than silently returning a zero
	// AdapterInfo. This is reachable without fault injection — a corrupt
	// config on disk is an ordinary user-facing failure, not a simulated IO
	// fault.
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte("{not valid json"), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter_config: %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("stub"), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter.safetensors: %s", result.Error())
	}

	_, err := InspectAdapter(dir)
	if err == nil {
		t.Fatal("InspectAdapter(malformed config) error = nil, want parse error")
	}
	if !core.Contains(err.Error(), "parse adapter_config.json") {
		t.Fatalf("error = %v, want parse-config context", err)
	}
}

func TestInspectAdapter_EmptyPath_Bad(t *testing.T) {
	// Bad: an empty adapter path is the guard at the top of Inspect — it
	// returns the shared errAdapterPathRequired sentinel before touching the
	// filesystem. InspectAdapter forwards path as both arguments, so the
	// public entry point exercises the same guard.
	if _, err := InspectAdapter(""); err != errAdapterPathRequired {
		t.Fatalf("InspectAdapter(\"\") error = %v, want errAdapterPathRequired", err)
	}
	if _, err := Inspect("", "/some/identity"); err != errAdapterPathRequired {
		t.Fatalf("Inspect(\"\", identity) error = %v, want errAdapterPathRequired", err)
	}
}

func TestInspectAdapter_UnreadableShardSkipped_Ugly(t *testing.T) {
	// Ugly: hashAdapter globs *.safetensors and hashes each match, skipping
	// any entry it cannot read (the !ok branch in hashWeightFile). A
	// *directory* whose name ends in .safetensors is matched by the glob
	// but fails ReadFile — so the skip fires without any permission games.
	// The resulting hash must still be deterministic and must equal the hash
	// of an adapter that never had the unreadable entry, proving the skipped
	// directory contributed nothing to the digest.
	makeAdapter := func(t *testing.T, withDir bool) AdapterInfo {
		t.Helper()
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter_config: %s", result.Error())
		}
		if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("real-weights"), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter.safetensors: %s", result.Error())
		}
		if withDir {
			// A directory matched by the *.safetensors glob — ReadFile on a
			// directory fails, so hashWeightFile returns ok=false and the
			// cursor does not advance for it.
			if result := core.MkdirAll(core.PathJoin(dir, "extra.safetensors"), 0o755); !result.OK {
				t.Fatalf("MkdirAll extra.safetensors: %s", result.Error())
			}
		}
		info, err := InspectAdapter(dir)
		if err != nil {
			t.Fatalf("InspectAdapter error = %v", err)
		}
		if info.Hash == "" {
			t.Fatalf("InspectAdapter produced empty hash: %+v", info)
		}
		return info
	}

	withSkip := makeAdapter(t, true)
	withoutSkip := makeAdapter(t, false)
	if withSkip.Hash != withoutSkip.Hash {
		t.Fatalf("unreadable .safetensors directory altered the digest: %q != %q", withSkip.Hash, withoutSkip.Hash)
	}
}

func TestAdapterConfigPath_DelegatesToPrecomputed_Good(t *testing.T) {
	// Good: adapterConfigPath is the convenience wrapper that computes the
	// .safetensors suffix check once and delegates to the precomputed
	// variant. Assert the delegation contract directly (not a bare call):
	// the wrapper must produce exactly what the precomputed form produces for
	// the same suffix classification, for both a directory and a weight-file
	// path.
	cases := []struct {
		name string
		path string
		want string
	}{
		{"dir", "/models/my-lora", "/models/my-lora/adapter_config.json"},
		{"safetensors", "/models/my-lora/adapter.safetensors", "/models/my-lora/adapter_config.json"},
		{"trailingSlash", "/models/my-lora/", "/models/my-lora/adapter_config.json"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := adapterConfigPath(tc.path)
			if got != tc.want {
				t.Fatalf("adapterConfigPath(%q) = %q, want %q", tc.path, got, tc.want)
			}
			precomputed := adapterConfigPathPrecomputed(tc.path, core.HasSuffix(tc.path, ".safetensors"))
			if got != precomputed {
				t.Fatalf("adapterConfigPath(%q) = %q, diverged from precomputed %q", tc.path, got, precomputed)
			}
		})
	}
}

func TestHashAdapter_DelegatesToPrecomputed_Good(t *testing.T) {
	// Good: hashAdapter is the convenience wrapper over hashAdapterPrecomputed.
	// Assert the delegation produces a byte-identical digest to the precomputed
	// form (same suffix classification) for both a directory adapter and a
	// direct .safetensors path — documents that the wrapper changes nothing but
	// the suffix-scan bookkeeping.
	dir := writeTestLoRAAdapter(t, `{"rank":4,"alpha":8,"target_modules":["q_proj"]}`)
	config := []byte(`{"rank":4,"alpha":8,"target_modules":["q_proj"]}`)

	dirHash := hashAdapter(dir, config)
	dirPrecomputed := hashAdapterPrecomputed(dir, config, false)
	if dirHash != dirPrecomputed || dirHash == "" {
		t.Fatalf("hashAdapter(dir) = %q, want non-empty == precomputed %q", dirHash, dirPrecomputed)
	}

	weightPath := core.PathJoin(dir, "adapter.safetensors")
	fileHash := hashAdapter(weightPath, config)
	filePrecomputed := hashAdapterPrecomputed(weightPath, config, true)
	if fileHash != filePrecomputed || fileHash == "" {
		t.Fatalf("hashAdapter(.safetensors) = %q, want non-empty == precomputed %q", fileHash, filePrecomputed)
	}
}

func TestJoinDirChildPattern_Branches_Ugly(t *testing.T) {
	// Ugly: joinDirChildPattern is the filepath.Clean-skipping join shared by
	// hashAdapterPrecomputed. It has three branches the hot tests never hit
	// individually — empty dir (relative passthrough), dir already ending in
	// '/' (collapse the duplicate separator), and the plain insert-separator
	// case. One assertion per branch.
	cases := []struct {
		name  string
		dir   string
		child string
		want  string
	}{
		{"emptyDir", "", "*.safetensors", "*.safetensors"},
		{"trailingSlash", "/models/lora/", "*.safetensors", "/models/lora/*.safetensors"},
		{"plainJoin", "/models/lora", "*.safetensors", "/models/lora/*.safetensors"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := joinDirChildPattern(tc.dir, tc.child); got != tc.want {
				t.Fatalf("joinDirChildPattern(%q, %q) = %q, want %q", tc.dir, tc.child, got, tc.want)
			}
		})
	}
}

func TestAdapterConfigPathPrecomputed_TrailingSlashCollapse_Ugly(t *testing.T) {
	// Ugly: the trailing-slash collapse branch — when the dir base already
	// ends in '/', the leading '/' of the suffix is dropped so the result
	// never contains "//adapter_config.json". Pair it with the non-slash
	// base to prove both arms produce the canonical single separator.
	if got := adapterConfigPathPrecomputed("/models/lora/", false); got != "/models/lora/adapter_config.json" {
		t.Fatalf("trailing-slash base = %q, want collapsed single separator", got)
	}
	if got := adapterConfigPathPrecomputed("/models/lora", false); got != "/models/lora/adapter_config.json" {
		t.Fatalf("plain base = %q, want single separator", got)
	}
	// Safetensors path: PathDir strips the weight file, then the parent dir
	// gets the suffix — exercises the isSafetensors arm.
	if got := adapterConfigPathPrecomputed("/models/lora/adapter.safetensors", true); got != "/models/lora/adapter_config.json" {
		t.Fatalf("safetensors base = %q, want parent-dir config path", got)
	}
}

func TestResultError_Branches_GoodBadUgly(t *testing.T) {
	// resultError unwraps a core.Result into a plain error for core.E
	// chaining. Three branches, all reachable with synthetic Results — no
	// fault injection needed:
	//   Good: an OK result has no error.
	//   Bad:  a failed result carrying an error returns that error.
	//   Ugly: a failed result whose Value is NOT an error falls back to the
	//         errResultFailed sentinel (the defensive path).
	if err := resultError(core.Ok([]byte("data"))); err != nil {
		t.Fatalf("resultError(Ok) = %v, want nil", err)
	}
	sentinel := core.NewError("boom")
	if err := resultError(core.Fail(sentinel)); err != sentinel {
		t.Fatalf("resultError(Fail(err)) = %v, want the wrapped error", err)
	}
	nonError := core.Result{Value: "a bare string, not an error", OK: false}
	if err := resultError(nonError); err != errResultFailed {
		t.Fatalf("resultError(non-error failure) = %v, want errResultFailed fallback", err)
	}
}

func writeTestLoRAAdapter(t *testing.T, config string) string {
	t.Helper()
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(config), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter_config: %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("stub-weights"), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter.safetensors: %s", result.Error())
	}
	return dir
}
