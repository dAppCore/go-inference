// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for inspect.go — kept separate from inspect_test.go so
// the godoc-attached usage snippets stay readable. The Inspect examples
// build a synthetic adapter under a temp directory and print only the
// content-stable identity fields (Name / Rank / Alpha / Scale / sorted
// TargetKeys). They deliberately never print AdapterInfo.Hash: it is derived
// from the weight bytes and the temp path, so it has no deterministic value
// to assert in an // Output: block.

package lora

import (
	"slices"

	core "dappco.re/go"
)

// ExampleAdapterInfo_IsEmpty shows the zero-value adapter identity reporting
// itself empty — the state callers check before treating an adapter as
// attached.
func ExampleAdapterInfo_IsEmpty() {
	var info AdapterInfo
	core.Println(info.IsEmpty())

	info.Name = "my-lora"
	core.Println(info.IsEmpty())

	// Output:
	// true
	// false
}

// ExampleAdapterInfo_Identity projects an inspected adapter into the
// portable state.AdapterIdentity shape a capability.AdapterModel or
// state.CheckWakeCompatibility caller consumes. Name and Scale do not
// appear on the far side — Identity is an identity projection, not a
// lossless mirror.
func ExampleAdapterInfo_Identity() {
	info := AdapterInfo{
		Name:       "my-lora",
		Path:       "/models/my-lora",
		Rank:       16,
		Alpha:      32,
		Scale:      2,
		TargetKeys: []string{"self_attn.q_proj"},
	}
	id := info.Identity()
	core.Println("path:", id.Path)
	core.Println("rank:", id.Rank)
	core.Println("alpha:", id.Alpha)
	core.Println("targets:", id.TargetKeys)

	// Output:
	// path: /models/my-lora
	// rank: 16
	// alpha: 32
	// targets: [self_attn.q_proj]
}

// ExampleInspectAdapter reads adapter_config.json from an adapter directory
// and reports the LoRA identity. Here the config uses the canonical rank /
// alpha fields and lists its targets under lora_layers (the mlx-lm spelling);
// Inspect derives the scale (alpha / rank) and surfaces the targets verbatim.
func ExampleInspectAdapter() {
	dir := core.MkdirTemp("", "lora-inspect-example-*").Value.(string)
	defer core.RemoveAll(dir)
	core.WriteFile(core.PathJoin(dir, "adapter_config.json"),
		[]byte(`{"rank":16,"alpha":32,"lora_layers":["self_attn.q_proj","self_attn.v_proj"]}`), 0o600)
	core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("synthetic-weights"), 0o600)

	info, err := InspectAdapter(dir)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("rank:", info.Rank)
	core.Println("alpha:", info.Alpha)
	core.Println("scale:", info.Scale)
	core.Println("targets:", info.TargetKeys)

	// Output:
	// rank: 16
	// alpha: 32
	// scale: 2
	// targets: [self_attn.q_proj self_attn.v_proj]
}

// ExampleInspect shows the two-argument identity split: the adapter is
// staged under a temp path, but the reported identity uses the caller's
// original path — the shape a Medium-staged adapter needs so its reported
// Name/Path survive the staging round-trip.
func ExampleInspect() {
	stagedDir := core.MkdirTemp("", "lora-inspect-staged-*").Value.(string)
	defer core.RemoveAll(stagedDir)
	core.WriteFile(core.PathJoin(stagedDir, "adapter_config.json"),
		[]byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600)
	core.WriteFile(core.PathJoin(stagedDir, "adapter.safetensors"), []byte("synthetic-weights"), 0o600)

	info, err := Inspect(stagedDir, "/adapters/original/support-tone")
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("name:", info.Name)
	core.Println("path:", info.Path)
	core.Println("rank:", info.Rank)

	// Output:
	// name: support-tone
	// path: /adapters/original/support-tone
	// rank: 8
}

// ExampleInspectAdapter_aliases shows the metadata-alias normalisation
// Inspect applies: a PEFT-style config (r / lora_alpha / target_modules) is
// read into the same canonical AdapterInfo as the mlx-lm spelling, so a
// single downstream consumer reads any adapter flavour uniformly.
func ExampleInspectAdapter_aliases() {
	dir := core.MkdirTemp("", "lora-inspect-aliases-example-*").Value.(string)
	defer core.RemoveAll(dir)
	core.WriteFile(core.PathJoin(dir, "adapter_config.json"),
		[]byte(`{"r":8,"lora_alpha":16,"target_modules":["v_proj","q_proj"]}`), 0o600)
	core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("synthetic-weights"), 0o600)

	info, err := InspectAdapter(dir)
	if err != nil {
		core.Println("error:", err)
		return
	}
	// target_modules preserves source order; sort for a stable example line.
	slices.Sort(info.TargetKeys)
	core.Println("rank:", info.Rank)
	core.Println("alpha:", info.Alpha)
	core.Println("targets:", info.TargetKeys)

	// Output:
	// rank: 8
	// alpha: 16
	// targets: [q_proj v_proj]
}
