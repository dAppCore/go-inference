// SPDX-Licence-Identifier: EUPL-1.2

// Package lora is the adapter-level multi-LoRA serving pool for the inference stack. One base
// model (held resident by the model-level pkg/residency policy) serves many LoRA
// adapters at once: each request selects an adapter by name, the Pool loads it on
// demand via a go-mlx Loader, keeps a bounded set resident, and evicts the
// least-recently-used adapter that is neither in-flight nor pinned when it hits
// capacity.
//
// Where pkg/residency reasons over MODELS and byte budgets — which whole models
// fit a 16 GB GPU / 96 GB M3 Ultra — this package reasons over ADAPTERS and a
// count cap: adapters are small (LoRA deltas), so the binding constraint is how
// many can be applied to the live base model at once, not their bytes. The two
// compose: residency keeps the base model loaded, this pool swaps adapters on top
// of it. Neither package touches a device; the caller injects the real go-mlx
// apply/unload behind the Loader interface and this package only decides what to
// load, what to evict, and which adapter is safe to evict.
//
//	pool := lora.NewPool(lora.Config{
//		Loader:   mlxLoader,                  // real go-mlx apply/unload
//		Policy:   lora.NewLRUEvictionPolicy(),
//		Capacity: 8,                          // max adapters resident at once
//	})
//	pool.Register(lora.AdapterRef{Name: "support-tone", Path: "/adapters/support", BaseModel: "gemma-e4b"})
//	id, release, err := pool.Use(ctx, "support-tone") // load-on-demand, ref-counted
//	if err != nil { return err }
//	defer release()                                    // drop the in-flight ref
//	// … run inference on the base model with adapter `id` applied …
//
// Ref-counting guarantees an adapter serving an in-flight request is never
// evicted: Use takes a ref, the returned release drops it, and only adapters with
// a zero ref-count (and not pinned) are eviction candidates.
package lora

import core "dappco.re/go"

// AdapterRef identifies one LoRA adapter: a human Name (the request-side selector
// and registry key), the Path the Loader applies from, and the BaseModel the
// adapter was trained against. The triple yields a stable ID — see ID.
//
//	r := lora.AdapterRef{Name: "support-tone", Path: "/adapters/support", BaseModel: "gemma-e4b"}
type AdapterRef struct {
	Name      string
	Path      string
	BaseModel string
}

// ID is the deterministic adapter id derived from Name and Path. Like SGLang's
// LoRARef.deterministic_id, it is stable across processes and machines for the
// same Name+Path so every node minting refs from the same --adapter-paths agrees
// on the id (a uuid4-style random id would diverge per process). The id is a
// content hash, so a re-pathed adapter of the same name is a distinct id.
//
//	lora.AdapterRef{Name: "a", Path: "/x"}.ID() // stable for ("a","/x")
func (r AdapterRef) ID() string {
	return core.SHA256HexString(deterministicSeed(r.Name, r.Path))
}

// deterministicSeed joins name and path with a NUL so ("ab","c") and ("a","bc")
// never collide. Caller-free helper, used only by ID.
func deterministicSeed(name, path string) string {
	return name + "\x00" + path
}
