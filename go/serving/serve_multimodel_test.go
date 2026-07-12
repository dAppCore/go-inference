// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// mustResolver builds a resolver from specs + opts, failing the test on a
// construction error.
func mustResolver(t *testing.T, specs []ModelSpec, opts MultiModelOptions) *multiModelResolver {
	t.Helper()
	r, err := newMultiModelResolver(specs, opts)
	if err != nil {
		t.Fatalf("newMultiModelResolver: %v", err)
	}
	return r
}

// countingLoader returns a loader that hands each path a fresh *mockTextModel
// (modelType = basename, so the routed model is identifiable) and records how
// many times each path was loaded. The returned models map keeps the latest
// model per path so a test can assert eviction Closed it.
func countingLoader() (loader ModelLoader, loads map[string]int, models map[string]*mockTextModel) {
	var mu sync.Mutex
	loads = map[string]int{}
	models = map[string]*mockTextModel{}
	loader = func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		mu.Lock()
		defer mu.Unlock()
		loads[path]++
		m := &mockTextModel{modelType: core.PathBase(path)}
		models[path] = m
		return m, nil
	}
	return loader, loads, models
}

// TestMultiModelResolver_New_NoModels_Bad proves construction refuses an empty
// registry rather than booting a resolver that can route nothing.
func TestMultiModelResolver_New_NoModels_Bad(t *testing.T) {
	if _, err := newMultiModelResolver(nil, MultiModelOptions{}); err == nil {
		t.Fatal("newMultiModelResolver(nil) should error")
	}
}

// TestMultiModelResolver_New_DuplicateID_Bad proves a colliding id/alias is
// refused at construction — routing must be unambiguous.
func TestMultiModelResolver_New_DuplicateID_Bad(t *testing.T) {
	_, err := newMultiModelResolver([]ModelSpec{
		{ID: "a", Path: "/m/a"},
		{ID: "b", Path: "/m/b", Aliases: []string{"a"}},
	}, MultiModelOptions{})
	if err == nil {
		t.Fatal("duplicate alias 'a' should error at construction")
	}
}

// TestMultiModelResolver_Route_ExactAndAlias_Good proves an exact id and an
// alias both route to the same model, and the load is lazy + once.
func TestMultiModelResolver_Route_ExactAndAlias_Good(t *testing.T) {
	loader, loads, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "qwen3", Path: "/m/qwen3", Aliases: []string{"qwen"}},
		{ID: "bge", Path: "/m/bge"},
	}, MultiModelOptions{})
	r.setLoader(loader)

	byID, err := r.ResolveModel(context.Background(), "qwen3")
	if err != nil {
		t.Fatalf("resolve qwen3: %v", err)
	}
	byAlias, err := r.ResolveModel(context.Background(), "qwen")
	if err != nil {
		t.Fatalf("resolve qwen: %v", err)
	}
	if byID.ModelType() != "qwen3" || byAlias.ModelType() != "qwen3" {
		t.Fatalf("id/alias routed to %q/%q, want both qwen3", byID.ModelType(), byAlias.ModelType())
	}
	if loads["/m/qwen3"] != 1 {
		t.Fatalf("qwen3 loaded %d times, want 1 (lazy once, alias reuses)", loads["/m/qwen3"])
	}
	if loads["/m/bge"] != 0 {
		t.Fatalf("bge loaded %d times, want 0 (never resolved)", loads["/m/bge"])
	}
}

// TestMultiModelResolver_Route_UnknownToDefault_Good proves an unrecognised name
// falls back to the default model (first pinned) — the single-model UX where a
// client echoing an arbitrary name is still served.
func TestMultiModelResolver_Route_UnknownToDefault_Good(t *testing.T) {
	loader, _, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "small", Path: "/m/small"},
		{ID: "main", Path: "/m/main", Pinned: true}, // pinned → default
	}, MultiModelOptions{})
	r.setLoader(loader)

	got, err := r.ResolveModel(context.Background(), "gpt-4-turbo")
	if err != nil {
		t.Fatalf("resolve unknown: %v", err)
	}
	if got.ModelType() != "main" {
		t.Fatalf("unknown name routed to %q, want the pinned default 'main'", got.ModelType())
	}
}

// TestMultiModelResolver_Route_Profile_Good proves `model:profile` resolves the
// model AND applies the preset (temperature reaches the wrapped model), while
// the bare model gets no preset.
func TestMultiModelResolver_Route_Profile_Good(t *testing.T) {
	spy := newPresetSpy()
	r := mustResolver(t, []ModelSpec{
		{ID: "qwen3", Path: "/m/qwen3", Profiles: map[string]ProfileConfig{
			"creative": {Temperature: ptrFloat32(0.9)},
		}},
	}, MultiModelOptions{})
	r.setLoader(func(string, ...inference.LoadOption) (inference.TextModel, error) { return spy, nil })

	profiled, err := r.ResolveModel(context.Background(), "qwen3:creative")
	if err != nil {
		t.Fatalf("resolve qwen3:creative: %v", err)
	}
	drain(profiled.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}))
	if spy.lastCfg.Temperature != 0.9 {
		t.Fatalf("profile temperature = %v, want 0.9", spy.lastCfg.Temperature)
	}

	bare, err := r.ResolveModel(context.Background(), "qwen3")
	if err != nil {
		t.Fatalf("resolve qwen3: %v", err)
	}
	drain(bare.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}))
	if spy.lastCfg.Temperature != 0.0 {
		t.Fatalf("bare model temperature = %v, want 0.0 (no preset)", spy.lastCfg.Temperature)
	}
}

// TestMultiModelResolver_Route_UnknownProfile_Bad proves a profile named against
// a known model that does not define it is an error — a real client mistake the
// resolver must surface rather than silently ignore.
func TestMultiModelResolver_Route_UnknownProfile_Bad(t *testing.T) {
	loader, _, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{{ID: "qwen3", Path: "/m/qwen3"}}, MultiModelOptions{})
	r.setLoader(loader)

	if _, err := r.ResolveModel(context.Background(), "qwen3:nonesuch"); err == nil {
		t.Fatal("resolve qwen3:nonesuch should error (undefined profile)")
	}
}

// TestMultiModelResolver_Evict_LRUOrder_Good pins the LRU victim selection: under
// a ceiling that holds two of three equal-sized models, loading the third evicts
// the least-recently-used, and the next reload evicts the new LRU.
func TestMultiModelResolver_Evict_LRUOrder_Good(t *testing.T) {
	loader, loads, models := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "a", Path: "/m/a", EstBytes: 100},
		{ID: "b", Path: "/m/b", EstBytes: 100},
		{ID: "c", Path: "/m/c", EstBytes: 100},
	}, MultiModelOptions{MemoryCeiling: 250})
	r.setLoader(loader)
	clk := time.Unix(0, 0)
	r.now = func() time.Time { return clk }

	if _, err := r.ResolveModel(context.Background(), "a"); err != nil { // A@t0
		t.Fatal(err)
	}
	clk = clk.Add(time.Second)
	if _, err := r.ResolveModel(context.Background(), "b"); err != nil { // B@t1, resident {A,B}=200
		t.Fatal(err)
	}
	clk = clk.Add(time.Second)
	if _, err := r.ResolveModel(context.Background(), "c"); err != nil { // C needs 100; 300>250 → evict LRU (A@t0)
		t.Fatal(err)
	}
	if !models["/m/a"].closed {
		t.Fatal("A should be evicted+closed as the LRU victim")
	}
	if r.entries["a"].model != nil {
		t.Fatal("A entry should be non-resident after eviction")
	}
	if r.entries["b"].model == nil || r.entries["c"].model == nil {
		t.Fatal("B and C should be resident")
	}

	// Re-resolve A@t3 → evict the new LRU, which is B@t1 (older than C@t2).
	clk = clk.Add(time.Second)
	if _, err := r.ResolveModel(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	if !models["/m/b"].closed {
		t.Fatal("B should be the next LRU victim (older than C)")
	}
	if loads["/m/a"] != 2 {
		t.Fatalf("A loaded %d times, want 2 (reload after eviction)", loads["/m/a"])
	}
}

// TestMultiModelResolver_Evict_PinExemption_Good proves a pinned model is never
// an LRU eviction victim: with a pinned + an unpinned resident, loading a third
// evicts the unpinned one even though the pinned one is older.
func TestMultiModelResolver_Evict_PinExemption_Good(t *testing.T) {
	loader, _, models := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "pinned", Path: "/m/pinned", EstBytes: 100, Pinned: true},
		{ID: "warm", Path: "/m/warm", EstBytes: 100},
		{ID: "cold", Path: "/m/cold", EstBytes: 100},
	}, MultiModelOptions{MemoryCeiling: 250})
	r.setLoader(loader)
	clk := time.Unix(0, 0)
	r.now = func() time.Time { return clk }

	r.ResolveModel(context.Background(), "pinned") // @t0, oldest
	clk = clk.Add(time.Second)
	r.ResolveModel(context.Background(), "warm") // @t1
	clk = clk.Add(time.Second)
	if _, err := r.ResolveModel(context.Background(), "cold"); err != nil { // evict → warm (pinned is exempt despite being oldest)
		t.Fatal(err)
	}
	if models["/m/pinned"].closed {
		t.Fatal("pinned model must never be evicted, even as the oldest")
	}
	if !models["/m/warm"].closed {
		t.Fatal("warm (unpinned) should be the eviction victim")
	}
}

// TestMultiModelResolver_Budget_RefuseTooBig_Bad proves a model larger than the
// whole ceiling is refused rather than loaded over budget.
func TestMultiModelResolver_Budget_RefuseTooBig_Bad(t *testing.T) {
	loader, loads, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{{ID: "big", Path: "/m/big", EstBytes: 200}}, MultiModelOptions{MemoryCeiling: 100})
	r.setLoader(loader)

	if _, err := r.ResolveModel(context.Background(), "big"); err == nil {
		t.Fatal("a model larger than the ceiling should be refused")
	}
	if loads["/m/big"] != 0 {
		t.Fatal("an over-ceiling model must not be loaded")
	}
}

// TestMultiModelResolver_Budget_RefuseAllPinned_Bad proves that when the only
// resident is pinned and there is no room, a new load is refused rather than
// breaking the ceiling or evicting a pinned model.
func TestMultiModelResolver_Budget_RefuseAllPinned_Bad(t *testing.T) {
	loader, loads, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "pinned", Path: "/m/pinned", EstBytes: 60, Pinned: true},
		{ID: "other", Path: "/m/other", EstBytes: 60},
	}, MultiModelOptions{MemoryCeiling: 100})
	r.setLoader(loader)

	r.ResolveModel(context.Background(), "pinned") // resident 60
	if _, err := r.ResolveModel(context.Background(), "other"); err == nil {
		t.Fatal("second model should be refused — 120 > 100 and the resident is pinned")
	}
	if loads["/m/other"] != 0 {
		t.Fatal("the refused model must not be loaded")
	}
}

// TestMultiModelResolver_SweepIdle_TTL_Good proves the idle sweep evicts a model
// past its TTL and spares a recently-used one.
func TestMultiModelResolver_SweepIdle_TTL_Good(t *testing.T) {
	loader, _, models := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "a", Path: "/m/a", EstBytes: 100},
		{ID: "b", Path: "/m/b", EstBytes: 100},
	}, MultiModelOptions{IdleTTL: 10 * time.Minute})
	r.setLoader(loader)
	clk := time.Unix(0, 0)
	r.now = func() time.Time { return clk }

	r.ResolveModel(context.Background(), "a") // A@t0
	r.ResolveModel(context.Background(), "b") // B@t0
	clk = clk.Add(5 * time.Minute)
	r.ResolveModel(context.Background(), "b") // B refreshed @t5

	clk = clk.Add(6 * time.Minute) // now t11: A idle 11m, B idle 6m
	if n := r.sweepIdle(clk); n != 1 {
		t.Fatalf("sweepIdle evicted %d, want 1 (only A past TTL)", n)
	}
	if !models["/m/a"].closed || r.entries["a"].model != nil {
		t.Fatal("A (idle 11m) should be idle-evicted")
	}
	if r.entries["b"].model == nil {
		t.Fatal("B (idle 6m) should survive the TTL sweep")
	}
}

// TestMultiModelResolver_SweepIdle_PinExempt_Good proves the idle sweep never
// evicts a pinned model, however long it sits idle.
func TestMultiModelResolver_SweepIdle_PinExempt_Good(t *testing.T) {
	loader, _, models := countingLoader()
	r := mustResolver(t, []ModelSpec{{ID: "a", Path: "/m/a", EstBytes: 100, Pinned: true}}, MultiModelOptions{IdleTTL: 10 * time.Minute})
	r.setLoader(loader)
	clk := time.Unix(0, 0)
	r.now = func() time.Time { return clk }

	r.ResolveModel(context.Background(), "a")
	clk = clk.Add(time.Hour)
	if n := r.sweepIdle(clk); n != 0 {
		t.Fatalf("sweepIdle evicted %d, want 0 (pinned exempt)", n)
	}
	if models["/m/a"].closed {
		t.Fatal("a pinned model must never be idle-evicted")
	}
}

// TestMultiModelResolver_Unload_OverridesPin_Good proves an explicit unload frees
// a pinned model's memory (unload is an operator action, so it overrides pin),
// and the spec stays registered so a later resolve reloads it.
func TestMultiModelResolver_Unload_OverridesPin_Good(t *testing.T) {
	loader, loads, models := countingLoader()
	r := mustResolver(t, []ModelSpec{{ID: "a", Path: "/m/a", EstBytes: 100, Pinned: true}}, MultiModelOptions{})
	r.setLoader(loader)

	r.ResolveModel(context.Background(), "a")
	if err := r.unloadModel("a"); err != nil {
		t.Fatalf("unload: %v", err)
	}
	if !models["/m/a"].closed || r.entries["a"].model != nil {
		t.Fatal("unload should evict+close even a pinned model")
	}
	if _, err := r.ResolveModel(context.Background(), "a"); err != nil {
		t.Fatalf("re-resolve after unload: %v", err)
	}
	if loads["/m/a"] != 2 {
		t.Fatalf("A loaded %d times, want 2 (reload after unload)", loads["/m/a"])
	}
}

// TestMultiModelResolver_SetPinned_Good proves pinning an unpinned model then
// exempts it from LRU eviction.
func TestMultiModelResolver_SetPinned_Good(t *testing.T) {
	loader, _, models := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "a", Path: "/m/a", EstBytes: 100},
		{ID: "b", Path: "/m/b", EstBytes: 100},
	}, MultiModelOptions{MemoryCeiling: 150})
	r.setLoader(loader)

	r.ResolveModel(context.Background(), "a")
	if err := r.setPinned("a", true); err != nil {
		t.Fatalf("setPinned: %v", err)
	}
	// B needs room; A is now pinned so B cannot fit (100+100 > 150, no unpinned
	// victim) → refused.
	if _, err := r.ResolveModel(context.Background(), "b"); err == nil {
		t.Fatal("after pinning A, B should be refused (A is now exempt)")
	}
	if models["/m/a"].closed {
		t.Fatal("A must not be evicted once pinned")
	}
}

// TestMultiModelResolver_ListedModelIDs_Good proves /v1/models enumeration lists
// every model plus its profile combinations in a stable order.
func TestMultiModelResolver_ListedModelIDs_Good(t *testing.T) {
	r := mustResolver(t, []ModelSpec{
		{ID: "qwen3", Path: "/m/qwen3", Profiles: map[string]ProfileConfig{
			"creative": {Temperature: ptrFloat32(0.9)},
			"precise":  {Temperature: ptrFloat32(0.0)},
		}},
		{ID: "bge", Path: "/m/bge"},
	}, MultiModelOptions{})

	got := r.listedModelIDs()
	want := []string{"qwen3", "qwen3:creative", "qwen3:precise", "bge"}
	if len(got) != len(want) {
		t.Fatalf("listedModelIDs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("listedModelIDs[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

// TestMultiModelResolver_List_ResidencySnapshot_Good proves list() reports
// residency + pin + est bytes accurately after a resolve.
func TestMultiModelResolver_List_ResidencySnapshot_Good(t *testing.T) {
	loader, _, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "a", Path: "/m/a", EstBytes: 100, Pinned: true},
		{ID: "b", Path: "/m/b", EstBytes: 200},
	}, MultiModelOptions{})
	r.setLoader(loader)
	r.ResolveModel(context.Background(), "a")

	statuses := r.list()
	if len(statuses) != 2 {
		t.Fatalf("list len = %d, want 2", len(statuses))
	}
	if !statuses[0].Resident || !statuses[0].Pinned || statuses[0].EstBytes != 100 {
		t.Fatalf("a status = %+v, want resident+pinned+100", statuses[0])
	}
	if statuses[1].Resident || statuses[1].EstBytes != 200 {
		t.Fatalf("b status = %+v, want non-resident+200", statuses[1])
	}
}

// TestMultiModelResolver_ReloadModel_MakesDefault_Good proves the legacy reload
// verb loads a new model, pins it, and makes it the default.
func TestMultiModelResolver_ReloadModel_MakesDefault_Good(t *testing.T) {
	loader, _, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{{ID: "a", Path: "/m/a"}}, MultiModelOptions{})
	r.setLoader(loader)

	prev, active, err := r.ReloadModel("/m/z", nil)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if prev != "/m/a" || active != "/m/z" {
		t.Fatalf("reload returned prev=%q active=%q, want /m/a and /m/z", prev, active)
	}
	if r.CurrentPath() != "/m/z" {
		t.Fatalf("CurrentPath = %q, want /m/z (new default)", r.CurrentPath())
	}
	got, err := r.ResolveModel(context.Background(), "unknown-name")
	if err != nil {
		t.Fatalf("resolve unknown after reload: %v", err)
	}
	if got.ModelType() != "z" {
		t.Fatalf("unknown routed to %q, want the new default z", got.ModelType())
	}
}

// TestMultiModelResolver_ConcurrentResolve_Race hammers ResolveModel + the
// control-plane operations from many goroutines to prove the single-mutex design
// is race-free (run under -race). Correctness here is "no panic, no race, every
// resolve returns a usable model".
func TestMultiModelResolver_ConcurrentResolve_Race(t *testing.T) {
	loader, _, _ := countingLoader()
	r := mustResolver(t, []ModelSpec{
		{ID: "a", Path: "/m/a", EstBytes: 100},
		{ID: "b", Path: "/m/b", EstBytes: 100},
		{ID: "c", Path: "/m/c", EstBytes: 100, Profiles: map[string]ProfileConfig{"fast": {MaxTokens: ptrInt(16)}}},
	}, MultiModelOptions{MemoryCeiling: 250, IdleTTL: time.Millisecond})
	r.setLoader(loader)

	names := []string{"a", "b", "c", "c:fast", "unknown"}
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			name := names[i%len(names)]
			if m, err := r.ResolveModel(context.Background(), name); err != nil || m == nil {
				// A budget refusal is a valid outcome under contention; only a
				// nil model with no error is a bug.
				if err == nil {
					t.Errorf("resolve %q returned nil model, no error", name)
				}
			}
			switch i % 7 {
			case 0:
				r.sweepIdle(r.now())
			case 1:
				_ = r.list()
			case 2:
				_ = r.setPinned("a", i%2 == 0)
			case 3:
				_ = r.unloadModel("b")
			case 4:
				_ = r.listedModelIDs()
			}
		}(i)
	}
	wg.Wait()
}

// TestEstimateModelBytes_PackSize_Good proves the disk sizer sums regular-file
// bytes across the model tree (including a subdirectory), the residency-budget
// proxy.
func TestEstimateModelBytes_PackSize_Good(t *testing.T) {
	dir := t.TempDir()
	writeFile := func(rel string, n int) {
		p := core.PathJoin(dir, rel)
		if d := core.PathDir(p); d != "" {
			if res := core.MkdirAll(d, 0o755); !res.OK {
				t.Fatalf("mkdir %s: %v", d, res.Value)
			}
		}
		if res := core.WriteFile(p, make([]byte, n), 0o644); !res.OK {
			t.Fatalf("write %s: %v", p, res.Value)
		}
	}
	writeFile("weights.bin", 1000)
	writeFile("config.json", 24)
	writeFile("assistant/draft.bin", 500)

	got := estimateModelBytes(dir)
	if got != 1524 {
		t.Fatalf("estimateModelBytes = %d, want 1524 (1000+24+500)", got)
	}
}

// TestEstimateModelBytes_MissingPath_Good proves an unreadable path sizes to 0 —
// the budget cannot gate what it cannot measure, so a load is never falsely
// refused.
func TestEstimateModelBytes_MissingPath_Good(t *testing.T) {
	if got := estimateModelBytes("/no/such/model/dir/anywhere"); got != 0 {
		t.Fatalf("estimateModelBytes(missing) = %d, want 0", got)
	}
}
