// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
)

func ExampleStateSession_WakeState() {
	store := state.NewInMemoryStore(nil)
	cache, _ := newROCmKVCache(rocmKVCacheModeQ8, 2)
	_ = cache.AppendVectors(0, 1, 1, []float32{1, 2}, []float32{2, 1})
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)
	_, _ = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store, EntryURI: "state://entry"})
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	wake, _ := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry"})
	core.Println(wake.PrefixTokens)
	// Output: 2
}

func ExampleStateSession_SleepState() {
	store := state.NewInMemoryStore(nil)
	cache, _ := newROCmKVCache(rocmKVCacheModeQ8, 2)
	_ = cache.AppendVectors(0, 1, 1, []float32{1, 2}, []float32{2, 1})
	session := newStateSessionWithRuntime(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, nil, cache)

	sleep, _ := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/sleep",
		Title:    "sleep",
	})
	core.Println(sleep.Entry.URI)
	core.Println(sleep.Labels["kv_serialize"])
	// Output:
	// state://entry/sleep
	// runtime_owned_blocks
}

func ExampleStateSession_SleepState_kvSnapshot() {
	store := state.NewInMemoryStore(nil)
	cache, _ := newROCmKVCache(rocmKVCacheModeQ8, 2)
	_ = cache.Append(0, []float32{1, 2, 3}, []float32{3, 2, 1})
	session := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)

	sleep, _ := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Encoding: rocmKVSnapshotEncoding,
	})
	core.Println(sleep.Encoding)
	core.Println(sleep.Labels["kv_serialize"])
	// Output:
	// rocm/kv-cache+json
	// runtime_owned
}

func ExampleStateSession_Close() {
	cache, _ := newROCmKVCache(rocmKVCacheModeQ8, 2)
	_ = cache.AppendVectors(0, 1, 1, []float32{1, 2}, []float32{3, 4})
	driver := &fakeHIPDriver{available: true}
	device, _ := cache.MirrorToDevice(driver)
	session := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, device)

	_ = session.Close()
	core.Println(device.closed)
	core.Println(len(driver.frees) == len(driver.allocations))
	// Output:
	// true
	// true
}

func ExampleStateSession_ForkState() {
	store := state.NewInMemoryStore(nil)
	cache, _ := newROCmKVCache(rocmKVCacheModeQ8, 2)
	_ = cache.AppendVectors(0, 1, 1, []float32{1, 2}, []float32{2, 1})
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)
	_, _ = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store, EntryURI: "state://entry"})
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	forked, wake, _ := session.ForkState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry"})
	core.Println(wake.PrefixTokens)
	core.Println(forked != session)
	// Output:
	// 2
	// true
}

func Example_rocmModel_ForkState() {
	store := state.NewInMemoryStore(nil)
	cache, _ := newROCmKVCache(rocmKVCacheModeQ8, 2)
	_ = cache.AppendVectors(0, 1, 1, []float32{1, 2}, []float32{3, 4})
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, _ = session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/fork-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: &fakeHIPDriver{available: true}},
	}

	forked, wake, _ := model.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/fork-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})
	forkedSession := forked.(*StateSession)
	device, remirrored := forkedSession.runtime.(*rocmDeviceKVCache)
	if remirrored {
		defer device.Close()
	}
	core.Println(wake.Labels["kv_restore"])
	core.Println(wake.Labels["kv_device_restore"])
	core.Println(remirrored)
	// Output:
	// device_mirror
	// mirrored
	// true
}

func Example_rocmModel_CaptureState() {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{},
	}

	bundle, _ := model.CaptureState(context.Background(), "hello world", inference.WithMaxTokens(8))
	core.Println(bundle.Version)
	core.Println(bundle.Labels["state_bundle"])
	// Output:
	// rocm-state-bundle-v1
	// metadata_only
}

func Example_rocmModel_RestoreState() {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	_ = model.RestoreState(context.Background(), &inference.StateBundle{
		Model:  inference.ModelIdentity{Architecture: "qwen3"},
		Labels: map[string]string{"tenant": "a"},
	})

	core.Println(model.state.labels["kv_restore"])
	core.Println(model.state.labels["tenant"])
	// Output:
	// metadata_only
	// a
}
