// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"context"
	"sync"
)

// fakeLoader records every Load/Unload the Pool drives — it stands in for the
// real go-mlx apply/unload that this package never performs itself. Set loadErr
// / unloadErr to exercise the failure paths.
type fakeLoader struct {
	mu        sync.Mutex
	loaded    []string // ids in load order
	unloaded  []string // ids in unload order
	loads     int
	unloads   int
	loadErr   error
	unloadErr error
}

func (f *fakeLoader) Load(_ context.Context, ref AdapterRef) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.loadErr != nil {
		return f.loadErr
	}
	f.loads++
	f.loaded = append(f.loaded, ref.ID())
	return nil
}

func (f *fakeLoader) Unload(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.unloadErr != nil {
		return f.unloadErr
	}
	f.unloads++
	f.unloaded = append(f.unloaded, id)
	return nil
}

// ref is a tiny helper so the tests read against adapter names, not paths.
func ref(name string) AdapterRef {
	return AdapterRef{Name: name, Path: "/models/" + name, BaseModel: "gemma-e4b"}
}

// errBoom is a sentinel Loader failure for the load-error path.
var errBoom = context.DeadlineExceeded
