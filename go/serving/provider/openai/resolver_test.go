// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestResolver_ResolverFunc_Good(t *testing.T) {
	model := &stubModel{}
	fn := ResolverFunc(func(ctx context.Context, name string) (inference.TextModel, error) {
		if name != "qwen" {
			t.Fatalf("name = %q, want qwen", name)
		}
		return model, nil
	})

	got, err := fn.ResolveModel(context.Background(), "qwen")
	if err != nil || got != model {
		t.Fatalf("ResolveModel() = %v, %v, want model", got, err)
	}
}

// TestResolver_ResolverFunc_Bad covers the nil-function guard — a
// zero-value ResolverFunc must fail loudly rather than panic on call.
func TestResolver_ResolverFunc_Bad(t *testing.T) {
	var fn ResolverFunc
	got, err := fn.ResolveModel(context.Background(), "qwen")
	if got != nil || err == nil || !core.Contains(err.Error(), "resolver is nil") {
		t.Fatalf("ResolveModel() = %v, %v, want nil-resolver error", got, err)
	}
}

// fakeLoadBackend is a minimal inference.Backend used to exercise
// BackendResolver without a real GPU runtime.
type fakeLoadBackend struct {
	name      string
	available bool
	model     inference.TextModel
	loadErr   error
}

func (b *fakeLoadBackend) Name() string { return b.name }

func (b *fakeLoadBackend) Available() bool { return b.available }

func (b *fakeLoadBackend) LoadModel(string, ...inference.LoadOption) core.Result {
	if b.loadErr != nil {
		return core.Fail(b.loadErr)
	}
	return core.Ok(b.model)
}

// wrongTypeBackend's LoadModel reports success with a Value that is
// not an inference.TextModel — exercises BackendResolver's defensive
// type assertion on the loaded value.
type wrongTypeBackend struct{ name string }

func (b *wrongTypeBackend) Name() string    { return b.name }
func (b *wrongTypeBackend) Available() bool { return true }
func (b *wrongTypeBackend) LoadModel(string, ...inference.LoadOption) core.Result {
	return core.Ok("not-a-text-model")
}

func TestResolver_NewBackendResolver_Good(t *testing.T) {
	r := NewBackendResolver(" test-backend ", " /models/x ", inference.WithContextLen(4096))

	if r.BackendName != "test-backend" || r.ModelPath != "/models/x" {
		t.Fatalf("NewBackendResolver() = %+v, want trimmed fields", r)
	}
	if len(r.LoadOptions) != 1 {
		t.Fatalf("LoadOptions = %d, want 1", len(r.LoadOptions))
	}
}

func TestResolver_BackendResolver_ResolveModel_Good(t *testing.T) {
	model := &stubModel{}
	backend := &fakeLoadBackend{name: "resolver-good-backend", available: true, model: model}
	inference.Register(backend)

	r := NewBackendResolver(backend.name, "/models/x")
	got, err := r.ResolveModel(context.Background(), "ignored")
	if err != nil || got != model {
		t.Fatalf("ResolveModel() = %v, %v, want model", got, err)
	}

	// Second call must reuse the cached model rather than loading again.
	got2, err := r.ResolveModel(context.Background(), "ignored")
	if err != nil || got2 != model {
		t.Fatalf("ResolveModel() cached = %v, %v, want same model", got2, err)
	}
}

// TestResolver_BackendResolver_ResolveModel_Bad covers the guard
// branches: nil receiver, empty ModelPath, and an already-cancelled
// context.
func TestResolver_BackendResolver_ResolveModel_Bad(t *testing.T) {
	var nilResolver *BackendResolver
	if _, err := nilResolver.ResolveModel(context.Background(), "x"); err == nil || !core.Contains(err.Error(), "resolver is nil") {
		t.Fatalf("nil BackendResolver.ResolveModel() error = %v, want nil-resolver error", err)
	}

	empty := &BackendResolver{}
	if _, err := empty.ResolveModel(context.Background(), "x"); err == nil || !core.Contains(err.Error(), "model path is required") {
		t.Fatalf("empty ModelPath ResolveModel() error = %v, want model-path-required error", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cancelled := NewBackendResolver("any-backend", "/models/x")
	if _, err := cancelled.ResolveModel(ctx, "x"); err == nil {
		t.Fatal("cancelled-context ResolveModel() error = nil, want ctx.Err()")
	}
}

// TestResolver_BackendResolver_ResolveModel_Ugly covers the two
// failure shapes that only surface once the backend is actually
// invoked: a failing LoadModel call, and a successful LoadModel call
// whose Value is not an inference.TextModel.
func TestResolver_BackendResolver_ResolveModel_Ugly(t *testing.T) {
	failing := &fakeLoadBackend{name: "resolver-ugly-failing-backend", available: true, loadErr: core.E("test", "load failed", nil)}
	inference.Register(failing)
	r := NewBackendResolver(failing.name, "/models/x")
	if _, err := r.ResolveModel(context.Background(), "x"); err == nil || !core.Contains(err.Error(), "load failed") {
		t.Fatalf("ResolveModel() error = %v, want wrapped load failure", err)
	}

	wrongType := &wrongTypeBackend{name: "resolver-ugly-wrong-type-backend"}
	inference.Register(wrongType)
	r2 := NewBackendResolver(wrongType.name, "/models/x")
	if _, err := r2.ResolveModel(context.Background(), "x"); err == nil || !core.Contains(err.Error(), "not an inference.TextModel") {
		t.Fatalf("ResolveModel() error = %v, want type-assertion failure", err)
	}
}

// TestResolver_StaticResolver_ResolveModel_Bad_CancelledContext covers
// the ctx.Done() short-circuit — a resolver holding the requested
// model must still refuse once the caller's context is already
// cancelled.
func TestResolver_StaticResolver_ResolveModel_Bad_CancelledContext(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := resolver.ResolveModel(ctx, "qwen"); err == nil {
		t.Fatal("ResolveModel(cancelled context) error = nil, want ctx.Err()")
	}
}

// TestResolver_StaticResolver_ResolveModel_Bad_NilReceiver covers the
// nil-receiver guard.
func TestResolver_StaticResolver_ResolveModel_Bad_NilReceiver(t *testing.T) {
	var resolver *StaticResolver

	if _, err := resolver.ResolveModel(context.Background(), "qwen"); err == nil || !core.Contains(err.Error(), "resolver is nil") {
		t.Fatalf("ResolveModel() error = %v, want nil-resolver error", err)
	}
}
