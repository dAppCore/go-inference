// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/engine/scheme"
)

// fakeQuant is a minimal QuantMatVec for exercising the (backend,kind) registry —
// it carries an identity tag (kind + bits) and a MatVec that just echoes its inputs
// back tagged, so a test can prove the registry handed back THIS impl (not another
// backend's same-kind impl). No real compute: the contract here is dispatch, the
// arithmetic lives in pkg/native / pkg/metal.
type fakeQuant struct {
	kind string
	bits int
	tag  byte // stamped into MatVec output so the resolved impl is identifiable
}

func (f fakeQuant) Kind() string { return f.kind }
func (f fakeQuant) Bits() int    { return f.bits }

func (f fakeQuant) MatVec(x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	out := make([]byte, outDim*bf16Size)
	if len(out) > 0 {
		out[0] = f.tag // identify which registered impl produced this
	}
	return out, nil
}

var _ QuantMatVec = fakeQuant{} // compile-time: a QuantMatVec is a QuantScheme + MatVec
var _ scheme.QuantScheme = fakeQuant{}

// TestBackendQuant_Keying is the (backend,kind) cross-section proof: two backends can
// register the SAME kind and a third (backend, same-kind) coexists — each resolves to
// its own impl. This is the one fact that distinguishes this registry from pkg/scheme's
// kind-only one (the contract's whole reason to exist).
func TestBackendQuant_Keying(t *testing.T) {
	// distinct backends, same scheme kind ("affine") — must NOT collide.
	nativeAffine := fakeQuant{kind: "affine", bits: 4, tag: 0xa1}
	metalAffine := fakeQuant{kind: "affine", bits: 4, tag: 0xb2}
	// a second kind on one backend — proves keying is (backend, kind) not backend-only.
	nativeQ40 := fakeQuant{kind: "q4_0", bits: 4, tag: 0x40}

	if r := RegisterBackendQuant("testnative", nativeAffine); !r.OK {
		t.Fatalf("RegisterBackendQuant(testnative/affine): %v", r.Error())
	}
	if r := RegisterBackendQuant("testmetal", metalAffine); !r.OK {
		t.Fatalf("RegisterBackendQuant(testmetal/affine): %v", r.Error())
	}
	if r := RegisterBackendQuant("testnative", nativeQ40); !r.OK {
		t.Fatalf("RegisterBackendQuant(testnative/q4_0): %v", r.Error())
	}

	cases := []struct {
		backend, kind string
		wantTag       byte
	}{
		{"testnative", "affine", nativeAffine.tag},
		{"testmetal", "affine", metalAffine.tag},
		{"testnative", "q4_0", nativeQ40.tag},
	}
	for _, c := range cases {
		q, ok := BackendQuant(c.backend, c.kind)
		if !ok {
			t.Fatalf("BackendQuant(%q,%q): not found", c.backend, c.kind)
		}
		if q.Kind() != c.kind {
			t.Fatalf("BackendQuant(%q,%q).Kind()=%q, want %q", c.backend, c.kind, q.Kind(), c.kind)
		}
		out, err := q.MatVec(nil, nil, nil, nil, 1, 1, 0, 4)
		if err != nil {
			t.Fatalf("MatVec: %v", err)
		}
		if out[0] != c.wantTag {
			t.Fatalf("BackendQuant(%q,%q) resolved the wrong impl: tag %#x, want %#x",
				c.backend, c.kind, out[0], c.wantTag)
		}
	}
}

// TestBackendQuant_Missing — an unregistered (backend,kind) reports ok=false with a nil
// impl, so the engine can detect "no backend serves this format" rather than panic.
func TestBackendQuant_Missing(t *testing.T) {
	if q, ok := BackendQuant("nosuchbackend", "affine"); ok || q != nil {
		t.Fatalf("BackendQuant(unregistered) = (%v,%v), want (nil,false)", q, ok)
	}
	// a registered backend but an unregistered kind on it is equally absent.
	if r := RegisterBackendQuant("partialbackend", fakeQuant{kind: "affine", bits: 4}); !r.OK {
		t.Fatalf("setup register: %v", r.Error())
	}
	if q, ok := BackendQuant("partialbackend", "q4_0"); ok || q != nil {
		t.Fatalf("BackendQuant(registered-backend, unregistered-kind) = (%v,%v), want (nil,false)", q, ok)
	}
}

// TestBackendQuant_LatestWins — the registry is Open (overwrite), so re-registering a
// (backend,kind) replaces the prior impl. A backend's init runs once, but a test/driver
// that re-registers must get the latest, not a stale or a duplicate-key error.
func TestBackendQuant_LatestWins(t *testing.T) {
	first := fakeQuant{kind: "affine", bits: 4, tag: 0x01}
	second := fakeQuant{kind: "affine", bits: 4, tag: 0x02}
	if r := RegisterBackendQuant("overwrite", first); !r.OK {
		t.Fatalf("first register: %v", r.Error())
	}
	if r := RegisterBackendQuant("overwrite", second); !r.OK {
		t.Fatalf("re-register (Open mode must overwrite, not reject): %v", r.Error())
	}
	q, ok := BackendQuant("overwrite", "affine")
	if !ok {
		t.Fatal("BackendQuant(overwrite/affine): not found after re-register")
	}
	out, _ := q.MatVec(nil, nil, nil, nil, 1, 1, 0, 4)
	if out[0] != second.tag {
		t.Fatalf("latest-wins broken: tag %#x, want %#x (the second registration)", out[0], second.tag)
	}
}

// TestBkKey is the key shape — backend and kind joined by "/", the string the registry
// indexes on. A direct check so the (backend,kind) composition is pinned (a change to the
// separator would silently un-collide or collide keys).
func TestBkKey(t *testing.T) {
	if got := bkKey("native", "affine"); got != "native/affine" {
		t.Fatalf("bkKey = %q, want %q", got, "native/affine")
	}
	if got := bkKey("metal", "q4_0"); got != "metal/q4_0" {
		t.Fatalf("bkKey = %q, want %q", got, "metal/q4_0")
	}
}

// TestQuant_RegisterBackendQuant_Good covers the ordinary registration: it reports OK,
// and the registered impl resolves through BackendQuant at its (backend,kind) key.
func TestQuant_RegisterBackendQuant_Good(t *testing.T) {
	r := RegisterBackendQuant("regtest-good", fakeQuant{kind: "affine", bits: 4, tag: 0x11})
	if !r.OK {
		t.Fatalf("RegisterBackendQuant: %s", r.Error())
	}
	q, ok := BackendQuant("regtest-good", "affine")
	if !ok || q.Kind() != "affine" {
		t.Fatalf("BackendQuant after registration = (%v,%v), want the registered impl", q, ok)
	}
}

// TestQuant_RegisterBackendQuant_Bad covers re-registration under the SAME (backend,kind)
// key: the registry is Open, so it still reports OK (never rejects a re-registration) and
// the new impl replaces the old.
func TestQuant_RegisterBackendQuant_Bad(t *testing.T) {
	if r := RegisterBackendQuant("regtest-bad", fakeQuant{kind: "affine", bits: 4, tag: 0x01}); !r.OK {
		t.Fatalf("first RegisterBackendQuant: %s", r.Error())
	}
	r := RegisterBackendQuant("regtest-bad", fakeQuant{kind: "affine", bits: 4, tag: 0x02})
	if !r.OK {
		t.Fatalf("re-RegisterBackendQuant must still report OK (Open registry): %s", r.Error())
	}
	q, _ := BackendQuant("regtest-bad", "affine")
	out, _ := q.MatVec(nil, nil, nil, nil, 1, 1, 0, 4)
	if out[0] != 0x02 {
		t.Fatalf("re-registration did not overwrite: tag %#x, want 0x02", out[0])
	}
}

// TestQuant_RegisterBackendQuant_Ugly covers registering DIFFERENT kinds under the SAME
// backend name: each occupies its own (backend,kind) slot, so the second registration
// must not clobber the first.
func TestQuant_RegisterBackendQuant_Ugly(t *testing.T) {
	if r := RegisterBackendQuant("regtest-ugly", fakeQuant{kind: "affine", bits: 4, tag: 0xa}); !r.OK {
		t.Fatalf("register affine: %s", r.Error())
	}
	if r := RegisterBackendQuant("regtest-ugly", fakeQuant{kind: "q4_0", bits: 4, tag: 0xb}); !r.OK {
		t.Fatalf("register q4_0: %s", r.Error())
	}
	qa, ok := BackendQuant("regtest-ugly", "affine")
	if !ok || qa.Kind() != "affine" {
		t.Fatalf("BackendQuant(regtest-ugly,affine) = (%v,%v), want the affine impl still present", qa, ok)
	}
	qb, ok := BackendQuant("regtest-ugly", "q4_0")
	if !ok || qb.Kind() != "q4_0" {
		t.Fatalf("BackendQuant(regtest-ugly,q4_0) = (%v,%v), want the q4_0 impl", qb, ok)
	}
}

// TestQuant_BackendQuant_Good covers the ordinary resolve: a registered (backend,kind)
// returns its exact impl.
func TestQuant_BackendQuant_Good(t *testing.T) {
	RegisterBackendQuant("resolvetest-good", fakeQuant{kind: "affine", bits: 8, tag: 0x9})
	q, ok := BackendQuant("resolvetest-good", "affine")
	if !ok {
		t.Fatal("BackendQuant: not found after registration")
	}
	if q.Bits() != 8 {
		t.Fatalf("BackendQuant.Bits() = %d, want 8", q.Bits())
	}
}

// TestQuant_BackendQuant_Bad covers an unregistered backend entirely: ok=false, nil impl.
func TestQuant_BackendQuant_Bad(t *testing.T) {
	if q, ok := BackendQuant("never-registered-backend", "affine"); ok || q != nil {
		t.Fatalf("BackendQuant(unregistered backend) = (%v,%v), want (nil,false)", q, ok)
	}
}

// TestQuant_BackendQuant_Ugly covers a REGISTERED backend but an unregistered kind on
// it: the (backend,kind) composite key means partial matches still miss.
func TestQuant_BackendQuant_Ugly(t *testing.T) {
	RegisterBackendQuant("resolvetest-ugly", fakeQuant{kind: "affine", bits: 4})
	if q, ok := BackendQuant("resolvetest-ugly", "never-registered-kind"); ok || q != nil {
		t.Fatalf("BackendQuant(registered backend, unregistered kind) = (%v,%v), want (nil,false)", q, ok)
	}
}
