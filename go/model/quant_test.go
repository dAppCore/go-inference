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
