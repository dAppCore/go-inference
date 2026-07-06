// SPDX-Licence-Identifier: EUPL-1.2

// Tests for errors.go — the typed admission-failure predicates.

package lora

import (
	"testing"

	core "dappco.re/go"
)

// TestErrors_IsCannotFit_Good covers the structural "can never fit" failure:
// an error minted by errCannotFit reports true, and the sibling CannotAdmit
// kind reports false for the same error (the two kinds are mutually
// exclusive).
func TestErrors_IsCannotFit_Good(t *testing.T) {
	err := errCannotFit("alpha")
	if !IsCannotFit(err) {
		t.Fatalf("IsCannotFit(errCannotFit) = false, want true")
	}
	if IsCannotAdmit(err) {
		t.Fatalf("IsCannotFit's error must not also match IsCannotAdmit")
	}
}

// TestErrors_IsCannotFit_Bad covers non-matches: nil, an unrelated error, and
// the sibling CannotAdmit kind must all report false rather than a false
// positive.
func TestErrors_IsCannotFit_Bad(t *testing.T) {
	if IsCannotFit(nil) {
		t.Fatalf("IsCannotFit(nil) = true, want false")
	}
	if IsCannotFit(errBoom) {
		t.Fatalf("IsCannotFit(unrelated error) = true, want false")
	}
	if IsCannotFit(errCannotAdmit("beta")) {
		t.Fatalf("IsCannotFit(errCannotAdmit) = true, want false")
	}
}

// TestErrors_IsCannotFit_Ugly covers the multi-level wrap case: fitKind walks
// err's whole tree via core.As, so a CannotFit error wrapped again by an
// unrelated core.E still reports true through two layers of Cause.
func TestErrors_IsCannotFit_Ugly(t *testing.T) {
	inner := errCannotFit("gamma")
	outer := core.E("lora.Use", "admission failed", inner)
	if !IsCannotFit(outer) {
		t.Fatalf("IsCannotFit must find a wrapped CannotFit error through nested Cause chain")
	}
}

// TestErrors_IsCannotAdmit_Good covers the transient "no evictable slot"
// failure: an error minted by errCannotAdmit reports true, and the sibling
// CannotFit kind reports false for the same error.
func TestErrors_IsCannotAdmit_Good(t *testing.T) {
	err := errCannotAdmit("alpha")
	if !IsCannotAdmit(err) {
		t.Fatalf("IsCannotAdmit(errCannotAdmit) = false, want true")
	}
	if IsCannotFit(err) {
		t.Fatalf("IsCannotAdmit's error must not also match IsCannotFit")
	}
}

// TestErrors_IsCannotAdmit_Bad covers non-matches: nil, an unrelated error,
// and the sibling CannotFit kind must all report false.
func TestErrors_IsCannotAdmit_Bad(t *testing.T) {
	if IsCannotAdmit(nil) {
		t.Fatalf("IsCannotAdmit(nil) = true, want false")
	}
	if IsCannotAdmit(errBoom) {
		t.Fatalf("IsCannotAdmit(unrelated error) = true, want false")
	}
	if IsCannotAdmit(errCannotFit("beta")) {
		t.Fatalf("IsCannotAdmit(errCannotFit) = true, want false")
	}
}

// TestErrors_IsCannotAdmit_Ugly covers the multi-level wrap case: a
// CannotAdmit error wrapped again by an unrelated core.E still reports true
// through two layers of Cause.
func TestErrors_IsCannotAdmit_Ugly(t *testing.T) {
	inner := errCannotAdmit("gamma")
	outer := core.E("lora.Use", "admission failed", inner)
	if !IsCannotAdmit(outer) {
		t.Fatalf("IsCannotAdmit must find a wrapped CannotAdmit error through nested Cause chain")
	}
}
