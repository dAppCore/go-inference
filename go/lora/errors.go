// SPDX-Licence-Identifier: EUPL-1.2

// Admission errors: the typed cannot-fit / cannot-admit failures and their predicates.

package lora

import core "dappco.re/go"

// fitError is the typed admission failure. Kind distinguishes a structural
// impossibility (CannotFit — the pool is too small even when empty) from a
// transient one (CannotAdmit — full of referenced/pinned adapters, retry once a
// lease is released). Test with IsCannotFit / IsCannotAdmit.
type fitError struct {
	kind string
	name string
}

const (
	kindCannotFit   = "cannot_fit"
	kindCannotAdmit = "cannot_admit"
)

// Error renders the admission failure via the Core error convention.
func (e *fitError) Error() string {
	switch e.kind {
	case kindCannotFit:
		return "lora: adapter cannot fit pool (capacity too small): " + e.name
	default:
		return "lora: cannot admit adapter, no evictable slot: " + e.name
	}
}

func errCannotFit(name string) error {
	return core.E("ai", (&fitError{kind: kindCannotFit, name: name}).Error(), &fitError{kind: kindCannotFit, name: name})
}

func errCannotAdmit(name string) error {
	return core.E("ai", (&fitError{kind: kindCannotAdmit, name: name}).Error(), &fitError{kind: kindCannotAdmit, name: name})
}

// IsCannotFit reports whether err is the structural "adapter can never fit this
// pool" failure (Capacity too small even when empty). The caller routes the
// request elsewhere rather than retrying.
//
//	if lora.IsCannotFit(err) { … route to another node … }
func IsCannotFit(err error) bool { return fitKind(err) == kindCannotFit }

// IsCannotAdmit reports whether err is the transient "no evictable slot" failure
// (the pool is full of in-flight or pinned adapters). The caller may retry once a
// lease is released.
//
//	if lora.IsCannotAdmit(err) { … backoff and retry … }
func IsCannotAdmit(err error) bool { return fitKind(err) == kindCannotAdmit }

// fitKind finds the kind of a fitError in err's chain via core.As (which walks
// the Core error tree, including the Cause of a core.E). Returns "" when err is
// not an admission failure.
func fitKind(err error) string {
	var fe *fitError
	if core.As(err, &fe) {
		return fe.kind
	}
	return ""
}
