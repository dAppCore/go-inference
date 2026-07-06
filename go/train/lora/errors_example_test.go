// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for errors.go — kept separate from errors_test.go so the
// godoc-attached usage snippets stay readable.

package lora

import (
	"context"

	core "dappco.re/go"
)

// ExampleIsCannotFit shows the caller-facing routing decision: a Pool that
// can never fit an adapter (Capacity 0) yields an error IsCannotFit
// recognises, telling the caller to route the request elsewhere rather than
// retry.
func ExampleIsCannotFit() {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 0})
	_ = p.Register(AdapterRef{Name: "a", Path: "/models/a"})

	_, _, err := p.Use(context.Background(), "a")
	core.Println(IsCannotFit(err))

	// Output:
	// true
}

// ExampleIsCannotAdmit shows the retry decision: a full pool where the sole
// resident adapter is still referenced (in flight) yields an error
// IsCannotAdmit recognises, telling the caller it may retry once the lease is
// released.
func ExampleIsCannotAdmit() {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	_ = p.Register(AdapterRef{Name: "a", Path: "/models/a"})
	_ = p.Register(AdapterRef{Name: "b", Path: "/models/b"})

	_, release, _ := p.Use(context.Background(), "a") // a resident + referenced
	defer release()

	_, _, err := p.Use(context.Background(), "b") // capacity 1, a can't be evicted
	core.Println(IsCannotAdmit(err))

	// Output:
	// true
}
