// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for pool.go — kept separate from pool_test.go so the
// godoc-attached usage snippets stay readable. Examples use a no-op Loader
// so they never touch a real go-mlx device.

package lora

import (
	"context"

	core "dappco.re/go"
)

// exampleLoader is a zero-work Loader for godoc examples — same shape as
// the real go-mlx apply/unload, but a no-op so examples run without a
// device.
type exampleLoader struct{}

func (exampleLoader) Load(context.Context, AdapterRef) error { return nil }
func (exampleLoader) Unload(context.Context, string) error   { return nil }

// ExampleNewPool shows building a serving pool from a Config.
func ExampleNewPool() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 8})
	core.Println(len(pool.Resident()))
	// Output:
	// 0
}

// ExamplePool_Register shows cataloguing an adapter with the Pool.
func ExamplePool_Register() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 8})
	err := pool.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support", BaseModel: "gemma-e4b"})
	core.Println(err)
	// Output:
	// <nil>
}

// ExamplePool_Unregister shows removing a free adapter from the pool's
// catalogue.
func ExamplePool_Unregister() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 8})
	_ = pool.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	err := pool.Unregister("support-tone")
	core.Println(err)
	// Output:
	// <nil>
}

// ExamplePool_Use shows the load-on-demand, ref-counted serving path.
func ExamplePool_Use() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 8})
	_ = pool.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})

	id, release, err := pool.Use(context.Background(), "support-tone")
	if err != nil {
		core.Println("error:", err)
		return
	}
	defer release()
	core.Println(id == (AdapterRef{Name: "support-tone", Path: "/adapters/support"}).ID())
	// Output:
	// true
}

// ExamplePool_Pin shows protecting a resident adapter from eviction.
func ExamplePool_Pin() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	_ = pool.Register(AdapterRef{Name: "a", Path: "/adapters/a"})
	_ = pool.Register(AdapterRef{Name: "b", Path: "/adapters/b"})

	_, release, _ := pool.Use(context.Background(), "a")
	release()
	pool.Pin("a")

	_, _, err := pool.Use(context.Background(), "b") // capacity 1, a pinned → can't admit
	core.Println(IsCannotAdmit(err))
	// Output:
	// true
}

// ExamplePool_Unpin shows returning a pinned adapter to normal eviction
// eligibility.
func ExamplePool_Unpin() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	_ = pool.Register(AdapterRef{Name: "a", Path: "/adapters/a"})
	_ = pool.Register(AdapterRef{Name: "b", Path: "/adapters/b"})

	_, release, _ := pool.Use(context.Background(), "a")
	release()
	pool.Pin("a")
	pool.Unpin("a")

	_, release2, err := pool.Use(context.Background(), "b") // a evictable again
	if err != nil {
		core.Println("error:", err)
		return
	}
	defer release2()
	core.Println(pool.IsResident("a"))
	// Output:
	// false
}

// ExamplePool_IsResident shows the residency query.
func ExamplePool_IsResident() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 8})
	_ = pool.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})

	core.Println(pool.IsResident("support-tone"))
	_, release, _ := pool.Use(context.Background(), "support-tone")
	defer release()
	core.Println(pool.IsResident("support-tone"))
	// Output:
	// false
	// true
}

// ExamplePool_Resident shows the sorted working-set snapshot.
func ExamplePool_Resident() {
	pool := NewPool(Config{Loader: exampleLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 8})
	_ = pool.Register(AdapterRef{Name: "beta", Path: "/adapters/beta"})
	_ = pool.Register(AdapterRef{Name: "alpha", Path: "/adapters/alpha"})

	_, r1, _ := pool.Use(context.Background(), "beta")
	defer r1()
	_, r2, _ := pool.Use(context.Background(), "alpha")
	defer r2()

	core.Println(pool.Resident())
	// Output:
	// [alpha beta]
}
