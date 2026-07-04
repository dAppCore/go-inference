// SPDX-Licence-Identifier: EUPL-1.2

package registry

import "sync"

// Ordered stores keyed extension registrations in first-registration order.
// Re-registering an existing key replaces the value while preserving order.
type Ordered[K comparable, V any] struct {
	mu     sync.RWMutex
	order  []K
	values map[K]V
}

// NewOrdered returns an empty ordered registry.
func NewOrdered[K comparable, V any]() *Ordered[K, V] {
	return &Ordered[K, V]{values: map[K]V{}}
}

// Put registers or replaces value for key.
func (registry *Ordered[K, V]) Put(key K, value V) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.values == nil {
		registry.values = map[K]V{}
	}
	if _, ok := registry.values[key]; !ok {
		registry.order = append(registry.order, key)
	}
	registry.values[key] = value
}

// Get returns the value registered for key.
func (registry *Ordered[K, V]) Get(key K) (V, bool) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	value, ok := registry.values[key]
	return value, ok
}

// Keys returns registered keys in first-registration order.
func (registry *Ordered[K, V]) Keys() []K {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	return append([]K(nil), registry.order...)
}

// Values returns registered values in first-registration order.
func (registry *Ordered[K, V]) Values() []V {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	out := make([]V, 0, len(registry.order))
	for _, key := range registry.order {
		value, ok := registry.values[key]
		if ok {
			out = append(out, value)
		}
	}
	return out
}

// Snapshot returns copy-safe ordered keys and values for tests that need to
// restore process-global extension registries.
func (registry *Ordered[K, V]) Snapshot() ([]K, map[K]V) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	order := append([]K(nil), registry.order...)
	values := make(map[K]V, len(registry.values))
	for key, value := range registry.values {
		values[key] = value
	}
	return order, values
}

// Restore replaces the registry state from a previous Snapshot.
func (registry *Ordered[K, V]) Restore(order []K, values map[K]V) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	registry.order = append([]K(nil), order...)
	registry.values = make(map[K]V, len(values))
	for key, value := range values {
		registry.values[key] = value
	}
}
