// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import "testing"

// TestBuiltinQuantKindsRegistered_Good pins builtin.go's quant catalogue: every
// kind the hip fork's own scheme package identifies (engine/hip/scheme/builtin.go)
// is mirrored here as identity (Kind+Bits), so scheme.QuantKinds() answers "what
// quant kinds exist" the same way regardless of which engine is loaded
// (design-rocm.md #14) — even on this engine, which has a registered MatVec
// (model.BackendQuant) for "affine" only.
func TestBuiltinQuantKindsRegistered_Good(t *testing.T) {
	for _, tc := range []struct {
		kind string
		bits int
	}{
		{"affine", 0},
		{"bf16", 16},
		{"mxfp4", 4},
		{"mxfp8", 8},
		{"nvfp4", 4},
		{"q4_0", 4},
		{"jangtq", 2},
	} {
		q, ok := QuantFor(tc.kind)
		if !ok {
			t.Errorf("QuantFor(%q) not registered", tc.kind)
			continue
		}
		if q.Kind() != tc.kind {
			t.Errorf("QuantFor(%q).Kind() = %q, want %q", tc.kind, q.Kind(), tc.kind)
		}
		if q.Bits() != tc.bits {
			t.Errorf("QuantFor(%q).Bits() = %d, want %d", tc.kind, q.Bits(), tc.bits)
		}
	}
}
