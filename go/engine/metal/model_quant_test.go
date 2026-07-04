// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func TestNativeAffineQuantRegistered(t *testing.T) {
	q, ok := model.BackendQuant("native", "affine")
	if !ok {
		t.Fatal("native affine quant backend is not registered")
	}
	if q.Kind() != "affine" {
		t.Fatalf("Kind() = %q, want affine", q.Kind())
	}
	if q.Bits() != 0 {
		t.Fatalf("Bits() = %d, want 0 so model config supplies the width", q.Bits())
	}
}

func TestAffineQMVZeroSizedMatVec(t *testing.T) {
	requireNativeRuntime(t)

	q := affineQMV{}
	got, err := q.MatVec(nil, nil, nil, nil, 0, 0, 64, 4)
	if err != nil {
		t.Fatalf("affineQMV zero-sized MatVec: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("affineQMV zero-sized MatVec length = %d, want 0", len(got))
	}
}
