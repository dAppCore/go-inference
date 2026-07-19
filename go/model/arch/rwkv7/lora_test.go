// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// mkLora builds a synthetic LoRA-MLP shaped [in]->[low]->[out]. bias=false omits Bias (the g_lora shape).
func mkLora(in, low, out, seed int, bias bool) lora {
	var b []float32
	if bias {
		b = syn(out, seed+2)
	}
	return lora{A: syn(low*in, seed), B: syn(out*low, seed+1), Bias: b, In: in, Low: low, Out: out}
}

// TestLora_forward_Good proves the LoRA-MLP produces [L,Out] and that its bias (when present) is actually
// added — comparing against a bias-free LoRA of otherwise identical A/B confirms Bias participates.
func TestLora_forward_Good(t *testing.T) {
	const L, in, low, out = 3, 5, 2, 4
	lo := mkLora(in, low, out, 1, true)
	x := syn(L*in, 9)

	got, err := lo.forward(x, L, tanhF32)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(got) != L*out {
		t.Fatalf("out len %d, want %d", len(got), L*out)
	}

	noBias := lo
	noBias.Bias = nil
	withoutBias, err := noBias.forward(x, L, tanhF32)
	if err != nil {
		t.Fatalf("forward (no bias): %v", err)
	}
	diff := false
	for i := range got {
		if got[i] != withoutBias[i] {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("Bias made no difference to the output — bias add not applied")
	}
	t.Logf("lora.forward: [%d,%d] -> [%d,%d], bias participates", L, in, L, out)
}

// TestLora_forward_Bad rejects a shape mismatch between x and A (In dimension).
func TestLora_forward_Bad(t *testing.T) {
	lo := mkLora(5, 2, 4, 1, true)
	if _, err := lo.forward(syn(3*4, 9), 3, tanhF32); err == nil {
		t.Fatal("x shaped for the wrong In dimension accepted")
	}
}

// TestLora_forward_Ugly proves a nil activation is treated as identity (fla's activation=None ⇒
// nn.Identity, used by a_lora/v_lora) rather than erroring or zeroing the hidden.
func TestLora_forward_Ugly(t *testing.T) {
	const L, in, low, out = 2, 4, 2, 3
	lo := mkLora(in, low, out, 5, false)
	x := syn(L*in, 11)

	got, err := lo.forward(x, L, nil)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	identity := func(v float32) float32 { return v }
	want, err := lo.forward(x, L, identity)
	if err != nil {
		t.Fatalf("forward (explicit identity): %v", err)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("nil-act[%d] = %v, want identity-act %v", i, got[i], want[i])
		}
	}
}
