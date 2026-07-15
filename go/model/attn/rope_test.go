// SPDX-Licence-Identifier: EUPL-1.2

package attn

import "testing"

func TestRopeParamsRotaryDim_Golden(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params RopeParams
		want   int
	}{
		{name: "full", params: RopeParams{HeadDim: 80, PartialRotaryFactor: 1}, want: 80},
		{name: "phi2 partial", params: RopeParams{HeadDim: 80, PartialRotaryFactor: 0.4}, want: 32},
		{name: "implicit full", params: RopeParams{HeadDim: 80}, want: 80},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.params.RotaryDim()
			if err != nil || got != tc.want {
				t.Fatalf("RotaryDim() = %d, %v; want %d, nil", got, err, tc.want)
			}
		})
	}
}

func TestRopeParamsRotaryDim_Bad(t *testing.T) {
	if _, err := (RopeParams{HeadDim: 80, PartialRotaryFactor: 0.33}).RotaryDim(); err == nil {
		t.Fatal("RotaryDim accepted a non-even resolved dimension")
	}
}

func TestRopeParamsRotaryDim_Ugly(t *testing.T) {
	if _, err := (RopeParams{}).RotaryDim(); err == nil {
		t.Fatal("RotaryDim accepted an empty declaration")
	}
}
