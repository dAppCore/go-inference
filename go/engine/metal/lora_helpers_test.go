// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestSelectProj(t *testing.T) {
	layer := DecodeLayerWeights{
		WQ:    []byte{1},
		WK:    []byte{2},
		WV:    []byte{3},
		WO:    []byte{4},
		WGate: []byte{5},
		WUp:   []byte{6},
		WDown: []byte{7},
	}
	tests := []struct {
		name string
		want byte
	}{
		{name: "wq", want: 1},
		{name: "wk", want: 2},
		{name: "wv", want: 3},
		{name: "wo", want: 4},
		{name: "wgate", want: 5},
		{name: "wup", want: 6},
		{name: "wdown", want: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := selectProj(&layer, tt.name)
			if got == nil || len(*got) != 1 || (*got)[0] != tt.want {
				t.Fatalf("selectProj(%q) = %v, want byte %d", tt.name, got, tt.want)
			}
			(*got)[0]++
			if (*got)[0] != tt.want+1 {
				t.Fatalf("selectProj(%q) did not return the live layer slice", tt.name)
			}
			(*got)[0] = tt.want
		})
	}
	if got := selectProj(&layer, "unknown"); got != nil {
		t.Fatalf("selectProj(unknown) = %v, want nil", got)
	}
}
