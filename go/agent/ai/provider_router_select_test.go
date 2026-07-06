// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	"testing"

	core "dappco.re/go"
)

// fixtureEndpoints returns a small heterogeneous candidate pool mirroring the
// §6.2 device model: a Metal box, a 16 GB GPU, a free OSS provider, and a paid
// provider — all serving the same model id.
func fixtureEndpoints() []Endpoint {
	return []Endpoint{
		{
			Provider: "openai", Model: "gemma-4", Quantisation: "bf16",
			PromptPrice: 0.5, CompletionPrice: 1.5, Latency: 80, Throughput: 120,
			DeviceID: "remote", Local: false, Free: false,
			Capabilities: []string{"tools", "streaming"},
		},
		{
			Provider: "nim", Model: "gemma-4", Quantisation: "bf16",
			PromptPrice: 0, CompletionPrice: 0, Latency: 200, Throughput: 60,
			DeviceID: "remote", Local: false, Free: true,
			Capabilities: []string{"tools", "streaming"},
		},
		{
			Provider: "local-gpu", Model: "gemma-4", Quantisation: "q4_0",
			PromptPrice: 0, CompletionPrice: 0, Latency: 40, Throughput: 90,
			DeviceID: "gpu-16gb", Local: true, Free: true,
			Capabilities: []string{"tools"},
		},
		{
			Provider: "local-metal", Model: "gemma-4", Quantisation: "bf16",
			PromptPrice: 0, CompletionPrice: 0, Latency: 60, Throughput: 50,
			DeviceID: "m3-ultra", Local: true, Free: true,
			Capabilities: []string{"tools", "streaming"},
		},
	}
}

func providerNames(endpoints []Endpoint) []string {
	return core.SliceMap(endpoints, func(e Endpoint) string { return e.Provider })
}

func TestProviderRouter_SelectEndpoints_Good(t *testing.T) {
	cases := []struct {
		name      string
		request   SelectRequest
		endpoints []Endpoint
		want      []string
	}{
		{
			name:      "default local-first then free-first",
			request:   SelectRequest{Model: "gemma-4"},
			endpoints: fixtureEndpoints(),
			// locals first (metal + gpu, in declared order among equals),
			// then free remote, then paid remote.
			want: []string{"local-gpu", "local-metal", "nim", "openai"},
		},
		{
			name:      "explicit order wins over defaults",
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Order: []string{"openai", "local-metal"}}},
			endpoints: fixtureEndpoints(),
			want:      []string{"openai", "local-metal"},
		},
		{
			name:      "sort by price keeps free ahead of paid",
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByPrice}},
			endpoints: fixtureEndpoints(),
			// three free endpoints (price 0) tie, ordered by input; paid last.
			want: []string{"nim", "local-gpu", "local-metal", "openai"},
		},
		{
			name:      "sort by latency ascending",
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByLatency}},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-gpu", "local-metal", "openai", "nim"},
		},
		{
			name:      "sort by throughput descending",
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Sort: SortByThroughput}},
			endpoints: fixtureEndpoints(),
			want:      []string{"openai", "local-gpu", "nim", "local-metal"},
		},
		{
			name:      "fallback model list expands candidate models in order",
			request:   SelectRequest{Model: "missing-primary", Models: []string{"missing-primary", "gemma-4"}},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-gpu", "local-metal", "nim", "openai"},
		},
		{
			name:      "quantisations filter restricts to q4_0",
			request:   SelectRequest{Model: "gemma-4", Quantisations: []string{"q4_0"}},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-gpu"},
		},
		{
			name:      "max_price ceiling drops the paid endpoint",
			request:   SelectRequest{Model: "gemma-4", MaxPrice: 0.1},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-gpu", "local-metal", "nim"},
		},
		{
			name: "only allow-list keeps just those providers in default order",
			// `only` filters but does not order; the default local-first
			// ordering still applies, so the local endpoint leads.
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Only: []string{"local-metal", "nim"}}},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-metal", "nim"},
		},
		{
			name:      "ignore deny-list removes a provider",
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{Ignore: []string{"local-gpu"}}},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-metal", "nim", "openai"},
		},
		{
			name:      "require_parameters drops endpoints missing a capability",
			request:   SelectRequest{Model: "gemma-4", RequireParameters: true, Capabilities: []string{"streaming"}},
			endpoints: fixtureEndpoints(),
			// local-gpu lacks "streaming"; dropped.
			want: []string{"local-metal", "nim", "openai"},
		},
		{
			name:      "zdr flag keeps only zero-data-retention endpoints",
			request:   SelectRequest{Model: "gemma-4", ZDR: true},
			endpoints: zdrEndpoints(),
			want:      []string{"local-metal", "nim-zdr"},
		},
		{
			name:      "allow_fallbacks false keeps only the primary route",
			request:   SelectRequest{Model: "gemma-4", Preferences: ProviderPreferences{AllowFallbacks: new(false)}},
			endpoints: fixtureEndpoints(),
			want:      []string{"local-gpu"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := SelectEndpoints(tc.request, tc.endpoints)
			if !result.OK {
				t.Fatalf("SelectEndpoints() error = %s", result.Error())
			}
			got := providerNames(result.Value.([]Endpoint))
			if !sliceEqual(got, tc.want) {
				t.Fatalf("SelectEndpoints() order = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestProviderRouter_SelectEndpoints_Bad(t *testing.T) {
	cases := []struct {
		name      string
		request   SelectRequest
		endpoints []Endpoint
		wantErr   string
	}{
		{
			name:      "no candidate matches the requested model",
			request:   SelectRequest{Model: "no-such-model"},
			endpoints: fixtureEndpoints(),
			wantErr:   "no endpoint",
		},
		{
			name:      "every candidate exceeds max_price",
			request:   SelectRequest{Model: "gemma-4", MaxPrice: 0.0001},
			endpoints: paidOnlyEndpoints(),
			wantErr:   "no endpoint",
		},
		{
			name:      "empty endpoint pool",
			request:   SelectRequest{Model: "gemma-4"},
			endpoints: nil,
			wantErr:   "no endpoint",
		},
		{
			name:      "no model specified at all",
			request:   SelectRequest{},
			endpoints: fixtureEndpoints(),
			wantErr:   "model is required",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := SelectEndpoints(tc.request, tc.endpoints)
			if result.OK {
				t.Fatalf("SelectEndpoints() OK = true, want failure")
			}
			if !core.Contains(result.Error(), tc.wantErr) {
				t.Fatalf("SelectEndpoints() error = %q, want %q", result.Error(), tc.wantErr)
			}
		})
	}
}

func TestProviderRouter_SelectEndpoints_Ugly(t *testing.T) {
	t.Run("empty order falls back to default ordering", func(t *testing.T) {
		result := SelectEndpoints(SelectRequest{
			Model:       "gemma-4",
			Preferences: ProviderPreferences{Order: []string{}},
		}, fixtureEndpoints())
		if !result.OK {
			t.Fatalf("SelectEndpoints() error = %s", result.Error())
		}
		got := providerNames(result.Value.([]Endpoint))
		want := []string{"local-gpu", "local-metal", "nim", "openai"}
		if !sliceEqual(got, want) {
			t.Fatalf("SelectEndpoints() order = %v, want default %v", got, want)
		}
	})

	t.Run("only and ignore conflict filters everything out", func(t *testing.T) {
		result := SelectEndpoints(SelectRequest{
			Model: "gemma-4",
			Preferences: ProviderPreferences{
				Only:   []string{"local-metal"},
				Ignore: []string{"local-metal"},
			},
		}, fixtureEndpoints())
		if result.OK {
			t.Fatalf("SelectEndpoints() OK = true, want conflict to filter all out")
		}
		if !core.Contains(result.Error(), "no endpoint") {
			t.Fatalf("SelectEndpoints() error = %q, want no-endpoint failure", result.Error())
		}
	})

	t.Run("required capability missing from all endpoints", func(t *testing.T) {
		result := SelectEndpoints(SelectRequest{
			Model:             "gemma-4",
			RequireParameters: true,
			Capabilities:      []string{"video"},
		}, fixtureEndpoints())
		if result.OK {
			t.Fatalf("SelectEndpoints() OK = true, want missing-capability failure")
		}
		if !core.Contains(result.Error(), "no endpoint") {
			t.Fatalf("SelectEndpoints() error = %q, want no-endpoint failure", result.Error())
		}
	})

	t.Run("quantisations filter removes every candidate", func(t *testing.T) {
		result := SelectEndpoints(SelectRequest{
			Model:         "gemma-4",
			Quantisations: []string{"w4a16"},
		}, fixtureEndpoints())
		if result.OK {
			t.Fatalf("SelectEndpoints() OK = true, want quant filter to empty pool")
		}
		if !core.Contains(result.Error(), "no endpoint") {
			t.Fatalf("SelectEndpoints() error = %q, want no-endpoint failure", result.Error())
		}
	})

	t.Run("order names an absent provider then a present one", func(t *testing.T) {
		result := SelectEndpoints(SelectRequest{
			Model:       "gemma-4",
			Preferences: ProviderPreferences{Order: []string{"ghost", "local-metal", "ghost"}},
		}, fixtureEndpoints())
		if !result.OK {
			t.Fatalf("SelectEndpoints() error = %s", result.Error())
		}
		got := providerNames(result.Value.([]Endpoint))
		if !sliceEqual(got, []string{"local-metal"}) {
			t.Fatalf("SelectEndpoints() order = %v, want only the present provider", got)
		}
	})

	t.Run("duplicate endpoints survive as distinct routes", func(t *testing.T) {
		endpoints := append(fixtureEndpoints(), fixtureEndpoints()[2]) // second local-gpu
		result := SelectEndpoints(SelectRequest{
			Model:         "gemma-4",
			Quantisations: []string{"q4_0"},
		}, endpoints)
		if !result.OK {
			t.Fatalf("SelectEndpoints() error = %s", result.Error())
		}
		if got := result.Value.([]Endpoint); len(got) != 2 {
			t.Fatalf("SelectEndpoints() len = %d, want both q4_0 endpoints retained", len(got))
		}
	})
}

func zdrEndpoints() []Endpoint {
	return []Endpoint{
		{Provider: "openai", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0.5, Local: false, Free: false, ZDR: false},
		{Provider: "nim-zdr", Model: "gemma-4", Quantisation: "bf16", Local: false, Free: true, ZDR: true},
		{Provider: "local-metal", Model: "gemma-4", Quantisation: "bf16", Local: true, Free: true, ZDR: true},
	}
}

func paidOnlyEndpoints() []Endpoint {
	return []Endpoint{
		{Provider: "openai", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0.5, CompletionPrice: 1.5, Local: false, Free: false},
		{Provider: "anthropic", Model: "gemma-4", Quantisation: "bf16", PromptPrice: 0.3, CompletionPrice: 1.2, Local: false, Free: false},
	}
}

//go:fix inline
func boolPtr(v bool) *bool { return new(v) }

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
