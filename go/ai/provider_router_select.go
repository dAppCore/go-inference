// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	core "dappco.re/go"
)

// SortMode ranks surviving endpoints by a single cost axis (§6.2 `sort`).
//
//	ai.SelectEndpoints(ai.SelectRequest{Model: "gemma-4", Preferences: ai.ProviderPreferences{Sort: ai.SortByLatency}}, pool)
type SortMode string

const (
	// SortDefault keeps the local-first then free-first ordering.
	SortDefault SortMode = ""
	// SortByPrice ranks by the higher of prompt/completion price, ascending.
	SortByPrice SortMode = "price"
	// SortByLatency ranks by rolling latency, ascending.
	SortByLatency SortMode = "latency"
	// SortByThroughput ranks by rolling throughput, descending.
	SortByThroughput SortMode = "throughput"
)

// Endpoint is one routable backend for a model — a local runtime (Metal /
// 16 GB GPU) or an external provider — carrying the stats §6.2 routes on.
//
//	ep := ai.Endpoint{Provider: "local-metal", Model: "gemma-4", Quantisation: "bf16", Local: true, Free: true}
type Endpoint struct {
	Provider        string
	Model           string
	Quantisation    string
	PromptPrice     float64
	CompletionPrice float64
	Latency         float64
	Throughput      float64
	DeviceID        string
	Capabilities    []string
	Local           bool
	Free            bool
	ZDR             bool
}

// ProviderPreferences carries the §6.2 routing preferences that shape which
// endpoints survive and in what order they are tried.
//
//	prefs := ai.ProviderPreferences{Order: []string{"local-metal", "nim"}}
type ProviderPreferences struct {
	Order          []string
	Only           []string
	Ignore         []string
	AllowFallbacks *bool
	Sort           SortMode
}

// SelectRequest is the routing need a caller hands the selector: the primary
// model plus an ordered fallback list, the required capabilities, and the
// quant / price / ZDR constraints from §6.2.
//
//	req := ai.SelectRequest{Model: "gemma-4", Models: []string{"gemma-4", "qwen"}, MaxPrice: 0.1}
type SelectRequest struct {
	Model             string
	Models            []string
	Capabilities      []string
	Quantisations     []string
	MaxPrice          float64
	ZDR               bool
	RequireParameters bool
	Preferences       ProviderPreferences
}

// SelectEndpoints returns the ordered endpoints to try for a request — the
// primary route plus fallbacks — applying every §6.2 preference and the
// default local-first then free-first ordering. It fails with a typed error
// when no endpoint satisfies the request.
//
//	result := ai.SelectEndpoints(ai.SelectRequest{Model: "gemma-4"}, pool)
//	if !result.OK {
//	    return result
//	}
//	routes := result.Value.([]ai.Endpoint)
func SelectEndpoints(request SelectRequest, endpoints []Endpoint) core.Result {
	wanted := requestedModels(request)
	if len(wanted) == 0 {
		return core.Fail(core.E("ai.SelectEndpoints", "model is required", nil))
	}

	candidates := filterCandidates(request, wanted, endpoints)
	if len(candidates) == 0 {
		return core.Fail(core.E("ai.SelectEndpoints", core.Sprintf("no endpoint satisfies request for model %q", wanted[0]), nil))
	}

	ordered := orderCandidates(request, wanted, candidates)
	if len(ordered) == 0 {
		return core.Fail(core.E("ai.SelectEndpoints", core.Sprintf("no endpoint satisfies request for model %q", wanted[0]), nil))
	}

	if !allowFallbacks(request.Preferences) {
		ordered = ordered[:1]
	}
	return core.Ok(ordered)
}

// requestedModels merges the primary model and fallback list into a
// duplicate-free ordered set; the primary always leads.
func requestedModels(request SelectRequest) []string {
	out := make([]string, 0, len(request.Models)+1)
	add := func(model string) {
		model = core.Trim(model)
		if model == "" || core.SliceContains(out, model) {
			return
		}
		out = append(out, model)
	}
	add(request.Model)
	for _, model := range request.Models {
		add(model)
	}
	return out
}

// filterCandidates drops every endpoint excluded by model, allow/deny lists,
// quantisations, max_price, require_parameters, and the ZDR flag.
func filterCandidates(request SelectRequest, wanted []string, endpoints []Endpoint) []Endpoint {
	out := make([]Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if !core.SliceContains(wanted, core.Trim(endpoint.Model)) {
			continue
		}
		if !providerAllowed(request.Preferences, endpoint.Provider) {
			continue
		}
		if !quantisationAllowed(request.Quantisations, endpoint.Quantisation) {
			continue
		}
		if !priceWithinCeiling(request.MaxPrice, endpoint) {
			continue
		}
		if request.RequireParameters && !endpointHasCapabilities(endpoint, request.Capabilities) {
			continue
		}
		if request.ZDR && !endpoint.ZDR {
			continue
		}
		out = append(out, endpoint)
	}
	return out
}

// providerAllowed honours `only` (allow-list) then `ignore` (deny-list).
func providerAllowed(preferences ProviderPreferences, provider string) bool {
	provider = core.Trim(provider)
	if len(preferences.Only) > 0 && !core.SliceContains(preferences.Only, provider) {
		return false
	}
	if core.SliceContains(preferences.Ignore, provider) {
		return false
	}
	return true
}

// quantisationAllowed keeps an endpoint when no quant filter is set, or when
// its quant is in the requested set.
func quantisationAllowed(quantisations []string, quantisation string) bool {
	if len(quantisations) == 0 {
		return true
	}
	return core.SliceContains(quantisations, core.Trim(quantisation))
}

// priceWithinCeiling keeps an endpoint when no ceiling is set, or when the
// higher of its prompt/completion price is at or below max_price.
func priceWithinCeiling(maxPrice float64, endpoint Endpoint) bool {
	if maxPrice <= 0 {
		return true
	}
	highest := endpoint.PromptPrice
	if endpoint.CompletionPrice > highest {
		highest = endpoint.CompletionPrice
	}
	return highest <= maxPrice
}

// endpointHasCapabilities reports whether an endpoint advertises every
// required capability (§6.2 require_parameters).
func endpointHasCapabilities(endpoint Endpoint, required []string) bool {
	for _, capability := range required {
		capability = core.Trim(capability)
		if capability == "" {
			continue
		}
		if !core.SliceContains(endpoint.Capabilities, capability) {
			return false
		}
	}
	return true
}

// orderCandidates applies explicit `order` (which also filters), else `sort`,
// else the default local-first then free-first ordering.
func orderCandidates(request SelectRequest, wanted []string, candidates []Endpoint) []Endpoint {
	if len(request.Preferences.Order) > 0 {
		return orderByExplicit(request.Preferences.Order, candidates)
	}
	return sortCandidates(request, wanted, candidates)
}

// orderByExplicit keeps only providers named in order, in that order; an
// absent name is skipped and a repeated name is honoured once.
func orderByExplicit(order []string, candidates []Endpoint) []Endpoint {
	out := make([]Endpoint, 0, len(candidates))
	seen := make(map[int]bool, len(candidates))
	for _, name := range order {
		name = core.Trim(name)
		for index, endpoint := range candidates {
			if seen[index] || core.Trim(endpoint.Provider) != name {
				continue
			}
			seen[index] = true
			out = append(out, endpoint)
		}
	}
	return out
}

// sortCandidates ranks by the requested sort axis, with the original input
// position as a deterministic tie-break so equal-cost endpoints keep their
// declared order.
func sortCandidates(request SelectRequest, wanted []string, candidates []Endpoint) []Endpoint {
	out := make([]Endpoint, len(candidates))
	copy(out, candidates)
	indexOf := candidateIndex(candidates)

	switch request.Preferences.Sort {
	case SortByPrice:
		core.SliceSortFunc(out, func(a, b Endpoint) bool {
			pa, pb := highestPrice(a), highestPrice(b)
			if pa != pb {
				return pa < pb
			}
			return indexOf(a) < indexOf(b)
		})
	case SortByLatency:
		core.SliceSortFunc(out, func(a, b Endpoint) bool {
			if a.Latency != b.Latency {
				return a.Latency < b.Latency
			}
			return indexOf(a) < indexOf(b)
		})
	case SortByThroughput:
		core.SliceSortFunc(out, func(a, b Endpoint) bool {
			if a.Throughput != b.Throughput {
				return a.Throughput > b.Throughput
			}
			return indexOf(a) < indexOf(b)
		})
	default:
		core.SliceSortFunc(out, func(a, b Endpoint) bool {
			if a.Local != b.Local {
				return a.Local
			}
			if a.Free != b.Free {
				return a.Free
			}
			if ma, mb := modelRank(wanted, a.Model), modelRank(wanted, b.Model); ma != mb {
				return ma < mb
			}
			return indexOf(a) < indexOf(b)
		})
	}
	return out
}

// candidateIndex returns a lookup from the pre-sort position of each
// endpoint, used as a stable tie-break inside the comparators.
func candidateIndex(candidates []Endpoint) func(Endpoint) int {
	positions := make(map[string]int, len(candidates))
	for index, endpoint := range candidates {
		key := endpointKey(endpoint)
		if _, ok := positions[key]; !ok {
			positions[key] = index
		}
	}
	return func(endpoint Endpoint) int {
		if index, ok := positions[endpointKey(endpoint)]; ok {
			return index
		}
		return len(candidates)
	}
}

// endpointKey identifies an endpoint for tie-break lookups; provider plus
// device plus quant is unique within a candidate pool.
func endpointKey(endpoint Endpoint) string {
	return core.Concat(core.Trim(endpoint.Provider), "|", core.Trim(endpoint.DeviceID), "|", core.Trim(endpoint.Quantisation), "|", core.Trim(endpoint.Model))
}

// modelRank returns the position of a model in the requested fallback order,
// so a primary-model endpoint ranks ahead of a fallback-model one.
func modelRank(wanted []string, model string) int {
	index := core.SliceIndex(wanted, core.Trim(model))
	if index < 0 {
		return len(wanted)
	}
	return index
}

// highestPrice returns the larger of an endpoint's prompt/completion price.
func highestPrice(endpoint Endpoint) float64 {
	if endpoint.CompletionPrice > endpoint.PromptPrice {
		return endpoint.CompletionPrice
	}
	return endpoint.PromptPrice
}

// allowFallbacks reports whether fallbacks are permitted; nil defaults to true.
func allowFallbacks(preferences ProviderPreferences) bool {
	if preferences.AllowFallbacks == nil {
		return true
	}
	return *preferences.AllowFallbacks
}
