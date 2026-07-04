// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"

type ProductionQuantizationTier = modelgemma4.ProductionQuantizationTier

type ProductionQuantizationSelectionInput = modelgemma4.ProductionQuantizationSelectionInput

type ProductionQuantizationChoice = modelgemma4.ProductionQuantizationChoice

var productionQuantizationTiers = modelgemma4.DefaultProductionQuantizationTiers()

// SelectProductionQuantizationTier chooses the app-facing Gemma4 E2B tier from
// backend-neutral device memory and workload shape. q6 is the normal path; q8
// is quality-first when memory headroom is proven, and q4 is constrained or
// fallback-only.
func SelectProductionQuantizationTier(input ProductionQuantizationSelectionInput) ProductionQuantizationChoice {
	return modelgemma4.SelectProductionQuantizationTier(input)
}

func productionQuantizationTierByBits(policy ProductionQuantizationPolicy, bits int) ProductionQuantizationTier {
	for _, tier := range policy.Tiers {
		if tier.Bits == bits {
			return tier
		}
	}
	return ProductionQuantizationTier{}
}

func productionQuantizationActiveWeightReadBytes(bits int) uint64 {
	return modelgemma4.ProductionQuantizationActiveWeightReadBytes(bits)
}
