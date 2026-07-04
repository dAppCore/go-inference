// SPDX-Licence-Identifier: EUPL-1.2

package autoround_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/autoround"
)

// ExampleBuiltinProfiles lists the three native AutoRound calibration profiles
// in their canonical default/best/light order.
func ExampleBuiltinProfiles() {
	for _, profile := range autoround.BuiltinProfiles() {
		core.Println(profile.ID, profile.Scheme, profile.Iters)
	}
	// Output:
	// auto-round W4A16 200
	// auto-round-best W2A16 1000
	// auto-round-light W4A16 50
}

// ExampleLookupProfile resolves a profile id to its tuning defaults, reporting
// the ok flag alongside the looked-up bit-width.
func ExampleLookupProfile() {
	profile, ok := autoround.LookupProfile(autoround.ProfileAutoRoundBest)
	core.Println(ok, profile.Scheme, profile.NSamples)
	_, missing := autoround.LookupProfile("does-not-exist")
	core.Println(missing)
	// Output:
	// true W2A16 512
	// false
}

// ExampleConfigFromProfile flattens a profile into the QuantizeConfig that
// drives QuantizeWeights; bits come from the resolved scheme.
func ExampleConfigFromProfile() {
	profile, _ := autoround.LookupProfile(autoround.ProfileAutoRound)
	cfg := autoround.ConfigFromProfile(profile)
	core.Println(cfg.Scheme, cfg.Bits, cfg.GroupSize, cfg.Symmetric)
	// Output: W4A16 4 128 true
}

// ExampleProfile_GroupScheme resolves a profile's scheme metadata, applying the
// profile's group-size override on top of the scheme defaults.
func ExampleProfile_GroupScheme() {
	profile := autoround.Profile{Scheme: autoround.SchemeW4A16, GroupSize: 64, Symmetric: true}
	info := profile.GroupScheme()
	core.Println(info.Scheme, info.Bits, info.GroupSize)
	// Output: W4A16 4 64
}

// ExampleResolveScheme normalises a free-form scheme alias to its canonical
// weight-only-quant descriptor.
func ExampleResolveScheme() {
	info, ok := autoround.ResolveScheme("w4a16")
	core.Println(ok, info.Scheme, info.Bits, info.GroupSize, info.Family)
	// Output: true W4A16 4 128 int_woq
}

// ExampleQuantizeWeights round-trips a one-group W4A16 tensor through symmetric
// RTN quantisation and reports the produced metadata.
func ExampleQuantizeWeights() {
	weights := make([]float32, 32)
	for i := range weights {
		weights[i] = float32(i-16) / 8
	}
	out, err := autoround.QuantizeWeights(weights, autoround.QuantizeConfig{Scheme: autoround.SchemeW4A16, GroupSize: 32})
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(out.Bits, out.GroupSize, len(out.QValues), len(out.Scales))
	// Output: 4 32 32 1
}
