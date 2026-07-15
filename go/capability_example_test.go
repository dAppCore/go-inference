// SPDX-Licence-Identifier: EUPL-1.2

package inference

import core "dappco.re/go"

func ExampleTokenizerModel() {
	model := &capabilityModel{}
	tokenizer, ok := any(model).(TokenizerModel)
	if !ok {
		return
	}

	core.Println(tokenizer.Decode(tokenizer.Encode("hello")))
	// Output: 1
}

func ExampleAdapterModel() {
	model := &capabilityModel{}
	adapter, ok := any(model).(AdapterModel)
	if !ok {
		return
	}

	identity, _ := adapter.LoadAdapter("/models/domain/adapter.safetensors")

	core.Println(identity.Format)
	// Output: lora
}

func ExampleCapabilityReporter() {
	model := &capabilityModel{}
	report, ok := CapabilitiesOf(model)
	if !ok {
		return
	}

	core.Println(report.Runtime.Backend)
	core.Println(report.Supports(CapabilityProbeEvents))
	// Output:
	// stub
	// true
}
