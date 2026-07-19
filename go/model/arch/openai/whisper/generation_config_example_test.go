// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleLoadGenerationConfig() {
	g, err := LoadGenerationConfig("testdata")
	core.Println(err == nil, g.DecoderStartTokenID, len(g.LangToID) > 0)
	// Output: true 50258 true
}

func ExampleGenerationConfig_LanguageTokenID() {
	g, err := LoadGenerationConfig("testdata")
	if err != nil {
		core.Println(err)
		return
	}
	id, ok := g.LanguageTokenID("en")
	core.Println(ok, id)
	// Output: true 50259
}

func ExampleGenerationConfig_LanguageCode() {
	g, err := LoadGenerationConfig("testdata")
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(g.LanguageCode(50259))
	// Output: en
}
