// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	core "dappco.re/go"
)

func ExampleTextModel_AcceptsImages() {
	m := &TextModel{tm: &fakeVisionTokenModel{accepts: true}}
	core.Println(m.AcceptsImages())
	// Output: true
}
