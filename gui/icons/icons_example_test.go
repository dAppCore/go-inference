package icons

import core "dappco.re/go"

func ExamplePlaceholder() {
	icon := Placeholder()

	core.Println(len(icon) > 0)
	core.Println(icon[0])
	// Output:
	// true
	// 137
}
