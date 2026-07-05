package icons

import (
	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestIcons_Placeholder_Good(t *core.T) {
	icon := Placeholder()
	signature := []byte{0x89, 0x50, 0x4e, 0x47}
	got := icon[:4]

	core.AssertEqual(t, signature, got)
	core.AssertTrue(t, len(icon) > 0)
}

func TestIcons_Placeholder_Bad(t *core.T) {
	icon := Placeholder()
	got := len(icon)
	want := 0

	core.AssertTrue(t, got > want)
	core.AssertNotEqual(t, want, got)
}

func TestIcons_Placeholder_Ugly(t *core.T) {
	first := Placeholder()
	second := Placeholder()
	first[0] = 0

	core.AssertNotEqual(t, first[0], second[0])
	core.AssertEqual(t, byte(0x89), second[0])
}
