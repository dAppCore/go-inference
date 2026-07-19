// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleDecodeWAV16Mono() {
	raw := buildWAV(1, 16000, 16, int16LEBytes([]int16{0, 16384, -16384}))
	samples, err := DecodeWAV16Mono(raw)
	core.Println(err == nil, len(samples))
	// Output: true 3
}
