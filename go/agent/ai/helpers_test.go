// SPDX-License-Identifier: EUPL-1.2

package ai

import core "dappco.re/go"

// repeatString builds count copies of value, for length-boundary tests.
//
//	repeatString("a", 3) // "aaa"
func repeatString(value string, count int) string {
	parts := make([]string, count)
	for i := range parts {
		parts[i] = value
	}
	return core.Join("", parts...)
}
