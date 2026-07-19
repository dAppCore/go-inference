// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	core "dappco.re/go"
)

// repeatStr returns s repeated count times (empty for count <= 0 or empty s).
func repeatStr(s string, count int) string {
	if count <= 0 || s == "" {
		return ""
	}
	// core.Repeat (strings.Repeat) presizes the buffer to the exact final
	// length — one allocation. The earlier Builder loop grew the buffer
	// geometrically, costing several reallocs + a final copy.
	return core.Repeat(s, count)
}

// userHomeDir returns the current user's home directory.
func userHomeDir() core.Result { return core.UserHomeDir() }

// hostname returns the system hostname.
func hostname() core.Result { return core.Hostname() }

// readAll reads all bytes from a reader, concentrating the core.ReadAll import.
//
//	r := readAll(resp.Body)
//	if !r.OK { return r }
//	data := r.Bytes()
func readAll(r any) core.Result {
	result := core.ReadAll(r)
	if !result.OK {
		return result
	}
	return core.Ok([]byte(result.String()))
}
