// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import core "dappco.re/go"

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

// repeatString returns part repeated count times (empty for count <= 0).
func repeatString(part string, count int) string {
	if count <= 0 {
		return ""
	}
	b := core.NewBuilder()
	for range count {
		b.WriteString(part)
	}
	return b.String()
}
