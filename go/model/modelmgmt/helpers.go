// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import core "dappco.re/go"

// readAll reads all bytes from a reader, concentrating the core.ReadAll import.
//
//	r := readAll(resp.Body)
//	if !r.OK { return r }
//	data := r.Value.([]byte)
func readAll(r any) core.Result {
	result := core.ReadAll(r)
	if !result.OK {
		return result
	}
	return core.Ok([]byte(result.Value.(string)))
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

// strVal extracts a string value from a row map, returning "" when the key is
// absent or the value is not a string.
func strVal(row map[string]any, key string) string {
	v, ok := row[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}
