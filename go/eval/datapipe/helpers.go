// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	core "dappco.re/go"
)

// readAll reads all bytes from a reader, concentrating the core.ReadAll import
// so the rest of the package stays free of the banned stdlib io package.
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
