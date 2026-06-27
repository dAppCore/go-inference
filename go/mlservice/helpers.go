// SPDX-Licence-Identifier: EUPL-1.2

package mlservice

import core "dappco.re/go"

// fprintf writes a formatted line to any io.Writer-shaped value, concentrating
// the formatting in one place.
func fprintf(w any, format string, args ...any) {
	if f, ok := w.(interface{ Write([]byte) (int, error) }); ok {
		_, _ = f.Write([]byte(core.Sprintf(format, args...)))
	}
}

// readAll reads all bytes from a reader, concentrating the core.ReadAll import.
func readAll(r any) core.Result {
	result := core.ReadAll(r)
	if !result.OK {
		return result
	}
	return core.Ok([]byte(result.Value.(string)))
}
