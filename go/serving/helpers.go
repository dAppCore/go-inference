// SPDX-Licence-Identifier: EUPL-1.2

// These are thin wrappers that avoid importing banned stdlib packages across
// the package, concentrating the imports in one file.
package serving

import (
	core "dappco.re/go"
)

// applyStopSequences truncates text at the earliest occurrence of any stop
// sequence. Empty stop sequences are ignored.
func applyStopSequences(text string, stopSequences []string) string {
	if text == "" || len(stopSequences) == 0 {
		return text
	}

	cut := len(text)
	for _, stop := range stopSequences {
		if stop == "" {
			continue
		}
		if idx := indexSubstr(text, stop); idx >= 0 && idx < cut {
			cut = idx
		}
	}

	return text[:cut]
}

// indexSubstr returns the index of the first occurrence of substr in s, or -1.
// It avoids the banned strings.Index.
func indexSubstr(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// readAll reads all bytes from a reader.
//
//	r := readAll(resp.Body)
//	if !r.OK { return r }
//	data := r.Value.([]byte)
func readAll(r any) core.Result {
	result := core.ReadAll(r)
	if !result.OK {
		return result
	}
	// core.ReadAll already owns a freshly-read buffer it exposed as a string;
	// AsBytes returns a read-only view of it rather than copying the whole
	// response body again. The only consumers (JSONUnmarshal, Sprintf) treat
	// the bytes as read-only, so the zero-copy view is safe.
	return core.Ok(core.AsBytes(result.Value.(string)))
}
