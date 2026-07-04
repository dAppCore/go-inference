// SPDX-Licence-Identifier: EUPL-1.2

package hip

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func mergeStringMaps(left, right map[string]string) map[string]string {
	out := cloneStringMap(left)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range right {
		out[key] = value
	}
	return out
}

func joinNonEmptyStrings(values []string, sep string) string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != "" {
			out = append(out, value)
		}
	}
	if len(out) == 0 {
		return ""
	}
	result := out[0]
	for _, value := range out[1:] {
		result += sep + value
	}
	return result
}
