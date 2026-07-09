// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import "fmt"

// ExampleParseJSONStringList shows the array-of-strings shape used for
// a model's stop-sequence list.
func ExampleParseJSONStringList() {
	stops, err := ParseJSONStringList([]byte(`["END","STOP"]`))
	fmt.Println(stops, err)
	// Output: [END STOP] <nil>
}

// ExampleParseJSONString shows decoding a single JSON string field.
func ExampleParseJSONString() {
	s, next, err := ParseJSONString([]byte(`"hello"`), 0)
	fmt.Println(s, next, err)
	// Output: hello 7 <nil>
}

// ExampleParseJSONStringRaw shows the no-copy variant returning a
// slice into the original data on the no-escape fast path.
func ExampleParseJSONStringRaw() {
	b, next, err := ParseJSONStringRaw([]byte(`"hello"`), 0)
	fmt.Println(string(b), next, err)
	// Output: hello 7 <nil>
}

// ExampleSkipJSONWhitespace shows advancing past leading whitespace to
// the next value boundary.
func ExampleSkipJSONWhitespace() {
	i := SkipJSONWhitespace([]byte("   {}"), 0)
	fmt.Println(i)
	// Output: 3
}

// ExampleParseJSONInt shows decoding a plain JSON integer.
func ExampleParseJSONInt() {
	n, next, err := ParseJSONInt([]byte(`42`), 0)
	fmt.Println(n, next, err)
	// Output: 42 2 <nil>
}

// ExampleParseJSONBool shows decoding the `true` literal.
func ExampleParseJSONBool() {
	v, next, err := ParseJSONBool([]byte(`true`), 0)
	fmt.Println(v, next, err)
	// Output: true 4 <nil>
}

// ExampleIsJSONNull shows the `null`-literal check used before a
// caller commits to decoding a typed value.
func ExampleIsJSONNull() {
	fmt.Println(IsJSONNull([]byte(`null`), 0))
	// Output: true
}

// ExampleSkipJSONValue shows skipping a whole nested object without
// materialising any of its fields — the single-pass dispatch shape
// used for ignored/unknown keys.
func ExampleSkipJSONValue() {
	next, err := SkipJSONValue([]byte(`{"a":1,"b":[2,3]}`), 0)
	fmt.Println(next, err)
	// Output: 17 <nil>
}

// ExampleSkipJSONString shows advancing past a string value without
// materialising it.
func ExampleSkipJSONString() {
	next, err := SkipJSONString([]byte(`"hello"`), 0)
	fmt.Println(next, err)
	// Output: 7 <nil>
}

// ExampleMatchObjectStart shows asserting and consuming an opening
// brace, whitespace-tolerant.
func ExampleMatchObjectStart() {
	i, err := MatchObjectStart([]byte(`  {"a":1}`), 0)
	fmt.Println(i, err)
	// Output: 3 <nil>
}

// ExampleMatchArrayStart shows asserting and consuming an opening
// bracket, whitespace-tolerant.
func ExampleMatchArrayStart() {
	i, err := MatchArrayStart([]byte(`  [1,2]`), 0)
	fmt.Println(i, err)
	// Output: 3 <nil>
}

// ExampleParseJSONFloat32 shows decoding a plain JSON decimal into a
// float32.
func ExampleParseJSONFloat32() {
	v, next, err := ParseJSONFloat32([]byte(`0.7`), 0)
	fmt.Println(v, next, err)
	// Output: 0.7 3 <nil>
}

// ExampleParseJSONFloat64 shows decoding a plain JSON decimal into a
// float64.
func ExampleParseJSONFloat64() {
	v, next, err := ParseJSONFloat64([]byte(`3.14`), 0)
	fmt.Println(v, next, err)
	// Output: 3.14 4 <nil>
}

// ExampleCountJSONArrayElements shows pre-counting an array body to
// size the destination slice in one allocation.
func ExampleCountJSONArrayElements() {
	count := CountJSONArrayElements([]byte(`1,2,3]`), 0)
	fmt.Println(count)
	// Output: 3
}
