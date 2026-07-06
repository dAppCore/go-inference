// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import "fmt"

// ExampleAppendJSONString shows encoding a single JSON string value.
func ExampleAppendJSONString() {
	buf := AppendJSONString(nil, "answer")
	fmt.Println(string(buf))
	// Output: "answer"
}

// ExampleAppendStringField shows the `"key":"value"` field shape.
func ExampleAppendStringField() {
	buf := AppendStringField(nil, "model", "qwen3", false)
	fmt.Println(string(buf))
	// Output: "model":"qwen3"
}

// ExampleAppendIntField shows the leading-comma call shape used for
// every field after the first in an object body.
func ExampleAppendIntField() {
	buf := AppendIntField(nil, "index", 0, true)
	fmt.Println(string(buf))
	// Output: ,"index":0
}

// ExampleAppendInt64Field shows a nanosecond-duration field, the wide
// value class the int64 variant exists for.
func ExampleAppendInt64Field() {
	buf := AppendInt64Field(nil, "total_duration", 1_500_000_000, false)
	fmt.Println(string(buf))
	// Output: "total_duration":1500000000
}

// ExampleAppendBoolField shows the Done-flag shape used on the final
// chunk of a streaming response.
func ExampleAppendBoolField() {
	buf := AppendBoolField(nil, "done", true, false)
	fmt.Println(string(buf))
	// Output: "done":true
}

// ExampleAppendFloat32Field shows a sampling-parameter field.
func ExampleAppendFloat32Field() {
	buf := AppendFloat32Field(nil, "temperature", 0.7, false)
	fmt.Println(string(buf))
	// Output: "temperature":0.7
}

// ExampleAppendFloat32 shows the bare-value shape used for per-element
// embedding-vector output.
func ExampleAppendFloat32() {
	buf := AppendFloat32(nil, 0.95)
	fmt.Println(string(buf))
	// Output: 0.95
}

// ExampleAppendFloat64 shows the bare-value shape used for score /
// probability output.
func ExampleAppendFloat64() {
	buf := AppendFloat64(nil, 0.12345)
	fmt.Println(string(buf))
	// Output: 0.12345
}

// ExampleHexChar shows the nibble-to-ASCII contract used by the
// \u00XX escape branch of AppendJSONString.
func ExampleHexChar() {
	fmt.Println(string(HexChar(10)))
	// Output: a
}
