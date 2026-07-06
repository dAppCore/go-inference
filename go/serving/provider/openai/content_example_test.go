// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"

	core "dappco.re/go"
)

func ExampleChatMessage_UnmarshalJSON() {
	var msg ChatMessage
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hello"}`), &msg); err != nil {
		core.Println(err)
		return
	}

	core.Println(msg.Role, msg.Content)
	// Output:
	// user hello
}
