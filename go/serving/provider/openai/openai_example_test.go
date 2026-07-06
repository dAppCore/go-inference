// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"

	core "dappco.re/go"
)

func ExampleStopList_UnmarshalJSON() {
	var stops StopList
	if err := json.Unmarshal([]byte(`["END","STOP"]`), &stops); err != nil {
		core.Println(err)
		return
	}

	core.Println(len(stops))
	core.Println(stops[0], stops[1])
	// Output:
	// 2
	// END STOP
}

func ExampleChatMessageDelta_MarshalJSON() {
	delta := ChatMessageDelta{Role: "assistant", Content: "hi"}

	data, err := delta.MarshalJSON()
	if err != nil {
		core.Println(err)
		return
	}

	core.Println(string(data))
	// Output:
	// {"role":"assistant","content":"hi"}
}
