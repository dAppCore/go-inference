// SPDX-Licence-Identifier: EUPL-1.2

package transform

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// ExampleMiddleOut demonstrates compressing an over-window conversation
// (RFC §6.11 "Message transforms"): the leading system turn and the
// most-recent turns are kept, the middle is elided down to a single
// placeholder, and MiddleOut shrinks the kept tail until the result fits.
func ExampleMiddleOut() {
	messages := []chat.Message{
		sys("be terse"),
		user("turn one"),
		asst("reply one"),
		user("turn two"),
		asst("reply two"),
		user("turn three"),
	}

	out, transformed, err := MiddleOut(messages, perMessage(10), 40)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(transformed)
	core.Println(len(out))
	core.Println(out[0].Text())
	core.Println(out[len(out)-1].Text())
	// Output:
	// true
	// 4
	// be terse
	// turn three
}
