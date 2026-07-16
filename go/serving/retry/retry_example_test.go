// SPDX-Licence-Identifier: EUPL-1.2

package retry

import (
	"context"
	"time"

	core "dappco.re/go"
)

// ExampleDo demonstrates the backoff loop: a call that fails twice with a
// retryable class then succeeds returns nil after three attempts, having
// slept out two backoffs (sleep is faked here so the example runs instantly).
func ExampleDo() {
	sleep, _ := recordSleeper()
	fn, calls := fakeFn(2, core.E("provider", "503", nil))
	p := Policy{
		InitialInterval: 100 * time.Millisecond,
		MaxInterval:     2 * time.Second,
		MaxElapsed:      10 * time.Second,
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassServiceUnavailable), p)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(*calls)
	// Output:
	// 3
}
