// SPDX-Licence-Identifier: EUPL-1.2

package classify

import (
	"testing"

	core "dappco.re/go"
)

// TestClassify_failResult_AllArms drives every branch of failResult, which was
// only 33.3% covered because callers exercise a single shape indirectly. It
// coerces five input shapes into a failed core.Result:
//   - an already-failed Result -> returned unchanged
//   - an OK Result carrying an error value -> Fail(that error)
//   - an OK Result carrying a non-error value -> Fail(NewError(r.Error()))
//   - a bare error -> Fail(that error)
//   - any other value -> Fail(NewError(Sprintf("%v", v)))
func TestClassify_failResult_AllArms(t *testing.T) {
	t.Run("already_failed_result_returned_unchanged", func(t *testing.T) {
		orig := core.Fail(core.NewError("boom"))
		got := failResult(orig)
		if got.OK || !core.Contains(got.Error(), "boom") {
			t.Fatalf("failResult(failed) = %#v, want failed carrying \"boom\"", got)
		}
	})
	t.Run("ok_result_with_error_value", func(t *testing.T) {
		got := failResult(core.Ok(core.NewError("inner")))
		if got.OK || !core.Contains(got.Error(), "inner") {
			t.Fatalf("failResult(Ok(error)) = %#v, want failed carrying \"inner\"", got)
		}
	})
	t.Run("ok_result_with_non_error_value", func(t *testing.T) {
		got := failResult(core.Ok("plain string value"))
		if got.OK {
			t.Fatalf("failResult(Ok(string)) = %#v, want failed", got)
		}
	})
	t.Run("bare_error", func(t *testing.T) {
		got := failResult(core.NewError("bare"))
		if got.OK || !core.Contains(got.Error(), "bare") {
			t.Fatalf("failResult(error) = %#v, want failed carrying \"bare\"", got)
		}
	})
	t.Run("non_error_value", func(t *testing.T) {
		got := failResult(42)
		if got.OK || !core.Contains(got.Error(), "42") {
			t.Fatalf("failResult(42) = %#v, want failed carrying \"42\"", got)
		}
	})
}
