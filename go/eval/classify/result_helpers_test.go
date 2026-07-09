package classify

import (
	"testing"

	"dappco.re/go"
	"dappco.re/go/i18n"
)

func valueFromResult[T any](r core.Result) (T, error) {
	var zero T
	if !r.OK {
		if err, ok := r.Value.(error); ok {
			return zero, err
		}
		return zero, core.NewError(r.Error())
	}
	v, ok := r.Value.(T)
	if !ok {
		return zero, core.NewError(core.Sprintf("unexpected result value %T", r.Value))
	}
	return v, nil
}

func serviceFromResult(r core.Result) (*i18n.Service, error) {
	return valueFromResult[*i18n.Service](r)
}

func errorFromResult(r core.Result) error {
	if r.OK {
		return nil
	}
	if err, ok := r.Value.(error); ok {
		return err
	}
	return core.NewError(r.Error())
}

// noPanicForAudit runs fn and fails the test if it panics. The audited
// functions return core.Result (which converts internal panics into failed
// Results with logging), so a normal recover guard is all the AX-7 triplets
// need — no global service/locale state machinery.
func noPanicForAudit(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("audit panic: %v", r)
		}
	}()
	fn()
}
