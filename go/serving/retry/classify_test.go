// SPDX-Licence-Identifier: EUPL-1.2

package retry

import (
	core "dappco.re/go"
)

// TestClassify_Classify_Good pins the retryable upstream-failure mappings
// (RFC §6.7): rate-limited, bad-gateway, and service-unavailable are the
// statuses retry.Do is built to back off and try again on.
func TestClassify_Classify_Good(t *core.T) {
	core.AssertEqual(t, ClassRateLimited, Classify(429))
	core.AssertEqual(t, ClassBadGateway, Classify(502))
	core.AssertEqual(t, ClassServiceUnavailable, Classify(503))
	core.AssertTrue(t, Retryable(Classify(429)))
}

// TestClassify_Classify_Bad pins the permanent client-error mappings — a
// caller mistake (bad request, missing credential, no such model) always
// maps to its documented class and is never retryable.
func TestClassify_Classify_Bad(t *core.T) {
	cases := map[int]Class{
		400: ClassBadRequest,
		401: ClassUnauthorised,
		402: ClassPaymentRequired,
		403: ClassForbidden,
		404: ClassNotFound,
		413: ClassPayloadTooLarge,
		422: ClassUnprocessable,
	}
	for status, want := range cases {
		got := Classify(status)
		core.AssertEqual(t, want, got)
		core.AssertFalse(t, Retryable(got))
	}
}

// TestClassify_Classify_Ugly pins the 2xx boundary: 200 and 299 are both
// ClassNone (success, not a failure), but 300 — one past the range — falls
// through to ClassInternal like any other unmapped status.
func TestClassify_Classify_Ugly(t *core.T) {
	core.AssertEqual(t, ClassNone, Classify(200))
	core.AssertEqual(t, ClassNone, Classify(299))
	core.AssertEqual(t, ClassInternal, Classify(300))
}

// TestClassify_Retryable_Good pins the exact retryable set the RFC names:
// rate-limited, provider-overloaded, timeout, bad-gateway, and
// service-unavailable — the classes retry.Do backs off and tries again on.
func TestClassify_Retryable_Good(t *core.T) {
	for _, c := range []Class{ClassRateLimited, ClassProviderOverloaded, ClassTimeout, ClassBadGateway, ClassServiceUnavailable} {
		core.AssertTrue(t, Retryable(c))
	}
}

// TestClassify_Retryable_Bad pins the permanent classes: none of the
// client-error or provider-internal classes are retryable, and neither is
// ClassNone — there is nothing to retry when the call already succeeded.
func TestClassify_Retryable_Bad(t *core.T) {
	for _, c := range []Class{ClassBadRequest, ClassUnauthorised, ClassPaymentRequired, ClassForbidden, ClassNotFound, ClassPayloadTooLarge, ClassUnprocessable, ClassInternal, ClassNone} {
		core.AssertFalse(t, Retryable(c))
	}
}

// TestClassify_Retryable_Ugly pins the out-of-range edge: a Class value
// outside the declared enum is not retryable — fail closed rather than
// retrying an unrecognised failure forever.
func TestClassify_Retryable_Ugly(t *core.T) {
	core.AssertFalse(t, Retryable(Class(9999)))
	core.AssertFalse(t, Retryable(Class(-1)))
}
