// SPDX-Licence-Identifier: EUPL-1.2

// Package retry classifies a failed inference call and retries the retryable
// ones with exponential backoff (RFC §6.7). The provider surface (§6.1)
// returns typed failures — bad request, unauthorised, rate-limited, provider
// overloaded, timeout, and so on — and only some of those are worth trying
// again. This package answers two questions the router asks on every error:
// which class is this, and should I back off and retry it or surface it now.
//
// classify.go maps HTTP-ish statuses onto the Class set and says which classes
// are retryable; retry.go drives the backoff loop (Do). Backoff is testable —
// the sleep function is injected, so tests assert the schedule without waiting.
//
//	c := retry.Classify(resp.StatusCode)
//	if !retry.Retryable(c) { return err }            // permanent — surface now
//	err := retry.Do(ctx, call, retry.ClassifyErr, retry.Policy{
//	    InitialInterval: 200 * time.Millisecond,
//	    MaxInterval:     10 * time.Second,
//	    MaxElapsed:      time.Minute,
//	    MaxAttempts:     5,
//	    Multiplier:      2.0,
//	})
package retry

// Class is a typed inference-failure class (RFC §6.7). ClassNone is the
// zero value and means "not a failure" — a 2xx status maps to it.
//
//	switch retry.Classify(status) {
//	case retry.ClassRateLimited:        // 429 — back off
//	case retry.ClassUnauthorised:       // 401 — surface, don't retry
//	}
type Class int

// The failure classes, in the order the RFC §6.7 lists them. ClassNone (0) is
// the absence of a failure; the rest each name one provider failure mode.
const (
	ClassNone               Class = iota // not a failure (e.g. 2xx)
	ClassBadRequest                      // 400 — malformed request
	ClassUnauthorised                    // 401 — missing / invalid credential
	ClassPaymentRequired                 // 402 — out of credit
	ClassForbidden                       // 403 — credential lacks access
	ClassNotFound                        // 404 — no such model / endpoint
	ClassPayloadTooLarge                 // 413 — request body over limit
	ClassUnprocessable                   // 422 — semantically invalid request
	ClassRateLimited                     // 429 — per-key / per-provider limit (retryable)
	ClassProviderOverloaded              // upstream overloaded (retryable)
	ClassTimeout                         // edge / request timeout (retryable)
	ClassBadGateway                      // 502 — bad gateway (retryable)
	ClassServiceUnavailable              // 503 — service unavailable (retryable)
	ClassInternal                        // 500 / unmapped — provider-internal
)

// Classify maps an HTTP-ish status code onto a Class. A 2xx status is
// ClassNone (success); a status with no specific class — including 0 and any
// unrecognised code — is ClassInternal, so an unknown failure fails closed
// (permanent) rather than being retried forever.
//
//	retry.Classify(429) // ClassRateLimited
//	retry.Classify(200) // ClassNone
//	retry.Classify(418) // ClassInternal (unmapped)
func Classify(status int) Class {
	switch status {
	case 400:
		return ClassBadRequest
	case 401:
		return ClassUnauthorised
	case 402:
		return ClassPaymentRequired
	case 403:
		return ClassForbidden
	case 404:
		return ClassNotFound
	case 413:
		return ClassPayloadTooLarge
	case 422:
		return ClassUnprocessable
	case 429:
		return ClassRateLimited
	case 502:
		return ClassBadGateway
	case 503:
		return ClassServiceUnavailable
	case 500:
		return ClassInternal
	}
	if status >= 200 && status < 300 {
		return ClassNone
	}
	return ClassInternal
}

// Retryable reports whether a Class is worth trying again (RFC §6.7):
// rate-limited, provider-overloaded, timeout, bad-gateway, and
// service-unavailable are retryable; every other class — including ClassNone
// and any unknown value — is permanent and surfaces immediately.
//
//	if retry.Retryable(c) { /* back off and try again */ }
func Retryable(c Class) bool {
	switch c {
	case ClassRateLimited, ClassProviderOverloaded, ClassTimeout, ClassBadGateway, ClassServiceUnavailable:
		return true
	default:
		return false
	}
}
