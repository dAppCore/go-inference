module dappco.re/go/inference

go 1.26.0

require (
	dappco.re/go v0.10.4
	dappco.re/go/api v0.15.0
	dappco.re/go/cli v0.10.0
	dappco.re/go/i18n v0.10.0
	dappco.re/go/io v0.11.0
	dappco.re/go/log v0.9.0
	dappco.re/go/process v0.10.0
	dappco.re/go/rag v0.14.0
	github.com/gin-gonic/gin v1.12.0
	github.com/google/uuid v1.6.0
	github.com/marcboeker/go-duckdb/v2 v2.4.3
)

// dappco.re/go/ratelimit is supplied by the go.work `use` directive
// (./external/go-ratelimit/go); it cannot be pinned here until it is published
// under the proxy's expected tag scheme (ratelimit/vX.Y.Z).

require (
	forge.lthn.ai/Snider/Enchantrix v0.0.5
	github.com/ProtonMail/go-crypto v1.3.0 // indirect
	github.com/cloudflare/circl v1.6.3 // indirect
	golang.org/x/crypto v0.48.0 // indirect
	golang.org/x/sys v0.41.0 // indirect
)
