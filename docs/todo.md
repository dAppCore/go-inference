# Repository TODO

Non-HIP issues noticed while working in `engine/hip`, for the owning area to
pick up. HIP engine implementation and performance work does not belong here.

## Tooling

- [ ] Let `bench:hip:gemma4-sweep` configure a longer Go test timeout. A 12B
  32K/tg512 receipt reached state materialization but was killed by `go test`'s
  default 10-minute timeout; the same focused test needs roughly 10-15 minutes
  on the RX 7800 XT.
- [ ] Restore the untagged Darwin implementation of
  `model/safetensors.float16SliceToFloat32`. On 2026-07-16,
  `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go vet ./engine/hip` failed while
  compiling `model/safetensors/index.go:696` and
  `model/safetensors/values.go:53` with `undefined: float16SliceToFloat32`.
  Linux HIP tests and vet pass; this is a shared safetensors portability issue,
  not an `engine/hip` implementation failure.
