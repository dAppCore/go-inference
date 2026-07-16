# Repository TODO

Non-HIP issues noticed while working in `engine/hip`, for the owning area to
pick up. HIP engine implementation and performance work does not belong here.

## Tooling

- [ ] Let `bench:hip:gemma4-sweep` configure a longer Go test timeout. A 12B
  32K/tg512 receipt reached state materialization but was killed by `go test`'s
  default 10-minute timeout; the same focused test needs roughly 10-15 minutes
  on the RX 7800 XT.
