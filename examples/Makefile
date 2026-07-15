# SPDX-Licence-Identifier: EUPL-1.2
#
# Build matrix for the examples — one binary per example per backend, named
# <example>-{mlx,amd,cuda,cpu}. This is the consumer-facing template: your own
# app builds the same four ways (see pkg/README.md). Apple is Taskfile
# territory (`task metallib` at the repo root supplies the shader library);
# the linux lanes mirror the repo-root Makefile's lthn-{amd,cuda,cpu-*}
# targets, which own the HIP static archives + kernel sidecars.

EXAMPLES := $(patsubst pkg/%/main.go,%,$(shell find pkg -name main.go | sort))
# Engine-importing examples that only build on darwin/arm64 (e.g. chat/mtp).
DARWIN_ONLY := chat/mtp
LINUX_EXAMPLES := $(filter-out $(DARWIN_ONLY),$(EXAMPLES))
BIN := bin

.PHONY: all mlx cpu amd cuda list clean

all: mlx

list:
	@printf '%s\n' $(EXAMPLES)

# Apple silicon (metal) — plain Go, no tags, no cgo. Runtime needs
# MLX_METALLIB_PATH (task metallib at the repo root builds it).
mlx:
	@mkdir -p $(BIN)
	@for e in $(EXAMPLES); do \
	  out=$(BIN)/$$(echo $$e | tr / -)-mlx; \
	  echo "  build $$out"; \
	  GOOS=darwin GOARCH=arm64 go build -o $$out ./pkg/$$e || exit 1; \
	done

# CPU lane — the hip engine's no-cgo lane: fully static, cross-compilable
# from any host. Pair the binary with the CPU kernel sidecar from the repo
# root (`make hip-cpu-x86_64` / `hip-cpu-aarch64`).
cpu:
	@mkdir -p $(BIN)
	@for e in $(LINUX_EXAMPLES); do \
	  out=$(BIN)/$$(echo $$e | tr / -)-cpu; \
	  echo "  build $$out"; \
	  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o $$out ./pkg/$$e || exit 1; \
	done

# AMD (ROCm) — cgo against the static HIP archives; build ON a linux/amd64
# host after preparing the archives at the repo root:
#   make -C .. hsa-static-archive hip-static-archive hip-amd
#   make -C .. hip-link-info        # prints the CGO_LDFLAGS value to export
amd:
	@test "$$(uname -s)/$$(uname -m)" = "Linux/x86_64" || { echo "amd lane builds on linux/amd64 (see pkg/README.md)"; exit 1; }
	@test -n "$$CGO_LDFLAGS" || { echo "set CGO_LDFLAGS from: make -C .. hip-link-info"; exit 1; }
	@mkdir -p $(BIN)
	@for e in $(LINUX_EXAMPLES); do \
	  out=$(BIN)/$$(echo $$e | tr / -)-amd; \
	  echo "  build $$out"; \
	  CGO_ENABLED=1 go build -tags rocm_static_hip -o $$out ./pkg/$$e || exit 1; \
	done

# CUDA — the hip-nvidia lane; same shape as amd with the nvidia kernel:
#   make -C .. hip-nvidia   (plus the archives above)
cuda:
	@test "$$(uname -s)/$$(uname -m)" = "Linux/x86_64" || { echo "cuda lane builds on linux/amd64 (see pkg/README.md)"; exit 1; }
	@test -n "$$CGO_LDFLAGS" || { echo "set CGO_LDFLAGS from: make -C .. hip-link-info"; exit 1; }
	@mkdir -p $(BIN)
	@for e in $(LINUX_EXAMPLES); do \
	  out=$(BIN)/$$(echo $$e | tr / -)-cuda; \
	  echo "  build $$out"; \
	  CGO_ENABLED=1 go build -tags rocm_static_hip -o $$out ./pkg/$$e || exit 1; \
	done

clean:
	rm -rf $(BIN)
