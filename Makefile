SHELL := /usr/bin/env bash

GO ?= go
CMAKE ?= cmake
CMAKE_GENERATOR ?= Ninja
HOST_CC ?= gcc
HOST_CXX ?= g++
READELF ?= readelf
SHA256SUM ?= sha256sum
TAR ?= tar
STRIP ?= strip
STRIP_AMD ?= $(STRIP)
STRIP_CUDA ?= $(STRIP)
STRIP_CPU_X86 ?= $(STRIP)
STRIP_CPU_AARCH64 ?= aarch64-linux-gnu-strip
GO_SUBTREE ?= go
# The CLI moved to its own module (cli/lem): go/ is a pure library. The four
# release builds run -C $(CLI_SUBTREE); module tests keep GO_SUBTREE.
CLI_SUBTREE ?= cli
CLI_CMD ?= .
# CLI_NAME names every produced binary (dev + release variants get
# -amd/-cuda/-cpu-* suffixes); CLI_VERSION threads into main.version.
CLI_NAME ?= lem
CLI_VERSION ?= dev
CLI_GO_LDFLAGS = -X main.version=$(CLI_VERSION)
BUILD_DIR ?= build
BIN_DIR ?= $(BUILD_DIR)/bin
DIST_DIR ?= $(BUILD_DIR)/dist
KERNEL_BUILD_DIR ?= $(BUILD_DIR)/kernels
HIP_RUNTIME_BUILD_DIR ?= $(BUILD_DIR)/rocm-clr
HIP_RUNTIME_INSTALL_DIR ?= $(BUILD_DIR)/rocm-clr-install
ROCR_RUNTIME_BUILD_DIR ?= $(BUILD_DIR)/rocr-runtime
ROCR_RUNTIME_INSTALL_DIR ?= $(BUILD_DIR)/rocr-runtime-install
ROCR_CMAKE_SHIM_DIR ?= $(BUILD_DIR)/cmake
KERNEL_SRC ?= go/engine/hip/kernels/rocm_kernels.hip
BIN_DIR_ABS := $(abspath $(BIN_DIR))
DIST_DIR_ABS := $(abspath $(DIST_DIR))
KERNEL_BUILD_DIR_ABS := $(abspath $(KERNEL_BUILD_DIR))
HIP_RUNTIME_BUILD_DIR_ABS := $(abspath $(HIP_RUNTIME_BUILD_DIR))
HIP_RUNTIME_INSTALL_DIR_ABS := $(abspath $(HIP_RUNTIME_INSTALL_DIR))
ROCR_RUNTIME_BUILD_DIR_ABS := $(abspath $(ROCR_RUNTIME_BUILD_DIR))
ROCR_RUNTIME_INSTALL_DIR_ABS := $(abspath $(ROCR_RUNTIME_INSTALL_DIR))
ROCR_CMAKE_SHIM_DIR_ABS := $(abspath $(ROCR_CMAKE_SHIM_DIR))
KERNEL_SRC_ABS := $(abspath $(KERNEL_SRC))
AMD_KERNEL_MODULE_NAME = rocm_kernels_$(AMD_HIP_ARCH).hsaco
AMD_KERNEL_MODULE = $(KERNEL_BUILD_DIR_ABS)/$(AMD_KERNEL_MODULE_NAME)
CUDA_KERNEL_MODULE_NAME = rocm_kernels_nvidia_$(NVIDIA_HIP_ARCH).o
CUDA_KERNEL_MODULE = $(KERNEL_BUILD_DIR_ABS)/$(CUDA_KERNEL_MODULE_NAME)
CPU_X86_KERNEL_MODULE_NAME = rocm_kernels_hip_cpu_x86_64.o
CPU_X86_KERNEL_MODULE = $(KERNEL_BUILD_DIR_ABS)/$(CPU_X86_KERNEL_MODULE_NAME)
CPU_AARCH64_KERNEL_MODULE_NAME = rocm_kernels_hip_cpu_aarch64.o
CPU_AARCH64_KERNEL_MODULE = $(KERNEL_BUILD_DIR_ABS)/$(CPU_AARCH64_KERNEL_MODULE_NAME)
TARGET_GOOS ?= linux
AMD_GOARCH ?= amd64
CUDA_GOARCH ?= amd64
CPU_X86_GOARCH ?= amd64
CPU_AARCH64_GOARCH ?= arm64
CPU_AARCH64_CC ?= aarch64-linux-gnu-gcc
CPU_AARCH64_CXX ?= aarch64-linux-gnu-g++
AMD_CGO_ENABLED ?= 1
CUDA_CGO_ENABLED ?= 1
# DuckDB (the go-store/chathistory driver) requires cgo — like engine/hip,
# the linux lanes build with cgo on. Only the Apple metal engine is no-cgo.
CPU_CGO_ENABLED ?= 1
RELEASE_BINS := $(CLI_NAME)-amd $(CLI_NAME)-cuda $(CLI_NAME)-cpu-x86 $(CLI_NAME)-cpu-aarch64
RELEASE_ARCHIVES := $(addsuffix -linux.tar.gz,$(RELEASE_BINS))
RELEASE_SIDECARS = $(AMD_KERNEL_MODULE_NAME) $(CUDA_KERNEL_MODULE_NAME) $(CPU_X86_KERNEL_MODULE_NAME) $(CPU_AARCH64_KERNEL_MODULE_NAME)

HIPCC ?= hipcc
AMD_HIP_ARCH ?= gfx1100
AMD_HIP_STD ?= c++23
AMD_HIP_OPT ?= -O3
NVIDIA_HIP_ARCH ?= sm_75
NVIDIA_HIP_STD ?= c++20
ROCM_INCLUDE_DIR ?= /opt/rocm/include
ROCM_PATH ?= /opt/rocm
ROCM_LIB_DIR ?= /opt/rocm/lib
ROCM_FALLBACK_PATH ?= /opt/rocm-7.2.0
ROCM_FALLBACK_LIB_DIR ?= /opt/rocm-7.2.0/lib
HIP_API_SOURCE_DIR ?= external/rocm-hip
HIP_RUNTIME_SOURCE_DIR ?= external/rocm-clr
ROCR_RUNTIME_SOURCE_DIR ?= external/rocr-runtime
HIP_API_SOURCE_DIR_ABS := $(abspath $(HIP_API_SOURCE_DIR))
HIP_RUNTIME_SOURCE_DIR_ABS := $(abspath $(HIP_RUNTIME_SOURCE_DIR))
ROCR_RUNTIME_SOURCE_DIR_ABS := $(abspath $(ROCR_RUNTIME_SOURCE_DIR))
HIP_RUNTIME_STATIC_ARCHIVE := $(HIP_RUNTIME_BUILD_DIR_ABS)/hipamd/lib/libamdhip64.a
ROCR_RUNTIME_STATIC_ARCHIVE := $(ROCR_RUNTIME_BUILD_DIR_ABS)/runtime/hsa-runtime/libhsa-runtime64.a
ROCR_HSAKMT_STATIC_ARCHIVE := $(ROCR_RUNTIME_BUILD_DIR_ABS)/libhsakmt/libhsakmt-staticdrm.a
HIP_RUNTIME_BUILD_JOBS ?= $(shell nproc 2>/dev/null || echo 4)
ROCR_RUNTIME_BUILD_JOBS ?= $(shell nproc 2>/dev/null || echo 4)
HIP_RUNTIME_CMAKE_ARGS ?=
ROCR_RUNTIME_CMAKE_ARGS ?=
HIP_DIRECT_GO_TAGS ?= rocm_static_hip
# Release targets build this archive themselves, so its link path must not
# depend on whether it happened to exist while Make was planning the build.
# Callers may still provide HIP_STATIC_ARCHIVE=/path/to/libamdhip64.a.
HIP_STATIC_ARCHIVE ?= $(HIP_RUNTIME_STATIC_ARCHIVE)
ROCR_CLANG ?= $(firstword $(wildcard $(ROCM_PATH)/lib/llvm/bin/clang $(ROCM_FALLBACK_PATH)/lib/llvm/bin/clang /usr/lib/llvm-18/bin/clang /usr/bin/clang-18 /usr/bin/clang))
ROCR_LLVM_OBJCOPY ?= $(firstword $(wildcard $(ROCM_PATH)/lib/llvm/bin/llvm-objcopy $(ROCM_FALLBACK_PATH)/lib/llvm/bin/llvm-objcopy /usr/lib/llvm-18/bin/llvm-objcopy /usr/bin/llvm-objcopy-18 /usr/bin/llvm-objcopy))
HOST_LIBSTDCXX_STATIC ?= $(shell $(HOST_CXX) -print-file-name=libstdc++.a 2>/dev/null || true)
HOST_LIBGCC_EH_STATIC ?= $(shell $(HOST_CC) -print-file-name=libgcc_eh.a 2>/dev/null || true)
DRM_AMDGPU_STATIC_ARCHIVE ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libdrm_amdgpu.a /lib/x86_64-linux-gnu/libdrm_amdgpu.a /opt/amdgpu/lib/x86_64-linux-gnu/libdrm_amdgpu.a))
DRM_STATIC_ARCHIVE ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libdrm.a /lib/x86_64-linux-gnu/libdrm.a /opt/amdgpu/lib/x86_64-linux-gnu/libdrm.a))
ELF_STATIC_ARCHIVE ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libelf.a /lib/x86_64-linux-gnu/libelf.a))
NUMA_STATIC_ARCHIVE ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libnuma.a /lib/x86_64-linux-gnu/libnuma.a))
HIP_STATIC_CXX_LDFLAGS ?= $(if $(wildcard $(HOST_LIBSTDCXX_STATIC)),$(HOST_LIBSTDCXX_STATIC),-lstdc++) $(if $(wildcard $(HOST_LIBGCC_EH_STATIC)),$(HOST_LIBGCC_EH_STATIC),)
HIP_STATIC_HSA_LDFLAGS ?= $(ROCR_RUNTIME_STATIC_ARCHIVE) $(ROCR_HSAKMT_STATIC_ARCHIVE) $(DRM_AMDGPU_STATIC_ARCHIVE) $(DRM_STATIC_ARCHIVE) $(ELF_STATIC_ARCHIVE)
HIP_STATIC_DEP_LDFLAGS ?= $(HIP_STATIC_HSA_LDFLAGS) $(HIP_STATIC_CXX_LDFLAGS) -lm -ldl -lpthread -lrt $(if $(NUMA_STATIC_ARCHIVE),$(NUMA_STATIC_ARCHIVE),-lnuma)
HIP_DIRECT_CGO_LDFLAGS ?= -Wl,--as-needed -L$(ROCM_LIB_DIR) -L$(ROCM_FALLBACK_LIB_DIR) -lamdhip64
HIP_RELEASE_CGO_LDFLAGS ?= $(if $(HIP_STATIC_ARCHIVE),$(HIP_STATIC_ARCHIVE) $(HIP_STATIC_DEP_LDFLAGS),$(HIP_DIRECT_CGO_LDFLAGS))
CUDA_PATH ?= /usr/local/cuda
CUDA_HOME ?= $(CUDA_PATH)
NVCC ?= $(CUDA_PATH)/bin/nvcc

HIP_CPU_INCLUDE ?= /opt/hip-cpu/include
HIP_CPU_CXX ?= g++
HIP_CPU_AARCH64_CXX ?= aarch64-linux-gnu-g++
HIP_CPU_STD ?= c++20
HIP_PRODUCTION_GGUF ?=
HIP_PRODUCTION_HSACO ?= $(AMD_KERNEL_MODULE)
HIP_PRODUCTION_MOE ?= 0

.PHONY: all help build build-cli lem-rocm lthn-rocm named-binaries release-binaries release-dependency-guard release-artifacts dist static-hip-binaries rocr-cmake-shims hsa-static-archive hip-static-archive require-static-hip-archive hip-link-info lem-amd lem-cuda lem-cpu-x86 lem-cpu-aarch64 lthn-amd lthn-cuda lthn-cpu-x86 lthn-cpu-aarch64 test test-cli test-all clean \
	hip hip-amd hip-nvidia hip-cpu hip-cpu-x86_64 hip-cpu-aarch64 \
	test-hip-amd test-hip-nvidia test-hip-cpu test-hip-cpu-runtime test-hip-cpu-kernel-runtime test-zluda-cuda test-hip-production \
	compile-matrix test-matrix

all: build

help:
	@printf '%s\n' \
		'Targets:' \
		'  lem-rocm           build the local development CLI binary plus AMD HSACO sidecar' \
		'  lem-amd            build the AMD ROCm release binary plus HSACO sidecar' \
		'  lem-cuda           build the HIP/CUDA release binary' \
		'  lem-cpu-x86        build the Linux amd64 CPU release binary' \
		'  lem-cpu-aarch64    build the Linux arm64 CPU release binary' \
		'  named-binaries     build all named release binaries' \
		'  release-artifacts  build archives and checksums under $(DIST_DIR)' \
		'  test               run the Go module test suite' \
		'  test-hip-production run the AMD Gemma4 release receipt (set HIP_PRODUCTION_GGUF)' \
		'  test-matrix        run the host suite plus every hip toolchain smoke, skipping what is absent' \
		'  clean              remove $(BUILD_DIR)'

build: build-cli

build-cli:
	mkdir -p "$(BIN_DIR_ABS)"
	$(GO) -C "$(CLI_SUBTREE)" build -ldflags "$(CLI_GO_LDFLAGS)" -o "$(BIN_DIR_ABS)/$(CLI_NAME)" "$(CLI_CMD)"

# Deprecated aliases (the lthn-* spellings) — the canonical verbs are lem-*.
lthn-rocm: lem-rocm
lthn-amd: lem-amd
lthn-cuda: lem-cuda
lthn-cpu-x86: lem-cpu-x86
lthn-cpu-aarch64: lem-cpu-aarch64

lem-rocm: build-cli hip-amd
	cp "$(AMD_KERNEL_MODULE)" "$(BIN_DIR_ABS)/$(AMD_KERNEL_MODULE_NAME)"

named-binaries: lem-amd lem-cuda lem-cpu-x86 lem-cpu-aarch64

release-binaries: named-binaries

release-dependency-guard: release-binaries
	@for bin in $(CLI_NAME)-amd $(CLI_NAME)-cuda; do \
		echo "checking release deps: $$bin"; \
		if $(READELF) -d "$(BIN_DIR_ABS)/$$bin" | grep -E 'NEEDED.*\[(libamdhip64|libhsa-runtime64|libhsakmt|libdrm|libelf|libnuma|libstdc\+\+|libgcc_s)' ; then \
			echo "forbidden shared ROCm/HIP dependency in $(BIN_DIR_ABS)/$$bin"; \
			exit 1; \
		fi; \
		if $(READELF) -d "$(BIN_DIR_ABS)/$$bin" | grep -E '\((RPATH|RUNPATH)\)' ; then \
			echo "release binary must not carry RPATH/RUNPATH: $(BIN_DIR_ABS)/$$bin"; \
			exit 1; \
		fi; \
	done
	@for bin in $(CLI_NAME)-cpu-x86 $(CLI_NAME)-cpu-aarch64; do \
		echo "checking static release deps: $$bin"; \
		if $(READELF) -d "$(BIN_DIR_ABS)/$$bin" 2>/dev/null | grep -E 'NEEDED|\((RPATH|RUNPATH)\)' ; then \
			echo "CPU release binary must be fully static: $(BIN_DIR_ABS)/$$bin"; \
			exit 1; \
		fi; \
	done

release-artifacts: release-binaries release-dependency-guard
	rm -rf "$(DIST_DIR_ABS)"
	mkdir -p "$(DIST_DIR_ABS)"
	for bin in $(RELEASE_BINS); do \
		cp "$(BIN_DIR_ABS)/$$bin" "$(DIST_DIR_ABS)/$$bin"; \
		chmod 0755 "$(DIST_DIR_ABS)/$$bin"; \
	done
	cp "$(BIN_DIR_ABS)/$(AMD_KERNEL_MODULE_NAME)" "$(DIST_DIR_ABS)/$(AMD_KERNEL_MODULE_NAME)"
	cp "$(KERNEL_BUILD_DIR_ABS)/$(CUDA_KERNEL_MODULE_NAME)" "$(DIST_DIR_ABS)/$(CUDA_KERNEL_MODULE_NAME)"
	cp "$(KERNEL_BUILD_DIR_ABS)/$(CPU_X86_KERNEL_MODULE_NAME)" "$(DIST_DIR_ABS)/$(CPU_X86_KERNEL_MODULE_NAME)"
	cp "$(KERNEL_BUILD_DIR_ABS)/$(CPU_AARCH64_KERNEL_MODULE_NAME)" "$(DIST_DIR_ABS)/$(CPU_AARCH64_KERNEL_MODULE_NAME)"
	for sidecar in $(RELEASE_SIDECARS); do \
		chmod 0644 "$(DIST_DIR_ABS)/$$sidecar"; \
	done
	$(STRIP_AMD) "$(DIST_DIR_ABS)/$(CLI_NAME)-amd"
	$(STRIP_CUDA) "$(DIST_DIR_ABS)/$(CLI_NAME)-cuda"
	$(STRIP_CPU_X86) "$(DIST_DIR_ABS)/$(CLI_NAME)-cpu-x86"
	$(STRIP_CPU_AARCH64) "$(DIST_DIR_ABS)/$(CLI_NAME)-cpu-aarch64"
	for bin in $(RELEASE_BINS); do \
		if [ "$$bin" = "$(CLI_NAME)-amd" ]; then \
			(cd "$(DIST_DIR_ABS)" && $(TAR) -czf "$$bin-linux.tar.gz" "$$bin" "$(AMD_KERNEL_MODULE_NAME)"); \
		elif [ "$$bin" = "$(CLI_NAME)-cuda" ]; then \
			(cd "$(DIST_DIR_ABS)" && $(TAR) -czf "$$bin-linux.tar.gz" "$$bin" "$(CUDA_KERNEL_MODULE_NAME)"); \
		elif [ "$$bin" = "$(CLI_NAME)-cpu-x86" ]; then \
			(cd "$(DIST_DIR_ABS)" && $(TAR) -czf "$$bin-linux.tar.gz" "$$bin" "$(CPU_X86_KERNEL_MODULE_NAME)"); \
		elif [ "$$bin" = "$(CLI_NAME)-cpu-aarch64" ]; then \
			(cd "$(DIST_DIR_ABS)" && $(TAR) -czf "$$bin-linux.tar.gz" "$$bin" "$(CPU_AARCH64_KERNEL_MODULE_NAME)"); \
		else \
			(cd "$(DIST_DIR_ABS)" && $(TAR) -czf "$$bin-linux.tar.gz" "$$bin"); \
		fi; \
	done
	(cd "$(DIST_DIR_ABS)" && $(SHA256SUM) $(RELEASE_BINS) $(RELEASE_SIDECARS) $(RELEASE_ARCHIVES) > SHA256SUMS)

dist: release-artifacts

static-hip-binaries: lem-amd lem-cuda

rocr-cmake-shims:
	@test -x "$(ROCR_CLANG)" || { echo "missing ROCr clang; install rocm-llvm or set ROCR_CLANG=/path/to/clang"; exit 1; }
	@test -x "$(ROCR_LLVM_OBJCOPY)" || { echo "missing ROCr llvm-objcopy; install rocm-llvm or set ROCR_LLVM_OBJCOPY=/path/to/llvm-objcopy"; exit 1; }
	mkdir -p "$(ROCR_CMAKE_SHIM_DIR_ABS)/clang" "$(ROCR_CMAKE_SHIM_DIR_ABS)/llvm"
	printf '%s\n' \
		'set(Clang_FOUND TRUE)' \
		'' \
		'if(NOT TARGET clang)' \
		'  add_executable(clang IMPORTED)' \
		'  set_target_properties(clang PROPERTIES IMPORTED_LOCATION "$(ROCR_CLANG)")' \
		'endif()' > "$(ROCR_CMAKE_SHIM_DIR_ABS)/clang/ClangConfig.cmake"
	printf '%s\n' \
		'set(LLVM_FOUND TRUE)' \
		'' \
		'if(NOT TARGET llvm-objcopy)' \
		'  add_executable(llvm-objcopy IMPORTED)' \
		'  set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION "$(ROCR_LLVM_OBJCOPY)")' \
		'endif()' > "$(ROCR_CMAKE_SHIM_DIR_ABS)/llvm/LLVMConfig.cmake"

hsa-static-archive:
	@test -f "$(ROCR_RUNTIME_SOURCE_DIR_ABS)/CMakeLists.txt" || { echo "ROCr runtime source submodule is not initialized: $(ROCR_RUNTIME_SOURCE_DIR)"; exit 1; }
	@test -n "$(DRM_AMDGPU_STATIC_ARCHIVE)" || { echo "missing static libdrm_amdgpu.a; install libdrm-amdgpu-dev"; exit 1; }
	@test -n "$(DRM_STATIC_ARCHIVE)" || { echo "missing static libdrm.a; install libdrm-dev"; exit 1; }
	@test -n "$(ELF_STATIC_ARCHIVE)" || { echo "missing static libelf.a; install libelf-dev"; exit 1; }
	$(MAKE) --no-print-directory rocr-cmake-shims
	$(CMAKE) -S "$(ROCR_RUNTIME_SOURCE_DIR_ABS)" -B "$(ROCR_RUNTIME_BUILD_DIR_ABS)" -G "$(CMAKE_GENERATOR)" \
		-DBUILD_SHARED_LIBS=OFF \
		-DClang_DIR="$(ROCR_CMAKE_SHIM_DIR_ABS)/clang" \
		-DLLVM_DIR="$(ROCR_CMAKE_SHIM_DIR_ABS)/llvm" \
		-DCMAKE_PREFIX_PATH="$(ROCM_PATH);$(ROCM_FALLBACK_PATH)" \
		-DCMAKE_INSTALL_PREFIX="$(ROCR_RUNTIME_INSTALL_DIR_ABS)" \
		-DCMAKE_BUILD_TYPE=Release $(ROCR_RUNTIME_CMAKE_ARGS)
	$(CMAKE) --build "$(ROCR_RUNTIME_BUILD_DIR_ABS)" --target hsa-runtime64_static --parallel "$(ROCR_RUNTIME_BUILD_JOBS)"
	@test -s "$(ROCR_RUNTIME_STATIC_ARCHIVE)" || { echo "expected static ROCr archive was not produced: $(ROCR_RUNTIME_STATIC_ARCHIVE)"; exit 1; }
	@test -s "$(ROCR_HSAKMT_STATIC_ARCHIVE)" || { echo "expected static HSAKMT archive was not produced: $(ROCR_HSAKMT_STATIC_ARCHIVE)"; exit 1; }

hip-static-archive:
	@test -f "$(HIP_API_SOURCE_DIR_ABS)/CMakeLists.txt" || { echo "HIP API source submodule is not initialized: $(HIP_API_SOURCE_DIR)"; exit 1; }
	@test -f "$(HIP_RUNTIME_SOURCE_DIR_ABS)/CMakeLists.txt" || { echo "HIP runtime source submodule is not initialized: $(HIP_RUNTIME_SOURCE_DIR)"; exit 1; }
	$(CMAKE) -S "$(HIP_RUNTIME_SOURCE_DIR_ABS)" -B "$(HIP_RUNTIME_BUILD_DIR_ABS)" -G "$(CMAKE_GENERATOR)" \
		-DCLR_BUILD_HIP=ON \
		-DCLR_BUILD_OCL=OFF \
		-DHIP_PLATFORM=amd \
		-DBUILD_SHARED_LIBS=OFF \
		-D__HIP_ENABLE_PCH=OFF \
		-DHIP_COMMON_DIR="$(HIP_API_SOURCE_DIR_ABS)" \
		-DHIPCC_BIN_DIR="$(ROCM_PATH)/bin" \
		-DAMD_OPENCL_PATH="$(HIP_RUNTIME_SOURCE_DIR_ABS)/opencl" \
		-DROCCLR_PATH="$(HIP_RUNTIME_SOURCE_DIR_ABS)/rocclr" \
		-DCMAKE_PREFIX_PATH="$(ROCM_PATH);$(ROCM_FALLBACK_PATH)" \
		-DCMAKE_INSTALL_PREFIX="$(HIP_RUNTIME_INSTALL_DIR_ABS)" \
		-DCMAKE_BUILD_TYPE=Release $(HIP_RUNTIME_CMAKE_ARGS)
	$(CMAKE) --build "$(HIP_RUNTIME_BUILD_DIR_ABS)" --target amdhip64 --parallel "$(HIP_RUNTIME_BUILD_JOBS)"
	@test -s "$(HIP_RUNTIME_STATIC_ARCHIVE)" || { echo "expected static HIP archive was not produced: $(HIP_RUNTIME_STATIC_ARCHIVE)"; exit 1; }

require-static-hip-archive: hip-static-archive
	@test -s "$(HIP_STATIC_ARCHIVE)" || { echo "expected static HIP archive was not produced: $(HIP_STATIC_ARCHIVE)"; exit 1; }

hip-link-info:
	@if [ -n "$(HIP_STATIC_ARCHIVE)" ]; then \
		echo "HIP link mode: static archive $(HIP_STATIC_ARCHIVE)"; \
		echo "HSA link mode: static archive $(ROCR_RUNTIME_STATIC_ARCHIVE)"; \
		echo "HIP release deps: $(HIP_STATIC_DEP_LDFLAGS)"; \
	else \
		echo "HIP link mode: direct shared ROCm link ($(HIP_DIRECT_CGO_LDFLAGS)); install libamdhip64.a for static HIP release binaries."; \
	fi

lem-amd: hsa-static-archive require-static-hip-archive hip-amd
	mkdir -p "$(BIN_DIR_ABS)"
	$(MAKE) --no-print-directory hip-link-info
	CGO_ENABLED=$(AMD_CGO_ENABLED) CGO_LDFLAGS="$(HIP_RELEASE_CGO_LDFLAGS)" GOOS=$(TARGET_GOOS) GOARCH=$(AMD_GOARCH) $(GO) -C "$(CLI_SUBTREE)" build -tags "$(HIP_DIRECT_GO_TAGS)" -ldflags "$(CLI_GO_LDFLAGS)" -o "$(BIN_DIR_ABS)/$(CLI_NAME)-amd" "$(CLI_CMD)"
	cp "$(AMD_KERNEL_MODULE)" "$(BIN_DIR_ABS)/$(AMD_KERNEL_MODULE_NAME)"

lem-cuda: hsa-static-archive require-static-hip-archive hip-nvidia
	mkdir -p "$(BIN_DIR_ABS)"
	$(MAKE) --no-print-directory hip-link-info
	CGO_ENABLED=$(CUDA_CGO_ENABLED) CGO_LDFLAGS="$(HIP_RELEASE_CGO_LDFLAGS)" GOOS=$(TARGET_GOOS) GOARCH=$(CUDA_GOARCH) $(GO) -C "$(CLI_SUBTREE)" build -tags "$(HIP_DIRECT_GO_TAGS)" -ldflags "$(CLI_GO_LDFLAGS)" -o "$(BIN_DIR_ABS)/$(CLI_NAME)-cuda" "$(CLI_CMD)"

lem-cpu-x86: hip-cpu-x86_64
	mkdir -p "$(BIN_DIR_ABS)"
	CGO_ENABLED=$(CPU_CGO_ENABLED) GOOS=$(TARGET_GOOS) GOARCH=$(CPU_X86_GOARCH) $(GO) -C "$(CLI_SUBTREE)" build -ldflags "$(CLI_GO_LDFLAGS)" -o "$(BIN_DIR_ABS)/$(CLI_NAME)-cpu-x86" "$(CLI_CMD)"

lem-cpu-aarch64: hip-cpu-aarch64
	mkdir -p "$(BIN_DIR_ABS)"
	CGO_ENABLED=$(CPU_CGO_ENABLED) CC=$(CPU_AARCH64_CC) CXX=$(CPU_AARCH64_CXX) GOOS=$(TARGET_GOOS) GOARCH=$(CPU_AARCH64_GOARCH) $(GO) -C "$(CLI_SUBTREE)" build -ldflags "$(CLI_GO_LDFLAGS)" -o "$(BIN_DIR_ABS)/$(CLI_NAME)-cpu-aarch64" "$(CLI_CMD)"

test:
	$(GO) -C "$(GO_SUBTREE)" test ./... -count=1

test-cli:
	$(GO) -C "$(CLI_SUBTREE)" test ./... -count=1

test-all: test test-cli

hip: hip-amd hip-nvidia hip-cpu

compile-matrix: build-cli named-binaries

hip-amd:
	mkdir -p "$(KERNEL_BUILD_DIR_ABS)"
	HIP_PLATFORM=amd $(HIPCC) --std=$(AMD_HIP_STD) --genco --offload-arch=$(AMD_HIP_ARCH) $(AMD_HIP_OPT) "$(KERNEL_SRC_ABS)" -o "$(KERNEL_BUILD_DIR_ABS)/rocm_kernels_$(AMD_HIP_ARCH).hsaco"

hip-nvidia:
	mkdir -p "$(KERNEL_BUILD_DIR_ABS)"
	HIP_PLATFORM=nvidia CUDA_PATH="$(CUDA_PATH)" CUDA_HOME="$(CUDA_HOME)" $(HIPCC) --std=$(NVIDIA_HIP_STD) -c -x cu -I"$(ROCM_INCLUDE_DIR)" -arch=$(NVIDIA_HIP_ARCH) "$(KERNEL_SRC_ABS)" -o "$(CUDA_KERNEL_MODULE)"

hip-cpu: hip-cpu-x86_64 hip-cpu-aarch64

hip-cpu-x86_64:
	mkdir -p "$(KERNEL_BUILD_DIR_ABS)"
	$(HIP_CPU_CXX) -std=$(HIP_CPU_STD) -O2 -x c++ -I"$(HIP_CPU_INCLUDE)" -c "$(KERNEL_SRC_ABS)" -o "$(CPU_X86_KERNEL_MODULE)"

hip-cpu-aarch64:
	mkdir -p "$(KERNEL_BUILD_DIR_ABS)"
	$(HIP_CPU_AARCH64_CXX) -std=$(HIP_CPU_STD) -O2 -x c++ -I"$(HIP_CPU_INCLUDE)" -D'VALGRIND_STACK_REGISTER(a,b)=((void)0)' -c "$(KERNEL_SRC_ABS)" -o "$(CPU_AARCH64_KERNEL_MODULE)"

test-hip-amd:
	GO_ROCM_RUN_AMD_HIP_COMPILE_TESTS=1 $(GO) -C "$(GO_SUBTREE)" test ./engine/hip -run TestHIPKernelSource_AMDHIPCompile_Good -count=1

test-hip-nvidia:
	GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS=1 CUDA_PATH="$(CUDA_PATH)" CUDA_HOME="$(CUDA_HOME)" $(GO) -C "$(GO_SUBTREE)" test ./engine/hip -run TestHIPKernelSource_NVIDIAHIPCompile_Good -count=1

test-hip-cpu:
	GO_ROCM_RUN_HIP_CPU_COMPILE_TESTS=1 GO_ROCM_HIP_CPU_INCLUDE="$(HIP_CPU_INCLUDE)" GO_ROCM_HIP_CPU_CXX="$(HIP_CPU_CXX)" GO_ROCM_HIP_CPU_AARCH64_CXX="$(HIP_CPU_AARCH64_CXX)" $(GO) -C "$(GO_SUBTREE)" test ./engine/hip -run TestHIPKernelSource_HIPCPUCompile_Good -count=1

test-hip-cpu-runtime:
	GO_ROCM_RUN_HIP_CPU_RUNTIME_TESTS=1 GO_ROCM_HIP_CPU_INCLUDE="$(HIP_CPU_INCLUDE)" GO_ROCM_HIP_CPU_CXX="$(HIP_CPU_CXX)" $(GO) -C "$(GO_SUBTREE)" test ./engine/hip -run TestHIPKernelSource_HIPCPURuntimeSmoke_Good -count=1

test-hip-cpu-kernel-runtime:
	GO_ROCM_RUN_HIP_CPU_KERNEL_RUNTIME_TESTS=1 GO_ROCM_HIP_CPU_INCLUDE="$(HIP_CPU_INCLUDE)" GO_ROCM_HIP_CPU_CXX="$(HIP_CPU_CXX)" $(GO) -C "$(GO_SUBTREE)" test ./engine/hip -run TestHIPKernelSource_HIPCPUProductionKernelRuntimeSmoke_Good -count=1

test-zluda-cuda:
	GO_ROCM_RUN_ZLUDA_CUDA_TESTS=1 CUDA_PATH="$(CUDA_PATH)" CUDA_HOME="$(CUDA_HOME)" $(GO) -C "$(GO_SUBTREE)" test ./engine/hip -run TestHIPKernelSource_ZLUDACUDARuntimeSmoke_Good -count=1

test-hip-production:
	@test -n "$(strip $(HIP_PRODUCTION_GGUF))" || { echo "HIP_PRODUCTION_GGUF must name a local Gemma4 GGUF"; exit 1; }
	@test -f "$(HIP_PRODUCTION_GGUF)" || { echo "HIP production GGUF not found: $(HIP_PRODUCTION_GGUF)"; exit 1; }
	$(MAKE) --no-print-directory hip-amd
	@test -f "$(HIP_PRODUCTION_HSACO)" || { echo "HIP production HSACO not found: $(HIP_PRODUCTION_HSACO)"; exit 1; }
	$(GO) -C "$(GO_SUBTREE)" test ./engine/hip -count=1 -run '^TestScheduler_'
	@set -o pipefail; \
		gguf="$$(realpath "$(HIP_PRODUCTION_GGUF)")"; \
		hsaco="$$(realpath "$(HIP_PRODUCTION_HSACO)")"; \
		tests='TestNativeDecodeSmokeKernelStatus_Good|TestHIPGemma4ExactStateContinuityHardware_Good|TestHIPLaneSetE2BHardwareMatchesSingleLanes_Good'; \
		if [ "$(HIP_PRODUCTION_MOE)" = "1" ]; then \
			tests="$$tests|TestHIPLaneSet26BMoEHardwareMatchesSingleLanes_Good"; \
		fi; \
		log="$$(mktemp)"; \
		trap 'rm -f "$$log"' EXIT; \
		HIP_VISIBLE_DEVICES="$${HIP_VISIBLE_DEVICES:-0}" \
		GO_ROCM_RUN_MODEL_TESTS=1 \
		GO_ROCM_RUN_MOE_LANE_TESTS="$(HIP_PRODUCTION_MOE)" \
		GO_ROCM_MODEL_PATH="$$gguf" \
		GO_ROCM_KERNEL_HSACO="$$hsaco" \
		$(GO) -C "$(GO_SUBTREE)" test ./engine/hip -count=1 -v -run "^($$tests)$$" | tee "$$log"; \
		status="$${PIPESTATUS[0]}"; \
		if grep -q -- '--- SKIP:' "$$log"; then \
			echo "HIP production receipt skipped a required hardware test"; \
			exit 1; \
		fi; \
		exit "$$status"

# test-matrix aggregates the host suite plus every hip toolchain smoke behind
# one command. Each leg probes its own toolchain first and skips with a
# one-line reason when it is absent, so the verb runs unattended on any box
# (a full ROCm+CUDA+ZLUDA+HIP-CPU rig, a bare AMD/linux box, or darwin, where
# every hip leg skips and only the host suite runs).
test-matrix: test
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		$(MAKE) --no-print-directory test-hip-amd; \
	else \
		echo "SKIP test-hip-amd: $(HIPCC) not found in PATH"; \
	fi
	@if command -v $(HIPCC) >/dev/null 2>&1 && { [ -x "$(CUDA_PATH)/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1; }; then \
		$(MAKE) --no-print-directory test-hip-nvidia; \
	else \
		echo "SKIP test-hip-nvidia: $(HIPCC) and/or a CUDA toolkit (nvcc) not found"; \
	fi
	@if { [ -x "$(CUDA_PATH)/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1; } && \
		{ [ -f "$${GO_ROCM_ZLUDA_DIR:-/opt/zluda/v5/zluda}/libcuda.so" ] || [ -f /tmp/zluda-v5/zluda/libcuda.so ]; }; then \
		$(MAKE) --no-print-directory test-zluda-cuda; \
	else \
		echo "SKIP test-zluda-cuda: nvcc and/or a ZLUDA v5 unpack (libcuda.so) not found"; \
	fi
	@if grep -qs __HIP_CPU_RT__ "$${GO_ROCM_HIP_CPU_INCLUDE:-$(HIP_CPU_INCLUDE)}/hip/hip_defines.h" 2>/dev/null; then \
		$(MAKE) --no-print-directory test-hip-cpu; \
		$(MAKE) --no-print-directory test-hip-cpu-runtime; \
		$(MAKE) --no-print-directory test-hip-cpu-kernel-runtime; \
	else \
		echo "SKIP test-hip-cpu/test-hip-cpu-runtime/test-hip-cpu-kernel-runtime: HIP-CPU headers not found under $${GO_ROCM_HIP_CPU_INCLUDE:-$(HIP_CPU_INCLUDE)} (clone https://github.com/ROCm/HIP-CPU or set GO_ROCM_HIP_CPU_INCLUDE)"; \
	fi

clean:
	rm -rf "$(BUILD_DIR)"
