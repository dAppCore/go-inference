// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && cgo && !rocm_legacy_server

package hip

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>

typedef int (*rocblas_create_handle_t)(void**);
typedef int (*rocblas_destroy_handle_t)(void*);
typedef int (*rocblas_sgemm_t)(void*, int, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);

static void* core_rocm_rocblas_lib = NULL;
static void* core_rocm_rocblas_handle = NULL;

static void* core_rocm_open_rocblas() {
	if (core_rocm_rocblas_lib != NULL) {
		return core_rocm_rocblas_lib;
	}
	const char* names[] = {"librocblas.so", "librocblas.so.5", "librocblas.so.4", "librocblas.so.3", "librocblas.so.2", NULL};
	for (int i = 0; names[i] != NULL; i++) {
		core_rocm_rocblas_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
		if (core_rocm_rocblas_lib != NULL) {
			return core_rocm_rocblas_lib;
		}
	}
	return NULL;
}

static int core_rocm_rocblas_available() {
	void* lib = core_rocm_open_rocblas();
	return lib != NULL && dlsym(lib, "rocblas_create_handle") != NULL &&
		dlsym(lib, "rocblas_destroy_handle") != NULL && dlsym(lib, "rocblas_sgemm") != NULL;
}

static int core_rocm_rocblas_sgemm(uintptr_t a, uintptr_t b, uintptr_t c, int m, int k, int n, int transpose_b) {
	void* lib = core_rocm_open_rocblas();
	if (lib == NULL) {
		return -200001;
	}
	rocblas_create_handle_t create_handle = (rocblas_create_handle_t)dlsym(lib, "rocblas_create_handle");
	rocblas_sgemm_t sgemm = (rocblas_sgemm_t)dlsym(lib, "rocblas_sgemm");
	if (create_handle == NULL || sgemm == NULL) {
		return -200002;
	}
	if (core_rocm_rocblas_handle == NULL) {
		int rc = create_handle(&core_rocm_rocblas_handle);
		if (rc != 0) {
			return rc;
		}
	}
	const float alpha = 1.0f;
	const float beta = 0.0f;
	if (transpose_b) {
		// Row-major C=A*B^T is column-major C^T=B*A^T.
		return sgemm(core_rocm_rocblas_handle, 112, 111, n, m, k, &alpha,
			(const float*)b, k, (const float*)a, k, &beta, (float*)c, n);
	}
	// Row-major C=A*B is column-major C^T=B^T*A^T.
	return sgemm(core_rocm_rocblas_handle, 111, 111, n, m, k, &alpha,
		(const float*)b, n, (const float*)a, k, &beta, (float*)c, n);
}
*/
import "C"

import (
	"sync"

	core "dappco.re/go"
)

var hipAudioROCBlasMu sync.Mutex

func (cgoHIPDriver) AudioGEMMAvailable() bool {
	return C.core_rocm_rocblas_available() != 0
}

func (driver cgoHIPDriver) AudioMatMul(a, b []float32, m, k, n int, transposeB bool) ([]float32, error) {
	if m <= 0 || k <= 0 || n <= 0 || len(a) != m*k || len(b) != k*n {
		return nil, core.NewError("rocm.hip.AudioMatMul: invalid matrix geometry")
	}
	aPointer, err := driver.Malloc(uint64(len(a) * 4))
	if err != nil {
		return nil, core.E("rocm.hip.AudioMatMul", "allocate A", err)
	}
	defer driver.Free(aPointer)
	bPointer, err := driver.Malloc(uint64(len(b) * 4))
	if err != nil {
		return nil, core.E("rocm.hip.AudioMatMul", "allocate B", err)
	}
	defer driver.Free(bPointer)
	cPointer, err := driver.Malloc(uint64(m * n * 4))
	if err != nil {
		return nil, core.E("rocm.hip.AudioMatMul", "allocate C", err)
	}
	defer driver.Free(cPointer)
	if err := driver.CopyHostToDevice(aPointer, hipAudioFloat32Bytes(a)); err != nil {
		return nil, core.E("rocm.hip.AudioMatMul", "copy A", err)
	}
	if err := driver.CopyHostToDevice(bPointer, hipAudioFloat32Bytes(b)); err != nil {
		return nil, core.E("rocm.hip.AudioMatMul", "copy B", err)
	}
	transposed := C.int(0)
	if transposeB {
		transposed = 1
	}
	hipAudioROCBlasMu.Lock()
	rc := C.core_rocm_rocblas_sgemm(C.uintptr_t(aPointer), C.uintptr_t(bPointer), C.uintptr_t(cPointer), C.int(m), C.int(k), C.int(n), transposed)
	hipAudioROCBlasMu.Unlock()
	if rc != 0 {
		return nil, core.E("rocm.hip.AudioMatMul", core.Sprintf("rocblas_sgemm returned %d", int(rc)), nil)
	}
	out := make([]float32, m*n)
	if err := driver.CopyDeviceToHost(cPointer, hipAudioFloat32Bytes(out)); err != nil {
		return nil, core.E("rocm.hip.AudioMatMul", "copy C", err)
	}
	return out, nil
}
