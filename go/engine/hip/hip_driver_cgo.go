// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && cgo && !rocm_legacy_server

package hip

/*
#cgo linux,!rocm_static_hip LDFLAGS: -ldl
#cgo rocm_static_hip CFLAGS: -DCORE_ROCM_STATIC_HIP=1
#cgo rocm_static_hip LDFLAGS: -Wl,--as-needed -L/opt/rocm/lib -L/opt/rocm-7.2.0/lib -lamdhip64
#include <stdint.h>
#include <stdlib.h>
#ifdef CORE_ROCM_STATIC_HIP
#include <string.h>
#else
#include <dlfcn.h>
#endif

typedef int (*hipGetDeviceCount_t)(int*);
typedef int (*hipSetDevice_t)(int);
typedef int (*hipMemGetInfo_t)(size_t*, size_t*);
typedef int (*hipRuntimeGetVersion_t)(int*);
typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipFreeAsync_t)(void*, void*);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);
typedef int (*hipMemcpyAsync_t)(void*, const void*, size_t, int, void*);
typedef int (*hipMemsetAsync_t)(void*, int, size_t, void*);
typedef int (*hipModuleLoadData_t)(void**, const void*);
typedef int (*hipModuleUnload_t)(void*);
typedef int (*hipModuleGetFunction_t)(void**, void*, const char*);
typedef int (*hipModuleLaunchKernel_t)(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**);
typedef int (*hipDeviceSynchronize_t)(void);
typedef int (*hipHostMalloc_t)(void**, size_t, unsigned int);
typedef int (*hipHostGetDevicePointer_t)(void**, void*, unsigned int);
typedef int (*hipHostFree_t)(void*);
typedef int (*hipEventCreateWithFlags_t)(void**, unsigned int);
typedef int (*hipEventRecord_t)(void*, void*);
typedef int (*hipEventSynchronize_t)(void*);
typedef int (*hipEventDestroy_t)(void*);

typedef struct {
	int rc;
	uintptr_t first;
	uintptr_t second;
} core_rocm_hip_uintptr2_result;

typedef struct {
	int rc;
	uint64_t value;
} core_rocm_hip_uint64_result;

typedef struct {
	int rc;
	uint32_t value;
} core_rocm_hip_uint32_result;

#ifdef CORE_ROCM_STATIC_HIP
extern int hipGetDeviceCount(int*);
extern int hipSetDevice(int);
extern int hipMemGetInfo(size_t*, size_t*);
extern int hipRuntimeGetVersion(int*);
extern int hipMalloc(void**, size_t);
extern int hipFree(void*);
extern int hipFreeAsync(void*, void*);
extern int hipMemcpy(void*, const void*, size_t, int);
extern int hipMemcpyAsync(void*, const void*, size_t, int, void*);
extern int hipMemsetAsync(void*, int, size_t, void*);
extern int hipModuleLoadData(void**, const void*);
extern int hipModuleUnload(void*);
extern int hipModuleGetFunction(void**, void*, const char*);
extern int hipModuleLaunchKernel(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**);
extern int hipDeviceSynchronize(void);
extern int hipHostMalloc(void**, size_t, unsigned int);
extern int hipHostGetDevicePointer(void**, void*, unsigned int);
extern int hipHostFree(void*);
extern int hipEventCreateWithFlags(void**, unsigned int);
extern int hipEventRecord(void*, void*);
extern int hipEventSynchronize(void*);
extern int hipEventDestroy(void*);

static void* core_rocm_open_hip() {
	return (void*)1;
}

static void* core_rocm_hip_symbol(const char* symbol_name) {
	if (strcmp(symbol_name, "hipGetDeviceCount") == 0) {
		return (void*)hipGetDeviceCount;
	}
	if (strcmp(symbol_name, "hipSetDevice") == 0) {
		return (void*)hipSetDevice;
	}
	if (strcmp(symbol_name, "hipMemGetInfo") == 0) {
		return (void*)hipMemGetInfo;
	}
	if (strcmp(symbol_name, "hipRuntimeGetVersion") == 0) {
		return (void*)hipRuntimeGetVersion;
	}
	if (strcmp(symbol_name, "hipMalloc") == 0) {
		return (void*)hipMalloc;
	}
	if (strcmp(symbol_name, "hipFree") == 0) {
		return (void*)hipFree;
	}
	if (strcmp(symbol_name, "hipFreeAsync") == 0) {
		return (void*)hipFreeAsync;
	}
	if (strcmp(symbol_name, "hipMemcpy") == 0) {
		return (void*)hipMemcpy;
	}
	if (strcmp(symbol_name, "hipMemcpyAsync") == 0) {
		return (void*)hipMemcpyAsync;
	}
	if (strcmp(symbol_name, "hipMemsetAsync") == 0) {
		return (void*)hipMemsetAsync;
	}
	if (strcmp(symbol_name, "hipModuleLoadData") == 0) {
		return (void*)hipModuleLoadData;
	}
	if (strcmp(symbol_name, "hipModuleUnload") == 0) {
		return (void*)hipModuleUnload;
	}
	if (strcmp(symbol_name, "hipModuleGetFunction") == 0) {
		return (void*)hipModuleGetFunction;
	}
	if (strcmp(symbol_name, "hipModuleLaunchKernel") == 0) {
		return (void*)hipModuleLaunchKernel;
	}
	if (strcmp(symbol_name, "hipDeviceSynchronize") == 0) {
		return (void*)hipDeviceSynchronize;
	}
	if (strcmp(symbol_name, "hipHostMalloc") == 0) {
		return (void*)hipHostMalloc;
	}
	if (strcmp(symbol_name, "hipHostGetDevicePointer") == 0) {
		return (void*)hipHostGetDevicePointer;
	}
	if (strcmp(symbol_name, "hipHostFree") == 0) {
		return (void*)hipHostFree;
	}
	if (strcmp(symbol_name, "hipEventCreateWithFlags") == 0) {
		return (void*)hipEventCreateWithFlags;
	}
	if (strcmp(symbol_name, "hipEventRecord") == 0) {
		return (void*)hipEventRecord;
	}
	if (strcmp(symbol_name, "hipEventSynchronize") == 0) {
		return (void*)hipEventSynchronize;
	}
	if (strcmp(symbol_name, "hipEventDestroy") == 0) {
		return (void*)hipEventDestroy;
	}
	return NULL;
}
#else
static void* core_rocm_hip_lib = NULL;

static void* core_rocm_open_hip() {
	if (core_rocm_hip_lib != NULL) {
		return core_rocm_hip_lib;
	}
	const char* names[] = {
		"libamdhip64.so",
		"libamdhip64.so.7",
		"libamdhip64.so.6",
		"libamdhip64.so.5",
		NULL,
	};
	for (int i = 0; names[i] != NULL; i++) {
		core_rocm_hip_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
		if (core_rocm_hip_lib != NULL) {
			return core_rocm_hip_lib;
		}
	}
	return NULL;
}

static void* core_rocm_hip_symbol(const char* name) {
	void* lib = core_rocm_open_hip();
	if (lib == NULL) {
		return NULL;
	}
	return dlsym(lib, name);
}
#endif

static int core_rocm_hip_device_count(int* count) {
	static hipGetDeviceCount_t cached = NULL;
	hipGetDeviceCount_t fn = cached;
	if (fn == NULL) {
		fn = (hipGetDeviceCount_t)core_rocm_hip_symbol("hipGetDeviceCount");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100001;
	}
	return fn(count);
}

static int core_rocm_hip_set_device(int device) {
	static hipSetDevice_t cached = NULL;
	hipSetDevice_t fn = cached;
	if (fn == NULL) {
		fn = (hipSetDevice_t)core_rocm_hip_symbol("hipSetDevice");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100002;
	}
	return fn(device);
}

static int core_rocm_hip_mem_info(size_t* free_bytes, size_t* total_bytes) {
	static hipMemGetInfo_t cached = NULL;
	hipMemGetInfo_t fn = cached;
	if (fn == NULL) {
		fn = (hipMemGetInfo_t)core_rocm_hip_symbol("hipMemGetInfo");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100003;
	}
	return fn(free_bytes, total_bytes);
}

static int core_rocm_hip_runtime_version(int* version) {
	static hipRuntimeGetVersion_t cached = NULL;
	hipRuntimeGetVersion_t fn = cached;
	if (fn == NULL) {
		fn = (hipRuntimeGetVersion_t)core_rocm_hip_symbol("hipRuntimeGetVersion");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100004;
	}
	return fn(version);
}

static int core_rocm_hip_malloc(uintptr_t* out, size_t size) {
	static hipMalloc_t cached = NULL;
	hipMalloc_t fn = cached;
	if (fn == NULL) {
		fn = (hipMalloc_t)core_rocm_hip_symbol("hipMalloc");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100005;
	}
	void* ptr = NULL;
	int rc = fn(&ptr, size);
	*out = (uintptr_t)ptr;
	return rc;
}

static core_rocm_hip_uintptr2_result core_rocm_hip_malloc_result(size_t size) {
	core_rocm_hip_uintptr2_result result = {0, 0, 0};
	uintptr_t ptr = 0;
	result.rc = core_rocm_hip_malloc(&ptr, size);
	result.first = ptr;
	return result;
}

static int core_rocm_hip_free(uintptr_t ptr) {
	static hipFree_t cached = NULL;
	hipFree_t fn = cached;
	if (fn == NULL) {
		fn = (hipFree_t)core_rocm_hip_symbol("hipFree");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100006;
	}
	return fn((void*)ptr);
}

static int core_rocm_hip_free_async(uintptr_t ptr) {
	static hipFreeAsync_t cached = NULL;
	hipFreeAsync_t fn = cached;
	if (fn == NULL) {
		fn = (hipFreeAsync_t)core_rocm_hip_symbol("hipFreeAsync");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100020;
	}
	return fn((void*)ptr, NULL);
}

static int core_rocm_hip_memcpy_htod(uintptr_t dst, void* src, size_t size) {
	static hipMemcpy_t cached = NULL;
	hipMemcpy_t fn = cached;
	if (fn == NULL) {
		fn = (hipMemcpy_t)core_rocm_hip_symbol("hipMemcpy");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100007;
	}
	return fn((void*)dst, src, size, 1);
}

static int core_rocm_hip_memcpy_dtoh(void* dst, uintptr_t src, size_t size) {
	static hipMemcpy_t cached = NULL;
	hipMemcpy_t fn = cached;
	if (fn == NULL) {
		fn = (hipMemcpy_t)core_rocm_hip_symbol("hipMemcpy");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100012;
	}
	return fn(dst, (void*)src, size, 2);
}

static core_rocm_hip_uint64_result core_rocm_hip_memcpy_dtoh_u64(uintptr_t src) {
	core_rocm_hip_uint64_result result = {0, 0};
	uint64_t value = 0;
	result.rc = core_rocm_hip_memcpy_dtoh(&value, src, sizeof(value));
	result.value = value;
	return result;
}

static core_rocm_hip_uint32_result core_rocm_hip_memcpy_dtoh_u32(uintptr_t src) {
	core_rocm_hip_uint32_result result = {0, 0};
	uint32_t value = 0;
	result.rc = core_rocm_hip_memcpy_dtoh(&value, src, sizeof(value));
	result.value = value;
	return result;
}

static int core_rocm_hip_memcpy_htod_async(uintptr_t dst, void* src, size_t size) {
	static hipMemcpyAsync_t cached = NULL;
	hipMemcpyAsync_t fn = cached;
	if (fn == NULL) {
		fn = (hipMemcpyAsync_t)core_rocm_hip_symbol("hipMemcpyAsync");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100015;
	}
	return fn((void*)dst, src, size, 1, NULL);
}

static int core_rocm_hip_memset_async(uintptr_t dst, int value, size_t size) {
	static hipMemsetAsync_t cached = NULL;
	hipMemsetAsync_t fn = cached;
	if (fn == NULL) {
		fn = (hipMemsetAsync_t)core_rocm_hip_symbol("hipMemsetAsync");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100019;
	}
	return fn((void*)dst, value, size, NULL);
}

static int core_rocm_hip_module_load_data(uintptr_t* out, void* image) {
	static hipModuleLoadData_t cached = NULL;
	hipModuleLoadData_t fn = cached;
	if (fn == NULL) {
		fn = (hipModuleLoadData_t)core_rocm_hip_symbol("hipModuleLoadData");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100008;
	}
	void* module = NULL;
	int rc = fn(&module, image);
	*out = (uintptr_t)module;
	return rc;
}

static core_rocm_hip_uintptr2_result core_rocm_hip_module_load_data_result(void* image) {
	core_rocm_hip_uintptr2_result result = {0, 0, 0};
	uintptr_t module = 0;
	result.rc = core_rocm_hip_module_load_data(&module, image);
	result.first = module;
	return result;
}

static int core_rocm_hip_module_unload(uintptr_t module) {
	static hipModuleUnload_t cached = NULL;
	hipModuleUnload_t fn = cached;
	if (fn == NULL) {
		fn = (hipModuleUnload_t)core_rocm_hip_symbol("hipModuleUnload");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100009;
	}
	return fn((void*)module);
}

static int core_rocm_hip_module_get_function(uintptr_t* out, uintptr_t module, const char* name) {
	static hipModuleGetFunction_t cached = NULL;
	hipModuleGetFunction_t fn = cached;
	if (fn == NULL) {
		fn = (hipModuleGetFunction_t)core_rocm_hip_symbol("hipModuleGetFunction");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100010;
	}
	void* function = NULL;
	int rc = fn(&function, (void*)module, name);
	*out = (uintptr_t)function;
	return rc;
}

static core_rocm_hip_uintptr2_result core_rocm_hip_module_get_function_result(uintptr_t module, const char* name) {
	core_rocm_hip_uintptr2_result result = {0, 0, 0};
	uintptr_t function = 0;
	result.rc = core_rocm_hip_module_get_function(&function, module, name);
	result.first = function;
	return result;
}

static int core_rocm_hip_module_launch_kernel(
	uintptr_t function,
	unsigned int grid_x,
	unsigned int grid_y,
	unsigned int grid_z,
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	unsigned int shared_mem_bytes,
	uintptr_t args
) {
	static hipModuleLaunchKernel_t cached = NULL;
	hipModuleLaunchKernel_t fn = cached;
	if (fn == NULL) {
		fn = (hipModuleLaunchKernel_t)core_rocm_hip_symbol("hipModuleLaunchKernel");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100011;
	}
	uintptr_t arg_ptr = args;
	void* kernel_params[] = { &arg_ptr };
	return fn((void*)function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes, NULL, kernel_params, NULL);
}

static int core_rocm_hip_device_synchronize() {
	static hipDeviceSynchronize_t cached = NULL;
	hipDeviceSynchronize_t fn = cached;
	if (fn == NULL) {
		fn = (hipDeviceSynchronize_t)core_rocm_hip_symbol("hipDeviceSynchronize");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100021;
	}
	return fn();
}

static int core_rocm_hip_host_malloc_mapped(uintptr_t* host_out, uintptr_t* device_out, size_t size) {
	static hipHostMalloc_t cached_malloc = NULL;
	static hipHostGetDevicePointer_t cached_pointer = NULL;
	static hipHostFree_t cached_free = NULL;
	hipHostMalloc_t malloc_fn = cached_malloc;
	hipHostGetDevicePointer_t pointer_fn = cached_pointer;
	hipHostFree_t free_fn = cached_free;
	if (malloc_fn == NULL) {
		malloc_fn = (hipHostMalloc_t)core_rocm_hip_symbol("hipHostMalloc");
		if (malloc_fn != NULL) {
			cached_malloc = malloc_fn;
		}
	}
	if (pointer_fn == NULL) {
		pointer_fn = (hipHostGetDevicePointer_t)core_rocm_hip_symbol("hipHostGetDevicePointer");
		if (pointer_fn != NULL) {
			cached_pointer = pointer_fn;
		}
	}
	if (free_fn == NULL) {
		free_fn = (hipHostFree_t)core_rocm_hip_symbol("hipHostFree");
		if (free_fn != NULL) {
			cached_free = free_fn;
		}
	}
	if (malloc_fn == NULL || pointer_fn == NULL || free_fn == NULL) {
		return -100013;
	}
	void* host = NULL;
	int rc = malloc_fn(&host, size, 0x40000002);
	if (rc != 0) {
		return rc;
	}
	void* device = NULL;
	rc = pointer_fn(&device, host, 0);
	if (rc != 0) {
		free_fn(host);
		return rc;
	}
	*host_out = (uintptr_t)host;
	*device_out = (uintptr_t)device;
	return 0;
}

static core_rocm_hip_uintptr2_result core_rocm_hip_host_malloc_mapped_result(size_t size) {
	core_rocm_hip_uintptr2_result result = {0, 0, 0};
	uintptr_t host = 0;
	uintptr_t device = 0;
	result.rc = core_rocm_hip_host_malloc_mapped(&host, &device, size);
	result.first = host;
	result.second = device;
	return result;
}

static int core_rocm_hip_host_free(uintptr_t host) {
	static hipHostFree_t cached = NULL;
	hipHostFree_t fn = cached;
	if (fn == NULL) {
		fn = (hipHostFree_t)core_rocm_hip_symbol("hipHostFree");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100014;
	}
	return fn((void*)host);
}

static int core_rocm_hip_host_malloc_pinned(uintptr_t* host_out, size_t size) {
	static hipHostMalloc_t cached = NULL;
	hipHostMalloc_t malloc_fn = cached;
	if (malloc_fn == NULL) {
		malloc_fn = (hipHostMalloc_t)core_rocm_hip_symbol("hipHostMalloc");
		if (malloc_fn != NULL) {
			cached = malloc_fn;
		}
	}
	if (malloc_fn == NULL) {
		return -100016;
	}
	void* host = NULL;
	int rc = malloc_fn(&host, size, 0);
	if (rc != 0) {
		return rc;
	}
	*host_out = (uintptr_t)host;
	return 0;
}

static core_rocm_hip_uintptr2_result core_rocm_hip_host_malloc_pinned_result(size_t size) {
	core_rocm_hip_uintptr2_result result = {0, 0, 0};
	uintptr_t host = 0;
	result.rc = core_rocm_hip_host_malloc_pinned(&host, size);
	result.first = host;
	return result;
}

static int core_rocm_hip_event_create(uintptr_t* out) {
	static hipEventCreateWithFlags_t cached = NULL;
	hipEventCreateWithFlags_t fn = cached;
	if (fn == NULL) {
		fn = (hipEventCreateWithFlags_t)core_rocm_hip_symbol("hipEventCreateWithFlags");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100017;
	}
	void* event = NULL;
	int rc = fn(&event, 0x2);
	*out = (uintptr_t)event;
	return rc;
}

static core_rocm_hip_uintptr2_result core_rocm_hip_event_create_result() {
	core_rocm_hip_uintptr2_result result = {0, 0, 0};
	uintptr_t event = 0;
	result.rc = core_rocm_hip_event_create(&event);
	result.first = event;
	return result;
}

static int core_rocm_hip_event_record(uintptr_t event) {
	static hipEventRecord_t cached = NULL;
	hipEventRecord_t fn = cached;
	if (fn == NULL) {
		fn = (hipEventRecord_t)core_rocm_hip_symbol("hipEventRecord");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100018;
	}
	return fn((void*)event, NULL);
}

static int core_rocm_hip_event_synchronize(uintptr_t event) {
	static hipEventSynchronize_t cached = NULL;
	hipEventSynchronize_t fn = cached;
	if (fn == NULL) {
		fn = (hipEventSynchronize_t)core_rocm_hip_symbol("hipEventSynchronize");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100019;
	}
	return fn((void*)event);
}

static int core_rocm_hip_event_destroy(uintptr_t event) {
	static hipEventDestroy_t cached = NULL;
	hipEventDestroy_t fn = cached;
	if (fn == NULL) {
		fn = (hipEventDestroy_t)core_rocm_hip_symbol("hipEventDestroy");
		if (fn != NULL) {
			cached = fn;
		}
	}
	if (fn == NULL) {
		return -100020;
	}
	return fn((void*)event);
}
*/
import "C"

import (
	"os"
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	corecgo "dappco.re/go/cgo"
)

type cgoHIPDriver struct {
	kernelModulePath string
}

func (cgoHIPDriver) rocmDefaultKVTensorPool() {}

const rocmHIPPinnedHostCopySupported = true

var cgoHIPAvailability = struct {
	sync.Once
	available bool
}{}

const (
	cgoHIPPoolMaxBufferBytes = 8 << 20
	cgoHIPPoolMaxTotalBytes  = 512 << 20
	cgoHIPPoolMaxPerSize     = 512
	cgoHIPPoolInitialPerSize = 8
	cgoHIPLaunchArgRingSize  = 16384
	cgoHIPAsyncCopyRingSize  = 8192
	cgoHIPAsyncCopyMaxBytes  = 1 << 20
)

type cgoHIPCachedModule struct {
	module    C.uintptr_t
	image     []byte
	scope     *corecgo.Scope
	functions map[string]C.uintptr_t
}

var cgoHIPModuleCache = struct {
	sync.Mutex
	modules map[string]*cgoHIPCachedModule
}{
	modules: map[string]*cgoHIPCachedModule{},
}

type cgoHIPFunctionCacheKey struct {
	module string
	kernel string
}

var cgoHIPFunctionCache sync.Map

var cgoHIPLaunchArgBuffer = struct {
	sync.Mutex
	pointer nativeDevicePointer
	host    unsafe.Pointer
	bytes   uint64
	mapped  bool
}{}

type cgoHIPLaunchArgSlot struct {
	pointer  nativeDevicePointer
	host     unsafe.Pointer
	event    C.uintptr_t
	bytes    uint64
	mapped   bool
	recorded bool
}

type cgoHIPLaunchArgLease struct {
	pointer    nativeDevicePointer
	syncBuffer bool
	asyncSlot  *cgoHIPLaunchArgSlot
	noEvent    bool
}

type cgoHIPLaunchArgMode struct {
	async  bool
	mapped bool
	events bool
}

var cgoHIPLaunchArgModeCache = struct {
	sync.Once
	mode cgoHIPLaunchArgMode
}{}

var cgoHIPLaunchArgRing = struct {
	sync.Mutex
	next    int
	wrapped bool
	slots   []cgoHIPLaunchArgSlot
}{
	slots: make([]cgoHIPLaunchArgSlot, cgoHIPLaunchArgRingSize),
}

type cgoHIPAsyncCopySlot struct {
	host     unsafe.Pointer
	event    C.uintptr_t
	bytes    uint64
	recorded bool
}

var cgoHIPAsyncCopyRing = struct {
	sync.Mutex
	next  int
	slots []cgoHIPAsyncCopySlot
}{
	slots: make([]cgoHIPAsyncCopySlot, cgoHIPAsyncCopyRingSize),
}

type cgoHIPMemoryPoolBucket struct {
	first nativeDevicePointer
	rest  []nativeDevicePointer
}

func (bucket cgoHIPMemoryPoolBucket) len() int {
	if bucket.first == 0 {
		return 0
	}
	return 1 + len(bucket.rest)
}

var cgoHIPMemoryPool = struct {
	sync.Mutex
	live      map[nativeDevicePointer]uint64
	free      map[uint64]cgoHIPMemoryPoolBucket
	freeBytes uint64
}{
	live: map[nativeDevicePointer]uint64{},
	free: map[uint64]cgoHIPMemoryPoolBucket{},
}

func newSystemHIPDriver() nativeHIPDriver {
	return cgoHIPDriver{kernelModulePath: hipKernelModulePath()}
}

func (cgoHIPDriver) Available() bool {
	cgoHIPAvailability.Do(func() {
		var count C.int
		if rc := C.core_rocm_hip_device_count(&count); rc == 0 && count > 0 {
			cgoHIPAvailability.available = true
		}
	})
	return cgoHIPAvailability.available
}

func (driver cgoHIPDriver) DeviceInfo() nativeDeviceInfo {
	var freeBytes C.size_t
	var totalBytes C.size_t
	if driver.Available() {
		_ = C.core_rocm_hip_set_device(0)
	}
	if rc := C.core_rocm_hip_mem_info(&freeBytes, &totalBytes); rc != 0 {
		if info, err := GetVRAMInfo(); err == nil {
			return nativeDeviceInfo{Name: "rocm", MemoryBytes: info.Total, FreeBytes: info.Free, Driver: "hip"}
		}
		return nativeDeviceInfo{Driver: "hip"}
	}
	var version C.int
	_ = C.core_rocm_hip_runtime_version(&version)
	return nativeDeviceInfo{
		Name:        "rocm",
		MemoryBytes: uint64(totalBytes),
		FreeBytes:   uint64(freeBytes),
		Driver:      core.Sprintf("hip:%d", int(version)),
	}
}

func (cgoHIPDriver) Malloc(size uint64) (nativeDevicePointer, error) {
	if size <= cgoHIPPoolMaxBufferBytes {
		cgoHIPMemoryPool.Lock()
		bucket := cgoHIPMemoryPool.free[size]
		if bucket.first != 0 {
			pointer := bucket.first
			if count := len(bucket.rest); count > 0 {
				bucket.first = bucket.rest[count-1]
				bucket.rest[count-1] = 0
				bucket.rest = bucket.rest[:count-1]
				cgoHIPMemoryPool.free[size] = bucket
			} else {
				bucket.first = 0
				cgoHIPMemoryPool.free[size] = bucket
			}
			cgoHIPMemoryPool.live[pointer] = size
			cgoHIPMemoryPool.freeBytes -= size
			cgoHIPMemoryPool.Unlock()
			return pointer, nil
		}
		cgoHIPMemoryPool.Unlock()
	}
	result := C.core_rocm_hip_malloc_result(C.size_t(size))
	if result.rc != 0 {
		return 0, hipReturnError("hipMalloc", int(result.rc))
	}
	pointer := nativeDevicePointer(result.first)
	cgoHIPMemoryPool.Lock()
	cgoHIPMemoryPool.live[pointer] = size
	cgoHIPMemoryPool.Unlock()
	return pointer, nil
}

func (cgoHIPDriver) Free(pointer nativeDevicePointer) error {
	if pointer == 0 {
		return nil
	}
	cgoHIPMemoryPool.Lock()
	size, tracked := cgoHIPMemoryPool.live[pointer]
	if tracked {
		delete(cgoHIPMemoryPool.live, pointer)
	}
	if tracked &&
		size <= cgoHIPPoolMaxBufferBytes &&
		cgoHIPMemoryPool.freeBytes+size <= cgoHIPPoolMaxTotalBytes &&
		cgoHIPMemoryPool.free[size].len() < cgoHIPPoolMaxPerSize {
		bucket := cgoHIPMemoryPool.free[size]
		if bucket.first == 0 {
			bucket.first = pointer
		} else {
			if bucket.rest == nil {
				bucket.rest = make([]nativeDevicePointer, 0, cgoHIPPoolInitialPerSize)
			}
			bucket.rest = append(bucket.rest, pointer)
		}
		cgoHIPMemoryPool.free[size] = bucket
		cgoHIPMemoryPool.freeBytes += size
		cgoHIPMemoryPool.Unlock()
		return nil
	}
	cgoHIPMemoryPool.Unlock()
	if os.Getenv("GO_ROCM_DISABLE_ASYNC_FREE") == "" {
		if rc := C.core_rocm_hip_free_async(C.uintptr_t(pointer)); rc == 0 {
			return nil
		}
	}
	if rc := C.core_rocm_hip_free(C.uintptr_t(pointer)); rc != 0 {
		return hipReturnError("hipFree", int(rc))
	}
	return nil
}

func (cgoHIPDriver) CopyHostToDevice(pointer nativeDevicePointer, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if rc := C.core_rocm_hip_memcpy_htod(C.uintptr_t(pointer), unsafe.Pointer(&data[0]), C.size_t(len(data))); rc != 0 {
		return hipReturnError("hipMemcpyHostToDevice", int(rc))
	}
	return nil
}

type nativeHIPPinnedHostToDevice interface {
	CopyPinnedHostToDevice(pointer nativeDevicePointer, host unsafe.Pointer, sizeBytes int) error
}

type nativeHIPLabeledPinnedHostToDevice interface {
	CopyPinnedHostToDeviceLabeled(pointer nativeDevicePointer, host unsafe.Pointer, sizeBytes int, operation, label string) error
}

func hipCopyPinnedHostToDevice(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte) error {
	return hipCopyPinnedHostToDeviceLabeled(driver, pointer, data, "", "")
}

func hipCopyPinnedHostToDeviceLabeled(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte, operation, label string) error {
	if len(data) == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.CopyPinnedHostToDevice", "device pointer is nil", nil)
	}
	if labeled, ok := driver.(nativeHIPLabeledPinnedHostToDevice); ok {
		var view core.PinnedView
		core.PinSlice(data, &view)
		defer view.Release()
		if err := labeled.CopyPinnedHostToDeviceLabeled(pointer, view.Ptr(), view.Bytes(), operation, label); err != nil {
			return err
		}
		runtime.KeepAlive(data)
		return nil
	}
	if pinned, ok := driver.(nativeHIPPinnedHostToDevice); ok {
		var view core.PinnedView
		core.PinSlice(data, &view)
		defer view.Release()
		if err := pinned.CopyPinnedHostToDevice(pointer, view.Ptr(), view.Bytes()); err != nil {
			return err
		}
		runtime.KeepAlive(data)
		return nil
	}
	if operation != "" || label != "" {
		return hipCopyHostToDeviceLabeled(driver, pointer, data, operation, label)
	}
	if err := hipCopyHostToDevice(driver, pointer, data); err != nil {
		return err
	}
	runtime.KeepAlive(data)
	return nil
}

func (cgoHIPDriver) CopyPinnedHostToDevice(pointer nativeDevicePointer, host unsafe.Pointer, sizeBytes int) error {
	if sizeBytes == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.CopyPinnedHostToDevice", "device pointer is nil", nil)
	}
	if host == nil {
		return core.E("rocm.hip.CopyPinnedHostToDevice", "host pointer is nil", nil)
	}
	if rc := C.core_rocm_hip_memcpy_htod(C.uintptr_t(pointer), host, C.size_t(sizeBytes)); rc != 0 {
		return hipReturnError("hipMemcpyHostToDevice", int(rc))
	}
	return nil
}

func (driver cgoHIPDriver) CopyHostToDeviceAsync(pointer nativeDevicePointer, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.CopyHostToDeviceAsync", "device pointer is nil", nil)
	}
	if os.Getenv("GO_ROCM_DISABLE_ASYNC_H2D") != "" || len(data) > cgoHIPAsyncCopyMaxBytes {
		return driver.CopyHostToDevice(pointer, data)
	}
	cgoHIPAsyncCopyRing.Lock()
	defer cgoHIPAsyncCopyRing.Unlock()
	slotIndex := cgoHIPAsyncCopyRing.next
	cgoHIPAsyncCopyRing.next = (cgoHIPAsyncCopyRing.next + 1) % len(cgoHIPAsyncCopyRing.slots)
	slot := &cgoHIPAsyncCopyRing.slots[slotIndex]
	if slot.recorded {
		if rc := C.core_rocm_hip_event_synchronize(slot.event); rc != 0 {
			return hipReturnError("hipEventSynchronize", int(rc))
		}
		slot.recorded = false
	}
	if slot.host == nil || slot.bytes < uint64(len(data)) {
		if err := driver.resizeAsyncCopySlot(slot, uint64(len(data))); err != nil {
			return core.E("rocm.hip.CopyHostToDeviceAsync", "allocate async copy staging slot", err)
		}
	}
	copy(unsafe.Slice((*byte)(slot.host), int(slot.bytes)), data)
	if rc := C.core_rocm_hip_memcpy_htod_async(C.uintptr_t(pointer), slot.host, C.size_t(len(data))); rc != 0 {
		return hipReturnError("hipMemcpyHostToDeviceAsync", int(rc))
	}
	if rc := C.core_rocm_hip_event_record(slot.event); rc != 0 {
		return hipReturnError("hipEventRecord", int(rc))
	}
	slot.recorded = true
	return nil
}

func (cgoHIPDriver) MemsetAsync(pointer nativeDevicePointer, value byte, size uint64) error {
	if size == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.MemsetAsync", "device pointer is nil", nil)
	}
	if rc := C.core_rocm_hip_memset_async(C.uintptr_t(pointer), C.int(value), C.size_t(size)); rc != 0 {
		return hipReturnError("hipMemsetAsync", int(rc))
	}
	return nil
}

func (cgoHIPDriver) CopyDeviceToHost(pointer nativeDevicePointer, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if rc := C.core_rocm_hip_memcpy_dtoh(unsafe.Pointer(&data[0]), C.uintptr_t(pointer), C.size_t(len(data))); rc != 0 {
		return hipReturnError("hipMemcpyDeviceToHost", int(rc))
	}
	return nil
}

func (cgoHIPDriver) CopyDeviceToHostUint64(pointer nativeDevicePointer) (uint64, error) {
	if pointer == 0 {
		return 0, nil
	}
	result := C.core_rocm_hip_memcpy_dtoh_u64(C.uintptr_t(pointer))
	if result.rc != 0 {
		return 0, hipReturnError("hipMemcpyDeviceToHost", int(result.rc))
	}
	return uint64(result.value), nil
}

func (cgoHIPDriver) CopyDeviceToHostUint32(pointer nativeDevicePointer) (uint32, error) {
	if pointer == 0 {
		return 0, nil
	}
	result := C.core_rocm_hip_memcpy_dtoh_u32(C.uintptr_t(pointer))
	if result.rc != 0 {
		return 0, hipReturnError("hipMemcpyDeviceToHost", int(result.rc))
	}
	return uint32(result.value), nil
}

func (driver cgoHIPDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	if !driver.Available() {
		return core.E("rocm.hip.LaunchKernel", "HIP driver is not available", nil)
	}
	modulePath := driver.kernelModulePath
	if modulePath == "" {
		modulePath = hipKernelModulePath()
	}
	if modulePath == "" {
		return core.E("rocm.hip.LaunchKernel", "kernel module sidecar or "+hipKernelModuleEnv+" is not set; native HIP kernels are not linked yet", nil)
	}
	function, err := cgoHIPCachedFunction(modulePath, config.Name)
	if err != nil {
		return err
	}

	args, err := driver.launchArgPointer(config.Args)
	hipReleaseLaunchPacket(config.Args)
	if err != nil {
		return err
	}
	if rc := C.core_rocm_hip_module_launch_kernel(
		function,
		C.uint(config.GridX),
		C.uint(config.GridY),
		C.uint(config.GridZ),
		C.uint(config.BlockX),
		C.uint(config.BlockY),
		C.uint(config.BlockZ),
		C.uint(config.SharedMemBytes),
		C.uintptr_t(args.pointer),
	); rc != 0 {
		_ = args.finish(false)
		return hipReturnError("hipModuleLaunchKernel", int(rc))
	}
	if err := args.finish(true); err != nil {
		return err
	}
	return nil
}

func (driver cgoHIPDriver) PrewarmKernelFunctions(kernelNames []string) {
	if !driver.Available() {
		return
	}
	modulePath := driver.kernelModulePath
	if modulePath == "" {
		modulePath = hipKernelModulePath()
	}
	if modulePath == "" {
		return
	}
	for _, name := range kernelNames {
		if name == "" {
			continue
		}
		_, _ = cgoHIPCachedFunction(modulePath, name)
	}
}

func (driver cgoHIPDriver) launchArgPointer(args []byte) (cgoHIPLaunchArgLease, error) {
	if cgoHIPLaunchArgModeConfig().async {
		return driver.launchArgPointerAsync(args)
	}
	return driver.launchArgPointerSync(args)
}

func (driver cgoHIPDriver) launchArgPointerSync(args []byte) (cgoHIPLaunchArgLease, error) {
	cgoHIPLaunchArgBuffer.Lock()
	want := uint64(len(args))
	if want < 256 {
		want = 256
	}
	if cgoHIPLaunchArgBuffer.pointer == 0 || cgoHIPLaunchArgBuffer.bytes < want {
		host, pointer, mapped, err := driver.allocateLaunchArgBuffer(want)
		if err != nil {
			cgoHIPLaunchArgBuffer.Unlock()
			return cgoHIPLaunchArgLease{}, core.E("rocm.hip.LaunchKernel", "allocate kernel argument packet", err)
		}
		previous := cgoHIPLaunchArgBuffer.pointer
		previousHost := cgoHIPLaunchArgBuffer.host
		previousMapped := cgoHIPLaunchArgBuffer.mapped
		cgoHIPLaunchArgBuffer.pointer = pointer
		cgoHIPLaunchArgBuffer.host = host
		cgoHIPLaunchArgBuffer.bytes = want
		cgoHIPLaunchArgBuffer.mapped = mapped
		if previous != 0 {
			_ = driver.freeLaunchArgBuffer(previousHost, previous, previousMapped)
		}
	}
	if cgoHIPLaunchArgBuffer.mapped {
		copy(unsafe.Slice((*byte)(cgoHIPLaunchArgBuffer.host), int(cgoHIPLaunchArgBuffer.bytes)), args)
	} else {
		if err := driver.CopyHostToDevice(cgoHIPLaunchArgBuffer.pointer, args); err != nil {
			cgoHIPLaunchArgBuffer.Unlock()
			return cgoHIPLaunchArgLease{}, core.E("rocm.hip.LaunchKernel", "copy kernel argument packet", err)
		}
	}
	return cgoHIPLaunchArgLease{pointer: cgoHIPLaunchArgBuffer.pointer, syncBuffer: true}, nil
}

func (driver cgoHIPDriver) launchArgPointerAsync(args []byte) (cgoHIPLaunchArgLease, error) {
	cgoHIPLaunchArgRing.Lock()
	syncOnWrap := !cgoHIPLaunchArgEventsEnabled()
	slotIndex := cgoHIPLaunchArgRing.next
	cgoHIPLaunchArgRing.next = (cgoHIPLaunchArgRing.next + 1) % len(cgoHIPLaunchArgRing.slots)
	if cgoHIPLaunchArgRing.next == 0 {
		cgoHIPLaunchArgRing.wrapped = true
	}
	if syncOnWrap && cgoHIPLaunchArgRing.wrapped && slotIndex == 0 {
		if rc := C.core_rocm_hip_device_synchronize(); rc != 0 {
			cgoHIPLaunchArgRing.Unlock()
			return cgoHIPLaunchArgLease{}, hipReturnError("hipDeviceSynchronize", int(rc))
		}
		for index := range cgoHIPLaunchArgRing.slots {
			cgoHIPLaunchArgRing.slots[index].recorded = false
		}
	}
	slot := &cgoHIPLaunchArgRing.slots[slotIndex]
	if !syncOnWrap && slot.recorded {
		if rc := C.core_rocm_hip_event_synchronize(slot.event); rc != 0 {
			cgoHIPLaunchArgRing.Unlock()
			return cgoHIPLaunchArgLease{}, hipReturnError("hipEventSynchronize", int(rc))
		}
		slot.recorded = false
	}
	want := uint64(len(args))
	if want < 256 {
		want = 256
	}
	if slot.pointer == 0 || slot.bytes < want {
		if err := driver.resizeLaunchArgSlot(slot, want); err != nil {
			cgoHIPLaunchArgRing.Unlock()
			return cgoHIPLaunchArgLease{}, core.E("rocm.hip.LaunchKernel", "allocate async kernel argument packet", err)
		}
	}
	hostBytes := unsafe.Slice((*byte)(slot.host), int(slot.bytes))
	copy(hostBytes, args)
	if !slot.mapped {
		if rc := C.core_rocm_hip_memcpy_htod_async(C.uintptr_t(slot.pointer), slot.host, C.size_t(len(args))); rc != 0 {
			cgoHIPLaunchArgRing.Unlock()
			return cgoHIPLaunchArgLease{}, hipReturnError("hipMemcpyHostToDeviceAsync", int(rc))
		}
	}
	return cgoHIPLaunchArgLease{pointer: slot.pointer, asyncSlot: slot, noEvent: syncOnWrap}, nil
}

func cgoHIPLaunchArgEventsEnabled() bool {
	return cgoHIPLaunchArgModeConfig().events
}

func cgoHIPLaunchArgModeConfig() cgoHIPLaunchArgMode {
	cgoHIPLaunchArgModeCache.Do(func() {
		cgoHIPLaunchArgModeCache.mode = cgoHIPLaunchArgMode{
			async:  os.Getenv("GO_ROCM_DISABLE_ASYNC_LAUNCH_ARGS") == "" || os.Getenv("GO_ROCM_ENABLE_MAPPED_LAUNCH_ARGS") != "",
			mapped: os.Getenv("GO_ROCM_DISABLE_MAPPED_LAUNCH_ARGS") == "",
			events: os.Getenv("GO_ROCM_ENABLE_LAUNCH_ARG_EVENTS") != "",
		}
	})
	return cgoHIPLaunchArgModeCache.mode
}

func (lease cgoHIPLaunchArgLease) finish(success bool) error {
	if lease.asyncSlot != nil {
		defer cgoHIPLaunchArgRing.Unlock()
		if !success || lease.noEvent || lease.asyncSlot.event == 0 {
			return nil
		}
		if rc := C.core_rocm_hip_event_record(lease.asyncSlot.event); rc != 0 {
			return hipReturnError("hipEventRecord", int(rc))
		}
		lease.asyncSlot.recorded = true
		return nil
	}
	if lease.syncBuffer {
		defer cgoHIPLaunchArgBuffer.Unlock()
		if success {
			if rc := C.core_rocm_hip_device_synchronize(); rc != 0 {
				return hipReturnError("hipDeviceSynchronize", int(rc))
			}
		}
	}
	return nil
}

func (driver cgoHIPDriver) allocateLaunchArgBuffer(size uint64) (unsafe.Pointer, nativeDevicePointer, bool, error) {
	if cgoHIPLaunchArgModeConfig().mapped {
		result := C.core_rocm_hip_host_malloc_mapped_result(C.size_t(size))
		if result.rc == 0 {
			return unsafe.Pointer(uintptr(result.first)), nativeDevicePointer(result.second), true, nil
		}
	}
	pointer, err := driver.Malloc(size)
	if err != nil {
		return nil, 0, false, err
	}
	return nil, pointer, false, nil
}

func (driver cgoHIPDriver) resizeLaunchArgSlot(slot *cgoHIPLaunchArgSlot, size uint64) error {
	if slot == nil {
		return core.E("rocm.hip.LaunchKernel", "launch argument slot is nil", nil)
	}
	if slot.recorded {
		if rc := C.core_rocm_hip_event_synchronize(slot.event); rc != 0 {
			return hipReturnError("hipEventSynchronize", int(rc))
		}
		slot.recorded = false
	}
	if err := driver.freeLaunchArgSlot(slot); err != nil {
		return err
	}
	if cgoHIPLaunchArgModeConfig().mapped {
		result := C.core_rocm_hip_host_malloc_mapped_result(C.size_t(size))
		if result.rc == 0 {
			event := C.uintptr_t(0)
			if cgoHIPLaunchArgEventsEnabled() {
				eventResult := C.core_rocm_hip_event_create_result()
				if eventResult.rc != 0 {
					_ = C.core_rocm_hip_host_free(result.first)
					return hipReturnError("hipEventCreateWithFlags", int(eventResult.rc))
				}
				event = eventResult.first
			}
			slot.host = unsafe.Pointer(uintptr(result.first))
			slot.pointer = nativeDevicePointer(result.second)
			slot.event = event
			slot.bytes = size
			slot.mapped = true
			return nil
		}
	}
	hostResult := C.core_rocm_hip_host_malloc_pinned_result(C.size_t(size))
	if hostResult.rc != 0 {
		return hipReturnError("hipHostMalloc", int(hostResult.rc))
	}
	pointer, err := driver.Malloc(size)
	if err != nil {
		_ = C.core_rocm_hip_host_free(hostResult.first)
		return err
	}
	event := C.uintptr_t(0)
	if cgoHIPLaunchArgEventsEnabled() {
		eventResult := C.core_rocm_hip_event_create_result()
		if eventResult.rc != 0 {
			_ = C.core_rocm_hip_host_free(hostResult.first)
			_ = driver.Free(pointer)
			return hipReturnError("hipEventCreateWithFlags", int(eventResult.rc))
		}
		event = eventResult.first
	}
	slot.host = unsafe.Pointer(uintptr(hostResult.first))
	slot.pointer = pointer
	slot.event = event
	slot.bytes = size
	slot.mapped = false
	return nil
}

func (driver cgoHIPDriver) freeLaunchArgSlot(slot *cgoHIPLaunchArgSlot) error {
	if slot == nil {
		return nil
	}
	var lastErr error
	if slot.recorded && slot.event != 0 {
		if rc := C.core_rocm_hip_event_synchronize(slot.event); rc != 0 {
			lastErr = hipReturnError("hipEventSynchronize", int(rc))
		}
		slot.recorded = false
	}
	if slot.event != 0 {
		if rc := C.core_rocm_hip_event_destroy(slot.event); rc != 0 {
			lastErr = hipReturnError("hipEventDestroy", int(rc))
		}
		slot.event = 0
	}
	if slot.host != nil {
		if rc := C.core_rocm_hip_host_free(C.uintptr_t(uintptr(slot.host))); rc != 0 {
			lastErr = hipReturnError("hipHostFree", int(rc))
		}
		slot.host = nil
	}
	if slot.pointer != 0 && !slot.mapped {
		if err := driver.Free(slot.pointer); err != nil {
			lastErr = err
		}
	}
	slot.pointer = 0
	slot.mapped = false
	slot.bytes = 0
	return lastErr
}

func (driver cgoHIPDriver) resizeAsyncCopySlot(slot *cgoHIPAsyncCopySlot, size uint64) error {
	if slot == nil {
		return core.E("rocm.hip.CopyHostToDeviceAsync", "async copy slot is nil", nil)
	}
	if slot.recorded {
		if rc := C.core_rocm_hip_event_synchronize(slot.event); rc != 0 {
			return hipReturnError("hipEventSynchronize", int(rc))
		}
		slot.recorded = false
	}
	if err := driver.freeAsyncCopySlot(slot); err != nil {
		return err
	}
	hostResult := C.core_rocm_hip_host_malloc_pinned_result(C.size_t(size))
	if hostResult.rc != 0 {
		return hipReturnError("hipHostMalloc", int(hostResult.rc))
	}
	eventResult := C.core_rocm_hip_event_create_result()
	if eventResult.rc != 0 {
		_ = C.core_rocm_hip_host_free(hostResult.first)
		return hipReturnError("hipEventCreateWithFlags", int(eventResult.rc))
	}
	slot.host = unsafe.Pointer(uintptr(hostResult.first))
	slot.event = eventResult.first
	slot.bytes = size
	return nil
}

func (driver cgoHIPDriver) freeAsyncCopySlot(slot *cgoHIPAsyncCopySlot) error {
	if slot == nil {
		return nil
	}
	var lastErr error
	if slot.recorded && slot.event != 0 {
		if rc := C.core_rocm_hip_event_synchronize(slot.event); rc != 0 {
			lastErr = hipReturnError("hipEventSynchronize", int(rc))
		}
		slot.recorded = false
	}
	if slot.event != 0 {
		if rc := C.core_rocm_hip_event_destroy(slot.event); rc != 0 {
			lastErr = hipReturnError("hipEventDestroy", int(rc))
		}
		slot.event = 0
	}
	if slot.host != nil {
		if rc := C.core_rocm_hip_host_free(C.uintptr_t(uintptr(slot.host))); rc != 0 {
			lastErr = hipReturnError("hipHostFree", int(rc))
		}
		slot.host = nil
	}
	slot.bytes = 0
	return lastErr
}

func (driver cgoHIPDriver) freeLaunchArgBuffer(host unsafe.Pointer, pointer nativeDevicePointer, mapped bool) error {
	if pointer == 0 {
		return nil
	}
	if mapped {
		if host == nil {
			return nil
		}
		if rc := C.core_rocm_hip_host_free(C.uintptr_t(uintptr(host))); rc != 0 {
			return hipReturnError("hipHostFree", int(rc))
		}
		return nil
	}
	return driver.Free(pointer)
}

func cgoHIPCachedFunction(modulePath, kernelName string) (C.uintptr_t, error) {
	key := cgoHIPFunctionCacheKey{module: modulePath, kernel: kernelName}
	if cached, ok := cgoHIPFunctionCache.Load(key); ok {
		return cached.(C.uintptr_t), nil
	}
	cgoHIPModuleCache.Lock()
	defer cgoHIPModuleCache.Unlock()
	if cached, ok := cgoHIPFunctionCache.Load(key); ok {
		return cached.(C.uintptr_t), nil
	}
	module := cgoHIPModuleCache.modules[modulePath]
	if module == nil {
		loaded, err := cgoHIPLoadModule(modulePath)
		if err != nil {
			return 0, err
		}
		module = loaded
		cgoHIPModuleCache.modules[modulePath] = module
	}
	if function, ok := module.functions[kernelName]; ok {
		cgoHIPFunctionCache.Store(key, function)
		return function, nil
	}
	function, err := cgoHIPModuleFunction(module.module, kernelName)
	if err != nil {
		return 0, core.E("rocm.hip.LaunchKernel", "resolve kernel "+kernelName, err)
	}
	module.functions[kernelName] = function
	cgoHIPFunctionCache.Store(key, function)
	return function, nil
}

func cgoHIPLoadModule(modulePath string) (*cgoHIPCachedModule, error) {
	image, err := os.ReadFile(modulePath)
	if err != nil {
		return nil, core.E("rocm.hip.LaunchKernel", "read kernel module "+modulePath, err)
	}
	if len(image) == 0 {
		return nil, core.E("rocm.hip.LaunchKernel", "kernel module is empty "+modulePath, nil)
	}
	scope := corecgo.NewScope()
	imageView := corecgo.PinIn(scope, image)

	moduleResult := C.core_rocm_hip_module_load_data_result(imageView.Ptr())
	if moduleResult.rc != 0 {
		scope.FreeAll()
		return nil, hipReturnError("hipModuleLoadData", int(moduleResult.rc))
	}
	return &cgoHIPCachedModule{module: moduleResult.first, image: image, scope: scope, functions: map[string]C.uintptr_t{}}, nil
}

func cgoHIPModuleFunction(module C.uintptr_t, kernelName string) (C.uintptr_t, error) {
	cName := corecgo.CStringPtr(kernelName)
	defer corecgo.Free(cName)
	functionResult := C.core_rocm_hip_module_get_function_result(module, (*C.char)(cName))
	if functionResult.rc != 0 {
		return 0, hipReturnError("hipModuleGetFunction", int(functionResult.rc))
	}
	return functionResult.first, nil
}

func hipReturnError(op string, code int) error {
	return core.E("rocm.hip."+op, core.Sprintf("HIP returned %d", code), nil)
}
