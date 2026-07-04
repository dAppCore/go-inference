// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package safetensors

/*
#cgo CFLAGS: -O3 -march=armv8-a+fp16
#include <arm_neon.h>
#include <stdint.h>

// neon_float16_to_float32 converts n contiguous IEEE-754 half precision values
// at src into n contiguous IEEE-754 single precision values at dst using the
// ARM64 FCVTL V.4S, V.4H instruction emitted by the vcvt_f32_f16 intrinsic.
// The tail (n % 4) is handled with vget_lane / vcvt scalar so that any input
// length, including <4, is supported. Output is bit-identical to the scalar
// Float16ToFloat32 reference for every non-NaN input (normals, subnormals,
// +/-0, +/-Inf). For NaN inputs the ARMv8 FCVTL instruction canonicalises
// signalling NaNs to quiet NaNs by setting the most-significant fraction bit,
// which is the IEEE-754-2008 hardware default and matches what x86 VCVTPH2PS
// does. No consumer in this tree distinguishes sNaN from qNaN (all use
// math.IsNaN), so the canonicalisation is an unobservable improvement; the
// equivalence is asserted in TestFloat16ToFloat32_NEONParity_BitExact.
static inline void neon_float16_to_float32(const uint16_t* src, float* dst, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
    for (; i < n; i++) {
        uint16x4_t lane = vld1_dup_u16(src + i);
        float16x4_t h = vreinterpret_f16_u16(lane);
        float32x4_t f = vcvt_f32_f16(h);
        dst[i] = vgetq_lane_f32(f, 0);
    }
}
*/
import "C"

import "unsafe"

// float16SliceToFloat32 converts n half-precision values from src into the
// first n elements of dst using a NEON FCVTL inner loop. The function name
// is dst-first to match Go's copy/append idiom. Caller guarantees
// len(src) >= n and len(dst) >= n.
//
// Build tag selection: this file is compiled only on darwin/arm64. All other
// platforms use float16_scalar.go which emits the scalar Go loop.
//
// Numerical guarantee: bit-exact against scalar Float16ToFloat32 for the
// full uint16 range — verified in TestFloat16ToFloat32_NEONParity_BitExact.
func float16SliceToFloat32(src []uint16, dst []float32, n int) {
	if n == 0 {
		return
	}
	C.neon_float16_to_float32(
		(*C.uint16_t)(unsafe.Pointer(unsafe.SliceData(src))),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(dst))),
		C.int(n),
	)
}
