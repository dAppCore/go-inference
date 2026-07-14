// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import "testing"

func BenchmarkHIPMoECombineNormsLaunchArgsBinaryInto_GemmaHidden2816(b *testing.B) {
	args := hipMoECombineNormsLaunchArgs{
		LocalInputPointer:    11,
		LocalWeightPointer:   12,
		ExpertInputPointer:   21,
		ExpertWeightPointer:  22,
		OutputPointer:        31,
		Count:                2816,
		LocalInputBytes:      2816 * 4,
		LocalWeightBytes:     2816 * 2,
		ExpertInputBytes:     2816 * 4,
		ExpertWeightBytes:    2816 * 4,
		OutputBytes:          2816 * 4,
		LocalEpsilon:         1e-6,
		LocalWeightEncoding:  hipRMSNormWeightEncodingBF16,
		LocalFlags:           hipRMSNormLaunchFlagAddUnitWeight,
		ExpertEpsilon:        1e-6,
		ExpertWeightEncoding: hipRMSNormWeightEncodingF32,
	}
	var scratch [hipMoECombineNormsLaunchArgsBytes]byte
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		payload, err := args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatalf("MoE combine norms launch args: %v", err)
		}
		if len(payload) != hipMoECombineNormsLaunchArgsBytes {
			b.Fatalf("MoE combine norms launch bytes len = %d, want %d", len(payload), hipMoECombineNormsLaunchArgsBytes)
		}
	}
}
