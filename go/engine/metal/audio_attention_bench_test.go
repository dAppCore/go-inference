// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkAudioAttention10x128(b *testing.B) {
	requireNativeRuntime(b)

	const hid, H, D, chunk, past, future, T = 128, 4, 32, 4, 2, 1, 10
	hd, P := H*D, past+1
	weights := &AudioAttentionWeights{
		QProj:         toBF16Bytes(syntheticFloat32(hd*hid, 3)),
		KProj:         toBF16Bytes(syntheticFloat32(hd*hid, 5)),
		VProj:         toBF16Bytes(syntheticFloat32(hd*hid, 7)),
		Post:          toBF16Bytes(syntheticFloat32(hid*hd, 9)),
		RelativeKProj: toBF16Bytes(syntheticFloat32(hd*hid, 11)),
		QScalePerDim:  syntheticFloat32(D, 13),
		PosEmbed:      syntheticFloat32(P*hid, 15),
		PosCount:      P,
	}
	cfg := AudioConfig{
		Hidden: hid, NumHeads: H, HeadDim: D, ChunkSize: chunk,
		PastHorizon: past, FutureHorizon: future,
		KScale: 0.5, LogitCap: 50, InvalidLogit: -1e9,
	}
	x := toBF16Bytes(syntheticFloat32(T*hid, 17))
	b.SetBytes(int64(len(x)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AudioAttention(x, weights, cfg); err != nil {
			b.Fatal(err)
		}
	}
}
