// SPDX-Licence-Identifier: EUPL-1.2

package engine

// MemoryReporter is an optional [TokenModel] capability: an engine that can
// read its device's allocation state reports it, and setMetrics folds the
// numbers into inference.GenerateMetrics after every generation. Engines
// without the capability leave PeakMemoryBytes/ActiveMemoryBytes at zero —
// honestly absent, never fabricated (mantis #1843).
type MemoryReporter interface {
	// ActiveMemoryBytes is the device's allocated bytes right now.
	ActiveMemoryBytes() uint64
	// PeakMemoryBytes is the device's allocation high-water since the current
	// operation began (engines reset the watermark at prefill entry). The
	// number is device-global: concurrent generations on one device share a
	// watermark, because allocation is a device-wide fact.
	PeakMemoryBytes() uint64
}
