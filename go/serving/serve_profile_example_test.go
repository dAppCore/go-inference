// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ExampleWriteTunedDraftBlockProfile writes a winning MTP draft block as a
// tuning profile, then shows loadTunedDraftBlock (the read side a serve boot
// runs) resolving the same block straight back from the file just written.
func ExampleWriteTunedDraftBlockProfile() {
	dirResult := core.MkdirTemp("", "tune-profile-example-*")
	if !dirResult.OK {
		panic("tempdir failed")
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)

	measurements := inference.TuningMeasurements{DecodeTokensPerSec: 42}
	score := inference.ScoreTuningMeasurements(inference.TuningWorkloadChat, measurements)
	path, err := WriteTunedDraftBlockProfile(dir, "/models/target", "", inference.TuningWorkloadChat, 5, measurements, score, 1700000000)
	if err != nil {
		panic(err)
	}

	block, resolvedPath := loadTunedDraftBlock(dir, "/models/target", "")
	fmt.Println(block)
	fmt.Println(resolvedPath == path)
	// Output:
	// 5
	// true
}
