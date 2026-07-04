// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"

	"dappco.re/go/inference/artifact"
)

// ExportArtifacts captures the session state and exports it as local
// artifacts via dappco.re/go/inference/artifact.
//
//	record, err := session.ExportArtifacts(artifact.Options{Model: "gemma3-1b"})
func (s *Session) ExportArtifacts(opts artifact.Options) (*artifact.Record, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return artifact.Export(context.Background(), snapshot, opts)
}
