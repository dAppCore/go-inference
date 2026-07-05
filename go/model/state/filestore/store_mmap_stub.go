// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin || linux || freebsd || netbsd || openbsd)

package filestore

func (s *Store) ensureMappedRegionLocked() error {
	return errMappedRegionInvalid
}

func (s *Store) unmapRegionLocked() {}
