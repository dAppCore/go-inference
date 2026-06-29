// SPDX-Licence-Identifier: EUPL-1.2

// filestore shared helpers: bounded payload writer, full-write loop, context gate and small numeric/result utilities.
package filestore

import (
	"context"
	stdio "io"

	core "dappco.re/go"
)

type limitedPayloadWriter struct {
	file      *core.OSFile
	remaining int
}

func (w *limitedPayloadWriter) Write(data []byte) (int, error) {
	if len(data) > w.remaining {
		return 0, errPayloadOversize
	}
	n, err := w.file.Write(data)
	w.remaining -= n
	if err != nil {
		return n, err
	}
	if n != len(data) {
		return n, stdio.ErrShortWrite
	}
	return n, nil
}

func writeAll(file stdio.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := file.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return stdio.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}

func checkContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func intFromUint64(value uint64, label string) (int, error) {
	max := uint64(maxInt())
	if value > max {
		return 0, core.NewError("state file store " + label + " is too large")
	}
	return int(value), nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
