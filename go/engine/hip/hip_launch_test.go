// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"testing"

	core "dappco.re/go"
)

type hipKernelBatchTestDriver struct {
	*fakeHIPDriver
	batches [][]hipKernelLaunchConfig
}

func (driver *hipKernelBatchTestDriver) LaunchKernelBatch(configs []hipKernelLaunchConfig) error {
	batch := append([]hipKernelLaunchConfig(nil), configs...)
	for index := range batch {
		batch[index].Args = append([]byte(nil), batch[index].Args...)
	}
	driver.batches = append(driver.batches, batch)
	return nil
}

func TestHIPLaunchKernelContext_BatchesUntilFlush_Good(t *testing.T) {
	driver := &hipKernelBatchTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	ctx, batch := hipBeginKernelLaunchBatch(context.Background(), driver)
	if batch == nil {
		t.Fatal("batch-capable driver did not start a launch batch")
	}
	first := hipKernelLaunchConfig{Name: "first", Args: []byte{1}, GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	second := hipKernelLaunchConfig{Name: "second", Args: []byte{2}, GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	core.RequireNoError(t, hipLaunchKernelContext(ctx, driver, first))
	core.RequireNoError(t, hipLaunchKernelContext(ctx, driver, second))
	core.AssertEqual(t, 0, len(driver.launches))
	core.AssertEqual(t, 0, len(driver.batches))

	core.RequireNoError(t, batch.Flush())
	core.AssertEqual(t, 1, len(driver.batches))
	core.AssertEqual(t, 2, len(driver.batches[0]))
	core.AssertEqual(t, "first", driver.batches[0][0].Name)
	core.AssertEqual(t, "second", driver.batches[0][1].Name)
}

func TestHIPBeginKernelLaunchBatch_FallsBackForOrdinaryDriver_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	ctx, batch := hipBeginKernelLaunchBatch(context.Background(), driver)
	if batch != nil {
		t.Fatal("ordinary driver unexpectedly started a launch batch")
	}
	config := hipKernelLaunchConfig{Name: "ordinary", Args: []byte{1}, GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	core.RequireNoError(t, hipLaunchKernelContext(ctx, driver, config))
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, "ordinary", driver.launches[0].Name)
}

func TestHIPKernelLaunchBatch_DiscardDoesNotLaunch_Good(t *testing.T) {
	driver := &hipKernelBatchTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	ctx, batch := hipBeginKernelLaunchBatch(context.Background(), driver)
	config := hipKernelLaunchConfig{Name: "discarded", Args: []byte{1}, GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	core.RequireNoError(t, hipLaunchKernelContext(ctx, driver, config))
	batch.Discard()
	core.AssertEqual(t, 0, len(driver.batches))
}

func TestHIPLaunchKernelContext_SnapshotsReusedArgs_Good(t *testing.T) {
	driver := &hipKernelBatchTestDriver{fakeHIPDriver: &fakeHIPDriver{available: true}}
	ctx, batch := hipBeginKernelLaunchBatch(context.Background(), driver)
	shared := []byte{1}
	config := hipKernelLaunchConfig{Name: "shared", Args: shared, GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	core.RequireNoError(t, hipLaunchKernelContext(ctx, driver, config))
	shared[0] = 2
	core.RequireNoError(t, hipLaunchKernelContext(ctx, driver, config))
	core.RequireNoError(t, batch.Flush())
	core.AssertEqual(t, byte(1), driver.batches[0][0].Args[0])
	core.AssertEqual(t, byte(2), driver.batches[0][1].Args[0])
}

func TestHIPDecodeKernelBatchEnabled_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DECODE_KERNEL_BATCH", "")
	core.AssertEqual(t, true, hipDecodeKernelBatchEnabled())
	t.Setenv("GO_ROCM_DISABLE_DECODE_KERNEL_BATCH", "1")
	core.AssertEqual(t, false, hipDecodeKernelBatchEnabled())
}
