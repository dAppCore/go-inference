//go:build linux && amd64

package hip

import (
	"strconv"
	"sync"
	"syscall"

	core "dappco.re/go"
)

var rocmVRAMInfoSysfsCache = struct {
	sync.Mutex
	usedPath string
	total    uint64
}{}

func warmROCmVRAMInfoCache() {
	_, _ = GetVRAMInfo()
}

//	info, err := GetVRAMInfo()
//	fmt.Printf("%d MiB free\n", info.Free>>20)
//
// GetVRAMInfo reads VRAM usage for the discrete GPU from sysfs. It identifies
// the dGPU by selecting the card with the largest VRAM total, which avoids
// hardcoding card numbers (e.g. card0=iGPU, card1=dGPU on Ryzen).
//
// Note: total and used are read non-atomically from sysfs; transient
// inconsistencies are possible under heavy allocation churn.
func GetVRAMInfo() (
	VRAMInfo,
	error,
) {
	rocmVRAMInfoSysfsCache.Lock()
	if rocmVRAMInfoSysfsCache.usedPath != "" && rocmVRAMInfoSysfsCache.total > 0 {
		usedPath := rocmVRAMInfoSysfsCache.usedPath
		total := rocmVRAMInfoSysfsCache.total
		rocmVRAMInfoSysfsCache.Unlock()
		return readCachedVRAMInfo(usedPath, total)
	}
	rocmVRAMInfoSysfsCache.Unlock()

	cards := core.PathGlob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total")
	if len(cards) == 0 {
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "no GPU VRAM info found in sysfs", nil)
	}

	var bestDir string
	var bestTotal uint64

	for _, totalPath := range cards {
		total, err := readSysfsUint64(totalPath)
		if err != nil {
			continue
		}
		if total > bestTotal {
			bestTotal = total
			bestDir = core.PathDir(totalPath)
		}
	}

	if bestDir == "" {
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "no readable VRAM sysfs entries", nil)
	}

	usedPath := core.PathJoin(bestDir, "mem_info_vram_used")
	used, err := readSysfsUint64(usedPath)
	if err != nil {
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "read vram used", err)
	}

	rocmVRAMInfoSysfsCache.Lock()
	if rocmVRAMInfoSysfsCache.usedPath == "" {
		rocmVRAMInfoSysfsCache.usedPath = usedPath
		rocmVRAMInfoSysfsCache.total = bestTotal
	}
	rocmVRAMInfoSysfsCache.Unlock()

	return vramInfoFromTotalUsed(bestTotal, used), nil
}

func readCachedVRAMInfo(usedPath string, total uint64) (VRAMInfo, error) {
	used, err := readSysfsUint64(usedPath)
	if err != nil {
		rocmVRAMInfoSysfsCache.Lock()
		if rocmVRAMInfoSysfsCache.usedPath == usedPath {
			rocmVRAMInfoSysfsCache.usedPath = ""
			rocmVRAMInfoSysfsCache.total = 0
		}
		rocmVRAMInfoSysfsCache.Unlock()
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "read cached vram used", err)
	}
	return vramInfoFromTotalUsed(total, used), nil
}

func vramInfoFromTotalUsed(total, used uint64) VRAMInfo {
	free := uint64(0)
	if total > used {
		free = total - used
	}

	return VRAMInfo{
		Total: total,
		Used:  used,
		Free:  free,
	}
}

func readSysfsUint64(path string) (
	uint64,
	error,
) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_CLOEXEC, 0)
	if err != nil {
		return 0, err
	}
	defer syscall.Close(fd)
	var buf [64]byte
	count, err := syscall.Read(fd, buf[:])
	if count <= 0 {
		if err != nil {
			return 0, err
		}
		return 0, strconv.ErrSyntax
	}
	var value uint64
	sawDigit := false
	for _, b := range buf[:count] {
		if b >= '0' && b <= '9' {
			digit := uint64(b - '0')
			if value > (^uint64(0)-digit)/10 {
				return 0, strconv.ErrRange
			}
			value = value*10 + digit
			sawDigit = true
			continue
		}
		if sawDigit {
			break
		}
		if b == ' ' || b == '\n' || b == '\r' || b == '\t' {
			continue
		}
		return 0, strconv.ErrSyntax
	}
	if !sawDigit {
		return 0, strconv.ErrSyntax
	}
	return value, nil
}
