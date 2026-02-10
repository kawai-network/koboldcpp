//go:build windows

package koboldcpp

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLoadLibraryW = kernel32.NewProc("LoadLibraryW")
	procGetProcAddr  = kernel32.NewProc("GetProcAddress")
)

// loadLibraryPlatform loads the library using Windows-specific APIs
func (k *KoboldCpp) loadLibraryPlatform(libPath string) (uintptr, error) {
	// Convert to UTF-16 for Windows API
	libPathUTF16, err := syscall.UTF16PtrFromString(libPath)
	if err != nil {
		return 0, fmt.Errorf("failed to convert path to UTF-16: %w", err)
	}

	// Load library using Windows LoadLibraryW
	handle, _, err := procLoadLibraryW.Call(uintptr(unsafe.Pointer(libPathUTF16)))
	if handle == 0 {
		return 0, fmt.Errorf("failed to load library: %w", err)
	}

	return handle, nil
}

// dlsymPlatform gets a symbol from the library using Windows-specific APIs
func dlsymPlatform(handle uintptr, symbol string) (uintptr, error) {
	// Convert symbol name to C string
	symbolPtr, err := syscall.BytePtrFromString(symbol)
	if err != nil {
		return 0, fmt.Errorf("failed to convert symbol name: %w", err)
	}

	// Get procedure address
	addr, _, err := procGetProcAddr.Call(handle, uintptr(unsafe.Pointer(symbolPtr)))
	if addr == 0 {
		return 0, fmt.Errorf("symbol not found: %s (%w)", symbol, err)
	}

	return addr, nil
}
