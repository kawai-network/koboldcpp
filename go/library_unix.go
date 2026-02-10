//go:build !windows

package koboldcpp

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// loadLibraryPlatform loads the library using Unix-specific purego APIs
func (k *KoboldCpp) loadLibraryPlatform(libPath string) (uintptr, error) {
	handle, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return 0, fmt.Errorf("failed to load library: %w", err)
	}
	return handle, nil
}

// dlsymPlatform gets a symbol from the library using Unix-specific purego APIs
func dlsymPlatform(handle uintptr, symbol string) (uintptr, error) {
	return purego.Dlsym(handle, symbol)
}
