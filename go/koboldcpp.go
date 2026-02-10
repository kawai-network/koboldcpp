package koboldcpp

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	// ErrLibraryNotLoaded is returned when attempting to use functions before loading the library
	ErrLibraryNotLoaded = errors.New("koboldcpp library not loaded")
)

// KoboldCpp represents the main interface to the KoboldCpp library
type KoboldCpp struct {
	handle   uintptr
	libPath  string
	isLoaded bool
}

// LibraryVariant represents different library variants
type LibraryVariant string

const (
	LibDefault        LibraryVariant = "default"
	LibFailsafe       LibraryVariant = "failsafe"
	LibNoAVX2         LibraryVariant = "noavx2"
	LibVulkan         LibraryVariant = "vulkan"
	LibVulkanFailsafe LibraryVariant = "vulkan_failsafe"
	LibVulkanNoAVX2   LibraryVariant = "vulkan_noavx2"
	LibCUBLAS         LibraryVariant = "cublas"
	LibHIPBLAS        LibraryVariant = "hipblas"
)

// New creates a new KoboldCpp instance
func New() *KoboldCpp {
	return &KoboldCpp{
		isLoaded: false,
	}
}

// LoadLibrary loads the KoboldCpp shared library
func (k *KoboldCpp) LoadLibrary(variant LibraryVariant, libDir string) error {
	if k.isLoaded {
		return errors.New("library already loaded")
	}

	libName := getLibraryName(variant)
	libPath := libDir + "/" + libName

	// Check if file exists
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		return fmt.Errorf("library file not found: %s", libPath)
	}

	// Load the library
	handle, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("failed to load library: %w", err)
	}

	k.handle = handle
	k.libPath = libPath
	k.isLoaded = true

	// Initialize function pointers
	if err := initWhisperFunctions(handle); err != nil {
		return fmt.Errorf("failed to initialize whisper functions: %w", err)
	}

	return nil
}

// Close closes the library handle
func (k *KoboldCpp) Close() error {
	if !k.isLoaded {
		return nil
	}

	// Note: purego doesn't provide dlclose, so we just mark as unloaded
	k.isLoaded = false
	k.handle = 0

	return nil
}

// IsLoaded returns whether the library is currently loaded
func (k *KoboldCpp) IsLoaded() bool {
	return k.isLoaded
}

// GetLibraryPath returns the path to the loaded library
func (k *KoboldCpp) GetLibraryPath() string {
	return k.libPath
}

// getLibraryName returns the platform-specific library name for the given variant
func getLibraryName(variant LibraryVariant) string {
	var prefix, suffix string

	switch runtime.GOOS {
	case "windows":
		prefix = "koboldcpp_"
		suffix = ".dll"
	default: // linux, darwin, and others - Makefile builds .so on all Unix platforms
		prefix = "koboldcpp_"
		suffix = ".so"
	}

	return prefix + string(variant) + suffix
}

// Helper functions for string conversion

// stringToCharPtr converts a Go string to a C char pointer (uintptr)
func stringToCharPtr(s string) uintptr {
	if s == "" {
		return 0
	}

	// Allocate memory for the string + null terminator
	bytes := []byte(s)
	bytes = append(bytes, 0) // null terminator

	// This is a simplified approach - in production you'd want proper memory management
	ptr := unsafe.Pointer(&bytes[0])
	return uintptr(ptr)
}

// freeCharPtr would free the allocated memory (no-op in Go with GC)
func freeCharPtr(ptr uintptr) {
	// In Go, the garbage collector handles this
	// This function exists for API consistency
}

// readFile reads a file and returns its contents
func readFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}
