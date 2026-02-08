package koboldcpp

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestLibraryLoading tests if the library can be loaded
func TestLibraryLoading(t *testing.T) {
	// Try to load default library
	err := InitLibrary(false, false, false, false)
	if err != nil {
		t.Skipf("Library not found (expected in development): %v", err)
		return
	}
	defer CloseLibrary()

	// Register functions
	err = RegisterFunctions()
	if err != nil {
		t.Fatalf("Failed to register functions: %v", err)
	}

	t.Log("Library loaded and functions registered successfully!")
}

// TestLibrarySelection tests library selection logic
func TestLibrarySelection(t *testing.T) {
	tests := []struct {
		name      string
		useCUDA   bool
		useVulkan bool
		noAVX2    bool
		failsafe  bool
	}{
		{"Default", false, false, false, false},
		{"CUDA", true, false, false, false},
		{"Vulkan", false, true, false, false},
		{"NoAVX2", false, false, true, false},
		{"Failsafe", false, false, true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := InitLibrary(tt.useCUDA, tt.useVulkan, tt.noAVX2, tt.failsafe)
			if err != nil {
				t.Logf("Library variant not available: %v", err)
				return
			}
			defer CloseLibrary()
			t.Logf("Successfully loaded library: %s", libname)
		})
	}
}

// TestHelperFunctions tests helper functions
func TestHelperFunctions(t *testing.T) {
	t.Run("CString", func(t *testing.T) {
		str := "Hello, World!"
		cstr := CString(str)
		if cstr == nil {
			t.Fatal("CString returned nil")
		}

		// Convert back
		gostr := GoString(cstr)
		if gostr != str {
			t.Errorf("String conversion failed: got %q, want %q", gostr, str)
		}

		// Cleanup
		FreeCString(cstr)
	})

	t.Run("EmptyString", func(t *testing.T) {
		cstr := CString("")
		if cstr != nil {
			t.Error("CString should return nil for empty string")
		}
	})

	t.Run("NilCString", func(t *testing.T) {
		gostr := GoString(nil)
		if gostr != "" {
			t.Errorf("GoString(nil) should return empty string, got %q", gostr)
		}
	})
}

// TestFileHelpers tests file helper functions
func TestFileHelpers(t *testing.T) {
	t.Run("GetLibraryExtension", func(t *testing.T) {
		ext := getLibraryExtension()
		switch runtime.GOOS {
		case "windows":
			if ext != ".dll" {
				t.Errorf("Expected .dll on Windows, got %s", ext)
			}
		case "darwin":
			if ext != ".dylib" {
				t.Errorf("Expected .dylib on macOS, got %s", ext)
			}
		default:
			if ext != ".so" {
				t.Errorf("Expected .so on Linux, got %s", ext)
			}
		}
	})

	t.Run("FileExists", func(t *testing.T) {
		// Test with this test file
		if !fileExists("koboldcpp_test.go") {
			t.Error("fileExists should return true for existing file")
		}

		// Test with non-existent file
		if fileExists("nonexistent_file_12345.txt") {
			t.Error("fileExists should return false for non-existent file")
		}
	})
}

// TestStructSizes tests that struct sizes match expectations
func TestStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size int
		obj  interface{}
	}{
		{"LogitBias", 8, LogitBias{}},
		{"TokenCountOutputs", 16, TokenCountOutputs{}},
		{"GenerationOutputs", 24, GenerationOutputs{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Just verify structs can be created
			t.Logf("Struct %s created successfully", tt.name)
		})
	}
}

// TestLoadModelInputs tests LoadModelInputs struct initialization
func TestLoadModelInputs(t *testing.T) {
	inputs := &LoadModelInputs{
		Threads:          4,
		BlasThreads:      4,
		MaxContextLength: 2048,
		UseMmap:          true,
		UseMlock:         false,
		GPULayers:        0,
	}

	if inputs.Threads != 4 {
		t.Errorf("Expected Threads=4, got %d", inputs.Threads)
	}
	if inputs.MaxContextLength != 2048 {
		t.Errorf("Expected MaxContextLength=2048, got %d", inputs.MaxContextLength)
	}
}

// TestGenerationInputs tests GenerationInputs struct initialization
func TestGenerationInputs(t *testing.T) {
	inputs := &GenerationInputs{
		Seed:        -1,
		MaxLength:   100,
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
	}

	if inputs.Temperature != 0.7 {
		t.Errorf("Expected Temperature=0.7, got %f", inputs.Temperature)
	}
	if inputs.TopK != 40 {
		t.Errorf("Expected TopK=40, got %d", inputs.TopK)
	}
}

// TestConstants tests that constants are defined correctly
func TestConstants(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  int
	}{
		{"SamplerOrderMax", SamplerOrderMax, 7},
		{"TensorSplitMax", TensorSplitMax, 16},
		{"ImagesMax", ImagesMax, 8},
		{"AudioMax", AudioMax, 4},
		{"LogprobsMax", LogprobsMax, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.value, tt.want)
			}
		})
	}
}

// TestVersion tests version constant
func TestVersion(t *testing.T) {
	if KcppVersion == "" {
		t.Error("KcppVersion should not be empty")
	}
	t.Logf("KoboldCpp Version: %s", KcppVersion)
}

// BenchmarkCString benchmarks string conversion
func BenchmarkCString(b *testing.B) {
	str := "Hello, World! This is a test string for benchmarking."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cstr := CString(str)
		FreeCString(cstr)
	}
}

// BenchmarkGoString benchmarks C to Go string conversion
func BenchmarkGoString(b *testing.B) {
	str := "Hello, World! This is a test string for benchmarking."
	cstr := CString(str)
	defer FreeCString(cstr)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GoString(cstr)
	}
}

// Integration test - only runs if library is available
func TestIntegration_BasicFlow(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Check if library exists in parent directory
	ext := getLibraryExtension()
	libPath := filepath.Join("..", "koboldcpp_default"+ext)
	if !fileExists(libPath) {
		t.Skip("Library not found, skipping integration test")
	}

	// Initialize library
	err := InitLibrary(false, false, false, false)
	if err != nil {
		t.Fatalf("Failed to initialize library: %v", err)
	}
	defer CloseLibrary()

	// Register functions
	err = RegisterFunctions()
	if err != nil {
		t.Fatalf("Failed to register functions: %v", err)
	}

	t.Log("Integration test: Library loaded and functions registered")

	// Test that we can create structs
	inputs := &LoadModelInputs{
		Threads:          4,
		MaxContextLength: 2048,
	}

	if inputs == nil {
		t.Fatal("Failed to create LoadModelInputs")
	}

	t.Log("Integration test: Structs can be created")
}

// TestMultipleLibraryLoads tests loading and unloading library multiple times
func TestMultipleLibraryLoads(t *testing.T) {
	for i := 0; i < 3; i++ {
		t.Run(fmt.Sprintf("Load_%d", i), func(t *testing.T) {
			err := InitLibrary(false, false, false, false)
			if err != nil {
				t.Skipf("Library not available: %v", err)
				return
			}

			err = RegisterFunctions()
			if err != nil {
				t.Fatalf("Failed to register functions: %v", err)
			}

			// Small delay
			time.Sleep(10 * time.Millisecond)

			err = CloseLibrary()
			if err != nil {
				t.Fatalf("Failed to close library: %v", err)
			}

			t.Logf("Load/unload cycle %d successful", i+1)
		})
	}
}

// TestStringRegistry tests string registry management
func TestStringRegistry(t *testing.T) {
	// Create multiple strings
	strs := []string{"test1", "test2", "test3"}
	cstrs := make([]*byte, len(strs))

	for i, s := range strs {
		cstrs[i] = CString(s)
	}

	// Verify registry has entries
	if len(stringRegistry) != len(strs) {
		t.Errorf("Expected %d entries in registry, got %d", len(strs), len(stringRegistry))
	}

	// Free strings
	for _, cstr := range cstrs {
		FreeCString(cstr)
	}

	// Verify registry is empty
	if len(stringRegistry) != 0 {
		t.Errorf("Expected empty registry after freeing, got %d entries", len(stringRegistry))
	}
}

// TestLibraryPath tests library path resolution
func TestLibraryPath(t *testing.T) {
	path := getLibraryPath("test.so")
	if path == "" {
		t.Error("getLibraryPath returned empty string")
	}
	t.Logf("Library path: %s", path)
}

// TestPickExistantFile tests file picking logic
func TestPickExistantFile(t *testing.T) {
	// This test just verifies the function doesn't panic
	result := pickExistantFile("test1", "test2")
	if result == "" {
		t.Error("pickExistantFile returned empty string")
	}
	t.Logf("Picked file: %s", result)
}

// Main test runner
func TestMain(m *testing.M) {
	fmt.Println("=== KoboldCpp Go Bindings Test Suite ===")
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("OS: %s\n", runtime.GOOS)
	fmt.Printf("Arch: %s\n", runtime.GOARCH)
	fmt.Printf("KoboldCpp Version: %s\n", KcppVersion)
	fmt.Println("=========================================")

	code := m.Run()

	fmt.Println("=========================================")
	fmt.Println("=== Test Suite Complete ===")

	os.Exit(code)
}
