package koboldcpp

import (
	"os"
	"testing"
)

func TestWhisperBinding(t *testing.T) {
	// Skip if library not available
	libPath := "../koboldcpp_default.so"
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skip("KoboldCpp library not found, skipping test")
	}

	// Create instance
	kcpp := New()
	if kcpp == nil {
		t.Fatal("Failed to create KoboldCpp instance")
	}

	// Load library
	err := kcpp.LoadLibrary(LibDefault, "..")
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer kcpp.Close()

	// Check if loaded
	if !kcpp.IsLoaded() {
		t.Fatal("Library should be loaded")
	}

	t.Log("Library loaded successfully:", kcpp.GetLibraryPath())
}

func TestWhisperModelLoad(t *testing.T) {
	// Skip if library or model not available
	libPath := "../koboldcpp_default.so"
	modelPath := "../models/whisper-base.bin"

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skip("KoboldCpp library not found, skipping test")
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Whisper model not found, skipping test")
	}

	// Create and load library
	kcpp := New()
	err := kcpp.LoadLibrary(LibDefault, "..")
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer kcpp.Close()

	// Load Whisper model
	err = kcpp.LoadWhisperModel(WhisperLoadModelInputs{
		ModelFilename:  modelPath,
		ExecutablePath: "..",
		Quiet:          true,
		DebugMode:      0,
	})

	if err != nil {
		t.Fatalf("Failed to load Whisper model: %v", err)
	}

	t.Log("Whisper model loaded successfully")
}

func TestWhisperTranscribe(t *testing.T) {
	// Skip if library, model, or audio not available
	libPath := "../koboldcpp_default.so"
	modelPath := "../models/whisper-base.bin"
	audioPath := "../test_audio.wav"

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skip("KoboldCpp library not found, skipping test")
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Whisper model not found, skipping test")
	}

	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		t.Skip("Test audio file not found, skipping test")
	}

	// Create and load library
	kcpp := New()
	err := kcpp.LoadLibrary(LibDefault, "..")
	if err != nil {
		t.Fatalf("Failed to load library: %v", err)
	}
	defer kcpp.Close()

	// Load Whisper model
	err = kcpp.LoadWhisperModel(WhisperLoadModelInputs{
		ModelFilename:  modelPath,
		ExecutablePath: "..",
		Quiet:          true,
	})
	if err != nil {
		t.Fatalf("Failed to load Whisper model: %v", err)
	}

	// Transcribe audio
	result, err := kcpp.WhisperTranscribeFile(audioPath, "en", "")
	if err != nil {
		t.Fatalf("Transcription failed: %v", err)
	}

	if result.Status != 1 {
		t.Fatalf("Expected status 1, got %d", result.Status)
	}

	if result.Text == "" {
		t.Fatal("Expected non-empty transcription text")
	}

	t.Logf("Transcription: %s", result.Text)

	// Check statistics
	totalGens := kcpp.GetTotalTranscribeGens()
	if totalGens < 1 {
		t.Errorf("Expected at least 1 transcription, got %d", totalGens)
	}
	t.Logf("Total transcriptions: %d", totalGens)
}

func TestLibraryVariants(t *testing.T) {
	variants := []LibraryVariant{
		LibDefault,
		LibVulkan,
		LibCUBLAS,
		LibNoAVX2,
	}

	for _, variant := range variants {
		t.Run(string(variant), func(t *testing.T) {
			libName := getLibraryName(variant)
			t.Logf("Library name for %s: %s", variant, libName)

			// Just check if the name is generated correctly
			if libName == "" {
				t.Errorf("Empty library name for variant %s", variant)
			}
		})
	}
}

func BenchmarkWhisperTranscribe(b *testing.B) {
	// Skip if library, model, or audio not available
	libPath := "../koboldcpp_default.so"
	modelPath := "../models/whisper-base.bin"
	audioPath := "../test_audio.wav"

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		b.Skip("KoboldCpp library not found, skipping benchmark")
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skip("Whisper model not found, skipping benchmark")
	}

	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		b.Skip("Test audio file not found, skipping benchmark")
	}

	// Setup
	kcpp := New()
	err := kcpp.LoadLibrary(LibDefault, "..")
	if err != nil {
		b.Fatalf("Failed to load library: %v", err)
	}
	defer kcpp.Close()

	err = kcpp.LoadWhisperModel(WhisperLoadModelInputs{
		ModelFilename:  modelPath,
		ExecutablePath: "..",
		Quiet:          true,
	})
	if err != nil {
		b.Fatalf("Failed to load Whisper model: %v", err)
	}

	// Benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := kcpp.WhisperTranscribeFile(audioPath, "en", "")
		if err != nil {
			b.Fatalf("Transcription failed: %v", err)
		}
	}
}
