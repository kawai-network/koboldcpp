package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/kawai-network/koboldcpp"
)

func main() {
	// Create a new KoboldCpp instance
	kcpp := koboldcpp.New()

	// Determine library directory (current directory or parent)
	libDir := "."
	if len(os.Args) > 1 {
		libDir = os.Args[1]
	}

	// Load the library (use Vulkan variant for GPU acceleration)
	fmt.Println("Loading KoboldCpp library...")
	err := kcpp.LoadLibrary(koboldcpp.LibVulkan, libDir)
	if err != nil {
		// Fallback to default if Vulkan not available
		fmt.Println("Vulkan library not found, trying default...")
		err = kcpp.LoadLibrary(koboldcpp.LibDefault, libDir)
		if err != nil {
			log.Fatalf("Failed to load library: %v", err)
		}
	}
	defer kcpp.Close()

	fmt.Printf("Library loaded: %s\n\n", kcpp.GetLibraryPath())

	// Example 1: Load Whisper model
	fmt.Println("=== Example 1: Loading Whisper Model ===")

	modelPath := "models/whisper-base.bin"
	if len(os.Args) > 2 {
		modelPath = os.Args[2]
	}

	// Check if model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s\nDownload from: https://huggingface.co/koboldcpp/whisper/tree/main", modelPath)
	}

	loadInputs := koboldcpp.WhisperLoadModelInputs{
		ModelFilename:  modelPath,
		ExecutablePath: libDir,
		MainGPU:        0,
		VulkanInfo:     "",
		Quiet:          false,
		DebugMode:      0,
	}

	err = kcpp.LoadWhisperModel(loadInputs)
	if err != nil {
		log.Fatalf("Failed to load Whisper model: %v", err)
	}

	fmt.Println("Whisper model loaded successfully!\n")

	// Example 2: Transcribe audio from file
	fmt.Println("=== Example 2: Transcribe Audio File ===")

	audioPath := "test_audio.wav"
	if len(os.Args) > 3 {
		audioPath = os.Args[3]
	}

	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		fmt.Printf("Audio file not found: %s\n", audioPath)
		fmt.Println("Skipping file transcription example...\n")
	} else {
		result, err := kcpp.WhisperTranscribeFile(
			audioPath,
			"en", // language code (en, id, ja, etc.)
			"",   // optional prompt for context
		)
		if err != nil {
			log.Printf("Transcription failed: %v", err)
		} else {
			fmt.Printf("Transcription: %s\n", result.Text)
			fmt.Printf("Status: %d\n\n", result.Status)
		}
	}

	// Example 3: Transcribe with base64 encoded audio
	fmt.Println("=== Example 3: Transcribe with Base64 Audio ===")

	// For this example, we'll use the same audio file if it exists
	if _, err := os.Stat(audioPath); err == nil {
		// Read and encode audio
		audioData, err := os.ReadFile(audioPath)
		if err != nil {
			log.Printf("Failed to read audio: %v", err)
		} else {
			// In real usage, you'd encode to base64
			// For now, we'll use the file method
			fmt.Println("Using file-based transcription (see Example 2)")
		}
	}

	// Example 4: Transcribe with language detection
	fmt.Println("=== Example 4: Transcribe with Auto Language Detection ===")

	if _, err := os.Stat(audioPath); err == nil {
		result, err := kcpp.WhisperTranscribeFile(
			audioPath,
			"auto", // auto-detect language
			"",
		)
		if err != nil {
			log.Printf("Transcription failed: %v", err)
		} else {
			fmt.Printf("Transcription: %s\n", result.Text)
			fmt.Printf("Status: %d\n\n", result.Status)
		}
	}

	// Example 5: Transcribe with context prompt
	fmt.Println("=== Example 5: Transcribe with Context Prompt ===")

	if _, err := os.Stat(audioPath); err == nil {
		result, err := kcpp.WhisperTranscribeFile(
			audioPath,
			"en",
			"This is a technical discussion about AI and machine learning.", // context prompt
		)
		if err != nil {
			log.Printf("Transcription failed: %v", err)
		} else {
			fmt.Printf("Transcription: %s\n", result.Text)
			fmt.Printf("Status: %d\n\n", result.Status)
		}
	}

	// Example 6: Get statistics
	fmt.Println("=== Example 6: Statistics ===")
	totalTranscriptions := kcpp.GetTotalTranscribeGens()
	fmt.Printf("Total transcriptions performed: %d\n", totalTranscriptions)

	fmt.Println("\n=== All Examples Complete ===")
	fmt.Println("\nUsage:")
	fmt.Printf("  %s [lib_dir] [model_path] [audio_path]\n", filepath.Base(os.Args[0]))
	fmt.Println("\nExample:")
	fmt.Printf("  %s . models/whisper-base.bin test_audio.wav\n", filepath.Base(os.Args[0]))
	fmt.Println("\nDownload Whisper models from:")
	fmt.Println("  https://huggingface.co/koboldcpp/whisper/tree/main")
}
