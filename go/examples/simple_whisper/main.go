package main

import (
	"fmt"
	"log"

	"github.com/kawai-network/koboldcpp"
)

// Simple example showing minimal code to transcribe audio
func main() {
	// Create and load library
	kcpp := koboldcpp.New()
	if err := kcpp.LoadLibrary(koboldcpp.LibDefault, "."); err != nil {
		log.Fatal(err)
	}
	defer kcpp.Close()

	// Load Whisper model
	if err := kcpp.LoadWhisperModel(koboldcpp.WhisperLoadModelInputs{
		ModelFilename:  "whisper-base.bin",
		ExecutablePath: ".",
	}); err != nil {
		log.Fatal(err)
	}

	// Transcribe audio
	result, err := kcpp.WhisperTranscribeFile("audio.wav", "en", "")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Transcription: %s\n", result.Text)
}
