package main

import (
	"fmt"
	"os"

	koboldcpp "github.com/kawai-network/koboldcpp"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: llama_example <library_dir> <model_path> <prompt>")
		fmt.Println("Example: llama_example . models/llama-2-7b.gguf \"Hello, how are you?\"")
		os.Exit(1)
	}

	libDir := os.Args[1]
	modelPath := os.Args[2]
	prompt := os.Args[3]

	fmt.Println("=== KoboldCpp Llama Go Bindings Example ===")
	fmt.Println()

	// Create KoboldCpp instance
	kcpp := koboldcpp.New()

	// Load library
	fmt.Println("Loading KoboldCpp library...")
	err := kcpp.LoadLibrary(koboldcpp.LibDefault, libDir)
	if err != nil {
		fmt.Printf("Error loading library: %v\n", err)
		os.Exit(1)
	}
	defer kcpp.Close()
	fmt.Println("Library loaded successfully!")
	fmt.Println()

	// Load model
	fmt.Printf("Loading model: %s\n", modelPath)
	err = kcpp.LoadModel(koboldcpp.LoadModelInputs{
		ModelFilename:    modelPath,
		ExecutablePath:   libDir,
		Threads:          4,
		MaxContextLength: 2048,
		GPULayers:        0, // CPU only
		BatchSize:        512,
		Quiet:            false,
	})
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Model loaded successfully!")
	fmt.Println()

	// Count tokens
	fmt.Printf("Counting tokens in prompt: %s\n", prompt)
	tokenCount, err := kcpp.TokenCount(prompt, true)
	if err != nil {
		fmt.Printf("Error counting tokens: %v\n", err)
	} else {
		fmt.Printf("Token count: %d\n", tokenCount.Count)
		fmt.Printf("Token IDs: %v\n", tokenCount.IDs)
	}
	fmt.Println()

	// Generate text
	fmt.Println("Generating text...")
	result, err := kcpp.Generate(koboldcpp.GenerationInputs{
		Prompt:      prompt,
		MaxLength:   100,
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        40,
		RepPen:      1.1,
	})
	if err != nil {
		fmt.Printf("Error generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("=== Generation Result ===")
	fmt.Printf("Status: %d\n", result.Status)
	fmt.Printf("Stop Reason: %d\n", result.StopReason)
	fmt.Printf("Prompt Tokens: %d\n", result.PromptTokens)
	fmt.Printf("Completion Tokens: %d\n", result.CompletionTokens)
	fmt.Println()
	fmt.Println("Generated Text:")
	fmt.Println(result.Text)
	fmt.Println()

	fmt.Println("=== Example Complete ===")
}
