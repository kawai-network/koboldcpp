package main

import (
	"fmt"
	"os"

	koboldcpp "github.com/kawai-network/koboldcpp/go"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: sd_example <library_dir> <model_path> <prompt> [output_image.png]")
		fmt.Println("Example: sd_example . models/sd_v1.5.gguf \"a beautiful sunset over mountains\" output.png")
		os.Exit(1)
	}

	libDir := os.Args[1]
	modelPath := os.Args[2]
	prompt := os.Args[3]
	outputPath := "output.png"
	if len(os.Args) > 4 {
		outputPath = os.Args[4]
	}

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

	// Load SD model
	fmt.Printf("Loading Stable Diffusion model: %s\n", modelPath)
	err = kcpp.LoadSDModel(koboldcpp.SDLoadModelInputs{
		ModelFilename:  modelPath,
		ExecutablePath: libDir,
		Threads:        4,
		Quiet:          false,
	})
	if err != nil {
		fmt.Printf("Error loading SD model: %v\n", err)
		os.Exit(1)
	}

	// Get model info
	info, err := kcpp.SDGetInfo()
	if err == nil {
		fmt.Printf("Model info: %s\n", info.Data)
	}

	// Generate image
	fmt.Printf("Generating image with prompt: %s\n", prompt)
	result, err := kcpp.SDGenerate(koboldcpp.SDGenerationInputs{
		Prompt:       prompt,
		Width:        512,
		Height:       512,
		SampleSteps:  20,
		CFGScale:     7.0,
		Seed:         -1, // Random seed
		SampleMethod: "euler_a",
	})
	if err != nil {
		fmt.Printf("Error generating image: %v\n", err)
		os.Exit(1)
	}

	// Save image
	fmt.Printf("Saving image to: %s\n", outputPath)
	err = koboldcpp.SaveImageToFile(result.Data, outputPath)
	if err != nil {
		fmt.Printf("Error saving image: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Image generated successfully!")
	if result.Animated > 0 {
		fmt.Println("Note: This is an animated image (GIF/video)")
	}
}
