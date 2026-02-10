package main

import (
	"fmt"
	"os"

	koboldcpp "github.com/kawai-network/koboldcpp/go"
)

func main() {
	if len(os.Args) < 5 {
		fmt.Println("Usage: sd_img2img_example <library_dir> <model_path> <init_image> <prompt> [output_image.png]")
		fmt.Println("Example: sd_img2img_example . models/sd_v1.5.gguf input.png \"enhance this image, make it more vibrant\" output.png")
		os.Exit(1)
	}

	libDir := os.Args[1]
	modelPath := os.Args[2]
	initImagePath := os.Args[3]
	prompt := os.Args[4]
	outputPath := "output.png"
	if len(os.Args) > 5 {
		outputPath = os.Args[5]
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

	// Generate image from init image (img2img)
	fmt.Printf("Generating image from: %s\n", initImagePath)
	fmt.Printf("Prompt: %s\n", prompt)

	result, err := kcpp.SDGenerateFromFile(prompt, initImagePath, koboldcpp.SDGenerationInputs{
		Width:             512,
		Height:            512,
		SampleSteps:       20,
		CFGScale:          7.0,
		DenoisingStrength: 0.75, // How much to change the init image (0.0-1.0)
		Seed:              -1,   // Random seed
		SampleMethod:      "euler_a",
		NegativePrompt:    "blurry, low quality, distorted",
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
}
