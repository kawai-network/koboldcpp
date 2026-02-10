# KoboldCpp Go Bindings

Go bindings for KoboldCpp using [purego](https://github.com/ebitengine/purego) for CGO-free FFI.

## Features

- ✅ **Llama Text Generation** - Generate text with GGUF models (llama.cpp)
- ✅ **Whisper Speech-to-Text** - Transcribe audio to text
- ✅ **Stable Diffusion Image Generation** - Generate images from text prompts
- 🚧 Text-to-Speech - Coming soon
- 🚧 Text Embeddings - Coming soon

## Installation

```bash
go get github.com/kawai-network/koboldcpp
```

## Requirements

1. **KoboldCpp Library**: You need the compiled KoboldCpp shared library (`.so`, `.dll`, or `.dylib`)
2. **Whisper Model**: Download from [HuggingFace](https://huggingface.co/koboldcpp/whisper/tree/main)

## Quick Start - Whisper Speech-to-Text

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/kawai-network/koboldcpp"
)

func main() {
    // Create instance
    kcpp := koboldcpp.New()
    
    // Load library
    err := kcpp.LoadLibrary(koboldcpp.LibDefault, ".")
    if err != nil {
        log.Fatal(err)
    }
    defer kcpp.Close()
    
    // Load Whisper model
    err = kcpp.LoadWhisperModel(koboldcpp.WhisperLoadModelInputs{
        ModelFilename:  "models/whisper-base.bin",
        ExecutablePath: ".",
        Quiet:          false,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Transcribe audio file
    result, err := kcpp.WhisperTranscribeFile(
        "audio.wav",
        "en",  // language code
        "",    // optional context prompt
    )
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Transcription: %s\n", result.Text)
}
```

## API Reference

### Core Functions

#### `New() *KoboldCpp`
Creates a new KoboldCpp instance.

#### `LoadLibrary(variant LibraryVariant, libDir string) error`
Loads the KoboldCpp shared library.

**Library Variants:**
- `LibDefault` - CPU-only, standard optimizations
- `LibVulkan` - GPU acceleration via Vulkan
- `LibCUBLAS` - NVIDIA GPU acceleration
- `LibHIPBLAS` - AMD GPU acceleration
- `LibNoAVX2` - CPU without AVX2 (older processors)
- `LibFailsafe` - Maximum compatibility

### Whisper Functions

#### `LoadWhisperModel(inputs WhisperLoadModelInputs) error`
Loads a Whisper model for speech-to-text.

**Parameters:**
```go
type WhisperLoadModelInputs struct {
    ModelFilename   string  // Path to whisper model (.bin)
    ExecutablePath  string  // Directory containing the library
    MainGPU         int32   // GPU device ID (0 = first GPU)
    VulkanInfo      string  // Vulkan device info (optional)
    DevicesOverride string  // Override device selection (optional)
    Quiet           bool    // Suppress console output
    DebugMode       int32   // Debug level (0 = off, 1 = on)
}
```

#### `WhisperTranscribe(inputs WhisperGenerationInputs) (*WhisperGenerationOutputs, error)`
Transcribes audio data (base64 encoded).

**Parameters:**
```go
type WhisperGenerationInputs struct {
    Prompt            string  // Context prompt (optional)
    AudioData         string  // Base64 encoded audio
    SuppressNonSpeech bool    // Filter non-speech sounds
    LanguageCode      string  // Language code (en, id, ja, etc.)
}
```

**Returns:**
```go
type WhisperGenerationOutputs struct {
    Status int32   // 1 = success, 0 = failure
    Text   string  // Transcribed text
}
```

#### `WhisperTranscribeFile(audioFilePath, languageCode, prompt string) (*WhisperGenerationOutputs, error)`
Convenience function to transcribe an audio file directly.

#### `GetTotalTranscribeGens() int32`
Returns the total number of transcriptions performed.

## Examples

### Example 1: Basic Transcription

```go
kcpp := koboldcpp.New()
kcpp.LoadLibrary(koboldcpp.LibDefault, ".")

kcpp.LoadWhisperModel(koboldcpp.WhisperLoadModelInputs{
    ModelFilename:  "whisper-base.bin",
    ExecutablePath: ".",
})

result, _ := kcpp.WhisperTranscribeFile("audio.wav", "en", "")
fmt.Println(result.Text)
```

### Example 2: GPU Acceleration

```go
kcpp := koboldcpp.New()

// Try Vulkan first, fallback to CPU
err := kcpp.LoadLibrary(koboldcpp.LibVulkan, ".")
if err != nil {
    kcpp.LoadLibrary(koboldcpp.LibDefault, ".")
}

kcpp.LoadWhisperModel(koboldcpp.WhisperLoadModelInputs{
    ModelFilename: "whisper-base.bin",
    ExecutablePath: ".",
    MainGPU: 0, // Use first GPU
})
```

### Example 3: Multiple Languages

```go
// English
result, _ := kcpp.WhisperTranscribeFile("audio_en.wav", "en", "")

// Indonesian
result, _ = kcpp.WhisperTranscribeFile("audio_id.wav", "id", "")

// Japanese
result, _ = kcpp.WhisperTranscribeFile("audio_ja.wav", "ja", "")

// Auto-detect
result, _ = kcpp.WhisperTranscribeFile("audio.wav", "auto", "")
```

### Example 4: With Context Prompt

```go
// Provide context for better accuracy
result, _ := kcpp.WhisperTranscribeFile(
    "technical_talk.wav",
    "en",
    "This is a discussion about artificial intelligence and machine learning.",
)
```

### Example 5: Base64 Audio Data

```go
import "encoding/base64"

// Read audio file
audioBytes, _ := os.ReadFile("audio.wav")
audioBase64 := base64.StdEncoding.EncodeToString(audioBytes)

// Transcribe
result, _ := kcpp.WhisperTranscribe(koboldcpp.WhisperGenerationInputs{
    AudioData:         audioBase64,
    LanguageCode:      "en",
    SuppressNonSpeech: true,
})
```

## Running the Example

```bash
# Build the example
cd go/examples
go build -o whisper_example whisper_example.go

# Run with default paths
./whisper_example

# Run with custom paths
./whisper_example /path/to/libs /path/to/model.bin /path/to/audio.wav
```

## Supported Audio Formats

The Whisper implementation supports various audio formats through automatic decoding:
- WAV (PCM)
- MP3
- FLAC
- OGG
- M4A
- And more...

Audio is automatically resampled to 16kHz mono for Whisper processing.

## Language Codes

Common language codes:
- `en` - English
- `id` - Indonesian
- `ja` - Japanese
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ko` - Korean
- `auto` - Auto-detect

See [Whisper documentation](https://github.com/openai/whisper) for full list.

## Model Recommendations

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 75 MB | Fastest | Basic | Real-time, low-resource |
| base | 142 MB | Fast | Good | General purpose |
| small | 466 MB | Medium | Better | Balanced |
| medium | 1.5 GB | Slow | Great | High accuracy |
| large | 2.9 GB | Slowest | Best | Maximum accuracy |

Download from: https://huggingface.co/koboldcpp/whisper/tree/main

## Performance Tips

1. **Use GPU acceleration** - Load `LibVulkan` or `LibCUBLAS` for 5-10x speedup
2. **Choose appropriate model** - Smaller models are faster but less accurate
3. **Provide context** - Use the prompt parameter for domain-specific audio
4. **Batch processing** - Load model once, transcribe multiple files
5. **Suppress non-speech** - Enable `SuppressNonSpeech` to filter background noise

## Architecture

```
Go Application
    ↓
purego (CGO-free FFI)
    ↓
KoboldCpp C API (expose.cpp)
    ↓
Whisper Adapter (whisper_adapter.cpp)
    ↓
Whisper.cpp Library
```

## Advantages of Purego

- ✅ **No CGO** - Pure Go, easier cross-compilation
- ✅ **No C compiler needed** - Just Go toolchain
- ✅ **Better performance** - Direct syscalls, no CGO overhead
- ✅ **Simpler builds** - No complex build configurations
- ✅ **Cross-platform** - Works on Windows, Linux, macOS

## Troubleshooting

### Library not found
```
Error: library file not found: koboldcpp_default.so
```
**Solution:** Ensure the KoboldCpp library is compiled and in the correct directory.

### Model load failed
```
Error: failed to load Whisper model
```
**Solution:** 
- Check model file path is correct
- Verify model file is not corrupted
- Ensure sufficient RAM/VRAM

### Transcription failed
```
Error: whisper transcription failed with status: 0
```
**Solution:**
- Check audio file format is supported
- Verify audio file is not corrupted
- Try with a different audio file

## License

This Go binding follows the same license as KoboldCpp (AGPL v3.0).

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## Roadmap

- [x] Whisper Speech-to-Text bindings
- [ ] Text generation (GGUF models)
- [ ] Stable Diffusion image generation
- [ ] Text-to-Speech (TTS)
- [ ] Text embeddings
- [ ] Streaming support
- [ ] Advanced sampling parameters
- [ ] State management (KV cache)

## Links

- [KoboldCpp](https://github.com/LostRuins/koboldcpp)
- [Whisper Models](https://huggingface.co/koboldcpp/whisper)
- [Purego](https://github.com/ebitengine/purego)


## Llama Text Generation

### Basic Text Generation

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/kawai-network/koboldcpp"
)

func main() {
    kcpp := koboldcpp.New()
    
    // Load library
    err := kcpp.LoadLibrary(koboldcpp.LibDefault, ".")
    if err != nil {
        log.Fatal(err)
    }
    defer kcpp.Close()
    
    // Load LLM model
    err = kcpp.LoadModel(koboldcpp.LoadModelInputs{
        ModelFilename:    "models/llama-2-7b.gguf",
        ExecutablePath:   ".",
        Threads:          4,
        MaxContextLength: 2048,
        GPULayers:        0, // CPU only, set > 0 for GPU
        BatchSize:        512,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Generate text
    result, err := kcpp.Generate(koboldcpp.GenerationInputs{
        Prompt:      "Hello, how are you?",
        MaxLength:   100,
        Temperature: 0.7,
        TopP:        0.9,
        TopK:        40,
        RepPen:      1.1,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Generated:", result.Text)
    fmt.Printf("Tokens: %d prompt + %d completion\n", 
        result.PromptTokens, result.CompletionTokens)
}
```

### Token Counting

```go
// Count tokens in text
tokenCount, err := kcpp.TokenCount("Hello, world!", true)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Token count: %d\n", tokenCount.Count)
fmt.Printf("Token IDs: %v\n", tokenCount.IDs)
```

### Streaming Generation

```go
// Start generation (non-blocking)
go func() {
    result, err := kcpp.Generate(koboldcpp.GenerationInputs{
        Prompt:    "Write a story:",
        MaxLength: 500,
    })
    if err != nil {
        log.Fatal(err)
    }
}()

// Poll for tokens
for !kcpp.HasFinished() {
    count := kcpp.GetStreamCount()
    for i := 0; i < count; i++ {
        token, err := kcpp.GetStreamToken(i)
        if err == nil {
            fmt.Print(token)
        }
    }
    time.Sleep(100 * time.Millisecond)
}
```

### Advanced Generation Parameters

```go
result, err := kcpp.Generate(koboldcpp.GenerationInputs{
    Prompt:           "Translate to French: Hello",
    MaxLength:        50,
    Temperature:      0.7,
    TopP:             0.9,
    TopK:             40,
    MinP:             0.05,
    RepPen:           1.1,
    RepPenRange:      256,
    PresencePenalty:  0.0,
    Mirostat:         0,
    Grammar:          "", // GBNF grammar
    StopSequence:     []string{"\n", "###"},
    Seed:             42,
})
```

## Stable Diffusion Image Generation

### Text-to-Image

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/kawai-network/koboldcpp"
)

func main() {
    kcpp := koboldcpp.New()
    
    // Load library
    err := kcpp.LoadLibrary(koboldcpp.LibDefault, ".")
    if err != nil {
        log.Fatal(err)
    }
    defer kcpp.Close()
    
    // Load Stable Diffusion model
    err = kcpp.LoadSDModel(koboldcpp.SDLoadModelInputs{
        ModelFilename:  "models/sd_v1.5.gguf",
        ExecutablePath: ".",
        Threads:        4,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Generate image
    result, err := kcpp.SDGenerate(koboldcpp.SDGenerationInputs{
        Prompt:       "a beautiful sunset over mountains, highly detailed",
        NegativePrompt: "blurry, low quality",
        Width:        512,
        Height:       512,
        SampleSteps:  20,
        CFGScale:     7.0,
        Seed:         -1, // Random seed
        SampleMethod: "euler_a",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Save image
    err = koboldcpp.SaveImageToFile(result.Data, "output.png")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Image generated successfully!")
}
```

### Image-to-Image (img2img)

```go
// Generate from existing image
result, err := kcpp.SDGenerateFromFile(
    "enhance this image, make it more vibrant",
    "input.png",
    koboldcpp.SDGenerationInputs{
        Width:             512,
        Height:            512,
        SampleSteps:       20,
        CFGScale:          7.0,
        DenoisingStrength: 0.75, // How much to change (0.0-1.0)
        Seed:              -1,
        SampleMethod:      "euler_a",
    },
)
```

### Image Upscaling

```go
// Upscale an image
result, err := kcpp.SDUpscale(koboldcpp.SDUpscaleInputs{
    InitImages:      base64EncodedImage,
    UpscalingResize: 2, // 2x upscale
})
```

### Advanced Options

```go
err = kcpp.LoadSDModel(koboldcpp.SDLoadModelInputs{
    ModelFilename:       "models/sd_v1.5.gguf",
    ExecutablePath:      ".",
    Threads:             4,
    Quant:               0,
    FlashAttention:      true,
    OffloadCPU:          false,
    VAECPU:              false,
    ClipCPU:             false,
    TAESD:               false,
    TiledVAEThreshold:   0,
    // Optional component models
    T5XXLFilename:       "",
    Clip1Filename:       "",
    Clip2Filename:       "",
    VAEFilename:         "",
    // LoRA support
    LoraFilename:        "models/lora.safetensors",
    LoraMultiplier:      1.0,
    LoraApplyMode:       0,
    // Upscaler
    UpscalerFilename:    "models/upscaler.pth",
    // Image limits
    ImgHardLimit:        2048,
    ImgSoftLimit:        1024,
    Quiet:               false,
})
```

### Generation Parameters

```go
result, err := kcpp.SDGenerate(koboldcpp.SDGenerationInputs{
    Prompt:            "your prompt here",
    NegativePrompt:    "things to avoid",
    InitImages:        "", // Base64 encoded init image for img2img
    Mask:              "", // Base64 encoded mask for inpainting
    ExtraImages:       []string{}, // Additional control images
    FlipMask:          false,
    DenoisingStrength: 0.75, // For img2img (0.0-1.0)
    CFGScale:          7.0,  // Classifier-free guidance scale
    DistilledGuidance: -1.0,
    ShiftedTimestep:   0,
    SampleSteps:       20,
    Width:             512,
    Height:            512,
    Seed:              -1, // -1 for random
    SampleMethod:      "euler_a", // euler, euler_a, heun, dpm2, etc.
    Scheduler:         "",
    ClipSkip:          -1,
    VidReqFrames:      1,    // For video generation
    VideoOutputType:   0,    // 0=gif, 1=avi, 2=both
    RemoveLimits:      false,
    CircularX:         false, // Tileable texture
    CircularY:         false,
    Upscale:           false,
})
```

## API Reference

### Stable Diffusion Types

#### SDLoadModelInputs
Parameters for loading a Stable Diffusion model.

#### SDGenerationInputs
Parameters for generating images.

#### SDGenerationOutputs
Output from image generation containing:
- `Status`: Generation status (1 = success)
- `Animated`: Whether output is animated (GIF/video)
- `Data`: Base64 encoded image data
- `DataExtra`: Additional data (for animated outputs)

#### SDUpscaleInputs
Parameters for upscaling images.

#### SDInfoOutputs
Information about the loaded model.

### Stable Diffusion Methods

#### LoadSDModel(inputs SDLoadModelInputs) error
Load a Stable Diffusion model.

#### SDGenerate(inputs SDGenerationInputs) (*SDGenerationOutputs, error)
Generate images from text prompts.

#### SDGenerateFromFile(prompt, initImagePath string, params SDGenerationInputs) (*SDGenerationOutputs, error)
Convenience method for img2img from file.

#### SDUpscale(inputs SDUpscaleInputs) (*SDGenerationOutputs, error)
Upscale images using the loaded upscaler model.

#### SDGetInfo() (*SDInfoOutputs, error)
Get information about the loaded SD model.

### Helper Functions

#### SaveImageToFile(base64Data, outputPath string) error
Save a base64 encoded image to a file.

## Examples

See the `examples/` directory for complete working examples:
- `whisper_example.go` - Whisper speech-to-text
- `simple_whisper.go` - Simple Whisper usage
- `sd_example.go` - Stable Diffusion text-to-image
- `sd_img2img_example.go` - Stable Diffusion image-to-image

## Building Examples

```bash
cd examples
go build -o whisper_example whisper_example.go
go build -o sd_example sd_example.go
go build -o sd_img2img_example sd_img2img_example.go
```

## Platform Support

- ✅ Linux (x86_64, ARM64)
- ✅ macOS (Intel, Apple Silicon)
- ✅ Windows (x86_64)

## License

This project follows the same license as KoboldCpp.
