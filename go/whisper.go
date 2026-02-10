package koboldcpp

import (
	"encoding/base64"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// WhisperLoadModelInputs represents the parameters for loading a Whisper model
type WhisperLoadModelInputs struct {
	ModelFilename   string
	ExecutablePath  string
	MainGPU         int32
	VulkanInfo      string
	DevicesOverride string
	Quiet           bool
	DebugMode       int32
}

// WhisperGenerationInputs represents the parameters for Whisper transcription
type WhisperGenerationInputs struct {
	Prompt            string
	AudioData         string // Base64 encoded audio data
	SuppressNonSpeech bool
	LanguageCode      string
}

// WhisperGenerationOutputs represents the output from Whisper transcription
type WhisperGenerationOutputs struct {
	Status int32
	Text   string
}

// C struct representations for purego
type cWhisperLoadModelInputs struct {
	modelFilename   uintptr // const char*
	executablePath  uintptr // const char*
	mainGPU         int32
	vulkanInfo      uintptr // const char*
	devicesOverride uintptr // const char*
	quiet           bool
	debugMode       int32
}

type cWhisperGenerationInputs struct {
	prompt            uintptr // const char*
	audioData         uintptr // const char*
	suppressNonSpeech bool
	langCode          uintptr // const char*
}

type cWhisperGenerationOutputs struct {
	status int32
	text   uintptr // const char*
}

// Whisper function pointers
var (
	whisperLoadModelPtr    func(inputs *cWhisperLoadModelInputs) bool
	whisperGeneratePtr     func(inputs *cWhisperGenerationInputs, outputs *cWhisperGenerationOutputs)
	getTotalTranscribeGens func() int32
)

// initWhisperFunctions initializes the Whisper-related function pointers
func initWhisperFunctions(handle uintptr) error {
	// whisper_load_model_ptr (pointer-based version for cross-platform compatibility)
	whisperLoadModelPtrPtr, err := dlsymPlatform(handle, "whisper_load_model_ptr")
	if err != nil {
		return fmt.Errorf("failed to load whisper_load_model_ptr: %w", err)
	}
	purego.RegisterFunc(&whisperLoadModelPtr, whisperLoadModelPtrPtr)

	// whisper_generate_ptr (pointer-based version for cross-platform compatibility)
	whisperGeneratePtrPtr, err := dlsymPlatform(handle, "whisper_generate_ptr")
	if err != nil {
		return fmt.Errorf("failed to load whisper_generate_ptr: %w", err)
	}
	purego.RegisterFunc(&whisperGeneratePtr, whisperGeneratePtrPtr)

	// get_total_transcribe_gens
	getTotalTranscribeGensPtr, err := dlsymPlatform(handle, "get_total_transcribe_gens")
	if err != nil {
		return fmt.Errorf("failed to load get_total_transcribe_gens: %w", err)
	}
	purego.RegisterFunc(&getTotalTranscribeGens, getTotalTranscribeGensPtr)

	return nil
}

// LoadWhisperModel loads a Whisper model for speech-to-text transcription
func (k *KoboldCpp) LoadWhisperModel(inputs WhisperLoadModelInputs) error {
	if k.handle == 0 {
		return ErrLibraryNotLoaded
	}

	// Convert Go strings to C strings
	// We need to keep the byte slices alive during the C call
	modelFilenameBytes := append([]byte(inputs.ModelFilename), 0)
	executablePathBytes := append([]byte(inputs.ExecutablePath), 0)
	vulkanInfoBytes := append([]byte(inputs.VulkanInfo), 0)
	devicesOverrideBytes := append([]byte(inputs.DevicesOverride), 0)

	cInputs := cWhisperLoadModelInputs{
		modelFilename:   uintptr(unsafe.Pointer(&modelFilenameBytes[0])),
		executablePath:  uintptr(unsafe.Pointer(&executablePathBytes[0])),
		mainGPU:         inputs.MainGPU,
		vulkanInfo:      uintptr(unsafe.Pointer(&vulkanInfoBytes[0])),
		devicesOverride: uintptr(unsafe.Pointer(&devicesOverrideBytes[0])),
		quiet:           inputs.Quiet,
		debugMode:       inputs.DebugMode,
	}

	success := whisperLoadModelPtr(&cInputs)

	// Keep byte slices alive until after the C call
	runtime.KeepAlive(modelFilenameBytes)
	runtime.KeepAlive(executablePathBytes)
	runtime.KeepAlive(vulkanInfoBytes)
	runtime.KeepAlive(devicesOverrideBytes)

	if !success {
		return fmt.Errorf("failed to load Whisper model: %s", inputs.ModelFilename)
	}

	return nil
}

// WhisperTranscribe performs speech-to-text transcription on audio data
func (k *KoboldCpp) WhisperTranscribe(inputs WhisperGenerationInputs) (*WhisperGenerationOutputs, error) {
	if k.handle == 0 {
		return nil, ErrLibraryNotLoaded
	}

	// Convert Go strings to C strings
	// Keep byte slices alive during the C call
	promptBytes := append([]byte(inputs.Prompt), 0)
	audioDataBytes := append([]byte(inputs.AudioData), 0)
	langCodeBytes := append([]byte(inputs.LanguageCode), 0)

	cInputs := cWhisperGenerationInputs{
		prompt:            uintptr(unsafe.Pointer(&promptBytes[0])),
		audioData:         uintptr(unsafe.Pointer(&audioDataBytes[0])),
		suppressNonSpeech: inputs.SuppressNonSpeech,
		langCode:          uintptr(unsafe.Pointer(&langCodeBytes[0])),
	}

	var cOutputs cWhisperGenerationOutputs
	whisperGeneratePtr(&cInputs, &cOutputs)

	// Keep byte slices alive until after the C call
	runtime.KeepAlive(promptBytes)
	runtime.KeepAlive(audioDataBytes)
	runtime.KeepAlive(langCodeBytes)

	outputs := &WhisperGenerationOutputs{
		Status: cOutputs.status,
		Text:   charPtrToString(cOutputs.text),
	}

	if outputs.Status != 1 {
		return outputs, fmt.Errorf("whisper transcription failed with status: %d", outputs.Status)
	}

	return outputs, nil
}

// WhisperTranscribeFile is a convenience function that reads an audio file and transcribes it
func (k *KoboldCpp) WhisperTranscribeFile(audioFilePath string, languageCode string, prompt string) (*WhisperGenerationOutputs, error) {
	// Read audio file
	audioData, err := readFileToBase64(audioFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio file: %w", err)
	}

	inputs := WhisperGenerationInputs{
		Prompt:            prompt,
		AudioData:         audioData,
		SuppressNonSpeech: true,
		LanguageCode:      languageCode,
	}

	return k.WhisperTranscribe(inputs)
}

// GetTotalTranscribeGens returns the total number of transcriptions performed
func (k *KoboldCpp) GetTotalTranscribeGens() int32 {
	if k.handle == 0 {
		return 0
	}
	return getTotalTranscribeGens()
}

// Helper function to read file and encode to base64
func readFileToBase64(filePath string) (string, error) {
	data, err := readFile(filePath)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

// Helper function to convert uintptr to string
// This mimics C.GoString behavior without CGO
//
//nolint:govet // unsafe.Pointer usage is intentional for FFI
func charPtrToString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}

	// Convert to unsafe.Pointer
	p := unsafe.Pointer(ptr)

	// Find string length
	length := 0
	for {
		// Use unsafe.Add for pointer arithmetic (Go 1.17+)
		b := *(*byte)(unsafe.Add(p, length))
		if b == 0 {
			break
		}
		length++
		// Safety limit
		if length > 1000000 {
			return ""
		}
	}

	if length == 0 {
		return ""
	}

	// Create byte slice from memory
	return string(unsafe.Slice((*byte)(p), length))
}
