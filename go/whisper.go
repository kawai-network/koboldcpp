package koboldcpp

import (
	"encoding/base64"
	"fmt"
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
	whisperLoadModel       func(inputs cWhisperLoadModelInputs) bool
	whisperGenerate        func(inputs cWhisperGenerationInputs) cWhisperGenerationOutputs
	getTotalTranscribeGens func() int32
)

// initWhisperFunctions initializes the Whisper-related function pointers
func initWhisperFunctions(handle uintptr) error {
	// whisper_load_model
	whisperLoadModelPtr, err := purego.Dlsym(handle, "whisper_load_model")
	if err != nil {
		return fmt.Errorf("failed to load whisper_load_model: %w", err)
	}
	purego.RegisterFunc(&whisperLoadModel, whisperLoadModelPtr)

	// whisper_generate
	whisperGeneratePtr, err := purego.Dlsym(handle, "whisper_generate")
	if err != nil {
		return fmt.Errorf("failed to load whisper_generate: %w", err)
	}
	purego.RegisterFunc(&whisperGenerate, whisperGeneratePtr)

	// get_total_transcribe_gens
	getTotalTranscribeGensPtr, err := purego.Dlsym(handle, "get_total_transcribe_gens")
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
	modelFilename := stringToCharPtr(inputs.ModelFilename)
	executablePath := stringToCharPtr(inputs.ExecutablePath)
	vulkanInfo := stringToCharPtr(inputs.VulkanInfo)
	devicesOverride := stringToCharPtr(inputs.DevicesOverride)

	defer func() {
		freeCharPtr(modelFilename)
		freeCharPtr(executablePath)
		freeCharPtr(vulkanInfo)
		freeCharPtr(devicesOverride)
	}()

	cInputs := cWhisperLoadModelInputs{
		modelFilename:   modelFilename,
		executablePath:  executablePath,
		mainGPU:         inputs.MainGPU,
		vulkanInfo:      vulkanInfo,
		devicesOverride: devicesOverride,
		quiet:           inputs.Quiet,
		debugMode:       inputs.DebugMode,
	}

	success := whisperLoadModel(cInputs)
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
	prompt := stringToCharPtr(inputs.Prompt)
	audioData := stringToCharPtr(inputs.AudioData)
	langCode := stringToCharPtr(inputs.LanguageCode)

	defer func() {
		freeCharPtr(prompt)
		freeCharPtr(audioData)
		freeCharPtr(langCode)
	}()

	cInputs := cWhisperGenerationInputs{
		prompt:            prompt,
		audioData:         audioData,
		suppressNonSpeech: inputs.SuppressNonSpeech,
		langCode:          langCode,
	}

	cOutputs := whisperGenerate(cInputs)

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
func charPtrToString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}

	// Read bytes until null terminator
	var bytes []byte
	for i := 0; ; i++ {
		b := *(*byte)(unsafe.Pointer(ptr + uintptr(i)))
		if b == 0 {
			break
		}
		bytes = append(bytes, b)
	}

	return string(bytes)
}
