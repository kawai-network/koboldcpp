package koboldcpp

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Constants - matching Python version
const (
	SamplerOrderMax    = 7
	TensorSplitMax     = 16
	ImagesMax          = 8
	AudioMax           = 4
	LogprobsMax        = 10
	DefaultDraftAmount = 8
	OverrideKVMax      = 4
	StopTokenMax       = 256
	BanTokenMax        = 768
	LogitBiasMax       = 512
	DrySeqBreakMax     = 128
	ExtraImagesMax     = 4
	MaxCStringLen      = 1024 * 1024 * 10 // 10MB max for C string safety
)

// Global variables - matching Python version
var (
	KcppVersion    = "1.107.3"
	handle         uintptr
	libname        string
	stringRegistry = make(map[uintptr][]byte) // Registry to prevent GC of C strings
)

// Library names - matching Python version
const (
	libDefault        = "koboldcpp_default"
	libFailsafe       = "koboldcpp_failsafe"
	libNoAVX2         = "koboldcpp_noavx2"
	libVulkanFailsafe = "koboldcpp_vulkan_failsafe"
	libCublas         = "koboldcpp_cublas"
	libHipblas        = "koboldcpp_hipblas"
	libVulkan         = "koboldcpp_vulkan"
	libVulkanNoAVX2   = "koboldcpp_vulkan_noavx2"
)

// Structs - matching Python ctypes structures

// LogitBias matches Python logit_bias
type LogitBias struct {
	TokenID int32
	Bias    float32
}

// TokenCountOutputs matches Python token_count_outputs
type TokenCountOutputs struct {
	Count int32
	IDs   *int32
}

// LogprobItem matches Python logprob_item
type LogprobItem struct {
	OptionCount     int32
	SelectedToken   *byte
	SelectedLogprob float32
	SelectedTokenID int32
	Tokens          [LogprobsMax]*byte
	TokenIDs        [LogprobsMax]int32
	Logprobs        *float32
}

// LastLogprobsOutputs matches Python last_logprobs_outputs
type LastLogprobsOutputs struct {
	Count        int32
	LogprobItems *LogprobItem
}

// LoadModelInputs matches Python load_model_inputs
type LoadModelInputs struct {
	Threads               int32
	BlasThreads           int32
	MaxContextLength      int32
	LowVRAM               bool
	UseMMQ                bool
	UseRowSplit           bool
	ExecutablePath        *byte
	ModelFilename         *byte
	LoraFilename          *byte
	DraftModelFilename    *byte
	DraftAmount           int32
	DraftGPULayers        int32
	DraftGPUSplit         [TensorSplitMax]float32
	MMProjFilename        *byte
	MMProjCPU             bool
	VisionMaxRes          int32
	UseMmap               bool
	UseMlock              bool
	UseSmartContext       bool
	UseContextShift       bool
	UseFastForward        bool
	KcppMainGPU           int32
	VulkanInfo            *byte
	BatchSize             int32
	Autofit               bool
	AutofitTaxMB          int32
	GPULayers             int32
	RopeFreqScale         float32
	RopeFreqBase          float32
	OverrideNativeContext int32
	MoeExperts            int32
	MoeCPU                int32
	NoBosToken            bool
	LoadGuidance          bool
	OverrideKV            [OverrideKVMax]*byte
	OverrideTensors       *byte
	FlashAttention        bool
	TensorSplit           [TensorSplitMax]float32
	QuantK                int32
	QuantV                int32
	CheckSlowness         bool
	HighPriority          bool
	SWASupport            bool
	SmartCache            bool
	SmartCacheSlots       int32
	PipelineParallel      bool
	LoraMultiplier        float32
	DevicesOverride       *byte
	Quiet                 bool
	DebugMode             int32
}

// GenerationInputs matches Python generation_inputs
type GenerationInputs struct {
	Seed                   int32
	Prompt                 *byte
	Memory                 *byte
	NegativePrompt         *byte
	GuidanceScale          float32
	Images                 [ImagesMax]*byte
	Audio                  [AudioMax]*byte
	MaxContextLength       int32
	MaxLength              int32
	Temperature            float32
	TopK                   int32
	TopA                   float32
	TopP                   float32
	MinP                   float32
	TypicalP               float32
	TFS                    float32
	NSigma                 float32
	RepPen                 float32
	RepPenRange            int32
	RepPenSlope            float32
	PresencePenalty        float32
	Mirostat               int32
	MirostatTau            float32
	MirostatEta            float32
	XTCThreshold           float32
	XTCProbability         float32
	SamplerOrder           [SamplerOrderMax]int32
	SamplerLen             int32
	AllowEOSToken          bool
	BypassEOSToken         bool
	ToolCallFix            bool
	RenderSpecial          bool
	StreamSSE              bool
	Grammar                *byte
	GrammarRetainState     bool
	DynatempRange          float32
	DynatempExponent       float32
	SmoothingFactor        float32
	SmoothingCurve         float32
	AdaptiveTarget         float32
	AdaptiveDecay          float32
	DryMultiplier          float32
	DryBase                float32
	DryAllowedLength       int32
	DryPenaltyLastN        int32
	DrySequenceBreakersLen int32
	DrySequenceBreakers    **byte
	StopSequenceLen        int32
	StopSequence           **byte
	LogitBiasesLen         int32
	LogitBiases            *LogitBias
	BannedTokensLen        int32
	BannedTokens           **byte
}

// GenerationOutputs matches Python generation_outputs
type GenerationOutputs struct {
	Status           int32
	StopReason       int32
	PromptTokens     int32
	CompletionTokens int32
	Text             *byte
}

// SDLoadModelInputs matches Python sd_load_model_inputs
type SDLoadModelInputs struct {
	ModelFilename       *byte
	ExecutablePath      *byte
	KcppMainGPU         int32
	VulkanInfo          *byte
	Threads             int32
	Quant               int32
	FlashAttention      bool
	OffloadCPU          bool
	VAECPU              bool
	ClipCPU             bool
	DiffusionConvDirect bool
	VAEConvDirect       bool
	TAESD               bool
	TiledVAEThreshold   int32
	T5XXLFilename       *byte
	Clip1Filename       *byte
	Clip2Filename       *byte
	VAEFilename         *byte
	LoraFilename        *byte
	LoraMultiplier      float32
	LoraApplyMode       int32
	PhotomakerFilename  *byte
	UpscalerFilename    *byte
	ImgHardLimit        int32
	ImgSoftLimit        int32
	DevicesOverride     *byte
	Quiet               bool
	DebugMode           int32
}

// SDGenerationInputs matches Python sd_generation_inputs
type SDGenerationInputs struct {
	Prompt            *byte
	NegativePrompt    *byte
	InitImages        *byte
	Mask              *byte
	ExtraImagesLen    int32
	ExtraImages       **byte
	FlipMask          bool
	DenoisingStrength float32
	CFGScale          float32
	DistilledGuidance float32
	ShiftedTimestep   int32
	SampleSteps       int32
	Width             int32
	Height            int32
	Seed              int32
	SampleMethod      *byte
	Scheduler         *byte
	ClipSkip          int32
	VidReqFrames      int32
	VideoOutputType   int32
	RemoveLimits      bool
	CircularX         bool
	CircularY         bool
	Upscale           bool
}

// SDGenerationOutputs matches Python sd_generation_outputs
type SDGenerationOutputs struct {
	Status    int32
	Animated  int32
	Data      *byte
	DataExtra *byte
}

// SDUpscaleInputs matches Python sd_upscale_inputs
type SDUpscaleInputs struct {
	InitImages      *byte
	UpscalingResize int32
}

// SDInfoOutputs matches Python sd_info_outputs
type SDInfoOutputs struct {
	Status int32
	Data   *byte
}

// WhisperLoadModelInputs matches Python whisper_load_model_inputs
type WhisperLoadModelInputs struct {
	ModelFilename   *byte
	ExecutablePath  *byte
	KcppMainGPU     int32
	VulkanInfo      *byte
	DevicesOverride *byte
	Quiet           bool
	DebugMode       int32
}

// WhisperGenerationInputs matches Python whisper_generation_inputs
type WhisperGenerationInputs struct {
	Prompt            *byte
	AudioData         *byte
	SuppressNonSpeech bool
	LangCode          *byte
}

// WhisperGenerationOutputs matches Python whisper_generation_outputs
type WhisperGenerationOutputs struct {
	Status int32
	Data   *byte
}

// TTSLoadModelInputs matches Python tts_load_model_inputs
type TTSLoadModelInputs struct {
	Threads          int32
	TTCModelFilename *byte
	CTSModelFilename *byte
	ExecutablePath   *byte
	KcppMainGPU      int32
	VulkanInfo       *byte
	GPULayers        int32
	FlashAttention   bool
	TTSMaxLen        int32
	DevicesOverride  *byte
	Quiet            bool
	DebugMode        int32
}

// TTSGenerationInputs matches Python tts_generation_inputs
type TTSGenerationInputs struct {
	Prompt             *byte
	SpeakerSeed        int32
	AudioSeed          int32
	CustomSpeakerVoice *byte
	CustomSpeakerText  *byte
	CustomSpeakerData  *byte
}

// TTSGenerationOutputs matches Python tts_generation_outputs
type TTSGenerationOutputs struct {
	Status int32
	Data   *byte
}

// EmbeddingsLoadModelInputs matches Python embeddings_load_model_inputs
type EmbeddingsLoadModelInputs struct {
	Threads          int32
	ModelFilename    *byte
	ExecutablePath   *byte
	KcppMainGPU      int32
	VulkanInfo       *byte
	GPULayers        int32
	FlashAttention   bool
	UseMmap          bool
	EmbeddingsMaxCtx int32
	DevicesOverride  *byte
	Quiet            bool
	DebugMode        int32
}

// EmbeddingsGenerationInputs matches Python embeddings_generation_inputs
type EmbeddingsGenerationInputs struct {
	Prompt   *byte
	Truncate bool
}

// EmbeddingsGenerationOutputs matches Python embeddings_generation_outputs
type EmbeddingsGenerationOutputs struct {
	Status int32
	Count  int32
	Data   *byte
}

// Helper functions

// getLibraryExtension returns the appropriate library extension for the OS
func getLibraryExtension() string {
	switch runtime.GOOS {
	case "windows":
		return ".dll"
	case "darwin":
		return ".dylib"
	default:
		return ".so"
	}
}

// pickExistantFile checks which library file exists
func pickExistantFile(ntOption, nonNTOption string) string {
	ext := getLibraryExtension()

	// Check if file exists
	if _, err := os.Stat(ntOption + ext); err == nil {
		return ntOption + ext
	}
	if _, err := os.Stat(nonNTOption + ext); err == nil {
		return nonNTOption + ext
	}

	// Default to first option
	return ntOption + ext
}

// fileExists checks if a file exists
func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return err == nil
}

// getLibraryPath returns the full path to the library
func getLibraryPath(libname string) string {
	// Try current directory first
	if fileExists(libname) {
		return libname
	}

	// Try parent directory (for when running from koboldcpp-go/)
	parentPath := filepath.Join("..", libname)
	if fileExists(parentPath) {
		return parentPath
	}

	// Get executable directory
	exePath, err := os.Executable()
	if err != nil {
		return libname
	}
	dirPath := filepath.Dir(exePath)
	return filepath.Join(dirPath, libname)
}

// InitLibrary initializes the C++ library - matches Python init_library()
func InitLibrary(useCUDA, useVulkan, noAVX2, failsafe bool) error {
	ext := getLibraryExtension()

	// Select library based on options - matching Python logic
	if noAVX2 {
		if failsafe && useVulkan && fileExists(libVulkanFailsafe+ext) {
			libname = libVulkanFailsafe + ext
		} else if useVulkan && fileExists(libVulkanNoAVX2+ext) {
			libname = libVulkanNoAVX2 + ext
		} else if failsafe && fileExists(libFailsafe+ext) {
			fmt.Println("!!! Attempting to use FAILSAFE MODE !!!")
			libname = libFailsafe + ext
		} else if fileExists(libNoAVX2 + ext) {
			libname = libNoAVX2 + ext
		} else {
			libname = libDefault + ext
		}
	} else if useCUDA {
		if fileExists(libCublas + ext) {
			libname = libCublas + ext
		} else if fileExists(libHipblas + ext) {
			libname = libHipblas + ext
		} else {
			libname = libDefault + ext
		}
	} else if useVulkan {
		if fileExists(libVulkan + ext) {
			libname = libVulkan + ext
		} else if fileExists(libVulkanNoAVX2 + ext) {
			libname = libVulkanNoAVX2 + ext
		} else {
			libname = libDefault + ext
		}
	} else {
		libname = libDefault + ext
		if !fileExists(libname) && fileExists(libNoAVX2+ext) {
			libname = libNoAVX2 + ext
		}
	}

	fmt.Printf("Initializing dynamic library: %s\n", libname)

	// Load library
	libPath := getLibraryPath(libname)
	var err error
	handle, err = purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("failed to load library %s: %w", libPath, err)
	}

	fmt.Println("Library loaded successfully!")
	return nil
}

// CloseLibrary closes the loaded library and releases resources
func CloseLibrary() error {
	if handle == 0 {
		return nil // Already closed or never opened
	}

	// Clear string registry
	stringRegistry = make(map[uintptr][]byte)

	// Close the library handle
	if err := purego.Dlclose(handle); err != nil {
		return fmt.Errorf("failed to close library: %w", err)
	}

	// Reset state
	handle = 0
	libname = ""

	fmt.Println("Library closed successfully!")
	return nil
}

func CString(s string) *byte {
	if s == "" {
		return nil
	}
	b := append([]byte(s), 0)
	ptr := &b[0]
	// Store in registry to prevent GC
	stringRegistry[uintptr(unsafe.Pointer(ptr))] = b
	return ptr
}

// FreeCString removes the string from registry, allowing GC
func FreeCString(ptr *byte) {
	if ptr != nil {
		delete(stringRegistry, uintptr(unsafe.Pointer(ptr)))
	}
}

// Helper to convert C string to Go string with bounded search
func GoString(cstr *byte) string {
	if cstr == nil {
		return ""
	}

	// Find null terminator with maximum length bound
	ptr := unsafe.Pointer(cstr)
	var length int
	for length < MaxCStringLen {
		if *(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(length))) == 0 {
			break
		}
		length++
	}

	// If we hit the limit without finding null terminator, truncate
	if length >= MaxCStringLen {
		length = MaxCStringLen - 1
	}

	// Convert to Go string
	return string(unsafe.Slice(cstr, length))
}

// Public API functions - matching Python functions

// LoadModel loads a model - matches Python load_model()
