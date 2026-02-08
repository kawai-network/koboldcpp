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

// Function pointers - will be registered with purego
var (
	loadModel              func(*LoadModelInputs) bool
	generate               func(*GenerationInputs) GenerationOutputs
	newToken               func(int32) *byte
	getStreamCount         func() int32
	hasFinished            func() bool
	hasAudioSupport        func() bool
	hasVisionSupport       func() bool
	getLastEvalTime        func() float32
	getLastProcessTime     func() float32
	getLastTokenCount      func() int32
	getLastInputCount      func() int32
	getLastSeed            func() int32
	getLastDraftSuccess    func() int32
	getLastDraftFailed     func() int32
	getTotalImgGens        func() int32
	getTotalTTSGens        func() int32
	getTotalTranscribeGens func() int32
	getTotalGens           func() int32
	getLastStopReason      func() int32
	abortGenerate          func() bool
	tokenCount             func(*byte, bool) TokenCountOutputs
	getPendingOutput       func() *byte
	getChatTemplate        func() *byte
	lastLogprobs           func() LastLogprobsOutputs
	detokenize             func(TokenCountOutputs) *byte

	// State management (KV cache)
	calcNewStateKV         func() uint64
	calcNewStateTokenCount func() uint64
	calcOldStateKV         func(int32) uint64
	calcOldStateTokenCount func(int32) uint64
	saveStateKV            func(int32) uint64
	loadStateKV            func(int32) bool
	clearStateKV           func() bool

	// SD functions
	sdLoadModel func(*SDLoadModelInputs) bool
	sdGenerate  func(*SDGenerationInputs) SDGenerationOutputs
	sdUpscale   func(*SDUpscaleInputs) SDGenerationOutputs
	sdGetInfo   func() SDInfoOutputs

	// Whisper functions
	whisperLoadModel func(*WhisperLoadModelInputs) bool
	whisperGenerate  func(*WhisperGenerationInputs) WhisperGenerationOutputs

	// TTS functions
	ttsLoadModel func(*TTSLoadModelInputs) bool
	ttsGenerate  func(*TTSGenerationInputs) TTSGenerationOutputs

	// Embeddings functions
	embeddingsLoadModel func(*EmbeddingsLoadModelInputs) bool
	embeddingsGenerate  func(*EmbeddingsGenerationInputs) EmbeddingsGenerationOutputs
)

// RegisterFunctions registers all C functions with purego - matches Python handle setup
func RegisterFunctions() error {
	// Register core functions
	purego.RegisterLibFunc(&loadModel, handle, "load_model")
	purego.RegisterLibFunc(&generate, handle, "generate")
	purego.RegisterLibFunc(&newToken, handle, "new_token")
	purego.RegisterLibFunc(&getStreamCount, handle, "get_stream_count")
	purego.RegisterLibFunc(&hasFinished, handle, "has_finished")
	purego.RegisterLibFunc(&hasAudioSupport, handle, "has_audio_support")
	purego.RegisterLibFunc(&hasVisionSupport, handle, "has_vision_support")
	purego.RegisterLibFunc(&getLastEvalTime, handle, "get_last_eval_time")
	purego.RegisterLibFunc(&getLastProcessTime, handle, "get_last_process_time")
	purego.RegisterLibFunc(&getLastTokenCount, handle, "get_last_token_count")
	purego.RegisterLibFunc(&getLastInputCount, handle, "get_last_input_count")
	purego.RegisterLibFunc(&getLastSeed, handle, "get_last_seed")
	purego.RegisterLibFunc(&getLastDraftSuccess, handle, "get_last_draft_success")
	purego.RegisterLibFunc(&getLastDraftFailed, handle, "get_last_draft_failed")
	purego.RegisterLibFunc(&getTotalImgGens, handle, "get_total_img_gens")
	purego.RegisterLibFunc(&getTotalTTSGens, handle, "get_total_tts_gens")
	purego.RegisterLibFunc(&getTotalTranscribeGens, handle, "get_total_transcribe_gens")
	purego.RegisterLibFunc(&getTotalGens, handle, "get_total_gens")
	purego.RegisterLibFunc(&getLastStopReason, handle, "get_last_stop_reason")
	purego.RegisterLibFunc(&abortGenerate, handle, "abort_generate")
	purego.RegisterLibFunc(&tokenCount, handle, "token_count")
	purego.RegisterLibFunc(&getPendingOutput, handle, "get_pending_output")
	purego.RegisterLibFunc(&getChatTemplate, handle, "get_chat_template")
	purego.RegisterLibFunc(&lastLogprobs, handle, "last_logprobs")
	purego.RegisterLibFunc(&detokenize, handle, "detokenize")

	// Register state management functions
	purego.RegisterLibFunc(&calcNewStateKV, handle, "calc_new_state_kv")
	purego.RegisterLibFunc(&calcNewStateTokenCount, handle, "calc_new_state_tokencount")
	purego.RegisterLibFunc(&calcOldStateKV, handle, "calc_old_state_kv")
	purego.RegisterLibFunc(&calcOldStateTokenCount, handle, "calc_old_state_tokencount")
	purego.RegisterLibFunc(&saveStateKV, handle, "save_state_kv")
	purego.RegisterLibFunc(&loadStateKV, handle, "load_state_kv")
	purego.RegisterLibFunc(&clearStateKV, handle, "clear_state_kv")

	// Register SD functions
	purego.RegisterLibFunc(&sdLoadModel, handle, "sd_load_model")
	purego.RegisterLibFunc(&sdGenerate, handle, "sd_generate")
	purego.RegisterLibFunc(&sdUpscale, handle, "sd_upscale")
	purego.RegisterLibFunc(&sdGetInfo, handle, "sd_get_info")

	// Register Whisper functions
	purego.RegisterLibFunc(&whisperLoadModel, handle, "whisper_load_model")
	purego.RegisterLibFunc(&whisperGenerate, handle, "whisper_generate")

	// Register TTS functions
	purego.RegisterLibFunc(&ttsLoadModel, handle, "tts_load_model")
	purego.RegisterLibFunc(&ttsGenerate, handle, "tts_generate")

	// Register Embeddings functions
	purego.RegisterLibFunc(&embeddingsLoadModel, handle, "embeddings_load_model")
	purego.RegisterLibFunc(&embeddingsGenerate, handle, "embeddings_generate")

	return nil
}

// Helper to convert Go string to C string
// Returns pointer and keeps reference in registry to prevent GC
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
func LoadModel(inputs *LoadModelInputs) bool {
	return loadModel(inputs)
}

// Generate generates text - matches Python generate()
func Generate(inputs *GenerationInputs) GenerationOutputs {
	return generate(inputs)
}

// SDLoadModel loads SD model - matches Python sd_load_model()
func SDLoadModel(inputs *SDLoadModelInputs) bool {
	return sdLoadModel(inputs)
}

// SDGenerate generates image - matches Python sd_generate()
func SDGenerate(inputs *SDGenerationInputs) SDGenerationOutputs {
	return sdGenerate(inputs)
}

// WhisperLoadModel loads Whisper model - matches Python whisper_load_model()
func WhisperLoadModel(inputs *WhisperLoadModelInputs) bool {
	return whisperLoadModel(inputs)
}

// WhisperGenerate transcribes audio - matches Python whisper_generate()
func WhisperGenerate(inputs *WhisperGenerationInputs) WhisperGenerationOutputs {
	return whisperGenerate(inputs)
}

// TTSLoadModel loads TTS model - matches Python tts_load_model()
func TTSLoadModel(inputs *TTSLoadModelInputs) bool {
	return ttsLoadModel(inputs)
}

// TTSGenerate generates speech - matches Python tts_generate()
func TTSGenerate(inputs *TTSGenerationInputs) TTSGenerationOutputs {
	return ttsGenerate(inputs)
}

// EmbeddingsLoadModel loads embeddings model - matches Python embeddings_load_model()
func EmbeddingsLoadModel(inputs *EmbeddingsLoadModelInputs) bool {
	return embeddingsLoadModel(inputs)
}

// EmbeddingsGenerate generates embeddings - matches Python embeddings_generate()
func EmbeddingsGenerate(inputs *EmbeddingsGenerationInputs) EmbeddingsGenerationOutputs {
	return embeddingsGenerate(inputs)
}

// HasFinished checks if generation is finished
func HasFinished() bool {
	return hasFinished()
}

// AbortGenerate aborts current generation
func AbortGenerate() bool {
	return abortGenerate()
}

// GetPendingOutput gets pending output
func GetPendingOutput() string {
	return GoString(getPendingOutput())
}

// GetChatTemplate gets chat template
func GetChatTemplate() string {
	return GoString(getChatTemplate())
}

// TokenCount counts tokens in text
func TokenCount(text string, addSpecial bool) TokenCountOutputs {
	return tokenCount(CString(text), addSpecial)
}

// GetStreamCount gets current stream count
func GetStreamCount() int32 {
	return getStreamCount()
}

// NewToken gets token at index
func NewToken(index int32) string {
	return GoString(newToken(index))
}

// GetLastEvalTime gets last evaluation time
func GetLastEvalTime() float32 {
	return getLastEvalTime()
}

// GetLastProcessTime gets last process time
func GetLastProcessTime() float32 {
	return getLastProcessTime()
}

// GetLastTokenCount gets last token count
func GetLastTokenCount() int32 {
	return getLastTokenCount()
}

// GetLastSeed gets last seed used
func GetLastSeed() int32 {
	return getLastSeed()
}

// GetTotalGens gets total generations
func GetTotalGens() int32 {
	return getTotalGens()
}

// SaveStateKV saves KV cache to slot
func SaveStateKV(slot int32) uint64 {
	return saveStateKV(slot)
}

// LoadStateKV loads KV cache from slot
func LoadStateKV(slot int32) bool {
	return loadStateKV(slot)
}

// ClearStateKV clears all saved states
func ClearStateKV() bool {
	return clearStateKV()
}
