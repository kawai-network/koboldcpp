package koboldcpp

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	TensorSplitMax = 16
	ImagesMax      = 8
	AudioMax       = 4
	OverrideKVMax  = 4
)

// LoadModelInputs represents the parameters for loading a language model
type LoadModelInputs struct {
	Threads               int
	BlasThreads           int
	MaxContextLength      int
	LowVRAM               bool
	UseMMQ                bool
	UseRowSplit           bool
	ExecutablePath        string
	ModelFilename         string
	LoraFilename          string
	DraftModelFilename    string
	DraftAmount           int
	DraftGPULayers        int
	DraftGPUSplit         []float32 // max 16
	MMProjFilename        string
	MMProjCPU             bool
	VisionMaxRes          int
	UseMMap               bool
	UseMLock              bool
	UseSmartContext       bool
	UseContextShift       bool
	UseFastForward        bool
	MainGPU               int
	VulkanInfo            string
	BatchSize             int
	Autofit               bool
	AutofitTaxMB          int
	GPULayers             int
	RopeFreqScale         float32
	RopeFreqBase          float32
	OverrideNativeContext int
	MoeExperts            int
	MoeCPU                int
	NoBOSToken            bool
	LoadGuidance          bool
	OverrideKV            []string // max 4
	OverrideTensors       string
	FlashAttention        bool
	TensorSplit           []float32 // max 16
	QuantK                int
	QuantV                int
	CheckSlowness         bool
	HighPriority          bool
	SWASupport            bool
	SmartCache            bool
	SmartCacheSlots       int
	PipelineParallel      bool
	LoraMultiplier        float32
	DevicesOverride       string
	Quiet                 bool
	DebugMode             int
}

// GenerationInputs represents the parameters for text generation
type GenerationInputs struct {
	Seed                int
	Prompt              string
	Memory              string
	NegativePrompt      string
	GuidanceScale       float32
	Images              []string // max 8
	Audio               []string // max 4
	MaxContextLength    int
	MaxLength           int
	Temperature         float32
	TopK                int
	TopA                float32
	TopP                float32
	MinP                float32
	TypicalP            float32
	TFS                 float32
	NSigma              float32
	RepPen              float32
	RepPenRange         int
	RepPenSlope         float32
	PresencePenalty     float32
	Mirostat            int
	MirostatTau         float32
	MirostatEta         float32
	XTCThreshold        float32
	XTCProbability      float32
	SamplerOrder        []int // sampler IDs
	AllowEOSToken       bool
	BypassEOSToken      bool
	ToolCallFix         bool
	RenderSpecial       bool
	StreamSSE           bool
	Grammar             string
	GrammarRetainState  bool
	DynatempRange       float32
	DynatempExponent    float32
	SmoothingFactor     float32
	SmoothingCurve      float32
	AdaptiveTarget      float32
	AdaptiveDecay       float32
	DryMultiplier       float32
	DryBase             float32
	DryAllowedLength    int
	DryPenaltyLastN     int
	DrySequenceBreakers []string
	StopSequence        []string
	LogitBiases         map[int]float32 // token_id -> bias
	BannedTokens        []string
}

// GenerationOutputs represents the output from text generation
type GenerationOutputs struct {
	Status           int
	StopReason       int
	PromptTokens     int
	CompletionTokens int
	Text             string
}

// TokenCountOutputs represents the output from token counting
type TokenCountOutputs struct {
	Count int
	IDs   []int
}

// C struct definitions matching expose.h
type cLoadModelInputs struct {
	threads               int32
	blasthreads           int32
	max_context_length    int32
	low_vram              bool
	use_mmq               bool
	use_rowsplit          bool
	executable_path       uintptr
	model_filename        uintptr
	lora_filename         uintptr
	draftmodel_filename   uintptr
	draft_amount          int32
	draft_gpulayers       int32
	draft_gpusplit        [TensorSplitMax]float32
	mmproj_filename       uintptr
	mmproj_cpu            bool
	visionmaxres          int32
	use_mmap              bool
	use_mlock             bool
	use_smartcontext      bool
	use_contextshift      bool
	use_fastforward       bool
	kcpp_main_gpu         int32
	vulkan_info           uintptr
	batchsize             int32
	autofit               bool
	autofit_tax_mb        int32
	gpulayers             int32
	rope_freq_scale       float32
	rope_freq_base        float32
	overridenativecontext int32
	moe_experts           int32
	moecpu                int32
	no_bos_token          bool
	load_guidance         bool
	override_kv           [OverrideKVMax]uintptr
	override_tensors      uintptr
	flash_attention       bool
	tensor_split          [TensorSplitMax]float32
	quant_k               int32
	quant_v               int32
	check_slowness        bool
	highpriority          bool
	swa_support           bool
	smartcache            bool
	smartcacheslots       int32
	pipelineparallel      bool
	lora_multiplier       float32
	devices_override      uintptr
	quiet                 bool
	debugmode             int32
}

type cGenerationInputs struct {
	seed                      int32
	prompt                    uintptr
	memory                    uintptr
	negative_prompt           uintptr
	guidance_scale            float32
	images                    [ImagesMax]uintptr
	audio                     [AudioMax]uintptr
	max_context_length        int32
	max_length                int32
	temperature               float32
	top_k                     int32
	top_a                     float32
	top_p                     float32
	min_p                     float32
	typical_p                 float32
	tfs                       float32
	nsigma                    float32
	rep_pen                   float32
	rep_pen_range             int32
	rep_pen_slope             float32
	presence_penalty          float32
	mirostat                  int32
	mirostat_tau              float32
	mirostat_eta              float32
	xtc_threshold             float32
	xtc_probability           float32
	sampler_order             [7]int32 // KCPP_SAMPLER_MAX
	sampler_len               int32
	allow_eos_token           bool
	bypass_eos_token          bool
	tool_call_fix             bool
	render_special            bool
	stream_sse                bool
	grammar                   uintptr
	grammar_retain_state      bool
	dynatemp_range            float32
	dynatemp_exponent         float32
	smoothing_factor          float32
	smoothing_curve           float32
	adaptive_target           float32
	adaptive_decay            float32
	dry_multiplier            float32
	dry_base                  float32
	dry_allowed_length        int32
	dry_penalty_last_n        int32
	dry_sequence_breakers_len int32
	dry_sequence_breakers     uintptr
	stop_sequence_len         int32
	stop_sequence             uintptr
	logit_biases_len          int32
	logit_biases              uintptr
	banned_tokens_len         int32
	banned_tokens             uintptr
}

type cGenerationOutputs struct {
	status            int32
	stopreason        int32
	prompt_tokens     int32
	completion_tokens int32
	text              uintptr
}

type cTokenCountOutputs struct {
	count int32
	ids   uintptr
}

var (
	loadModelFunc      func(inputs *cLoadModelInputs) bool
	generateFunc       func(inputs *cGenerationInputs, outputs *cGenerationOutputs)
	abortGenerateFunc  func() bool
	tokenCountFunc     func(input uintptr, addbos bool, output *cTokenCountOutputs)
	newTokenFunc       func(idx int32) uintptr
	getStreamCountFunc func() int32
	hasFinishedFunc    func() bool
)

// initLlamaFunctions initializes the Llama function pointers
func initLlamaFunctions(handle uintptr) error {
	// load_model
	loadModelPtr, err := dlsymPlatform(handle, "load_model")
	if err != nil {
		return fmt.Errorf("failed to load load_model: %w", err)
	}
	purego.RegisterFunc(&loadModelFunc, loadModelPtr)

	// generate_ptr (pointer-based for cross-platform compatibility)
	generatePtr, err := dlsymPlatform(handle, "generate_ptr")
	if err != nil {
		return fmt.Errorf("failed to load generate_ptr: %w", err)
	}
	purego.RegisterFunc(&generateFunc, generatePtr)

	// abort_generate
	abortGeneratePtr, err := dlsymPlatform(handle, "abort_generate")
	if err != nil {
		return fmt.Errorf("failed to load abort_generate: %w", err)
	}
	purego.RegisterFunc(&abortGenerateFunc, abortGeneratePtr)

	// token_count_ptr (pointer-based for cross-platform compatibility)
	tokenCountPtr, err := dlsymPlatform(handle, "token_count_ptr")
	if err != nil {
		return fmt.Errorf("failed to load token_count_ptr: %w", err)
	}
	purego.RegisterFunc(&tokenCountFunc, tokenCountPtr)

	// new_token
	newTokenPtr, err := dlsymPlatform(handle, "new_token")
	if err != nil {
		return fmt.Errorf("failed to load new_token: %w", err)
	}
	purego.RegisterFunc(&newTokenFunc, newTokenPtr)

	// get_stream_count
	getStreamCountPtr, err := dlsymPlatform(handle, "get_stream_count")
	if err != nil {
		return fmt.Errorf("failed to load get_stream_count: %w", err)
	}
	purego.RegisterFunc(&getStreamCountFunc, getStreamCountPtr)

	// has_finished
	hasFinishedPtr, err := dlsymPlatform(handle, "has_finished")
	if err != nil {
		return fmt.Errorf("failed to load has_finished: %w", err)
	}
	purego.RegisterFunc(&hasFinishedFunc, hasFinishedPtr)

	return nil
}

// LoadModel loads a language model
func (k *KoboldCpp) LoadModel(inputs LoadModelInputs) error {
	if k.handle == 0 {
		return ErrLibraryNotLoaded
	}

	// Helper function to convert string to C string pointer (empty string if not provided)
	toCString := func(s string) (uintptr, []byte) {
		bytes := append([]byte(s), 0)
		return uintptr(unsafe.Pointer(&bytes[0])), bytes
	}

	// Convert Go strings to C strings (always non-NULL, can be empty string)
	executablePathPtr, executablePathBytes := toCString(inputs.ExecutablePath)
	modelFilenamePtr, modelFilenameBytes := toCString(inputs.ModelFilename)
	loraFilenamePtr, loraFilenameBytes := toCString(inputs.LoraFilename)
	draftModelFilenamePtr, draftModelFilenameBytes := toCString(inputs.DraftModelFilename)
	mmprojFilenamePtr, mmprojFilenameBytes := toCString(inputs.MMProjFilename)
	vulkanInfoPtr, vulkanInfoBytes := toCString(inputs.VulkanInfo)
	overrideTensorsPtr, overrideTensorsBytes := toCString(inputs.OverrideTensors)
	devicesOverridePtr, devicesOverrideBytes := toCString(inputs.DevicesOverride)

	// Prepare override_kv array
	var overrideKVPtrs [OverrideKVMax]uintptr
	var overrideKVBytes [][]byte
	for i := 0; i < OverrideKVMax && i < len(inputs.OverrideKV); i++ {
		kvBytes := append([]byte(inputs.OverrideKV[i]), 0)
		overrideKVBytes = append(overrideKVBytes, kvBytes)
		overrideKVPtrs[i] = uintptr(unsafe.Pointer(&kvBytes[0]))
	}

	// Prepare tensor_split and draft_gpusplit arrays
	var tensorSplit [TensorSplitMax]float32
	for i := 0; i < TensorSplitMax && i < len(inputs.TensorSplit); i++ {
		tensorSplit[i] = inputs.TensorSplit[i]
	}

	var draftGPUSplit [TensorSplitMax]float32
	for i := 0; i < TensorSplitMax && i < len(inputs.DraftGPUSplit); i++ {
		draftGPUSplit[i] = inputs.DraftGPUSplit[i]
	}

	cInputs := cLoadModelInputs{
		threads:               int32(inputs.Threads),
		blasthreads:           int32(inputs.BlasThreads),
		max_context_length:    int32(inputs.MaxContextLength),
		low_vram:              inputs.LowVRAM,
		use_mmq:               inputs.UseMMQ,
		use_rowsplit:          inputs.UseRowSplit,
		executable_path:       executablePathPtr,
		model_filename:        modelFilenamePtr,
		lora_filename:         loraFilenamePtr,
		draftmodel_filename:   draftModelFilenamePtr,
		draft_amount:          int32(inputs.DraftAmount),
		draft_gpulayers:       int32(inputs.DraftGPULayers),
		draft_gpusplit:        draftGPUSplit,
		mmproj_filename:       mmprojFilenamePtr,
		mmproj_cpu:            inputs.MMProjCPU,
		visionmaxres:          int32(inputs.VisionMaxRes),
		use_mmap:              inputs.UseMMap,
		use_mlock:             inputs.UseMLock,
		use_smartcontext:      inputs.UseSmartContext,
		use_contextshift:      inputs.UseContextShift,
		use_fastforward:       inputs.UseFastForward,
		kcpp_main_gpu:         int32(inputs.MainGPU),
		vulkan_info:           vulkanInfoPtr,
		batchsize:             int32(inputs.BatchSize),
		autofit:               inputs.Autofit,
		autofit_tax_mb:        int32(inputs.AutofitTaxMB),
		gpulayers:             int32(inputs.GPULayers),
		rope_freq_scale:       inputs.RopeFreqScale,
		rope_freq_base:        inputs.RopeFreqBase,
		overridenativecontext: int32(inputs.OverrideNativeContext),
		moe_experts:           int32(inputs.MoeExperts),
		moecpu:                int32(inputs.MoeCPU),
		no_bos_token:          inputs.NoBOSToken,
		load_guidance:         inputs.LoadGuidance,
		override_kv:           overrideKVPtrs,
		override_tensors:      overrideTensorsPtr,
		flash_attention:       inputs.FlashAttention,
		tensor_split:          tensorSplit,
		quant_k:               int32(inputs.QuantK),
		quant_v:               int32(inputs.QuantV),
		check_slowness:        inputs.CheckSlowness,
		highpriority:          inputs.HighPriority,
		swa_support:           inputs.SWASupport,
		smartcache:            inputs.SmartCache,
		smartcacheslots:       int32(inputs.SmartCacheSlots),
		pipelineparallel:      inputs.PipelineParallel,
		lora_multiplier:       inputs.LoraMultiplier,
		devices_override:      devicesOverridePtr,
		quiet:                 inputs.Quiet,
		debugmode:             int32(inputs.DebugMode),
	}

	success := loadModelFunc(&cInputs)

	// Keep byte slices alive
	runtime.KeepAlive(executablePathBytes)
	runtime.KeepAlive(modelFilenameBytes)
	runtime.KeepAlive(loraFilenameBytes)
	runtime.KeepAlive(draftModelFilenameBytes)
	runtime.KeepAlive(mmprojFilenameBytes)
	runtime.KeepAlive(vulkanInfoBytes)
	runtime.KeepAlive(overrideTensorsBytes)
	runtime.KeepAlive(devicesOverrideBytes)
	runtime.KeepAlive(overrideKVBytes)

	if !success {
		return fmt.Errorf("failed to load model: %s", inputs.ModelFilename)
	}

	return nil
}

// Generate generates text from the loaded model
func (k *KoboldCpp) Generate(inputs GenerationInputs) (*GenerationOutputs, error) {
	if k.handle == 0 {
		return nil, ErrLibraryNotLoaded
	}

	// Convert Go strings to C strings
	promptBytes := append([]byte(inputs.Prompt), 0)
	memoryBytes := append([]byte(inputs.Memory), 0)
	negativePromptBytes := append([]byte(inputs.NegativePrompt), 0)
	grammarBytes := append([]byte(inputs.Grammar), 0)

	// Prepare images array
	var imagesPtrs [ImagesMax]uintptr
	var imagesBytes [][]byte
	for i := 0; i < ImagesMax && i < len(inputs.Images); i++ {
		imgBytes := append([]byte(inputs.Images[i]), 0)
		imagesBytes = append(imagesBytes, imgBytes)
		imagesPtrs[i] = uintptr(unsafe.Pointer(&imgBytes[0]))
	}

	// Prepare audio array
	var audioPtrs [AudioMax]uintptr
	var audioBytes [][]byte
	for i := 0; i < AudioMax && i < len(inputs.Audio); i++ {
		audBytes := append([]byte(inputs.Audio[i]), 0)
		audioBytes = append(audioBytes, audBytes)
		audioPtrs[i] = uintptr(unsafe.Pointer(&audBytes[0]))
	}

	// Prepare sampler_order
	var samplerOrder [7]int32
	for i := 0; i < 7 && i < len(inputs.SamplerOrder); i++ {
		samplerOrder[i] = int32(inputs.SamplerOrder[i])
	}

	cInputs := cGenerationInputs{
		seed:                 int32(inputs.Seed),
		prompt:               uintptr(unsafe.Pointer(&promptBytes[0])),
		memory:               uintptr(unsafe.Pointer(&memoryBytes[0])),
		negative_prompt:      uintptr(unsafe.Pointer(&negativePromptBytes[0])),
		guidance_scale:       inputs.GuidanceScale,
		images:               imagesPtrs,
		audio:                audioPtrs,
		max_context_length:   int32(inputs.MaxContextLength),
		max_length:           int32(inputs.MaxLength),
		temperature:          inputs.Temperature,
		top_k:                int32(inputs.TopK),
		top_a:                inputs.TopA,
		top_p:                inputs.TopP,
		min_p:                inputs.MinP,
		typical_p:            inputs.TypicalP,
		tfs:                  inputs.TFS,
		nsigma:               inputs.NSigma,
		rep_pen:              inputs.RepPen,
		rep_pen_range:        int32(inputs.RepPenRange),
		rep_pen_slope:        inputs.RepPenSlope,
		presence_penalty:     inputs.PresencePenalty,
		mirostat:             int32(inputs.Mirostat),
		mirostat_tau:         inputs.MirostatTau,
		mirostat_eta:         inputs.MirostatEta,
		xtc_threshold:        inputs.XTCThreshold,
		xtc_probability:      inputs.XTCProbability,
		sampler_order:        samplerOrder,
		sampler_len:          int32(len(inputs.SamplerOrder)),
		allow_eos_token:      inputs.AllowEOSToken,
		bypass_eos_token:     inputs.BypassEOSToken,
		tool_call_fix:        inputs.ToolCallFix,
		render_special:       inputs.RenderSpecial,
		stream_sse:           inputs.StreamSSE,
		grammar:              uintptr(unsafe.Pointer(&grammarBytes[0])),
		grammar_retain_state: inputs.GrammarRetainState,
		dynatemp_range:       inputs.DynatempRange,
		dynatemp_exponent:    inputs.DynatempExponent,
		smoothing_factor:     inputs.SmoothingFactor,
		smoothing_curve:      inputs.SmoothingCurve,
		adaptive_target:      inputs.AdaptiveTarget,
		adaptive_decay:       inputs.AdaptiveDecay,
		dry_multiplier:       inputs.DryMultiplier,
		dry_base:             inputs.DryBase,
		dry_allowed_length:   int32(inputs.DryAllowedLength),
		dry_penalty_last_n:   int32(inputs.DryPenaltyLastN),
	}

	// TODO: Handle arrays: dry_sequence_breakers, stop_sequence, logit_biases, banned_tokens

	var cOutputs cGenerationOutputs
	generateFunc(&cInputs, &cOutputs)

	// Keep byte slices alive
	runtime.KeepAlive(promptBytes)
	runtime.KeepAlive(memoryBytes)
	runtime.KeepAlive(negativePromptBytes)
	runtime.KeepAlive(grammarBytes)
	runtime.KeepAlive(imagesBytes)
	runtime.KeepAlive(audioBytes)

	outputs := &GenerationOutputs{
		Status:           int(cOutputs.status),
		StopReason:       int(cOutputs.stopreason),
		PromptTokens:     int(cOutputs.prompt_tokens),
		CompletionTokens: int(cOutputs.completion_tokens),
		Text:             charPtrToString(cOutputs.text),
	}

	return outputs, nil
}

// AbortGenerate stops the current generation
func (k *KoboldCpp) AbortGenerate() error {
	if k.handle == 0 {
		return ErrLibraryNotLoaded
	}

	success := abortGenerateFunc()
	if !success {
		return fmt.Errorf("failed to abort generation")
	}

	return nil
}

// TokenCount counts tokens in the input text
func (k *KoboldCpp) TokenCount(input string, addBOS bool) (*TokenCountOutputs, error) {
	if k.handle == 0 {
		return nil, ErrLibraryNotLoaded
	}

	inputBytes := append([]byte(input), 0)
	var cOutputs cTokenCountOutputs
	tokenCountFunc(uintptr(unsafe.Pointer(&inputBytes[0])), addBOS, &cOutputs)

	runtime.KeepAlive(inputBytes)

	// Convert C array to Go slice
	var ids []int
	if cOutputs.count > 0 && cOutputs.ids != 0 {
		// Convert uintptr to pointer for array access
		// This is safe because we know the array size from count
		// and the memory is managed by the C library
		//nolint:unsafeptr // Required for FFI with C arrays
		idsSlice := unsafe.Slice((*int32)(unsafe.Pointer(cOutputs.ids)), cOutputs.count)
		ids = make([]int, cOutputs.count)
		for i := 0; i < int(cOutputs.count); i++ {
			ids[i] = int(idsSlice[i])
		}
	}

	return &TokenCountOutputs{
		Count: int(cOutputs.count),
		IDs:   ids,
	}, nil
}

// GetStreamToken gets a token from the streaming output
func (k *KoboldCpp) GetStreamToken(idx int) (string, error) {
	if k.handle == 0 {
		return "", ErrLibraryNotLoaded
	}

	tokenPtr := newTokenFunc(int32(idx))
	if tokenPtr == 0 {
		return "", fmt.Errorf("token index %d out of range", idx)
	}

	return charPtrToString(tokenPtr), nil
}

// GetStreamCount gets the number of tokens in the stream
func (k *KoboldCpp) GetStreamCount() int {
	if k.handle == 0 {
		return 0
	}

	return int(getStreamCountFunc())
}

// HasFinished checks if generation has finished
func (k *KoboldCpp) HasFinished() bool {
	if k.handle == 0 {
		return true
	}

	return hasFinishedFunc()
}
