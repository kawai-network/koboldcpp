package koboldcpp

import (
	"encoding/base64"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// SDLoadModelInputs represents the parameters for loading a Stable Diffusion model
type SDLoadModelInputs struct {
	ModelFilename       string
	ExecutablePath      string
	MainGPU             int
	VulkanInfo          string
	Threads             int
	Quant               int
	FlashAttention      bool
	OffloadCPU          bool
	VAECPU              bool
	ClipCPU             bool
	DiffusionConvDirect bool
	VAEConvDirect       bool
	TAESD               bool
	TiledVAEThreshold   int
	T5XXLFilename       string
	Clip1Filename       string
	Clip2Filename       string
	VAEFilename         string
	LoraFilename        string
	LoraMultiplier      float32
	LoraApplyMode       int
	PhotomakerFilename  string
	UpscalerFilename    string
	ImgHardLimit        int
	ImgSoftLimit        int
	DevicesOverride     string
	Quiet               bool
	DebugMode           int
}

// SDGenerationInputs represents the parameters for generating images
type SDGenerationInputs struct {
	Prompt            string
	NegativePrompt    string
	InitImages        string // Base64 encoded image
	Mask              string // Base64 encoded mask
	ExtraImages       []string
	FlipMask          bool
	DenoisingStrength float32
	CFGScale          float32
	DistilledGuidance float32
	ShiftedTimestep   int
	SampleSteps       int
	Width             int
	Height            int
	Seed              int
	SampleMethod      string
	Scheduler         string
	ClipSkip          int
	VidReqFrames      int
	VideoOutputType   int // 0=gif, 1=avi, 2=both
	RemoveLimits      bool
	CircularX         bool
	CircularY         bool
	Upscale           bool
}

// SDGenerationOutputs represents the output from image generation
type SDGenerationOutputs struct {
	Status    int
	Animated  int
	Data      string // Base64 encoded image
	DataExtra string // Base64 encoded extra data
}

// SDUpscaleInputs represents the parameters for upscaling images
type SDUpscaleInputs struct {
	InitImages      string // Base64 encoded image
	UpscalingResize int
}

// SDInfoOutputs represents the output from getting SD info
type SDInfoOutputs struct {
	Status int
	Data   string
}

// C struct definitions matching expose.h
type cSDLoadModelInputs struct {
	modelFilename       uintptr
	executablePath      uintptr
	mainGPU             int32
	vulkanInfo          uintptr
	threads             int32
	quant               int32
	flashAttention      bool
	offloadCPU          bool
	vaeCPU              bool
	clipCPU             bool
	diffusionConvDirect bool
	vaeConvDirect       bool
	taesd               bool
	tiledVAEThreshold   int32
	t5xxlFilename       uintptr
	clip1Filename       uintptr
	clip2Filename       uintptr
	vaeFilename         uintptr
	loraFilename        uintptr
	loraMultiplier      float32
	loraApplyMode       int32
	photomakerFilename  uintptr
	upscalerFilename    uintptr
	imgHardLimit        int32
	imgSoftLimit        int32
	devicesOverride     uintptr
	quiet               bool
	debugMode           int32
}

type cSDGenerationInputs struct {
	prompt            uintptr
	negativePrompt    uintptr
	initImages        uintptr
	mask              uintptr
	extraImagesLen    int32
	extraImages       uintptr
	flipMask          bool
	denoisingStrength float32
	cfgScale          float32
	distilledGuidance float32
	shiftedTimestep   int32
	sampleSteps       int32
	width             int32
	height            int32
	seed              int32
	sampleMethod      uintptr
	scheduler         uintptr
	clipSkip          int32
	vidReqFrames      int32
	videoOutputType   int32
	removeLimits      bool
	circularX         bool
	circularY         bool
	upscale           bool
}

type cSDGenerationOutputs struct {
	status    int32
	animated  int32
	data      uintptr
	dataExtra uintptr
}

type cSDUpscaleInputs struct {
	initImages      uintptr
	upscalingResize int32
}

type cSDInfoOutputs struct {
	status int32
	data   uintptr
}

var (
	sdLoadModelPtr func(inputs *cSDLoadModelInputs) bool
	sdGeneratePtr  func(inputs *cSDGenerationInputs, outputs *cSDGenerationOutputs)
	sdUpscalePtr   func(inputs *cSDUpscaleInputs, outputs *cSDGenerationOutputs)
	sdGetInfoPtr   func(outputs *cSDInfoOutputs)
)

// initSDFunctions initializes the Stable Diffusion function pointers
func initSDFunctions(handle uintptr) error {
	// sd_load_model_ptr
	sdLoadModelPtrPtr, err := dlsymPlatform(handle, "sd_load_model_ptr")
	if err != nil {
		return fmt.Errorf("failed to load sd_load_model_ptr: %w", err)
	}
	purego.RegisterFunc(&sdLoadModelPtr, sdLoadModelPtrPtr)

	// sd_generate_ptr
	sdGeneratePtrPtr, err := dlsymPlatform(handle, "sd_generate_ptr")
	if err != nil {
		return fmt.Errorf("failed to load sd_generate_ptr: %w", err)
	}
	purego.RegisterFunc(&sdGeneratePtr, sdGeneratePtrPtr)

	// sd_upscale_ptr
	sdUpscalePtrPtr, err := dlsymPlatform(handle, "sd_upscale_ptr")
	if err != nil {
		return fmt.Errorf("failed to load sd_upscale_ptr: %w", err)
	}
	purego.RegisterFunc(&sdUpscalePtr, sdUpscalePtrPtr)

	// sd_get_info_ptr
	sdGetInfoPtrPtr, err := dlsymPlatform(handle, "sd_get_info_ptr")
	if err != nil {
		return fmt.Errorf("failed to load sd_get_info_ptr: %w", err)
	}
	purego.RegisterFunc(&sdGetInfoPtr, sdGetInfoPtrPtr)

	return nil
}

// LoadSDModel loads a Stable Diffusion model
func (k *KoboldCpp) LoadSDModel(inputs SDLoadModelInputs) error {
	if k.handle == 0 {
		return ErrLibraryNotLoaded
	}

	// Convert Go strings to C strings - keep byte slices alive
	modelFilenameBytes := append([]byte(inputs.ModelFilename), 0)
	executablePathBytes := append([]byte(inputs.ExecutablePath), 0)
	vulkanInfoBytes := append([]byte(inputs.VulkanInfo), 0)
	t5xxlFilenameBytes := append([]byte(inputs.T5XXLFilename), 0)
	clip1FilenameBytes := append([]byte(inputs.Clip1Filename), 0)
	clip2FilenameBytes := append([]byte(inputs.Clip2Filename), 0)
	vaeFilenameBytes := append([]byte(inputs.VAEFilename), 0)
	loraFilenameBytes := append([]byte(inputs.LoraFilename), 0)
	photomakerFilenameBytes := append([]byte(inputs.PhotomakerFilename), 0)
	upscalerFilenameBytes := append([]byte(inputs.UpscalerFilename), 0)
	devicesOverrideBytes := append([]byte(inputs.DevicesOverride), 0)

	cInputs := cSDLoadModelInputs{
		modelFilename:       uintptr(unsafe.Pointer(&modelFilenameBytes[0])),
		executablePath:      uintptr(unsafe.Pointer(&executablePathBytes[0])),
		mainGPU:             int32(inputs.MainGPU),
		vulkanInfo:          uintptr(unsafe.Pointer(&vulkanInfoBytes[0])),
		threads:             int32(inputs.Threads),
		quant:               int32(inputs.Quant),
		flashAttention:      inputs.FlashAttention,
		offloadCPU:          inputs.OffloadCPU,
		vaeCPU:              inputs.VAECPU,
		clipCPU:             inputs.ClipCPU,
		diffusionConvDirect: inputs.DiffusionConvDirect,
		vaeConvDirect:       inputs.VAEConvDirect,
		taesd:               inputs.TAESD,
		tiledVAEThreshold:   int32(inputs.TiledVAEThreshold),
		t5xxlFilename:       uintptr(unsafe.Pointer(&t5xxlFilenameBytes[0])),
		clip1Filename:       uintptr(unsafe.Pointer(&clip1FilenameBytes[0])),
		clip2Filename:       uintptr(unsafe.Pointer(&clip2FilenameBytes[0])),
		vaeFilename:         uintptr(unsafe.Pointer(&vaeFilenameBytes[0])),
		loraFilename:        uintptr(unsafe.Pointer(&loraFilenameBytes[0])),
		loraMultiplier:      inputs.LoraMultiplier,
		loraApplyMode:       int32(inputs.LoraApplyMode),
		photomakerFilename:  uintptr(unsafe.Pointer(&photomakerFilenameBytes[0])),
		upscalerFilename:    uintptr(unsafe.Pointer(&upscalerFilenameBytes[0])),
		imgHardLimit:        int32(inputs.ImgHardLimit),
		imgSoftLimit:        int32(inputs.ImgSoftLimit),
		devicesOverride:     uintptr(unsafe.Pointer(&devicesOverrideBytes[0])),
		quiet:               inputs.Quiet,
		debugMode:           int32(inputs.DebugMode),
	}

	success := sdLoadModelPtr(&cInputs)

	// Keep byte slices alive until after the C call
	runtime.KeepAlive(modelFilenameBytes)
	runtime.KeepAlive(executablePathBytes)
	runtime.KeepAlive(vulkanInfoBytes)
	runtime.KeepAlive(t5xxlFilenameBytes)
	runtime.KeepAlive(clip1FilenameBytes)
	runtime.KeepAlive(clip2FilenameBytes)
	runtime.KeepAlive(vaeFilenameBytes)
	runtime.KeepAlive(loraFilenameBytes)
	runtime.KeepAlive(photomakerFilenameBytes)
	runtime.KeepAlive(upscalerFilenameBytes)
	runtime.KeepAlive(devicesOverrideBytes)

	if !success {
		return fmt.Errorf("failed to load SD model: %s", inputs.ModelFilename)
	}

	return nil
}

// SDGenerate generates images using Stable Diffusion
func (k *KoboldCpp) SDGenerate(inputs SDGenerationInputs) (*SDGenerationOutputs, error) {
	if k.handle == 0 {
		return nil, ErrLibraryNotLoaded
	}

	// Convert Go strings to C strings - keep byte slices alive
	promptBytes := append([]byte(inputs.Prompt), 0)
	negativePromptBytes := append([]byte(inputs.NegativePrompt), 0)
	initImagesBytes := append([]byte(inputs.InitImages), 0)
	maskBytes := append([]byte(inputs.Mask), 0)
	sampleMethodBytes := append([]byte(inputs.SampleMethod), 0)
	schedulerBytes := append([]byte(inputs.Scheduler), 0)

	// Handle extra images array
	var extraImagesPtr uintptr
	var extraImagesPtrs []uintptr
	if len(inputs.ExtraImages) > 0 {
		extraImagesPtrs = make([]uintptr, len(inputs.ExtraImages))
		for i, img := range inputs.ExtraImages {
			imgBytes := append([]byte(img), 0)
			extraImagesPtrs[i] = uintptr(unsafe.Pointer(&imgBytes[0]))
			runtime.KeepAlive(imgBytes)
		}
		extraImagesPtr = uintptr(unsafe.Pointer(&extraImagesPtrs[0]))
	}

	cInputs := cSDGenerationInputs{
		prompt:            uintptr(unsafe.Pointer(&promptBytes[0])),
		negativePrompt:    uintptr(unsafe.Pointer(&negativePromptBytes[0])),
		initImages:        uintptr(unsafe.Pointer(&initImagesBytes[0])),
		mask:              uintptr(unsafe.Pointer(&maskBytes[0])),
		extraImagesLen:    int32(len(inputs.ExtraImages)),
		extraImages:       extraImagesPtr,
		flipMask:          inputs.FlipMask,
		denoisingStrength: inputs.DenoisingStrength,
		cfgScale:          inputs.CFGScale,
		distilledGuidance: inputs.DistilledGuidance,
		shiftedTimestep:   int32(inputs.ShiftedTimestep),
		sampleSteps:       int32(inputs.SampleSteps),
		width:             int32(inputs.Width),
		height:            int32(inputs.Height),
		seed:              int32(inputs.Seed),
		sampleMethod:      uintptr(unsafe.Pointer(&sampleMethodBytes[0])),
		scheduler:         uintptr(unsafe.Pointer(&schedulerBytes[0])),
		clipSkip:          int32(inputs.ClipSkip),
		vidReqFrames:      int32(inputs.VidReqFrames),
		videoOutputType:   int32(inputs.VideoOutputType),
		removeLimits:      inputs.RemoveLimits,
		circularX:         inputs.CircularX,
		circularY:         inputs.CircularY,
		upscale:           inputs.Upscale,
	}

	var cOutputs cSDGenerationOutputs
	sdGeneratePtr(&cInputs, &cOutputs)

	// Keep byte slices alive until after the C call
	runtime.KeepAlive(promptBytes)
	runtime.KeepAlive(negativePromptBytes)
	runtime.KeepAlive(initImagesBytes)
	runtime.KeepAlive(maskBytes)
	runtime.KeepAlive(sampleMethodBytes)
	runtime.KeepAlive(schedulerBytes)
	runtime.KeepAlive(extraImagesPtrs)

	outputs := &SDGenerationOutputs{
		Status:    int(cOutputs.status),
		Animated:  int(cOutputs.animated),
		Data:      charPtrToString(cOutputs.data),
		DataExtra: charPtrToString(cOutputs.dataExtra),
	}

	if outputs.Status != 1 {
		return outputs, fmt.Errorf("SD generation failed with status: %d", outputs.Status)
	}

	return outputs, nil
}

// SDUpscale upscales images using the loaded upscaler model
func (k *KoboldCpp) SDUpscale(inputs SDUpscaleInputs) (*SDGenerationOutputs, error) {
	if k.handle == 0 {
		return nil, ErrLibraryNotLoaded
	}

	// Convert Go strings to C strings - keep byte slices alive
	initImagesBytes := append([]byte(inputs.InitImages), 0)

	cInputs := cSDUpscaleInputs{
		initImages:      uintptr(unsafe.Pointer(&initImagesBytes[0])),
		upscalingResize: int32(inputs.UpscalingResize),
	}

	var cOutputs cSDGenerationOutputs
	sdUpscalePtr(&cInputs, &cOutputs)

	// Keep byte slices alive until after the C call
	runtime.KeepAlive(initImagesBytes)

	outputs := &SDGenerationOutputs{
		Status:    int(cOutputs.status),
		Animated:  int(cOutputs.animated),
		Data:      charPtrToString(cOutputs.data),
		DataExtra: charPtrToString(cOutputs.dataExtra),
	}

	if outputs.Status != 1 {
		return outputs, fmt.Errorf("SD upscale failed with status: %d", outputs.Status)
	}

	return outputs, nil
}

// SDGetInfo gets information about the loaded SD model
func (k *KoboldCpp) SDGetInfo() (*SDInfoOutputs, error) {
	if k.handle == 0 {
		return nil, ErrLibraryNotLoaded
	}

	var cOutputs cSDInfoOutputs
	sdGetInfoPtr(&cOutputs)

	outputs := &SDInfoOutputs{
		Status: int(cOutputs.status),
		Data:   charPtrToString(cOutputs.data),
	}

	return outputs, nil
}

// SDGenerateFromFile is a convenience function that reads an image file and generates from it
func (k *KoboldCpp) SDGenerateFromFile(prompt string, initImagePath string, params SDGenerationInputs) (*SDGenerationOutputs, error) {
	// Read init image if provided
	if initImagePath != "" {
		imageData, err := readFileToBase64(initImagePath)
		if err != nil {
			return nil, fmt.Errorf("failed to read init image: %w", err)
		}
		params.InitImages = imageData
	}

	params.Prompt = prompt
	return k.SDGenerate(params)
}

// Helper function to save generated image to file
func SaveImageToFile(base64Data string, outputPath string) error {
	// Decode base64
	imageData, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return fmt.Errorf("failed to decode base64 image: %w", err)
	}

	// Write to file
	return writeFile(outputPath, imageData)
}
