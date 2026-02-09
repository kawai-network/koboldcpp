package koboldcpp

/*
#include <stdint.h>
#include <stdbool.h>

// Define all the C structures that match the C++ header
const int tensor_split_max = 16;
const int images_max = 8;
const int audio_max = 4;
const int logprobs_max = 10;
const int overridekv_max = 4;

// Enums
enum samplers
{
    KCPP_SAMPLER_TOP_K=0,
    KCPP_SAMPLER_TOP_A=1,
    KCPP_SAMPLER_TOP_P=2,
    KCPP_SAMPLER_TFS=3,
    KCPP_SAMPLER_TYP=4,
    KCPP_SAMPLER_TEMP=5,
    KCPP_SAMPLER_REP_PEN=6,
    KCPP_SAMPLER_MAX=7
};

enum stop_reason
{
    ERROR_ENCOUNTERED=-2,
    INVALID=-1,
    OUT_OF_TOKENS=0,
    EOS_TOKEN_HIT=1,
    CUSTOM_STOPPER=2,
};

typedef struct {
    int32_t token_id;
    float bias;
} logit_bias;

typedef struct {
    int threads;
    int blasthreads;
    int max_context_length;
    bool low_vram;
    bool use_mmq;
    bool use_rowsplit;
    const char * executable_path;
    const char * model_filename;
    const char * lora_filename;
    const char * draftmodel_filename;
    int draft_amount;
    int draft_gpulayers;
    float draft_gpusplit[16];
    const char * mmproj_filename;
    bool mmproj_cpu;
    int visionmaxres;
    bool use_mmap;
    bool use_mlock;
    bool use_smartcontext;
    bool use_contextshift;
    bool use_fastforward;
    int kcpp_main_gpu;
    const char * vulkan_info;
    int batchsize;
    bool autofit;
    int autofit_tax_mb;
    int gpulayers;
    float rope_freq_scale;
    float rope_freq_base;
    int overridenativecontext;
    int moe_experts;
    int moecpu;
    bool no_bos_token;
    bool load_guidance;
    const char * override_kv[4];
    const char * override_tensors;
    bool flash_attention;
    float tensor_split[16];
    int quant_k;
    int quant_v;
    bool check_slowness;
    bool highpriority;
    bool swa_support;
    bool smartcache;
    int smartcacheslots;
    bool pipelineparallel;
    float lora_multiplier;
    const char * devices_override;
    bool quiet;
    int debugmode;
} load_model_inputs;

typedef struct {
    int seed;
    const char * prompt;
    const char * memory;
    const char * negative_prompt;
    float guidance_scale;
    const char * images[8];
    const char * audio[4];
    int max_context_length;
    int max_length;
    float temperature;
    int top_k;
    float top_a;
    float top_p;
    float min_p;
    float typical_p;
    float tfs;
    float nsigma;
    float rep_pen;
    int rep_pen_range;
    float rep_pen_slope;
    float presence_penalty;
    int mirostat;
    float mirostat_tau;
    float mirostat_eta;
    float xtc_threshold;
    float xtc_probability;
    enum samplers sampler_order[7]; // KCPP_SAMPLER_MAX = 7
    int sampler_len;
    bool allow_eos_token;
    bool bypass_eos_token;
    bool tool_call_fix;
    bool render_special;
    bool stream_sse;
    const char * grammar;
    bool grammar_retain_state;
    float dynatemp_range;
    float dynatemp_exponent;
    float smoothing_factor;
    float smoothing_curve;
    float adaptive_target;
    float adaptive_decay;
    float dry_multiplier;
    float dry_base;
    int dry_allowed_length;
    int dry_penalty_last_n;
    int dry_sequence_breakers_len;
    const char ** dry_sequence_breakers;
    int stop_sequence_len;
    const char ** stop_sequence;
    int logit_biases_len;
    const logit_bias * logit_biases;
    int banned_tokens_len;
    const char ** banned_tokens;
} generation_inputs;

typedef struct {
    int status;
    int stopreason;
    int prompt_tokens;
    int completion_tokens;
    const char * text;
} generation_outputs;

typedef struct {
    int count;
    int * ids;
} token_count_outputs;

typedef struct {
    int option_count;
    const char * selected_token;
    float selected_logprob;
    int32_t selected_token_id;
    const char * tokens[10];
    int32_t token_ids[10];
    float * logprobs;
} logprob_item;

typedef struct {
    int count;
    logprob_item * logprob_items;
} last_logprobs_outputs;

typedef struct {
    const char * model_filename;
    const char * executable_path;
    int kcpp_main_gpu;
    const char * vulkan_info;
    int threads;
    int quant;
    bool flash_attention;
    bool offload_cpu;
    bool vae_cpu;
    bool clip_cpu;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool taesd;
    int tiled_vae_threshold;
    const char * t5xxl_filename;
    const char * clip1_filename;
    const char * clip2_filename;
    const char * vae_filename;
    const char * lora_filename;
    float lora_multiplier;
    int lora_apply_mode;
    const char * photomaker_filename;
    const char * upscaler_filename;
    int img_hard_limit;
    int img_soft_limit;
    const char * devices_override;
    bool quiet;
    int debugmode;
} sd_load_model_inputs;

typedef struct {
    const char * prompt;
    const char * negative_prompt;
    const char * init_images;
    const char * mask;
    int extra_images_len;
    const char ** extra_images;
    bool flip_mask;
    float denoising_strength;
    float cfg_scale;
    float distilled_guidance;
    int shifted_timestep;
    int sample_steps;
    int width;
    int height;
    int seed;
    const char * sample_method;
    const char * scheduler;
    int clip_skip;
    int vid_req_frames;
    int video_output_type;
    bool remove_limits;
    bool circular_x;
    bool circular_y;
    bool upscale;
} sd_generation_inputs;

typedef struct {
    int status;
    int animated;
    const char * data;
    const char * data_extra;
} sd_generation_outputs;

typedef struct {
    const char * init_images;
    int upscaling_resize;
} sd_upscale_inputs;

typedef struct {
    int status;
    const char * data;
} sd_info_outputs;

typedef struct {
    const char * model_filename;
    const char * executable_path;
    int kcpp_main_gpu;
    const char * vulkan_info;
    const char * devices_override;
    bool quiet;
    int debugmode;
} whisper_load_model_inputs;

typedef struct {
    const char * prompt;
    const char * audio_data;
    bool suppress_non_speech;
    const char * langcode;
} whisper_generation_inputs;

typedef struct {
    int status;
    const char * text;
} whisper_generation_outputs;

typedef struct {
    int threads;
    const char * ttc_model_filename;
    const char * cts_model_filename;
    const char * executable_path;
    int kcpp_main_gpu;
    const char * vulkan_info;
    int gpulayers;
    bool flash_attention;
    int ttsmaxlen;
    const char * devices_override;
    bool quiet;
    int debugmode;
} tts_load_model_inputs;

typedef struct {
    const char * prompt;
    int speaker_seed;
    int audio_seed;
    const char * custom_speaker_voice;
    const char * custom_speaker_text;
    const char * custom_speaker_data;
} tts_generation_inputs;

typedef struct {
    int status;
    const char * data;
} tts_generation_outputs;

typedef struct {
    int threads;
    const char * model_filename;
    const char * executable_path;
    int kcpp_main_gpu;
    const char * vulkan_info;
    int gpulayers;
    bool flash_attention;
    bool use_mmap;
    int embeddingsmaxctx;
    const char * devices_override;
    bool quiet;
    int debugmode;
} embeddings_load_model_inputs;

typedef struct {
    const char * prompt;
    bool truncate;
} embeddings_generation_inputs;

typedef struct {
    int status;
    int count;
    const char * data;
} embeddings_generation_outputs;

// Function declarations
bool load_model(load_model_inputs inputs);
generation_outputs generate(generation_inputs inputs);
const char* new_token(int index);
int get_stream_count();
bool has_finished();
bool abort_generate();

bool has_audio_support();
bool has_vision_support();

float get_last_eval_time();
float get_last_process_time();
int get_last_token_count();
int get_last_input_count();
int get_last_seed();
int get_total_gens();
int get_total_img_gens();
int get_total_tts_gens();
int get_total_transcribe_gens();
int get_last_draft_success();
int get_last_draft_failed();
int get_last_stop_reason(); // returns enum stop_reason

token_count_outputs token_count(const char* text, bool add_special_tokens);
const char* get_pending_output();
const char* get_chat_template();
const char* detokenize(token_count_outputs tokens);
last_logprobs_outputs last_logprobs();

size_t calc_new_state_kv();
size_t calc_new_state_tokencount();
size_t calc_old_state_kv(int n);
size_t calc_old_state_tokencount(int n);
size_t save_state_kv(int slot);
bool load_state_kv(int slot);
bool clear_state_kv();

bool sd_load_model(sd_load_model_inputs inputs);
sd_generation_outputs sd_generate(sd_generation_inputs inputs);
sd_generation_outputs sd_upscale(sd_upscale_inputs inputs);
sd_info_outputs sd_get_info();

bool whisper_load_model(whisper_load_model_inputs inputs);
whisper_generation_outputs whisper_generate(whisper_generation_inputs inputs);

bool tts_load_model(tts_load_model_inputs inputs);
tts_generation_outputs tts_generate(tts_generation_inputs inputs);

bool embeddings_load_model(embeddings_load_model_inputs inputs);
embeddings_generation_outputs embeddings_generate(embeddings_generation_inputs inputs);

*/
import "C"
import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Constants
const (
	tensorSplitMax = 16
	imagesMax      = 8
	audioMax       = 4
	logprobsMax    = 10
	overridekvMax  = 4
)

// Enums
type Samplers int

const (
	KCPP_SAMPLER_TOP_K   Samplers = 0
	KCPP_SAMPLER_TOP_A   Samplers = 1
	KCPP_SAMPLER_TOP_P   Samplers = 2
	KCPP_SAMPLER_TFS     Samplers = 3
	KCPP_SAMPLER_TYP     Samplers = 4
	KCPP_SAMPLER_TEMP    Samplers = 5
	KCPP_SAMPLER_REP_PEN Samplers = 6
	KCPP_SAMPLER_MAX     Samplers = 7
)

type StopReason int

const (
	ERROR_ENCOUNTERED StopReason = -2
	INVALID           StopReason = -1
	OUT_OF_TOKENS     StopReason = 0
	EOS_TOKEN_HIT     StopReason = 1
	CUSTOM_STOPPER    StopReason = 2
)

// Structure definitions

type LogitBias struct {
	TokenID int32
	Bias    float32
}

type TokenCountOutputs struct {
	Count int
	IDs   []int
}

type LogprobItem struct {
	OptionCount       int
	SelectedToken     string
	SelectedLogprob   float32
	SelectedTokenID   int32
	Tokens            [logprobsMax]string
	TokenIDs          [logprobsMax]int32
	Logprobs          []float32
}

type LastLogprobsOutputs struct {
	Count        int
	LogprobItems []LogprobItem
}

type LoadModelInputs struct {
	Threads                 int
	Blasthreads             int
	MaxContextLength        int
	LowVram                 bool
	UseMMQ                  bool
	UseRowsplit             bool
	ExecutablePath          string
	ModelFilename           string
	LoraFilename            string
	DraftmodelFilename      string
	DraftAmount             int
	DraftGpulayers          int
	DraftGpusplit           [tensorSplitMax]float32
	MmprojFilename          string
	MmprojCPU               bool
	Visionmaxres            int
	UseMmap                 bool
	UseMlock                bool
	UseSmartcontext         bool
	UseContextshift         bool
	UseFastforward          bool
	KcppMainGPU             int
	VulkanInfo              string
	Batchsize               int
	Autofit                 bool
	AutofitTaxMB            int
	Gpulayers               int
	RopeFreqScale           float32
	RopeFreqBase            float32
	Overridenativecontext   int
	MoeExperts              int
	Moecpu                  int
	NoBosToken              bool
	LoadGuidance            bool
	OverrideKV              [overridekvMax]string
	OverrideTensors         string
	FlashAttention          bool
	TensorSplit             [tensorSplitMax]float32
	QuantK                  int
	QuantV                  int
	CheckSlowness           bool
	Highpriority            bool
	SwaSupport              bool
	Smartcache              bool
	Smartcacheslots         int
	Pipelineparallel        bool
	LoraMultiplier          float32
	DevicesOverride         string
	Quiet                   bool
	Debugmode               int
}

type GenerationInputs struct {
	Seed                    int
	Prompt                  string
	Memory                  string
	NegativePrompt          string
	GuidanceScale           float32
	Images                  [imagesMax]string
	Audio                   [audioMax]string
	MaxContextLength        int
	MaxLength               int
	Temperature             float32
	TopK                    int
	TopA                    float32
	TopP                    float32
	MinP                    float32
	TypicalP                float32
	TFS                     float32
	NSigma                  float32
	RepPen                  float32
	RepPenRange             int
	RepPenSlope             float32
	PresencePenalty         float32
	Mirostat                int
	MirostatTau             float32
	MirostatEta             float32
	XTCTreshold             float32
	XTCProbability          float32
	SamplerOrder            [KCPP_SAMPLER_MAX]Samplers
	SamplerLen              int
	AllowEOSToken           bool
	BypassEOSToken          bool
	ToolCallFix             bool
	RenderSpecial           bool
	StreamSSE               bool
	Grammar                 string
	GrammarRetainState      bool
	DynatempRange           float32
	DynatempExponent        float32
	SmoothingFactor         float32
	SmoothingCurve          float32
	AdaptiveTarget          float32
	AdaptiveDecay           float32
	DRYMultiplier           float32
	DRYBase                 float32
	DRYAllowedLength        int
	DRYPenaltyLastN         int
	DRYSequenceBreakersLen  int
	DRYSequenceBreakers     []string
	StopSequenceLen         int
	StopSequence            []string
	LogitBiasesLen          int
	LogitBiases             []LogitBias
	BannedTokensLen         int
	BannedTokens            []string
}

type GenerationOutputs struct {
	Status           int
	Stopreason       StopReason
	PromptTokens     int
	CompletionTokens int
	Text             string
}

type SDLoadModelInputs struct {
	ModelFilename       string
	ExecutablePath      string
	KcppMainGPU         int
	VulkanInfo          string
	Threads             int
	Quant               int
	FlashAttention      bool
	OffloadCPU          bool
	VAECPU              bool
	CLIPCPU             bool
	DiffusionConvDirect bool
	VAEConvDirect       bool
	TAESD               bool
	TiledVAEThreshold   int
	T5XXLFilename       string
	CLIP1Filename       string
	CLIP2Filename       string
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
	Debugmode           int
}

type SDGenerationInputs struct {
	Prompt                 string
	NegativePrompt         string
	InitImages             string
	Mask                   string
	ExtraImagesLen         int
	ExtraImages            []string
	FlipMask               bool
	DenoisingStrength      float32
	CFGScale               float32
	DistilledGuidance      float32
	ShiftedTimestep        int
	SampleSteps            int
	Width                  int
	Height                 int
	Seed                   int
	SampleMethod           string
	Scheduler              string
	ClipSkip               int
	VidReqFrames           int
	VideoOutputType        int
	RemoveLimits           bool
	CircularX              bool
	CircularY              bool
	Upscale                bool
}

type SDGenerationOutputs struct {
	Status     int
	Animated   int
	Data       string
	DataExtra  string
}

type SDUpscaleInputs struct {
	InitImages      string
	UpscalingResize int
}

type SDInfoOutputs struct {
	Status int
	Data   string
}

type WhisperLoadModelInputs struct {
	ModelFilename   string
	ExecutablePath  string
	KcppMainGPU     int
	VulkanInfo      string
	DevicesOverride string
	Quiet           bool
	Debugmode       int
}

type WhisperGenerationInputs struct {
	Prompt            string
	AudioData         string
	SuppressNonSpeech bool
	Langcode          string
}

type WhisperGenerationOutputs struct {
	Status int
	Text   string
}

type TTSLoadModelInputs struct {
	Threads           int
	TTCModelFilename  string
	CTSModelFilename  string
	ExecutablePath    string
	KcppMainGPU       int
	VulkanInfo        string
	Gpulayers         int
	FlashAttention    bool
	TTSMaxLen         int
	DevicesOverride   string
	Quiet             bool
	Debugmode         int
}

type TTSGenerationInputs struct {
	Prompt              string
	SpeakerSeed         int
	AudioSeed           int
	CustomSpeakerVoice  string
	CustomSpeakerText   string
	CustomSpeakerData   string
}

type TTSGenerationOutputs struct {
	Status int
	Data   string
}

type EmbeddingsLoadModelInputs struct {
	Threads           int
	ModelFilename     string
	ExecutablePath    string
	KcppMainGPU       int
	VulkanInfo        string
	Gpulayers         int
	FlashAttention    bool
	UseMmap           bool
	Embeddingsmaxctx  int
	DevicesOverride   string
	Quiet             bool
	Debugmode         int
}

type EmbeddingsGenerationInputs struct {
	Prompt   string
	Truncate bool
}

type EmbeddingsGenerationOutputs struct {
	Status int
	Count  int
	Data   string
}

// Library file names
var (
	LIB_DEFAULT           string
	LIB_FAILSAFE          string
	LIB_NOAVX2            string
	LIB_VULKAN_FAILSAFE   string
	LIB_CUBLAS            string
	LIB_HIPBLAS           string
	LIB_VULKAN            string
	LIB_VULKAN_NOAVX2     string
)

func init() {
	// Initialize library constants
	LIB_DEFAULT = pickExistantFile("koboldcpp_default.dll", "koboldcpp_default.so")
	LIB_FAILSAFE = pickExistantFile("koboldcpp_failsafe.dll", "koboldcpp_failsafe.so")
	LIB_NOAVX2 = pickExistantFile("koboldcpp_noavx2.dll", "koboldcpp_noavx2.so")
	LIB_VULKAN_FAILSAFE = pickExistantFile("koboldcpp_vulkan_failsafe.dll", "koboldcpp_vulkan_failsafe.so")
	LIB_CUBLAS = pickExistantFile("koboldcpp_cublas.dll", "koboldcpp_cublas.so")
	LIB_HIPBLAS = pickExistantFile("koboldcpp_hipblas.dll", "koboldcpp_hipblas.so")
	LIB_VULKAN = pickExistantFile("koboldcpp_vulkan.dll", "koboldcpp_vulkan.so")
	LIB_VULKAN_NOAVX2 = pickExistantFile("koboldcpp_vulkan_noavx2.dll", "koboldcpp_vulkan_noavx2.so")
}

func pickExistantFile(ntOption, nonNTOption string) string {
	precompiledPrefix := "precompiled_"
	
	fileExists := func(filename string) bool {
		_, err := os.Stat(filename)
		return err == nil
	}
	
	var ntExist, nonNTExist, precompiledNTExist, precompiledNonNTExist bool
	
	isWindows := runtime.GOOS == "windows"
	
	if isWindows {
		ntExist = fileExists(ntOption)
		nonNTExist = fileExists(nonNTOption)
		precompiledNTExist = fileExists(precompiledPrefix + ntOption)
		precompiledNonNTExist = fileExists(precompiledPrefix + nonNTOption)
		
		if !ntExist && precompiledNTExist {
			return precompiledPrefix + ntOption
		}
		if nonNTExist && !ntExist {
			return nonNTOption
		}
		return ntOption
	} else {
		ntExist = fileExists(ntOption)
		nonNTExist = fileExists(nonNTOption)
		precompiledNTExist = fileExists(precompiledPrefix + ntOption)
		precompiledNonNTExist = fileExists(precompiledPrefix + nonNTOption)
		
		if !nonNTExist && precompiledNonNTExist {
			return precompiledPrefix + nonNTOption
		}
		if ntExist && !nonNTExist {
			return ntOption
		}
		return nonNTOption
	}
}

// LibraryHandle represents a handle to the loaded library
type LibraryHandle struct {
	lib uintptr
}

// InitLibrary initializes the KoboldCpp C++ library and sets up all function bindings
func InitLibrary(libName string, dirPath string) (*LibraryHandle, error) {
	libPath := filepath.Join(dirPath, libName)
	
	lib, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("failed to load library %s: %w", libPath, err)
	}
	
	handle := &LibraryHandle{lib: lib}
	
	// Set up function bindings
	handle.setupFunctionBindings()
	
	return handle, nil
}

func (h *LibraryHandle) setupFunctionBindings() {
	// Text/GGUF model functions
	purego.RegisterLibFunc(&h.loadModel, h.lib, "load_model")
	purego.RegisterLibFunc(&h.generate, h.lib, "generate")
	purego.RegisterLibFunc(&h.newToken, h.lib, "new_token")
	purego.RegisterLibFunc(&h.getStreamCount, h.lib, "get_stream_count")
	purego.RegisterLibFunc(&h.hasFinished, h.lib, "has_finished")
	purego.RegisterLibFunc(&h.abortGenerate, h.lib, "abort_generate")
	
	// Capability checks
	purego.RegisterLibFunc(&h.hasAudioSupport, h.lib, "has_audio_support")
	purego.RegisterLibFunc(&h.hasVisionSupport, h.lib, "has_vision_support")
	
	// Performance metrics
	purego.RegisterLibFunc(&h.getLastEvalTime, h.lib, "get_last_eval_time")
	purego.RegisterLibFunc(&h.getLastProcessTime, h.lib, "get_last_process_time")
	purego.RegisterLibFunc(&h.getLastTokenCount, h.lib, "get_last_token_count")
	purego.RegisterLibFunc(&h.getLastInputCount, h.lib, "get_last_input_count")
	purego.RegisterLibFunc(&h.getLastSeed, h.lib, "get_last_seed")
	purego.RegisterLibFunc(&h.getTotalGens, h.lib, "get_total_gens")
	purego.RegisterLibFunc(&h.getTotalImgGens, h.lib, "get_total_img_gens")
	purego.RegisterLibFunc(&h.getTotalTTSGens, h.lib, "get_total_tts_gens")
	purego.RegisterLibFunc(&h.getTotalTranscribeGens, h.lib, "get_total_transcribe_gens")
	purego.RegisterLibFunc(&h.getLastDraftSuccess, h.lib, "get_last_draft_success")
	purego.RegisterLibFunc(&h.getLastDraftFailed, h.lib, "get_last_draft_failed")
	purego.RegisterLibFunc(&h.getLastStopReason, h.lib, "get_last_stop_reason")
	
	// Token operations
	purego.RegisterLibFunc(&h.tokenCount, h.lib, "token_count")
	purego.RegisterLibFunc(&h.getPendingOutput, h.lib, "get_pending_output")
	purego.RegisterLibFunc(&h.getChatTemplate, h.lib, "get_chat_template")
	purego.RegisterLibFunc(&h.detokenize, h.lib, "detokenize")
	purego.RegisterLibFunc(&h.lastLogprobs, h.lib, "last_logprobs")
	
	// State management (KV Cache)
	purego.RegisterLibFunc(&h.calcNewStateKv, h.lib, "calc_new_state_kv")
	purego.RegisterLibFunc(&h.calcNewStateTokencount, h.lib, "calc_new_state_tokencount")
	purego.RegisterLibFunc(&h.calcOldStateKv, h.lib, "calc_old_state_kv")
	purego.RegisterLibFunc(&h.calcOldStateTokencount, h.lib, "calc_old_state_tokencount")
	purego.RegisterLibFunc(&h.saveStateKv, h.lib, "save_state_kv")
	purego.RegisterLibFunc(&h.loadStateKv, h.lib, "load_state_kv")
	purego.RegisterLibFunc(&h.clearStateKv, h.lib, "clear_state_kv")
	
	// Stable Diffusion functions
	purego.RegisterLibFunc(&h.sdLoadModel, h.lib, "sd_load_model")
	purego.RegisterLibFunc(&h.sdGenerate, h.lib, "sd_generate")
	purego.RegisterLibFunc(&h.sdUpscale, h.lib, "sd_upscale")
	purego.RegisterLibFunc(&h.sdGetInfo, h.lib, "sd_get_info")
	
	// Whisper (speech-to-text) functions
	purego.RegisterLibFunc(&h.whisperLoadModel, h.lib, "whisper_load_model")
	purego.RegisterLibFunc(&h.whisperGenerate, h.lib, "whisper_generate")
	
	// TTS (text-to-speech) functions
	purego.RegisterLibFunc(&h.ttsLoadModel, h.lib, "tts_load_model")
	purego.RegisterLibFunc(&h.ttsGenerate, h.lib, "tts_generate")
	
	// Embeddings functions
	purego.RegisterLibFunc(&h.embeddingsLoadModel, h.lib, "embeddings_load_model")
	purego.RegisterLibFunc(&h.embeddingsGenerate, h.lib, "embeddings_generate")
}

// Function type definitions
type (
	loadModelFunc func(C.load_model_inputs) bool
	generateFunc func(C.generation_inputs) C.generation_outputs
	newTokenFunc func(C.int) *C.char
	getStreamCountFunc func() C.int
	hasFinishedFunc func() bool
	abortGenerateFunc func() bool
	
	hasAudioSupportFunc func() bool
	hasVisionSupportFunc func() bool
	
	getLastEvalTimeFunc func() C.float
	getLastProcessTimeFunc func() C.float
	getLastTokenCountFunc func() C.int
	getLastInputCountFunc func() C.int
	getLastSeedFunc func() C.int
	getTotalGensFunc func() C.int
	getTotalImgGensFunc func() C.int
	getTotalTTSGensFunc func() C.int
	getTotalTranscribeGensFunc func() C.int
	getLastDraftSuccessFunc func() C.int
	getLastDraftFailedFunc func() C.int
	getLastStopReasonFunc func() C.int
	
	tokenCountFunc func(*C.char, bool) C.token_count_outputs
	getPendingOutputFunc func() *C.char
	getChatTemplateFunc func() *C.char
	detokenizeFunc func(C.token_count_outputs) *C.char
	lastLogprobsFunc func() C.last_logprobs_outputs
	
	calcNewStateKvFunc func() C.size_t
	calcNewStateTokencountFunc func() C.size_t
	calcOldStateKvFunc func(C.int) C.size_t
	calcOldStateTokencountFunc func(C.int) C.size_t
	saveStateKvFunc func(C.int) C.size_t
	loadStateKvFunc func(C.int) bool
	clearStateKvFunc func() bool
	
	sdLoadModelFunc func(C.sd_load_model_inputs) bool
	sdGenerateFunc func(C.sd_generation_inputs) C.sd_generation_outputs
	sdUpscaleFunc func(C.sd_upscale_inputs) C.sd_generation_outputs
	sdGetInfoFunc func() C.sd_info_outputs
	
	whisperLoadModelFunc func(C.whisper_load_model_inputs) bool
	whisperGenerateFunc func(C.whisper_generation_inputs) C.whisper_generation_outputs
	
	ttsLoadModelFunc func(C.tts_load_model_inputs) bool
	ttsGenerateFunc func(C.tts_generation_inputs) C.tts_generation_outputs
	
	embeddingsLoadModelFunc func(C.embeddings_load_model_inputs) bool
	embeddingsGenerateFunc func(C.embeddings_generation_inputs) C.embeddings_generation_outputs
)

// Function variables
var (
	// These will be assigned by RegisterLibFunc calls
	loadModel          loadModelFunc
	generate           generateFunc
	newToken           newTokenFunc
	getStreamCount     getStreamCountFunc
	hasFinished        hasFinishedFunc
	abortGenerate      abortGenerateFunc
	
	hasAudioSupport    hasAudioSupportFunc
	hasVisionSupport   hasVisionSupportFunc
	
	getLastEvalTime        getLastEvalTimeFunc
	getLastProcessTime     getLastProcessTimeFunc
	getLastTokenCount      getLastTokenCountFunc
	getLastInputCount      getLastInputCountFunc
	getLastSeed            getLastSeedFunc
	getTotalGens           getTotalGensFunc
	getTotalImgGens        getTotalImgGensFunc
	getTotalTTSGens        getTotalTTSGensFunc
	getTotalTranscribeGens getTotalTranscribeGensFunc
	getLastDraftSuccess    getLastDraftSuccessFunc
	getLastDraftFailed     getLastDraftFailedFunc
	getLastStopReason      getLastStopReasonFunc
	
	tokenCount      tokenCountFunc
	getPendingOutput getPendingOutputFunc
	getChatTemplate  getChatTemplateFunc
	detokenize      detokenizeFunc
	lastLogprobs    lastLogprobsFunc
	
	calcNewStateKv           calcNewStateKvFunc
	calcNewStateTokencount   calcNewStateTokencountFunc
	calcOldStateKv           calcOldStateKvFunc
	calcOldStateTokencount   calcOldStateTokencountFunc
	saveStateKv              saveStateKvFunc
	loadStateKv              loadStateKvFunc
	clearStateKv             clearStateKvFunc
	
	sdLoadModel sdLoadModelFunc
	sdGenerate  sdGenerateFunc
	sdUpscale   sdUpscaleFunc
	sdGetInfo   sdGetInfoFunc
	
	whisperLoadModel whisperLoadModelFunc
	whisperGenerate  whisperGenerateFunc
	
	ttsLoadModel ttsLoadModelFunc
	ttsGenerate  ttsGenerateFunc
	
	embeddingsLoadModel embeddingsLoadModelFunc
	embeddingsGenerate  embeddingsGenerateFunc
)

// Wrapper methods for the library functions
func (h *LibraryHandle) LoadModel(inputs LoadModelInputs) bool {
	cInputs := convertToCLoadModelInputs(inputs)
	return h.loadModel(cInputs)
}

func (h *LibraryHandle) Generate(inputs GenerationInputs) GenerationOutputs {
	cInputs := convertToCGenerationInputs(inputs)
	cResult := h.generate(cInputs)
	return convertFromCGenerationOutputs(cResult)
}

func (h *LibraryHandle) NewToken(index int) string {
	cResult := h.newToken(C.int(index))
	return C.GoString(cResult)
}

func (h *LibraryHandle) GetStreamCount() int {
	return int(h.getStreamCount())
}

func (h *LibraryHandle) HasFinished() bool {
	return h.hasFinished()
}

func (h *LibraryHandle) AbortGenerate() bool {
	return h.abortGenerate()
}

func (h *LibraryHandle) HasAudioSupport() bool {
	return h.hasAudioSupport()
}

func (h *LibraryHandle) HasVisionSupport() bool {
	return h.hasVisionSupport()
}

func (h *LibraryHandle) GetLastEvalTime() float32 {
	return float32(h.getLastEvalTime())
}

func (h *LibraryHandle) GetLastProcessTime() float32 {
	return float32(h.getLastProcessTime())
}

func (h *LibraryHandle) GetLastTokenCount() int {
	return int(h.getLastTokenCount())
}

func (h *LibraryHandle) GetLastInputCount() int {
	return int(h.getLastInputCount())
}

func (h *LibraryHandle) GetLastSeed() int {
	return int(h.getLastSeed())
}

func (h *LibraryHandle) GetTotalGens() int {
	return int(h.getTotalGens())
}

func (h *LibraryHandle) GetTotalImgGens() int {
	return int(h.getTotalImgGens())
}

func (h *LibraryHandle) GetTotalTTSGens() int {
	return int(h.getTotalTTSGens())
}

func (h *LibraryHandle) GetTotalTranscribeGens() int {
	return int(h.getTotalTranscribeGens())
}

func (h *LibraryHandle) GetLastDraftSuccess() int {
	return int(h.getLastDraftSuccess())
}

func (h *LibraryHandle) GetLastDraftFailed() int {
	return int(h.getLastDraftFailed())
}

func (h *LibraryHandle) GetLastStopReason() StopReason {
	return StopReason(h.getLastStopReason())
}

func (h *LibraryHandle) TokenCount(text string, addSpecialTokens bool) TokenCountOutputs {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	
	cResult := h.tokenCount(cText, addSpecialTokens)
	result := TokenCountOutputs{
		Count: int(cResult.count),
	}
	
	if cResult.ids != nil && cResult.count > 0 {
		// Convert C array to Go slice
		goSlice := (*[1 << 28]C.int)(unsafe.Pointer(cResult.ids))[:cResult.count:cResult.count]
		result.IDs = make([]int, cResult.count)
		for i, id := range goSlice {
			result.IDs[i] = int(id)
		}
	}
	
	return result
}

func (h *LibraryHandle) GetPendingOutput() string {
	cResult := h.getPendingOutput()
	if cResult == nil {
		return ""
	}
	return C.GoString(cResult)
}

func (h *LibraryHandle) GetChatTemplate() string {
	cResult := h.getChatTemplate()
	if cResult == nil {
		return ""
	}
	return C.GoString(cResult)
}

func (h *LibraryHandle) Detokenize(tokens TokenCountOutputs) string {
	// Create C structure for function call
	cTokens := C.token_count_outputs{}
	cTokens.count = C.int(tokens.Count)
	
	if len(tokens.IDs) > 0 {
		// Allocate C array and copy Go slice
		cArray := C.malloc(C.size_t(len(tokens.IDs)) * C.sizeof_int)
		defer C.free(cArray)
		
		goSlice := (*[1 << 28]C.int)(cArray)[:len(tokens.IDs):len(tokens.IDs)]
		for i, id := range tokens.IDs {
			goSlice[i] = C.int(id)
		}
		cTokens.ids = (*C.int)(cArray)
	}
	
	cResult := h.detokenize(cTokens)
	if cResult == nil {
		return ""
	}
	return C.GoString(cResult)
}

func (h *LibraryHandle) LastLogprobs() LastLogprobsOutputs {
	cResult := h.lastLogprobs()
	result := LastLogprobsOutputs{
		Count: int(cResult.count),
	}
	
	if cResult.logprob_items != nil && cResult.count > 0 {
		// Convert C array of logprob_item to Go slice
		goSlice := (*[1 << 28]C.logprob_item)(unsafe.Pointer(cResult.logprob_items))[:cResult.count:cResult.count]
		result.LogprobItems = make([]LogprobItem, cResult.count)
		
		for i, item := range goSlice {
			result.LogprobItems[i] = convertFromCLogprobItem(item)
		}
	}
	
	return result
}

func (h *LibraryHandle) CalcNewStateKv() uint {
	return uint(h.calcNewStateKv())
}

func (h *LibraryHandle) CalcNewStateTokencount() uint {
	return uint(h.calcNewStateTokencount())
}

func (h *LibraryHandle) CalcOldStateKv(n int) uint {
	return uint(h.calcOldStateKv(C.int(n)))
}

func (h *LibraryHandle) CalcOldStateTokencount(n int) uint {
	return uint(h.calcOldStateTokencount(C.int(n)))
}

func (h *LibraryHandle) SaveStateKv(slot int) uint {
	return uint(h.saveStateKv(C.int(slot)))
}

func (h *LibraryHandle) LoadStateKv(slot int) bool {
	return h.loadStateKv(C.int(slot))
}

func (h *LibraryHandle) ClearStateKv() bool {
	return h.clearStateKv()
}

func (h *LibraryHandle) SDLoadModel(inputs SDLoadModelInputs) bool {
	cInputs := convertToCSDLoadModelInputs(inputs)
	return h.sdLoadModel(cInputs)
}

func (h *LibraryHandle) SDGenerate(inputs SDGenerationInputs) SDGenerationOutputs {
	cInputs := convertToCSDGenerationInputs(inputs)
	cResult := h.sdGenerate(cInputs)
	return convertFromCSDGenerationOutputs(cResult)
}

func (h *LibraryHandle) SDUpscale(inputs SDUpscaleInputs) SDGenerationOutputs {
	cInputs := convertToCSDUpscaleInputs(inputs)
	cResult := h.sdUpscale(cInputs)
	return convertFromCSDGenerationOutputs(cResult)
}

func (h *LibraryHandle) SDGetInfo() SDInfoOutputs {
	cResult := h.sdGetInfo()
	return convertFromCSDInfoOutputs(cResult)
}

func (h *LibraryHandle) WhisperLoadModel(inputs WhisperLoadModelInputs) bool {
	cInputs := convertToCWhisperLoadModelInputs(inputs)
	return h.whisperLoadModel(cInputs)
}

func (h *LibraryHandle) WhisperGenerate(inputs WhisperGenerationInputs) WhisperGenerationOutputs {
	cInputs := convertToCWhisperGenerationInputs(inputs)
	cResult := h.whisperGenerate(cInputs)
	return convertFromCWhisperGenerationOutputs(cResult)
}

func (h *LibraryHandle) TTSLoadModel(inputs TTSLoadModelInputs) bool {
	cInputs := convertToCTTSLoadModelInputs(inputs)
	return h.ttsLoadModel(cInputs)
}

func (h *LibraryHandle) TTSGenerate(inputs TTSGenerationInputs) TTSGenerationOutputs {
	cInputs := convertToCTTSGenerationInputs(inputs)
	cResult := h.ttsGenerate(cInputs)
	return convertFromCTTSGenerationOutputs(cResult)
}

func (h *LibraryHandle) EmbeddingsLoadModel(inputs EmbeddingsLoadModelInputs) bool {
	cInputs := convertToCEmbeddingsLoadModelInputs(inputs)
	return h.embeddingsLoadModel(cInputs)
}

func (h *LibraryHandle) EmbeddingsGenerate(inputs EmbeddingsGenerationInputs) EmbeddingsGenerationOutputs {
	cInputs := convertToCEmbeddingsGenerationInputs(inputs)
	cResult := h.embeddingsGenerate(cInputs)
	return convertFromCEmbeddingsGenerationOutputs(cResult)
}

// Helper conversion functions
func convertToCLoadModelInputs(inputs LoadModelInputs) C.load_model_inputs {
	cInputs := C.load_model_inputs{}
	
	cInputs.threads = C.int(inputs.Threads)
	cInputs.blasthreads = C.int(inputs.Blasthreads)
	cInputs.max_context_length = C.int(inputs.MaxContextLength)
	cInputs.low_vram = C.bool(inputs.LowVram)
	cInputs.use_mmq = C.bool(inputs.UseMMQ)
	cInputs.use_rowsplit = C.bool(inputs.UseRowsplit)
	
	if inputs.ExecutablePath != "" {
		cInputs.executable_path = C.CString(inputs.ExecutablePath)
	}
	if inputs.ModelFilename != "" {
		cInputs.model_filename = C.CString(inputs.ModelFilename)
	}
	if inputs.LoraFilename != "" {
		cInputs.lora_filename = C.CString(inputs.LoraFilename)
	}
	if inputs.DraftmodelFilename != "" {
		cInputs.draftmodel_filename = C.CString(inputs.DraftmodelFilename)
	}
	
	cInputs.draft_amount = C.int(inputs.DraftAmount)
	cInputs.draft_gpulayers = C.int(inputs.DraftGpulayers)
	
	// Copy tensor split arrays
	for i := 0; i < tensorSplitMax; i++ {
		cInputs.draft_gpusplit[i] = C.float(inputs.DraftGpusplit[i])
		cInputs.tensor_split[i] = C.float(inputs.TensorSplit[i])
	}
	
	if inputs.MmprojFilename != "" {
		cInputs.mmproj_filename = C.CString(inputs.MmprojFilename)
	}
	
	cInputs.mmproj_cpu = C.bool(inputs.MmprojCPU)
	cInputs.visionmaxres = C.int(inputs.Visionmaxres)
	cInputs.use_mmap = C.bool(inputs.UseMmap)
	cInputs.use_mlock = C.bool(inputs.UseMlock)
	cInputs.use_smartcontext = C.bool(inputs.UseSmartcontext)
	cInputs.use_contextshift = C.bool(inputs.UseContextshift)
	cInputs.use_fastforward = C.bool(inputs.UseFastforward)
	cInputs.kcpp_main_gpu = C.int(inputs.KcppMainGPU)
	
	if inputs.VulkanInfo != "" {
		cInputs.vulkan_info = C.CString(inputs.VulkanInfo)
	}
	
	cInputs.batchsize = C.int(inputs.Batchsize)
	cInputs.autofit = C.bool(inputs.Autofit)
	cInputs.autofit_tax_mb = C.int(inputs.AutofitTaxMB)
	cInputs.gpulayers = C.int(inputs.Gpulayers)
	cInputs.rope_freq_scale = C.float(inputs.RopeFreqScale)
	cInputs.rope_freq_base = C.float(inputs.RopeFreqBase)
	cInputs.overridenativecontext = C.int(inputs.Overridenativecontext)
	cInputs.moe_experts = C.int(inputs.MoeExperts)
	cInputs.moecpu = C.int(inputs.Moecpu)
	cInputs.no_bos_token = C.bool(inputs.NoBosToken)
	cInputs.load_guidance = C.bool(inputs.LoadGuidance)
	
	// Override KV strings
	for i := 0; i < overridekvMax; i++ {
		if inputs.OverrideKV[i] != "" {
			cInputs.override_kv[i] = C.CString(inputs.OverrideKV[i])
		}
	}
	
	if inputs.OverrideTensors != "" {
		cInputs.override_tensors = C.CString(inputs.OverrideTensors)
	}
	
	cInputs.flash_attention = C.bool(inputs.FlashAttention)
	cInputs.quant_k = C.int(inputs.QuantK)
	cInputs.quant_v = C.int(inputs.QuantV)
	cInputs.check_slowness = C.bool(inputs.CheckSlowness)
	cInputs.highpriority = C.bool(inputs.Highpriority)
	cInputs.swa_support = C.bool(inputs.SwaSupport)
	cInputs.smartcache = C.bool(inputs.Smartcache)
	cInputs.smartcacheslots = C.int(inputs.Smartcacheslots)
	cInputs.pipelineparallel = C.bool(inputs.Pipelineparallel)
	cInputs.lora_multiplier = C.float(inputs.LoraMultiplier)
	
	if inputs.DevicesOverride != "" {
		cInputs.devices_override = C.CString(inputs.DevicesOverride)
	}
	
	cInputs.quiet = C.bool(inputs.Quiet)
	cInputs.debugmode = C.int(inputs.Debugmode)
	
	return cInputs
}

func convertToCGenerationInputs(inputs GenerationInputs) C.generation_inputs {
	cInputs := C.generation_inputs{}
	
	cInputs.seed = C.int(inputs.Seed)
	
	if inputs.Prompt != "" {
		cInputs.prompt = C.CString(inputs.Prompt)
	}
	if inputs.Memory != "" {
		cInputs.memory = C.CString(inputs.Memory)
	}
	if inputs.NegativePrompt != "" {
		cInputs.negative_prompt = C.CString(inputs.NegativePrompt)
	}
	
	cInputs.guidance_scale = C.float(inputs.GuidanceScale)
	
	// Copy image and audio arrays
	for i := 0; i < imagesMax; i++ {
		if inputs.Images[i] != "" {
			cInputs.images[i] = C.CString(inputs.Images[i])
		}
	}
	for i := 0; i < audioMax; i++ {
		if inputs.Audio[i] != "" {
			cInputs.audio[i] = C.CString(inputs.Audio[i])
		}
	}
	
	cInputs.max_context_length = C.int(inputs.MaxContextLength)
	cInputs.max_length = C.int(inputs.MaxLength)
	cInputs.temperature = C.float(inputs.Temperature)
	cInputs.top_k = C.int(inputs.TopK)
	cInputs.top_a = C.float(inputs.TopA)
	cInputs.top_p = C.float(inputs.TopP)
	cInputs.min_p = C.float(inputs.MinP)
	cInputs.typical_p = C.float(inputs.TypicalP)
	cInputs.tfs = C.float(inputs.TFS)
	cInputs.nsigma = C.float(inputs.NSigma)
	cInputs.rep_pen = C.float(inputs.RepPen)
	cInputs.rep_pen_range = C.int(inputs.RepPenRange)
	cInputs.rep_pen_slope = C.float(inputs.RepPenSlope)
	cInputs.presence_penalty = C.float(inputs.PresencePenalty)
	cInputs.mirostat = C.int(inputs.Mirostat)
	cInputs.mirostat_tau = C.float(inputs.MirostatTau)
	cInputs.mirostat_eta = C.float(inputs.MirostatEta)
	cInputs.xtc_threshold = C.float(inputs.XTCTreshold)
	cInputs.xtc_probability = C.float(inputs.XTCProbability)
	
	// Copy sampler order (now using enum values)
	for i := 0; i < int(KCPP_SAMPLER_MAX); i++ {
		cInputs.sampler_order[i] = C.enum_samplers(inputs.SamplerOrder[i])
	}
	
	cInputs.sampler_len = C.int(inputs.SamplerLen)
	cInputs.allow_eos_token = C.bool(inputs.AllowEOSToken)
	cInputs.bypass_eos_token = C.bool(inputs.BypassEOSToken)
	cInputs.tool_call_fix = C.bool(inputs.ToolCallFix)
	cInputs.render_special = C.bool(inputs.RenderSpecial)
	cInputs.stream_sse = C.bool(inputs.StreamSSE)
	
	if inputs.Grammar != "" {
		cInputs.grammar = C.CString(inputs.Grammar)
	}
	
	cInputs.grammar_retain_state = C.bool(inputs.GrammarRetainState)
	cInputs.dynatemp_range = C.float(inputs.DynatempRange)
	cInputs.dynatemp_exponent = C.float(inputs.DynatempExponent)
	cInputs.smoothing_factor = C.float(inputs.SmoothingFactor)
	cInputs.smoothing_curve = C.float(inputs.SmoothingCurve)
	cInputs.adaptive_target = C.float(inputs.AdaptiveTarget)
	cInputs.adaptive_decay = C.float(inputs.AdaptiveDecay)
	cInputs.dry_multiplier = C.float(inputs.DRYMultiplier)
	cInputs.dry_base = C.float(inputs.DRYBase)
	cInputs.dry_allowed_length = C.int(inputs.DRYAllowedLength)
	cInputs.dry_penalty_last_n = C.int(inputs.DRYPenaltyLastN)
	cInputs.dry_sequence_breakers_len = C.int(inputs.DRYSequenceBreakersLen)
	
	// Handle dynamic arrays (need to allocate C memory)
	if len(inputs.DRYSequenceBreakers) > 0 {
		cArray := C.malloc(C.size_t(len(inputs.DRYSequenceBreakers)) * C.sizeof_char_p)
		defer C.free(cArray)
		
		goSlice := (*[1 << 28]*C.char)(cArray)[:len(inputs.DRYSequenceBreakers):len(inputs.DRYSequenceBreakers)]
		for i, str := range inputs.DRYSequenceBreakers {
			goSlice[i] = C.CString(str)
		}
		cInputs.dry_sequence_breakers = (**C.char)(cArray)
	}
	
	cInputs.stop_sequence_len = C.int(inputs.StopSequenceLen)
	if len(inputs.StopSequence) > 0 {
		cArray := C.malloc(C.size_t(len(inputs.StopSequence)) * C.sizeof_char_p)
		defer C.free(cArray)
		
		goSlice := (*[1 << 28]*C.char)(cArray)[:len(inputs.StopSequence):len(inputs.StopSequence)]
		for i, str := range inputs.StopSequence {
			goSlice[i] = C.CString(str)
		}
		cInputs.stop_sequence = (**C.char)(cArray)
	}
	
	cInputs.logit_biases_len = C.int(inputs.LogitBiasesLen)
	if len(inputs.LogitBiases) > 0 {
		cArray := C.malloc(C.size_t(len(inputs.LogitBiases)) * C.sizeof_struct_logit_bias)
		defer C.free(cArray)
		
		goSlice := (*[1 << 28]C.logit_bias)(cArray)[:len(inputs.LogitBiases):len(inputs.LogitBiases)]
		for i, bias := range inputs.LogitBiases {
			goSlice[i].token_id = C.int32_t(bias.TokenID)
			goSlice[i].bias = C.float(bias.Bias)
		}
		cInputs.logit_biases = (*C.logit_bias)(cArray)
	}
	
	cInputs.banned_tokens_len = C.int(inputs.BannedTokensLen)
	if len(inputs.BannedTokens) > 0 {
		cArray := C.malloc(C.size_t(len(inputs.BannedTokens)) * C.sizeof_char_p)
		defer C.free(cArray)
		
		goSlice := (*[1 << 28]*C.char)(cArray)[:len(inputs.BannedTokens):len(inputs.BannedTokens)]
		for i, str := range inputs.BannedTokens {
			goSlice[i] = C.CString(str)
		}
		cInputs.banned_tokens = (**C.char)(cArray)
	}
	
	return cInputs
}

func convertFromCGenerationOutputs(cResult C.generation_outputs) GenerationOutputs {
	result := GenerationOutputs{
		Status:           int(cResult.status),
		Stopreason:       StopReason(cResult.stopreason),
		PromptTokens:     int(cResult.prompt_tokens),
		CompletionTokens: int(cResult.completion_tokens),
	}
	
	if cResult.text != nil {
		result.Text = C.GoString(cResult.text)
	}
	
	return result
}

func convertFromCSDGenerationOutputs(cResult C.sd_generation_outputs) SDGenerationOutputs {
	result := SDGenerationOutputs{
		Status:   int(cResult.status),
		Animated: int(cResult.animated),
	}
	
	if cResult.data != nil {
		result.Data = C.GoString(cResult.data)
	}
	if cResult.data_extra != nil {
		result.DataExtra = C.GoString(cResult.data_extra)
	}
	
	return result
}

func convertFromCSDInfoOutputs(cResult C.sd_info_outputs) SDInfoOutputs {
	result := SDInfoOutputs{
		Status: int(cResult.status),
	}
	
	if cResult.data != nil {
		result.Data = C.GoString(cResult.data)
	}
	
	return result
}

func convertFromCWhisperGenerationOutputs(cResult C.whisper_generation_outputs) WhisperGenerationOutputs {
	result := WhisperGenerationOutputs{
		Status: int(cResult.status),
	}
	
	if cResult.text != nil {
		result.Text = C.GoString(cResult.text)
	}
	
	return result
}

func convertFromCTTSGenerationOutputs(cResult C.tts_generation_outputs) TTSGenerationOutputs {
	result := TTSGenerationOutputs{
		Status: int(cResult.status),
	}
	
	if cResult.data != nil {
		result.Data = C.GoString(cResult.data)
	}
	
	return result
}

func convertFromCEmbeddingsGenerationOutputs(cResult C.embeddings_generation_outputs) EmbeddingsGenerationOutputs {
	result := EmbeddingsGenerationOutputs{
		Status: int(cResult.status),
		Count:  int(cResult.count),
	}
	
	if cResult.data != nil {
		result.Data = C.GoString(cResult.data)
	}
	
	return result
}

func convertFromCLogprobItem(cItem C.logprob_item) LogprobItem {
	item := LogprobItem{
		OptionCount:       int(cItem.option_count),
		SelectedToken:     C.GoString(cItem.selected_token),
		SelectedLogprob:   float32(cItem.selected_logprob),
		SelectedTokenID:   int32(cItem.selected_token_id),
	}
	
	// Copy tokens array
	for i := 0; i < logprobsMax; i++ {
		item.Tokens[i] = C.GoString(cItem.tokens[i])
		item.TokenIDs[i] = int32(cItem.token_ids[i])
	}
	
	// Copy logprobs array if it exists
	if cItem.logprobs != nil {
		// Since we don't know the exact size, we'll assume it matches option_count
		count := item.OptionCount
		if count > logprobsMax {
			count = logprobsMax
		}
		goSlice := (*[1 << 28]C.float)(unsafe.Pointer(cItem.logprobs))[:count:count]
		item.Logprobs = make([]float32, count)
		for i, val := range goSlice {
			item.Logprobs[i] = float32(val)
		}
	}
	
	return item
}

// Additional conversion functions for other structures
func convertToCSDLoadModelInputs(inputs SDLoadModelInputs) C.sd_load_model_inputs {
	cInputs := C.sd_load_model_inputs{}
	
	if inputs.ModelFilename != "" {
		cInputs.model_filename = C.CString(inputs.ModelFilename)
	}
	if inputs.ExecutablePath != "" {
		cInputs.executable_path = C.CString(inputs.ExecutablePath)
	}
	cInputs.kcpp_main_gpu = C.int(inputs.KcppMainGPU)
	if inputs.VulkanInfo != "" {
		cInputs.vulkan_info = C.CString(inputs.VulkanInfo)
	}
	cInputs.threads = C.int(inputs.Threads)
	cInputs.quant = C.int(inputs.Quant)
	cInputs.flash_attention = C.bool(inputs.FlashAttention)
	cInputs.offload_cpu = C.bool(inputs.OffloadCPU)
	cInputs.vae_cpu = C.bool(inputs.VAECPU)
	cInputs.clip_cpu = C.bool(inputs.CLIPCPU)
	cInputs.diffusion_conv_direct = C.bool(inputs.DiffusionConvDirect)
	cInputs.vae_conv_direct = C.bool(inputs.VAEConvDirect)
	cInputs.taesd = C.bool(inputs.TAESD)
	cInputs.tiled_vae_threshold = C.int(inputs.TiledVAEThreshold)
	
	if inputs.T5XXLFilename != "" {
		cInputs.t5xxl_filename = C.CString(inputs.T5XXLFilename)
	}
	if inputs.CLIP1Filename != "" {
		cInputs.clip1_filename = C.CString(inputs.CLIP1Filename)
	}
	if inputs.CLIP2Filename != "" {
		cInputs.clip2_filename = C.CString(inputs.CLIP2Filename)
	}
	if inputs.VAEFilename != "" {
		cInputs.vae_filename = C.CString(inputs.VAEFilename)
	}
	if inputs.LoraFilename != "" {
		cInputs.lora_filename = C.CString(inputs.LoraFilename)
	}
	cInputs.lora_multiplier = C.float(inputs.LoraMultiplier)
	cInputs.lora_apply_mode = C.int(inputs.LoraApplyMode)
	
	if inputs.PhotomakerFilename != "" {
		cInputs.photomaker_filename = C.CString(inputs.PhotomakerFilename)
	}
	if inputs.UpscalerFilename != "" {
		cInputs.upscaler_filename = C.CString(inputs.UpscalerFilename)
	}
	cInputs.img_hard_limit = C.int(inputs.ImgHardLimit)
	cInputs.img_soft_limit = C.int(inputs.ImgSoftLimit)
	
	if inputs.DevicesOverride != "" {
		cInputs.devices_override = C.CString(inputs.DevicesOverride)
	}
	cInputs.quiet = C.bool(inputs.Quiet)
	cInputs.debugmode = C.int(inputs.Debugmode)
	
	return cInputs
}

func convertToCSDGenerationInputs(inputs SDGenerationInputs) C.sd_generation_inputs {
	cInputs := C.sd_generation_inputs{}
	
	if inputs.Prompt != "" {
		cInputs.prompt = C.CString(inputs.Prompt)
	}
	if inputs.NegativePrompt != "" {
		cInputs.negative_prompt = C.CString(inputs.NegativePrompt)
	}
	if inputs.InitImages != "" {
		cInputs.init_images = C.CString(inputs.InitImages)
	} else {
		cInputs.init_images = C.CString("")
	}
	if inputs.Mask != "" {
		cInputs.mask = C.CString(inputs.Mask)
	} else {
		cInputs.mask = C.CString("")
	}
	
	cInputs.extra_images_len = C.int(inputs.ExtraImagesLen)
	if len(inputs.ExtraImages) > 0 {
		cArray := C.malloc(C.size_t(len(inputs.ExtraImages)) * C.sizeof_char_p)
		defer C.free(cArray)
		
		goSlice := (*[1 << 28]*C.char)(cArray)[:len(inputs.ExtraImages):len(inputs.ExtraImages)]
		for i, str := range inputs.ExtraImages {
			goSlice[i] = C.CString(str)
		}
		cInputs.extra_images = (**C.char)(cArray)
	}
	
	cInputs.flip_mask = C.bool(inputs.FlipMask)
	cInputs.denoising_strength = C.float(inputs.DenoisingStrength)
	cInputs.cfg_scale = C.float(inputs.CFGScale)
	cInputs.distilled_guidance = C.float(inputs.DistilledGuidance)
	cInputs.shifted_timestep = C.int(inputs.ShiftedTimestep)
	cInputs.sample_steps = C.int(inputs.SampleSteps)
	cInputs.width = C.int(inputs.Width)
	cInputs.height = C.int(inputs.Height)
	cInputs.seed = C.int(inputs.Seed)
	
	if inputs.SampleMethod != "" {
		cInputs.sample_method = C.CString(inputs.SampleMethod)
	}
	if inputs.Scheduler != "" {
		cInputs.scheduler = C.CString(inputs.Scheduler)
	}
	cInputs.clip_skip = C.int(inputs.ClipSkip)
	cInputs.vid_req_frames = C.int(inputs.VidReqFrames)
	cInputs.video_output_type = C.int(inputs.VideoOutputType)
	cInputs.remove_limits = C.bool(inputs.RemoveLimits)
	cInputs.circular_x = C.bool(inputs.CircularX)
	cInputs.circular_y = C.bool(inputs.CircularY)
	cInputs.upscale = C.bool(inputs.Upscale)
	
	return cInputs
}

func convertToCSDUpscaleInputs(inputs SDUpscaleInputs) C.sd_upscale_inputs {
	cInputs := C.sd_upscale_inputs{}
	
	if inputs.InitImages != "" {
		cInputs.init_images = C.CString(inputs.InitImages)
	} else {
		cInputs.init_images = C.CString("")
	}
	cInputs.upscaling_resize = C.int(inputs.UpscalingResize)
	
	return cInputs
}

func convertToCWhisperLoadModelInputs(inputs WhisperLoadModelInputs) C.whisper_load_model_inputs {
	cInputs := C.whisper_load_model_inputs{}
	
	if inputs.ModelFilename != "" {
		cInputs.model_filename = C.CString(inputs.ModelFilename)
	}
	if inputs.ExecutablePath != "" {
		cInputs.executable_path = C.CString(inputs.ExecutablePath)
	}
	cInputs.kcpp_main_gpu = C.int(inputs.KcppMainGPU)
	if inputs.VulkanInfo != "" {
		cInputs.vulkan_info = C.CString(inputs.VulkanInfo)
	}
	if inputs.DevicesOverride != "" {
		cInputs.devices_override = C.CString(inputs.DevicesOverride)
	}
	cInputs.quiet = C.bool(inputs.Quiet)
	cInputs.debugmode = C.int(inputs.Debugmode)
	
	return cInputs
}

func convertToCWhisperGenerationInputs(inputs WhisperGenerationInputs) C.whisper_generation_inputs {
	cInputs := C.whisper_generation_inputs{}
	
	if inputs.Prompt != "" {
		cInputs.prompt = C.CString(inputs.Prompt)
	}
	if inputs.AudioData != "" {
		cInputs.audio_data = C.CString(inputs.AudioData)
	}
	cInputs.suppress_non_speech = C.bool(inputs.SuppressNonSpeech)
	if inputs.Langcode != "" {
		cInputs.langcode = C.CString(inputs.Langcode)
	}
	
	return cInputs
}

func convertToCTTSLoadModelInputs(inputs TTSLoadModelInputs) C.tts_load_model_inputs {
	cInputs := C.tts_load_model_inputs{}
	
	cInputs.threads = C.int(inputs.Threads)
	if inputs.TTCModelFilename != "" {
		cInputs.ttc_model_filename = C.CString(inputs.TTCModelFilename)
	}
	if inputs.CTSModelFilename != "" {
		cInputs.cts_model_filename = C.CString(inputs.CTSModelFilename)
	}
	if inputs.ExecutablePath != "" {
		cInputs.executable_path = C.CString(inputs.ExecutablePath)
	}
	cInputs.kcpp_main_gpu = C.int(inputs.KcppMainGPU)
	if inputs.VulkanInfo != "" {
		cInputs.vulkan_info = C.CString(inputs.VulkanInfo)
	}
	cInputs.gpulayers = C.int(inputs.Gpulayers)
	cInputs.flash_attention = C.bool(inputs.FlashAttention)
	cInputs.ttsmaxlen = C.int(inputs.TTSMaxLen)
	
	if inputs.DevicesOverride != "" {
		cInputs.devices_override = C.CString(inputs.DevicesOverride)
	}
	cInputs.quiet = C.bool(inputs.Quiet)
	cInputs.debugmode = C.int(inputs.Debugmode)
	
	return cInputs
}

func convertToCTTSGenerationInputs(inputs TTSGenerationInputs) C.tts_generation_inputs {
	cInputs := C.tts_generation_inputs{}
	
	if inputs.Prompt != "" {
		cInputs.prompt = C.CString(inputs.Prompt)
	}
	cInputs.speaker_seed = C.int(inputs.SpeakerSeed)
	cInputs.audio_seed = C.int(inputs.AudioSeed)
	
	if inputs.CustomSpeakerVoice != "" {
		cInputs.custom_speaker_voice = C.CString(inputs.CustomSpeakerVoice)
	} else {
		cInputs.custom_speaker_voice = C.CString("")
	}
	if inputs.CustomSpeakerText != "" {
		cInputs.custom_speaker_text = C.CString(inputs.CustomSpeakerText)
	} else {
		cInputs.custom_speaker_text = C.CString("")
	}
	if inputs.CustomSpeakerData != "" {
		cInputs.custom_speaker_data = C.CString(inputs.CustomSpeakerData)
	} else {
		cInputs.custom_speaker_data = C.CString("")
	}
	
	return cInputs
}

func convertToCEmbeddingsLoadModelInputs(inputs EmbeddingsLoadModelInputs) C.embeddings_load_model_inputs {
	cInputs := C.embeddings_load_model_inputs{}
	
	cInputs.threads = C.int(inputs.Threads)
	if inputs.ModelFilename != "" {
		cInputs.model_filename = C.CString(inputs.ModelFilename)
	}
	if inputs.ExecutablePath != "" {
		cInputs.executable_path = C.CString(inputs.ExecutablePath)
	}
	cInputs.kcpp_main_gpu = C.int(inputs.KcppMainGPU)
	if inputs.VulkanInfo != "" {
		cInputs.vulkan_info = C.CString(inputs.VulkanInfo)
	}
	cInputs.gpulayers = C.int(inputs.Gpulayers)
	cInputs.flash_attention = C.bool(inputs.FlashAttention)
	cInputs.use_mmap = C.bool(inputs.UseMmap)
	cInputs.embeddingsmaxctx = C.int(inputs.Embeddingsmaxctx)
	
	if inputs.DevicesOverride != "" {
		cInputs.devices_override = C.CString(inputs.DevicesOverride)
	}
	cInputs.quiet = C.bool(inputs.Quiet)
	cInputs.debugmode = C.int(inputs.Debugmode)
	
	return cInputs
}

func convertToCEmbeddingsGenerationInputs(inputs EmbeddingsGenerationInputs) C.embeddings_generation_inputs {
	cInputs := C.embeddings_generation_inputs{}
	
	if inputs.Prompt != "" {
		cInputs.prompt = C.CString(inputs.Prompt)
	}
	cInputs.truncate = C.bool(inputs.Truncate)
	
	return cInputs
}