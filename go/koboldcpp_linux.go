package koboldcpp

import "github.com/ebitengine/purego"

// Function pointers - will be registered with purego
// On Linux/Windows, purego does not support struct returns directly.
// We must use hidden pointer arguments for output structs.
// AND input structs must be passed BY VALUE to match C++ ABI.
var (
	loadModel              func(LoadModelInputs) bool
	generate               func(*GenerationOutputs, GenerationInputs) // *Out, In value
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
	tokenCount             func(*TokenCountOutputs, *byte, bool)
	getPendingOutput       func() *byte
	getChatTemplate        func() *byte
	lastLogprobs           func(*LastLogprobsOutputs)
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
	sdLoadModel func(SDLoadModelInputs) bool
	sdGenerate  func(*SDGenerationOutputs, SDGenerationInputs)
	sdUpscale   func(*SDGenerationOutputs, SDUpscaleInputs)
	sdGetInfo   func(*SDInfoOutputs) // No input

	// Whisper functions
	whisperLoadModel func(WhisperLoadModelInputs) bool
	whisperGenerate  func(*WhisperGenerationOutputs, WhisperGenerationInputs)

	// TTS functions
	ttsLoadModel func(TTSLoadModelInputs) bool
	ttsGenerate  func(*TTSGenerationOutputs, TTSGenerationInputs)

	// Embeddings functions
	embeddingsLoadModel func(EmbeddingsLoadModelInputs) bool
	embeddingsGenerate  func(*EmbeddingsGenerationOutputs, EmbeddingsGenerationInputs)
)

// RegisterFunctions registers all C functions with purego
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

// Public API functions - Wrappers around hidden pointer implementation

// LoadModel loads a model - matches Python load_model()
func LoadModel(inputs *LoadModelInputs) bool {
	return loadModel(*inputs)
}

// Generate generates text - matches Python generate()
func Generate(inputs *GenerationInputs) GenerationOutputs {
	var output GenerationOutputs
	generate(&output, *inputs)
	return output
}

// SDLoadModel loads SD model - matches Python sd_load_model()
func SDLoadModel(inputs *SDLoadModelInputs) bool {
	return sdLoadModel(*inputs)
}

// SDGenerate generates image - matches Python sd_generate()
func SDGenerate(inputs *SDGenerationInputs) SDGenerationOutputs {
	var output SDGenerationOutputs
	sdGenerate(&output, *inputs)
	return output
}

// SDUpscale upscales image - matches Python sd_upscale()
func SDUpscale(inputs *SDUpscaleInputs) SDGenerationOutputs {
	var output SDGenerationOutputs
	sdUpscale(&output, *inputs)
	return output
}

// SDGetInfo gets SD info - matches Python sd_get_info()
func SDGetInfo() SDInfoOutputs {
	var output SDInfoOutputs
	sdGetInfo(&output)
	return output
}

// WhisperLoadModel loads Whisper model - matches Python whisper_load_model()
func WhisperLoadModel(inputs *WhisperLoadModelInputs) bool {
	return whisperLoadModel(*inputs)
}

// WhisperGenerate transcribes audio - matches Python whisper_generate()
func WhisperGenerate(inputs *WhisperGenerationInputs) WhisperGenerationOutputs {
	var output WhisperGenerationOutputs
	whisperGenerate(&output, *inputs)
	return output
}

// TTSLoadModel loads TTS model - matches Python tts_load_model()
func TTSLoadModel(inputs *TTSLoadModelInputs) bool {
	return ttsLoadModel(*inputs)
}

// TTSGenerate generates speech - matches Python tts_generate()
func TTSGenerate(inputs *TTSGenerationInputs) TTSGenerationOutputs {
	var output TTSGenerationOutputs
	ttsGenerate(&output, *inputs)
	return output
}

// EmbeddingsLoadModel loads embeddings model - matches Python embeddings_load_model()
func EmbeddingsLoadModel(inputs *EmbeddingsLoadModelInputs) bool {
	return embeddingsLoadModel(*inputs)
}

// EmbeddingsGenerate generates embeddings - matches Python embeddings_generate()
func EmbeddingsGenerate(inputs *EmbeddingsGenerationInputs) EmbeddingsGenerationOutputs {
	var output EmbeddingsGenerationOutputs
	embeddingsGenerate(&output, *inputs)
	return output
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
	var output TokenCountOutputs
	tokenCount(&output, CString(text), addSpecial)
	return output
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

// GetLastDraftSuccess gets last draft success count
func GetLastDraftSuccess() int32 {
	return getLastDraftSuccess()
}

// GetLastDraftFailed gets last draft failed count
func GetLastDraftFailed() int32 {
	return getLastDraftFailed()
}

// GetTotalImgGens gets total image generations
func GetTotalImgGens() int32 {
	return getTotalImgGens()
}

// GetTotalTTSGens gets total TTS generations
func GetTotalTTSGens() int32 {
	return getTotalTTSGens()
}

// GetTotalTranscribeGens gets total transcribe generations
func GetTotalTranscribeGens() int32 {
	return getTotalTranscribeGens()
}

// GetTotalGens gets total generations
func GetTotalGens() int32 {
	return getTotalGens()
}

// GetLastStopReason gets last stop reason
func GetLastStopReason() int32 {
	return getLastStopReason()
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

// LastLogprobs gets last logprobs
func LastLogprobs() LastLogprobsOutputs {
	var output LastLogprobsOutputs
	lastLogprobs(&output)
	return output
}

// Detokenize detokenizes token IDs
func Detokenize(inputs TokenCountOutputs) string {
	return GoString(detokenize(inputs))
}

// CalcNewStateKV calculates new state KV
func CalcNewStateKV() uint64 {
	return calcNewStateKV()
}

// CalcNewStateTokenCount calculates new state token count
func CalcNewStateTokenCount() uint64 {
	return calcNewStateTokenCount()
}

// CalcOldStateKV calculates old state KV
func CalcOldStateKV(slot int32) uint64 {
	return calcOldStateKV(slot)
}

// CalcOldStateTokenCount calculates old state token count
func CalcOldStateTokenCount(slot int32) uint64 {
	return calcOldStateTokenCount(slot)
}

// HasAudioSupport checks if audio support is available
func HasAudioSupport() bool {
	return hasAudioSupport()
}

// HasVisionSupport checks if vision support is available
func HasVisionSupport() bool {
	return hasVisionSupport()
}

// GetLastInputCount gets last input count
func GetLastInputCount() int32 {
	return getLastInputCount()
}
