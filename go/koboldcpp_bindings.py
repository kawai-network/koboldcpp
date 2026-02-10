#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
KoboldCpp C++ Bindings
Extracted from koboldcpp.py - Contains all ctypes structures and function bindings
for interfacing with the KoboldCpp C++ library

This module provides a clean interface to the KoboldCpp C++ library through ctypes.
It includes all necessary structures and function bindings for:
- Text generation (GGUF models)
- Image generation (Stable Diffusion)
- Speech-to-text (Whisper)
- Text-to-speech (TTS)
- Text embeddings
- State management (KV cache)
- Performance monitoring

Note: This module contains ONLY the C++ library bindings. Application-level features
from koboldcpp.py (such as MCP protocol support, web server, etc.) are not included.
"""

import ctypes
import os
import time

# ============================================================================
# CONSTANTS
# ============================================================================

sampler_order_max = 7
tensor_split_max = 16
images_max = 8
audio_max = 4
bias_min_value = -100.0
bias_max_value = 100.0
logprobs_max = 10
default_draft_amount = 8
default_ttsmaxlen = 4096
default_visionmaxres = 1024
net_save_slots = 12
savestate_limit_default = 5
default_vae_tile_threshold = 768
default_native_ctx = 16384
overridekv_max = 4

# Abuse prevention limits
stop_token_max = 256
ban_token_max = 768
logit_bias_max = 512
dry_seq_break_max = 128
extra_images_max = 4  # for kontext/qwen img

# ============================================================================
# CTYPES STRUCTURES - Data Models
# ============================================================================

class logit_bias(ctypes.Structure):
    """Structure for logit bias configuration"""
    _fields_ = [("token_id", ctypes.c_int32),
                ("bias", ctypes.c_float)]

class token_count_outputs(ctypes.Structure):
    """Structure for token counting output"""
    _fields_ = [("count", ctypes.c_int),
                ("ids", ctypes.POINTER(ctypes.c_int))]

class logprob_item(ctypes.Structure):
    """Structure for individual logprob item (returns top 5 logprobs per token)"""
    _fields_ = [("option_count", ctypes.c_int),
                ("selected_token", ctypes.c_char_p),
                ("selected_logprob", ctypes.c_float),
                ("selected_token_id", ctypes.c_int32),
                ("tokens", ctypes.c_char_p * logprobs_max),
                ("token_ids", ctypes.c_int32 * logprobs_max),
                ("logprobs", ctypes.POINTER(ctypes.c_float))]

class last_logprobs_outputs(ctypes.Structure):
    """Structure for last logprobs output"""
    _fields_ = [("count", ctypes.c_int),
                ("logprob_items", ctypes.POINTER(logprob_item))]

# ============================================================================
# MODEL LOADING STRUCTURES
# ============================================================================

class load_model_inputs(ctypes.Structure):
    """Structure for text/GGUF model loading parameters"""
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("low_vram", ctypes.c_bool),
                ("use_mmq", ctypes.c_bool),
                ("use_rowsplit", ctypes.c_bool),
                ("executable_path", ctypes.c_char_p),
                ("model_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("draftmodel_filename", ctypes.c_char_p),
                ("draft_amount", ctypes.c_int),
                ("draft_gpulayers", ctypes.c_int),
                ("draft_gpusplit", ctypes.c_float * tensor_split_max),
                ("mmproj_filename", ctypes.c_char_p),
                ("mmproj_cpu", ctypes.c_bool),
                ("visionmaxres", ctypes.c_int),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("use_smartcontext", ctypes.c_bool),
                ("use_contextshift", ctypes.c_bool),
                ("use_fastforward", ctypes.c_bool),
                ("kcpp_main_gpu", ctypes.c_int),
                ("vulkan_info", ctypes.c_char_p),
                ("batchsize", ctypes.c_int),
                ("autofit", ctypes.c_bool),
                ("autofit_tax_mb", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("rope_freq_scale", ctypes.c_float),
                ("rope_freq_base", ctypes.c_float),
                ("overridenativecontext", ctypes.c_int),
                ("moe_experts", ctypes.c_int),
                ("moecpu", ctypes.c_int),
                ("no_bos_token", ctypes.c_bool),
                ("load_guidance", ctypes.c_bool),
                ("override_kv", ctypes.c_char_p * overridekv_max),
                ("override_tensors", ctypes.c_char_p),
                ("flash_attention", ctypes.c_bool),
                ("tensor_split", ctypes.c_float * tensor_split_max),
                ("quant_k", ctypes.c_int),
                ("quant_v", ctypes.c_int),
                ("check_slowness", ctypes.c_bool),
                ("highpriority", ctypes.c_bool),
                ("swa_support", ctypes.c_bool),
                ("smartcache", ctypes.c_bool),
                ("smartcacheslots", ctypes.c_int),
                ("pipelineparallel", ctypes.c_bool),
                ("lora_multiplier", ctypes.c_float),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class sd_load_model_inputs(ctypes.Structure):
    """Structure for Stable Diffusion model loading parameters"""
    _fields_ = [("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("vulkan_info", ctypes.c_char_p),
                ("threads", ctypes.c_int),
                ("quant", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("offload_cpu", ctypes.c_bool),
                ("vae_cpu", ctypes.c_bool),
                ("clip_cpu", ctypes.c_bool),
                ("diffusion_conv_direct", ctypes.c_bool),
                ("vae_conv_direct", ctypes.c_bool),
                ("taesd", ctypes.c_bool),
                ("tiled_vae_threshold", ctypes.c_int),
                ("t5xxl_filename", ctypes.c_char_p),
                ("clip1_filename", ctypes.c_char_p),
                ("clip2_filename", ctypes.c_char_p),
                ("vae_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("lora_multiplier", ctypes.c_float),
                ("lora_apply_mode", ctypes.c_int),
                ("photomaker_filename", ctypes.c_char_p),
                ("upscaler_filename", ctypes.c_char_p),
                ("img_hard_limit", ctypes.c_int),
                ("img_soft_limit", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class whisper_load_model_inputs(ctypes.Structure):
    """Structure for Whisper model loading parameters"""
    _fields_ = [("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("vulkan_info", ctypes.c_char_p),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class tts_load_model_inputs(ctypes.Structure):
    """Structure for TTS model loading parameters"""
    _fields_ = [("threads", ctypes.c_int),
                ("ttc_model_filename", ctypes.c_char_p),
                ("cts_model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("vulkan_info", ctypes.c_char_p),
                ("gpulayers", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("ttsmaxlen", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class embeddings_load_model_inputs(ctypes.Structure):
    """Structure for embeddings model loading parameters"""
    _fields_ = [("threads", ctypes.c_int),
                ("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("vulkan_info", ctypes.c_char_p),
                ("gpulayers", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("embeddingsmaxctx", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

# ============================================================================
# GENERATION INPUT STRUCTURES
# ============================================================================

class generation_inputs(ctypes.Structure):
    """Structure for text generation input parameters"""
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("memory", ctypes.c_char_p),
                ("negative_prompt", ctypes.c_char_p),
                ("guidance_scale", ctypes.c_float),
                ("images", ctypes.c_char_p * images_max),
                ("audio", ctypes.c_char_p * audio_max),
                ("max_context_length", ctypes.c_int),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_a", ctypes.c_float),
                ("top_p", ctypes.c_float),
                ("min_p", ctypes.c_float),
                ("typical_p", ctypes.c_float),
                ("tfs", ctypes.c_float),
                ("nsigma", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int),
                ("rep_pen_slope", ctypes.c_float),
                ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("xtc_threshold", ctypes.c_float),
                ("xtc_probability", ctypes.c_float),
                ("sampler_order", ctypes.c_int * sampler_order_max),
                ("sampler_len", ctypes.c_int),
                ("allow_eos_token", ctypes.c_bool),
                ("bypass_eos_token", ctypes.c_bool),
                ("tool_call_fix", ctypes.c_bool),
                ("render_special", ctypes.c_bool),
                ("stream_sse", ctypes.c_bool),
                ("grammar", ctypes.c_char_p),
                ("grammar_retain_state", ctypes.c_bool),
                ("dynatemp_range", ctypes.c_float),
                ("dynatemp_exponent", ctypes.c_float),
                ("smoothing_factor", ctypes.c_float),
                ("smoothing_curve", ctypes.c_float),
                ("adaptive_target", ctypes.c_float),
                ("adaptive_decay", ctypes.c_float),
                ("dry_multiplier", ctypes.c_float),
                ("dry_base", ctypes.c_float),
                ("dry_allowed_length", ctypes.c_int),
                ("dry_penalty_last_n", ctypes.c_int),
                ("dry_sequence_breakers_len", ctypes.c_int),
                ("dry_sequence_breakers", ctypes.POINTER(ctypes.c_char_p)),
                ("stop_sequence_len", ctypes.c_int),
                ("stop_sequence", ctypes.POINTER(ctypes.c_char_p)),
                ("logit_biases_len", ctypes.c_int),
                ("logit_biases", ctypes.POINTER(logit_bias)),
                ("banned_tokens_len", ctypes.c_int),
                ("banned_tokens", ctypes.POINTER(ctypes.c_char_p))]

class sd_generation_inputs(ctypes.Structure):
    """Structure for Stable Diffusion generation input parameters"""
    _fields_ = [("prompt", ctypes.c_char_p),
                ("negative_prompt", ctypes.c_char_p),
                ("init_images", ctypes.c_char_p),
                ("mask", ctypes.c_char_p),
                ("extra_images_len", ctypes.c_int),
                ("extra_images", ctypes.POINTER(ctypes.c_char_p)),
                ("flip_mask", ctypes.c_bool),
                ("denoising_strength", ctypes.c_float),
                ("cfg_scale", ctypes.c_float),
                ("distilled_guidance", ctypes.c_float),
                ("shifted_timestep", ctypes.c_int),
                ("sample_steps", ctypes.c_int),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int),
                ("seed", ctypes.c_int),
                ("sample_method", ctypes.c_char_p),
                ("scheduler", ctypes.c_char_p),
                ("clip_skip", ctypes.c_int),
                ("vid_req_frames", ctypes.c_int),
                ("video_output_type", ctypes.c_int),
                ("remove_limits", ctypes.c_bool),
                ("circular_x", ctypes.c_bool),
                ("circular_y", ctypes.c_bool),
                ("upscale", ctypes.c_bool)]

class sd_upscale_inputs(ctypes.Structure):
    """Structure for Stable Diffusion upscale input parameters"""
    _fields_ = [("init_images", ctypes.c_char_p),
                ("upscaling_resize", ctypes.c_int)]

class whisper_generation_inputs(ctypes.Structure):
    """Structure for Whisper generation input parameters"""
    _fields_ = [("prompt", ctypes.c_char_p),
                ("audio_data", ctypes.c_char_p),
                ("suppress_non_speech", ctypes.c_bool),
                ("langcode", ctypes.c_char_p)]

class tts_generation_inputs(ctypes.Structure):
    """Structure for TTS generation input parameters"""
    _fields_ = [("prompt", ctypes.c_char_p),
                ("speaker_seed", ctypes.c_int),
                ("audio_seed", ctypes.c_int),
                ("custom_speaker_voice", ctypes.c_char_p),
                ("custom_speaker_text", ctypes.c_char_p),
                ("custom_speaker_data", ctypes.c_char_p)]

class embeddings_generation_inputs(ctypes.Structure):
    """Structure for embeddings generation input parameters"""
    _fields_ = [("prompt", ctypes.c_char_p),
                ("truncate", ctypes.c_bool)]

# ============================================================================
# OUTPUT STRUCTURES
# ============================================================================

class generation_outputs(ctypes.Structure):
    """Structure for text generation output"""
    _fields_ = [("status", ctypes.c_int),
                ("stopreason", ctypes.c_int),
                ("prompt_tokens", ctypes.c_int),
                ("completion_tokens", ctypes.c_int),
                ("text", ctypes.c_char_p)]

class sd_generation_outputs(ctypes.Structure):
    """Structure for Stable Diffusion generation output"""
    _fields_ = [("status", ctypes.c_int),
                ("animated", ctypes.c_int),
                ("data", ctypes.c_char_p),
                ("data_extra", ctypes.c_char_p)]

class sd_info_outputs(ctypes.Structure):
    """Structure for Stable Diffusion info output"""
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class whisper_generation_outputs(ctypes.Structure):
    """Structure for Whisper generation output"""
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class tts_generation_outputs(ctypes.Structure):
    """Structure for TTS generation output"""
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class embeddings_generation_outputs(ctypes.Structure):
    """Structure for embeddings generation output"""
    _fields_ = [("status", ctypes.c_int),
                ("count", ctypes.c_int),
                ("data", ctypes.c_char_p)]

# ============================================================================
# LIBRARY INITIALIZATION AND FUNCTION BINDINGS
# ============================================================================

# Library file names (platform-specific)
def pick_existant_file(ntoption, nonntoption):
    """
    Helper function to pick the correct library file based on platform
    
    Args:
        ntoption: Windows library name (.dll)
        nonntoption: Unix library name (.so)
        
    Returns:
        The appropriate library filename for the current platform
    """
    precompiled_prefix = "precompiled_"
    
    def file_exists(filename):
        return os.path.exists(filename)
    
    ntexist = file_exists(ntoption)
    nonntexist = file_exists(nonntoption)
    precompiled_ntexist = file_exists(precompiled_prefix + ntoption)
    precompiled_nonntexist = file_exists(precompiled_prefix + nonntoption)
    
    if os.name == 'nt':
        if not ntexist and precompiled_ntexist:
            return (precompiled_prefix + ntoption)
        if nonntexist and not ntexist:
            return nonntoption
        return ntoption
    else:
        if not nonntexist and precompiled_nonntexist:
            return (precompiled_prefix + nonntoption)
        if ntexist and not nonntexist:
            return ntoption
        return nonntoption


# Available library variants
LIB_DEFAULT = pick_existant_file("koboldcpp_default.dll", "koboldcpp_default.so")
LIB_FAILSAFE = pick_existant_file("koboldcpp_failsafe.dll", "koboldcpp_failsafe.so")
LIB_NOAVX2 = pick_existant_file("koboldcpp_noavx2.dll", "koboldcpp_noavx2.so")
LIB_VULKAN_FAILSAFE = pick_existant_file("koboldcpp_vulkan_failsafe.dll", "koboldcpp_vulkan_failsafe.so")
LIB_CUBLAS = pick_existant_file("koboldcpp_cublas.dll", "koboldcpp_cublas.so")
LIB_HIPBLAS = pick_existant_file("koboldcpp_hipblas.dll", "koboldcpp_hipblas.so")
LIB_VULKAN = pick_existant_file("koboldcpp_vulkan.dll", "koboldcpp_vulkan.so")
LIB_VULKAN_NOAVX2 = pick_existant_file("koboldcpp_vulkan_noavx2.dll", "koboldcpp_vulkan_noavx2.so")


def init_library(libname, dir_path):
    """
    Initialize the KoboldCpp C++ library and set up all function bindings
    
    Args:
        libname: Name of the library file to load (use LIB_* constants)
        dir_path: Directory path where the library is located
        
    Returns:
        handle: ctypes.CDLL handle to the loaded library
        
    Example:
        handle = init_library(LIB_DEFAULT, "/path/to/library")
    """
    # Load the library
    handle = ctypes.CDLL(os.path.join(dir_path, libname))
    
    # ========================================================================
    # TEXT/GGUF MODEL FUNCTIONS
    # ========================================================================
    
    handle.load_model.argtypes = [load_model_inputs]
    handle.load_model.restype = ctypes.c_bool
    
    handle.generate.argtypes = [generation_inputs]
    handle.generate.restype = generation_outputs
    
    handle.new_token.restype = ctypes.c_char_p
    handle.new_token.argtypes = [ctypes.c_int]
    
    handle.get_stream_count.restype = ctypes.c_int
    handle.has_finished.restype = ctypes.c_bool
    handle.abort_generate.restype = ctypes.c_bool
    
    # ========================================================================
    # CAPABILITY CHECKS
    # ========================================================================
    
    handle.has_audio_support.restype = ctypes.c_bool
    handle.has_vision_support.restype = ctypes.c_bool
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    handle.get_last_eval_time.restype = ctypes.c_float
    handle.get_last_process_time.restype = ctypes.c_float
    handle.get_last_token_count.restype = ctypes.c_int
    handle.get_last_input_count.restype = ctypes.c_int
    handle.get_last_seed.restype = ctypes.c_int
    handle.get_last_draft_success.restype = ctypes.c_int
    handle.get_last_draft_failed.restype = ctypes.c_int
    handle.get_total_img_gens.restype = ctypes.c_int
    handle.get_total_tts_gens.restype = ctypes.c_int
    handle.get_total_transcribe_gens.restype = ctypes.c_int
    handle.get_total_gens.restype = ctypes.c_int
    handle.get_last_stop_reason.restype = ctypes.c_int
    
    # ========================================================================
    # TOKEN OPERATIONS
    # ========================================================================
    
    handle.token_count.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    handle.token_count.restype = token_count_outputs
    handle.get_pending_output.restype = ctypes.c_char_p
    handle.get_chat_template.restype = ctypes.c_char_p
    handle.detokenize.argtypes = [token_count_outputs]
    handle.detokenize.restype = ctypes.c_char_p
    handle.last_logprobs.restype = last_logprobs_outputs
    
    # ========================================================================
    # STATE MANAGEMENT (KV Cache)
    # ========================================================================
    
    handle.calc_new_state_kv.restype = ctypes.c_size_t
    handle.calc_new_state_tokencount.restype = ctypes.c_size_t
    
    handle.calc_old_state_kv.argtypes = [ctypes.c_int]
    handle.calc_old_state_kv.restype = ctypes.c_size_t
    
    handle.calc_old_state_tokencount.argtypes = [ctypes.c_int]
    handle.calc_old_state_tokencount.restype = ctypes.c_size_t
    
    handle.save_state_kv.argtypes = [ctypes.c_int]
    handle.save_state_kv.restype = ctypes.c_size_t
    
    handle.load_state_kv.argtypes = [ctypes.c_int]
    handle.load_state_kv.restype = ctypes.c_bool
    
    handle.clear_state_kv.restype = ctypes.c_bool
    
    # ========================================================================
    # STABLE DIFFUSION FUNCTIONS
    # ========================================================================
    
    handle.sd_load_model.argtypes = [sd_load_model_inputs]
    handle.sd_load_model.restype = ctypes.c_bool
    
    handle.sd_generate.argtypes = [sd_generation_inputs]
    handle.sd_generate.restype = sd_generation_outputs
    
    handle.sd_upscale.argtypes = [sd_upscale_inputs]
    handle.sd_upscale.restype = sd_generation_outputs
    
    handle.sd_get_info.argtypes = []
    handle.sd_get_info.restype = sd_info_outputs
    
    # ========================================================================
    # WHISPER (SPEECH-TO-TEXT) FUNCTIONS
    # ========================================================================
    
    handle.whisper_load_model.argtypes = [whisper_load_model_inputs]
    handle.whisper_load_model.restype = ctypes.c_bool
    
    handle.whisper_generate.argtypes = [whisper_generation_inputs]
    handle.whisper_generate.restype = whisper_generation_outputs
    
    # ========================================================================
    # TTS (TEXT-TO-SPEECH) FUNCTIONS
    # ========================================================================
    
    handle.tts_load_model.argtypes = [tts_load_model_inputs]
    handle.tts_load_model.restype = ctypes.c_bool
    
    handle.tts_generate.argtypes = [tts_generation_inputs]
    handle.tts_generate.restype = tts_generation_outputs
    
    # ========================================================================
    # EMBEDDINGS FUNCTIONS
    # ========================================================================
    
    handle.embeddings_load_model.argtypes = [embeddings_load_model_inputs]
    handle.embeddings_load_model.restype = ctypes.c_bool
    
    handle.embeddings_generate.argtypes = [embeddings_generation_inputs]
    handle.embeddings_generate.restype = embeddings_generation_outputs
    
    return handle


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the bindings:
    
    # 1. Initialize library
    handle = init_library(LIB_DEFAULT, "/path/to/library")
    
    # 2. Load a text model
    inputs = load_model_inputs()
    inputs.threads = 4
    inputs.blasthreads = 4
    inputs.max_context_length = 2048
    inputs.model_filename = b"/path/to/model.gguf"
    inputs.executable_path = b"/path/to/library/"
    inputs.use_mmap = True
    inputs.use_mlock = False
    inputs.gpulayers = 0  # 0 for CPU only
    # ... set other parameters as needed
    success = handle.load_model(inputs)
    
    if not success:
        print("Failed to load model!")
        exit(1)
    
    # 3. Generate text
    gen_inputs = generation_inputs()
    gen_inputs.prompt = b"Once upon a time"
    gen_inputs.max_length = 100
    gen_inputs.temperature = 0.7
    gen_inputs.top_k = 40
    gen_inputs.top_p = 0.9
    gen_inputs.seed = -1  # random seed
    # ... set other sampling parameters
    
    result = handle.generate(gen_inputs)
    
    if result.status == 1:
        print(f"Generated: {result.text.decode('utf-8')}")
        print(f"Prompt tokens: {result.prompt_tokens}")
        print(f"Completion tokens: {result.completion_tokens}")
    else:
        print("Generation failed!")
    
    # 4. Token counting
    text_to_count = b"Hello, world!"
    token_data = handle.token_count(text_to_count, True)  # True = add special tokens
    print(f"Token count: {token_data.count}")
    
    # 5. Load and use Stable Diffusion model
    sd_inputs = sd_load_model_inputs()
    sd_inputs.model_filename = b"/path/to/sd_model.safetensors"
    sd_inputs.executable_path = b"/path/to/library/"
    sd_inputs.threads = 4
    # ... set other SD parameters
    
    sd_success = handle.sd_load_model(sd_inputs)
    
    if sd_success:
        sd_gen = sd_generation_inputs()
        sd_gen.prompt = b"a beautiful landscape"
        sd_gen.negative_prompt = b"ugly, blurry"
        sd_gen.width = 512
        sd_gen.height = 512
        sd_gen.sample_steps = 20
        sd_gen.cfg_scale = 7.0
        sd_gen.seed = -1
        
        sd_result = handle.sd_generate(sd_gen)
        if sd_result.status == 1:
            # sd_result.data contains base64 encoded image
            print("Image generated successfully!")
    
    # 6. Load and use Whisper model (speech-to-text)
    whisper_inputs = whisper_load_model_inputs()
    whisper_inputs.model_filename = b"/path/to/whisper_model.bin"
    whisper_inputs.executable_path = b"/path/to/library/"
    
    whisper_success = handle.whisper_load_model(whisper_inputs)
    
    if whisper_success:
        whisper_gen = whisper_generation_inputs()
        whisper_gen.audio_data = b"base64_encoded_audio_data"
        whisper_gen.langcode = b"en"
        
        whisper_result = handle.whisper_generate(whisper_gen)
        if whisper_result.status == 1:
            print(f"Transcription: {whisper_result.data.decode('utf-8')}")
    
    # 7. Load and use TTS model (text-to-speech)
    tts_inputs = tts_load_model_inputs()
    tts_inputs.ttc_model_filename = b"/path/to/tts_model.gguf"
    tts_inputs.executable_path = b"/path/to/library/"
    tts_inputs.threads = 4
    
    tts_success = handle.tts_load_model(tts_inputs)
    
    if tts_success:
        tts_gen = tts_generation_inputs()
        tts_gen.prompt = b"Hello, this is a test."
        tts_gen.speaker_seed = -1
        tts_gen.audio_seed = -1
        
        tts_result = handle.tts_generate(tts_gen)
        if tts_result.status == 1:
            # tts_result.data contains audio data
            print("Audio generated successfully!")
    
    # 8. Load and use Embeddings model
    emb_inputs = embeddings_load_model_inputs()
    emb_inputs.model_filename = b"/path/to/embeddings_model.gguf"
    emb_inputs.executable_path = b"/path/to/library/"
    emb_inputs.threads = 4
    
    emb_success = handle.embeddings_load_model(emb_inputs)
    
    if emb_success:
        emb_gen = embeddings_generation_inputs()
        emb_gen.prompt = b"This is a test sentence."
        emb_gen.truncate = False
        
        emb_result = handle.embeddings_generate(emb_gen)
        if emb_result.status == 1:
            print(f"Embedding dimension: {emb_result.count}")
            # emb_result.data contains the embedding vector as JSON
    
    # 9. Streaming generation
    gen_inputs.stream_sse = True
    result = handle.generate(gen_inputs)
    
    current_token = 0
    while not handle.has_finished():
        stream_count = handle.get_stream_count()
        while current_token < stream_count:
            token = handle.new_token(current_token)
            print(token.decode('utf-8'), end='', flush=True)
            current_token += 1
        time.sleep(0.01)  # Small delay to avoid busy waiting
    
    print()  # New line after streaming
    
    # 10. State management (save/load KV cache)
    # Save current state to slot 0
    saved_size = handle.save_state_kv(0)
    print(f"Saved {saved_size} bytes of KV cache")
    
    # Load state from slot 0
    load_success = handle.load_state_kv(0)
    if load_success:
        print("State loaded successfully!")
    
    # Clear all saved states
    handle.clear_state_kv()
    
    # 11. Get performance metrics
    eval_time = handle.get_last_eval_time()
    process_time = handle.get_last_process_time()
    token_count = handle.get_last_token_count()
    print(f"Eval time: {eval_time}ms, Process time: {process_time}ms")
    print(f"Tokens: {token_count}, Speed: {token_count/eval_time*1000:.2f} t/s")
    
    # 12. Abort generation (if needed)
    # handle.abort_generate()
    """
    print("KoboldCpp Bindings Module")
    print("Import this module to use KoboldCpp C++ library bindings")
    print()
    print("Available library variants:")
    print(f"  LIB_DEFAULT: {LIB_DEFAULT}")
    print(f"  LIB_CUBLAS: {LIB_CUBLAS}")
    print(f"  LIB_HIPBLAS: {LIB_HIPBLAS}")
    print(f"  LIB_VULKAN: {LIB_VULKAN}")
    print(f"  LIB_NOAVX2: {LIB_NOAVX2}")
    print(f"  LIB_VULKAN_NOAVX2: {LIB_VULKAN_NOAVX2}")
    print(f"  LIB_VULKAN_FAILSAFE: {LIB_VULKAN_FAILSAFE}")
    print(f"  LIB_FAILSAFE: {LIB_FAILSAFE}")