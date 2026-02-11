#!/usr/bin/env python3
"""
Example: Using standalone TTS library with Python ctypes
"""

import ctypes
import os
import base64

# Load the TTS library
lib_path = "./build_tts/libkcpp_tts.so"  # or .dylib on macOS
if not os.path.exists(lib_path):
    lib_path = "./build_tts/libkcpp_tts.dylib"
    
tts_lib = ctypes.CDLL(lib_path)

# Define structures matching expose.h
class tts_load_model_inputs(ctypes.Structure):
    _fields_ = [
        ("threads", ctypes.c_int),
        ("ttc_model_filename", ctypes.c_char_p),
        ("cts_model_filename", ctypes.c_char_p),
        ("gpulayers", ctypes.c_int),
        ("flash_attention", ctypes.c_bool),
        ("ttsmaxlen", ctypes.c_int),
        ("debugmode", ctypes.c_int)
    ]

class tts_generation_inputs(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("speaker_seed", ctypes.c_int),
        ("audio_seed", ctypes.c_int),
        ("custom_speaker_voice", ctypes.c_char_p),
        ("custom_speaker_text", ctypes.c_char_p),
        ("custom_speaker_data", ctypes.c_char_p)
    ]

class tts_generation_outputs(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("data", ctypes.c_char_p)
    ]

# Set function signatures
tts_lib.tts_load_model.argtypes = [tts_load_model_inputs]
tts_lib.tts_load_model.restype = ctypes.c_bool

tts_lib.tts_generate.argtypes = [tts_generation_inputs]
tts_lib.tts_generate.restype = tts_generation_outputs


def load_tts_model(model_path, tokenizer_path="", threads=4):
    """Load TTS model"""
    inputs = tts_load_model_inputs()
    inputs.ttc_model_filename = model_path.encode("UTF-8")
    inputs.cts_model_filename = tokenizer_path.encode("UTF-8")
    inputs.threads = threads
    inputs.gpulayers = 0  # CPU only, set to 999 for GPU
    inputs.flash_attention = False
    inputs.ttsmaxlen = 4096
    inputs.debugmode = 0
    
    result = tts_lib.tts_load_model(inputs)
    return result


def generate_speech(text, voice_seed=1, audio_seed=-1, voice_name=""):
    """Generate speech from text"""
    inputs = tts_generation_inputs()
    inputs.prompt = text.encode("UTF-8")
    inputs.speaker_seed = voice_seed
    inputs.audio_seed = audio_seed
    inputs.custom_speaker_voice = voice_name.encode("UTF-8")
    inputs.custom_speaker_text = b""
    inputs.custom_speaker_data = b""
    
    result = tts_lib.tts_generate(inputs)
    
    if result.status == 1:
        # Data is base64 encoded WAV
        wav_base64 = result.data.decode("UTF-8")
        wav_data = base64.b64decode(wav_base64)
        return wav_data
    else:
        return None


def main():
    print("=== Standalone TTS Library Example ===\n")
    
    # Load model
    model_path = "path/to/your/tts_model.gguf"
    print(f"Loading TTS model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("\nDownload TTS models from:")
        print("https://huggingface.co/koboldcpp/tts/tree/main")
        return
    
    success = load_tts_model(model_path, threads=4)
    
    if not success:
        print("Failed to load TTS model!")
        return
    
    print("Model loaded successfully!\n")
    
    # Generate speech
    text = "Hello, this is a test of the standalone TTS library."
    print(f"Generating speech for: '{text}'")
    
    wav_data = generate_speech(text, voice_seed=1)
    
    if wav_data:
        output_file = "output.wav"
        with open(output_file, "wb") as f:
            f.write(wav_data)
        print(f"\nSpeech generated successfully!")
        print(f"Output saved to: {output_file}")
        print(f"File size: {len(wav_data)} bytes")
    else:
        print("Failed to generate speech!")


if __name__ == "__main__":
    main()
