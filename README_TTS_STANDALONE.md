# Standalone TTS Library untuk KoboldCpp

Library TTS standalone yang di-extract dari KoboldCpp, bisa digunakan secara independen tanpa perlu compile seluruh KoboldCpp.

## Fitur

- **Multi-engine TTS**: Mendukung Parler TTS, Kokoro, Dia, dan Orpheus
- **Lightweight**: Hanya compile komponen TTS yang diperlukan
- **Python binding**: Mudah digunakan dengan ctypes
- **Cross-platform**: Linux dan macOS

## Cara Build

### Prerequisites

- GCC/Clang compiler dengan C++17 support
- Make atau CMake (opsional)
- Python 3.x (untuk testing)

### Build Library

```bash
chmod +x build_tts_standalone.sh
./build_tts_standalone.sh
```

Output: `build_tts/libkcpp_tts.so` (Linux) atau `libkcpp_tts.dylib` (macOS)

### Build dengan GPU Support (CUDA)

Untuk CUDA support, edit `build_tts_standalone.sh` dan tambahkan:

```bash
CFLAGS="$CFLAGS -DGGML_USE_CUDA"
CXXFLAGS="$CXXFLAGS -DGGML_USE_CUDA"
LDFLAGS="$LDFLAGS -lcuda -lcublas -lcudart"
```

## Cara Pakai

### Python Example

```python
import ctypes
import base64

# Load library
tts_lib = ctypes.CDLL("./build_tts/libkcpp_tts.so")

# Define structures (lihat example_tts_usage.py untuk lengkapnya)
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

# Load model
inputs = tts_load_model_inputs()
inputs.ttc_model_filename = b"model.gguf"
inputs.threads = 4
tts_lib.tts_load_model(inputs)

# Generate speech
gen_inputs = tts_generation_inputs()
gen_inputs.prompt = b"Hello world"
gen_inputs.speaker_seed = 1
result = tts_lib.tts_generate(gen_inputs)

# Save WAV file
if result.status == 1:
    wav_data = base64.b64decode(result.data.decode())
    with open("output.wav", "wb") as f:
        f.write(wav_data)
```

### C/C++ Example

```cpp
#include "expose.h"

int main() {
    // Load model
    tts_load_model_inputs load_inputs = {};
    load_inputs.ttc_model_filename = "model.gguf";
    load_inputs.threads = 4;
    
    bool loaded = tts_load_model(load_inputs);
    
    // Generate speech
    tts_generation_inputs gen_inputs = {};
    gen_inputs.prompt = "Hello world";
    gen_inputs.speaker_seed = 1;
    
    tts_generation_outputs output = tts_generate(gen_inputs);
    
    if (output.status == 1) {
        // output.data contains base64 encoded WAV
        printf("Generated: %s\n", output.data);
    }
    
    return 0;
}
```

Compile:
```bash
g++ -o tts_test test.cpp -L./build_tts -lkcpp_tts -pthread
```

## Download Model

Download model TTS dari:
- https://huggingface.co/koboldcpp/tts/tree/main

Model yang tersedia:
- **Kokoro** (Recommended): Cepat dan berkualitas baik
- **Parler TTS Mini**: Lebih kecil, lebih cepat
- **Parler TTS Large**: Kualitas terbaik, lebih lambat
- **Dia**: Voice customization lebih baik
- **Orpheus**: Experimental

## Voice Selection

Gunakan `speaker_seed` untuk memilih voice:

```python
# Preset voices (1-5)
voice_seed = 1  # kobo
voice_seed = 2  # cheery
voice_seed = 3  # sleepy
voice_seed = 4  # shouty
voice_seed = 5  # chatty

# Custom seed (any integer)
voice_seed = 12345
```

Atau gunakan nama voice:
```python
gen_inputs.custom_speaker_voice = b"kobo"
```

## Voice Cloning

Untuk clone voice dari audio sample:

```python
# Load voice data dari JSON
speaker_json = {
    "phrase": "Sample text",
    "voice": "<voice_encoding_data>"
}

gen_inputs.custom_speaker_text = speaker_json["phrase"].encode()
gen_inputs.custom_speaker_data = speaker_json["voice"].encode()
gen_inputs.speaker_seed = 100  # Special seed untuk custom voice
```

## API Reference

### tts_load_model_inputs

| Field | Type | Description |
|-------|------|-------------|
| threads | int | Jumlah thread untuk inference |
| ttc_model_filename | char* | Path ke model GGUF |
| cts_model_filename | char* | Path ke tokenizer (opsional) |
| gpulayers | int | Jumlah layer di GPU (0=CPU, 999=semua) |
| flash_attention | bool | Enable flash attention |
| ttsmaxlen | int | Max sequence length (default: 4096) |
| debugmode | int | Debug level (0=off) |

### tts_generation_inputs

| Field | Type | Description |
|-------|------|-------------|
| prompt | char* | Text untuk di-generate |
| speaker_seed | int | Voice selection seed |
| audio_seed | int | Random seed untuk audio (-1=random) |
| custom_speaker_voice | char* | Nama voice preset |
| custom_speaker_text | char* | Text untuk voice cloning |
| custom_speaker_data | char* | Voice encoding data |

### tts_generation_outputs

| Field | Type | Description |
|-------|------|-------------|
| status | int | 1=success, -1=error |
| data | char* | Base64 encoded WAV file |

## Output Format

Output audio adalah WAV file dengan format:
- **Sample rate**: 44100 Hz (tergantung model)
- **Channels**: 1 (mono)
- **Bit depth**: 16-bit PCM
- **Encoding**: Base64 string

Decode dengan:
```python
import base64
wav_bytes = base64.b64decode(output.data.decode())
```

## Troubleshooting

### Library tidak bisa di-load
```bash
# Check dependencies
ldd build_tts/libkcpp_tts.so

# Set LD_LIBRARY_PATH jika perlu
export LD_LIBRARY_PATH=./build_tts:$LD_LIBRARY_PATH
```

### Model tidak bisa di-load
- Pastikan path model benar
- Check format model (harus GGUF)
- Coba dengan debugmode=1 untuk detail error

### Audio quality buruk
- Coba model yang lebih besar (Parler Large)
- Adjust speaker_seed untuk voice yang berbeda
- Pastikan input text tidak terlalu panjang

## Performance Tips

1. **CPU**: Gunakan thread count = jumlah core CPU
2. **GPU**: Set `gpulayers=999` untuk offload semua layer
3. **Memory**: Model besar butuh 2-4GB RAM
4. **Speed**: Kokoro paling cepat, Parler Large paling lambat

## Lisensi

- TTS.cpp: MIT License
- KoboldCpp: AGPL v3.0 License
- GGML: MIT License

## Credits

- TTS.cpp: https://github.com/mmwillet/TTS.cpp
- KoboldCpp: https://github.com/LostRuins/koboldcpp
- GGML: https://github.com/ggerganov/ggml

## Support

Untuk issues dan questions:
- TTS.cpp issues: https://github.com/mmwillet/TTS.cpp/issues
- KoboldCpp issues: https://github.com/LostRuins/koboldcpp/issues
