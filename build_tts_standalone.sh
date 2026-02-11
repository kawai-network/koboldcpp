#!/bin/bash
# Standalone TTS Library Builder for KoboldCpp
# This script builds only the TTS components as a shared library

set -e

echo "==================================="
echo "Building Standalone TTS Library"
echo "==================================="

# Compiler settings
CC=${CC:-gcc}
CXX=${CXX:-g++}

# Build flags
CFLAGS="-I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Iinclude -Isrc -Icommon -Ivendor -Ivendor/stb -Iotherarch -Iotherarch/tools -Iotherarch/ttscpp/include -Iotherarch/ttscpp/src -O3 -fno-finite-math-only -std=c11 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -pthread"

CXXFLAGS="-I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Iinclude -Isrc -Icommon -Ivendor -Ivendor/stb -Iotherarch -Iotherarch/tools -Iotherarch/ttscpp/include -Iotherarch/ttscpp/src -O3 -fno-finite-math-only -std=c++17 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -pthread -Wno-multichar -Wno-write-strings -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-variable"

LDFLAGS="-pthread -lm"

# Detect OS
UNAME_S=$(uname -s)
if [ "$UNAME_S" = "Darwin" ]; then
    LDFLAGS="$LDFLAGS -framework Accelerate"
    LIB_EXT="dylib"
elif [ "$UNAME_S" = "Linux" ]; then
    LDFLAGS="$LDFLAGS -ldl"
    LIB_EXT="so"
else
    echo "Unsupported OS: $UNAME_S"
    exit 1
fi

# Create build directory
mkdir -p build_tts

echo "Compiling GGML core components..."

# GGML core - compile from root, output to build_tts
$CC $CFLAGS -c ggml/src/ggml.c -o build_tts/ggml.o
$CC $CFLAGS -c ggml/src/ggml-alloc.c -o build_tts/ggml-alloc.o
$CC $CFLAGS -c ggml/src/ggml-quants.c -o build_tts/ggml-quants.o
$CC $CFLAGS -c ggml/src/ggml-cpu/ggml-cpu.c -o build_tts/ggml-cpu.o
$CC $CFLAGS -c ggml/src/ggml-cpu/quants.c -o build_tts/ggml-cpu-quants.o
$CC $CFLAGS -c ggml/src/ggml-cpu/kcpp-quantmapper.c -o build_tts/kcpp-quantmapper.o

# GGML C++ components
$CXX $CXXFLAGS -c ggml/src/ggml-backend.cpp -o build_tts/ggml-backend.o
$CXX $CXXFLAGS -c ggml/src/ggml-backend-reg.cpp -o build_tts/ggml-backend-reg.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/traits.cpp -o build_tts/ggml-cpu-traits.o
$CXX $CXXFLAGS -c ggml/src/ggml-threading.cpp -o build_tts/ggml-threading.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/ggml-cpu.cpp -o build_tts/ggml-cpu-cpp.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/kcpp-repackmapper.cpp -o build_tts/kcpp-repackmapper.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/llamafile/sgemm.cpp -o build_tts/sgemm.o
$CXX $CXXFLAGS -c ggml/src/gguf.cpp -o build_tts/gguf.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/binary-ops.cpp -o build_tts/ggml-binops.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/unary-ops.cpp -o build_tts/ggml-unops.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/ops.cpp -o build_tts/ggml-ops.o
$CXX $CXXFLAGS -c ggml/src/ggml-cpu/vec.cpp -o build_tts/ggml-vec.o

echo "Compiling common utilities..."

# Common utilities
$CXX $CXXFLAGS -c src/unicode.cpp -o build_tts/unicode.o
$CXX $CXXFLAGS -c src/unicode-data.cpp -o build_tts/unicode-data.o
$CXX $CXXFLAGS -c src/llama-impl.cpp -o build_tts/llama-impl.o
$CXX $CXXFLAGS -c common/common.cpp -o build_tts/common.o
$CXX $CXXFLAGS -c common/sampling.cpp -o build_tts/sampling.o
$CXX $CXXFLAGS -c otherarch/utils.cpp -o build_tts/kcpputils.o
$CXX $CXXFLAGS -c tools/mtmd/mtmd-audio.cpp -o build_tts/mtmdaudio.o

echo "Compiling llama.cpp..."

# Llama.cpp main file
$CXX $CXXFLAGS -c src/llama.cpp -o build_tts/llama.o

echo "Compiling TTS adapter..."

# TTS adapter (includes all TTS.cpp files)
$CXX $CXXFLAGS -c otherarch/tts_adapter.cpp -o build_tts/tts_adapter.o

echo "Creating TTS wrapper for standalone use..."

# Create a simple wrapper
cat > build_tts/tts_wrapper.cpp << 'EOF'
#include <string>
#include <vector>
#include <cstring>
#include "../expose.h"

extern "C" {
    // Forward declarations from tts_adapter
    bool ttstype_load_model(const tts_load_model_inputs inputs);
    tts_generation_outputs ttstype_generate(const tts_generation_inputs inputs);
    
    // Exported functions
    bool tts_load_model(const tts_load_model_inputs inputs) {
        return ttstype_load_model(inputs);
    }
    
    tts_generation_outputs tts_generate(const tts_generation_inputs inputs) {
        return ttstype_generate(inputs);
    }
}
EOF

$CXX $CXXFLAGS -c build_tts/tts_wrapper.cpp -o build_tts/tts_wrapper.o

echo "Linking shared library..."

# Link everything into shared library
$CXX -shared -o build_tts/libkcpp_tts.$LIB_EXT \
    build_tts/ggml.o build_tts/ggml-alloc.o build_tts/ggml-quants.o build_tts/ggml-cpu.o build_tts/ggml-cpu-quants.o build_tts/kcpp-quantmapper.o \
    build_tts/ggml-backend.o build_tts/ggml-backend-reg.o build_tts/ggml-cpu-traits.o build_tts/ggml-threading.o build_tts/ggml-cpu-cpp.o \
    build_tts/kcpp-repackmapper.o build_tts/sgemm.o build_tts/gguf.o build_tts/ggml-binops.o build_tts/ggml-unops.o build_tts/ggml-ops.o build_tts/ggml-vec.o \
    build_tts/unicode.o build_tts/unicode-data.o build_tts/llama-impl.o build_tts/common.o build_tts/sampling.o build_tts/kcpputils.o build_tts/mtmdaudio.o \
    build_tts/llama.o build_tts/tts_adapter.o build_tts/tts_wrapper.o \
    $LDFLAGS

echo ""
echo "==================================="
echo "Build complete!"
echo "Library: build_tts/libkcpp_tts.$LIB_EXT"
echo "==================================="
echo ""
echo "To use this library:"
echo "1. Copy libkcpp_tts.$LIB_EXT to your project"
echo "2. Include expose.h for struct definitions"
echo "3. Link with -lkcpp_tts"
echo ""

# Test trigger for GitHub Actions
# This line added to trigger workflow
