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
CFLAGS="-I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Iinclude -Isrc -I./common -I./vendor -I./vendor/stb -I./include -I./otherarch -I./otherarch/tools -I./otherarch/ttscpp/include -I./otherarch/ttscpp/src -O3 -fno-finite-math-only -std=c11 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -pthread"

CXXFLAGS="-I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Iinclude -Isrc -I./common -I./vendor -I./vendor/stb -I./include -I./otherarch -I./otherarch/tools -I./otherarch/ttscpp/include -I./otherarch/ttscpp/src -O3 -fno-finite-math-only -std=c++17 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -pthread -Wno-multichar -Wno-write-strings -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-variable"

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
cd build_tts

echo "Compiling GGML core components..."

# GGML core
$CC $CFLAGS -c ../ggml/src/ggml.c -o ggml.o
$CC $CFLAGS -c ../ggml/src/ggml-alloc.c -o ggml-alloc.o
$CC $CFLAGS -c ../ggml/src/ggml-quants.c -o ggml-quants.o
$CC $CFLAGS -c ../ggml/src/ggml-cpu/ggml-cpu.c -o ggml-cpu.o
$CC $CFLAGS -c ../ggml/src/ggml-cpu/quants.c -o ggml-cpu-quants.o
$CC $CFLAGS -c ../ggml/src/ggml-cpu/kcpp-quantmapper.c -o kcpp-quantmapper.o

# GGML C++ components
$CXX $CXXFLAGS -c ../ggml/src/ggml-backend.cpp -o ggml-backend.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-backend-reg.cpp -o ggml-backend-reg.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/traits.cpp -o ggml-cpu-traits.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-threading.cpp -o ggml-threading.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/ggml-cpu.cpp -o ggml-cpu-cpp.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/kcpp-repackmapper.cpp -o kcpp-repackmapper.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/llamafile/sgemm.cpp -o sgemm.o
$CXX $CXXFLAGS -c ../ggml/src/gguf.cpp -o gguf.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/binary-ops.cpp -o ggml-binops.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/unary-ops.cpp -o ggml-unops.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/ops.cpp -o ggml-ops.o
$CXX $CXXFLAGS -c ../ggml/src/ggml-cpu/vec.cpp -o ggml-vec.o

echo "Compiling common utilities..."

# Common utilities
$CXX $CXXFLAGS -c ../src/unicode.cpp -o unicode.o
$CXX $CXXFLAGS -c ../src/unicode-data.cpp -o unicode-data.o
$CXX $CXXFLAGS -c ../src/llama-impl.cpp -o llama-impl.o
$CXX $CXXFLAGS -c ../common/common.cpp -o common.o
$CXX $CXXFLAGS -c ../common/sampling.cpp -o sampling.o
$CXX $CXXFLAGS -c ../otherarch/utils.cpp -o kcpputils.o
$CXX $CXXFLAGS -c ../tools/mtmd/mtmd-audio.cpp -o mtmdaudio.o

echo "Compiling TTS adapter..."

# TTS adapter (includes all TTS.cpp files)
$CXX $CXXFLAGS -c ../otherarch/tts_adapter.cpp -o tts_adapter.o

echo "Creating TTS wrapper for standalone use..."

# Create a simple wrapper
cat > tts_wrapper.cpp << 'EOF'
#include "../../expose.h"
#include <cstring>

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

$CXX $CXXFLAGS -c tts_wrapper.cpp -o tts_wrapper.o

echo "Linking shared library..."

# Link everything into shared library
$CXX -shared -o libkcpp_tts.$LIB_EXT \
    ggml.o ggml-alloc.o ggml-quants.o ggml-cpu.o ggml-cpu-quants.o kcpp-quantmapper.o \
    ggml-backend.o ggml-backend-reg.o ggml-cpu-traits.o ggml-threading.o ggml-cpu-cpp.o \
    kcpp-repackmapper.o sgemm.o gguf.o ggml-binops.o ggml-unops.o ggml-ops.o ggml-vec.o \
    unicode.o unicode-data.o llama-impl.o common.o sampling.o kcpputils.o mtmdaudio.o \
    tts_adapter.o tts_wrapper.o \
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
