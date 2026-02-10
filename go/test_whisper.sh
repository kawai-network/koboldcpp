#!/bin/bash
# Test script for Whisper Go bindings

set -e

echo "=== KoboldCpp Whisper Go Bindings Test ==="
echo ""

# Check if library exists
if [ ! -f "../koboldcpp_default.so" ] && [ ! -f "../koboldcpp_vulkan.so" ]; then
    echo "Error: KoboldCpp library not found!"
    echo "Please compile KoboldCpp first:"
    echo "  cd .. && make"
    exit 1
fi

# Check if model exists
MODEL_PATH="../models/whisper-base.bin"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Whisper model not found at $MODEL_PATH"
    echo "Download from: https://huggingface.co/koboldcpp/whisper/tree/main"
    echo ""
    echo "Example:"
    echo "  mkdir -p ../models"
    echo "  wget https://huggingface.co/koboldcpp/whisper/resolve/main/ggml-base.bin -O $MODEL_PATH"
    echo ""
fi

# Build the example
echo "Building example..."
cd examples
go build -o whisper_example whisper_example.go

echo ""
echo "Build successful!"
echo ""
echo "To run the example:"
echo "  cd examples"
echo "  ./whisper_example ../.. $MODEL_PATH /path/to/audio.wav"
echo ""
echo "Or run the simple example:"
echo "  go run simple_whisper.go"
