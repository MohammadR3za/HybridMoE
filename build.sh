#!/usr/bin/env bash
# Builds the hybrid-moe branch of llama.cpp with the Vulkan backend.
#
# The Vulkan and SPIR-V headers live in deps/ because this machine has the
# Vulkan loader and glslc installed but not the header packages, and installing
# them needs root. Both are header-only, so a local copy is all the build needs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD:-$ROOT/llama.cpp/build}"   # BUILD=... builds elsewhere, e.g. while build/ is in use

# The project path contains a space, which CMAKE_CXX_FLAGS would split, so the
# SPIR-V include directory goes through gcc's CPATH instead.
export CPATH="$ROOT/deps/spirv/include${CPATH:+:$CPATH}"

cmake -S "$ROOT/llama.cpp" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CCACHE=OFF \
    -DLLAMA_CURL=OFF \
    -DCMAKE_PREFIX_PATH="$ROOT/deps/spirv" \
    -DVulkan_INCLUDE_DIR="$ROOT/deps/vulkan-headers/include" \
    -DVulkan_LIBRARY=/usr/lib/libvulkan.so

cmake --build "$BUILD" -j16 --target llama-bench llama-cli llama-server llama-perplexity "$@"
