#!/bin/sh

cd ~/llama.cpp
git pull origin master

rm -rf build

cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build build -j

./build/bin/llama-server --help
