# llama-v100-builder

Automated CI/CD builder using GitHub Actions to compile [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) specifically targeted for the **NVIDIA Tesla V100 (Volta sm_70)** with native SASS code generation (`-DCMAKE_CUDA_ARCHITECTURES="70"`).

## Features
- **100% Portable**: Produces a zero-install zip artifact containing `llama-server.exe`, `llama-cli.exe`, `ggml-cuda.dll` and runtime CUDA DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`).
- **No Driver JIT Overhead**: Compiles native `sm_70` Tensor Core / SASS kernels directly.
- **Volta Odd-GQA FlashAttention Fix**: Automatically patches `fattn.cu` to route non-power-of-2 GQA architectures (like Nanbeige 4.3 33B with 56:8 / 7:1 GQA) to the safe `TILE` FlashAttention kernel, eliminating WMMA unaligned warp deadlocks on `sm_70`.
- **Workflow Dispatch**: Trigger anytime via GitHub UI or CLI (`gh workflow run build-volta.yml`).
