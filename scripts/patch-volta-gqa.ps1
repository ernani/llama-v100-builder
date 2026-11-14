param(
    [string]$LlamaDir = "llama.cpp"
)

$ErrorActionPreference = "Stop"

# 1. Locate target files
$fattnPath = [System.IO.Path]::Combine($LlamaDir, "ggml", "src", "ggml-cuda", "fattn.cu")
$commonPath = [System.IO.Path]::Combine($LlamaDir, "ggml", "src", "ggml-cuda", "fattn-common.cuh")

if (-not (Test-Path $fattnPath)) {
    throw "Target file not found: $fattnPath"
}
if (-not (Test-Path $commonPath)) {
    throw "Target file not found: $commonPath"
}

# ==============================================================================
# Patch 1: fattn.cu
# ==============================================================================
Write-Host "Reading $fattnPath..."
$contentFattn = Get-Content $fattnPath -Raw

# 1.1 Route odd GQA ratio to BEST_FATTN_KERNEL_TILE on Volta in ggml_cuda_get_best_fattn_kernel
$target1 = "if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {"
$replacement1 = @"
if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (gqa_ratio % 2 != 0) {
            return BEST_FATTN_KERNEL_TILE; // Fallback to TILE kernel on Volta for odd GQA (e.g. 7:1) to prevent unaligned WMMA deadlock
        }
"@

if ($contentFattn.Contains($target1)) {
    $contentFattn = $contentFattn.Replace($target1, $replacement1)
    Write-Host "[OK] Patched kernel selection for Volta odd GQA"
} else {
    throw "Could not find target1 in $fattnPath"
}

# 1.2 Safety net in switch_ncols2 on Volta: call ext_tile instead of ncols1<1>
$target2 = "ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);"
$replacement2 = "ggml_cuda_flash_attn_ext_tile(ctx, dst); // Volta odd-GQA fallback"

if ($contentFattn.Contains($target2)) {
    $contentFattn = $contentFattn.Replace($target2, $replacement2)
    Write-Host "[OK] Patched switch_ncols2 fallback for Volta odd GQA"
} else {
    throw "Could not find target2 in $fattnPath"
}

# 1.3 Quantized KV cache: dynamic pool allocation prevents continuous-batching dst overflow
$target3 = "return f16_extra.end - (uintptr_t) dst->data;"
$replacement3 = "return ggml_nbytes(dst); // Dynamic KV pool allocation prevents continuous-batching dst overflow"

if ($contentFattn.Contains($target3)) {
    $contentFattn = $contentFattn.Replace($target3, $replacement3)
    Write-Host "[OK] Patched get_alloc_size in fattn.cu (quantized KV cache fix)"
} else {
    throw "Could not find target3 in $fattnPath"
}

[System.IO.File]::WriteAllText($fattnPath, $contentFattn, [System.Text.Encoding]::UTF8)
Write-Host "[SUCCESS] fattn.cu patched successfully!"

# ==============================================================================
# Patch 2: fattn-common.cuh
# ==============================================================================
Write-Host "Reading $commonPath..."
$contentCommon = Get-Content $commonPath -Raw

# 2.1 Add dynamic pool allocators for K and V
$target4 = "ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);"
$replacement4 = @"
ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    ggml_cuda_pool_alloc<half>   K_f16_alloc(pool);
    ggml_cuda_pool_alloc<half>   V_f16_alloc(pool);
"@

if ($contentCommon.Contains($target4)) {
    $contentCommon = $contentCommon.Replace($target4, $replacement4)
    Write-Host "[OK] Added K/V pool allocators in fattn-common.cuh"
} else {
    throw "Could not find target4 in $commonPath"
}

# 2.2 Allocate K_f16 from pool dynamically
$target5 = "GGML_ASSERT(f16_extra.K != 0);"
$replacement5 = "// GGML_ASSERT(f16_extra.K != 0); // Replaced by dynamic pool allocation"

if ($contentCommon.Contains($target5)) {
    $contentCommon = $contentCommon.Replace($target5, $replacement5)
    Write-Host "[OK] Patched K assertion in fattn-common.cuh"
} else {
    throw "Could not find target5 in $commonPath"
}

$target6 = "half * K_f16 = (half *) f16_extra.K;"
$replacement6 = "half * K_f16 = K_f16_alloc.alloc(ggml_nelements(K));"

if ($contentCommon.Contains($target6)) {
    $contentCommon = $contentCommon.Replace($target6, $replacement6)
    Write-Host "[OK] Patched K pool allocation in fattn-common.cuh"
} else {
    throw "Could not find target6 in $commonPath"
}

# 2.3 Allocate V_f16 from pool dynamically
$target7 = "GGML_ASSERT(f16_extra.V != 0);"
$replacement7 = "// GGML_ASSERT(f16_extra.V != 0); // Replaced by dynamic pool allocation"

if ($contentCommon.Contains($target7)) {
    $contentCommon = $contentCommon.Replace($target7, $replacement7)
    Write-Host "[OK] Patched V assertion in fattn-common.cuh"
} else {
    throw "Could not find target7 in $commonPath"
}

$target8 = "half * V_f16 = (half *) f16_extra.V;"
$replacement8 = "half * V_f16 = V_f16_alloc.alloc(ggml_nelements(V));"

if ($contentCommon.Contains($target8)) {
    $contentCommon = $contentCommon.Replace($target8, $replacement8)
    Write-Host "[OK] Patched V pool allocation in fattn-common.cuh"
} else {
    throw "Could not find target8 in $commonPath"
}

[System.IO.File]::WriteAllText($commonPath, $contentCommon, [System.Text.Encoding]::UTF8)
Write-Host "[SUCCESS] fattn-common.cuh patched successfully!"
