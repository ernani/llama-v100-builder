param(
    [string]$LlamaDir = "llama.cpp"
)

$ErrorActionPreference = "Stop"

$fattnPath = Join-Path $LlamaDir "ggml\src\ggml-cuda\fattn.cu"
if (-not (Test-Path $fattnPath)) {
    throw "Target file not found: $fattnPath"
}

Write-Host "Reading $fattnPath..."
$content = Get-Content $fattnPath -Raw

# 1. Route odd GQA ratio to BEST_FATTN_KERNEL_TILE on Volta in ggml_cuda_get_best_fattn_kernel
$target1 = "if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {"
$replacement1 = @"
if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (gqa_ratio % 2 != 0) {
            return BEST_FATTN_KERNEL_TILE; // Fallback to TILE kernel on Volta for odd GQA (e.g. 7:1) to prevent unaligned WMMA deadlock
        }
"@

if ($content.Contains($target1)) {
    $content = $content.Replace($target1, $replacement1)
    Write-Host "[OK] Patched kernel selection for Volta odd GQA"
} else {
    Write-Warning "Could not find target1 in $fattnPath"
}

# 2. Safety net in switch_ncols2 on Volta: call ext_tile instead of ncols1<1>
$target2 = "ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);"
$replacement2 = "ggml_cuda_flash_attn_ext_tile(ctx, dst); // Volta odd-GQA fallback"

if ($content.Contains($target2)) {
    $content = $content.Replace($target2, $replacement2)
    Write-Host "[OK] Patched switch_ncols2 fallback for Volta odd GQA"
} else {
    Write-Warning "Could not find target2 in $fattnPath"
}

[System.IO.File]::WriteAllText($fattnPath, $content, [System.Text.Encoding]::UTF8)
Write-Host "[SUCCESS] fattn.cu patched successfully!"
