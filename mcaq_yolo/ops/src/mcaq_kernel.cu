// mcaq_yolo/ops/src/mcaq_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS 1024

// Takes a variable bit-width (2~8 bits) and quantizes per pixel.
// Paper Listing 1 (tile-wise spatial quantization) + Listing 2 (fused soft
// mask): if a mask is given, the kernel also performs the Eq.(19) product
// X_q(p) = m(p) * Q_{bT(p)}(X(p)) (mask == nullptr means quantization only).
__global__ void SpatialAdaptiveQuantizationKernel(
    const float* __restrict__ input,        // (N, C, H, W)
    const float* __restrict__ bit_map,      // (N, H_tiles, W_tiles)
    const float* __restrict__ min_vals,     // (1, C, 1, 1) — per-channel (paper Sec IV-D)
    const float* __restrict__ max_vals,     // (1, C, 1, 1)
    const float* __restrict__ mask,         // (N, 1, H, W) or nullptr — Eq.(19) m(p)
    float* __restrict__ output,
    int batch, int channels, int height, int width,
    int tile_h, int tile_w,
    int n_tiles_h, int n_tiles_w            // actual bit_map grid size (N, Ht, Wt)
) {
    // REVIEW FIX (int32 overflow + launch-grid coupling): the previous
    // `int idx` / `int total_elements` overflow for N*C*H*W >= 2^31 and
    // required the launch grid to cover every element. 64-bit grid-stride
    // loop is correct for any tensor size and any grid.
    // NOTE: not compiled in the review environment (no nvcc available) —
    // semantics are unchanged w.r.t. the numerically verified PyTorch path.
    const long long total_elements =
        (long long)batch * channels * height * width;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += (long long)blockDim.x * gridDim.x) {

    // coordinate computation
    int w = (int)(idx % width);
    int h = (int)((idx / width) % height);
    int c = (int)((idx / ((long long)width * height)) % channels);
    int n = (int)(idx / ((long long)width * height * channels));

    // 1. Look up the bit-width for this pixel.
    // bit_map is a downsampled tile grid, so an index conversion is needed.
    // The grid size is NOT recomputed as height/tile_h; the actual bit_map
    // size (n_tiles_h/w) is used directly — since tile_h = floor(H / Ht), if H
    // is not divisible by Ht the recomputed grid is larger than the real one
    // and goes OOB (e.g. H=10, Ht=4 -> tile_h=2 -> 10/2=5 > 4; from external
    // review). Remainder pixels are assigned to the last tile.
    int tile_idx_h = h / tile_h;
    int tile_idx_w = w / tile_w;
    if (tile_idx_h >= n_tiles_h) tile_idx_h = n_tiles_h - 1;
    if (tile_idx_w >= n_tiles_w) tile_idx_w = n_tiles_w - 1;

    int map_idx = n * (n_tiles_h * n_tiles_w) + tile_idx_h * n_tiles_w + tile_idx_w;

    float bits_float = bit_map[map_idx];
    int bits = (int)(bits_float + 0.5f); // Round

    // clamp bit range (2 ~ 8)
    if (bits < 2) bits = 2;
    if (bits > 8) bits = 8;

    // 2. Compute quantization parameters (Scale, ZeroPoint)
    // Signed symmetric range: -(2^(b-1)) ~ 2^(b-1) - 1
    // (same formula as QuantizationParameters in core/quantization.py)
    float qmin = -(powf(2.0f, bits - 1));
    float qmax = powf(2.0f, bits - 1) - 1.0f;

    float x_min = min_vals[c]; // Per-channel (the caller guarantees C entries)
    float x_max = max_vals[c];

    // compute scale
    float range = x_max - x_min;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / (qmax - qmin);
    float zero_point = qmin - x_min / scale;

    // Clamp Zero Point
    if (zero_point < qmin) zero_point = qmin;
    if (zero_point > qmax) zero_point = qmax;

    // 3. Quantize & Dequantize (Simulated)
    float val = input[idx];
    float q_val = roundf(val / scale + zero_point);

    if (q_val < qmin) q_val = qmin;
    if (q_val > qmax) q_val = qmax;

    float deq = (q_val - zero_point) * scale;

    // 4. Paper Listing 2 / Eq.(19): fused learned soft-mask multiply
    // (mask index in 64-bit as well — (n*H+h)*W+w overflows int32 for the
    // same tensor sizes the grid-stride fix targets)
    if (mask != nullptr) {
        deq *= mask[((long long)n * height + h) * width + w];
    }

    output[idx] = deq;
    }  // grid-stride loop
}

// Function called from the C++ wrapper
void launch_spatial_quantization(
    const float* input, const float* bit_map,
    const float* min_vals, const float* max_vals,
    const float* mask,
    float* output,
    int N, int C, int H, int W,
    int tile_h, int tile_w,
    int n_tiles_h, int n_tiles_w,
    cudaStream_t stream
) {
    const long long total_elements = (long long)N * C * H * W;
    const int threads = 256;
    long long want = (total_elements + threads - 1) / threads;
    // Grid-stride kernel: cap the grid, the loop covers any remainder.
    int blocks = (int)(want < (long long)(1 << 20) ? want : (long long)(1 << 20));
    if (blocks < 1) blocks = 1;

    SpatialAdaptiveQuantizationKernel<<<blocks, threads, 0, stream>>>(
        input, bit_map, min_vals, max_vals, mask, output,
        N, C, H, W, tile_h, tile_w, n_tiles_h, n_tiles_w
    );
}
