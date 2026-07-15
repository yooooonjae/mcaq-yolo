// mcaq_yolo/ops/src/mcaq_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS 1024

// 유동적인 비트(2~8 bits)를 받아 픽셀별로 양자화 수행.
// Paper Listing 1 (tile-wise spatial quantization) + Listing 2 (fused soft
// mask): mask가 주어지면 Eq.(19) X_q(p) = m(p) * Q_{bT(p)}(X(p))의 곱을
// 커널 안에서 함께 수행한다 (mask == nullptr이면 양자화만).
__global__ void SpatialAdaptiveQuantizationKernel(
    const float* __restrict__ input,        // (N, C, H, W)
    const float* __restrict__ bit_map,      // (N, H_tiles, W_tiles)
    const float* __restrict__ min_vals,     // (1, C, 1, 1) — per-channel (paper Sec IV-D)
    const float* __restrict__ max_vals,     // (1, C, 1, 1)
    const float* __restrict__ mask,         // (N, 1, H, W) or nullptr — Eq.(19) m(p)
    float* __restrict__ output,
    int batch, int channels, int height, int width,
    int tile_h, int tile_w,
    int n_tiles_h, int n_tiles_w            // 실제 bit_map 그리드 크기 (N, Ht, Wt)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;

    if (idx >= total_elements) return;

    // 좌표 계산
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    // 1. 해당 픽셀의 비트 심도(Bit-width) 조회
    // bit_map은 다운샘플링된 타일 그리드이므로 인덱스 변환 필요.
    // 그리드 크기는 height/tile_h로 재계산하지 않고 실제 bit_map 크기
    // (n_tiles_h/w)를 그대로 쓴다 — tile_h = floor(H / Ht)라서 H가 Ht로
    // 나누어떨어지지 않으면 재계산 그리드가 실제보다 커져 OOB가 난다
    // (예: H=10, Ht=4 -> tile_h=2 -> 10/2=5 > 4; codex 리뷰 반영).
    // 잔여 픽셀은 마지막 타일에 귀속.
    int tile_idx_h = h / tile_h;
    int tile_idx_w = w / tile_w;
    if (tile_idx_h >= n_tiles_h) tile_idx_h = n_tiles_h - 1;
    if (tile_idx_w >= n_tiles_w) tile_idx_w = n_tiles_w - 1;

    int map_idx = n * (n_tiles_h * n_tiles_w) + tile_idx_h * n_tiles_w + tile_idx_w;

    float bits_float = bit_map[map_idx];
    int bits = (int)(bits_float + 0.5f); // Round

    // 비트 범위 클램핑 (2 ~ 8)
    if (bits < 2) bits = 2;
    if (bits > 8) bits = 8;

    // 2. Quantization Parameters 계산 (Scale, ZeroPoint)
    // Signed Symmetric 범위: -(2^(b-1)) ~ 2^(b-1) - 1
    // (core/quantization.py의 QuantizationParameters와 동일한 식)
    float qmin = -(powf(2.0f, bits - 1));
    float qmax = powf(2.0f, bits - 1) - 1.0f;

    float x_min = min_vals[c]; // Per-channel (호출측이 C개 엔트리를 보장)
    float x_max = max_vals[c];

    // Scale 계산
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
    if (mask != nullptr) {
        deq *= mask[(n * height + h) * width + w];
    }

    output[idx] = deq;
}

// C++ Wrapper에서 호출할 함수
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
    int total_elements = N * C * H * W;
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;

    SpatialAdaptiveQuantizationKernel<<<blocks, threads, 0, stream>>>(
        input, bit_map, min_vals, max_vals, mask, output,
        N, C, H, W, tile_h, tile_w, n_tiles_h, n_tiles_w
    );
}
