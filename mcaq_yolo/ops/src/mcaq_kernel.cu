// mcaq_yolo/ops/src/mcaq_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS 1024

// 유동적인 비트(3~8 bits)를 받아 픽셀별로 양자화 수행
__global__ void SpatialAdaptiveQuantizationKernel(
    const float* __restrict__ input,        // (N, C, H, W)
    const float* __restrict__ bit_map,      // (N, 1, H_tiles, W_tiles) or (N, 1, H, W) if upsampled
    const float* __restrict__ min_vals,     // (N, C, 1, 1) or (1, C, 1, 1)
    const float* __restrict__ max_vals,     // (N, C, 1, 1) or (1, C, 1, 1)
    float* __restrict__ output,
    int batch, int channels, int height, int width,
    int tile_h, int tile_w
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
    // bit_map은 보통 다운샘플링된 타일 크기이므로 인덱스 변환 필요
    int tile_idx_h = h / tile_h;
    int tile_idx_w = w / tile_w;
    
    // bit_map shape: (N, H_tiles, W_tiles) 로 가정 시 오프셋 계산
    // gridDim_h = height / tile_h
    int grid_w = width / tile_w;
    int map_idx = n * ( (height/tile_h) * grid_w ) + tile_idx_h * grid_w + tile_idx_w;
    
    float bits_float = bit_map[map_idx];
    int bits = (int)(bits_float + 0.5f); // Round
    
    // 비트 범위 클램핑 (3 ~ 8)
    if (bits < 2) bits = 2;
    if (bits > 8) bits = 8;

    // 2. Quantization Parameters 계산 (Scale, ZeroPoint)
    // qmin, qmax는 비트에 따라 결정됨 (Symmetric or Asymmetric)
    // 여기서는 Signed Symmetric 가정: -(2^(b-1)) ~ 2^(b-1) - 1
    float qmin = -(powf(2.0f, bits - 1));
    float qmax = powf(2.0f, bits - 1) - 1.0f;
    
    float x_min = min_vals[c]; // Per-channel 가정
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
    
    output[idx] = (q_val - zero_point) * scale;
}

// C++ Wrapper에서 호출할 함수
void launch_spatial_quantization(
    const float* input, const float* bit_map, 
    const float* min_vals, const float* max_vals, 
    float* output,
    int N, int C, int H, int W,
    int tile_h, int tile_w, 
    cudaStream_t stream
) {
    int total_elements = N * C * H * W;
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    
    SpatialAdaptiveQuantizationKernel<<<blocks, threads, 0, stream>>>(
        input, bit_map, min_vals, max_vals, output,
        N, C, H, W, tile_h, tile_w
    );
}