#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void launch_spatial_quantization(
    const float* input, const float* bit_map, 
    const float* min_vals, const float* max_vals, 
    float* output,
    int N, int C, int H, int W,
    int tile_h, int tile_w, 
    cudaStream_t stream
);

// PyTorch Interface
torch::Tensor spatial_quantize_cuda(
    torch::Tensor input,
    torch::Tensor bit_map,
    torch::Tensor min_vals,
    torch::Tensor max_vals,
    int tile_h, int tile_w
) {
    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    launch_spatial_quantization(
        input.data_ptr<float>(),
        bit_map.data_ptr<float>(),
        min_vals.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        tile_h, tile_w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_quantize", &spatial_quantize_cuda, "MCAQ Spatial Quantization (CUDA)");
}