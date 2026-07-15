#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>  // at::cuda::getCurrentCUDAStream
#include <c10/util/Optional.h>
#include <vector>

// CUDA forward declaration
void launch_spatial_quantization(
    const float* input, const float* bit_map,
    const float* min_vals, const float* max_vals,
    const float* mask,
    float* output,
    int N, int C, int H, int W,
    int tile_h, int tile_w,
    int n_tiles_h, int n_tiles_w,
    cudaStream_t stream
);

// PyTorch Interface
// mask: the learned soft mask m(p) from Eq.(19), (N, 1, H, W) — if given, the
// kernel fuses quantization with the multiply (paper Listing 2). If omitted,
// quantization only.
torch::Tensor spatial_quantize_cuda(
    torch::Tensor input,
    torch::Tensor bit_map,
    torch::Tensor min_vals,
    torch::Tensor max_vals,
    int tile_h, int tile_w,
    c10::optional<torch::Tensor> mask
) {
    auto output = torch::empty_like(input);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(min_vals.numel() == C && max_vals.numel() == C,
                "min_vals/max_vals must have one entry per channel (C=", C,
                "), got ", min_vals.numel(), " — expand per-tensor stats "
                "before calling (see SpatialAdaptiveQuantization._forward_cuda)");

    const float* mask_ptr = nullptr;
    if (mask.has_value() && mask->defined()) {
        TORCH_CHECK(mask->numel() == static_cast<int64_t>(N) * H * W,
                    "mask must be (N, 1, H, W)");
        mask_ptr = mask->data_ptr<float>();
    }

    // actual bit_map grid size — if the kernel recomputes it as height/tile_h,
    // it goes OOB in the non-divisible case (from external review)
    int n_tiles_h = bit_map.size(1);
    int n_tiles_w = bit_map.size(2);

    launch_spatial_quantization(
        input.data_ptr<float>(),
        bit_map.data_ptr<float>(),
        min_vals.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        mask_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        tile_h, tile_w,
        n_tiles_h, n_tiles_w,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_quantize", &spatial_quantize_cuda,
          "MCAQ Spatial Quantization (CUDA) — Eq.19 fused soft mask (Listing 2)",
          py::arg("input"), py::arg("bit_map"),
          py::arg("min_vals"), py::arg("max_vals"),
          py::arg("tile_h"), py::arg("tile_w"),
          py::arg("mask") = py::none());
}
