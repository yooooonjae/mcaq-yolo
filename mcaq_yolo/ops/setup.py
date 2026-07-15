"""
Build script for the custom MCAQ CUDA kernel (the paper's Listing 1).

Requires an NVIDIA GPU + CUDA toolkit (nvcc) + torch with CUDA.
Cannot be built on Apple Silicon / CPU-only machines — the Python code
falls back to SpatialAdaptiveQuantization._forward_pytorch automatically
(same math, slower).

Build & install into the active environment:

    cd mcaq_yolo/ops
    pip install -e .          # or: python setup.py build_ext --inplace

Verify:

    python -c "import mcaq_cuda_ops; print('CUDA kernel OK')"

After this, core/quantization.py prints no fallback warning and routes
inference through mcaq_cuda_ops.spatial_quantize.
"""

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "torch with CUDA support is required to build mcaq_cuda_ops "
        f"({e})"
    )

setup(
    name="mcaq_cuda_ops",
    version="0.1.0",
    description="MCAQ-YOLO tile-wise spatial adaptive quantization CUDA kernel",
    ext_modules=[
        CUDAExtension(
            name="mcaq_cuda_ops",
            sources=[
                "src/mcaq_ops.cpp",
                "src/mcaq_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
