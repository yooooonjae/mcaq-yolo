import os
from setuptools import setup, find_packages

# README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]


def _cuda_extension():
    """
    REVIEW FIX (packaging): the previous top-level
    `from torch.utils.cpp_extension import ...` made `pip install .` fail on
    any machine without torch preinstalled, and the unconditional
    CUDAExtension failed again on CPU-only machines (no nvcc) — contradicting
    the README's own CPU instructions. Build the kernel only when torch is
    importable AND a CUDA toolchain is discoverable (CUDA_HOME) or FORCE_CUDA=1
    is set; otherwise install pure-Python and rely on the documented
    SpatialAdaptiveQuantization._forward_pytorch fallback.
    Set MCAQ_SKIP_CUDA=1 to skip unconditionally.
    """
    if os.environ.get("MCAQ_SKIP_CUDA", "0") == "1":
        print("[mcaq-yolo setup] MCAQ_SKIP_CUDA=1 — skipping the CUDA kernel.")
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError:
        print("[mcaq-yolo setup] torch not importable at build time — installing "
              "without the CUDA kernel (pure-PyTorch fallback will be used).")
        return [], {}
    if CUDA_HOME is None and not os.environ.get("FORCE_CUDA"):
        print("[mcaq-yolo setup] CUDA toolkit (nvcc) not found — installing "
              "without the CUDA kernel (pure-PyTorch fallback will be used). "
              "Set FORCE_CUDA=1 to override.")
        return [], {}
    ops_src_dir = os.path.join("mcaq_yolo", "ops", "src")
    ext = CUDAExtension(
        name="mcaq_cuda_ops",
        sources=[
            os.path.join(ops_src_dir, "mcaq_ops.cpp"),
            os.path.join(ops_src_dir, "mcaq_kernel.cu"),
        ],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = _cuda_extension()

setup(
    name="mcaq-yolo",
    version="0.2.1",
    author="Yoonjae Seo",
    author_email="ssyyjj0517@naver.com",
    description="Morphological Complexity-Aware Quantization for YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yooooonjae/mcaq-yolo",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={"dev": ["pytest>=7.0"]},
    entry_points={
        "console_scripts": [
            "mcaq-yolo-train=mcaq_yolo.train:main",
            "mcaq-yolo-infer=mcaq_yolo.inference:main",
        ],
    },
)
