import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# README.md 읽기
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements.txt 읽기
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# CUDA 소스 파일 경로 설정 (mcaq_yolo/ops/src 디렉토리)
ops_src_dir = os.path.join('mcaq_yolo', 'ops', 'src')

setup(
    name="mcaq-yolo",
    version="0.1.0",
    author="Yoonjae Seo",
    author_email="ssyyjj0517@naver.com",
    description="Morphological Complexity-Aware Quantization for YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yooooonjae/mcaq-yolo",
    packages=find_packages(),  # 패키지 자동 탐색
    ext_modules=[
        CUDAExtension(
            name='mcaq_cuda_ops',
            sources=[
                os.path.join(ops_src_dir, 'mcaq_ops.cpp'),
                os.path.join(ops_src_dir, 'mcaq_kernel.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mcaq-yolo-train=mcaq_yolo.train:main",
            "mcaq-yolo-infer=mcaq_yolo.inference:main",
        ],
    },
)