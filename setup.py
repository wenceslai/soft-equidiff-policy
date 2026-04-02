from setuptools import setup, find_packages

setup(
    name="soft-equidiff-policy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "escnn>=0.1.9",
        "einops>=0.6.0",
        "diffusers>=0.18.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.9",
)
