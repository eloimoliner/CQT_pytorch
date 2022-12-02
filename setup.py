from setuptools import find_packages, setup

setup(
    name="cqt-nsgt-pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.6",
    license="MIT",
    description="Pytorch implementation of an invertible and differentiable Constant-Q Transform based Non-stationary Gabor Transform (NSGT) for audio processing.",
    long_description_content_type="text/markdown",
    author="Eloi Moliner",
    author_email="eloi.moliner@aalto.fi",
    url="https://github.com/eloimoliner/CQT_pytorch",
    keywords=["audio processing", "constant-q transform", "deep learning", "pytorch", "nsgt"],
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.19.5",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)