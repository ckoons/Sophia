#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="sophia",
    version="0.1.0",
    description="Machine learning components for the Tekton ecosystem",
    author="Tekton Team",
    author_email="noreply@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
