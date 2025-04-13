# CUDA Matrix Multiplication

This project implements **Matrix Multiplication** using **CUDA** for parallel processing. It demonstrates how to leverage GPU acceleration to perform matrix operations, which is significantly faster than CPU-based implementations, especially for large matrices.

### Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
- [How to Use](#how-to-use)
- [Example](#example)
- [License](#license)

---

## Project Description

Matrix multiplication is a fundamental operation in many scientific computing and machine learning applications. This project uses CUDA to parallelize matrix multiplication, utilizing the massive parallelism of GPUs to speed up the computation. The matrix multiplication is done between two \(n \times n\) matrices.

### Key Concepts:
- **CUDA Programming**: A parallel computing platform and application programming interface model created by NVIDIA for general computing on GPUs.
- **Matrix Multiplication**: The process of multiplying two matrices. In this case, the product of two square matrices is calculated.

---

## Features

- **Parallelized Matrix Multiplication** using CUDA.
- **Efficient memory handling** with device memory allocation.
- **Scalable**: Can handle large matrices by utilizing the GPU.
- **Uses blocks and threads**: Efficient GPU computation with thread synchronization and memory management.

---

## Prerequisites

Before running the project, ensure you have the following installed:

1. **CUDA Toolkit**: The CUDA toolkit must be installed on your system to compile and run CUDA code. You can download it from [NVIDIA's CUDA toolkit page](https://developer.nvidia.com/cuda-toolkit).
2. **NVIDIA Driver**: An NVIDIA GPU with the correct driver version for CUDA support.
3. **Visual Studio Code**: A lightweight code editor (can be installed via [VS Code website](https://code.visualstudio.com/)).
4. **C++ Compiler with CUDA support**: Ensure you have a C++ compiler that supports CUDA (`nvcc`).

---

## Installation Instructions

### Step 1: Install CUDA Toolkit and Driver

1. Update your Ubuntu package list and install the necessary dependencies:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-460 nvidia-cuda-toolkit
