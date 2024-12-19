# FlashAttention Demo

A minimal re-implementation of FlashAttention using CUDA and PyTorch. Explore the efficiency of IO-aware attention optimizations with our streamlined implementation. For reference, check out the official [FlashAttention](https://github.com/Dao-AILab/flash-attention) implementation.

## Features

- **Forward Pass Implementation**: Complete forward pass written in `flash.cu`.
- **Consistent Naming Conventions**: Variable names align with those used in the original [FlashAttention paper](https://arxiv.org/abs/2205.14135).
- **Optimized Memory Management**: Utilizes shared memory (SRAM) to minimize high-bandwidth memory (HBM) accesses.

## Prerequisites

- **PyTorch**: Ensure PyTorch is installed with CUDA support.
- **Ninja**: Required for building the C++ extensions.

## Installation

1. **Clone the Repository**:
    ```
    git clone https://github.com/yourusername/flash-attention-demo.git
    cd flash-attention-demo
    ```

2. **Install Dependencies**:
    Ensure PyTorch with CUDA is installed. Install Ninja if it's not already available:
    ```
    pip install ninja
    ```

3. **Build the Extension**:
    The extension will be automatically built when you run the benchmarking script.

## Usage

### Benchmarking

Compare the wall-clock time between manual attention and the minimal FlashAttention implementation by running:

```
python bench.py
```

## Sample Output
```
=== Profiling Manual Attention ===
...
Self CPU time total: 52.389ms
Self CUDA time total: 52.545ms

=== Profiling Minimal Flash Attention === 
...  
Self CPU time total: 11.452ms
Self CUDA time total: 3.908ms
```
