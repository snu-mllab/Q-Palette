# Q-Palette Kernels

This repository contains optimized CUDA kernels for fused dequantization and matrix multiplication tailored for Q-Palette quantizers. 
Our kernel implementation is built on top of [QTIP kernels](https://github.com/Cornell-RelaxML/qtip/tree/main/qtip-kernels) and [Any-precision kernels](https://github.com/SNU-ARC/any-precision-llm/tree/main/any_precision/modules/kernels). 

## Setup Environment

```bash
# Create and activate conda environment
conda create -n qpal python=3.11 -y
conda activate qpal
export CONDA_HOME=<YOUR_CONDA_HOME>
export CUDA_HOME=<YOUR_CUDA12.4_HOME>

# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_HOME}/envs/qpal/lib/python3.11/site-packages/torch/lib

# Install kernel for Fast Hadamard Transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
cd ..

# Compile CUDA kernels (about 30 mins)
source install_kernels.sh 
```

Note that this codebase is optimized for NVIDIA RTX4090 GPU. 