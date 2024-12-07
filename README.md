# cuda-from-python

```shell
nvcc --version
# 12.1
uv --version
# 0.4.0
```

compile the shared library

```shell
nvcc --shared -Xcompiler -fPIC -o libvector_add.so vector_add.cu
```
- [NVIDIA CUDA Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)

Run python script

```shell
uv run main.py
```

