# Compile cuda code

```shell
nvcc --shared -Xcompiler -fPIC -o libvector_add.so vector_add.cu
```
- [NVIDIA CUDA Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)


```shell
uv run main.py
```

