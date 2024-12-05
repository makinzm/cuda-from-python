#include <cuda_runtime.h>

extern "C" void vector_add(float *a, float *b, float *c, int n);

__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add(float *a, float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // デバイスメモリの割り当て
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // ホストからデバイスへデータを転送
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // カーネルの起動
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // デバイスからホストへ結果を転送
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // デバイスメモリの解放
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

