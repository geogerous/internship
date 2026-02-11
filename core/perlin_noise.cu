#include "perlin_noise.cuh"
// ...existing code...

__host__ __device__ float perlin_noise(float x, float y, float z) {
    // ...existing noise 实现保持不变...
}

// 如果有其它重载，同样改为 __host__ __device__：
__host__ __device__ float perlin_noise(float2 p) { ... }
__host__ __device__ float perlin_noise(float3 p) { ... }

// ...existing code...
