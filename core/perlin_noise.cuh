#ifndef PERLIN_NOISE_CUH
#define PERLIN_NOISE_CUH

__host__ __device__ float perlin_noise(float x, float y, float z) {
    // Simple noise function - replace with actual implementation
    return sinf(x * 0.1f) * cosf(y * 0.1f) * sinf(z * 0.1f);
}

#endif // PERLIN_NOISE_CUH
