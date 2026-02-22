#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_functions.h>
#include <curand_kernel.h>
#include <cuda_texture_types.h>

#include <vector_types.h>
#include <algorithm>

#include "vector.cuh"

// Modern CUDA Texture Objects (Bindless)
extern __device__ cudaTextureObject_t _HGLut;
extern __device__ cudaTextureObject_t _DensityVolume;
extern __device__ cudaTextureObject_t _HDRI;

// Mipmap arrays
extern __device__ cudaTextureObject_t _Mips[9];
extern __device__ cudaTextureObject_t _Var_Mips[9];
extern __device__ cudaTextureObject_t _TR_Mips[8];

#define MipDensityStatic(mip, pos) tex3D<float>(_Mips[mip], (pos).z + 0.5f, (pos).y + 0.5f, (pos).x + 0.5f)
#define MipTrStatic(mip, pos) tex3D<float>(_TR_Mips[mip], (pos).z + 0.5f, (pos).y + 0.5f, (pos).x + 0.5f)

__device__ float MipDensityDynamic(int mip, float3 pos);
__device__ float MipTrDynamic(int mip, float3 pos);
__device__ float MipVarianceDynamic(int mip, float3 pos);

#define MipDensity MipDensityDynamic
#define MipTr MipTrDynamic

__device__ float3 ShadowTerm_TRTex(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float g, int mip);