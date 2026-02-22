#include "renderResources.cuh"

__device__ cudaTextureObject_t _HGLut;
__device__ cudaTextureObject_t _DensityVolume;
__device__ cudaTextureObject_t _HDRI;

__device__ cudaTextureObject_t _Mips[9];
__device__ cudaTextureObject_t _Var_Mips[9];
__device__ cudaTextureObject_t _TR_Mips[8];

__device__ float MipDensityDynamic(int mip, float3 pos) {
    if (mip < 0) mip = 0;
    if (mip > 8) mip = 8;
    float3 uv = pos + 0.5f;
    return tex3D<float>(_Mips[mip], uv.z, uv.y, uv.x);
}

__device__ float MipTrDynamic(int mip, float3 pos) {
    if (mip < 0) mip = 0;
    if (mip > 7) mip = 7;
    float3 uv = pos + 0.5f;
    return tex3D<float>(_TR_Mips[mip], uv.z, uv.y, uv.x);
}

__device__ float MipVarianceDynamic(int mip, float3 pos) {
    if (mip < 0) mip = 0;
    if (mip > 8) mip = 8;
    float3 uv = pos + 0.5f;
    return tex3D<float>(_Var_Mips[mip], uv.z, uv.y, uv.x);
}

__device__ float3 ShadowTerm_TRTex(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float g, int mip)
{
    if (ori.x < -0.5f || ori.y < -0.5f || ori.z < -0.5f || ori.x > 0.5f || ori.y > 0.5f || ori.z > 0.5f)
    {
        float offset = RayBoxOffset(ori, lightDir);
        if (offset >= 0)
        {
            return lightColor * MipTr(mip, ori + lightDir * offset);
        }
        else
        {
            return lightColor * float3{ 1.0f,1.0f,1.0f };
        }
    }
    else
    {
        return lightColor * MipTr(mip, ori);
    }
}