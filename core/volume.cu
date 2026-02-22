#include "volume.hpp"

#include "hdr_loader.h"

#include "render.cuh"

#include "platform.h"
#include <thread>

#include <iostream>
#include <fstream>
#include <sstream>
#include "omp.hpp"
using namespace std;

#define CheckError { auto error = cudaGetLastError(); if (error != 0) cout << cudaGetErrorString(error); }

// [New] Kernel to fill HG LUT
__global__ void Fill_Hg(cudaSurfaceObject_t surf, float g) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= LUT_SIZE || y >= LUT_SIZE) return;

    float cos_theta = (float)x / (float)LUT_SIZE * 2.0f - 1.0f;
    // float angle = (float)y / (float)LUT_SIZE; // Unused in standard HG?

    // Standard HG Phase Function
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    float res = (1.0f - g2) / (4.0f * 3.14159265f * powf(denom, 1.5f));

    surf2Dwrite(res, surf, x * sizeof(float), y);
}

// [New] Kernel to fill TR Mipmaps
__global__ void Fill_TR(cudaSurfaceObject_t tr_surf, cudaTextureObject_t density_tex, int res, float alpha, float3 lightDir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= res || y >= res || z >= res) return;

    // UV coordinates [0, 1]
    float3 uv;
    uv.x = (x + 0.5f) / (float)res;
    uv.y = (y + 0.5f) / (float)res;
    uv.z = (z + 0.5f) / (float)res;
    
    // Raymarch parameters
    const int MaxStep = 128;
    float3 ori;
    ori.x = uv.x - 0.5f;
    ori.y = uv.y - 0.5f;
    ori.z = uv.z - 0.5f;
    
    float dis = RayBoxDistance(ori, lightDir);
    float MaxStepInv = dis / MaxStep;
    float3 Lpos = ori;
    float shadowdist = 0;
    
    for (int i = 0; i < MaxStep; i++) {
        Lpos.x += lightDir.x * MaxStepInv;
        Lpos.y += lightDir.y * MaxStepInv;
        Lpos.z += lightDir.z * MaxStepInv;
        
        // Sample density from texture (normalized coords)
        float3 tex_uv;
        tex_uv.x = Lpos.x + 0.5f;
        tex_uv.y = Lpos.y + 0.5f;
        tex_uv.z = Lpos.z + 0.5f;
        float lsample = tex3D<float>(density_tex, tex_uv.z, tex_uv.y, tex_uv.x);
        shadowdist += lsample;
    }
    
    float shadowterm = expf(-shadowdist * alpha * MaxStepInv); // Phase is 1.0 here
    // Legacy had TR_MUL? 
    // #define TR_MUL 1.0f
    
    surf3Dwrite(shadowterm, tr_surf, x * sizeof(float), y, z);
}

template<int type>
__global__ void CalculateRadianceMulti(volatile int* record, float3* result, float3* ori, float3* dir, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float3 res;
    if (type == 0)
        res = make_float3(CalculateRadiance(ori[idx], dir[idx], lightDir, lightColor, alpha, multiScatter, g, sampleNum));
    else if (type == 1)
        res = make_float3(NNPredict<Type::RPNN>(ori[idx], dir[idx], lightDir, lightColor, alpha, g));
    else
        res = make_float3(NNPredict<Type::MRPNN>(ori[idx], dir[idx], lightDir, lightColor, alpha, g));

    result[idx] = res;

    if (threadIdx.x == 0)
        atomicAdd((int*)record, 1);
}

__global__ void GetSampleMulti(volatile int* record, int task_num, float3* result, float* alpha, float3* ori, float3* dir, float3* lightDir, float* g, float* scatters, float3 lightColor = { 1, 1, 1 }, int multiScatter = 1, int sampleNum = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= task_num) return;

    float3 res = GetSample(ori[idx], dir[idx], lightDir[idx], lightColor, scatters[idx], alpha[idx], multiScatter, g[idx], sampleNum);

    result[idx] = res;

    if (threadIdx.x == 0)
        atomicAdd((int*)record, 1);
}

__device__ int dev_checkboard = 0;
__device__ int flip = 0;

__device__ float exposure = 1;

__device__ float3 lori;
__device__ float3 lup;
__device__ float3 lright;
template<bool predict, int type = Type::MRPNN>
__global__ void RenderCamera(float3* target, Histogram* histo_buffer, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    if (dev_checkboard) j = (blockIdx.y * blockDim.y + threadIdx.y) * 2 + ((i + flip) % 2);
    else j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * size.x + j;

    curandState seed;
    InitRand(&seed);

    float u = 1 - (j + Rand(&seed)) / size.x;
    float v = (i + Rand(&seed)) / size.y;

    float3 forward = normalize(-ori);
    float3 dir = forward + (right * (u * 2 - 1)) + (up * (v * 2 - 1));
    dir = normalize(dir);

    float4 res_dis;
    if (predict)
        res_dis = NNPredict<type>(ori, dir, lightDir, lightColor, alpha, g);
    else
        res_dis = CalculateRadiance(ori, dir, lightDir, lightColor, alpha, multiScatter, g, 1);
    float3 res = make_float3(res_dis);
    bool sky = res_dis.w < 0;
    float dis = max(0.001f, res_dis.w < 0 ? 10.0f : res_dis.w);

    res = max(float3{ 0 }, res);


    // show Lut
    {
        //float aaa = tex2D<float>(_HGLut, 1.2 * u - 0.1, 1.2 * v - 0.1);
        //res = { aaa,aaa,aaa };

        //if (abs(abs(1.2 * u - 0.1 - 0.5) - 0.5) < 0.001 || abs(abs(1.2 * v - 0.1 - 0.5) - 0.5) < 0.001) {
        //    res = { 1, 0, 1 };
        //}
    }

    if (i >= size.x || j >= size.y) return;


    int fNum = dev_checkboard ? frameNum / 2 : frameNum;
    if (!predict) {

        if (fNum == 0)
            histo_buffer[idx] = { 0 };

        float lerp_rate = 1.0f / (1 + (fNum));
        target[idx] = lerp(target[idx], res, lerp_rate);

        res = res / (res + 1);

        histo_buffer[idx].totalSampleNum += 1;

        int3 bin_idx = floor(min(res, float3{ 0.999f, 0.999f, 0.999f }) * HISTO_SIZE);

        histo_buffer[idx].bin[bin_idx.x] += 1;
        histo_buffer[idx].bin[bin_idx.y + HISTO_SIZE] += 1;
        histo_buffer[idx].bin[bin_idx.z + HISTO_SIZE * 2] += 1;

        float l = dot(res, { 1, 1, 1 });

        histo_buffer[idx].x = lerp(histo_buffer[idx].x, l, 1.0f / (1 + fNum));
        histo_buffer[idx].x2 = lerp(histo_buffer[idx].x2, l * l, 1.0f / (1 + fNum));
    }
    else {

        int lidx;
        {   // reprojection
            float3 motion_pos = ori + dir * dis;
            float3 ldir = motion_pos - lori;
            float3 lforward = normalize(lori);
            ldir = ldir / dot(ldir, lforward);
            ldir = ldir - lforward;
            float lu = dot(ldir, lright) * 0.5 + 0.5;
            float lv = dot(ldir, lup) * -0.5 + 0.5;
            int2 lxy = int2{ min(size.x - 1, max(0, int(lu * size.x))), min(size.y - 1, max(0, int(lv * size.y))) };
            lidx = lxy.y * size.x + lxy.x;
        }

        float lerp_rate = 1.0f / (1 + fNum);
        if (!sky && !dev_checkboard) lerp_rate = min(0.2f, lerp_rate);
        float3 his = float3{ histo_buffer[lidx].totalSampleNum,histo_buffer[lidx].x, histo_buffer[lidx].x2 };
        target[idx] = lerp(his, res, lerp_rate);
    }
}

template<bool predict, int type = Type::MRPNN>
__global__ void RenderCamera(float3* result, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (blockIdx.y * blockDim.y + threadIdx.y);

    int idx = i * size.x + j;

    curandState seed;
    InitRand(&seed);

    float u = 1 - (j + Rand(&seed)) / size.x;
    float v = 1 - (i + Rand(&seed)) / size.y;

    float3 forward = normalize(-ori);
    float3 dir = forward + (right * (u * 2 - 1)) + (up * (v * 2 - 1));
    dir = normalize(dir);

    float4 res_dis;
    if (predict)
        res_dis = NNPredict<type>(ori, dir, lightDir, lightColor, alpha, g);
    else
        res_dis = CalculateRadiance(ori, dir, lightDir, lightColor, alpha, multiScatter, g, 1);
    float3 res = make_float3(res_dis);
    float dis = max(0.001f, res_dis.w < 0 ? 10.0f : res_dis.w);

    res = max(float3{ 0 }, res);

    if (i >= size.x || j >= size.y) return;

    float lerp_rate = 1.0f / (1 + frameNum);
    result[idx] = lerp(result[idx], res, lerp_rate);
}

__device__ int UnRoll(int2 idx, int2 wh) {
    idx.y = min(max(0, idx.y), wh.x - 1);
    idx.x = min(max(0, idx.x), wh.y - 1);
    return idx.y + wh.x * idx.x;
}

__device__ float Compare(Histogram x, Histogram y) {
    float nx = x.totalSampleNum;
    float ny = y.totalSampleNum;
    float sqrt_y_x = sqrt(ny / nx);
    float sqrt_x_y = sqrt(nx / ny);

    float p = 0;
    float res = 0;
    for (int i = 0; i < HISTO_SIZE * 3; i++)
    {
        float hx = x.bin[i];
        float hy = y.bin[i];
        if (hx != 0 || hy != 0) {
            p++;
            float t = sqrt_y_x * hx - sqrt_x_y * hy;
            res += t * t / (hx + hy);
        }
    }
    return p == 0 ? 0 : res / p;
}

template<bool denoise>
__global__ void Denoise(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, int toneType) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size.x * size.y) return;

    int2 idx = int2{ id / size.x, id % size.x };

    float3 res;

    if (frameNum == 0) {
        if ((((idx.x + flip) % 2) + idx.y) % 2 == 0) {
            res = target[id];
        }
        else {
            res = target[UnRoll({ idx.x - 1, idx.y }, size)] + target[UnRoll({ idx.x + 1, idx.y }, size)]
                    + target[UnRoll({ idx.x, idx.y - 1 }, size)] + target[UnRoll({ idx.x, idx.y + 1 }, size)];
            res = res / 4;
        }
    }
    else {
        float variance = abs(histo_buffer[id].x2 - histo_buffer[id].x * histo_buffer[id].x);

        if (denoise && variance > 0.01) {
            Histogram center = histo_buffer[UnRoll(idx, size)];
            res = { 0 };
            float ws = 0;
            for (int i = -5; i <= 5; i++)
            {
                for (int j = -5; j <= 5; j++)
                {
                    int2 pairId = { idx.x + i, idx.y + j };
                    int t = UnRoll(pairId, size);
                    float w = max(0.0f, 1.0f - 1.2 * Compare(center, histo_buffer[t]));
                    res = res + target[t] * w;
                    ws += w;
                }
            }
            res = res / ws;
        }
        else {
            res = target[id];
        }
    }

    float3 val = res * exposure;

    if (toneType == 1)
        val = Gamma(val);
    else if (toneType == 2)
        val = ACES(val);

    const unsigned int red = (unsigned int)(255.0f * saturate_(val.x));
    const unsigned int gre = (unsigned int)(255.0f * saturate_(val.y));
    const unsigned int blu = (unsigned int)(255.0f * saturate_(val.z));
    target2[id] = 0xff000000 | (red << 16) | (gre << 8) | blu;
}

__global__ void ReprojectionDenoise(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, int toneType) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size.x * size.y) return;

    histo_buffer[id].totalSampleNum = target[id].x;
    histo_buffer[id].x = target[id].y;
    histo_buffer[id].x2 = target[id].z;
    float3 res = target[id];

    float3 val = res * exposure;

    if (toneType == 1)
        val = Gamma(val);
    else if (toneType == 2)
        val = ACES(val);

    const unsigned int red = (unsigned int)(255.0f * saturate_(val.x));
    const unsigned int gre = (unsigned int)(255.0f * saturate_(val.y));
    const unsigned int blu = (unsigned int)(255.0f * saturate_(val.z));
    target2[id] = 0xff000000 | (red << 16) | (gre << 8) | blu;
}

__global__ void ClearHis(Histogram* histo_buffer, int2 size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size.x * size.y) return;

    histo_buffer[id] = { 0 };
}

__global__ void CalculateShadowTerm_TR(float3* result, float3 ori, float3 dir, float3 lightDir, float alpha = 1, float g = 0) {

    float3 res = ShadowTerm_TR(ori, dir, lightDir, alpha, g);

    result[blockIdx.x * blockDim.x + threadIdx.x] = res;
}

float3 VolumeRender::GetTr(float3 ori, float3 dir, float3 lightDir, float alpha,float g, int sampleNum) const 
{
    int group = 32;
    int group_num = sampleNum / group + (sampleNum % group != 0 ? 1 : 0);
    float3* results;
    cudaMalloc(&results, sizeof(float3) * group_num * group);
    CheckError;
    CalculateShadowTerm_TR << <group_num, group >> > (results, ori, dir, lightDir,  alpha, g);
    float3* res_cpu = new float3[group_num * group];
    cudaDeviceSynchronize();
    CheckError;
    cudaMemcpy(res_cpu, results, sizeof(float3) * group * group_num, cudaMemcpyDeviceToHost);
    CheckError;
    cudaFree(results);
    CheckError;
    float3 res = { 0,0,0 };
    for (int i = 0; i < group * group_num; i++)
    {
        res = res + res_cpu[i];
    }
    delete[]res_cpu;
    return res / (group * group_num);
}

vector<float3> VolumeRender::GetRadiances(vector<float3> ori, vector<float3> dir, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int sampleNum, RenderType rt) {

    if (rt != RenderType::PT) {
        UpdateHGLut(g);
        Update_TR(lightDir, alpha);
    }

    int task_num = ori.size();

    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    float3* results;
    cudaMalloc(&results, sizeof(float3) * task_num);
    float3* oris;
    cudaMalloc(&oris, sizeof(float3) * task_num);
    float3* dirs;
    cudaMalloc(&dirs, sizeof(float3) * task_num);

    CheckError;

    cudaMemcpy(oris, ori.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);

    CheckError;

    cudaMemcpy(dirs, dir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    
    CheckError;
    
    volatile int* d_rec, *h_rec;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_rec, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_rec, (int*)h_rec, 0);
    *h_rec = 0;
    if (rt == RenderType::PT)
        CalculateRadianceMulti<0><<<group_num, group>>>(d_rec, results, oris, dirs, lightDir, lightColor, alpha, multiScatter, g, sampleNum);
    else if (rt == RenderType::RPNN)
        CalculateRadianceMulti<1><<<group_num, group>>>(d_rec, results, oris, dirs, lightDir, lightColor, alpha, multiScatter, g, sampleNum);
    else        
        CalculateRadianceMulti<2><<<group_num, group>>>(d_rec, results, oris, dirs, lightDir, lightColor, alpha, multiScatter, g, sampleNum);
    
    auto call_back = thread([&](){
        int value = 0;
        do {
            int value1 = *h_rec;
            if (value1 > value) {
                printf("Rendering: %6.2f%%\n", value1 * 100.0f / group_num);
                value = value1;
            }
            wait(1000);
        } while (value < group_num);
    });
    
    cudaDeviceSynchronize();

    call_back.join();

    cudaFreeHost((void*)h_rec);

    CheckError;

    vector<float3> res_cpu(task_num);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * task_num, cudaMemcpyDeviceToHost);

    CheckError;

    cudaFree(results);
    cudaFree(oris);
    cudaFree(dirs);

    CheckError;

    return res_cpu;
}

int flip_cpu = 0;
int rand_cpu = 0;

float3 last_ori, last_up, last_right;


vector<float3> VolumeRender::Render(int2 size, float3 ori, float3 up, float3 right, float3 lightDir, RenderType rt, float g, float alpha, float3 lightColor, int multiScatter, int sampleNum) {

    float3* results;
    cudaMalloc(&results, size.x * size.y * sizeof(float3));

    for (int i = 0; i < sampleNum; i++)
    {
        if (env_tex_dev != NULL && (rt != RenderType::PT)) {

            float rate = hdri_exp * 4 / (hdri_exp * 4 + max(lightColor.x, max(lightColor.y, lightColor.z)));

            if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < rate) {
                float3 rnd = Roberts2(rand_cpu);
                float3 dir = UniformSampleSphere(float2{ rnd.x, rnd.y });

                float2 uv = float2{ atan2f(-dir.z, dir.x) * (float)(0.5 / 3.1415926) + 0.5f, acosf(fmaxf(fminf(dir.y, 1.0f), -1.0f)) * (float)(1.0 / 3.1415926) };

                lightDir = dir;
                lightColor = hdri_img.Sample(uv) * hdri_exp * 4 / rate;
            }
            else
                lightColor = lightColor / (1 - rate);
        }

        if (rt != RenderType::PT) {
            UpdateHGLut(g);
            Update_TR(lightDir, alpha);
        }

        cudaMemcpyToSymbol(frameNum, &i, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(randNum, &i, sizeof(int), 0, cudaMemcpyHostToDevice);

        dim3 dimBlock(8, 4);
        dim3 dimGrid;

        dimGrid.x = (size.x + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (size.y + dimBlock.y - 1) / dimBlock.y;
         
        if (rt == RenderType::PT)
            RenderCamera<false><<<dimGrid, dimBlock>>>(results, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
        else if (rt == RenderType::RPNN)
            RenderCamera<true, Type::RPNN><<<dimGrid, dimBlock>>>(results, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
        else 
            RenderCamera<true, Type::MRPNN><<<dimGrid, dimBlock>>>(results, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);

        cudaDeviceSynchronize();

        CheckError;

    }

    vector<float3> res_cpu(size.x * size.y);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * size.x * size.y, cudaMemcpyDeviceToHost);
    cudaFree(results);

    return res_cpu;
}

void VolumeRender::Render(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int randseed, RenderType rt, int toneType, bool denoise) {

    if (env_tex_dev != NULL && (rt != RenderType::PT)) {

        float rate = hdri_exp * 4 / (hdri_exp * 4 + max(lightColor.x, max(lightColor.y, lightColor.z)));

        if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < rate) {
            float3 rnd = Roberts2(rand_cpu);
            float3 dir = UniformSampleSphere(float2{ rnd.x, rnd.y });

            float2 uv = float2{ atan2f(-dir.z, dir.x) * (float)(0.5 / 3.1415926) + 0.5f, acosf(fmaxf(fminf(dir.y, 1.0f), -1.0f)) * (float)(1.0 / 3.1415926) };

            lightDir = dir;
            lightColor = hdri_img.Sample(uv) * hdri_exp * 4 / rate;
        }
        else
            lightColor = lightColor / (1 - rate);
    }

    if (rt != RenderType::PT) {
        UpdateHGLut(g);
        Update_TR(lightDir, alpha);
    }

    cudaMemcpyToSymbol(frameNum, &randseed, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(randNum, &rand_cpu, sizeof(int), 0, cudaMemcpyHostToDevice);    
    cudaMemcpyToSymbol(flip, &flip_cpu, sizeof(int), 0, cudaMemcpyHostToDevice);

    flip_cpu = (flip_cpu + 1) % 2;    
    rand_cpu++;

    dim3 dimBlock(8, 4);
    dim3 dimGrid;

    dimGrid.x = (size.x + dimBlock.x - 1) / dimBlock.x;
    if (checkboard)
        dimGrid.y = (size.y + dimBlock.y * 2 - 1) / (dimBlock.y * 2);
    else
        dimGrid.y = (size.y + dimBlock.y) / dimBlock.y;

    int task_num = size.x * size.y;
    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    bool predict = rt != RenderType::PT;
    if (!last_predict && predict) {
        ClearHis<<<group_num, group>>>(histo_buffer, size);
    }
    last_predict = predict;

    if (rt == RenderType::PT)
        RenderCamera<false><<<dimGrid, dimBlock>>>(target, histo_buffer, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
    else if (rt == RenderType::RPNN)
        RenderCamera<true, Type::RPNN><<<dimGrid, dimBlock>>>(target, histo_buffer, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
    else
        RenderCamera<true, Type::MRPNN><<<dimGrid, dimBlock>>>(target, histo_buffer, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);

    if (!predict) {
        if (denoise)
            Denoise<true><<<group_num, group>>>(target, histo_buffer, target2, size, toneType);
        else
            Denoise<false><<<group_num, group>>>(target, histo_buffer, target2, size, toneType);
    }
    else
        ReprojectionDenoise<<<group_num, group>>>(target, histo_buffer, target2, size, toneType);

    cudaMemcpyToSymbol(lori, &ori, sizeof(float3), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lup, &up, sizeof(float3), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lright, &right, sizeof(float3), 0, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    CheckError;

    last_ori = ori;
    last_up = up;
    last_right = right;
    return;
}


vector<float3> VolumeRender::GetSamples(vector<float> alpha, vector<float3> ori, vector<float3> dir, vector<float3> lightDir, vector<float> g, vector<float> scatter, float3 lightColor, int multiScatter, int sampleNum) const {

    int task_num = ori.size();

    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    float3* results;
    cudaMalloc(&results, sizeof(float3) * task_num);
    float3* oris;
    cudaMalloc(&oris, sizeof(float3) * task_num);
    float3* dirs;
    cudaMalloc(&dirs, sizeof(float3) * task_num);
    float3* ldirs;
    cudaMalloc(&ldirs, sizeof(float3) * task_num);
    float* as;
    cudaMalloc(&as, sizeof(float) * task_num);
    float* gs;
    cudaMalloc(&gs, sizeof(float) * task_num);
    float* scatters;
    cudaMalloc(&scatters, sizeof(float) * task_num);
    CheckError;

    cudaMemcpy(oris, ori.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dirs, dir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(ldirs, lightDir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(as, alpha.data(), sizeof(float) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(gs, g.data(), sizeof(float) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scatters, scatter.data(), sizeof(float) * task_num, cudaMemcpyHostToDevice);

    CheckError;

    volatile int* d_rec, * h_rec;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_rec, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_rec, (int*)h_rec, 0);
    *h_rec = 0;
    GetSampleMulti<<<group_num, group>>>(d_rec, task_num, results, as, oris, dirs, ldirs, gs, scatters, lightColor, multiScatter, sampleNum);

    auto call_back = thread([&]() {
        int value = 0;
        do {
            int value1 = *h_rec;
            if (value1 > value) {
                printf("Rendering: %6.2f%%\n", value1 * 100.0f / group_num);
                value = value1;
            }
            wait(1000);
        } while (value < group_num);
        });

    cudaDeviceSynchronize();

    call_back.join();

    cudaFreeHost((void*)h_rec);

    CheckError;

    vector<float3> res_cpu(task_num);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * task_num, cudaMemcpyDeviceToHost);

    CheckError;

    cudaFree(results);
    cudaFree(oris);
    cudaFree(dirs);
    cudaFree(ldirs);
    cudaFree(as);
    cudaFree(gs);
    cudaFree(scatters);

    CheckError;

    return res_cpu;
}

__global__ void GetTrMulti(int task_num, float3* result, float alpha, float3* ori, float3* dir, float3 lightDir, float3 lightColor,float g = 0, int sampleNum = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= task_num) return;

    float3 res = ShadowTerm_TRs(ori[idx], dir[idx], lightDir, lightColor,alpha, g, sampleNum);
    result[idx] = res;
}
//这里的dir应该是 中心位置看向偏移位置
vector<float3> VolumeRender::GetTrs(float alpha, vector<float3> ori, vector<float3> dir, float3 lightDir,float3 lightColor, float g, int sampleNum) const {

    int task_num = ori.size();

    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    float3* results;
    cudaMalloc(&results, sizeof(float3) * task_num);
    float3* oris;
    cudaMalloc(&oris, sizeof(float3) * task_num);
    float3* dirs;
    cudaMalloc(&dirs, sizeof(float3) * task_num);

    CheckError;

    cudaMemcpy(oris, ori.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dirs, dir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);

    CheckError;

    GetTrMulti<<<group_num, group>>>(task_num, results, alpha, oris, dirs, lightDir, lightColor,g, sampleNum);

    cudaDeviceSynchronize();
    CheckError;

    vector<float3> res_cpu(task_num);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * task_num, cudaMemcpyDeviceToHost);

    CheckError;

    cudaFree(results);
    cudaFree(oris);
    cudaFree(dirs);
    CheckError;

    return res_cpu;
}
inline float Sample(float* data, int res, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= res || y >= res || z >= res) return 0;
    return max(0.0f, data[((x * res) + y) * res + z]);
}
inline float SampleClamp(float* data, int res, int x, int y, int z) {
    x = max(min(x,res-1),0);
    y = max(min(y,res-1),0);
    z = max(min(z,res-1),0);
    return max(0.0f, data[((x * res) + y) * res + z]);
}
inline float Sample(float* data, int res, float3 uv) {
    float3 pos = uv * res - 0.5;
    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);
    float3 w = pos - make_float3(x, y, z);

    return
        lerp(
            lerp(
                lerp(Sample(data, res, x, y, z), Sample(data, res, x, y, z + 1), w.z),
                lerp(Sample(data, res, x, y + 1, z), Sample(data, res, x, y + 1, z + 1), w.z),
                w.y),
            lerp(
                lerp(Sample(data, res, x + 1, y, z), Sample(data, res, x + 1, y, z + 1), w.z),
                lerp(Sample(data, res, x + 1, y + 1, z), Sample(data, res, x + 1, y + 1, z + 1), w.z),
            w.y),
        w.x);
}
inline float SampleClamp(float* data, int res, float3 uv) {
    float3 pos = uv * res - 0.5;
    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);
    float3 w = pos - make_float3(x, y, z);

    return
        lerp(
            lerp(
                lerp(SampleClamp(data, res, x, y, z), SampleClamp(data, res, x, y, z + 1), w.z),
                lerp(SampleClamp(data, res, x, y + 1, z), SampleClamp(data, res, x, y + 1, z + 1), w.z),
                w.y),
            lerp(
                lerp(SampleClamp(data, res, x + 1, y, z), SampleClamp(data, res, x + 1, y, z + 1), w.z),
                lerp(SampleClamp(data, res, x + 1, y + 1, z), SampleClamp(data, res, x + 1, y + 1, z + 1), w.z),
                w.y),
            w.x);
}

//#define TR_MUL 3.1415926535f
#define TR_MUL 1.0f
inline float Sample_TR(float* data, int res, float3 uv,float alpha,float3 lightDir) {
    const int MaxStep = 128;
    float3 ori = uv - 0.5;
    float dis = RayBoxDistance(ori, lightDir);
    float MaxStepInv = dis / MaxStep;
    float phase = 1.0;
    float3 Lpos = ori;
    float shadowdist = 0;
    for (int i = 0; i < MaxStep; i++)
    {
        Lpos = Lpos + lightDir * MaxStepInv;
        float lsample = Sample(data, res, Lpos+0.5f);
        shadowdist = shadowdist + lsample;
    }
    float shadowterm = exp(-shadowdist * alpha * MaxStepInv) * phase;
    return TR_MUL * shadowterm;//
}

// 安全版 AABB 相交：返回 [tmin, tmax]
inline bool RayBoxIntersectSafe(const float3& ori, const float3& dir,
                                const float3& bmin, const float3& bmax,
                                float& tmin, float& tmax)
{
    float3 invDir = make_float3(
        dir.x != 0.0f ? 1.0f / dir.x : 1e30f,
        dir.y != 0.0f ? 1.0f / dir.y : 1e30f,
        dir.z != 0.0f ? 1.0f / dir.z : 1e30f);

    float3 t0 = (bmin - ori) * invDir;
    float3 t1 = (bmax - ori) * invDir;

    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);

    tmin = max(tmin3.x, max(tmin3.y, tmin3.z));
    tmax = min(tmax3.x, min(tmax3.y, tmax3.z));

    return tmax >= max(tmin, 0.0f);
}

struct InitWeight {
    InitWeight();
};

void VolumeRender::MallocMemory() {
    cudaFree(0);

    datas = new float[resolution * resolution * resolution];
    hglut = new float[LUT_SIZE * LUT_SIZE];
    channel_desc = cudaCreateChannelDesc<float>();
    size = cudaExtent{ (size_t)resolution, (size_t)resolution, (size_t)resolution };
    cudaMalloc3DArray(&datas_dev, &channel_desc, size);

    // Create Density Texture Object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = datas_dev;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = true; // Use normalized coords as per legacy code
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp; // Use Clamp? Legacy used Border/Clamp?
    // Legacy Density used tex3D in render.cuh. Usually Clamp or Border.
    // Let's assume Clamp for now. 
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&density_tex, &resDesc, &texDesc, NULL);

    for (int i = 0; i < 9; i++) {
        int reso = 256 >> i;
        mips[i] = new float[reso * reso * reso];
        var_mips[i] = new float[reso * reso * reso];
        mip_size[i] = cudaExtent{ (size_t)reso, (size_t)reso, (size_t)reso };
        cudaMalloc3DArray(mips_dev + i, &channel_desc, mip_size[i]);
        cudaMalloc3DArray(var_mips_dev + i, &channel_desc, mip_size[i]);

        // Create Texture Objects for Mips
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = mips_dev[i];
        cudaCreateTextureObject(&mips_tex[i], &resDesc, &texDesc, NULL);

        // Create Texture Objects for Variance Mips
        resDesc.res.array.array = var_mips_dev[i];
        cudaCreateTextureObject(&var_mips_tex[i], &resDesc, &texDesc, NULL);
    }
    for (int i = 0; i < 8; i++) {
        int reso = 128 >> i;
        tr_mips[i] = new float[reso * reso * reso];
        tr_mip_size[i] = cudaExtent{ (size_t)reso, (size_t)reso, (size_t)reso };
        cudaMalloc3DArray(tr_mips_dev + i, &channel_desc, tr_mip_size[i], cudaArraySurfaceLoadStore);

        // Create Texture Object for TR Mips
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = tr_mips_dev[i];
        cudaCreateTextureObject(&tr_mips_tex[i], &resDesc, &texDesc, NULL);

        // Create Surface Object for TR Mips (for writing)
        cudaResourceDesc surfResDesc;
        memset(&surfResDesc, 0, sizeof(surfResDesc));
        surfResDesc.resType = cudaResourceTypeArray;
        surfResDesc.res.array.array = tr_mips_dev[i];
        cudaCreateSurfaceObject(&tr_mips_surf[i], &surfResDesc);
    }

    cudaMallocArray(&hglut_dev, &channel_desc, LUT_SIZE, LUT_SIZE, cudaArraySurfaceLoadStore);
    
    // Create HGLut Texture and Surface
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = hglut_dev;
    cudaCreateTextureObject(&hglut_tex, &resDesc, &texDesc, NULL); // Reuse texDesc

    cudaResourceDesc surfResDesc;
    memset(&surfResDesc, 0, sizeof(surfResDesc));
    surfResDesc.resType = cudaResourceTypeArray;
    surfResDesc.res.array.array = hglut_dev;
    cudaCreateSurfaceObject(&hglut_surf, &surfResDesc);
}

void VolumeRender::GenerateVarianceMipmaps() {
    // 并行生成 mip[0] 的均值和方差（把原始分辨率降到 256，如果需要）
    const int base_res = 256;
    const int ratio = max(1, resolution / base_res);

    ParallelFill(mips[0], var_mips[0], base_res,
        [&](int x, int y, int z, float u, float v, float w) -> float2 {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int cnt = 0;

            // 以 box filter 的方式从原始体素降采样
            for (int dz = 0; dz < ratio; ++dz)
                for (int dy = 0; dy < ratio; ++dy)
                    for (int dx = 0; dx < ratio; ++dx) {
                        float3 samplePos = float3{
                            u + (dx - ratio * 0.5f + 0.5f) / float(resolution),
                            v + (dy - ratio * 0.5f + 0.5f) / float(resolution),
                            w + (dz - ratio * 0.5f + 0.5f) / float(resolution)
                        };
                        float val = Sample(datas, resolution, samplePos);
                        sum     += val;
                        sum_sq  += val * val;
                        ++cnt;
                    }

            float inv_cnt = 1.0f / max(1, cnt);
            float mean = sum * inv_cnt;
            float variance = sum_sq * inv_cnt - mean * mean;
            variance = max(0.0f, variance);
            return make_float2(mean, variance);
        });

    // 逐级生成更粗的 mip（密度 + 方差），从 mip 0 -> 8
    for (int mip = 0; mip < 8; ++mip) {
        int cur_res  = base_res >> mip;
        int next_res = base_res >> (mip + 1);

        ParallelFill(mips[mip + 1], var_mips[mip + 1], next_res,
            [&](int x, int y, int z, float u, float v, float w) -> float2 {
                // 从上一层的 2x2x2 区域聚合
                float sum = 0.0f;
                float sum_sq = 0.0f;
                for (int iz = 0; iz < 2; ++iz)
                    for (int iy = 0; iy < 2; ++iy)
                        for (int ix = 0; ix < 2; ++ix) {
                            float3 sp = float3{
                                u + (ix - 0.5f) / float(cur_res),
                                v + (iy - 0.5f) / float(cur_res),
                                w + (iz - 0.5f) / float(cur_res)
                            };
                            float val = Sample(mips[mip], cur_res, sp);
                            sum    += val;
                            sum_sq += val * val;
                        }

                const float inv8 = 1.0f / 8.0f;
                float mean = sum * inv8;
                float variance = sum_sq * inv8 - mean * mean;
                variance = max(0.0f, variance);
                return make_float2(mean, variance);
            });
    }
}

float VolumeRender::DensityAtUV(float mip, float3 uv) {
    int a = int(mip);
    float w = mip - a;
    return lerp(DensityAtUV(a, uv), DensityAtUV(a + 1, uv), w);
}

inline float trilinear_interp(float* data, int res, float3 pos) {
    // pos in voxel space [0, res)
    int x0 = (int)floorf(pos.x);
    int y0 = (int)floorf(pos.y);
    int z0 = (int)floorf(pos.z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float fx = pos.x - x0;
    float fy = pos.y - y0;
    float fz = pos.z - z0;

    auto get_val = [&](int x, int y, int z) -> float {
        if (x < 0 || x >= res || y < 0 || y >= res || z < 0 || z >= res)
            return 0.0f;
        return data[(z * res + y) * res + x];
    };

    float v000 = get_val(x0, y0, z0);
    float v100 = get_val(x1, y0, z0);
    float v010 = get_val(x0, y1, z0);
    float v110 = get_val(x1, y1, z0);
    float v001 = get_val(x0, y0, z1);
    float v101 = get_val(x1, y0, z1);
    float v011 = get_val(x0, y1, z1);
    float v111 = get_val(x1, y1, z1);

    float v00 = lerp(v000, v100, fx);
    float v01 = lerp(v001, v101, fx);
    float v10 = lerp(v010, v110, fx);
    float v11 = lerp(v011, v111, fx);

    float v0 = lerp(v00, v10, fy);
    float v1 = lerp(v01, v11, fy);

    return lerp(v0, v1, fz);
}

float VolumeRender::VarianceAtUV(int mip, float3 uv) {
    if (mip < 0 || mip >= 9)
        return 0.0f;

    int res = 256 >> mip;
    // uv in [0,1) -> position in voxel space
    float3 pos = uv * float(res);
    return trilinear_interp(var_mips[mip], res, pos);
}

// ==== 以下为缺失实现，按 MRPNN1 逻辑补回 ====

void VolumeRender::Update() {
    // 1. Compute actual Max Density on CPU
    float current_max = 0.0f;
    long long total_voxels = (long long)resolution * resolution * resolution;
    #pragma omp parallel for reduction(max:current_max)
    for (long long i = 0; i < total_voxels; i++) {
        if (datas[i] > current_max) current_max = datas[i];
    }
    this->max_density = max(0.0001f, current_max);

    // 2. Sync to GPU Global Symbols
    cudaMemcpyToSymbol(maxDensity, &this->max_density, sizeof(float));
    cudaMemcpyToSymbol(Resolution, &this->resolution, sizeof(int));
    CheckError;

    // 3. Data upload
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)datas, resolution * sizeof(float), resolution, resolution);
    copyParams.dstArray = datas_dev;
    copyParams.extent = make_cudaExtent(resolution, resolution, resolution);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    CheckError;

    // 4. Generate Mipmaps
    GenerateVarianceMipmaps();

    for (int i = 0; i < 9; i++) {
        int reso = 256 >> i;
        copyParams.srcPtr = make_cudaPitchedPtr((void*)mips[i], reso * sizeof(float), reso, reso);
        copyParams.dstArray = mips_dev[i];
        copyParams.extent = make_cudaExtent(reso, reso, reso);
        cudaMemcpy3D(&copyParams);

        copyParams.srcPtr = make_cudaPitchedPtr((void*)var_mips[i], reso * sizeof(float), reso, reso);
        copyParams.dstArray = var_mips_dev[i];
        cudaMemcpy3D(&copyParams);
    }
    CheckError;

    // 5. Update Global Texture Objects
    cudaMemcpyToSymbol(_DensityVolume, &density_tex, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(_Mips, mips_tex, sizeof(cudaTextureObject_t) * 9);
    cudaMemcpyToSymbol(_Var_Mips, var_mips_tex, sizeof(cudaTextureObject_t) * 9);
    CheckError;
}

void VolumeRender::UpdateHGLut(float g)
{
    if (g == hginlut) return;

    // Kernel launch with Surface Object
    dim3 block(8, 8);
    dim3 grid((LUT_SIZE + block.x - 1) / block.x,
              (LUT_SIZE + block.y - 1) / block.y);
    Fill_Hg<<<grid, block>>>(hglut_surf, g);
    CheckError;

    // Update global texture object symbol
    cudaMemcpyToSymbol(_HGLut, &hglut_tex, sizeof(cudaTextureObject_t));
    CheckError;

    hginlut = g;
}

void VolumeRender::Update_TR(float3 lightDir, float alpha, bool CPU)
{
    if (lightDir.x == tr_lightDir.x && lightDir.y == tr_lightDir.y &&
        lightDir.z == tr_lightDir.z && alpha == tr_alpha)
        return;

    tr_lightDir = lightDir;
    tr_alpha = alpha;

    if (CPU) {
        int res = 256;
        for (int tr_mip = 0; tr_mip < 8; ++tr_mip)
        {
            float* source = nullptr;
            res = 128 >> tr_mip;
            int source_res = res;

            // 使用密度 mip[mip+1] 做 TR 采样
            source = mips[tr_mip + 1];
            float alphaScale = pow(1.73f, tr_mip + 1.0f);

            ParallelFill(tr_mips[tr_mip], res,
                [&](int x, int y, int z, float u, float v, float w) {
                    return Sample_TR(source, source_res,
                                     float3{u, v, w},
                                     alpha / alphaScale, lightDir);
                });

            // 上传到 GPU
            cudaMemcpy3DParms copyParams = {0};
            copyParams.srcPtr = make_cudaPitchedPtr(
                (void*)tr_mips[tr_mip], source_res * sizeof(float),
                source_res, source_res);
            copyParams.dstArray = tr_mips_dev[tr_mip];
            copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
            copyParams.kind = cudaMemcpyHostToDevice;
            cudaMemcpy3D(&copyParams);
            CheckError;
        }
    } else {
        int res = 256;
        for (int tr_mip = 0; tr_mip < 8; ++tr_mip)
        {
            res = 128 >> tr_mip;
            float alphaScale = pow(1.73f, tr_mip + 1.0f);

            // Use density texture from mip+1
            // Use TR surface for current mip
            cudaTextureObject_t dens_tex = mips_tex[tr_mip + 1];
            cudaSurfaceObject_t tr_surf = tr_mips_surf[tr_mip];

            dim3 block(4, 4, 4);
            dim3 grid((res + 3) / 4, (res + 3) / 4, (res + 3) / 4);
            Fill_TR<<<grid, block>>>(tr_surf, dens_tex, res, alpha / alphaScale, lightDir);
        }
    }

    // Update global texture object array
    cudaMemcpyToSymbol(_TR_Mips, tr_mips_tex, sizeof(cudaTextureObject_t) * 8);
    CheckError;
}

float VolumeRender::DensityAtPosition(int mip, float3 pos) {
    return Sample(mips[mip], 256 >> mip, pos + 0.5f);
}

float VolumeRender::DensityAtUV(int mip, float3 uv) {
    return Sample(mips[mip], 256 >> mip, uv);
}

// 已有的 float 版本在上面
// float VolumeRender::DensityAtUV(float mip, float3 uv) { ... }

VolumeRender::VolumeRender(int resolution) : resolution(resolution)
{
    MallocMemory();
}

VolumeRender::VolumeRender(std::string path)
{
    // 简化版：保持原有加载逻辑
    if (FILE* file = fopen((path + ".bin").c_str(), "rb")) {
        fread(&resolution, sizeof(int), 1, file);
        MallocMemory();
        fread(datas, sizeof(float), resolution * resolution * resolution, file);
        fclose(file);
        Update();
        return;
    }

    std::string format = path.substr(path.size() - 3, 3);
    if (format == "txt") {
        FILE* f = fopen(path.c_str(), "r");
        fscanf(f, "%d", &resolution);
        int total = resolution * resolution * resolution;
        MallocMemory();
        int index = 0;
        int loop_num = (total / 8) + (total % 8 != 0 ? 1 : 0);
        while (index < loop_num) {
            fscanf(f, "%f %f %f %f %f %f %f %f",
                   datas + index * 8,
                   datas + index * 8 + 1,
                   datas + index * 8 + 2,
                   datas + index * 8 + 3,
                   datas + index * 8 + 4,
                   datas + index * 8 + 5,
                   datas + index * 8 + 6,
                   datas + index * 8 + 7);
            index++;
        }
        fclose(f);
    } else if (format == "vox") {
        resolution = 256;
        MallocMemory();
        std::ifstream infile(path);
        std::string line;
        int i = 0;
        float inv = 64.0f / 255.0f;
        while (!infile.eof()) {
            std::getline(infile, line);
            if (line.empty()) continue;
            std::string firstc = line.substr(0, 1);
            if (firstc != "w" && firstc != "h" && firstc != "d") {
                if (i >= 256 * 256 * 256) break;
                if (i % (256 * 256 * 32) == 0)
                    printf("Loading vox percent:%.2f%%\n",
                           100.0 * (float)i / (256 * 256 * 256));
                std::stringstream data(line);
                int d[64];
                for (int j = 0; j < 64; ++j) data >> d[j];
                for (int j = 0; j < 64; ++j)
                    datas[i + j] = (float)d[j] * inv;
                i += 64;
            }
        }
        infile.close();
    } else {
        printf("File not found!\n");
        return;
    }

    Update();

    // 缓存为 .bin，和原版一致
    if (FILE* file = fopen((path + ".bin").c_str(), "wb")) {
        fwrite(&resolution, sizeof(int), 1, file);
        fwrite(datas, sizeof(float),
               resolution * resolution * resolution, file);
        fclose(file);
    }
}

VolumeRender::~VolumeRender()
{
    // Destroy Texture/Surface Objects
    if (density_tex) cudaDestroyTextureObject(density_tex);
    if (hdri_tex) cudaDestroyTextureObject(hdri_tex);
    if (hglut_tex) cudaDestroyTextureObject(hglut_tex);
    if (hglut_surf) cudaDestroySurfaceObject(hglut_surf);

    for (int i = 0; i < 9; ++i) {
        if (mips_tex[i]) cudaDestroyTextureObject(mips_tex[i]);
        if (var_mips_tex[i]) cudaDestroyTextureObject(var_mips_tex[i]);
    }
    for (int i = 0; i < 8; ++i) {
        if (tr_mips_tex[i]) cudaDestroyTextureObject(tr_mips_tex[i]);
        if (tr_mips_surf[i]) cudaDestroySurfaceObject(tr_mips_surf[i]);
    }

    cudaFreeArray(datas_dev);
    cudaFreeArray(hglut_dev);
    delete[] datas;
    delete[] hglut;

    for (int i = 0; i < 9; ++i) {
        delete[] mips[i];
        delete[] var_mips[i];
        cudaFreeArray(mips_dev[i]);
        cudaFreeArray(var_mips_dev[i]);
    }
    for (int i = 0; i < 8; ++i) {
        delete[] tr_mips[i];
        cudaFreeArray(tr_mips_dev[i]);
    }

    if (env_tex_dev != 0) {
        cudaFreeArray(env_tex_dev);
    }
    if (hdri_img.data != 0) {
        delete hdri_img.data;
    }
}

void VolumeRender::SetHDRI(string path) {
    unsigned int x, y;
    float* data = nullptr;
    if (load_hdr_float4(&data, &x, &y, path.c_str())) {
        if (hdri_img.data) free(hdri_img.data);
        hdri_img.data = (float4*)data;
        hdri_img.sx = x;
        hdri_img.sy = y;

        if (env_tex_dev) cudaFreeArray(env_tex_dev);
        
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&env_tex_dev, &channelDesc, x, y);
        cudaMemcpyToArray(env_tex_dev, 0, 0, data, x * y * sizeof(float4), cudaMemcpyHostToDevice);

        // Create Texture Object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = env_tex_dev;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = true;

        if (hdri_tex) cudaDestroyTextureObject(hdri_tex);
        cudaCreateTextureObject(&hdri_tex, &resDesc, &texDesc, NULL);
        
        cudaMemcpyToSymbol(_HDRI, &hdri_tex, sizeof(cudaTextureObject_t));
    } else {
        printf("Failed to load HDRI: %s\n", path.c_str());
    }
}

void VolumeRender::SetCheckboard(bool checkboard) {
    this->checkboard = checkboard;
    int cb = checkboard ? 1 : 0;
    cudaMemcpyToSymbol(dev_checkboard, &cb, sizeof(int));
}

void VolumeRender::SetEnvExp(float exp) {
    hdri_exp = exp;
    cudaMemcpyToSymbol(enviroment_exp, &exp, sizeof(float));
}

void VolumeRender::SetTrScale(float scale) {
    cudaMemcpyToSymbol(tr_scale, &scale, sizeof(float));
}

void VolumeRender::SetScatterRate(float rate) {
    float3 r = {rate, rate, rate};
    cudaMemcpyToSymbol(scatter_rate, &r, sizeof(float3));
}

void VolumeRender::SetScatterRate(float3 rate) {
    cudaMemcpyToSymbol(scatter_rate, &rate, sizeof(float3));
}

void VolumeRender::SetExposure(float exp) {
    cudaMemcpyToSymbol(exposure, &exp, sizeof(float));
}

void VolumeRender::SetSurfaceIOR(float ior) {
    cudaMemcpyToSymbol(IOR, &ior, sizeof(float));
}

void VolumeRender::SetData(int x, int y, int z, float value) {
    if (x >= 0 && x < resolution && y >= 0 && y < resolution && z >= 0 && z < resolution) {
        datas[x * resolution * resolution + y * resolution + z] = value;
    }
}

void VolumeRender::SetDatas(FillFunc func) {
    #pragma omp parallel for
    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            for (int k = 0; k < resolution; k++)
            {
                datas[(i * resolution + j) * resolution + k] = func(i, j, k, (float)i / resolution, (float)j / resolution, (float)k / resolution);
            }
        }
    }
}

float VolumeRender::GetHGLut(float cos, float angle) {
    // Not implemented on CPU
    return 0.0f;
}

// ...existing tail code (SetHDRI, SetEnvExp, etc.)...