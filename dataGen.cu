#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>      // 新增：构造每行 CSV
#include <omp.h>        // 新增：OpenMP

using std::vector;
using std::string;

// 1. 结构体定义：在原有 density 基础上加 variance
class SamplePoint {
public:
    float3 Position;
    float3 ViewDir;
    float3 LightDir;
    float Alpha;
    float g;

    SamplePoint(float3 p, float3 v, float3 l, float a, float g_) 
        : Position(p), ViewDir(v), LightDir(l), Alpha(a), g(g_) {}
};

class DisneyDescriptor {
public:
    class Layer {
    public:
        static const size_t SIZE_X = 5;
        static const size_t SIZE_Y = 5;
        static const size_t SIZE_Z = 9;
        static const size_t LAYER_SIZE = SIZE_Z * SIZE_Y * SIZE_X;

        float density[LAYER_SIZE];
        float variance[LAYER_SIZE];
    };

    static const size_t LAYERS_CNT = 10;

    Layer layers[LAYERS_CNT];
    float Gamma = 0.0f;
    float Radiance = 0.0f;
};

// 2. 工具函数 & 几何：完全用 MRPNN1 版本
float hash1() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float3 hash31sphere() {
    float3 Rands = float3{ hash1(),hash1(),hash1() };
    float theta = 2 * 3.14159265358979f * Rands.x;
    float phi   = acosf(2 * Rands.y - 1.0f);
    float3 fp   = float3{ cosf(theta) * sinf(phi), sinf(theta) * sinf(phi), cosf(phi) };
    return normalize(fp);
}

float3 hash3box(float scale = 0.01f) {
    float3 Rands = hash31sphere() * cbrtf(hash1());
    return Rands * scale;
}

float RayBoxOffset_(float3 p, float3 dir)
{
    dir = inv(dir);
    float3 bmax = { 0.4999f, 0.4999f, 0.4999f };
    float3 to_axil_dis = -p * dir;
    float3 axil_to_face_dis = bmax * dir;

    float3 dis0 = to_axil_dis + axil_to_face_dis;
    float3 dis1 = to_axil_dis - axil_to_face_dis;

    float3 tmin = min(dis0, dis1);
    float3 tmax = max(dis0, dis1);

    float tmi = max(tmin.x, max(tmin.y, tmin.z));
    float tma = min(tmax.x, min(tmax.y, tmax.z));

    return tma >= tmi ? max(tmi, 0.0f) : -1.0f;
}

float RayBoxDistance_(float3 p, float3 dir)
{
    dir = inv(dir);
    float3 bmax = { 0.5f, 0.5f, 0.5f };
    float3 to_axil_dis     = -p * dir;
    float3 axil_to_face_dis= bmax * dir;

    float3 dis0 = to_axil_dis + axil_to_face_dis;
    float3 dis1 = to_axil_dis - axil_to_face_dis;  // 修正这里

    float3 tmax = max(dis0, dis1);
    float tma   = min(tmax.x, min(tmax.y, tmax.z));

    return tma;
}

bool DeterminateNextVertex(VolumeRender& CurrentVolume, float alpha, float g,
                           float3 pos, float3 dir, float dis,
                           float3* nextPos, float3* nextDir)
{
    float SMax = CurrentVolume.max_density * alpha;
    float t = 0.0f;
    int loop_num = 0;
    while (loop_num++ < 10000)
    {
        float rk = hash1();
        t -= logf(1.0f - rk) / SMax;

        if (t > dis)
        {
            *nextPos = { 0,0,0 };
            *nextDir = { 0,0,0 };
            return false;
        }
        else
        {
            rk = hash1();
            float density = CurrentVolume.DensityAtPosition(0, pos + dir * t);
            float S = density * alpha;
            if (S / SMax > rk)
                break;

            if (density < 0.0f)
                t -= density;
        }
    }
    *nextDir = SampleHenyeyGreenstein(hash1(), hash1(), dir, g);
    *nextPos = pos + dir * t;
    return true;
}

void MeanFreePathSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples,
                        float3 ori, float3 dir, float3 lightDir,
                        int maxcount, float alpha, float g)
{
    dir = normalize(dir);
    lightDir = normalize(lightDir);

    float dis = RayBoxOffset_(ori, dir);
    if (dis < 0.0f)
        return;

    float3 samplePosition = ori + dir * dis;
    float3 rayDirection   = dir;
    for (int i = 0; i < 4; ++i)
    {
        float3 nextPos, nextDir;
        float dLimit = RayBoxDistance_(samplePosition, rayDirection);
        bool in_volume = DeterminateNextVertex(CurrentVolume, alpha, g,
                                               samplePosition, rayDirection,
                                               dLimit, &nextPos, &nextDir);

        if (!in_volume || Samples.size() >= (size_t)maxcount)
            return;

        if (i == 0 ||
            dot(samplePosition - nextPos, samplePosition - nextPos) > 1.0 / 64.0 ||
            hash1() > 0.9f)
        {
            Samples.push_back(
                SamplePoint(
                    hash1() > 0.5f ? nextPos + hash3box(1.0f / 128.0f) : nextPos,
                    hash1() > 0.25f ? dir : rayDirection,
                    lightDir, alpha, g));
        }
        samplePosition = nextPos;
        rayDirection   = nextDir;
    }
}

void GetDesiredCountSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples,
                           int Count, float density_min, float density_max)
{
    Samples.clear();
    int last_print = 0;
    int print_per  = std::max(1, Count / 8);
    while ((int)Samples.size() < Count)
    {
        if ((int)Samples.size() / print_per > last_print)
        {
            printf("Getting Samples: %.5f%%\n",
                   float(Samples.size()) / Count * 100.0f);
            last_print = (int)Samples.size() / print_per;
        }

        float3 ori  = hash31sphere() * 3.0f;
        float3 dir  = normalize(hash31sphere() + normalize(-ori));
        float3 ldir = hash31sphere();
        float Alpha = lerp(density_min, density_max, hash1());
        float g     = 0.857f;

        MeanFreePathSample(CurrentVolume, Samples, ori, dir, ldir, Count, Alpha, g);
    }
}

// 5. 描述符提取：在 MRPNN1 的基础上多读取 variance
DisneyDescriptor GetDisneyDesc(VolumeRender& volume, float3 uv, float3 v, float3 s,
                               float alpha, float descSizeAtLevel0)
{
    DisneyDescriptor descriptor;
    v = normalize(v);
    const float3 eZ = normalize(s);
    const float3 eX = normalize(cross(eZ, v));
    const float3 eY = cross(eX, eZ);
    descriptor.Gamma = acosf(dot(v, eZ));

    const float3 origin = uv;
    float scale        = 0.5f * descSizeAtLevel0;
    float mipmapLevel  = 0.0f;

    for (size_t layerId = 0; layerId < DisneyDescriptor::LAYERS_CNT; ++layerId)
    {
        float currentmipmapLevel = max(min(mipmapLevel - 1.0f, 9.0f), 0.0f);
        uint32_t sampleId = 0;
        for (int z = -2; z <= 6; ++z)
        {
            for (int y = -2; y <= 2; ++y)
            {
                for (int x = -2; x <= 2; ++x)
                {
                    float3 offset = (eX * x + eY * y + eZ * z) * scale;
                    const float3 pos = origin + offset;

                    float d = volume.DensityAtUV((int)(currentmipmapLevel + 0.001f), pos);
                    descriptor.layers[layerId].density[sampleId]  = d * alpha / 64.0f;
                    descriptor.layers[layerId].variance[sampleId] = volume.VarianceAtUV((int)(currentmipmapLevel + 0.001f), pos);
                    ++sampleId;
                }
            }
        }
        scale *= 2.0f;
        mipmapLevel += 1.0f;
    }
    return descriptor;
}

// 6. 主函数：保持你现有的数据路径和输出格式，只调用上面的采样和描述符
int main() {
    std::string DataPath    = "/home/guoshudan/Downloads/Data/";
    std::string DataName    = "DS_10000_with_variance.csv";
    std::string RelativePath= "/home/guoshudan/Downloads/Data/";
    
    vector<std::string> DataList;
    DataList.push_back("CLOUD0");

    const int CountAll     = 10000;
    int CountPerModel      = CountAll / (int)DataList.size();

    std::ofstream outfile(DataPath + DataName);
    outfile << "# Samples | Density_Layers(10) | Variance_Layers(10) | Gamma | Radiance\n";

    for (int i = 0; i < (int)DataList.size(); ++i) {
        VolumeRender v(RelativePath + DataList[i]);
        printf("\nComputing: %s\n", DataList[i].c_str());
        printf(">>> [Debug] Max Density: %.6f\n", v.max_density);
        
        std::vector<SamplePoint> Samples;
        GetDesiredCountSample(v, Samples, CountPerModel, 0.5f, 6.0f);

        std::vector<float3> SOris, SDirs, SLDirs;
        std::vector<float>  SAlphas, SGs, SScatters;
        for (auto& s : Samples) {
            SOris.push_back(s.Position);
            SDirs.push_back(s.ViewDir);
            SLDirs.push_back(s.LightDir);
            SAlphas.push_back(s.Alpha);
            SGs.push_back(s.g);
            SScatters.push_back(1.0f);
        }

        printf("GPU Rendering Path Tracing Ground Truth...\n");
        std::vector<float3> Radiances =
            v.GetSamples(SAlphas, SOris, SDirs, SLDirs, SGs, SScatters,
                         float3{1,1,1}, 512, 1024);
        printf("RealRadianceSet Size: %zu\n", Radiances.size());

        float gap = 0.25f / 1024.0f;

        // 预先分配一块缓存，每个样本一行字符串
        std::vector<std::string> lines(Radiances.size());

        // 用 OpenMP 并行每个样本的 descriptor + 行构建
        #pragma omp parallel for schedule(dynamic)
        for (long long k = 0; k < (long long)Radiances.size(); ++k) {
            // 1. 构造 descriptor（原逻辑不变）
            DisneyDescriptor desc =
                GetDisneyDesc(v, SOris[k] + float3{0.5f,0.5f,0.5f},
                              SDirs[k], SLDirs[k], SAlphas[k], gap);

            // 2. 把一整行写到 stringstream（线程本地）
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6);

            // 密度
            for (int j = 0; j < (int)DisneyDescriptor::LAYERS_CNT; ++j)
                for (int n = 0; n < (int)DisneyDescriptor::Layer::LAYER_SIZE; ++n)
                    oss << desc.layers[j].density[n] << ",";

            // 方差
            for (int j = 0; j < (int)DisneyDescriptor::LAYERS_CNT; ++j)
                for (int n = 0; n < (int)DisneyDescriptor::Layer::LAYER_SIZE; ++n)
                    oss << desc.layers[j].variance[n] << ",";

            oss << desc.Gamma << "," << Radiances[k].x << "\n";

            // 3. 写回共享数组（每个线程只写自己行，不需要锁）
            lines[k] = oss.str();
        }

        // 单线程把缓存里的行顺序写到文件，避免并发 IO
        for (const auto& line : lines)
            outfile << line;
    }

    outfile.close();
    printf("\nAll Done. File saved to %s\n", (DataPath + DataName).c_str());
    return 0;
}