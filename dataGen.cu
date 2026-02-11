#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include "core/perlin_noise.cuh" // Corrected include path
#include <chrono>
#include <random>
#include <iomanip>
#include <vector>

using std::vector;

// Define SamplePoint before usage
class SamplePoint {
public:
    float3 Position;
    float3 ViewDir;
    float3 LightDir;
    float Alpha;
    float g;

    SamplePoint(float3 p, float3 v, float3 l, float a, float g_) {
        Position = p;
        ViewDir = v;
        LightDir = l;
        Alpha = a;
        g = g_;
    }
};

// Define Samples before usage
std::vector<SamplePoint> Samples;

// Replace the old device-side hash1 using rand()
/*
__device__ float hash1() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
*/

// New: simple host-side RNG helper (all call sites are host functions)
float hash1() {
    static thread_local std::mt19937 gen(
        static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
            reinterpret_cast<uintptr_t>(&gen)
        )
    );
    static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

class DisneyDescriptor {
public:
    static const size_t SIZE_X = 5;
    static const size_t SIZE_Y = 5;
    static const size_t SIZE_Z = 9;
    static const size_t LAYER_SIZE = SIZE_Z * SIZE_Y * SIZE_X;
    static const size_t LAYERS_CNT = 10;

    class Layer {
    public:
        float density[LAYER_SIZE];
        float variance[LAYER_SIZE]; // Added variance data
        __device__ float operator[](int i) const {
            return density[i];
        }
    };

    Layer layers[LAYERS_CNT];
    float Gamma = 0.0f;
    float Radiance = 0.0f;
};

DisneyDescriptor GetDisneyDesc(VolumeRender& volume, float3 uv, float3 v, float3 s, float alpha, float descSizeAtLevel0) {
    DisneyDescriptor descriptor;
    v = normalize(v);
    const float3 eZ = normalize(s);
    const float3 eX = normalize(cross(eZ, v));
    const float3 eY = cross(eX, eZ);
    descriptor.Gamma = acos(dot(v, eZ));
    const float3 origin = uv;
    float mipmapLevel = 0;

    for (size_t layerId = 0; layerId < DisneyDescriptor::LAYERS_CNT; layerId++) {
        float currentmipmapLevel = max(min(mipmapLevel - 1.0f, 9.0f), 0.0f);
        uint32_t sampleId = 0;
        for (int z = -2; z <= 6; z++) {
            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    float3 offset = (eX * x + eY * y + eZ * z) * 0.5f * descSizeAtLevel0;
                    const float3 pos = origin + offset;
                    float density = volume.DensityAtUV(int(currentmipmapLevel + 0.001f), pos);
                    descriptor.layers[layerId].density[sampleId] = density * alpha / 64.0f;
                    float variance = volume.VarianceAtUV(int(currentmipmapLevel + 0.001f), pos);
                    descriptor.layers[layerId].variance[sampleId] = variance;
                    sampleId++;
                }
            }
        }
        mipmapLevel++;
    }

    return descriptor;
}

float3 hash31sphere() {
    float3 Rands = float3{ hash1(), hash1(), hash1() };
    float theta = 2 * 3.14159265358979 * Rands.x;
    float phi = acos(2 * Rands.y - 1.0);
    float3 fp = float3{ cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi) };
    return normalize(fp);
}

float3 hash3box(float scale = 0.01) {
    float3 Rands = hash31sphere() * cbrt(hash1());
    return Rands * scale;
}

float RayBoxOffset_(float3 p, float3 dir) {
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
    return tma >= tmi ? max(tmi, 0.0f) : -1;
}

float RayBoxDistance_(float3 p, float3 dir) {
    dir = inv(dir);
    float3 bmax = { 0.5f, 0.5f, 0.5f };
    float3 to_axil_dis = -p * dir;
    float3 axil_to_face_dis = bmax * dir;
    float3 dis0 = to_axil_dis + axil_to_face_dis;
    float3 dis1 = to_axil_dis - axil_to_face_dis;
    float3 tmin = min(dis0, dis1);
    float3 tmax = max(dis0, dis1);
    float tmi = max(tmin.x, max(tmin.y, tmin.z));
    float tma = min(tmax.x, min(tmax.y, tmax.z));
    return tma;
}

bool DeterminateNextVertex(VolumeRender& CurrentVolume, float alpha, float g, float3 pos, float3 dir, float dis, float3* nextPos, float3* nextDir) {
    float SMax = CurrentVolume.max_density * alpha;
    float t = 0;
    int loop_num = 0;
    while (loop_num++ < 10000) {
        float rk = hash1();
        t -= log(1 - rk) / SMax;
        if (t > dis) {
            *nextPos = { 0, 0, 0 };
            *nextDir = { 0, 0, 0 };
            return false;
        }
        float density = CurrentVolume.DensityAtPosition(0, pos + (dir * t));
        float S = density * alpha;
        if (S / SMax > rk) {
            break;
        }
        if (density < 0) {
            t -= density;
        }
    }
    *nextDir = SampleHenyeyGreenstein(hash1(), hash1(), dir, g);
    *nextPos = (dir * t) + pos;
    return true;
}

void MeanFreePathSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples, float3 ori, float3 dir, float3 lightDir, int maxcount, float alpha, float g) {
    dir = normalize(dir);
    lightDir = normalize(lightDir);

    float dis = RayBoxOffset_(ori, dir);
    if (dis < 0) {
        return;
    }

    float3 samplePosition = ori + dir * dis;
    float3 rayDirection = dir;
    for (int i = 0; i < 4; i++) {
        float3 nextPos, nextDir;
        float dis = RayBoxDistance_(samplePosition, rayDirection);
        bool in_volume = DeterminateNextVertex(CurrentVolume, alpha, g, samplePosition, rayDirection, dis, &nextPos, &nextDir);
        if (!in_volume || Samples.size() >= maxcount) {
            return;
        }
        if (i == 0 || dot(samplePosition - nextPos, samplePosition - nextPos) > 1.0 / 64.0 || hash1() > 0.9) {
            Samples.push_back(SamplePoint(hash1() > 0.5 ? nextPos + hash3box(1.0 / 128.0) : nextPos, hash1() > 0.25 ? dir : rayDirection, lightDir, alpha, g));
        }
        samplePosition = nextPos;
        rayDirection = nextDir;
    }
}

void GetDesiredCountSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples, int Count, float density_min, float density_max) {
    Samples.clear();
    int last_print = 0;
    int print_per = Count / 8;
    while (Samples.size() < Count) {
        if (Samples.size() / print_per > last_print) {
            printf("Getting Samples: %.5f%%\n", float(Samples.size()) / Count * 100.0f);
            last_print = Samples.size() / print_per;
        }
        float3 ori = hash31sphere() * 0.5f;
        float3 dir = normalize(hash31sphere() + normalize(-ori));
        float3 ldir = hash31sphere();
        float Alpha = lerp(density_min, density_max, hash1());
        float g = 0.857f;
        MeanFreePathSample(CurrentVolume, Samples, ori, dir, ldir, Count, Alpha, g);
    }
}

void DebugSamples(string vpath, string outpath, int count = 512, float alpha = 1.0, float alpha_max = 5.0) {
    VolumeRender v(vpath);
    vector<SamplePoint> Samples;
    GetDesiredCountSample(v, Samples, count, alpha, alpha_max);
    std::ofstream outfile(outpath);
    for (SamplePoint& s : Samples) {
        outfile << setiosflags(ios::fixed) << s.Position.x << ",";
        outfile << setiosflags(ios::fixed) << s.Position.y << ",";
        outfile << setiosflags(ios::fixed) << s.Position.z << ",";
        outfile << setiosflags(ios::fixed) << s.ViewDir.x << ",";
        outfile << setiosflags(ios::fixed) << s.ViewDir.y << ",";
        outfile << setiosflags(ios::fixed) << s.ViewDir.z << ",";
        outfile << setiosflags(ios::fixed) << s.LightDir.x << ",";
        outfile << setiosflags(ios::fixed) << s.LightDir.y << ",";
        outfile << setiosflags(ios::fixed) << s.LightDir.z << ",";
        outfile << std::endl;
    }
}

int main() {
    std::string DataPath = "./Data/";
    std::string DataName = "DS_10000.csv";
    std::string RelativePath = "./Data/";
    vector<std::string> DataList;
    DataList.push_back("cloud_0");
    // DataList.push_back("mediocris_high.512.txt");
    // DataList.push_back("cumulus_humilis.512.txt");
    // DataList.push_back("cumulus_congestus1.512.txt");
    vector<float> DensityMin;
    DensityMin.push_back(0.5f);
    DensityMin.push_back(0.5f);
    DensityMin.push_back(1.0f);
    DensityMin.push_back(0.5f);
    vector<float> DensityMax;
    DensityMax.push_back(6.0f);
    DensityMax.push_back(12.0f);
    DensityMax.push_back(40.0f);
    DensityMax.push_back(6.0f);
    const int CountAll = 10;
    const int MiniLoopCount = 1;
    int Computed = 0;
    int CountPer = CountAll / DataList.size() / MiniLoopCount;

    vector<DisneyDescriptor> Data;
    std::ofstream outfile(DataPath + DataName);
    outfile << "# " << CountAll << "x" << "(5,5,9)" << "x" << "10 Layers" << std::endl;
    for (int l = 0; l < MiniLoopCount; l++) {
        for (int i = 0; i < DataList.size(); i++) {
            Data.clear();
            printf("Processing %.2f%%\n", 100.0 * float(Computed) / Computed);
            printf("Computing:%s\n", DataList[i].c_str());
            printf("Desired Size:%d\n", CountPer);
            std::string CurrentData = DataList[i];
            float CurrentDensityMin = DensityMin[i];
            float CurrentDensityMax = DensityMax[i];
            VolumeRender v(RelativePath + CurrentData);

            printf(">>> [Debug] Model: %s\n", DataList[i].c_str());
            printf(">>> [Debug] Resolution: %d\n", v.resolution);
            printf(">>> [Debug] Max Density: %.6f\n", v.max_density);
            if (v.max_density <= 0.000001f) {
                printf("!!! ERROR: Max density is zero or near zero. Woodcock sampling will fail!\n");
                // 可以尝试打印前几个数据点，看看是不是读到的全是 0
                // printf("First voxel value: %.6f\n", v.datas[0]); 
            }
            printf("Getting Mean Free Path Samples\n");
            vector<SamplePoint> Samples;
            GetDesiredCountSample(v, Samples, CountPer, CurrentDensityMin, CurrentDensityMax);
            vector<float3> SampleOris;
            vector<float3> SampleDirs;
            vector<float3> SampleLDirs;
            vector<float> SampleAlphas;
            vector<float> SampleGs;
            vector<float> SampleScatters;
            for (int i = 0; i < Samples.size(); i++) {
                SamplePoint CurrentSample = Samples[i];
                SampleOris.push_back(CurrentSample.Position);
                SampleDirs.push_back(CurrentSample.ViewDir);
                SampleLDirs.push_back(CurrentSample.LightDir);
                SampleAlphas.push_back(CurrentSample.Alpha);
                SampleGs.push_back(CurrentSample.g);
                SampleScatters.push_back(1.0f);
            }
            float3 LightColor = { 1.0, 1.0, 1.0 };
            vector<float3> CurrentRadiances = v.GetSamples(SampleAlphas, SampleOris, SampleDirs, SampleLDirs, SampleGs, SampleScatters, LightColor, 512, 1024);
            printf("RealRadianceSet Size:%d\n", CurrentRadiances.size());
            float gap = 0.25f / 1024.0f;
            for (int i = 0; i < CurrentRadiances.size(); i++) {
                DisneyDescriptor desc = GetDisneyDesc(v, SampleOris[i] + float3{ 0.5f, 0.5f, 0.5f }, SampleDirs[i], SampleLDirs[i], SampleAlphas[i], gap);
                desc.Radiance = CurrentRadiances[i].x;
                Data.push_back(desc);
            }
            printf("FinalDescSet Size:%d\n", Data.size());

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::shuffle(Data.begin(), Data.end(), e);
            const int samplecount = 5 * 5 * 9;
            for (int i = 0; i < Data.size(); i++) {
                if (i % (Data.size() / 8) == 0) {
                    printf("Output Shuffle_Dataset:%.2f%%\n", 100.0f * (float)i / Data.size());
                }
                DisneyDescriptor& CS = Data[i];
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < samplecount; k++) {
                        outfile << setiosflags(ios::fixed) << CS.layers[j].density[k] << ",";
                    }
                }
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < samplecount; k++) {
                        outfile << setiosflags(ios::fixed) << CS.layers[j].variance[k] << ",";
                    }
                }
                outfile << setiosflags(ios::fixed) << CS.Gamma << ",";
                outfile << setiosflags(ios::fixed) << CS.Radiance << std::endl;
                Computed++;
            }
        }
    }

    // New: Generate Perlin Noise Volume
    {
        printf("Generating Perlin Noise Volume...\n");
        VolumeRender v_noise(256);


        v_noise.SetDatas([&](int x, int y, int z, float u, float v, float w) {
            float freq1 = 8.0f, amp1 = 1.0f;
            float freq2 = 32.0f, amp2 = 0.25f;
            float val = perlin_noise(u * freq1, v * freq1, w * freq1) * amp1;
            val += perlin_noise(u * freq2, v * freq2, w * freq2) * amp2;
            val = (val + 1.0f) * 0.5f;
            return (val > 0.7f) ? val * 5.0f : 0.0f;
        });
        v_noise.Update();
        GetDesiredCountSample(v_noise, Samples, CountPer, 0.5f, 5.0f);
    }

    outfile.close();
    return 0;
}
// int main() {
//     // 1. 路径适配 (建议使用相对路径，避免 Linux/Windows 环境冲突)
//     std::string DataPath = "./Output/";      // 存储生成的 CSV
//     std::string DataName = "TrainingData_5M_with_Var.csv";
//     std::string RelativePath = "./Data/";    // 你存放 33 个 txt 的地方
    
//     // 确保目录存在
//     system("mkdir -p ./Output/");

//     // 2. 自动构建 33 个模型的列表
//     vector<std::string> DataList;
//     vector<float> DensityMin;
//     vector<float> DensityMax;

//     for(int i = 0; i < 26; i++) {
//         DataList.push_back("cloud_" + std::to_string(i) + ".txt");
//         DensityMin.push_back(0.5f); DensityMax.push_back(10.0f); // 云彩密度
//     }
//     for(int i = 0; i < 7; i++) {
//         DataList.push_back("object_" + std::to_string(i) + ".txt");
//         DensityMin.push_back(1.0f); DensityMax.push_back(40.0f); // 破碎物体密度
//     }

//     // 3. 样本量计算
//     const int CountAll = 3300; 
//     int CountPerModel = CountAll / DataList.size(); // 每个模型采样数 (约 15 万)
//     int Computed = 0;

//     std::ofstream outfile(DataPath + DataName);
//     outfile << "# Samples with Variance | 10 Layers | (5,5,9) Stencil" << std::endl;

//     // 4. 开始主循环
//     for (int i = 0; i < DataList.size(); i++) {
//         printf("\n--- Processing Model [%d/33]: %s ---\n", i + 1, DataList[i].c_str());
        
//         // 加载体积模型
//         VolumeRender v(RelativePath + DataList[i]);
//         vector<DisneyDescriptor> ModelData;

//         // 论文细节：单模型应用多组随机参数
//         // 我们将 CountPerModel 分成几批，每批使用不同的随机光照/浓度
//         int Batches = 50; 
//         int SamplesPerBatch = CountPerModel / Batches;

//         for (int b = 0; b < Batches; b++) {
//             // 随机参数设计
//             float currentAlpha = lerp(DensityMin[i], DensityMax[i], hash1()); // 密度缩放
//             float currentG = lerp(0.2f, 0.9f, hash1());                      // 相位函数
//             float3 randomLightDir = normalize(hash31sphere());             // 随机光照

//             vector<SamplePoint> BatchSamples;
//             GetDesiredCountSample(v, BatchSamples, SamplesPerBatch, currentAlpha, currentAlpha); // 固定本批次 Alpha

//             // 转换并进行路径追踪 (Path Tracing) 获取真值
//             // ... (提取 Position, ViewDir, LightDir 等向量) ...
            
//             float3 LightColor = { 1.0, 1.0, 1.0 };
//             vector<float3> Radiances = v.GetSamples(SampleAlphas, SampleOris, SampleDirs, SampleLDirs, SampleGs, SampleScatters, LightColor, 512, 1024);

//             // 5. 提取描述符（包含方差）并保存
//             float gap = 0.25f / 1024.0f;
//             for (int k = 0; k < Radiances.size(); k++) {
//                 // GetDisneyDesc 内部已经包含了你写的 Variance 采样逻辑
//                 DisneyDescriptor desc = GetDisneyDesc(v, SampleOris[k] + float3{0.5f, 0.5f, 0.5f}, SampleDirs[k], SampleLDirs[k], SampleAlphas[k], gap);
//                 desc.Radiance = Radiances[k].x;

//                 // 立即写入文件，避免内存溢出
//                 WriteSampleToCSV(outfile, desc);
//                 Computed++;
//             }
//             printf("Progress: %.2f%% (%d samples)\n", 100.0f * Computed / CountAll, Computed);
//         }
//     }
//     outfile.close();
//     return 0;
// }
// #include "volume.hpp"
// #include "camera.hpp"
// #include "sample_method.hpp"
// #include "GUI.hpp"
// #include "core/perlin_noise.cuh" 
// #include <chrono>
// #include <random>
// #include <iomanip>
// #include <vector>
// #include <iostream>
// #include <fstream>

// using std::vector;
// using std::string;

// // 采样点结构体
// class SamplePoint {
// public:
//     float3 Position;
//     float3 ViewDir;
//     float3 LightDir;
//     float Alpha;
//     float g;

//     SamplePoint(float3 p, float3 v, float3 l, float a, float g_) 
//         : Position(p), ViewDir(v), LightDir(l), Alpha(a), g(g_) {}
// };

// // 描述符结构体：包含密度和方差
// class DisneyDescriptor {
// public:
//     static const size_t SIZE_X = 5;
//     static const size_t SIZE_Y = 5;
//     static const size_t SIZE_Z = 9;
//     static const size_t LAYER_SIZE = SIZE_Z * SIZE_Y * SIZE_X;
//     static const size_t LAYERS_CNT = 10;

//     class Layer {
//     public:
//         float density[LAYER_SIZE];
//         float variance[LAYER_SIZE]; 
//     };

//     Layer layers[LAYERS_CNT];
//     float Gamma = 0.0f;
//     float Radiance = 0.0f;
// };

// // Host端随机数生成
// float hash1() {
//     static thread_local std::mt19937 gen(std::random_device{}());
//     static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
//     return dist(gen);
// }

// float3 hash31sphere() {
//     float theta = 2 * 3.1415926f * hash1();
//     float phi = acos(2 * hash1() - 1.0f);
//     return float3{ cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi) };
// }

// float3 hash3box(float scale = 0.01) {
//     float3 sphere = hash31sphere();
//     float r = cbrt(hash1()) * scale;
//     return float3{ sphere.x * r, sphere.y * r, sphere.z * r };
// }

// // 辅助函数：将单个样本写入CSV
// void WriteSampleToCSV(std::ofstream& f, const DisneyDescriptor& CS) {
//     const int samplecount = DisneyDescriptor::LAYER_SIZE;
//     // 1. 写入 10 层密度描述符
//     for (int j = 0; j < 10; j++) {
//         for (int k = 0; k < samplecount; k++) {
//             f << std::fixed << std::setprecision(6) << CS.layers[j].density[k] << ",";
//         }
//     }
//     // 2. 写入 10 层方差描述符 (核心改进特征)
//     for (int j = 0; j < 10; j++) {
//         for (int k = 0; k < samplecount; k++) {
//             f << std::fixed << std::setprecision(6) << CS.layers[j].variance[k] << ",";
//         }
//     }
//     // 3. 写入 Gamma 参数和 Radiance 真值
//     f << CS.Gamma << "," << CS.Radiance << "\n";
// }

// // 修正：显式分量操作，避免 float3 转换报错
// float RayBoxOffset_(float3 p, float3 dir) {
//     float3 invDir = {1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z};
//     float3 bmax = { 0.4999f, 0.4999f, 0.4999f };
//     float3 t0 = { (-p.x - bmax.x) * invDir.x, (-p.y - bmax.y) * invDir.y, (-p.z - bmax.z) * invDir.z };
//     float3 t1 = { (-p.x + bmax.x) * invDir.x, (-p.y + bmax.y) * invDir.y, (-p.z + bmax.z) * invDir.z };
//     float tmi = fmaxf(fminf(t0.x, t1.x), fmaxf(fminf(t0.y, t1.y), fminf(t0.z, t1.z)));
//     float tma = fminf(fmaxf(t0.x, t1.x), fminf(fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z)));
//     return tma >= tmi ? fmaxf(tmi, 0.0f) : -1.0f;
// }

// float RayBoxDistance_(float3 p, float3 dir) {
//     float3 invDir = {1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z};
//     float3 bmax = { 0.5f, 0.5f, 0.5f };
//     float3 t0 = { (-p.x - bmax.x) * invDir.x, (-p.y - bmax.y) * invDir.y, (-p.z - bmax.z) * invDir.z };
//     float3 t1 = { (-p.x + bmax.x) * invDir.x, (-p.y + bmax.y) * invDir.y, (-p.z + bmax.z) * invDir.z };
//     float tma = fminf(fmaxf(t0.x, t1.x), fminf(fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z)));
//     return tma;
// }

// // 确定下一个散射点
// bool DeterminateNextVertex(VolumeRender& CurrentVolume, float alpha, float g, float3 pos, float3 dir, float dis, float3* nextPos, float3* nextDir) {
//     float SMax = CurrentVolume.max_density * alpha;
//     float t = 0;
//     while (t < 1000.0f) {
//         float rk = hash1();
//         t -= logf(1.0f - rk) / SMax;
//         if (t > dis) return false;
//         float3 currentP = { pos.x + dir.x * t, pos.y + dir.y * t, pos.z + dir.z * t };
//         float density = CurrentVolume.DensityAtPosition(0, currentP);
//         if ((density * alpha) / SMax > hash1()) break;
//     }
//     *nextDir = SampleHenyeyGreenstein(hash1(), hash1(), dir, g);
//     *nextPos = { pos.x + dir.x * t, pos.y + dir.y * t, pos.z + dir.z * t };
//     return true;
// }

// void MeanFreePathSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples, float3 ori, float3 dir, float3 lightDir, int maxcount, float alpha, float g) {
//     float dis = RayBoxOffset_(ori, dir);
//     if (dis < 0) return;
//     float3 samplePos = { ori.x + dir.x * dis, ori.y + dir.y * dis, ori.z + dir.z * dis };
//     float3 rayDir = dir;
//     for (int i = 0; i < 4; i++) {
//         float3 nPos, nDir;
//         float dLimit = RayBoxDistance_(samplePos, rayDir);
//         if (!DeterminateNextVertex(CurrentVolume, alpha, g, samplePos, rayDir, dLimit, &nPos, &nDir)) return;
//         if (Samples.size() >= maxcount) return;
        
//         float3 storedPos = nPos;
//         if (hash1() > 0.5) {
//             float3 box = hash3box(0.0078f);
//             storedPos.x += box.x; storedPos.y += box.y; storedPos.z += box.z;
//         }
//         Samples.push_back(SamplePoint(storedPos, hash1() > 0.25 ? dir : rayDir, lightDir, alpha, g));
//         samplePos = nPos; rayDir = nDir;
//     }
// }

// void GetDesiredCountSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples, int Count, float d_min, float d_max) {
//     Samples.clear();
//     while (Samples.size() < Count) {
//         float3 ori = hash31sphere();
//         ori.x *= 3.0f; ori.y *= 3.0f; ori.z *= 3.0f;
//         float3 sphere = hash31sphere();
//         float3 dir = normalize(float3{sphere.x - ori.x/3.0f, sphere.y - ori.y/3.0f, sphere.z - ori.z/3.0f});
//         float3 ldir = hash31sphere();
//         float Alpha = d_min + (d_max - d_min) * hash1();
//         MeanFreePathSample(CurrentVolume, Samples, ori, dir, ldir, Count, Alpha, 0.857f);
//     }
// }

// // 修正：GetDisneyDesc 彻底解决 float3 运算报错问题
// DisneyDescriptor GetDisneyDesc(VolumeRender& volume, float3 uv, float3 v, float3 s, float alpha, float descSizeAtLevel0) {
//     DisneyDescriptor descriptor;
//     v = normalize(v);
//     float3 eZ = normalize(s);
//     float3 eX = normalize(cross(eZ, v));
//     float3 eY = cross(eX, eZ);
//     descriptor.Gamma = acosf(fmaxf(-1.0f, fminf(1.0f, v.x * eZ.x + v.y * eZ.y + v.z * eZ.z)));

//     for (size_t layerId = 0; layerId < DisneyDescriptor::LAYERS_CNT; layerId++) {
//         int mipLevel = (int)layerId;
//         uint32_t sampleId = 0;
//         float currentScale = 0.5f * descSizeAtLevel0 * powf(2.0f, (float)mipLevel);

//         for (int z = -2; z <= 6; z++) {
//             for (int y = -2; y <= 2; y++) {
//                 for (int x = -2; x <= 2; x++) {
//                     float3 offset;
//                     offset.x = (eX.x * x + eY.x * y + eZ.x * z) * currentScale;
//                     offset.y = (eX.y * x + eY.y * y + eZ.y * z) * currentScale;
//                     offset.z = (eX.z * x + eY.z * y + eZ.z * z) * currentScale;

//                     float3 samplePos = { uv.x + offset.x, uv.y + offset.y, uv.z + offset.z };
                    
//                     descriptor.layers[layerId].density[sampleId] = volume.DensityAtUV(mipLevel, samplePos) * alpha / 64.0f;
//                     descriptor.layers[layerId].variance[sampleId] = volume.VarianceAtUV(mipLevel, samplePos);
//                     sampleId++;
//                 }
//             }
//         }
//     }
//     return descriptor;
// }

// int main() {
//     string DataPath = "./Output/";
//     string DataName = "TrainingData_5M.csv";
//     string RelativePath = "./Data/";
//     system("mkdir -p ./Output/");

//     vector<string> DataList;
//     vector<float> DensityMin, DensityMax;

//     // 自动扫描 33 个模型
//     for(int i = 0; i < 26; i++) {
//         DataList.push_back("cloud_" + std::to_string(i) );
//         DensityMin.push_back(0.5f); DensityMax.push_back(10.0f);
//     }
//     for(int i = 0; i < 7; i++) {
//         DataList.push_back("object_" + std::to_string(i) );
//         DensityMin.push_back(1.0f); DensityMax.push_back(40.0f);
//     }

//     const int CountAll = 330; 
//     int CountPerModel = CountAll / DataList.size();
//     int Computed = 0;

//     std::ofstream outfile(DataPath + DataName);
//     outfile << "# 5M Samples | Density + Variance | 10 Layers" << std::endl;

//     for (int i = 0; i < DataList.size(); i++) {
//         printf("\nProcessing [%d/33]: %s\n", i + 1, DataList[i].c_str());
//         VolumeRender v(RelativePath + DataList[i]);
        
//         int Batches = 50; 
//         int SamplesPerBatch = CountPerModel / Batches;

//         for (int b = 0; b < Batches; b++) {
//             float currentAlpha = DensityMin[i] + (DensityMax[i] - DensityMin[i]) * hash1();
//             float currentG = 0.2f + 0.7f * hash1();
//             float3 randomLightDir = normalize(hash31sphere());

//             vector<SamplePoint> BatchSamples;
//             GetDesiredCountSample(v, BatchSamples, SamplesPerBatch, currentAlpha, currentAlpha);

//             vector<float3> SOris, SDirs, SLDirs;
//             vector<float> SAlphas, SGs, SScatters;
//             for (auto& s : BatchSamples) {
//                 SOris.push_back(s.Position); SDirs.push_back(s.ViewDir);
//                 SLDirs.push_back(randomLightDir); SAlphas.push_back(currentAlpha);
//                 SGs.push_back(currentG); SScatters.push_back(1.0f);
//             }

//             float3 LightColor = { 1.0, 1.0, 1.0 };
//             vector<float3> Radiances = v.GetSamples(SAlphas, SOris, SDirs, SLDirs, SGs, SScatters, LightColor, 512, 1024);

//             float gap = 0.25f / 1024.0f;
//             for (int k = 0; k < Radiances.size(); k++) {
//                 float3 offsetUV = { SOris[k].x + 0.5f, SOris[k].y + 0.5f, SOris[k].z + 0.5f };
//                 DisneyDescriptor desc = GetDisneyDesc(v, offsetUV, SDirs[k], SLDirs[k], SAlphas[k], gap);
//                 desc.Radiance = Radiances[k].x;
//                 WriteSampleToCSV(outfile, desc);
//                 Computed++;
//             }
//             printf("\rOverall Progress: %.2f%% (%d/%d)", 100.0f * Computed / CountAll, Computed, CountAll);
//             std::cout.flush();
//         }
//         outfile.flush();
//     }
//     outfile.close();
//     return 0;
// }