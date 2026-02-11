#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "core/perlin_noise.cuh" // 确保包含你项目中的噪声函数

using namespace std;

// 确保 perlin_noise 在 host 端可用。如果报错，请在定义处加 __host__
// 或者是简单实现一个常用的噪声函数

struct VoxelConfig {
    string name;
    float frequency;
    float threshold;
    float amplitude;
    bool is_fractured; // 是否为极破碎形状
};

void SaveAsTxt(const string& path, const vector<float>& data, int res) {
    ofstream f(path);
    if (!f.is_open()) return;
    for (float v : data) {
        f << v << " ";
    }
    f.close();
}

// 核心生成逻辑
void GenerateModel(VoxelConfig config, int res, string saveDir) {
    vector<float> volume(res * res * res);
    cout << "Generating: " << config.name << "..." << endl;

    for (int z = 0; z < res; z++) {
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                float u = (float)x / res, v = (float)y / res, w = (float)z / res;
                float density = 0.0f;

                if (!config.is_fractured) {
                    // 【普通云彩类】：分形布朗运动 (fBm)
                    float f = config.frequency;
                    density = perlin_noise(u * f, v * f, w * f) * 1.0f;
                    density += perlin_noise(u * f * 2, v * f * 2, w * f * 2) * 0.5f;
                    density += perlin_noise(u * f * 4, v * f * 4, w * f * 4) * 0.25f;
                    density = (density + 1.0f) * 0.5f; // 映射到 0-1
                    density = max(0.0f, (density - config.threshold) * config.amplitude);
                } else {
                    // 【非云/破碎类】：极高频噪声，模拟细碎固体
                    // 这部分数据是验证“方差改进”的关键
                    density = perlin_noise(u * config.frequency, v * config.frequency, w * config.frequency);
                    density = (density > config.threshold) ? 1.0f : 0.0f;
                }
                
                // 应用球形 Mask 确保模型在包围盒中心
                float dx = u - 0.5f, dy = v - 0.5f, dz = w - 0.5f;
                float dist = sqrt(dx*dx + dy*dy + dz*dz);
                if (dist > 0.45f) density = 0.0f;

                volume[x + y * res + z * res * res] = density;
            }
        }
    }
    SaveAsTxt(saveDir + config.name + ".txt", volume, res);
}

int main() {
    int res =128;
    string saveDir = "./Data/";
    system("mkdir -p ./Data/");

    // 1. 生成 26 个云形状 (训练/验证 + 测试)
    for (int i = 0; i < 26; i++) {
        VoxelConfig cloud;
        cloud.name = "cloud_" + to_string(i);
        cloud.frequency = 4.0f + (i * 0.5f); // 逐渐增加复杂度
        cloud.threshold = 0.4f + (float)(rand() % 20) / 100.0f;
        cloud.amplitude = 2.0f;
        cloud.is_fractured = false;
        GenerateModel(cloud, res, saveDir);
    }

    // 2. 生成 7 个非云形状 (主要是碎片化形状，用于方差测试)
    for (int i = 0; i < 7; i++) {
        VoxelConfig obj;
        obj.name = "object_" + to_string(i);
        obj.frequency = 32.0f + (i * 10.0f); // 极高频
        obj.threshold = 0.1f;
        obj.is_fractured = true;
        GenerateModel(obj, res, saveDir);
    }

    cout << "All 33 models generated in ./Data/" << endl;
    return 0;
}