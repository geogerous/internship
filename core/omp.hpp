#pragma once

#include <functional>
#include <vector>
#include <string>
#include <cuda_runtime.h> // Ensure float2 is defined

using namespace std;

typedef function<float(int x, int y, int z, float u, float v, float w)> FillFunc;
typedef function<float2(int x, int y, int z, float u, float v, float w)> FillFunc2;

void ParallelFill(float* datas, int resolution, FillFunc fillFunc);
void ParallelFill(float* target1, float* target2, int resolution, FillFunc2 fillFunc);

typedef function<float(int index)> LoopFunc;

void ParallelFor(float* result, int length, LoopFunc func);