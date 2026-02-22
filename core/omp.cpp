#include "omp.hpp"
#include <fstream>
#include <cuda_runtime.h> // For float2

void ParallelFill(float* datas, int resolution, FillFunc fillFunc) {
	#pragma omp parallel for
	for (int i = 0; i < resolution; i++)
		for (int j = 0; j < resolution; j++)
			for (int k = 0; k < resolution; k++)
			{
				float u, v, w;
				u = (i + 0.5) / resolution;
				v = (j + 0.5) / resolution;
				w = (k + 0.5) / resolution;
				datas[(i * resolution + j) * resolution + k] = fillFunc(i, j, k, u, v, w);
			}
}

void ParallelFill(float* target1, float* target2, int resolution, FillFunc2 fillFunc) {
	#pragma omp parallel for
	for (int i = 0; i < resolution; i++)
		for (int j = 0; j < resolution; j++)
			for (int k = 0; k < resolution; k++)
			{
				float u, v, w;
				u = (i + 0.5f) / resolution;
				v = (j + 0.5f) / resolution;
				w = (k + 0.5f) / resolution;
				float2 res = fillFunc(i, j, k, u, v, w);
				int index = (i * resolution + j) * resolution + k;
				target1[index] = res.x;
				target2[index] = res.y;
			}
}

void ParallelFor(float* result, int length, LoopFunc func) {
	#pragma omp parallel for
	for (int i = 0; i < length; i++) {
		result[i] = func(i);
	}	
}