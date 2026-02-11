#include "omp.hpp"
#include <fstream>

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
				float res = fillFunc(i, j, k, u, v, w);
				int index = (i * resolution + j) * resolution + k;
				datas[index] = res;
			}
}

void ParallelFill(float* target1, float* target2, int resolution, FillFunc2 fillFunc) {
    #pragma omp parallel for
    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                float u = (k + 0.5f) / resolution;
                float v = (j + 0.5f) / resolution;
                float w = (i + 0.5f) / resolution;

                // Call the lambda function returning float2
                float2 result = fillFunc(k, j, i, u, v, w);

                int index = (i * resolution + j) * resolution + k;

                // Write results to the two target arrays
                target1[index] = result.x; // Mean
                target2[index] = result.y; // Variance
            }
        }
    }
}

void ParallelFor(float* result, int length, LoopFunc func) {
	#pragma omp parallel for
	for (int i = 0; i < length; i++) {
		result[i] = func(i);
	}	
}