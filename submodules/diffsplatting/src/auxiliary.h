#ifndef CUDA_SPLATTER_AUXILIARY_H_INCLUDED
#define CUDA_SPLATTER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <math.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define NUM_WARPS (BLOCK_SIZE/32)


__forceinline__ __device__ void getBox3D(const float3 p, const float scale, const float gamma_m, uint3& box_min, uint3& box_max, int resolution, dim3 grid)
{	
	//float radius = 2.f * scale;
	// float gamma_eps_scale = 0.1f;
	// float gamma_m = 2.f;
	// float gamma_kappa = 2.f;
	// float eps = fmaxf(scale * gamma_eps_scale, 1e-6f);
	// float support = gamma_m * gamma_kappa * eps;
	float support = 2.f * scale;
	float radius = fmaxf(support, 1e-6f);
	box_min = {
		min(grid.x, max((int)0, (int)((p.x - radius) * resolution / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - radius) * resolution / BLOCK_Y))),
		min(grid.z, max((int)0, (int)((p.z - radius) * resolution / BLOCK_Z)))
	};
	box_max = {
		min(grid.x, max((int)0, (int)(((p.x + radius) * resolution + BLOCK_X - 1) / BLOCK_X + 1))),
		min(grid.y, max((int)0, (int)(((p.y + radius) * resolution + BLOCK_Y - 1) / BLOCK_Y + 1))),
		min(grid.z, max((int)0, (int)(((p.z + radius) * resolution + BLOCK_Z - 1) / BLOCK_Z + 1))),
	};

}


__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif