#ifndef CUDA_SPLATTER_BACKWARD_H_INCLUDED
#define CUDA_SPLATTER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{	
	void splat(
		const dim3 grid, 
		const dim3 block,
		const int num_computed,
		const int num_phases,
		const int num_points,
		const int resolution, 
		const float* means3D,
		const float* normals,
		const float* scales,
		const float* gamma_m,
		const float* features,
		const uint2* phases_ranges,
		const uint32_t* phases_tiles,
		const uint32_t* point_list,
		const uint32_t* tiles_list,
		const int* grad_flags,
		const float* out_sums,
		const float* out_sdfs,
		const float* out_feat,
		const float* out_lap_aux,
		const float* dL_dout_sums,
		const float* dL_dout_sdfs,
		const float* dL_dout_feat,
		float* dL_dmeans3D,
		float* dL_dnormals,
		float* dL_dscales,
		float* dL_dgamma_m,
		float* dL_dfeatures,
		float laplacian_lambda,
		const float sp_alpha,
		const int sp_seed);

}


#endif