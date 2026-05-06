#include <iostream>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
namespace cg = cooperative_groups;

#include "forward.h"
#include "auxiliary.h"

// ================= Histogram logging =================

constexpr int HIST_BINS = 64;

// log10 ranges
constexpr float LOG_SCALE_MIN = -6.0f;   // 1e-6
constexpr float LOG_SCALE_MAX =  0.0f;   // 1

constexpr float LOG_THETA_MIN = -6.0f;   // 1e-6
constexpr float LOG_THETA_MAX =  0.0f;   // 1

// gamma_m is not log-scaled in the kernel, so use a linear range.
// Pick a wide-enough default range for debugging; adjust if you expect gamma_m outside it.
constexpr float GAMMA_M_MIN = 0.0f;
constexpr float GAMMA_M_MAX = 5.0f;

__device__ __forceinline__
int log_bin(float x, float log_min, float log_max)
{
    float lx = log10f(fmaxf(x, 1e-20f));
    int bin = (int)((lx - log_min) / (log_max - log_min) * HIST_BINS);
    return max(0, min(HIST_BINS - 1, bin));
}

__device__ __forceinline__
int lin_bin(float x, float x_min, float x_max)
{
    float t = (x - x_min) / (x_max - x_min);
    int bin = (int)(t * HIST_BINS);
    return max(0, min(HIST_BINS - 1, bin));
}

// γ(s) = u^{2m} v with u = 1 - s/(mk), v = 2s/k + 1; radial ∇²γ = γ'' + (2/s)γ'
__device__ __forceinline__
void iml_gamma_laplacian(float s, float k_raw, float m_in,
	float& gamma, float& dgamma_ds, float& lap_gamma)
{
	float m = fminf(fmaxf(m_in, 1.0f), 3.0f);
	float k = fmaxf(k_raw, 1e-6f);
	float cutoff = fmaxf(m * k, 1e-6f);

	gamma = 0.f;
	dgamma_ds = 0.f;
	lap_gamma = 0.f;

	if (s > cutoff)
		return;

	float inv_cut = 1.f / cutoff;
	float u = fmaxf(1.f - s * inv_cut, 0.f);
	float v = 2.f * s / k + 1.f;
	gamma = powf(u, 2.f * m) * v;

	float u_pow_2m1 = powf(u, 2.f * m - 1.f);
	float coeff = -2.f * (2.f * m + 1.f) / (m * k * k);
	dgamma_ds = coeff * s * u_pow_2m1;

	float pow_exp = 2.f * m - 2.f;
	float u_pow_2m2 = (pow_exp > 0.f) ? powf(u, pow_exp) : 1.f;
	// float bracket = 5.f - 2.f * s * (m + 2.f) / (m * k);
	float bracket = 3.f - 2.f * s * (m + 1.f) / (m * k);
	lap_gamma = coeff * u_pow_2m2 * bracket;
}

__device__ __forceinline__ uint32_t xorshift32(uint32_t x)
{
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

__device__ __forceinline__ float u01(uint32_t& state)
{
	state = xorshift32(state);
	return (state & 0x00FFFFFF) / 16777216.0f;
}

__device__ __forceinline__ float reflect01(float x)
{
	// Reflect into [0,1] using period-2 reflection.
	float t = fmodf(x, 2.0f);
	if (t < 0.f) t += 2.0f;
	return (t > 1.0f) ? (2.0f - t) : t;
}

// __device__ __forceinline__ void jitter_query_pos(float& x, float& y, float& z, float alpha, uint32_t seed)
// {
// 	if (alpha <= 0.f) return;

// 	uint32_t state = seed;
// 	// 3D direction from normal distribution via hash RNG
// 	float dx = u01(state) * 2.f - 1.f;
// 	float dy = u01(state) * 2.f - 1.f;
// 	float dz = u01(state) * 2.f - 1.f;
// 	float n2 = dx*dx + dy*dy + dz*dz;
// 	if (n2 < 1e-12f) { dx = 1.f; dy = 0.f; dz = 0.f; n2 = 1.f; }
// 	float invn = rsqrtf(n2);
// 	dx *= invn; dy *= invn; dz *= invn;

// 	// uniform in ball: r = alpha * U^{1/3}
// 	float r = alpha * cbrtf(fmaxf(u01(state), 1e-12f));
// 	x = reflect01(x + dx * r);
// 	y = reflect01(y + dy * r);
// 	z = reflect01(z + dz * r);
// }


// Box-Muller transform: consumes 2 RNG steps, returns one N(0,1) sample.
__device__ __forceinline__ float box_muller_sample(uint32_t& state)
{
    float u1 = fmaxf(u01(state), 1e-7f);   // guard against log(0)
    float u2 = u01(state);
    return sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265358979f * u2);
}

// Perturb query position with isotropic Gaussian noise: delta ~ N(0, alpha^2 * I)
// Matches Eq. 2 of Ling et al. 2025 exactly.
__device__ __forceinline__ void jitter_query_pos(float& x, float& y, float& z, float alpha, uint32_t seed)
{
    if (alpha <= 0.f) return;

    uint32_t state = seed;
    // 6 RNG steps total (2 per axis via Box-Muller)
    float dx = alpha * box_muller_sample(state);
    float dy = alpha * box_muller_sample(state);
    float dz = alpha * box_muller_sample(state);

    // Period-2 reflection boundary (Eq. 3, Ling et al.)
    x = reflect01(x + dx);
    y = reflect01(y + dy);
    z = reflect01(z + dz);
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	int num_points, 
	int resolution,
	const dim3   grid,
	const float* means3D,
	const float* scales,
	const float* gamma_m,
	const float* features,
	uint2* bbox,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_points)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	tiles_touched[idx] = 0;

	// Transform point by projecting
	float3 p_orig = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };

	// Compute extent in 3D space
	uint3 box_min, box_max;
	getBox3D(p_orig, scales[idx], gamma_m[idx], box_min, box_max, resolution, grid);
	if ((box_max.x - box_min.x) + (box_max.y - box_min.y) + (box_max.z - box_min.z) == 0)
		return;

	// Store some useful helper data for the next steps.
	bbox[idx * 3 + 0] = {box_min.x, box_max.x}; // x
	bbox[idx * 3 + 1] = {box_min.y, box_max.y}; // y
	bbox[idx * 3 + 2] = {box_min.z, box_max.z}; // z
	tiles_touched[idx] = (box_max.x-box_min.x)*(box_max.y-box_min.y)*(box_max.z - box_min.z);
	
}


void FORWARD::preprocess(
	int num_points, 
	int resolution,
	const dim3 grid,
	const float* means3D,
	const float* scales,
	const float* gamma_m,
	const float* features,
	uint2* bbox,
	uint32_t* tiles_touched)
{
	preprocessCUDA<NUM_CHANNELS> << <(num_points + 255) / 256, 256 >> > (
		num_points, 
		resolution,
		grid,
		means3D,
		scales,
		gamma_m,
		features,
		bbox,
		tiles_touched);
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
splatCUDA(
	int num_phases,
	int resolution,
	const float* __restrict__ means3D,
	const float* __restrict__ normals,
	const float* __restrict__ scales,
	const float* __restrict__ gamma_m,
	const float* __restrict__ features,
	const uint2* __restrict__ spa_ranges,
	const uint32_t* __restrict__ idx_tile,
	const uint32_t* __restrict__ point_list,
	float* __restrict__ out_sums,
	float* __restrict__ out_sdfs,
	float* __restrict__ out_feat,
	float* __restrict__ out_lap_aux,
	uint32_t* __restrict__ hist_scale,
    uint32_t* __restrict__ hist_theta,
	uint32_t* __restrict__ hist_gamma_m,
	float sp_alpha,
	int sp_seed)

{	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	int phase_id  = block.group_index().x;
	int thread_id = block.thread_index().x;

	uint32_t tile_id = idx_tile[phase_id];
	uint2    range   = spa_ranges[phase_id];

	uint32_t num_blocks_x = (resolution + BLOCK_X - 1) / BLOCK_X;
	uint32_t num_blocks_y = (resolution + BLOCK_Y - 1) / BLOCK_Y;

	int block_idz = (int)( tile_id / (num_blocks_y*num_blocks_x));
	int block_idy = (int)((tile_id - block_idz*(num_blocks_y*num_blocks_x)) / num_blocks_x);
	int block_idx = (int)( tile_id - block_idz*(num_blocks_y*num_blocks_x) - block_idy*num_blocks_x);

	int thread_idz = (int)(thread_id / (BLOCK_Y*BLOCK_X));
	int thread_idy = (int)((thread_id - thread_idz*(BLOCK_Y*BLOCK_X)) / BLOCK_X);
	int thread_idx = (int)( thread_id - thread_idz*(BLOCK_Y*BLOCK_X) - thread_idy*BLOCK_X);	

	uint3 cell_id_min = {block_idx * BLOCK_X, block_idy * BLOCK_Y, block_idz * BLOCK_Z};
	uint3 cell_id_max = {min(cell_id_min.x+BLOCK_X, resolution), min(cell_id_min.y+BLOCK_Y, resolution), min(cell_id_min.z+BLOCK_Z, resolution) };

	uint3    cell_id      = { cell_id_min.x + thread_idx, cell_id_min.y + thread_idy, cell_id_min.z + thread_idz };
	uint32_t cell_id_flat = cell_id.z * resolution * resolution + cell_id.y * resolution + cell_id.x;

	float cell_x = (float)cell_id.x / (resolution-1);
	float cell_y = (float)cell_id.y / (resolution-1);
	float cell_z = (float)cell_id.z / (resolution-1);

	// Stochastic preconditioning: jitter query position q per voxel deterministically from (seed, voxel id).
	// NOTE: must match backward kernel jitter exactly.
	if (sp_alpha > 0.f)
	{
		uint32_t seed = (uint32_t)sp_seed ^ (uint32_t)cell_id_flat * 9781u;
		jitter_query_pos(cell_x, cell_y, cell_z, sp_alpha, seed);
	}

	bool inside = cell_id.x < resolution && cell_id.y < resolution && cell_id.z < resolution;
	bool done = !inside;

	int toDo = range.y - range.x;

	const int num_cell = resolution * resolution * resolution;

	float theta_sum = 0;
	float theta_proj_sum = 0;
	float theta_feat_sum[CHANNELS] = { 0 };

	float acc_lapB = 0.f;
	float acc_lapA = 0.f;
	float acc_gBx = 0.f, acc_gBy = 0.f, acc_gBz = 0.f;
	float acc_gAx = 0.f, acc_gAy = 0.f, acc_gAz = 0.f;

	__shared__ int    collected_id[LEN_PHASE];
	__shared__ float  collected_scales[LEN_PHASE];
	__shared__ float  collected_gamma_m[LEN_PHASE];
	__shared__ float3 collected_means3D[LEN_PHASE];
	__shared__ float3 collected_normals[LEN_PHASE];

	if (range.x + thread_id < range.y)
	{
		int coll_id = point_list[range.x + thread_id];
		collected_id[thread_id]      = coll_id;
		collected_scales[thread_id]  = scales[coll_id];
		collected_gamma_m[thread_id] = gamma_m[coll_id];
		collected_means3D[thread_id] = {means3D[coll_id*3], means3D[coll_id*3+1], means3D[coll_id*3+2]};
		collected_normals[thread_id] = {normals[coll_id*3], normals[coll_id*3+1], normals[coll_id*3+2]};
	}
	block.sync();

	int valid_point_num = 0;
	for (int i = 0; i < toDo; i++)
	{	
		float3 point  = collected_means3D[i];
		float3 normal = collected_normals[i];
		float  scale  = collected_scales[i];
		float  gamma_m_val = collected_gamma_m[i];
		// Clamp effective gamma_m used in the kernel to [1.0, 5.0]
		// gamma_m_val = fminf(fmaxf(gamma_m_val, 1.0f), 3.0f);
		gamma_m_val = fminf(fmaxf(gamma_m_val, 1.0f), 5.0f);


		float3 vec   = {cell_x-point.x, cell_y-point.y, cell_z-point.z};
		float  proj  = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;
		float  norm  = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
		float kappa_eps = fmaxf(scale, 1e-6f);
		float dist = sqrtf(norm + 1e-12f);

		float gamma, dg_ds, lap_g;
		iml_gamma_laplacian(dist, kappa_eps, gamma_m_val, gamma, dg_ds, lap_g);
		float theta = gamma;

		if (theta < 1e-4f)
			continue;
		
		int b_scale = log_bin(scale, LOG_SCALE_MIN, LOG_SCALE_MAX);
		int b_theta = log_bin(theta, LOG_THETA_MIN, LOG_THETA_MAX);

		atomicAdd(&hist_scale[b_scale], 1);
		atomicAdd(&hist_theta[b_theta], 1);

		valid_point_num += 1;

		float inv_s = 1.f / dist;
		float rn = proj * inv_s;

		acc_lapB += lap_g;
		acc_lapA += lap_g * proj + 2.f * dg_ds * rn;
		acc_gBx += dg_ds * vec.x * inv_s;
		acc_gBy += dg_ds * vec.y * inv_s;
		acc_gBz += dg_ds * vec.z * inv_s;
		acc_gAx += dg_ds * vec.x * inv_s * proj + gamma * normal.x;
		acc_gAy += dg_ds * vec.y * inv_s * proj + gamma * normal.y;
		acc_gAz += dg_ds * vec.z * inv_s * proj + gamma * normal.z;

		theta_sum      += theta;
		theta_proj_sum += theta * proj;
		for (int ch = 0; ch < CHANNELS; ch++)
			theta_feat_sum[ch] += theta * features[collected_id[i]*CHANNELS+ch];
	}

	if ((theta_sum >= 1e-4f) && (inside) && (valid_point_num > 1))
	{	
		//  change valid_point_num > 1 to valid_point_num >= 1
		atomicAdd(&out_sums[cell_id_flat], theta_sum);
		atomicAdd(&out_sdfs[cell_id_flat], theta_proj_sum);
		for (int ch = 0; ch < CHANNELS; ch++)
			atomicAdd(&out_feat[(ch*resolution*resolution*resolution) + cell_id_flat], theta_feat_sum[ch]);

		atomicAdd(&out_lap_aux[0 * num_cell + cell_id_flat], acc_lapB);
		atomicAdd(&out_lap_aux[1 * num_cell + cell_id_flat], acc_lapA);
		atomicAdd(&out_lap_aux[2 * num_cell + cell_id_flat], acc_gBx);
		atomicAdd(&out_lap_aux[3 * num_cell + cell_id_flat], acc_gBy);
		atomicAdd(&out_lap_aux[4 * num_cell + cell_id_flat], acc_gBz);
		atomicAdd(&out_lap_aux[5 * num_cell + cell_id_flat], acc_gAx);
		atomicAdd(&out_lap_aux[6 * num_cell + cell_id_flat], acc_gAy);
		atomicAdd(&out_lap_aux[7 * num_cell + cell_id_flat], acc_gAz);
	}

	// Histogram logging: count gamma_m values used in this phase.
	// This is for debugging only (used by the host-side CSV dump).
	if (hist_gamma_m)
	{
		for (int i = 0; i < valid_point_num; i++)
		{
			float gm = collected_gamma_m[i];
			int bin = lin_bin(gm, GAMMA_M_MIN, GAMMA_M_MAX);
			atomicAdd(&hist_gamma_m[bin], 1u);
		}
	}
}

void FORWARD::splat(
	const dim3 grid, 
	const dim3 block,
	int num_phases,
	int resolution, 
	const float* means3D,
	const float* normals,
	const float* scales,
	const float* gamma_m,
	const float* features,
	const uint2* spa_ranges,
	const uint32_t* idx_tile,
	const uint32_t* point_list,
	float* out_sums,
	float* out_sdfs,
	float* out_feat,
	float* out_lap_aux,
	float sp_alpha,
	int sp_seed)
{	
	static uint32_t* d_hist_scale = nullptr;
	static uint32_t* d_hist_theta = nullptr;
	static uint32_t* d_hist_gamma_m = nullptr;

	if (!d_hist_scale) {
		cudaMalloc(&d_hist_scale, HIST_BINS * sizeof(uint32_t));
		cudaMalloc(&d_hist_theta, HIST_BINS * sizeof(uint32_t));
		cudaMalloc(&d_hist_gamma_m, HIST_BINS * sizeof(uint32_t));
	}

	cudaMemset(d_hist_scale, 0, HIST_BINS * sizeof(uint32_t));
	cudaMemset(d_hist_theta, 0, HIST_BINS * sizeof(uint32_t));
	cudaMemset(d_hist_gamma_m, 0, HIST_BINS * sizeof(uint32_t));


	splatCUDA<NUM_CHANNELS> << <num_phases, block.x*block.y*block.z >> > (
		num_phases,
		resolution,
		means3D,
		normals,
		scales,
		gamma_m,
		features,
		spa_ranges,
		idx_tile,
		point_list,
		out_sums,
		out_sdfs,
		out_feat,
		out_lap_aux,
		d_hist_scale,
    	d_hist_theta,
		d_hist_gamma_m,
		sp_alpha,
		sp_seed);

	std::vector<uint32_t> h_scale(HIST_BINS);
	std::vector<uint32_t> h_theta(HIST_BINS);
	std::vector<uint32_t> h_gamma_m(HIST_BINS);

	cudaMemcpy(h_scale.data(), d_hist_scale,
			HIST_BINS * sizeof(uint32_t),
			cudaMemcpyDeviceToHost);

	cudaMemcpy(h_theta.data(), d_hist_theta,
			HIST_BINS * sizeof(uint32_t),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(h_gamma_m.data(), d_hist_gamma_m,
			HIST_BINS * sizeof(uint32_t),
			cudaMemcpyDeviceToHost);
	
	static int iter = 0;   // counts how many times splat() was called
	iter++;

	if (iter % 5000 != 0)
    return;
	std::string theta_fname = "/home/nandhana-u/Improved-IMLS-Splatting/gaussian_ranges/theta_hist_iter_" + std::to_string(iter) + ".csv";
	std::ofstream f_theta(theta_fname);
	f_theta << "log10_theta,count\n";

	for (int i = 0; i < HIST_BINS; i++) {
		float center =
			LOG_THETA_MIN +
			(i + 0.5f) * (LOG_THETA_MAX - LOG_THETA_MIN) / HIST_BINS;

		f_theta << std::fixed << std::setprecision(6)
				<< center << "," << h_theta[i] << "\n";
	}
	f_theta.close();
	std::string scale_fname ="/home/nandhana-u/Improved-IMLS-Splatting/gaussian_ranges/scale_hist_iter_" + std::to_string(iter) + ".csv";
	std::ofstream f_scale(scale_fname);
	f_scale << "log10_scale,count\n";

	for (int i = 0; i < HIST_BINS; i++) {
		float center =
			LOG_SCALE_MIN +
			(i + 0.5f) * (LOG_SCALE_MAX - LOG_SCALE_MIN) / HIST_BINS;

		f_scale << std::fixed << std::setprecision(6)
				<< center << "," << h_scale[i] << "\n";
	}
	f_scale.close();
	std::string gamma_m_fname ="/home/nandhana-u/Improved-IMLS-Splatting/gaussian_ranges/gamma_m_hist_iter_" + std::to_string(iter) + ".csv";
	std::ofstream f_gamma_m(gamma_m_fname);
	f_gamma_m << "gamma_m,count\n";
	uint64_t gamma_m_total = 0;
	double gamma_m_sum = 0.0;
	for (int i = 0; i < HIST_BINS; i++) {
		float center =
			GAMMA_M_MIN +
			(i + 0.5f) * (GAMMA_M_MAX - GAMMA_M_MIN) / HIST_BINS;
		f_gamma_m << std::fixed << std::setprecision(6)
				<< center << "," << h_gamma_m[i] << "\n";
		gamma_m_total += (uint64_t)h_gamma_m[i];
		gamma_m_sum += (double)center * (double)h_gamma_m[i];
	}
	f_gamma_m.close();

	// Representative gamma_m (weighted mean from histogram bins).
	if (gamma_m_total > 0)
	{
		double gamma_m_mean = gamma_m_sum / (double)gamma_m_total;
		std::cout << "[gamma_m] iter " << iter
				  << " mean≈" << std::fixed << std::setprecision(6) << gamma_m_mean
				  << " (hist range " << GAMMA_M_MIN << ".." << GAMMA_M_MAX << ")"
				  << std::endl;
	}

	


}


template <uint32_t CHANNELS>
__global__ void 
weightingCUDA(
	int resolution,
	float* __restrict__ out_sums,
	float* __restrict__ out_sdfs,
	float* __restrict__ out_feat,
	const float* __restrict__ out_lap_aux,
	float* __restrict__ out_sdfs_base,
	float laplacian_lambda)
{	
	auto thread_idx = cg::this_grid().thread_rank();

	int num_cell = resolution * resolution * resolution;
	if (thread_idx >= num_cell)
		return;

	float out_sum = out_sums[thread_idx];
	float out_sdf = out_sdfs[thread_idx];

	if (out_sum >= 1e-4f)
	{
		float SB = out_sum;
		float SA = out_sdf;
		float invB = 1.f / SB;
		float invB2 = invB * invB;

		float lapB = out_lap_aux[0 * num_cell + thread_idx];
		float lapA = out_lap_aux[1 * num_cell + thread_idx];
		float GBx = out_lap_aux[2 * num_cell + thread_idx];
		float GBy = out_lap_aux[3 * num_cell + thread_idx];
		float GBz = out_lap_aux[4 * num_cell + thread_idx];
		float GAx = out_lap_aux[5 * num_cell + thread_idx];
		float GAy = out_lap_aux[6 * num_cell + thread_idx];
		float GAz = out_lap_aux[7 * num_cell + thread_idx];

		float gradFx = (SB * GAx - SA * GBx) * invB2;
		float gradFy = (SB * GAy - SA * GBy) * invB2;
		float gradFz = (SB * GAz - SA * GBz) * invB2;

		float lapF = lapA * invB - SA * lapB * invB2
			- 2.f * (GBx * gradFx + GBy * gradFy + GBz * gradFz) * invB;

		float F = SA / SB;
		out_sdfs_base[thread_idx] = F;
		out_sdfs[thread_idx] = F + laplacian_lambda * lapF;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_feat[(ch*num_cell) + thread_idx] = out_feat[(ch*num_cell) + thread_idx] / out_sum;
	}
}

void FORWARD::weighting(
	int resolution,
	float* out_sums,
	float* out_sdfs,
	float* out_feat,
	const float* out_lap_aux,
	float* out_sdfs_base,
	float laplacian_lambda)
{	
	int num_cell = resolution * resolution * resolution;
	int num_blocks = (num_cell + 255) / 256;
	weightingCUDA<NUM_CHANNELS> << <num_blocks, 256 >> > (
		resolution,
		out_sums,
		out_sdfs,
		out_feat,
		out_lap_aux,
		out_sdfs_base,
		laplacian_lambda);
}


