#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

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
	float t = fmodf(x, 2.0f);
	if (t < 0.f) t += 2.0f;
	return (t > 1.0f) ? (2.0f - t) : t;
}

// __device__ __forceinline__ void jitter_query_pos(float& x, float& y, float& z, float alpha, uint32_t seed)
// {
// 	if (alpha <= 0.f) return;

// 	uint32_t state = seed;
// 	float dx = u01(state) * 2.f - 1.f;
// 	float dy = u01(state) * 2.f - 1.f;
// 	float dz = u01(state) * 2.f - 1.f;
// 	float n2 = dx*dx + dy*dy + dz*dz;
// 	if (n2 < 1e-12f) { dx = 1.f; dy = 0.f; dz = 0.f; n2 = 1.f; }
// 	float invn = rsqrtf(n2);
// 	dx *= invn; dy *= invn; dz *= invn;

// 	float r = alpha * cbrtf(fmaxf(u01(state), 1e-12f));
// 	x = reflect01(x + dx * r);
// 	y = reflect01(y + dy * r);
// 	z = reflect01(z + dz * r);
// }

// Matches forward.cu iml_gamma_laplacian; provides ∂²γ/∂s², ∂(∇²γ)/∂s, and ∂/∂k for scale backprop.
// γ(s) = u^(2m) v, u = max(1 - s/(mk), 0), v = 2s/k + 1; forward uses dg/ds = C s u^(2m-1),
// ∇²γ (radial) = C u^(2m-2) bracket with bracket = 3 - 2s(m+1)/(mk).
__device__ __forceinline__
void iml_gamma_laplacian_reverse(
	float s, float k_raw, float m_in,
	float gamma, float dg_ds, float lap_g_in,
	float& d2g_ds2, float& dlap_ds, float& dgamma_dk, float& d_dg_ds_dk, float& dlap_dk)
{
	float m = fminf(fmaxf(m_in, 1.0f), 3.0f);
	float k = fmaxf(k_raw, 1e-6f);
	float cutoff = fmaxf(m * k, 1e-6f);

	d2g_ds2 = 0.f;
	dlap_ds = 0.f;
	dgamma_dk = 0.f;
	d_dg_ds_dk = 0.f;
	dlap_dk = 0.f;

	if (s > cutoff)
		return;

	float inv_cut = 1.f / cutoff;
	float u = fmaxf(1.f - s * inv_cut, 0.f);
	if (u <= 0.f)
		return;

	float coeff = -2.f * (2.f * m + 1.f) / (m * k * k);
	float u_pow_2m1 = powf(u, 2.f * m - 1.f);
	float du_ds = -inv_cut;
	// d²γ/ds² from γ' = coeff * s * u^(2m-1)
	d2g_ds2 = coeff * ((2.f * m - 1.f) * s * powf(u, 2.f * m - 2.f) * du_ds + u_pow_2m1);

	float pow_exp = 2.f * m - 2.f;
	float u_pow_2m2 = (pow_exp > 0.f) ? powf(u, pow_exp) : 1.f;
	float bracket = 3.f - 2.f * s * (m + 1.f) / (m * k);
	float dbracket_ds = -2.f * (m + 1.f) / (m * k);
	if (pow_exp > 0.f)
		dlap_ds = coeff * ((2.f * m - 2.f) * powf(u, 2.f * m - 3.f) * du_ds * bracket + u_pow_2m2 * dbracket_ds);
	else
		dlap_ds = coeff * dbracket_ds;

	float v = 2.f * s / k + 1.f;
	// ∂u/∂k = s / (m k^2) since cutoff = m k
	float du_dk = s / (m * k * k);
	float dcoeff_dk = 4.f * (2.f * m + 1.f) / (m * powf(k, 3.f));
	float dv_dk = -2.f * s / (k * k);
	dgamma_dk = powf(u, 2.f * m) * dv_dk + 2.f * m * powf(u, 2.f * m - 1.f) * du_dk * v;
	d_dg_ds_dk = dcoeff_dk * s * u_pow_2m1 + coeff * s * (2.f * m - 1.f) * powf(u, 2.f * m - 2.f) * du_dk;

	float dbracket_dk = 2.f * s * (m + 1.f) / (m * k * k);
	float du_pow_2m2_dk = (pow_exp > 0.f) ? ((2.f * m - 2.f) * powf(u, 2.f * m - 3.f) * du_dk) : 0.f;
	dlap_dk = dcoeff_dk * u_pow_2m2 * bracket
		+ coeff * du_pow_2m2_dk * bracket
		+ coeff * u_pow_2m2 * dbracket_dk;

	(void)gamma;
	(void)dg_ds;
	(void)lap_g_in;
}

__device__ __forceinline__
void iml_gamma_laplacian_fwd(float s, float k_raw, float m_in,
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
	float bracket = 3.f - 2.f * s * (m + 1.f) / (m * k);
	lap_gamma = coeff * u_pow_2m2 * bracket;
}

__device__ __forceinline__ float box_muller_sample(uint32_t& state)
{
    float u1 = fmaxf(u01(state), 1e-7f);
    float u2 = u01(state);
    return sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265358979f * u2);
}

__device__ __forceinline__ void jitter_query_pos(float& x, float& y, float& z, float alpha, uint32_t seed)
{
    if (alpha <= 0.f) return;

    uint32_t state = seed;
    float dx = alpha * box_muller_sample(state);
    float dy = alpha * box_muller_sample(state);
    float dz = alpha * box_muller_sample(state);

    x = reflect01(x + dx);
    y = reflect01(y + dy);
    z = reflect01(z + dz);
}

template <uint32_t CHANNELS>
__global__ void
splatCUDA(
	int num_computed,
	int num_points,
	int resolution,
	const float* __restrict__ means3D,
	const float* __restrict__ normals,
	const float* __restrict__ scales,
	const float* __restrict__ gamma_m,
	const float* __restrict__ features,
	const uint2* __restrict__ phases_ranges,
	const uint32_t* __restrict__ phases_tiles,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ tiles_list,
	const int* __restrict__ grad_flags,
	const float* __restrict__ out_sums,
	const float* __restrict__ out_sdfs,
	const float* __restrict__ out_feat,
	const float* __restrict__ out_lap_aux,
	const float* __restrict__ dL_dout_sums,
	const float* __restrict__ dL_dout_sdfs,
	const float* __restrict__ dL_dout_feat,
	float* __restrict__ dL_dmeans3D,
	float* __restrict__ dL_dnormals,
	float* __restrict__ dL_dscales,
	float* __restrict__ dL_dgamma_m,
	float* __restrict__ dL_dfeatures,
	float laplacian_lambda,
	float sp_alpha,
	int sp_seed
)
{
	// Identify current tile and associated min/max pixel range.
	auto idx = cg::this_grid().thread_rank();

	if (idx >= num_computed)
		return;
	
	int    point_id = point_list[idx];
	float3 point    = {means3D[point_id*3], means3D[point_id*3+1], means3D[point_id*3+2]};
	float3 normal   = {normals[point_id*3], normals[point_id*3+1], normals[point_id*3+2]};
	float  scale    = scales[point_id];
			float  gamma_m_val = gamma_m[point_id];
			// Clamp effective gamma_m used in the backward kernel to [1.0, 5.0]
			// gamma_m_val = fminf(fmaxf(gamma_m_val, 1.0f), 3.0f);
			gamma_m_val = fminf(fmaxf(gamma_m_val, 1.0f), 5.0f);	

	uint32_t tile_id  = tiles_list[idx];

	uint32_t num_blocks_x = (resolution + BLOCK_X - 1) / BLOCK_X;
	uint32_t num_blocks_y = (resolution + BLOCK_Y - 1) / BLOCK_Y;

	int block_idz = (int)( tile_id / (num_blocks_y*num_blocks_x));
	int block_idy = (int)((tile_id - block_idz*(num_blocks_y*num_blocks_x)) / num_blocks_x);
	int block_idx = (int)( tile_id - block_idz*(num_blocks_y*num_blocks_x) - block_idy*num_blocks_x);

	uint3 cell_id_min = {block_idx * BLOCK_X, block_idy * BLOCK_Y, block_idz * BLOCK_Z};
	uint3 cell_id_max = {min(cell_id_min.x+BLOCK_X, resolution), min(cell_id_min.y+BLOCK_Y, resolution), min(cell_id_min.z+BLOCK_Z, resolution) };

	float dL_dp_i_x = 0.0f;
	float dL_dp_i_y = 0.0f;
	float dL_dp_i_z = 0.0f;
	float dL_dn_i_x = 0.0f;
	float dL_dn_i_y = 0.0f;
	float dL_dn_i_z = 0.0f;
	float dL_ds_i   = 0.0f;
	float dL_dgamma_m_i = 0.0f;
	float dL_da_i   = 0.0f;
	float dL_dfeature_i[CHANNELS] = { 0 };

	float feat[CHANNELS]     = { 0 };
	float dL_dfeat[CHANNELS] = { 0 };

	int all_valid_flag = 0;
	for (int cell_id_x = cell_id_min.x; cell_id_x < cell_id_max.x; cell_id_x++)
	{
		for (int cell_id_y = cell_id_min.y; cell_id_y < cell_id_max.y; cell_id_y++)
		{	
			for (int cell_id_z = cell_id_min.z; cell_id_z < cell_id_max.z; cell_id_z++)
			{
				uint32_t cell_id_flat = cell_id_z * resolution * resolution + cell_id_y * resolution + cell_id_x;

				if (grad_flags[cell_id_flat] == 1)
				{
					all_valid_flag = 1;

					float3 cell = {(float)cell_id_x / (resolution-1), (float)cell_id_y / (resolution-1), (float)cell_id_z / (resolution-1)};
					if (sp_alpha > 0.f)
					{
						uint32_t cell_id_flat = cell_id_z * resolution * resolution + cell_id_y * resolution + cell_id_x;
						uint32_t seed = (uint32_t)sp_seed ^ (uint32_t)cell_id_flat * 9781u;
						jitter_query_pos(cell.x, cell.y, cell.z, sp_alpha, seed);
					}

					// get cell info.
					float sum = out_sums[cell_id_flat];
					float sdf = out_sdfs[cell_id_flat];
					for (int ch = 0; ch < CHANNELS; ch++)
						feat[ch] = out_feat[(ch*resolution*resolution*resolution) + cell_id_flat];

					float dL_dsum = dL_dout_sums[cell_id_flat];
					float dL_dsdf = dL_dout_sdfs[cell_id_flat];
					for (int ch = 0; ch < CHANNELS; ch++)
						dL_dfeat[ch] = dL_dout_feat[(ch*resolution*resolution*resolution) + cell_id_flat];

					// for backprop
					const int num_cell = resolution * resolution * resolution;
					float gL = laplacian_lambda * dL_dsdf;
					float bar_A_lap = 0.f, bar_B_lap = 0.f, bar_LA = 0.f, bar_LB = 0.f;
					float bar_GAx = 0.f, bar_GAy = 0.f, bar_GAz = 0.f;
					float bar_GBx = 0.f, bar_GBy = 0.f, bar_GBz = 0.f;
					if (gL != 0.f && sum >= 1e-4f)
					{
						float B = sum;
						float F = sdf;
						float A = F * B;
						float LB = out_lap_aux[0 * num_cell + cell_id_flat];
						float LA = out_lap_aux[1 * num_cell + cell_id_flat];
						float GBx = out_lap_aux[2 * num_cell + cell_id_flat];
						float GBy = out_lap_aux[3 * num_cell + cell_id_flat];
						float GBz = out_lap_aux[4 * num_cell + cell_id_flat];
						float GAx = out_lap_aux[5 * num_cell + cell_id_flat];
						float GAy = out_lap_aux[6 * num_cell + cell_id_flat];
						float GAz = out_lap_aux[7 * num_cell + cell_id_flat];
						float invB = 1.f / B;
						float invB2 = invB * invB;
						float invB3 = invB2 * invB;
						float Gx = (B * GAx - A * GBx) * invB2;
						float Gy = (B * GAy - A * GBy) * invB2;
						float Gz = (B * GAz - A * GBz) * invB2;
						float GBnorm2 = GBx * GBx + GBy * GBy + GBz * GBz;
						float GBdotG = GBx * Gx + GBy * Gy + GBz * Gz;
						float GBdotGA = GBx * GAx + GBy * GAy + GBz * GAz;
						bar_LA = gL * invB;
						bar_LB = gL * (-A * invB2);
						bar_A_lap = gL * (-LB * invB2 + 2.f * GBnorm2 * invB3);
						// bar_B_lap = gL * (-LA * invB2 + 2.f * A * LB * invB3 + 6.f * GBdotG * invB2 - 2.f * GBdotGA * invB3);
						bar_B_lap = gL * (-LA * invB2 
							+ 2.f * A * LB * invB3 
							+ 2.f * GBdotG * invB2          // was 6.f ← wrong
							+ 2.f * GBdotGA * invB3          // sign flipped ← wrong
							- 4.f * A * GBnorm2 * invB2 * invB2);  // missing term ← wrong
						bar_GAx = gL * (-2.f * invB2) * GBx;
						bar_GAy = gL * (-2.f * invB2) * GBy;
						bar_GAz = gL * (-2.f * invB2) * GBz;
						bar_GBx = gL * (-2.f * invB) * (Gx - A * GBx * invB2);
						bar_GBy = gL * (-2.f * invB) * (Gy - A * GBy * invB2);
						bar_GBz = gL * (-2.f * invB) * (Gz - A * GBz * invB2);
					}

					float3 vec   = {cell.x-point.x, cell.y-point.y, cell.z-point.z};
					float  proj  = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;
					float  norm  = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
					//float  theta = __expf(-1.0f * norm / (scale * scale));

					// float gamma_eps_scale = 0.1f;
					// float gamma_kappa = 2.f;
					// float eps = fmaxf(scale * gamma_eps_scale, 1e-6f);
					// float kappa_eps = fmaxf(0.1, 1e-6f);
					float kappa_eps = fmaxf(scale,  1e-6f);
					float support = gamma_m_val * kappa_eps;
					float inv_eps = 1.f / (support);
					// float kappa_eps = fmaxf(gamma_m * kappa_eps, 1e-6f);
					float inv_kappa_eps = 1.f / kappa_eps;
					float cutoff = fmaxf(support, 1e-6f);
					float inv_cutoff = 1.f / cutoff;
					float dist = sqrtf(norm + 1e-12f);

					float theta = 0.f;
					float dtheta_dp_base = 0.f;
					float dtheta_dscale = 0.f;
					float dtheta_dgamma_m = 0.f;
					if (dist <= cutoff)
					{
						float t = dist * inv_cutoff;
						float one_minus_tval = fmaxf(0.f, 1.f - t);
						float pow_base = powf(one_minus_tval, 2.f * gamma_m_val); 
						float lin = 2.f * dist * inv_kappa_eps + 1.f;
						theta = pow_base * lin;

						// Derivative calculations
						float pow_deriv_base = (gamma_m_val > 0.f) ? powf(one_minus_tval, 2.f * gamma_m_val - 1.f)*(2.f * inv_kappa_eps) : 0.f;
						float poly_deriv_base = 2.f *inv_kappa_eps;
						// checked until here
						// float dpoly_ds = -2.f * gamma_m * inv_cutoff * pow_deriv_base;
						// float dlin_ds = 2.f * inv_kappa_eps;
						// float dtheta_ds = dpoly_ds * lin + pow_base * dlin_ds;

						// float inv_s = rsqrtf(norm + 1e-12f);

						dtheta_dp_base = (1.f/dist)  * (pow_deriv_base * lin + -2.f*pow_base * poly_deriv_base);
						//taking kappa_eps as scale

						// float dpoly_deps = (2.f * gamma_m * t * inv_eps) * pow_deriv_base;
						// float dlin_deps = -2.f * dist * inv_kappa_eps * inv_eps;
						// float dtheta_deps = dpoly_deps * lin + pow_base * dlin_deps;
						// dtheta_dscale = dtheta_deps*gamma_eps_scale;
						float dpow_dscale_base = (gamma_m_val > 0.f) ? powf(one_minus_tval, 2.f * gamma_m_val - 1.f) : 0.f;
						dtheta_dscale = (gamma_m_val > 0.f) ? (2.f * dist *inv_kappa_eps *inv_kappa_eps)*((dpow_dscale_base*lin) - pow_base) : 0;
						
						// Compute dtheta/dgamma_m
						// theta = pow_base * lin where pow_base = (1-t)^(2*gamma_m)
						// cutoff = gamma_m * kappa_eps, so t = dist / cutoff = dist / (gamma_m * kappa_eps)
						// d(pow_base)/dgamma_m = (1-t)^(2*gamma_m) * 2 * ln(1-t) + 2*t*(1-t)^(2*gamma_m-1)
						if (gamma_m_val > 0.f && one_minus_tval > 1e-6f)
						{
							float log_one_minus_t = logf(one_minus_tval);
							float pow_deriv_gamma_m = powf(one_minus_tval, 2.f * gamma_m_val - 1.f);
							dtheta_dgamma_m = lin * (pow_base * 2.f * log_one_minus_t + 2.f * t * pow_deriv_gamma_m);
						}
						else
						{
							dtheta_dgamma_m = 0.f;
						}
					}

					if (theta < 1e-4f)
						continue;

					// float dtheta_i_dp_i_base = 2 * theta / (scale * scale);
					float dtheta_i_dp_i_base = dtheta_dp_base;
					float dfeat_ch_dfeature_i_ch = theta / sum;

					// dL_dtheta_i
					float dL_dtheta_i = dL_dsdf * ((proj - sdf) / sum) + dL_dsum;
					// dL_dproj_i
					float dL_dproj_i = dL_dsdf * (theta / sum);

					// Laplacian term S = F + λ lapF: chain through per-point radial γ, γ′, ∇²γ (matches forward.cu iml_gamma_laplacian)
					if (gL != 0.f && sum >= 1e-4f)
					{
						float ig, idg, ilap;
						iml_gamma_laplacian_fwd(dist, kappa_eps, gamma_m_val, ig, idg, ilap);
						float d2g_ds2, dlap_ds, dgamma_dk, d_dg_ds_dk, dlap_dk;
						iml_gamma_laplacian_reverse(dist, kappa_eps, gamma_m_val, ig, idg, ilap,
							d2g_ds2, dlap_ds, dgamma_dk, d_dg_ds_dk, dlap_dk);
						float inv_s = 1.f / dist;
						float barGA_dot_v = bar_GAx * vec.x + bar_GAy * vec.y + bar_GAz * vec.z;
						float barGB_dot_v = bar_GBx * vec.x + bar_GBy * vec.y + bar_GBz * vec.z;
						float c_lap = bar_LB + bar_LA * proj;
						float c_dg = bar_LA * (2.f * proj * inv_s) + barGA_dot_v * proj * inv_s + barGB_dot_v * inv_s;
						float c_gamma = bar_GAx * normal.x + bar_GAy * normal.y + bar_GAz * normal.z;
						float c_proj = bar_LA * (ilap + 2.f * idg * inv_s) + idg * inv_s * barGA_dot_v;
						float radial_q = c_lap * dlap_ds + c_dg * d2g_ds2 + c_gamma * idg;
						float lap_dscale = c_lap * dlap_dk + c_dg * d_dg_ds_dk + c_gamma * dgamma_dk;
						dL_dtheta_i += bar_A_lap * proj + bar_B_lap + c_gamma;
						dL_dproj_i += c_proj;
						dL_dp_i_x += -radial_q * inv_s * vec.x;
						dL_dp_i_y += -radial_q * inv_s * vec.y;
						dL_dp_i_z += -radial_q * inv_s * vec.z;
						dL_ds_i += lap_dscale;
					}

					for (int ch = 0; ch < CHANNELS; ch++)
						dL_dtheta_i += dL_dfeat[ch] * (features[point_id*CHANNELS+ch] - feat[ch]) / sum;

					// // dL_dproj_i
					// float dL_dproj_i = dL_dsdf * (theta / sum);

					// dL_dfeature_i
					for (int ch = 0; ch < CHANNELS; ch++)
						dL_dfeature_i[ch] += dL_dfeat[ch] * dfeat_ch_dfeature_i_ch;

					// dL_dp_i
					dL_dp_i_x += dL_dtheta_i * (dtheta_i_dp_i_base * vec.x) + dL_dproj_i * (- normal.x);
					dL_dp_i_y += dL_dtheta_i * (dtheta_i_dp_i_base * vec.y) + dL_dproj_i * (- normal.y);
					dL_dp_i_z += dL_dtheta_i * (dtheta_i_dp_i_base * vec.z) + dL_dproj_i * (- normal.z);

					// dL_dn_i
					dL_dn_i_x += dL_dproj_i * vec.x;
					dL_dn_i_y += dL_dproj_i * vec.y;
					dL_dn_i_z += dL_dproj_i * vec.z;

					// dL_ds_i
					//dL_ds_i   += dL_dtheta_i * (dtheta_i_dp_i_base * norm / scale);
					dL_ds_i   += dL_dtheta_i * dtheta_dscale;
					
					// dL_dgamma_m_i
					dL_dgamma_m_i += dL_dtheta_i * dtheta_dgamma_m;

				}
			}
		}
	}

	if (all_valid_flag != 0)
	{
		atomicAdd(&dL_dmeans3D[point_id*3 + 0], dL_dp_i_x);
		atomicAdd(&dL_dmeans3D[point_id*3 + 1], dL_dp_i_y);
		atomicAdd(&dL_dmeans3D[point_id*3 + 2], dL_dp_i_z);

		atomicAdd(&dL_dnormals[point_id*3 + 0], dL_dn_i_x);
		atomicAdd(&dL_dnormals[point_id*3 + 1], dL_dn_i_y);
		atomicAdd(&dL_dnormals[point_id*3 + 2], dL_dn_i_z);
		
		atomicAdd(&dL_dscales[point_id], dL_ds_i);
		atomicAdd(&dL_dgamma_m[point_id], dL_dgamma_m_i);

		for (int ch = 0; ch < CHANNELS; ch++)
			atomicAdd(&dL_dfeatures[point_id*CHANNELS + ch], dL_dfeature_i[ch]);
	}
}

void BACKWARD::splat(
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
	const int sp_seed)
{
	splatCUDA<NUM_CHANNELS> << <(num_computed + 255) / 256, 256 >> >(
		num_computed,
		num_points,
		resolution, 
		means3D,
		normals,
		scales,
		gamma_m,
		features,
		phases_ranges,
		phases_tiles,
		point_list,
		tiles_list,
		grad_flags,
		out_sums,
		out_sdfs,
		out_feat,
		out_lap_aux,
		dL_dout_sums,
		dL_dout_sdfs,
		dL_dout_feat,
		dL_dmeans3D,
		dL_dnormals,
		dL_dscales,
		dL_dgamma_m,
		dL_dfeatures,
		laplacian_lambda,
		sp_alpha,
		sp_seed);
}
