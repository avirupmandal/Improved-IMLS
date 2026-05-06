#include "splatter_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


Splatter::PhaseState Splatter::PhaseState::fromChunk(char*& chunk, size_t P)
{
	PhaseState spa;
	obtain(chunk, spa.keys,   P, 128);
	obtain(chunk, spa.ranges, P, 128);
	return spa;
}

// forward buffer
Splatter::ForwardPointState Splatter::ForwardPointState::fromChunk(char*& chunk, size_t P)
{
	ForwardPointState point;
	obtain(chunk, point.bbox, P * 3, 128); // x,  y,  z
	obtain(chunk, point.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, point.scan_size, point.tiles_touched, point.tiles_touched, P);
	obtain(chunk, point.scanning_space, point.scan_size, 128);
	obtain(chunk, point.tiles_offsets, P, 128);
	return point;
}

Splatter::ForwardBinState Splatter::ForwardBinState::fromChunk(char*& chunk, size_t P)
{
	ForwardBinState bin;
	obtain(chunk, bin.point_list, P, 128);
	obtain(chunk, bin.tiles_list, P, 128);

	obtain(chunk, bin.point_list_unsorted, P, 128);
	obtain(chunk, bin.tiles_list_unsorted, P, 128);

	cub::DeviceRadixSort::SortPairs(
		nullptr, bin.sorting_size,
		bin.tiles_list_unsorted, bin.tiles_list,
		bin.point_list_unsorted, bin.point_list, P);
	obtain(chunk, bin.list_sorting_space, bin.sorting_size, 128);
	return bin;
}

Splatter::ForwardTileState Splatter::ForwardTileState::fromChunk(char*& chunk, size_t N)
{
	ForwardTileState tile;
	obtain(chunk, tile.ranges, N, 128);
	obtain(chunk, tile.phases_touched, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, tile.scan_size, tile.phases_touched, tile.phases_touched, N);
	obtain(chunk, tile.scanning_space, tile.scan_size, 128);
	obtain(chunk, tile.phases_offsets, N, 128);
	return tile;
}



Splatter::BackwardCellState Splatter::BackwardCellState::fromChunk(char*& chunk, size_t N)
{
	BackwardCellState cell;
	obtain(chunk, cell.grad_flags, N, 128);
	return cell;
}


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithTiles(
	int num_points,
	const uint2* bbox,
	const uint32_t* tiles_offsets,
	uint32_t* tiles_list_unsorted,
	uint32_t* point_list_unsorted,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_points)
		return;

	uint32_t off = (idx == 0) ? 0 : tiles_offsets[idx - 1];

	uint2 bbox_x = bbox[idx * 3 + 0];
	uint2 bbox_y = bbox[idx * 3 + 1];
	uint2 bbox_z = bbox[idx * 3 + 2];

	for (int x = bbox_x.x; x < bbox_x.y; x++)
	{
		for (int y = bbox_y.x; y < bbox_y.y; y++)
		{	
			for (int z = bbox_z.x; z < bbox_z.y; z++)
			{
				uint32_t key = z*(grid.y*grid.x) + y*(grid.x) + x;
				tiles_list_unsorted[off] = key;
				point_list_unsorted[off] = idx;
				off++;
			}
		}
	}

}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyRanges(int L, uint32_t* elem_list, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read ID from key. Update start/end of elem range if at limit.
	uint32_t currelem = elem_list[idx];
	if (idx == 0)
		ranges[currelem].x = 0;
	else
	{
		uint32_t prevelem = elem_list[idx - 1];
		if (currelem != prevelem)
		{
			ranges[prevelem].y = idx;
			ranges[currelem].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currelem].y = L;
}

// Compute the Phase Number from ranges
__global__ void ComputePhaseNumber(int L, uint2* ranges, uint32_t* phases_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;
	
	phases_touched[idx] = 0;

	int num_touched = ranges[idx].y - ranges[idx].x;
	if (num_touched == 0)
		return;
	
	phases_touched[idx] = (int)((num_touched + LEN_PHASE - 1) / LEN_PHASE);
}


__global__ void SplitPhasefromRanges(int L, uint2* ranges, uint32_t* phases_touched, uint32_t* phases_offsets, uint32_t* phases_keys, uint2* phases_ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;
	
	uint32_t phases_index;
	uint32_t val_index_begin;
	uint32_t val_index_end;

	if (idx==0)
		phases_index = 0;
	else
	{
		phases_index = phases_offsets[idx-1];
	}
	
	for (int k = 0; k < phases_touched[idx]; k++)
	{	
		if (k == (phases_touched[idx] - 1))
		{	
			val_index_begin = ranges[idx].x + k * LEN_PHASE;
			val_index_end   = ranges[idx].y;
		}
		else
		{
			val_index_begin = ranges[idx].x + k * LEN_PHASE;
			val_index_end   = ranges[idx].x + (k + 1) * LEN_PHASE;
		}
		phases_ranges[phases_index].x = val_index_begin;
		phases_ranges[phases_index].y = val_index_end;
		phases_keys[phases_index]     = idx;
		phases_index++;
	}
}


__global__ void duplicateWithCells(
	const int num_points,
	const int resolution,
	const uint2* bbox,
	const uint32_t* cells_offsets,
	const int* grad_flags,
	uint32_t* tiles_list_unsorted,
	uint32_t* point_list_unsorted,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_points)
		return;

	uint32_t off = (idx == 0) ? 0 : cells_offsets[idx - 1];

	uint2 bbox_x = bbox[idx * 3 + 0];
	uint2 bbox_y = bbox[idx * 3 + 1];
	uint2 bbox_z = bbox[idx * 3 + 2];

	for (int x = bbox_x.x; x < bbox_x.y; x++)
	{
		for (int y = bbox_y.x; y < bbox_y.y; y++)
		{	
			for (int z = bbox_z.x; z < bbox_z.y; z++)
			{
				int3 cell_id_min = {x * BLOCK_X, y * BLOCK_Y, z * BLOCK_Z};
				int3 cell_id_max = {min(cell_id_min.x+BLOCK_X, resolution), min(cell_id_min.y+BLOCK_Y, resolution), min(cell_id_min.z+BLOCK_Z, resolution) };

				for (int cell_id_x = cell_id_min.x; cell_id_x < cell_id_max.x; cell_id_x++)
				{
					for (int cell_id_y = cell_id_min.y; cell_id_y < cell_id_max.y; cell_id_y++)
					{	
						for (int cell_id_z = cell_id_min.z; cell_id_z < cell_id_max.z; cell_id_z++)
						{
							uint32_t cell_id_flat = cell_id_z * resolution * resolution + cell_id_y * resolution + cell_id_x;
							int valid_flag = grad_flags[cell_id_flat];

							if (valid_flag == 1)
							{
								tiles_list_unsorted[off] = cell_id_flat;
								point_list_unsorted[off] = idx;
								off++;
							}
						}
					}
				}
			}
		}
	}


}


template <uint32_t CHANNELS>
__global__ void compute_grad_flag(
	const int num_cells,
	const int resolution,
	const float* out_sums,
	const float* dL_dout_sums,
	const float* dL_dout_sdfs,
	const float* dL_dout_feat,
	int* grad_flags)
{
	auto cell_id_flat = cg::this_grid().thread_rank();
	if (cell_id_flat >= num_cells)
		return;
	
	float sums = out_sums[cell_id_flat];
	float dL_dsum = dL_dout_sums[cell_id_flat];
	float dL_dsdf = dL_dout_sdfs[cell_id_flat];

	int grad_flag = (dL_dsum != 0) || (dL_dsdf != 0);
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		grad_flag = grad_flag || (dL_dout_feat[(ch*resolution*resolution*resolution) + cell_id_flat] != 0);
	}
	grad_flag = grad_flag && (sums >= 1e-4f);

	grad_flags[cell_id_flat] = grad_flag;
}



int2 Splatter::Splatter::forward(
	std::function<char* (size_t)> pntBuffer,
	std::function<char* (size_t)> binBuffer,
	std::function<char* (size_t)> tilBuffer,
	std::function<char* (size_t)> spaBuffer,
	const int num_points, 
	const int resolution,
	const float* means3D,
	const float* normals,
	const float* scales,
	const float* gamma_m,
	const float* features,
	float* out_sums,
	float* out_sdfs,
	float* out_feat,
	float* out_lap_aux,
	float* out_sdfs_base,
	float laplacian_lambda,
	float sp_alpha,
	int sp_seed)
{	
	size_t point_chunksize = required<ForwardPointState>(num_points);
	char*  point_chunkptr  = pntBuffer(point_chunksize);
	ForwardPointState pntState = ForwardPointState::fromChunk(point_chunkptr, num_points);

	dim3 tile_grid((resolution + BLOCK_X - 1)/BLOCK_X, (resolution + BLOCK_Y - 1)/BLOCK_Y, (resolution + BLOCK_Z - 1)/BLOCK_Z);
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

	int   tile_chunksize = required<ForwardTileState>(tile_grid.x * tile_grid.y * tile_grid.z);
	char* tile_chunkptr  = tilBuffer(tile_chunksize);
	ForwardTileState tilState = ForwardTileState::fromChunk(tile_chunkptr, tile_grid.x * tile_grid.y * tile_grid.z);

	FORWARD::preprocess(
		num_points, 
		resolution,
		tile_grid,
		means3D,
		scales,
		gamma_m,
		features,
		pntState.bbox,
		pntState.tiles_touched);

	cub::DeviceScan::InclusiveSum(pntState.scanning_space, pntState.scan_size,
		pntState.tiles_touched, pntState.tiles_offsets, num_points);

    int num_computed;
    cudaMemcpy(&num_computed, pntState.tiles_offsets + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost);

	int   bin_chunksize = required<ForwardBinState>(num_computed);
	char* bin_chunkptr  = binBuffer(bin_chunksize);
	ForwardBinState binState = ForwardBinState::fromChunk(bin_chunkptr, num_computed);

	duplicateWithTiles << <(num_points + 255) / 256, 256 >> > (
		num_points,
		pntState.bbox,
		pntState.tiles_offsets,
		binState.tiles_list_unsorted,
		binState.point_list_unsorted,
		tile_grid);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y * tile_grid.z);

	cub::DeviceRadixSort::SortPairs(
		binState.list_sorting_space,
		binState.sorting_size,
		binState.tiles_list_unsorted, binState.tiles_list,
		binState.point_list_unsorted, binState.point_list,
		num_computed, 0, bit);

	cudaMemset(tilState.ranges, 0, tile_grid.x * tile_grid.y * tile_grid.z * sizeof(uint2));

	if (num_computed > 0)
		identifyRanges << <(num_computed + 255) / 256, 256 >> > (
			num_computed,
			binState.tiles_list,
			tilState.ranges);

	int num_tiles = tile_grid.x * tile_grid.y * tile_grid.z;
	ComputePhaseNumber << <(num_tiles + 255) / 256, 256 >> > (
		num_tiles,
		tilState.ranges,
		tilState.phases_touched);
	
	cub::DeviceScan::InclusiveSum(tilState.scanning_space, tilState.scan_size,
		tilState.phases_touched, tilState.phases_offsets, num_tiles);

	int num_phases;
    cudaMemcpy(&num_phases, tilState.phases_offsets + num_tiles - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t spa_chunksize = required<PhaseState>(num_phases);
	char*  spa_chunkptr  = spaBuffer(spa_chunksize);
	PhaseState spaState = PhaseState::fromChunk(spa_chunkptr, num_phases);

	SplitPhasefromRanges << <(num_tiles + 255) / 256, 256 >> > (
		num_tiles,
		tilState.ranges,
		tilState.phases_touched,
		tilState.phases_offsets,
		spaState.keys,
		spaState.ranges);
		
	FORWARD::splat(
		tile_grid, 
		block,
		num_phases,
		resolution,
		means3D,
		normals,
		scales,
		gamma_m,
		features,
		spaState.ranges,
		spaState.keys,
		binState.point_list,
		out_sums,
		out_sdfs,
		out_feat,
		out_lap_aux,
		sp_alpha,
		sp_seed);

	FORWARD::weighting(
		resolution,
		out_sums,
		out_sdfs,
		out_feat,
		out_lap_aux,
		out_sdfs_base,
		laplacian_lambda);

	int2 numbers = {num_computed, num_phases};
	return numbers;
}


void Splatter::Splatter::backward(
	const int num_computed,
	const int num_phases,
	const int num_points,
	const int resolution,
	const float* means3D,
	const float* normals,
	const float* scales,
	const float* gamma_m,
	const float* features,
	char* forwardPntBuffer,
	char* forwardBinBuffer,
	char* forwardTilBuffer,
	char* spaBuffer,
	std::function<char* (size_t)> celBuffer,
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
	float sp_alpha,
	int sp_seed)
{
	dim3 tile_grid((resolution + BLOCK_X - 1)/BLOCK_X, (resolution + BLOCK_Y - 1)/BLOCK_Y, (resolution + BLOCK_Z - 1)/BLOCK_Z);
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
	
	ForwardPointState PntState = ForwardPointState::fromChunk(forwardPntBuffer, num_points);
	ForwardTileState  TilState = ForwardTileState::fromChunk(forwardTilBuffer, tile_grid.x * tile_grid.y * tile_grid.z);
	ForwardBinState   BinState = ForwardBinState::fromChunk(forwardBinBuffer, num_computed);
	PhaseState        SpaState = PhaseState::fromChunk(spaBuffer, num_phases);

	int num_cells = (resolution * resolution * resolution);
	size_t cell_chunksize = required<BackwardCellState>(num_cells);
	char*  cell_chunkptr  = celBuffer(cell_chunksize);
	BackwardCellState CellState = BackwardCellState::fromChunk(cell_chunkptr, num_cells);

	compute_grad_flag<NUM_CHANNELS> << <(num_cells + 255) / 256, 256 >> > (
		num_cells,
		resolution,
		out_sums, 
		dL_dout_sums,
		dL_dout_sdfs,
		dL_dout_feat,
		CellState.grad_flags);

	BACKWARD::splat(
		tile_grid,
		block,
		num_computed,
		num_phases, 
		num_points,
		resolution,
		means3D,
		normals,
		scales,
		gamma_m,
		features,
		SpaState.ranges,
		SpaState.keys,
		BinState.point_list,
		BinState.tiles_list,
		CellState.grad_flags,
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
