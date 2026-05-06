#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <cuda_runtime_api.h>

namespace Splatter
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct PhaseState
	{	
		uint32_t* keys;
		uint2*    ranges;

		static PhaseState fromChunk(char*& chunk, size_t P);
	};

	struct ForwardPointState
	{
		size_t scan_size;
		char* scanning_space;

		uint2*  bbox;
		uint32_t* tiles_touched;
		uint32_t* tiles_offsets;

		static ForwardPointState fromChunk(char*& chunk, size_t P);
	};

	struct ForwardBinState
	{
		size_t sorting_size;
		char* list_sorting_space;

		uint32_t* point_list;
		uint32_t* tiles_list;
		uint32_t* point_list_unsorted;
		uint32_t* tiles_list_unsorted;

		static ForwardBinState fromChunk(char*& chunk, size_t P);
	};

	struct ForwardTileState
	{	
		size_t scan_size;
		char* scanning_space;

		uint2* ranges;
		uint32_t* phases_touched;
		uint32_t* phases_offsets;

		static ForwardTileState fromChunk(char*& chunk, size_t N);
	};

	struct BackwardCellState
	{	
		int* grad_flags;

		static BackwardCellState fromChunk(char*& chunk, size_t N);
	};	

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	class Splatter
	{
	public:
		static int2 forward(
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
			int sp_seed);

		static void backward(
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
			int sp_seed);
	};
};

