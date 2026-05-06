#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>

#include "src/config.h"
#include "src/splatter_impl.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}


// std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SplatterForward(
	const torch::Tensor& means3D,
    const torch::Tensor& normals,
    const torch::Tensor& scales,
    const torch::Tensor& gamma_m,
    const torch::Tensor& features,
    const int resolution,
    const float laplacian_lambda,
    const float sp_alpha,
    const int sp_seed
	)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    
    const int num_points = means3D.size(0);

    auto int_opts   = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_sums  = torch::full({1, resolution, resolution, resolution}, 0.0, float_opts);
    torch::Tensor out_sdfs  = torch::full({1, resolution, resolution, resolution}, 0.0, float_opts);
    torch::Tensor out_feat  = torch::full({NUM_CHANNELS, resolution, resolution, resolution}, 0.0, float_opts);
    torch::Tensor out_lap_aux = torch::zeros({8, resolution, resolution, resolution}, float_opts);
    torch::Tensor out_sdfs_base = torch::zeros({1, resolution, resolution, resolution}, float_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor pntBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binBuffer = torch::empty({0}, options.device(device));
    torch::Tensor tilBuffer = torch::empty({0}, options.device(device));
    torch::Tensor spaBuffer = torch::empty({0}, options.device(device));

    std::function<char*(size_t)> pntFunc = resizeFunctional(pntBuffer);
    std::function<char*(size_t)> binFunc = resizeFunctional(binBuffer);
    std::function<char*(size_t)> tilFunc = resizeFunctional(tilBuffer);
    std::function<char*(size_t)> spaFunc = resizeFunctional(spaBuffer);

	int num_computed = 0;
    int num_phases   = 0;
	if (num_points == 0)
	{
        // return std::make_tuple(num_computed, num_phases, out_sums, out_sdfs, out_feat, out_sdfs_base, pntBuffer, binBuffer, tilBuffer, spaBuffer);
        return std::make_tuple(num_computed, num_phases, out_sums, out_sdfs, out_feat, out_sdfs_base, out_lap_aux, pntBuffer, binBuffer, tilBuffer, spaBuffer);
    }

	int2 out_numbers = Splatter::Splatter::forward(
		pntFunc, 
        binFunc, 
        tilFunc,
        spaFunc,
		num_points, 
        resolution, 
		means3D.contiguous().data<float>(),
		normals.contiguous().data<float>(),
        scales.contiguous().data_ptr<float>(),
        gamma_m.contiguous().data_ptr<float>(),
        features.contiguous().data_ptr<float>(),
        out_sums.contiguous().data<float>(),
        out_sdfs.contiguous().data<float>(),
        out_feat.contiguous().data<float>(),
        out_lap_aux.contiguous().data<float>(),
        out_sdfs_base.contiguous().data<float>(),
        laplacian_lambda,
        sp_alpha,
        sp_seed
        );
    
    num_computed = out_numbers.x;
    num_phases   = out_numbers.y;

	// return std::make_tuple(num_computed, num_phases, out_sums, out_sdfs, out_feat, out_sdfs_base, pntBuffer, binBuffer, tilBuffer, spaBuffer);
    return std::make_tuple(num_computed, num_phases, out_sums, out_sdfs, out_feat, out_sdfs_base, out_lap_aux, pntBuffer, binBuffer, tilBuffer, spaBuffer);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SplatterBackward(
    const int num_computed,
    const int num_phases,
    const int resolution,
	const torch::Tensor& means3D,
    const torch::Tensor& normals,
    const torch::Tensor& scales,
    const torch::Tensor& gamma_m,
    const torch::Tensor& features,
    const torch::Tensor& forwardpntBuffer,
    const torch::Tensor& forwardbinBuffer,
    const torch::Tensor& forwardtilBuffer,
    const torch::Tensor& forwardspaBuffer,
    const torch::Tensor& out_sums,
    const torch::Tensor& out_sdfs,
    const torch::Tensor& out_feat,
    const torch::Tensor& out_lap_aux,
    const torch::Tensor& dL_dout_sums,
    const torch::Tensor& dL_dout_sdfs,
    const torch::Tensor& dL_dout_feat,
    const float laplacian_lambda,
    const float sp_alpha,
    const int sp_seed
	)
{
    const int num_points = means3D.size(0);
    torch::Tensor dL_dmeans3D   = torch::zeros({num_points, 3}, means3D.options());
    torch::Tensor dL_dnormals   = torch::zeros({num_points, 3}, means3D.options());
    torch::Tensor dL_dscales    = torch::zeros({num_points, 1}, means3D.options());
    torch::Tensor dL_dgamma_m  = torch::zeros({num_points, 1}, means3D.options());
    torch::Tensor dL_dfeatures  = torch::zeros({num_points, NUM_CHANNELS}, means3D.options());

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor celBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> celFunc = resizeFunctional(celBuffer);

    if (num_points != 0)
    {
        Splatter::Splatter::backward(
            num_computed,
            num_phases,
            num_points,
            resolution,
            means3D.contiguous().data<float>(),
            normals.contiguous().data<float>(),
            scales.contiguous().data_ptr<float>(),
            gamma_m.contiguous().data_ptr<float>(),
            features.contiguous().data<float>(),
            reinterpret_cast<char*>(forwardpntBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(forwardbinBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(forwardtilBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(forwardspaBuffer.contiguous().data_ptr()),
            celFunc,
            out_sums.contiguous().data<float>(),
            out_sdfs.contiguous().data<float>(),
            out_feat.contiguous().data<float>(),
            out_lap_aux.contiguous().data<float>(),
            dL_dout_sums.contiguous().data<float>(),
            dL_dout_sdfs.contiguous().data<float>(),
            dL_dout_feat.contiguous().data<float>(),
			dL_dmeans3D.contiguous().data<float>(),
			dL_dnormals.contiguous().data<float>(),
            dL_dscales.contiguous().data<float>(),
            dL_dgamma_m.contiguous().data_ptr<float>(),
            dL_dfeatures.contiguous().data<float>(),
            laplacian_lambda,
            sp_alpha,
            sp_seed
            );
    }
    
    return std::make_tuple(dL_dmeans3D, dL_dnormals, dL_dscales, dL_dgamma_m, dL_dfeatures);
}


