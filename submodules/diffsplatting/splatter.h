#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

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
    const int sp_seed);

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
    const torch::Tensor& geoBuffer,
    const torch::Tensor& binBuffer,
    const torch::Tensor& voxBuffer,
    const torch::Tensor& spaBuffer,
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
	);


