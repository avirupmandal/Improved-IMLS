#include "cumc.h"
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace cumc
{

  template <typename Scalar, typename IndexType>
  class CUMC
  {
    CuMC<Scalar, IndexType> mc;

    static_assert(std::is_same<Scalar, double>() ||
                  std::is_same<Scalar, float>());
    static_assert(std::is_same<IndexType, long>() ||
                  std::is_same<IndexType, int>());

  public:
    ~CUMC()
    {
      cudaDeviceSynchronize();
      cudaFree(mc.temp_storage);
      cudaFree(mc.first_cell_used);
      cudaFree(mc.used_to_first_mc_vert);
      cudaFree(mc.used_to_first_mc_tri);
      cudaFree(mc.used_cell_code);
      cudaFree(mc.used_cell_index);
      cudaFree(mc.verts_type);
      cudaFree(mc.tris);
      cudaFree(mc.verts);
      cudaFree(mc.feats);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor sdfsgrid,
                                                                    torch::Tensor featgrid,
                                                                    Scalar isovalue,
                                                                    IndexType dim_feats)
    {
      CHECK_INPUT(sdfsgrid);
      CHECK_INPUT(featgrid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(sdfsgrid.dtype() == scalarType,
                            "sdfsgrid type must match the mc class");
      TORCH_INTERNAL_ASSERT(featgrid.dtype() == scalarType,
                            "featgrid type must match the mc class");

      torch::ScalarType indexType = torch::kInt;
      if constexpr (std::is_same<IndexType, int>())
      {
        indexType = torch::kInt;
      }
      else
      {
        indexType = torch::kLong;
      }

      IndexType dimX = sdfsgrid.size(0);
      IndexType dimY = sdfsgrid.size(1);
      IndexType dimZ = sdfsgrid.size(2);

      mc.forward(sdfsgrid.data_ptr<Scalar>(), 
                reinterpret_cast<Feature<Scalar> *>(featgrid.data_ptr<Scalar>()), 
                dimX, dimY, dimZ, isovalue, sdfsgrid.device().index());

      auto verts =
          torch::from_blob(
              mc.verts, torch::IntArrayRef{mc.n_verts, 3},
              sdfsgrid.options().dtype(scalarType)).clone();

      auto feats =
          torch::from_blob(
              mc.feats, torch::IntArrayRef{mc.n_verts, dim_feats},
              sdfsgrid.options().dtype(scalarType)).clone();

      auto tris =
          torch::from_blob(
              mc.tris, torch::IntArrayRef{mc.n_tris, 3},
              sdfsgrid.options().dtype(indexType)).clone();

      return {verts, feats, tris};
    }


    void backward(torch::Tensor sdfsgrid, torch::Tensor featgrid, Scalar isovalue, IndexType dim_feats, 
                  torch::Tensor adj_verts, torch::Tensor adj_feats,
                  torch::Tensor adj_sdfsgrid, torch::Tensor adj_featgrid)
    {
      CHECK_INPUT(adj_verts);
      CHECK_INPUT(adj_feats);
      CHECK_INPUT(adj_sdfsgrid);
      CHECK_INPUT(adj_featgrid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(adj_verts.dtype() == scalarType,
                            "adj_verts type must match the mc class");
      TORCH_INTERNAL_ASSERT(adj_sdfsgrid.dtype() == scalarType,
                            "adj_sdfsgrid type must match the mc class");
      TORCH_INTERNAL_ASSERT(adj_featgrid.dtype() == scalarType,
                            "adj_featgrid type must match the mc class");

      IndexType dimX = sdfsgrid.size(0);
      IndexType dimY = sdfsgrid.size(1);
      IndexType dimZ = sdfsgrid.size(2);

      mc.backward(
          sdfsgrid.data_ptr<Scalar>(),
          reinterpret_cast<Feature<Scalar> *>(featgrid.data_ptr<Scalar>()),
          reinterpret_cast<Vertex<Scalar> *>(adj_verts.data_ptr<Scalar>()),
          reinterpret_cast<Feature<Scalar> *>(adj_feats.data_ptr<Scalar>()),
          adj_sdfsgrid.data_ptr<Scalar>(),
          reinterpret_cast<Feature<Scalar> *>(adj_featgrid.data_ptr<Scalar>()),
          isovalue,
          sdfsgrid.device().index());
    }
  };

} // namespace cumc



template <class C>
void register_mc_class(pybind11::module m, std::string name)
{
  pybind11::class_<C>(m, name)
      .def("forward",  pybind11::overload_cast<torch::Tensor, torch::Tensor, C, C>(&C::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, C, C, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, C>(&C::backward));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  pybind11::class_<cumc::CUMC<double, int>>(m, "CUMCDouble")
      .def(py::init<>())
      .def("forward" , pybind11::overload_cast<torch::Tensor, torch::Tensor, double, int>(&cumc::CUMC<double, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, double, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(&cumc::CUMC<double, int>::backward));

  pybind11::class_<cumc::CUMC<float, int>>(m, "CUMCFloat")
      .def(py::init<>())
      .def("forward" , pybind11::overload_cast<torch::Tensor, torch::Tensor, float , int>(&cumc::CUMC<float, int>::forward))
      .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float , int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(&cumc::CUMC<float, int>::backward));

}