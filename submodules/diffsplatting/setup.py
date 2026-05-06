from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob

os.path.dirname(os.path.abspath(__file__))

setup(
    name="diffsplatting",
    ext_modules=[
        CUDAExtension(
            name="diffsplatting",
            sources=[
            "src/splatter_impl.cu",
            "src/forward.cu",
            "src/backward.cu",
            "splatter.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-O3",  "-use_fast_math", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
