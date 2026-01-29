import sys
import os
import shutil
import re
import subprocess
import torch

from setuptools import setup
from kernels_logic_parser import generate_cu_from_proto


def _pick_gpp():
    # Prefer specific versions if present, else fall back to g++
    candidates = ["g++-13","g++-12","g++-11","g++-10","g++-9","g++-8","g++"]
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    raise RuntimeError("No g++ found in PATH")


GPP_PATH = _pick_gpp()
GPP_DIR = os.path.dirname(GPP_PATH)


generate_cu_from_proto(
    'connections_manager/connections_manager_kernels_logic.proto',
    'connections_manager/aux/connections_manager_kernels_logic.cu'
)
generate_cu_from_proto(
    'spnet/spnet_runtime_kernels_logic.proto',
    'spnet/aux/spnet_runtime_kernels_logic.cu'
)
generate_cu_from_proto(
    'lut/lut_runtime_kernels_logic.proto',
    'lut/aux/lut_runtime_kernels_logic.cu'
)
generate_cu_from_proto(
    'lut/lut_compile_time_kernels_logic.proto',
    'lut/aux/lut_compile_time_kernels_logic.cu'
)
generate_cu_from_proto(
    'synapse_growth/synapse_growth_kernels_logic.proto',
    'synapse_growth/aux/synapse_growth_kernels_logic.cu'
)
generate_cu_from_proto(
    'misc/spike_storage_kernels_logic.proto',
    'misc/aux/spike_storage_kernels_logic.cu'
)
generate_cu_from_proto(
    'torch_utils/torch_utils_kernels_logic.proto',
    'torch_utils/aux/torch_utils_kernels_logic.cu'
)

BUILD_INTEGERS_VERSION = True
BUILD_INTEGERS_COMPILE_ARGS = ['-DBUILD_INTEGERS_VERSION'] if BUILD_INTEGERS_VERSION else []

sources_list_cuda = [
    'connections_manager/connections_manager.cu',
    'misc/spike_storage.cu',
    'misc/firing_buffer.cu',
    'misc/concurrent_ds.cu',
    'misc/misc.cpp',
    'spnet/spnet.cu',
    'spnet/spnet_runtime.cu',
    'lut/lut.cu',
    'lut/lut_runtime.cu',
    'synapse_growth/synapse_growth.cu',
    'torch_utils/torch_utils.cu',
    'spiky_py.cpp'
]
if BUILD_INTEGERS_VERSION:
    sources_list_cuda += [
        'spnet/aux/spnet_I.cu',
        'spnet/aux/spnet_runtime_I.cu',
        'lut/aux/lut_I.cu',
        'lut/aux/lut_runtime_I.cu'
    ]

sources_list_no_cuda = [
    'connections_manager/aux/connections_manager.cpp',
    'misc/aux/spike_storage.cpp',
    'misc/aux/firing_buffer.cpp',
    'misc/aux/concurrent_ds.cpp',
    'misc/misc.cpp',
    'spnet/aux/spnet.cpp',
    'spnet/aux/spnet_runtime.cpp',
    'lut/aux/lut.cpp',
    'lut/aux/lut_runtime.cpp',
    'synapse_growth/aux/synapse_growth.cpp',
    'torch_utils/aux/torch_utils.cpp',
    'spiky_py.cpp',
]
if BUILD_INTEGERS_VERSION:
    sources_list_no_cuda += [
        'spnet/aux/spnet_I.cpp',
        'spnet/aux/spnet_runtime_I.cpp',
        'lut/aux/lut_I.cpp',
        'lut/aux/lut_runtime_I.cpp'
    ]

if hasattr(sys, 'getwindowsversion'):
    if torch.cuda.is_available():
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        setup(
            name='spiky_cuda',
            version='0.1',
            ext_modules=[
                CUDAExtension(
                    'spiky_cuda', sources_list_cuda,
                    extra_compile_args={
                        'cxx': ["-O2"] + BUILD_INTEGERS_COMPILE_ARGS,
                        'nvcc': ['-O3'] + BUILD_INTEGERS_COMPILE_ARGS
                    },
                    libraries=['cuda']
                ),
            ],
            cmdclass={
                'build_ext': BuildExtension
            }
        )
    else:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        setup(
            name='spiky_cuda',
            version='0.1',
            ext_modules=[
                CppExtension(
                    'spiky_cuda', sources_list_no_cuda,
                    extra_compile_args=["-DNO_CUDA", "-O2"] + BUILD_INTEGERS_COMPILE_ARGS
                )
            ],
            cmdclass={
                'build_ext': BuildExtension
            }
        )
else:
    if torch.cuda.is_available():
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        setup(
            name='spiky_cuda',
            version='0.1',
            ext_modules=[
                CUDAExtension(
                    'spiky_cuda', sources_list_cuda,
                    extra_compile_args={
                        'cxx': [
                           '-I', '/usr/local/cuda/include', "-Ofast"
                        ] + BUILD_INTEGERS_COMPILE_ARGS,
                        'nvcc': [
                            '-I', '/usr/local/cuda/include', f'--compiler-bindir={GPP_DIR}', '-O3', '-Xptxas="-v"'
                        ] + BUILD_INTEGERS_COMPILE_ARGS
                    },
                    extra_link_args=['-lcuda'],
                    library_dirs=['/usr/local/cuda/lib64']
                )
            ],
            cmdclass={
                'build_ext': BuildExtension
            }
        )
    else:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        setup(
            name='spiky_cuda',
            version='0.1',
            ext_modules=[
                CppExtension(
                    'spiky_cuda', sources_list_no_cuda,
                    extra_compile_args=["-DNO_CUDA", "-O3"] + BUILD_INTEGERS_COMPILE_ARGS,
                )
            ],
            cmdclass={
                'build_ext': BuildExtension
            }
        )
