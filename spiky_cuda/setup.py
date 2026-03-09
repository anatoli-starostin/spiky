import os
import shutil
import sys

from setuptools import setup


def _pick_gpp():
    candidates = ["g++-13", "g++-12", "g++-11", "g++-10", "g++-9", "g++-8", "g++"]
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    raise RuntimeError("No g++ found in PATH")


def _run_codegen():
    from kernels_logic_parser import generate_cu_from_proto
    generate_cu_from_proto(
        'connections_manager/connections_manager_kernels_logic.proto',
        'connections_manager/aux_/connections_manager_kernels_logic.cu'
    )
    generate_cu_from_proto(
        'spnet/spnet_runtime_kernels_logic.proto',
        'spnet/aux_/spnet_runtime_kernels_logic.cu'
    )
    generate_cu_from_proto(
        'lut/lut_runtime_kernels_logic.proto',
        'lut/aux_/lut_runtime_kernels_logic.cu'
    )
    generate_cu_from_proto(
        'lut/lut_compile_time_kernels_logic.proto',
        'lut/aux_/lut_compile_time_kernels_logic.cu'
    )
    generate_cu_from_proto(
        'synapse_growth/synapse_growth_kernels_logic.proto',
        'synapse_growth/aux_/synapse_growth_kernels_logic.cu'
    )
    generate_cu_from_proto(
        'misc/spike_storage_kernels_logic.proto',
        'misc/aux_/spike_storage_kernels_logic.cu'
    )
    generate_cu_from_proto(
        'torch_utils/torch_utils_kernels_logic.proto',
        'torch_utils/aux_/torch_utils_kernels_logic.cu'
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
    'lutorch/lutorch.cu',
    'spiky_py.cpp'
]
if BUILD_INTEGERS_VERSION:
    sources_list_cuda += [
        'spnet/aux_/spnet_I.cu',
        'spnet/aux_/spnet_runtime_I.cu',
        'lut/aux_/lut_I.cu',
        'lut/aux_/lut_runtime_I.cu'
    ]

sources_list_no_cuda = [
    'connections_manager/aux_/connections_manager.cpp',
    'misc/aux_/spike_storage.cpp',
    'misc/aux_/firing_buffer.cpp',
    'misc/aux_/concurrent_ds.cpp',
    'misc/misc.cpp',
    'spnet/aux_/spnet.cpp',
    'spnet/aux_/spnet_runtime.cpp',
    'lut/aux_/lut.cpp',
    'lut/aux_/lut_runtime.cpp',
    'synapse_growth/aux_/synapse_growth.cpp',
    'torch_utils/aux_/torch_utils.cpp',
    'lutorch/aux_/lutorch.cpp',
    'spiky_py.cpp',
]
if BUILD_INTEGERS_VERSION:
    sources_list_no_cuda += [
        'spnet/aux_/spnet_I.cpp',
        'spnet/aux_/spnet_runtime_I.cpp',
        'lut/aux_/lut_I.cpp',
        'lut/aux_/lut_runtime_I.cpp'
    ]


def _torch_cuda_available():
    import torch
    return torch.cuda.is_available()


def _get_build_extension():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension


def _get_ext_modules():
    if hasattr(sys, "getwindowsversion"):
        if _torch_cuda_available():
            from torch.utils.cpp_extension import CUDAExtension

            return [
                CUDAExtension(
                    "spiky_cuda",
                    sources_list_cuda,
                    extra_compile_args={
                        "cxx": ["-O2"] + BUILD_INTEGERS_COMPILE_ARGS,
                        "nvcc": [
                            "-O3",
                            "-v",
                            "-allow-unsupported-compiler",
                            '-Xptxas="-v"',
                        ] + BUILD_INTEGERS_COMPILE_ARGS,
                    },
                    libraries=["cuda"],
                )
            ]
        else:
            from torch.utils.cpp_extension import CppExtension

            return [
                CppExtension(
                    "spiky_cuda",
                    sources_list_no_cuda,
                    extra_compile_args=["-DNO_CUDA", "-O2"] + BUILD_INTEGERS_COMPILE_ARGS,
                )
            ]
    else:
        if _torch_cuda_available():
            from torch.utils.cpp_extension import CUDAExtension

            gpp_path = _pick_gpp()
            gpp_dir = os.path.dirname(gpp_path)

            return [
                CUDAExtension(
                    "spiky_cuda",
                    sources_list_cuda,
                    extra_compile_args={
                        "cxx": [
                            "-I",
                            "/usr/local/cuda/include",
                            "-Ofast",
                        ] + BUILD_INTEGERS_COMPILE_ARGS,
                        "nvcc": [
                            "-I",
                            "/usr/local/cuda/include",
                            f"--compiler-bindir={gpp_dir}",
                            "-O3",
                            '-Xptxas="-v"',
                        ] + BUILD_INTEGERS_COMPILE_ARGS,
                    },
                    extra_link_args=["-lcuda"],
                    library_dirs=["/usr/local/cuda/lib64"],
                )
            ]
        else:
            from torch.utils.cpp_extension import CppExtension

            return [
                CppExtension(
                    "spiky_cuda",
                    sources_list_no_cuda,
                    extra_compile_args=["-DNO_CUDA", "-O3"] + BUILD_INTEGERS_COMPILE_ARGS,
                )
            ]


_run_codegen()

setup(
    name="spiky_cuda",
    version="0.1",
    ext_modules=_get_ext_modules(),
    cmdclass={"build_ext": _get_build_extension()},
)
