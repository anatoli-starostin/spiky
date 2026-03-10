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
    """
    Run proto-to-CUDA codegen for the main `spiky_cuda` extension.

    After the native refactor, all generated kernels live under
    `native/spiky/**`. The generator script `kernels_logic_parser.py`
    now lives in `native/common`.
    """
    here = os.path.dirname(os.path.abspath(__file__))  # native/spiky
    spiky_root = here

    project_root = os.path.dirname(os.path.dirname(here))
    kparser_dir = os.path.join(project_root, "native", "common")
    sys.path.insert(0, kparser_dir)
    from kernels_logic_parser import generate_cu_from_proto

    generate_cu_from_proto(
        os.path.join(spiky_root, "connections_manager", "connections_manager_kernels_logic.proto"),
        os.path.join(spiky_root, "connections_manager", "aux_", "connections_manager_kernels_logic.cu"),
    )
    generate_cu_from_proto(
        os.path.join(spiky_root, "spnet", "spnet_runtime_kernels_logic.proto"),
        os.path.join(spiky_root, "spnet", "aux_", "spnet_runtime_kernels_logic.cu"),
    )
    generate_cu_from_proto(
        os.path.join(spiky_root, "lut", "lut_runtime_kernels_logic.proto"),
        os.path.join(spiky_root, "lut", "aux_", "lut_runtime_kernels_logic.cu"),
    )
    generate_cu_from_proto(
        os.path.join(spiky_root, "lut", "lut_compile_time_kernels_logic.proto"),
        os.path.join(spiky_root, "lut", "aux_", "lut_compile_time_kernels_logic.cu"),
    )
    generate_cu_from_proto(
        os.path.join(spiky_root, "synapse_growth", "synapse_growth_kernels_logic.proto"),
        os.path.join(spiky_root, "synapse_growth", "aux_", "synapse_growth_kernels_logic.cu"),
    )
    generate_cu_from_proto(
        os.path.join(spiky_root, "misc", "spike_storage_kernels_logic.proto"),
        os.path.join(spiky_root, "misc", "aux_", "spike_storage_kernels_logic.cu"),
    )
    generate_cu_from_proto(
        os.path.join(spiky_root, "torch_utils", "torch_utils_kernels_logic.proto"),
        os.path.join(spiky_root, "torch_utils", "aux_", "torch_utils_kernels_logic.cu"),
    )


BUILD_INTEGERS_VERSION = True
BUILD_INTEGERS_COMPILE_ARGS = ['-DBUILD_INTEGERS_VERSION'] if BUILD_INTEGERS_VERSION else []

here = os.path.dirname(os.path.abspath(__file__))  # native/spiky
spiky_root = here

sources_list_cuda = [
    os.path.join(spiky_root, 'connections_manager', 'connections_manager.cu'),
    os.path.join(spiky_root, 'misc', 'spike_storage.cu'),
    os.path.join(spiky_root, 'misc', 'firing_buffer.cu'),
    os.path.join(spiky_root, 'misc', 'concurrent_ds.cu'),
    os.path.join(spiky_root, 'misc', 'misc.cpp'),
    os.path.join(spiky_root, 'spnet', 'spnet.cu'),
    os.path.join(spiky_root, 'spnet', 'spnet_runtime.cu'),
    os.path.join(spiky_root, 'lut', 'lut.cu'),
    os.path.join(spiky_root, 'lut', 'lut_runtime.cu'),
    os.path.join(spiky_root, 'synapse_growth', 'synapse_growth.cu'),
    os.path.join(spiky_root, 'torch_utils', 'torch_utils.cu'),
    # LUTorch kernels are built as part of the dedicated `lutorch_cuda` module.
    os.path.join(spiky_root, 'spiky_py.cpp'),
]
if BUILD_INTEGERS_VERSION:
    sources_list_cuda += [
        'spnet/aux_/spnet_I.cu',
        'spnet/aux_/spnet_runtime_I.cu',
        'lut/aux_/lut_I.cu',
        'lut/aux_/lut_runtime_I.cu'
    ]

sources_list_no_cuda = [
    os.path.join(spiky_root, 'connections_manager', 'aux_', 'connections_manager.cpp'),
    os.path.join(spiky_root, 'misc', 'aux_', 'spike_storage.cpp'),
    os.path.join(spiky_root, 'misc', 'aux_', 'firing_buffer.cpp'),
    os.path.join(spiky_root, 'misc', 'aux_', 'concurrent_ds.cpp'),
    os.path.join(spiky_root, 'misc', 'misc.cpp'),
    os.path.join(spiky_root, 'spnet', 'aux_', 'spnet.cpp'),
    os.path.join(spiky_root, 'spnet', 'aux_', 'spnet_runtime.cpp'),
    os.path.join(spiky_root, 'lut', 'aux_', 'lut.cpp'),
    os.path.join(spiky_root, 'lut', 'aux_', 'lut_runtime.cpp'),
    os.path.join(spiky_root, 'synapse_growth', 'aux_', 'synapse_growth.cpp'),
    os.path.join(spiky_root, 'torch_utils', 'aux_', 'torch_utils.cpp'),
    os.path.join(spiky_root, 'spiky_py.cpp'),
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
