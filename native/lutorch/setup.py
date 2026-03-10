import os
import sys
from setuptools import setup


BUILD_INTEGERS_VERSION = True
BUILD_INTEGERS_COMPILE_ARGS = ["-DBUILD_INTEGERS_VERSION"] if BUILD_INTEGERS_VERSION else []


def _torch_cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _pick_gpp() -> str:
    import shutil

    candidates = ["g++-13", "g++-12", "g++-11", "g++-10", "g++-9", "g++-8", "g++"]
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    raise RuntimeError("No g++ found in PATH")


def _get_ext_modules():
    """
    Build a minimal extension that only contains the LUTorch manager
    and its supporting utilities, exposed as the `lutorch_cuda` module.

    Sources live under `native/common` and `native/lutorch`.
    """
    from torch.utils.cpp_extension import CUDAExtension, CppExtension

    # Compute project root from this file (native/lutorch/setup.py -> project root).
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))

    sources_cuda = [
        os.path.join(project_root, "native", "common", "misc.cpp"),
        os.path.join(project_root, "native", "lutorch", "lutorch.cu"),
    ]
    sources_no_cuda = [
        os.path.join(project_root, "native", "common", "misc.cpp"),
        os.path.join(project_root, "native", "lutorch", "aux_", "lutorch.cpp"),
    ]

    if hasattr(sys, "getwindowsversion"):
        if _torch_cuda_available():
            return [
                CUDAExtension(
                    "lutorch_cuda",
                    sources_cuda,
                    extra_compile_args={
                        "cxx": ["-O2"] + BUILD_INTEGERS_COMPILE_ARGS,
                        "nvcc": [
                            "-O3",
                            "-v",
                            "-allow-unsupported-compiler",
                            '-Xptxas="-v"',
                        ]
                        + BUILD_INTEGERS_COMPILE_ARGS,
                    },
                    libraries=["cuda"],
                )
            ]
        return [
            CppExtension(
                "lutorch_cuda",
                sources_no_cuda,
                extra_compile_args=["-DNO_CUDA", "-O2"] + BUILD_INTEGERS_COMPILE_ARGS,
            )
        ]

    # Non-Windows (Linux, macOS)
    if _torch_cuda_available():
        gpp_path = _pick_gpp()
        gpp_dir = os.path.dirname(gpp_path)

        return [
            CUDAExtension(
                "lutorch_cuda",
                sources_cuda,
                extra_compile_args={
                    "cxx": [
                        "-I",
                        "/usr/local/cuda/include",
                        "-Ofast",
                    ]
                    + BUILD_INTEGERS_COMPILE_ARGS,
                    "nvcc": [
                        "-I",
                        "/usr/local/cuda/include",
                        f"--compiler-bindir={gpp_dir}",
                        "-O3",
                        '-Xptxas="-v"',
                    ]
                    + BUILD_INTEGERS_COMPILE_ARGS,
                },
                extra_link_args=["-lcuda"],
                library_dirs=["/usr/local/cuda/lib64"],
            )
        ]

    return [
        CppExtension(
            "lutorch_cuda",
            sources_no_cuda,
            extra_compile_args=["-DNO_CUDA", "-O3"] + BUILD_INTEGERS_COMPILE_ARGS,
        )
    ]


def _get_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension


if __name__ == "__main__":
    setup(
        name="lutorch_cuda",
        version="0.1",
        ext_modules=_get_ext_modules(),
        cmdclass={"build_ext": _get_build_extension()},
    )

