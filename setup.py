import setuptools

version = '0.0.1'

if __name__ == '__main__':
    setuptools.setup(
        name='spiky',
        version=version,
        description='Several spiky neural models',
        long_description='',
        author='Anatoli Starostin',
        author_email='anatoli.starostin@gmail.com',
        package_dir={"": "src"},
        packages=[
            "spiky.util",
            "spiky.lut_fused",
            "spiky.lutorch",
            "spiky.spnet",
            "spiky.lutorch_ex",
            "spiky.lutorch_ex.cartridges",
            "spiky.lutorch_ex.tests",
        ],
        package_data={"spiky.lutorch_ex.cartridges": ["csrc/*.cu", "csrc/*.cuh", "csrc/*.cpp", "csrc/*.h"]},
        # ninja is required by torch.utils.cpp_extension.load to JIT-build lutorch_ex's co-located
        # CUDA extensions (lutorch_ex_lprojection / pow2_int8 / single_anchor / softsign).
        install_requires=['ninja'],
    )
