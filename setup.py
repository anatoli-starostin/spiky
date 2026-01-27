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
            "spiky.spnet"
        ],
    )
