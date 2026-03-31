from setuptools import setup, find_packages

install_requires = [
    "asyncssh",
    "caiman>=1.10.0",
    "dask",
    "fastplotlib~=0.1.0.a17",
    "h5py",
    "hdf5storage",
    "ipympl",
    "mesmerize-core",
    "mesmerize-viz",
    "mslex",
    "nest-asyncio",
    "opencv-python",
    "optype[numpy]",
    "pydantic"
]


setup(
    name="cmcode",
    description="Caiman 2P analysis code for Proekt Lab",
    version="0.1.0",
    install_requires=install_requires,
    packages=find_packages(),
    entry_points={
        'console_scripts': ['caimanlab = cmcode.remote.caimanlab:main']
    },
    author="Ethan Blackwood",
    author_email="ethanbb@pennmedicine.upenn.edu"
)