try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'utils.libmcubes.mcubes',
    sources=[
        'utils/libmcubes/mcubes.pyx',
        'utils/libmcubes/pywrapper.cpp',
        'utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'utils.libmise.mise',
    sources=[
        'utils/libmise/mise.pyx'
    ],
)




# Gather all extension modules
ext_modules = [
    mcubes_module,
    mise_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)