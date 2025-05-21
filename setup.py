import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

class CUDABuildExtension(build_ext):
    def build_extension(self, ext):
        # Compile CUDA files first
        cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
        other_sources = [s for s in ext.sources if not s.endswith('.cu')]
        
        objects = []
        if cuda_sources:
            # Compile CUDA sources with nvcc
            for source in cuda_sources:
                output = source.replace('.cu', '.o')
                cmd = [
                    'nvcc',
                    source,
                    '-c',
                    '-o', output,
                    '--extended-lambda',
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '-arch=compute_86',
                    '--allow-unsupported-compiler',
                    '-Xcompiler=-fPIC',
                    '-I' + os.path.dirname(sys.executable) + '/include/python' + sys.version[:3],
                    '-I/usr/local/cuda-11.8/include'
                ]
                subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                objects.append(output)
        
        # Add compiled objects to extension
        ext.extra_objects = objects
        ext.sources = other_sources  # Only keep non-CUDA sources

        super().build_extension(ext)

ext_modules = [
    Pybind11Extension(
        'qsim',
        sources=['qsim_wrapper.cpp', 'qsim.cu'],  # Keep CUDA files in sources list
        include_dirs=['/usr/local/cuda-11.8/include'],
        library_dirs=['/usr/local/cuda-11.8/lib64'],
        libraries=['cudart'],
        extra_compile_args=['-O3', '-fPIC'],
        language='c++',
    ),
]

setup(
    name='qsim',
    version='0.1',
    cmdclass={'build_ext': CUDABuildExtension},
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6'],
    install_requires=['numpy'],
    zip_safe=False,
)