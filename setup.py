import os
import sys
from numpy import get_include as get_numpy_include
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def get_pybind_include():
    pybind_inc = os.path.join(os.path.dirname(__file__), 'pybind11', 'include')
    assert os.path.exists(pybind_inc)
    return pybind_inc


def get_spdlog_include():
    inc = os.path.join(os.path.dirname(__file__), 'spdlog', 'include')
    assert os.path.exists(inc)
    return inc


def get_project_include():
    inc = os.path.join(os.path.dirname(__file__), 'cpp', 'include')
    assert os.path.exists(inc)
    return inc


ext_modules = [
    Extension(
        'readdy_learn.analyze_tools',
        sources=['cpp/src/main.cpp', 'cpp/src/lasso_minimizer_objective_fun.cpp'],
        language='c++',
        include_dirs=[
            get_pybind_include(), get_spdlog_include(), get_project_include(), get_numpy_include()
        ],
        extra_compile_args=['-std=c++14', '-O3', '-fvisibility=hidden']
    ),
]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append('-fvisibility=hidden')
        opts.append('-std=c++14')
        opts.append('-O3')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='readdy_learn',
    version='0.0.1',
    author='ReaDDy team',
    author_email='clonker@gmail.com',
    url='https://github.com/readdy/readdy_learn',
    description='readdy learn',
    long_description='some more description',
    ext_modules=ext_modules,
    packages=find_packages(),
    # cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
