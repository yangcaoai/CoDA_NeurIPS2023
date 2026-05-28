# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="box_intersection",
    sources=["box_intersection.pyx"],   # 如果已经有 .c 也可以用 ["box_intersection.c"]
    include_dirs=[np.get_include()],    # 关键：自动拿到 .../numpy/_core/include
    extra_compile_args=["-O3", "-Wall"],
)

setup(
    name="box_intersection",
    ext_modules=cythonize([ext], language_level="3"),
)