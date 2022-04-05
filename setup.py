#!/usr/bin/env python3

import sys
from skbuild import setup

setup(
    name = "imreadfast",
    version = "0.0.3.dev0",
    description = "Python wrapper of the KLB file format, a high-performance file format for up to 5-dimensional arrays.",
    url = "https://github.com/Onetism/fastImageRead.git",
    packages=["imreadfast"],
    setup_requires = ["numpy"],
    install_requires = ["cython", "numpy"]
)