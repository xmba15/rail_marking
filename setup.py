#!/usr/bin/env python
import os
import re
import sys
import platform
import subprocess as sp
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


install_requires = []


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = sp.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        dirname = os.path.dirname(self.get_ext_fullpath(ext.name))
        extdir = os.path.abspath(dirname)
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            liboutput = "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}"
            liboutput = liboutput.format(cfg.upper(), extdir)
            cmake_args += [liboutput]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            if not is_pipeline():
                build_args += ["--", "-j6"]

        env = os.environ.copy()
        cxxflags = '{} -DVERSION_INFO=\\"{}\\"'
        cxxflags = cxxflags.format(env.get("CXXFLAGS", ""), self.distribution.get_version())
        env["CXXFLAGS"] = cxxflags
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        sp.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
            env=env,
        )
        sp.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    name="rail_marking",
    python_requires=">3",
    description="rail marking module",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    packages=find_packages("rail_marking"),
    package_dir={"": "rail_marking"},
    test_suite="tests",
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
