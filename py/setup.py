import os
import platform
import shutil
import sys

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class CustomBuildPy(build_py_orig):
    def run(self):
        machine = platform.machine().lower()
        if sys.platform.startswith("linux"):
            platform_key = f"linux_{machine}"
            lib_ext = ".so"
        else:
            raise RuntimeError(f"Unsupported platform for carbon wheel: {sys.platform}")

        build_dir = os.environ.get("CARBONPY_BUILD_DIR")
        if not build_dir:
            build_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "build-py")
            )

        carbon_lib = os.path.join(build_dir, "libcarbon.so")
        carbonpy_lib = os.path.join(build_dir, "libcarbonpy", f"libcarbonpy{lib_ext}")
        for src in (carbon_lib, carbonpy_lib):
            if not os.path.isfile(src):
                raise RuntimeError(
                    f"Missing built library {src}; configure/build Carbon with "
                    "-DCARBON_ENABLE_PYTHON=ON and set CARBONPY_BUILD_DIR if needed."
                )

        dest_dir = os.path.join(self.build_lib, "carbon", "lib", platform_key)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(carbonpy_lib, os.path.join(dest_dir, f"libcarbonpy{lib_ext}"))
        shutil.copy2(carbon_lib, os.path.join(dest_dir, f"libcarbon{lib_ext}"))

        super().run()


setup(
    name="carbon-warp",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "carbon": [
            "lib/**/*.so",
        ],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass={"build_py": CustomBuildPy},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0",
    ],
)
