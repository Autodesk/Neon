import os
import shutil
import sys
import platform
from setuptools import setup, find_packages
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py as build_py_orig

class BinaryDistribution(Distribution):
    """Distribution that includes binary components."""
    def has_ext_modules(self):
        return True

class CustomBuildPy(build_py_orig):
    """Custom build command to copy the appropriate shared library."""
    def run(self):
        # Determine platform and architecture
        current_platform = sys.platform
        machine = platform.machine().lower()

        # Map sys.platform to your binaries directory
        if current_platform.startswith("linux"):
            platform_key = "linux_" + machine
            lib_extension = ".so"
        elif current_platform.startswith("darwin"):
            platform_key = "macos_" + machine
            lib_extension = ".dylib"
        elif current_platform.startswith("win"):
            platform_key = "windows_" + machine
            lib_extension = ".dll"
        else:
            raise RuntimeError(f"Unsupported platform: {current_platform}")

        # Define the path to the shared library. TODO: This is a hack to get the path to the shared library. We need to find a better way to do this for multiple platforms.
        source_so = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../cmake-build-debug/libNeonPy/liblibNeonPy{lib_extension}'))

        # Define the destination directory within the build directory
        destination_dir = os.path.join(self.build_lib, 'neon', 'lib', platform_key)
        destination_so = os.path.join(destination_dir, os.path.basename(source_so))

        # Ensure the destination directory exists
        os.makedirs(destination_dir, exist_ok=True)

        # Copy the shared library to the destination directory
        try:
            shutil.copy2(source_so, destination_so)
            print(f"Copied {source_so} to {destination_so}")
        except IOError as e:
            print(f"Error copying {source_so} to {destination_so}: {e}")
            raise e

        # Continue with the standard build process
        super().run()

setup(
    name="neon",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "neon": [
            "**/*.h",             # Include all .h files recursively
        ]
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass={
        'build_py': CustomBuildPy,
    },
    python_requires=">=3.10", # This should be the minimum version of python that warp supports.
    # TODO: We need to add warp to the requirements later. Currently we're using a custom build of warp.
    install_requires=[
        "numpy>=2.0",
        "nvtx"
    ],
)
