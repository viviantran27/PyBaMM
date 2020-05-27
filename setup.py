import os
import glob
import logging
import subprocess
from pathlib import Path
from platform import system
import wheel.bdist_wheel as orig

try:
    from setuptools import setup, find_packages, Extension
    from setuptools.command.install import install
except ImportError:
    from distutils.core import setup, find_packages
    from distutils.command.install import install

import CMakeBuild

default_lib_dir = (
    "" if system() == "Windows" else os.path.join(os.getenv("HOME"), ".local")
)

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger("PyBaMM setup")

# To override the default severity of logging
logger.setLevel("INFO")

# Use FileHandler() to log to a file
logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.log")
file_handler = logging.FileHandler(logfile)
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

# Add the file handler
logger.addHandler(file_handler)
logger.info("Starting PyBaMM setup")


class CustomInstall(install):
    """A custom install command to add 2 build options"""

    user_options = install.user_options + [
        ("suitesparse-root=", None, "suitesparse source location"),
        ("sundials-root=", None, "sundials source location"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.suitesparse_root = None
        self.sundials_root = None

    def finalize_options(self):
        install.finalize_options(self)
        if not self.suitesparse_root:
            self.suitesparse_root = default_lib_dir
        if not self.sundials_root:
            self.sundials_root = default_lib_dir

    def run(self):
        install.run(self)


class bdist_wheel(orig.bdist_wheel):
    """A custom install command to add 2 build options"""

    user_options = orig.bdist_wheel.user_options + [
        ("suitesparse-root=", None, "suitesparse source location"),
        ("sundials-root=", None, "sundials source location"),
    ]

    def initialize_options(self):
        orig.bdist_wheel.initialize_options(self)
        self.suitesparse_root = None
        self.sundials_root = None

    def finalize_options(self):
        orig.bdist_wheel.finalize_options(self)
        if not self.suitesparse_root:
            self.suitesparse_root = default_lib_dir
        if not self.sundials_root:
            self.sundials_root = default_lib_dir

    def run(self):
        orig.bdist_wheel.run(self)


def load_version():
    # Read version number from file
    try:
        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, "pybamm", "version"), "r") as f:
            version = f.read().strip().split(",")
        return ".".join([str(int(x)) for x in version])
    except Exception as e:
        raise RuntimeError("Unable to read version number (" + str(e) + ").")


def compile_KLU():
    # Return whether or not the KLU extension should be compiled.
    # Return True if:
    # - Not running on Windows AND
    # - CMake is found AND
    # - The pybind11 directory is found in the PyBaMM project directory
    CMakeFound = True
    PyBind11Found = True
    windows = (not system()) or system() == "Windows"

    msg = "Running on Windows" if windows else "Not running on windows"
    logger.info(msg)

    try:
        subprocess.run(["cmake", "--version"])
        logger.info("Found CMake.")
    except OSError:
        CMakeFound = False
        logger.info("Could not find CMake. Skipping compilation of KLU module.")

    pybamm_project_dir = os.path.dirname(os.path.abspath(__file__))
    pybind11_dir = os.path.join(pybamm_project_dir, "pybind11")
    try:
        open(os.path.join(pybind11_dir, "tools", "pybind11Tools.cmake"))
        logger.info("Found pybind11 directory ({})".format(pybind11_dir))
    except FileNotFoundError:
        PyBind11Found = False
        msg = (
            "Could not find PyBind11 directory ({})."
            " Skipping compilation of KLU module.".format(pybind11_dir)
        )
        logger.info(msg)

    return CMakeFound and PyBind11Found and (not windows)


# Build the list of package data files to be included in the PyBaMM package.
# These are mainly the parameter files located in the input/parameters/ subdirectories.
pybamm_data = []
for file_ext in ["*.csv", "*.py", "*.md"]:
    # Get all the files ending in file_ext in pybamm/input dir.
    # list_of_files = [
    #    'pybamm/input/drive_cycles/car_current.csv',
    #    'pybamm/input/drive_cycles/US06.csv',
    # ...
    list_of_files = glob.glob("pybamm/input/**/" + file_ext, recursive=True)

    # Add these files to pybamm_data.
    # The path must be relative to the package dir (pybamm/), so
    # must process the content of list_of_files to take out the top
    # pybamm/ dir, i.e.:
    # ['input/drive_cycles/car_current.csv',
    #  'input/drive_cycles/US06.csv',
    # ...
    pybamm_data.extend(
        [os.path.join(*Path(filename).parts[1:]) for filename in list_of_files]
    )
pybamm_data.append("./version")
pybamm_data.append("./CITATIONS.txt")

idaklu_ext = Extension("idaklu", ["pybamm/solvers/c_solvers/idaklu.cpp"])
ext_modules = [idaklu_ext] if compile_KLU() else []

setup(
    name="pybamm",
    version=load_version() + ".post3",
    description="Python Battery Mathematical Modelling.",
    long_description="description",
    long_description_content_type="text/markdown",
    url="https://github.com/pybamm-team/PyBaMM",
    packages=find_packages(include=("pybamm", "pybamm.*")),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": CMakeBuild.CMakeBuild,
        "bdist_wheel": bdist_wheel,
        "install": CustomInstall,
    },
    package_data={"pybamm": pybamm_data},
    # List of dependencies
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.3",
        "pandas>=0.24",
        "anytree>=2.4.3",
        "autograd>=1.2",
        "scikit-fem>=0.2.0",
        "casadi>=3.5.0",
        "jupyter",  # For example notebooks
        # Note: Matplotlib is loaded for debug plots, but to ensure pybamm runs
        # on systems without an attached display, it should never be imported
        # outside of plot() methods.
        # Should not be imported
        "matplotlib>=2.0",
    ],
    extras_require={
        "docs": ["sphinx>=1.5", "guzzle-sphinx-theme"],  # For doc generation
        "dev": [
            "flake8>=3",  # For code style checking
            "black",  # For code style auto-formatting
        ],
    },
    entry_points={
        "console_scripts": [
            "pybamm_edit_parameter = pybamm.parameters_cli:edit_parameter",
            "pybamm_add_parameter = pybamm.parameters_cli:add_parameter",
            "pybamm_list_parameters = pybamm.parameters_cli:list_parameters",
            "pybamm_install_odes = pybamm.install_odes:main",
        ],
    },
)
