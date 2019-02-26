""" A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path


_HERE = path.abspath(path.dirname(__file__))
_README_FILE_NAME = "README.md"
_README_FILE_ENCODING = "utf8"


# Get the long description from the README file
with open(path.join(_HERE, _README_FILE_NAME),
          encoding=_README_FILE_ENCODING) as file:
    long_description = file.read()

setup(
    name="clustering",

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.0.1",

    author="BIZZOZZÉRO Nicolas",
    author_email="nicolasbizzozzero@gmail.com",

    description="A complete fuzzy clustering framework",
    long_description=long_description,
    long_description_content_type="text/markdown",

    license="gpl-v3",
    url="https://github.com/NicolasBizzozzero/clustering",
    download_url="",

    # See https://pypi.org/classifiers
    # or https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],

    # What does your project relate to?
    keywords=["cli", "clustering", "fuzzy", "fuzzy_clustering", "research",
              "upmc", "sorbonne", "sorbonne_universite"],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "click >= 7.0",
        "numpy >= 1.16.1",
        "pandas >= 0.24.1",
        "scikit-learn >= 0.20.2"
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={

    },

    # Set to True if we use MANIFEST.in
    include_package_data=False,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            "clus = clustering.main:main"
        ],
    },
)
