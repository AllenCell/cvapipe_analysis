#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black",
    "flake8",
    "flake8-debugger",
    "pytest",
    "pytest-cov",
    "pytest-raises",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version",
    "coverage",
    "ipython",
    "m2r2",
    "pytest-runner",
    "sphinx",
    "furo",
    "tox",
    "twine",
    "wheel",
]

requirements = [
    "jupyter",
    "aicsshparam",
    "aicscytoparam",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ]
}

setup(
    author="Matheus Viana",
    author_email="matheus.viana@alleninstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only"
    ],
    description="Analysis pipeline usinf in Integrated intracellular organization and its variations in human iPS cells",
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cvapipe_analysis",
    name="cvapipe_analysis",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.9",
    setup_requires=setup_requirements,
    test_suite="cvapipe_analysis/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/cvapipe_analysis",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.2.0",
    zip_safe=False,
)