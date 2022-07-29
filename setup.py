##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import setuptools
import pathlib


def parse_requirements(filename):
    return pathlib.Path(filename).read_text().replace("==", ">=").split("\n")


setuptools.setup(
    name="lnn",
    version="1.0",
    author="IBM Research",
    description="A `Neuro = Symbolic` framework for weighted " "real-valued logic",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/LNN",
    packages=setuptools.find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "test": parse_requirements("requirements_test.txt"),
        "plot": parse_requirements("requirements_plot.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
