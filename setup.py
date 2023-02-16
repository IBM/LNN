##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import setuptools
import pathlib


setuptools.setup(
    name="lnn",
    version="1.0",
    author="IBM Research",
    description="A `Neural = Symbolic` framework for weighted " "real-valued logic",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/LNN",
    packages=setuptools.find_packages(),
    install_requires=pathlib.Path("requirements.txt").read_text().replace("==", ">="),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
