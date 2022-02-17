##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().replace("==", ">=")

setuptools.setup(
    name="lnn",
    version="1.0",
    author="IBM Research",
    description="A `Neuro = Symbolic` framework for weighted " "real-valued logic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/LNN",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

print(setuptools)
