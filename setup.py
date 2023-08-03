import setuptools
from stroke_outcome import __version__

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

# this is documentation...
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stroke-outcome",
    # set version number using value read in from __init__()
    version=__version__,
    author="Anna Laws",
    author_email="a.laws2@exeter.ac.uk",
    license="MIT License (MIT)",
    description="Stroke modelling: outcome calculations with time to treatment.",
    # read in above from readme.md and will appear on PyPi
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stroke-modelling/stroke-outcome",
    packages=setuptools.find_packages(),
    # if true look in MANIFEST.in for data files to include
    include_package_data=True,
    # 2nd approach to include data is include_package_data=False
    package_data={"stroke_outcome": ["data/*.csv"]},
    # these are for PyPi documentation
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.17',
    # pip requirements read in above
    install_requires=requirements,
)
