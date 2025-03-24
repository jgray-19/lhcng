import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lhcng",
    version="0.1.0",
    author="Joshua Gray",
    description="A package for MAD-NG utilities focused on the LHC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgray-19/lhcng",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "tfs-pandas",       # For TFS file handling
        "cpymad",    # For MAD-X interaction
        "pymadng",   # For MAD-NG routines
        "omc3",      # For optics and accelerator modeling
    ],
)
