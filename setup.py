from setuptools import setup, find_packages

setup(
    name="AlphaDSSP",
    version="0.4.2",
    packages=find_packages(),
    description="Tool for converting Alphafold tar shards to a database of DSSP secondary structure information.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Noel Garber",
    author_email="ngarber93@gmail.com",
    url="https://github.com/noelgarber/AlphaDSSP",
    install_requires=[
        "numpy",
        "tqdm",
        "biopython"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
