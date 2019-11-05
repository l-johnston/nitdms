"""setuptools install script"""
from setuptools import setup


def readme():
    """Get readme contents"""
    with open("README.md") as f:
        return f.read()


setup(
    name="nitdms",
    version="1.0.5",
    author="Lee Johnston",
    author_email="lee.johnston.100@gmail.com",
    description="A pythonic reader for TDMS files generated by LabVIEW",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=["nitdms"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    url="https://github.com/l-johnston/nitdms",
    install_requires=["numpy"],
)
