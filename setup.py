from pathlib import Path

from setuptools import find_packages, setup

__version__ = "0.1.0"
__name__ = "hope"
__doc__ = "Evaluation of chunking"

requirements = (Path(__file__).resolve().parent / "requirements.txt").read_text()
readme = (Path(__file__).resolve().parent / "README.md").read_text()


setup(
    name=__name__,
    version=__version__,
    description=__doc__,
    long_description=readme,
    author="Henrik Brådland",
    author_email="henrik.bradland@uia.no",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    dependency_links=["https://download.pytorch.org/whl/cu118"],
)
