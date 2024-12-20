from setuptools import setup, find_packages

setup(
    name="irtracking",
    version="0.1.0",
    package_dir={"": "irtracking"},
    packages=find_packages(where="irtracking"),
    python_requires=">=3.8",
)