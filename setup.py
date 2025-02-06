from setuptools import setup, find_packages

setup(
    name="irtracking",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "opencv-python",
        "rerun-sdk",
    ],
)