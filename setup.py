from setuptools import setup, find_packages

setup(
    name="lateral-project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.7.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
    ],
)
