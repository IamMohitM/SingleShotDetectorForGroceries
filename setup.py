from setuptools import setup, find_packages

setup(
    name='infilect_object_detection',
    author='Mohit Motwani',
    packages=find_packages(include=["src", "src.*"]),
    install_requires=['torch-tools', 'tensorboard', 'torch', 'torchvision', 'pandas']
)
