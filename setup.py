from distutils.core import setup
from setuptools import find_packages

setup(
    name='Standard_ML_Library',
    version='0.0.1dev',
    packages=find_packages(),
    license='BSD3 License',
    long_description=open('README.md').read(),
)