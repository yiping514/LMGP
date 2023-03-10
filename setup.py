# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as fh:
    requirements = [line.strip() for line in fh.readlines()]

setup(name='lvgp-pytorch',
      version='0.1.1',
      description=('LVGP-PyTorch is a fast and robust implementation of Latent variable '
      'Gaussian process (LVGP) for modeling systems with one or more qualitative inputs.'),
      author='Suraj Yerramilli, Wei Chen, Daniel W. Apley, Ramin Bostanabad',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests","notebooks"]),
      install_requires=requirements,
      extras_requires={
          "docs":['sphinx','sphinx-rtd-theme','nbsphinx'],
          "notebooks":['ipython','jupyter','matplotlib']
      },
      zip_safe=False)