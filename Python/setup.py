import os
import sys
from setuptools import setup

version_py = os.path.join(os.path.dirname(__file__), 'phate', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 2.7 or >=3.5 required.")

setup(name='phate',
      version=version,
      description='PHATE',
      author='Daniel Burkhardt, Krishnaswamy Lab, Yale University',
      author_email='daniel.burkhardt@yale.edu',
      packages=['phate', ],
      license='GNU General Public License Version 2',
      install_requires=['numpy>=1.10.0', 'pandas>=0.18.0', 'scipy>=0.14.0',
                        'matplotlib', 'sklearn', 'future'],
      long_description=open('README.md').read(),
      )

# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
