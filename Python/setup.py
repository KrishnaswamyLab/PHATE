import os
import sys
import shutil
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('PHATE requires Python 3')

setup(name='phate',
      version='0.1',
      description='PHATE',
      author='Daniel Burkhardt, Krishnaswamy Lab, Yale University',
      author_email='daniel.burkhardt@yale.edu',
      packages=['phate',],
      license='GNU General Public License Version 2',
      install_requires=['numpy>=1.10.0', 'pandas>=0.18.0', 'scipy>=0.14.0',
          'matplotlib', 'sklearn'],
       long_description=open('README.md').read(),
      )

# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
