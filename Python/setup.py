import os
import sys
from setuptools import setup

install_requires = [
    'numpy>=1.14.0',
    'pandas>=0.21.0',
    'scipy>=1.1.0',
    'matplotlib>=2.0.1',
    'scikit-learn>=0.19.1',
    'future',
    'graphtools>=0.1.8.1',
    'tasklogger',
]

test_requires = [
    'nose2']

doc_requires = [
    'sphinx',
    'sphinxcontrib-napoleon']

version_py = os.path.join(os.path.dirname(__file__), 'phate', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 2.7 or >=3.5 required.")

readme = open('README.rst').read()

setup(name='phate',
      version=version,
      description='PHATE',
      author='Daniel Burkhardt, Krishnaswamy Lab, Yale University',
      author_email='daniel.burkhardt@yale.edu',
      packages=['phate', ],
      license='GNU General Public License Version 2',
      install_requires=install_requires,
      extras_require={
          'test': test_requires,
          'doc': doc_requires},
      test_suite='nose2.collector.collector',
      long_description=readme,
      url='https://github.com/KrishnaswamyLab/PHATE',
      download_url="https://github.com/KrishnaswamyLab/PHATE/archive/v{}.tar.gz".format(
          version),
      keywords=['visualization',
                'big-data',
                'dimensionality-reduction',
                'embedding',
                'manifold-learning',
                'computational-biology'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Visualization',
      ]
      )

# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
