from setuptools import setup

setup(name='nlsa',
      version='0.0.1',
      description='Nonlinear Laplacian Spectral Analysis in Python',
      url='https://github.com/nbren12/nlsa',
      author='Noah D. Brenowitz',
      author_email='nbren12@uw.edu',
      license='MIT',
      packages=['nlsa'],
      zip_safe=False,
      include_package_data=True,
      entry_points = {
          'console_scripts': ['nlsa=nlsa.cli:main'],
      })
