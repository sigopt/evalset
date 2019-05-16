from setuptools import setup

install_requires=['numpy>=1.10.1', 'scipy>=0.16.0']

setup(
  name='evalset',
  version='1.2.3',
  description='Benchmark suite of test functions suitable for evaluating black-box optimization strategies',
  author='SigOpt',
  author_email='support@sigopt.com',
  url='https://sigopt.com/',
  packages=['evalset'],
  install_requires=install_requires,
  classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
  ]
)
