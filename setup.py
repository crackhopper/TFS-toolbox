import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
  user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

  def initialize_options(self):
    TestCommand.initialize_options(self)
    self.pytest_args = []

  def run_tests(self):
    #import here, cause outside the eggs aren't loaded
    import pytest
    errno = pytest.main(self.pytest_args)
    sys.exit(errno)

setup(
  name='tfs',
  author='crackhopper',
  author_email='crackhopper@gmail.com',
  version='0.1',
  url='https://github.com/crackhopper/TFS-toolbox',
  packages=find_packages(),
  install_requires=[
    'numpy>=1.12.0',
    'tensorflow>=0.12',
    'scikit-learn>=0.18.1'
  ],
  tests_require=[
    'pytest'
  ],
  cmdclass = {
    'pytest': PyTest
  },
)
