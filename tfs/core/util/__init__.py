from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfs.core.util.util import *
import tensorflow as tf
# decorators
def run_once_for_each_obj(f):
  """decorate the method which only run once for each object
  """
  def wrapper(self,*args, **kwargs):
    if not hasattr(self,'_has_run'): self._has_run={}
    assert f.__name__ not in self._has_run
    self._has_run[f.__name__] = True
    return f(self,*args, **kwargs)
  wrapper.__name__ = f.__name__
  return wrapper

# http://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): pass

import inspect
def get_arg_dict(excludes=[]):
  frame=inspect.currentframe()
  f = frame.f_back
  args, _, _, values = inspect.getargvalues(f)
  return {k:values[k] for k in args if k not in excludes}
