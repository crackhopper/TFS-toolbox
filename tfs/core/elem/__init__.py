import types

FUNC_FULL_DISPLAY=False
class Param(object):
  def __init__(self,**kwargs):
    self.__dict__ = kwargs

  def __str__(self):
    info=[]
    for k in self.__dict__:
      value = self.__dict__[k]
      if isinstance(value,types.FunctionType):
        if FUNC_FULL_DISPLAY:
          value = value.__module__ +'.'+ value.__name__
        else:
          value = value.__name__
      info.append('%s=%s'%(k,str(value)))
    return ','.join(info)

  def __repr__(self):
    return 'Param(%s)'%self

  def __getitem__(self,key):
    return self.__dict__[key]

  def __iter__(self):
    return iter(self.__dict__)

  def __eq__(self,other):
    return self.__dict__==other.__dict__

  def copy(self):
    obj = Param()
    obj.__dict__ = self.__dict__.copy()
    return obj
