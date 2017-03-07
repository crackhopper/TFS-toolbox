from os.path import expanduser
from tfs.core.util import Singleton
import json
import os
_home = expanduser("~")
class GlobalConfig(Singleton):
  _cfgfile = _home+'/.tfs.json'
  def __init__(self):
    if os.path.exists(self._cfgfile):
      f = open(self._cfgfile,'r')
      self._cfg = json.load(f)
      f.close()
    else:
      self._cfg={}

    # default configuration for submodules
    self.dataset = DatasetConfig(self['dataset'])

  def __del__(self):
    f = open(self._cfgfile,'w')
    json.dump(self._cfg,f)
    f.close()

  def __getitem__(self,key):
    if key in self._cfg:
      return self._cfg[key]
    else:
      self._cfg[key]={}
      return self._cfg[key]

class DatasetConfig(object):
  def __init__(self,cfgobj):
    if 'basedir' not in cfgobj:
      cfgobj['basedir'] = _home+'/tfs_data/'
    if 'loaded' not in cfgobj:
      cfgobj['loaded'] = {}
    if 'downlaod' not in cfgobj:
      cfgobj['download'] = _home+'/tfs_data/'+"download"
    self.cfg = cfgobj

  def getdir(self,dataset):
    typename = type(dataset).__name__
    if typename in self.cfg['loaded']:
      return self.cfg['loaded'][typename]
    return None

  def setdir(self,dataset,_dir):
    typename = type(dataset).__name__
    self.cfg['loaded'][typename] = _dir

  @property
  def base_dir(self):
    return self.cfg['basedir']

  @property
  def download_dir(self):
    return self.cfg['download']

