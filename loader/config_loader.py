# -*- encoding: utf-8 -*-

"""
@File    :   config_loader.py
@Time    :   2020/05/31 18:02:58
@Author  :   silist
@Version :   1.0
@Desc    :   Loader for YAML config.
"""

import yaml
from inria_loader import InriaDataLoader

class Config(object):
    def __init__(self, cfg_path):
        self.cfg = self._load(cfg_path)
    
    def _load(self, cfg_path):
        cfg = yaml.safe_load(open(cfg_path, 'r'))
        # Check necessary parts
        for p in ['dataset', 'hog', 'train']:
            if not p in cfg:
                raise KeyError('"%s" must be contained in config!' % p)
        # Data format
        # if 'resize' in cfg['dataset'] and cfg['dataset']['resize']:
        #     cfg['dataset']['resize'] = list(map(int, cfg['dataset']['resize'].split()))
        
        return cfg
    
    def __getitem__(self, key):
        return self.cfg[key]

if __name__ == "__main__":
    cfg_path = './config/hog_svm_inria.yml'
    config = Config(cfg_path)
    print(config.cfg)
    dataloader = InriaDataLoader(config['dataset'])
    