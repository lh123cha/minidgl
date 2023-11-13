from __future__ import absolute_import
from collections import Any, MutableMapping , namedtuple

import sys
import numpy as np
from python.needle import Tensor,Device
import base

# Frame 存储节点特征以及边特征，以torch Tensor形式存储
class Scheme(namedtuple('Scheme',['shape','dtype'])):
    pass

def infer_scheme(tensor):
    return Scheme(tuple(tensor.shape[1:]),tensor.dtype)

class Column(object):
    """
    Column作为存储nodes和edges特征的抽象对象
    使用tensor作为底层存储
    
    Parameters
    --------------
    data : Tensor
        The initial data for the column
    scheme : Scheme 
        The scheme of the column.Will be inferred if not provided
    """
    def __init__(self, data ,scheme= None):
        self.data = data
        self.scheme = scheme if scheme else infer_scheme(data)
    
    def __len__(self):
        return self.data.shape[0]
    
    @property
    def shape(self):
        return self.scheme.shape
    
    def __getitem__(self,idx):
        """Retrun the feature data at the given index
        
        Parameters
        -------------
        idx : slices or utils.Index

        Returns
        -------------
        Tensor
            The feature data
        """
        if isinstance(idx,slice):
            return self.data[idx]
        else:
            pass
    def _setitem__(self,idx,feats):
        """Update the feature data at the given index

        """
    def update(self,idx,feats,inplace=True):
        """Update the feature data at the given index

        Parameter:
        --------------
        idx : slice
            The index
        feats : Tensor
            The new features
        inplace : bool
        """
        feat_scheme = infer_scheme(feats)
        if feat_scheme != self.scheme:
            raise ValueError()

        if isinstance(idx,slice):
            raise ValueError("Cannot upadte column of schemem %s using feature of column %s" % (feat_scheme,self.scheme))
        if inplace:
            self.data[idx] = feats
        # out-place wirte
        else:
            if isinstance(idx,slice):
                part1 = self.data[0:idx.start]
                part2 = feats
                part3 = self.data[idx.stop:len(self)]
            else:
                self.data[idx] = feats
    @staticmethod
    def create(data):
        if isinstance(data,Column):
            return Column(data.data,data.scheme)
        else:
            return Column(data)
        
class Frame(MutableMapping):
    """
    Frame是feature filed到feature tensor的映射
    Frame是从特征字段到特征列的字典。所有列应该具有相同的行数（即相同的第一维度）
    
    Parameters
    ---------------------
    data : data是特征字段与特征向量组成的字典,e.g. x['feat']=[[0,1,0,1],[1,0,0,1],...,[0,0,4,1]]
    num_rows : int, optional[default=0]
    """
    def __init(self,data=None,num_rows=0):
        if data is None:
            self._columns = dict()
            self._num_rows = num_rows
        else:
            # 总是从提供的data重新创建column，因为这样不会导致两个不同的Frame
            # 使用同样的column
            self._columns = {k : Column(v) for k,v in data.item()}
            if len(self._columns) != 0:
                #获得frame中一个field中tensor的行数
                self._num_rows = len(next(iter(self._columns.values())))
            else:
                self._num_rows = 0
            # 检查每一个feature field的feature tensor函数是否相等
            for name,col in self._columns.items():
                if len(col)!=self._num_rows:
                    raise AssertionError('All columns must have same rows')
        self._initializers = {}
        self.default_initializer = None
    
    @property
    def schemes(self):
        """Return the dictionary of column name to column schemes
        """
        return {k:col.scheme for k,col in self._columns.items()}
    @property
    def num_columns(self):
        return len(self._columns)
    @property
    def 



