from __future__ import absolute_import
from collections.abc import MutableMapping
from collections import namedtuple
import sys
import numpy as np
import sys
sys.path.append('../python')
import needle as ndl
import needle.backend_ndarray as bn

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
    def __init__(self, data : ndl.Tensor ,scheme= None):
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
        self.update(idx, feats, inplace=True)
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
    def extend(self,feats : ndl.Tensor,feat_scheme=None):
        """在Column中添加数据
            [[0,1,11],[11,2,2]]-----(添加一行)---->[[0,1,11],[11,2,2],[2,2,3]]
        """
        if feat_scheme is None:
            feat_scheme = infer_scheme(feats)
        if feat_scheme != self.scheme:
            raise ValueError("Cannot update column of scheme %s using feature of scheme %s."
                    % (feat_scheme, self.scheme))
        #初始化Tensor设置device为self.data.device
        feats = ndl.Tensor(feats,device=self.data.device)
        self.data = ndl.ops.cat((self.data,feats),dim=0)

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
        num_rows代表data中每个feat field中的节点数。
    """
    def __init__(self,data=None,num_rows=0):
        if data is None:
            self._columns = dict()
            self._num_rows = num_rows
        else:
            # 总是从提供的data重新创建column，因为这样不会导致两个不同的Frame
            # 使用同样的column
            self._columns = {k : Column(v) for k,v in data.items()}
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
        ret = {}
        for k,col in self._columns.items():
            ret[k] = col.scheme
        return ret
    @property
    def num_columns(self):
        return len(self._columns)
    @property
    def num_rows(self):
        return self._num_rows
    def __contains__(self, name):
        return name in self._columns
    
    def __getitem__(self, name):
        return self._columns[name]
    def __setitem__(self, name,data):
        return self.update_column(name,data)
    def __delitem__(self, name) -> None:
        del self._columns[name]
    def __iter__(self):
        return iter(self._columns)
    def __len__(self):
        return self.num_columns
    
    def _append(self,other):
        if self._num_rows == 0:
            #本身为空frame，则直接赋值
            self._columns = {key:Column.create(value) for key,value in other.items()}
        else:
            for k,v in other.items():
                #判断重复的feature field
                if k not in self._columns:
                    self.add_column(k,v.scheme,v.device)
                #重复的feature field则添加扩展feature tensor
                self._columns[k].extend(v.data,v.scheme)
    def add_column(self,name:str , scheme:Scheme, ctx : bn.BackendDevice):
        """Add a new column to the Frame

            Use init.rand to initialize the column

            相当于添加新的节点特征field
        Parameter
        --------------
        name : Str
            The column name.
        scheme : Scheme
            The column scheme 
        ctx : 
            Device used in needle gpu/cpu           
        """
        if name in self._columns:
            raise NameError('Column "%s" already exist')
        temp_shape = (self.num_rows,)+scheme.shape
        print(temp_shape)
        #rand函数参数需要使用*shape
        init_data = ndl.init.rand(*temp_shape,device=ctx)
        self._columns[name] = Column(init_data,scheme)
    def add_rows(self,num_rows):
        """Add blank rows to the frame
        
        For existing fields , 
        相当于
        Parameters
        ----------------
        num_rows:
            The number of new rows
        """
        new_feats = {}
        for k,v in self._columns.items():
            scheme =v.scheme
            device = v.data.device
            temp_shape = (num_rows,)+scheme.shape
            init_data = ndl.init.rand(*temp_shape,device=device)
            new_feats[k] = init_data
        self._append(Frame(new_feats))
        self._num_rows += num_rows
    
    def update_column(self,name,data):
        """Add or replace the column with new name and data

        Parameters
        ------------
        name : str
            The column name
        data : Column
            The column data
        """
        col = Column.create(data)
        if len(col) != self.num_rows:
            raise ValueError('Expected data to have %d rows, got %d.' %
                           (self.num_rows, len(col)))
        self._columns[name] = col
    def show_column(self,name):
        """Debug used,print the given column data

        Parameters
        --------------
        name : str
            The given column name
        """
        if name not in self._columns:
            raise ValueError('Given name :%s is not in Frame' % (name))
        print(self._columns[name].data)

    def append(self,other):
        """Append other frame's  data to this frame

        If the self column is empty,It will use the other's column.
        Otherwise,the given data should cotain all the column keys of 
        the self column

        Parameters
        ----------------
        other : Frame or dict
            The Frame data to be appended
        """
        if not isinstance(other,Frame):
            other = Frame(other)
        self._append(other)
        self._num_rows+=other.num_columns
    def clear(self) -> None:
        self._num_rows = 0
        self._columns = {}
    def keys(self):
        return self._columns.keys()

class FrameRef(MutableMapping):
    """Reference object to a frame on a subset of rows.
    Frame的上层抽象，用于存储整个图的Frame的一个子集，例如仅包含节点1,2,5的特征
    的Frame子集

    Parameters
    ----------
    frame : Frame, optional
        The underlying frame. If not given, the reference will point to a
        new empty frame.
    index : iterable, slice, or int, optional
        The rows that are referenced in the underlying frame. If not given,
        the whole frame is referenced. The index should be distinct (no
        duplication is allowed).

        Note that if a slice is given, the step must be None.
    """
    def __init__(self, frame=None, index=None):
        self._frame = frame if frame is not None else Frame()
        if index is None:
            # _index_data can be either a slice or an iterable
            self._index_data = slice(0, self._frame.num_rows)
        else:
            # TODO(minjie): check no duplication
            self._index_data = index
        self._index = None
        self._index_or_slice = None

    @property
    def schemes(self):
        """Return the frame schemes.

        Returns
        -------
        dict of str to Scheme
            The frame schemes.
        """
        return self._frame.schemes

    @property
    def num_columns(self):
        """Return the number of columns in the referred frame."""
        return self._frame.num_columns
    @property
    def num_rows(self):
        """Return the number of rows referred."""
        if isinstance(self._index_data, slice):
            # NOTE: we always assume that slice.step is None
            return self._index_data.stop - self._index_data.start
        else:
            return len(self._index_data)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
