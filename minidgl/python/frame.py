from __future__ import absolute_import
from ast import Num
from collections.abc import MutableMapping
from collections import namedtuple
import sys
from tempfile import tempdir
import numpy as np
import utils
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
        #TODO: 支持idx为Index object的情况
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
        
    def __contains__(self, name):
        """Return whether the column name exists."""
        return name in self._frame
    def __iter__(self):
        """Return the iterator of the columns."""
        return iter(self._frame)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    def __len__(self):
        """Return the number of columns."""
        return self.num_columns
    def keys(self):
        """Return the keys."""
        return self._frame.keys()
    def __getitem__(self, key):
        """Get data from the frame

        

        """
        if isinstance(key,str):
            return self.select_columnn(key)
        elif isinstance(key,slice) and key == slice(0,self.num_rows):
            return self
        elif isinstance(key,utils.Index) and key.is_slice(0,self.num_rows):
            return self
        else:
            return self.select_rows(key)


    def is_contiguous(self):
        """Check whether this refer to a contiguous range of rows
        """
        # minidglGraph接受的节点slice默认slice.step是None
        return isinstance(self._index_data,slice)
    
    def is_span_whole_column(self):
        """Check Whether this refers to all the rows
        """
        return self.is_contiguous() and self.num_rows == self._frame.num_rows
    def index(self):
        """Return the index object

        Returns
        -----------
        utils.Index
            The index
        """
        if self._index is None:
            if self.is_contiguous():
                self._index = utils.toindex(
                    np.arange(self._index_data.start,self._index_data.stop)
                )
            else:
                self._index = utils.toindex(self._index_data)
        return self._index
    
    def index_or_slice(self):
        """Return index object or slice

        Returns
        ----------
        utils.Index or slice
            The index or slice data
        """
        if self.index_or_slice is None:
            if self.is_contiguous():
                self._index_or_slice = self._index_data
            else:
                self._index_or_slice = utils.toindex(self._index_data)
        return self._index_or_slice
            
    def select_columnn(self,name):
        """ Return the column of given name

        Parameters
        ------------
        name : str
            The column name
        
        Returns 
        -----------
        Tensor
            The column data.
        """
        col = self._frame[name]
        if self.is_span_whole_column():
            return col.data
        else:
            return col[self.index_or_slice()]
        
    def transform_rows(self,query:utils.Index):
        """将self.Index()加上key Index,在原有的Index基础上选择子集。

        Parameters
        ---------------
        query : Index
            the index object

        Returns
        ----------------
        Index

        """
        if self.is_contiguous():
            start = self._index_data.start
            if start==0:
                return query
            elif isinstance(query,slice):
                return slice(query.start+start,query.stop+start)
            else:
                temp = query.tonumpy()
                return utils.toindex(temp+start)
        else:
            temp_index = query.tonumpy()
            frame_index = self.index().tonumpy()
            return utils.toindex(frame_index[temp_index])
    def select_rows(self,key : utils.Index):
        """Return the rows of the given index

        Parameters
        ------------
        key : Index
            The index obejct 
        
        Returns
        ------------
        FrameRef
            The sub Frame on the key index 
            
        """
        after_transform_rows = self.transform_rows(key)
        return utils.LazyDict(lambda key : self._frame[key][after_transform_rows],keys=self.keys())
    
    
    def update_column(self,name,data,inplace=True):
        """update the column given name and data
        
        Parameters
        ------------
        name : str 
            The column name 
        data : Tensor
            The given column data
        """    
        if self.is_span_whole_column():
        #update the whole Frame whith a new Frame
            new_column_data = Column.create(data=data)

            if self._frame.num_rows==0:
                #empty frame
                self._index_data = slice(0,len(new_column_data))
            self._frame[name] = new_column_data
            
        else:
            #if name in frame
            if name not in self._frame:
                ctx = data.device
                self._frame.add_column(name,infer_scheme(data),ctx)
            #使用Column的update更新指定索引下的数据，
            #Frame的update_column只能更新指定域名下的全部数据
            #使用Frame的update_column也可以如下：
            # new_column = self._frame[name].update(self._index_or_slice,data,inplace)
            # self._frame.update_column(name,new_column)
            self._frame[name].update(self._index_or_slice,data,inplace)
    def add_rows(self,num_rows):
        """Add rows to all Columns in the frame
        equal to add new nodes to the graph
        
        Parameter
        -------------
        num_rows : int
            number of blank rows given to add to the frame
        """
        if not self.is_span_whole_column():
            raise ValueError("The added FrameRef must span all Frame")
        self._frame.add_rows(num_rows)
        if self._index.slice_data is not None:
            slc = self._index.slice_data
            self._index = utils.toindex(slice(slc.start,slc.stop+num_rows))
        else:
            #非slice类型的index
            #需要在原index之后加上顺序的num_rows个索引
            newidxdata = self._index.tonumpy()
            newdata = np.arange(self.num_rows,self.num_rows+num_rows)
            self._index = utils.toindex(np.concatenate((newidxdata,newdata),axis=0))

    def update_rows(self,query:utils.Index , data , inplace=True):
        """Update the given rows

        注意query是在自己FrameRef上的查询，需要转换到底层的Frame

        Parameters
        -------------
        query : Index

        data : dict or Column

        inplace : bool
        """
        newrows = self.transform_rows(query)
        for k,v in data.items():
            if k not in self._frame:
                temp = FrameRef(self._frame,newrows)
                temp.update_column(k,v,inplace)
            else:
                self._frame[k].update(newrows,v,inplace)
    def __delitem__(self, key):
        """Delete data in the frame.

        If the provided key is a string, the corresponding column will be deleted.
        If the provided key is an index object or a slice, the corresponding rows will
        be deleted.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not reflect on the other. However, deleting columns is real.

        Parameters
        ----------
        key : str or utils.Index
            The key.
        """
        if not isinstance(key, (str, utils.Index)):
            raise ValueError('Argument "key" must be either str or utils.Index type.')
        if isinstance(key, str):
            del self._frame[key]
        else:
            self.delete_rows(key)
    def delete_rows(self, query):
        """Delete rows.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not reflect on the other. By contrast, deleting columns is real.

        Parameters
        ----------
        query : utils.Index
            The rows to be deleted.
        """
        query = query.tonumpy()
        index = self._index.tonumpy()
        self._index = utils.toindex(np.delete(index, query))

    def append(self, other):
        """Append another frame into this one.

        Parameters
        ----------
        other : dict of str to tensor
            The data to be appended.
        """
        old_nrows = self._frame.num_rows
        self._frame.append(other)
        new_nrows = self._frame.num_rows
        # update index
        if (self._index.slice_data is not None
                and self._index.slice_data.stop == old_nrows):
            # Self index is a slice and index.stop is equal to the size of the
            # underlying frame. Can still use a slice for the new index.
            oldstart = self._index.slice_data.start
            self._index = utils.toindex(slice(oldstart, new_nrows))
        else:
            # convert it to user tensor and concat
            selfidxdata = self._index.tonumpy()
            newdata = np.arange(old_nrows, new_nrows)
            self._index = utils.toindex(F.cat([selfidxdata, newdata], dim=0))
            self._index = utils.toindex(np.concatenate((selfidxdata,newdata),axis=0))
    def clear(self):
        """Clear the frame."""
        self._frame.clear()
        self._index = utils.toindex(slice(0, 0))
    