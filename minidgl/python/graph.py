from python.needle import Tensor
from cppbackend.graph_backend import graph_backend 



class MinidglGraph(object):
    """一个Graph object包含node frame，edge frame


    """
    def __init__(self,graph_Data = None,node_frame = None,
                 edge_frame=None,multigraph=False,readonly=False):
        self.readonly = readonly
        self._graph = 

    #存储Graph为List:[src] List:[dst] List:[edge_id]
    def CreateGraph(self) -> None:
        pass
    