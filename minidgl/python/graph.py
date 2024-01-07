from python.needle import Tensor
from cppbackend.graph_backend import graph_backend 
import graph_structure as gs
from frame import FrameRef,Frame


class MinidglGraph(object):
    """一个Graph object包含node frame，edge frame


    """
    def __init__(self,graph_data = None,node_frame = None,
                 edge_frame=None,multigraph=False,readonly=False):
        self.readonly = readonly
        self._graph_structure = gs.create_graph_index(graph_data,multigraph)
        if node_frame is None:
            self._node_frame = FrameRef(Frame(num_rows=self.num))
        else:
            self._node_frame = node_frame
        if edge_frame is None:
            self._edge_frame = FrameRef(Frame(num_rows=self.num))

    #存储Graph为List:[src] List:[dst] List:[edge_id]

    def number_of_nodes(self):
        return self._graph_structure.number_of_nodes()
    def CreateGraph(self) -> None:
        pass
    