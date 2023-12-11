import numpy as np
from minidgl.cppbackend.graph_backend import graph_backend
import scipy
import indexutils as utils

class GraphStructure(object):
    """Graph structure object

    Parameters
    --------------
    handle : C++ backend graph structure
    """
    def __init__(self,handler):
        self.handler = handler
        self.cache = {}
    def clear(self):
        self.cache.clear()
        pass
    def add_nodes(self,num_nodes):
        pass
    def add_edge(self,u,v):
        pass
    def number_of_nodes(self):
        pass
    def has_node(self,nodeid):
        pass
    def has_edge_between(self,u,v):
        pass
    def has_edges_between(self,src_array,dst_array):
        pass
    def edge_id(self,u,v):
        pass
    def edge_ids(self,src_array,dst_array):
        pass
    def add_edges(self,src_array,dst_array):
        pass
    def edges(self,sorted=False):
        """Return all edges
        """
        pass
    
    def from_edge_list(self,edge_list):
        """Convert edge list to graphstruture

        Parameter
        -----------
        edge_list:
            list of tuple (src,dst)
        """
        self.clear()
        src,dst = zip(*edge_list)
        src = np.array(src)
        dst = np.array(dst)
        num_nodes = max(src.max(),dst.max())+1
        num_edges = len(src)
        min_nodes_id = min(src.min(),dst.min())
        self.add_nodes(num_nodes)
        self.add_edges(src_array=src,dst_array=dst)
    def from_scipy_sparse_matrix(self,adj):
        self.clear()
        self.add_nodes(adj.shape[0])
        adj_coo = adj.tocoo()
        src = utils.toindex(adj_coo.row)
        dst = utils.toindex(adj_coo.col)
        self.add_edges(src,dst)


def create_graph_index(graph_data = None,multigraph = False):
    """Create a graph index object
    
    Parameters
    -----------
    graph_data : (list tuple),GraphStruture,
        Data
    
    """
    if isinstance(graph_data,GraphStructure):
        return graph_data
    handler = graph_backend.Graph(multigraph)
    gs = GraphStructure(handler)
    if graph_data is None:
        return gs
    if isinstance(graph_data,(list,tuple)):
        gs.from_edge_list(graph_data)
    elif isinstance(graph_data,(list)):
        gs.