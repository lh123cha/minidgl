#include <algorithm>
#include <vector>
#include <unordered_map>
#include <set>
#include <functional>
#include <tuple>
#include "../include/graph.h"



namespace minidgl{


namespace py = pybind11;

//注意这里PYBIND11_MODULE的名字要与编译生成的.s文件的名字相同，否则会报ImportError: dynamic module does not define module export function (PyInit_graph_backend)错误
//且使用readelf -s graph_backend.cpython-38-x86_64-linux-gnu.so | grep PyInit命令查看，.so文件中的export function name。

//包的名称一定要与二进制.so文件的名称相同

void Graph::AddVertices(minidgl_id_t num_vertices){
    adj_.resize(adj_.size()+num_vertices);
    reverse_adj_.resize(reverse_adj_.size()+num_vertices);
}
void Graph::AddEdge(minidgl_id_t src,minidgl_id_t dst){
    CHECK(HasVertex(src) && HasVertex(dst))
    << "Invalid vertices: src=" << src << " dst=" << dst;
    minidgl_id_t edge_id = num_edges_++;

    adj_[src].succ.push_back(dst);
    adj_[src].edge_id.push_back(edge_id);
    reverse_adj_[dst].succ.push_back(src);
    reverse_adj_[dst].edge_id.push_back(edge_id);
    all_edges_src.push_back(src);
    all_edges_dst.push_back(dst);
}
void Graph::AddEdges(IDArray srcs,IDArray dsts){
    for(int i=0;i<srcs.size();i++){
        AddEdge(srcs[i],dsts[i]);
    }
}
BoolArray Graph::HastheVertices(IDArray vertexid_array){
    BoolArray res;
    for(size_t i=0;i<vertexid_array.size();i++){
        res.push_back(HastheVertex(vertexid_array[i]))
    }
    return res;
}
bool Graph::HasEdgeBetween(minidgl_id_t src,minidgl_id_t dst) const{
    std::vector<minidgl_id_t> src_succ_list = adj_[src].succ;
    if(std::find(src_succ_list.begin(),src_succ_list.end(),dst)!=src_succ_list.end()){
        return true;
    }
    return false;
}
BoolArray Graph::HasEdgesBetween(IDArray src_ids,IDArray dst_ids) const{
    BoolArray a = BoolArray();
    for(int i=0;i<src_ids.size();i++){
        a.push_back(HasEdgeBetween(src_ids[i],dst_ids[i]));
    }
    return a;
}
IDArray Graph::EdgeId(minidgl_id_t src,minidgl_id_t dst){
    CHECK(HastheVertex(src)&&HastheVertex(dst))<< "Invalid vertex src=" << src <<" dst=" << dst;
    auto succ = adj_[src].succ;
    IDArray edgeid_list;
    for(size_t i=0;i<succ){
        if(succ[i]==dst){
            edgeid_list.push_back(adj_[src].edge_id[i]);
        }
    }
    return edgeid_list;
}
Graph::EdgeArray Graph::EdgesIds(IDArray src_array,IDArray dst_array){
    //broadcast
    const int srclen = src_array.size();
    const int dstlen = dst_array.size();
    CHECK((srclen == dstlen) || (srclen == 1) || (dstlen == 1))
    << "Invalid src and dst id array.";
    const int64_t src_stride = (srclen==1&&dstlen!=1) ? 0:1;
    const int64_t dst_stride = (dstlen==1&&srclen!=1) ? 0 : 1;
    std::vector<minidgl_id_t> src,dst,eid;

    for(int i=0,j=0;i<srclen&&j<dstlen;i+=src_stride,j+=dst_stride){
        const minidgl_id_t src_id = src_array[i],dst_id = dst_array[j];
        CHECK(HastheVertex(src_id)&&HastheVertex(dst_id));
        const auto& succ = adj_[src_id].succ
        for(size_t k=0;k<=succ.size();k++){
            if(succ[k]==dst_id){
                src.push_back(src_id);
                dst.push_back(dst_id);
                eid.push_back(adj_[src_id].edge_id[k]);
            }
        }
    }
    return Graph::EdgeArray(src,dst,eid);
}
namespace py = pybind11;
PYBIND11_MODULE(graph_backend, m) {
    py::class_<EdgeList>(m,"EdgeList")
        .def(py::init<>())
        .def_readwrite("succ",&EdgeList::succ)
        .def_readwrite("pred",&EdgeList::pred)
        .def("appendsucc",&EdgeList::appennd_succ)
        .def("appendpred",&EdgeList::append_pred);
    // py::bind_vector<std::vector<EdgeList>>(m, "EdgeListVector");
    // 直接bind_vector会出现递归调用错误
    m.def("create_edge_list_vector", []() {
        return std::vector<EdgeList>();
    });
    py::class_<EdgeListVector>(m,"ELVector")
        .def(py::init<>())
        .def("append",(void(EdgeListVector::*)(const EdgeList &)) & EdgeListVector::push_back)
        .def("__len__",[](const EdgeListVector& v){
            return v.size();
        })
        .def(
            "__iter__",
            [](EdgeListVector &v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>())
        .def("__getitem__",[](const EdgeListVector& v,int pos){
            return v[pos];
        });
    py::class_<Graph>(m, "Graph")
        .def(py::init<bool>())
        .def_property_readonly("adj", &Graph::Getadjlist,py::return_value_policy::reference_internal)
        .def_property_readonly("reverseadj",&Graph::Getreveradjlist,py::return_value_policy::reference_internal)
        .def_property_readonly("all_edges_src",&Graph::Get_all_edges_src)
        .def_property_readonly("all_edges_dst",&Graph::Get_all_edges_dst)
        .def_property("num_edges",&Graph::Getnumedges,&Graph::Setnumedges)
        .def_property("is_multigraph",&Graph::Getmultigraph,&Graph::Setmultigraph)
        .def("AddVertices",&Graph::AddVertices)
        .def("AddEdge",&Graph::AddEdge)
        .def("AddEdges",&Graph::AddEdges)
        .def("HastheVertices",&Graph::HastheVertices)
        .def("HastheVertex",&Graph::HastheVertex)
        .def("HasEdgeBetween",&Graph::HasEdgeBetween)
        .def("HasEdgesBetween",&Graph::HasEdgesBetween)
        .def("EdgeId",&Graph::EdgeId)
        .def("EdgeIds",&Graph::EdgeIds);
}

}
