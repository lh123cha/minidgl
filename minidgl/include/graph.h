#include <cstdint>
#include <utility>
#include <tuple>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>



#define CHECK

using namespace std;


typedef uint64_t minidgl_id_t;

struct EdgeList
{   
    EdgeList(){};
    
    /* data */
    void appennd_succ(minidgl_id_t v){
        succ.push_back(v);
    };
    void append_pred(minidgl_id_t v){
        pred.push_back(v);
    };
    std::vector<minidgl_id_t> succ;
    std::vector<minidgl_id_t> edge_id;
};
PYBIND11_MAKE_OPAQUE(std::vector<EdgeList>);
using EdgeListVector = std::vector<EdgeList>;

namespace minidgl{

    
    typedef std::vector<minidgl_id_t> IDArray;
    typedef std::vector<int> DegreeArray;
    typedef std::vector<bool> BoolArray;
    typedef std::vector<int> IntArray;

    class Graph;
    class GraphOp;


    class Graph{
        //只提供存储
        protected:
            friend class GraphOp;

            EdgeListVector adj_;
            
            EdgeListVector reverse_adj_;

            
            std::vector<minidgl_id_t> all_edges_src;
            
            std::vector<minidgl_id_t> all_edges_dst;

            bool is_multigraph_;

            uint64_t num_edges_;

        public:
            typedef struct 
            {
                IDArray src,dst,id;
            }EdgeArray;
            
            explicit Graph(bool multigraph = false) : is_multigraph_(multigraph){}

            Graph(const Graph& other) = default;

            Graph& operator=(const Graph& other) = default;

            // Default 
            ~Graph() = default;

            EdgeListVector Getadjlist(){
                return adj_;
            }
            EdgeListVector Getreveradjlist(){
                return reverse_adj_;
            }
            std::vector<minidgl_id_t> Get_all_edges_src(){
                return all_edges_src;
            }
            std::vector<minidgl_id_t> Get_all_edges_dst(){
                return all_edges_dst;
            }
            const bool Getmultigraph(){
                return is_multigraph_;
            }
            void Setmultigraph(const bool multigraph){
                is_multigraph_ = multigraph;
            }
            const uint64_t Getnumedges(){
                return num_edges_;
            }
            void Setnumedges(const uint64_t numedges){
                num_edges_ = numedges;
            }
            /**
             * \brief Add vertices to the graph
             * \note 
            */
           void AddVertices(minidgl_id_t num_vertices);
            /*
            * \brief Add an edge to graph
            */
           void AddEdge(minidgl_id_t src,minidgl_id_t dst);
            /**
             *  \brief Add a set of edges to the graph
             *  \param srcs vector<int>
             *  \param dsts vector<int>
            */
           void AddEdges(IDArray srcs,IDArray dsts);

           void Clear(){
                adj_.clear();
                reverse_adj_.clear();   
                all_edges_src.clear();
                all_edges_dst.clear();
                num_edges_ = 0;
           }

           bool IsMultigraph() const{
                return is_multigraph_;
           }
           uint64_t NumVertices() const{
                return adj_.size();
           }
           
           uint64_t NumEdges() const {
                return num_edges_;
           }
           
           bool HastheVertex(minidgl_id_t vertexid){
            return vertexid<NumVertices();
           }
           BoolArray HastheVertices(IDArray vertexid_array);

           bool HasEdgeBetween(minidgl_id_t src,minidgl_id_t dst) const;

           BoolArray HasEdgesBetween(IDArray src_ids,IDArray dst_ids) const;

           IDArray EdgeId(minidgl_id_t src,minidgl_id_t dst);

           EdgeArray EdgesIds(IDArray src_array,IDArray dst_array);

           EdgeArray edges(bool sorted);


    };

    struct SubGraph{
        //backend full Graph
        Graph graph;

        IDArray induced_vertices;

        IDArray induced_edges;

    };
}










