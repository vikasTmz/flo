#include <igl/circulation.h>
#include <igl/collapse_edge.h>
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>
#include <iostream>
#include <set>


int main(int argc, char * argv[])
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;
  
  const std::string in_name = argv[1];
  const std::string out_name = argv[2];

  MatrixXd V,OV;
  MatrixXi F,OF;
  read_triangle_mesh(in_name,OV,OF);

  decimate(OV, OF, 100, V, F);
  writeOBJ(out_name,V,F);

  // Prepare array-based edge data structures and priority queue
  VectorXi EMAP;
  MatrixXi E,EF,EI;
  igl::min_heap< std::tuple<double,int,int> > Q;
  Eigen::VectorXi EQ;
  // If an edge were collapsed, we'd collapse it to these points:
  MatrixXd C;
  int num_collapsed;

  // Function to reset original mesh and data structures
  F = OF;
  V = OV;
  edge_flaps(F,E,EMAP,EF,EI);
  C.resize(E.rows(),V.cols());
  VectorXd costs(E.rows());
  // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
  // Q.clear();
  Q = {};
  EQ = Eigen::VectorXi::Zero(E.rows());
  {
    Eigen::VectorXd costs(E.rows());
    igl::parallel_for(E.rows(),[&](const int e)
    {
      double cost = e;
      RowVectorXd p(1,3);
      shortest_edge_and_midpoint(e,V,F,E,EMAP,EF,EI,cost,p);
      C.row(e) = p;
      costs(e) = cost;
    },10000);
    for(int e = 0;e<E.rows();e++)
    {
      Q.emplace(costs(e),e,0);
    }
  }

  num_collapsed = 0;


  bool something_collapsed = false;
  // collapse edge
  const int max_iter = std::ceil(0.01*Q.size());
  for(int j = 0;j<max_iter;j++)
  {
    if(!collapse_edge(shortest_edge_and_midpoint,V,F,E,EMAP,EF,EI,Q,EQ,C))
    {
      break;
    }
    something_collapsed = true;
    num_collapsed++;
  }

  // writeOBJ(out_name,V,F);

  return 0;
}
