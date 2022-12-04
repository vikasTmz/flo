#include <igl/read_triangle_mesh.h>
#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readDMAT.h>
#include <igl/readOFF.h>
#include <igl/repdiag.h>
#include <igl/writeOBJ.h>

#include <iostream>

Eigen::MatrixXd V,U;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L;

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  const std::string in_name = argv[1];
  const int max_iter = std::stoi(argv[2]);
  const std::string out_name = argv[3];

  // Load a mesh in OFF format
  // igl::readOFF(argv[1], V, F);
  igl::read_triangle_mesh(in_name,V,F);

  // Compute Laplace-Beltrami operator: #V by #V
  igl::cotmatrix(V,F,L);

  // Alternative construction of same Laplacian
  SparseMatrix<double> G,K;
  // Gradient/Divergence
  igl::grad(V,F,G);
  // Diagonal per-triangle "mass matrix"
  VectorXd dblA;
  igl::doublearea(V,F,dblA);
  // Place areas along diagonal #dim times
  const auto & T = 1.*(dblA.replicate(3,1)*0.5).asDiagonal();
  // Laplacian K built as discrete divergence of gradient or equivalently
  // discrete Dirichelet energy Hessian
  K = -G.transpose() * T * G;
  cout<<"|K-L|: "<<(K-L).norm()<<endl;

  U = V;

  for (int iterations = 0; iterations < max_iter; ++iterations)
  {
    // Recompute just mass matrix on each step
    SparseMatrix<double> M;
    igl::massmatrix(U,F,igl::MASSMATRIX_TYPE_BARYCENTRIC,M);
    // Solve (M-delta*L) U = M*U
    const auto & S = (M - 0.001*L);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
    assert(solver.info() == Eigen::Success);
    U = solver.solve(M*U).eval();
    // Compute centroid and subtract (also important for numerics)
    VectorXd dblA;
    igl::doublearea(U,F,dblA);
    double area = 0.5*dblA.sum();
    MatrixXd BC;
    igl::barycenter(U,F,BC);
    RowVector3d centroid(0,0,0);
    for(int i = 0;i<BC.rows();i++)
    {
      centroid += 0.5*dblA(i)/area*BC.row(i);
    }
    U.rowwise() -= centroid;
    // Normalize to unit surface area (important for numerics)
    U.array() /= sqrt(area);

    // std::stringstream ss(std::to_string(iterations));
    // std::string newString = out_name + ss.str() + ".obj";
    // igl::writeOBJ(newString, U, F);
  }

  std::stringstream ss(std::to_string(max_iter));
  std::string newString = out_name + ss.str() + ".obj";
  igl::writeOBJ(newString,U,F);
  return 1;
}
