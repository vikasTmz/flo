#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOFF.h>


template <typename DerivedV, typename DerivedF, typename BinaryOp>
FLO_API void willmore_flow(Eigen::MatrixBase<DerivedV>& V,
                           const Eigen::MatrixBase<DerivedF>& F,
                           BinaryOp integrator)
{
  using namespace Eigen;
  std::cout << "Calculate smooth vertex normals" << std::endl;

  // Calculate smooth vertex normals
  Matrix<real, Dynamic, 3> N;
  {
    using namespace igl;
    igl::per_vertex_normals(V, F, PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);
  }

  std::cout << "Calculate the cotangent laplacian for our mesh" << std::endl;
  // Calculate the cotangent laplacian for our mesh
  SparseMatrix<real> L;
  igl::cotmatrix(V, F, L);
  L = (-L.eval());

  std::cout << "Calculate the vertex masses for our mesh" << std::endl;
  // Calculate the vertex masses for our mesh
  Matrix<real, Dynamic, 1> M;
  vertex_mass(V, F, M);

  std::cout << "Build our constraints {1, N.x, N.y, N.z}" << std::endl;
  // Build our constraints {1, N.x, N.y, N.z}
  Matrix<real, Dynamic, 4> constraints(N.rows(), 4);
  constraints.col(0) = Matrix<real, Dynamic, 1>::Ones(N.rows());
  constraints.col(1) = N.col(0);
  constraints.col(2) = N.col(1);
  constraints.col(3) = N.col(2);

  // Declare an immersed inner-product using the mass matrix
  const auto ip = [&M](const Matrix<real, Dynamic, 1>& x,
                       const Matrix<real, Dynamic, 1>& y) -> real {
    auto single_mat = (x.transpose() * M.asDiagonal() * y).eval();
    return single_mat(0, 0);
  };

  // Build a constraint basis using the Gram–Schmidt process
  Matrix<real, Dynamic, Dynamic> U;
  orthonormalize(constraints, ip, U);

  // Calculate the signed mean curvature based on our vertex normals
  Matrix<real, Dynamic, 1> H;
  signed_mean_curvature(V, L, M, N, H);
  // Apply our flow direction to the mean curvature half density
  H *= -1.f;

  Matrix<real, Dynamic, 1> HP = H;
  //// Build our constraints {1, N.x, N.y, N.z} curvature
  project_basis(HP, U, ip);
  // take a time step
  integrator(H, HP);

  // spin transform using our change in mean curvature half-density
  spin_xform(V, F, H, L);
}

