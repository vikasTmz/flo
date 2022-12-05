#include <iostream>
#include <numeric>
#include <sstream>

#include <igl/write_triangle_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/adjacency_matrix.h>
#include <igl/writeOBJ.h>
#include <igl/decimate.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

using namespace Eigen;

template <typename T>
struct ForwardEuler
{
  T tao = 0.95f;

  ForwardEuler(T i_tao) : tao(std::move(i_tao))
  {
  }

  void operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& i_x,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1>& i_dx) const
  {
    i_x += i_dx * tao;
  }
};

Matrix<float, 4, 1>
hammilton_product(const Matrix<float, 4, 1>& i_rhs,
                  const Matrix<float, 4, 1>& i_lhs)
{
  using namespace Eigen;
  const auto a1 = i_rhs.w();
  const auto b1 = i_rhs.x();
  const auto c1 = i_rhs.y();
  const auto d1 = i_rhs.z();
  const auto a2 = i_lhs.w();
  const auto b2 = i_lhs.x();
  const auto c2 = i_lhs.y();
  const auto d2 = i_lhs.z();
  // W is last in a vector
  return Matrix<float, 4, 1>(a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
                            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2);
}

Matrix<float, 4, 1>
hammilton_product(const Matrix<float, 3, 1>& i_rhs,
                  const Matrix<float, 3, 1>& i_lhs)
{
  using namespace Eigen;
  return hammilton_product(
    Matrix<float, 4, 1>{i_rhs.x(), i_rhs.y(), i_rhs.z(), 0.},
    Matrix<float, 4, 1>{i_lhs.x(), i_lhs.y(), i_lhs.z(), 0.});
}

template <int R, int C>
void insert_block_sparse(const Matrix<float, R, C>& i_block,
                                 SparseMatrix<float>& i_mat,
                                 int i_x,
                                 int i_y)
{
  // might be a dynamic matrix so need to get dim from member funcs
  int n_rows = i_block.cols();
  int n_cols = i_block.rows();
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
    {
      i_mat.coeffRef(i_x * n_rows + r, i_y * n_cols + c) = i_block(r, c);
    }
}


Matrix<float, 4, 4>
quat_to_block(const Matrix<float, 4, 1>& i_quat)
{
  using namespace Eigen;
  const auto a = i_quat.w();
  const auto b = i_quat.x();
  const auto c = i_quat.y();
  const auto d = i_quat.z();

  Matrix<float, 4, 4> block;
  block << a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a;
  return block;
}

Matrix<float, 4, 1>
conjugate(const Matrix<float, 4, 1>& i_quat)
{
  return Matrix<float, 4, 1>(-i_quat.x(), -i_quat.y(), -i_quat.z(), i_quat.w());
}


inline SparseMatrix<float>
to_real_quaternion_matrix(const SparseMatrix<float>& i_real_matrix)
{
  SparseMatrix<float> quat_matrix(i_real_matrix.rows() * 4,
                                        i_real_matrix.cols() * 4);

  Matrix<int, Dynamic, 1> dim(i_real_matrix.cols() * 4, 1);
  for (int i = 0; i < i_real_matrix.cols(); ++i)
  {
    auto num = 4 * i_real_matrix.col(i).nonZeros();
    dim(i * 4 + 0) = num;
    dim(i * 4 + 1) = num;
    dim(i * 4 + 2) = num;
    dim(i * 4 + 3) = num;
  }
  quat_matrix.reserve(dim);

  using iter_t = SparseMatrix<float>::InnerIterator;
  for (int i = 0; i < i_real_matrix.outerSize(); ++i)
  {
    for (iter_t it(i_real_matrix, i); it; ++it)
    {
      auto r = it.row();
      auto c = it.col();

      Matrix<float, 4, 1> real_quat(0.f, 0.f, 0.f, it.value());
      auto block = quat_to_block(real_quat);
      insert_block_sparse(block, quat_matrix, r, c);
    }
  }
  return quat_matrix;
}

// template <typename DerivedF,
//           typename DerivedVVAK,
//           typename DerivedVVA,
//           typename DerivedVVV,
//           typename DerivedVVCV>
void vertex_vertex_adjacency(const Matrix<int, Eigen::Dynamic, 3>& F,
                                     Matrix<int, Dynamic, 1> & VVAK,
                                     Matrix<int, Dynamic, 1> & VVA,
                                     Matrix<int, Dynamic, 1> & VVV,
                                     Matrix<int, Dynamic, 1> & VVCV)
{
  SparseMatrix<int> A;
  igl::adjacency_matrix(F, A);

  // Get the vertex valences
  VVV.resize(A.cols());
  VVCV.resize(A.cols() + 1);
  VVCV(0) = 0;
  for (int i = 0; i < A.cols(); ++i)
  {
    // Get the valence for this vertex
    VVV(i) = A.col(i).nonZeros();
    // Accumulate the valence for this vertex
    VVCV(i + 1) = VVCV(i) + VVV(i);
  }

  // Get the vertex adjacencies
  VVAK.resize(VVCV(VVCV.size() - 1));
  VVA.resize(VVCV(VVCV.size() - 1));
  for (int k = 0; k < A.outerSize(); ++k)
  {
    int j = 0;
    for (SparseMatrix<int>::InnerIterator it(A, k); it; ++it, ++j)
    {
      VVAK(VVCV(k) + j) = it.col();
      VVA(VVCV(k) + j) = it.row();
    }
  }
}

// template <typename DerivedV,
//           typename DerivedF,
//           typename DerivedVV,
//           typename DerivedA,
//           typename DerivedP>
void intrinsic_dirac(const Matrix<float, Eigen::Dynamic, 3>& V,
                             const Matrix<int, Eigen::Dynamic, 3>& F,
                             const Matrix<int, Dynamic, 1>& VV,
                             const Matrix<float, Dynamic, 1>  & A,
                             const Matrix<float, Dynamic, 1>& P,
                             SparseMatrix<float>& D)
{
  // Find the max valence
  const int nnz = V.rows() * VV.maxCoeff() * 16;
  const int dim = V.rows() * 4;
  // Allocate for our Eigen problem matrix
  D.conservativeResize(dim, dim);
  D.reserve(VectorXi::Constant(dim, VV.maxCoeff() * 4));

  // For every face
  for (int k = 0; k < F.rows(); ++k)
  {
    // Get a reference to the face vertex indices
    const auto& f = F.row(k);
    // Compute components of the matrix calculation for this face
    auto a = -1.f / (4.f * A(k));
    auto b = 1.f / 6.f;
    auto c = A(k) / 9.f;

    // Compute edge vectors as imaginary quaternions
    std::array<Matrix<float, 4, 1>, 3> edges;
    // opposing edge per vertex i.e. vertex one opposes edge 1->2
    edges[0].head<3>() = V.row(f(2)) - V.row(f(1));
    edges[1].head<3>() = V.row(f(0)) - V.row(f(2));
    edges[2].head<3>() = V.row(f(1)) - V.row(f(0));
    // Initialize real part to zero
    edges[0].w() = 0.f;
    edges[1].w() = 0.f;
    edges[2].w() = 0.f;

    // increment matrix entry for each ordered pair of vertices
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
      {
        // W comes first in a quaternion but last in a vector
        Matrix<float, 4, 1> cur_quat(D.coeff(f[i] * 4 + 1, f[j] * 4),
                                    D.coeff(f[i] * 4 + 2, f[j] * 4),
                                    D.coeff(f[i] * 4 + 3, f[j] * 4),
                                    D.coeff(f[i] * 4 + 0, f[j] * 4));

        // Calculate the matrix component
        Matrix<float, 4, 1> q = a * hammilton_product(edges[i], edges[j]) +
                               b * (P(f(i)) * edges[j] - P(f(j)) * edges[i]);
        q.w() += P(f(i)) * P(f(j)) * c;
        // Sum it with any existing value
        cur_quat += q;

        // Write it back into our matrix
        auto block = quat_to_block(cur_quat);
        insert_block_sparse(block, D, f(i), f(j));
      }
  }
}

void similarity_xform(const SparseMatrix<float>& D,
                              Matrix<float, Dynamic, 4>& X,
                              int back_substitutions = 0)
{
  // Calculate the length of our matrix,
  // and hence the number of quaternions we should expect
  const int vlen = D.cols();
  const int qlen = vlen / 4;

  SimplicialLLT<SparseMatrix<float>, Lower> cg;
  cg.compute(D);

  // Init every real part to 1, all imaginary parts to zero
  Matrix<float, Dynamic, 1> lambda(vlen, 1);
  for (int i = 0; i < qlen; ++i)
  {
    lambda(i * 4 + 0) = 1.f;
    lambda(i * 4 + 1) = 0.f;
    lambda(i * 4 + 2) = 0.f;
    lambda(i * 4 + 3) = 0.f;
  }

  // Solve the smallest Eigen value problem DL = EL
  // Where D is the self adjoint intrinsic dirac operator,
  // L is the similarity transformation, and E are the eigen values
  // Usually converges in 3 iterations or less
  lambda.normalize();
  for (int i = 0; i < back_substitutions + 1; ++i)
  {
    lambda = cg.solve(lambda.eval());
    lambda.normalize();
  }

  X.resize(qlen, 4);
  for (int i = 0; i < qlen; ++i)
  {
    X.row(i)(0) = lambda(i * 4 + 1);
    X.row(i)(1) = lambda(i * 4 + 2);
    X.row(i)(2) = lambda(i * 4 + 3);
    X.row(i)(3) = lambda(i * 4 + 0);
  }
}

void divergent_edges(const Matrix<float, Eigen::Dynamic, 3>& V,
                             const Matrix<int, Eigen::Dynamic, 3>& F,
                             const Matrix<float, Dynamic, 4> & X,
                             const SparseMatrix<float>& L,
                             Matrix<float, Dynamic, 4>& E)
{
  using namespace Eigen;
  E.resize(V.rows(), 4);
  E.setConstant(0.f);
  // For every face
  for (int k = 0; k < F.rows(); ++k)
  {
    const auto& f = F.row(k);
    // For each edge in the face
    for (int i = 0; i < 3; ++i)
    {
      int a = f((i + 1) % 3);
      int b = f((i + 2) % 3);
      if (a > b)
        std::swap(a, b);

      const auto& l1 = X.row(a);
      const auto& l2 = X.row(b);

      Matrix<float, 3, 1> edge = V.row(a) - V.row(b);
      Matrix<float, 4, 1> e(edge[0], edge[1], edge[2], 0.f);

      constexpr auto third = 1.f / 3.f;
      constexpr auto sixth = 1.f / 6.f;

      Matrix<float, 4, 1> et =
        hammilton_product(hammilton_product(third * conjugate(l1), e), l1) +
        hammilton_product(hammilton_product(sixth * conjugate(l1), e), l2) +
        hammilton_product(hammilton_product(sixth * conjugate(l2), e), l1) +
        hammilton_product(hammilton_product(third * conjugate(l2), e), l2);

      auto cot_alpha = L.coeff(a, b) * 0.5f;
      E.row(a) -= et * cot_alpha;
      E.row(b) += et * cot_alpha;
    }
  }
}

void spin_positions(const SparseMatrix<float>& QL,
                            const Matrix<float, Dynamic, 4>& QE,
                            Matrix<float, Dynamic, Dynamic>& V)
{
  // Solve for our new positions
  // If float precision, use Cholmod solver
  // Cholmod not supported for single precision
  SimplicialLLT<SparseMatrix<float>, Lower> cg;
  cg.compute(QL);

  Matrix<float, Dynamic, 4, RowMajor> QEr = QE;

  for (int i = 0; i < QEr.size() / 4; ++i)
  {
    const float z = QEr(i * 4 + 3);
    QEr(i * 4 + 3) = QEr(i * 4 + 2);
    QEr(i * 4 + 2) = QEr(i * 4 + 1);
    QEr(i * 4 + 1) = QEr(i * 4 + 0);
    QEr(i * 4 + 0) = z;
  }
  Map<Matrix<float, Dynamic, 1>> b(QEr.data(), QEr.size());
  Matrix<float, Dynamic, 1> flat = cg.solve(b);

  V.resize((flat.size() / 4), 4);
  for (int i = 0; i < flat.size() / 4; ++i)
  {
    const float z = flat(i * 4 + 0);
    V.row(i)(0) = flat(i * 4 + 1);
    V.row(i)(1) = flat(i * 4 + 2);
    V.row(i)(2) = flat(i * 4 + 3);
    V.row(i)(3) = z;
  }

  // Remove the mean to center the positions
  const Matrix<float, 1, 4, RowMajor> average =
    V.colwise().sum().array() / V.rows();
  V.rowwise() -= average;
  // Normalize positions
  const float max_dist = std::sqrt(V.rowwise().squaredNorm().maxCoeff());
  V *= (1.f / max_dist);
}



// template <typename DerivedV, typename DerivedF, typename DerivedP>
void spin_xform(Matrix<float, Eigen::Dynamic, 3>& V,
                        const Matrix<int, Eigen::Dynamic, 3>& F,
                        const Matrix<float, Dynamic, 1>& P,
                        const SparseMatrix<float>& L)
{
  using namespace Eigen;
  // Calculate the real matrix from our quaternion edges
  auto QL = to_real_quaternion_matrix(L);

  // Calculate all face areas
  Matrix<float, Dynamic, 1> A;
  igl::doublearea(V, F, A);
  A *= 0.5f;

  // Calculate the valence of every vertex to allocate sparse matrices
  Matrix<int, Dynamic, 1> VV, VA, VAK, VCV;
  vertex_vertex_adjacency(F, VAK, VA, VV, VCV);

  // Calculate the intrinsic dirac operator matrix
  SparseMatrix<float> D;
  intrinsic_dirac(V, F, VV, A, P, D);

  // Calculate the scaling and rotation for our spin transformation
  Matrix<float, Dynamic, 4> X;
  similarity_xform(D, X);

  // Calculate our transformed edges
  Matrix<float, Dynamic, 4> E;
  divergent_edges(V, F, X, L, E);

  // Solve the final vertex positions
  Matrix<float, Dynamic, Dynamic> NV;
  spin_positions(QL, E, NV);
  NV.conservativeResize(NoChange, 3);
  V = NV;
}



int main(int argc, char* argv[])
{
  // Command line arguments
  const std::string in_name = argv[1];
  const std::string out_name = argv[2];
  const int max_iter = std::stoi(argv[3]);
  const float tao = std::stof(argv[4]);
  const std::string is_decimate_str = argv[5]
  bool is_decimate;
  istringstream(is_decimate_str) >> is_decimate;
  // flo::host::Surface surf;
  // igl::read_triangle_mesh(in_name, surf.vertices, surf.faces);

  Matrix<float, Eigen::Dynamic, 3> V;
  Matrix<int, Eigen::Dynamic, 3> F;
  igl::read_triangle_mesh(in_name,V,F);

  ForwardEuler<float> integrator(tao);

  for (int iter = 0; iter < max_iter; ++iter)
  {
    std::cout << "Iteration: " << iter << '\n';
    // flo::host::willmore_flow(surf.vertices, surf.faces, integrator);

    // Verbose - Willmore Flow

    std::cout << "Calculate smooth vertex normals" << std::endl;

    // Calculate smooth vertex normals
    // MatrixXd N;
    Matrix<float, Eigen::Dynamic, 3> N;
    igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);

    std::cout << "Calculate the cotangent laplacian for our mesh" << std::endl;
    // Calculate the cotangent laplacian for our mesh
    SparseMatrix<float> L;
    igl::cotmatrix(V, F, L);
    L = (-L.eval());

    std::cout << "Calculate the vertex masses for our mesh" << std::endl;
    // Calculate the vertex masses for our mesh
    Matrix<float, Dynamic, 1> M;
    M.resize(V.rows());
    M.setConstant(0.0f);
    // Calculate all face areas
    Matrix<float, Dynamic, 1> A;
    igl::doublearea(V, F, A);

    // For every face
    for (int i = 0; i < F.rows(); ++i)
    {
      const auto& f = F.row(i);
      constexpr auto sixth = 1.f / 6.f;
      const auto thirdArea = A(i) * sixth;

      M(f(0)) += thirdArea;
      M(f(1)) += thirdArea;
      M(f(2)) += thirdArea;
    }

    std::cout << "Build our constraints {1, N.x, N.y, N.z}" << std::endl;
    // Build our constraints {1, N.x, N.y, N.z}
    Matrix<float, Dynamic, 4> constraints(N.rows(), 4);
    constraints.col(0) = Matrix<float, Dynamic, 1>::Ones(N.rows());
    constraints.col(1) = N.col(0);
    constraints.col(2) = N.col(1);
    constraints.col(3) = N.col(2);

    // Declare an immersed inner-product using the mass matrix
    const auto inner_product = [&M](const Matrix<float, Dynamic, 1>& x,
                         const Matrix<float, Dynamic, 1>& y) -> float {
      auto single_mat = (x.transpose() * M.asDiagonal() * y).eval();
      return single_mat(0, 0);
    };

    // Build a constraint basis using the Gram–Schmidt process
    Matrix<float, Dynamic, Dynamic> U;
    // flo::host::orthonormalize(constraints, inner_product, U);
    auto normalize = [&](const Matrix<float, Dynamic, 1>& x) {
      return x.array() / std::sqrt(inner_product(x, x));
    };
    // Dimensionality of vectors
    const auto nvectors = V.cols();
    const auto dim = V.rows();

    // Allocate space for our final basis matrix
    U.resize(dim, nvectors);

    // The first u0 is v0 normalized
    U.col(0) = normalize(V.col(0));
    // Gramm Schmit process
    for (int i = 1; i < nvectors; ++i)
    {
      U.col(i) = V.col(i) - inner_product(V.col(i), U.col(0)) * U.col(0);
      for (int k = 1; k < i; ++k)
      {
        U.col(i) -= inner_product(U.col(i), U.col(k)) * U.col(k);
      }
      U.col(i) = normalize(U.col(i).eval());
    }

    // Calculate the signed mean curvature based on our vertex normals
    Matrix<float, Dynamic, 1> H;
    // signed_mean_curvature(V, L, M, N, H);
    Matrix<float, Dynamic, 3> HN;
    // mean_curvature_normal(V, L, M, HN);
    Matrix<float, Dynamic, 1> Minv = 1.f / (12.f * M.array());
    HN = (-Minv).asDiagonal() * (2.0f * L * V);
    H.resize(HN.rows());

    for (int i = 0; i < HN.rows(); ++i)
    {
      // if the angle between the unit and curvature normals is obtuse,
      // we need to flow in the opposite direction, and hence invert our sign
      auto NdotH = -N.row(i).dot(HN.row(i));
      H(i) = std::copysign(HN.row(i).norm(), std::move(NdotH));
    }

    // Apply our flow direction to the mean curvature half density
    H *= -1.f;

    Matrix<float, Dynamic, 1> HP = H;
    //// Build our constraints {1, N.x, N.y, N.z} curvature
    // project_basis(HP, U, ip);
    for (int i = 0; i < U.cols(); ++i)
    {
      HP -= inner_product(HP, U.col(i)) * U.col(i);
    }

    // take a time step
    integrator(H, HP);

    // // spin transform using our change in mean curvature half-density
    spin_xform(V, F, H, L);

    if (iter % 3 == 0)
    {
      std::stringstream ss(std::to_string(iter));
      std::string newString = out_name + ss.str() + ".obj";
      igl::writeOBJ(newString, V, F);
    }

    if (is_decimate) 
    {
      const auto num_face = F.rows();
      igl::decimate(V,F,num_face / 2,V,F);
    }

  }

  // std::stringstream ss(std::to_string(max_iter));
  // std::string newString = out_name + ss.str() + ".obj";
  // igl::writeOBJ(newString,V,F);

  return 0;
}
