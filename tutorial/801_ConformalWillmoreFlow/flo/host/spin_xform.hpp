#ifndef FLO_HOST_INCLUDED_SPIN_XFORM
#define FLO_HOST_INCLUDED_SPIN_XFORM

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "spin_xform.hpp"
#include "flo_matrix_operation.hpp"
#include "intrinsic_dirac.hpp"
#include "vertex_vertex_adjacency.hpp"
#include "similarity_xform.hpp"
#include "divergent_edges.hpp"
#include "spin_positions.hpp"
#include <igl/doublearea.h>

FLO_HOST_NAMESPACE_BEGIN

/// @breif Solves the spin transformation given a change in mean curvature
//  @param V #Vx3 A column major matrix of vertex positions
//  @param F #Fx3 A column major matrix of face vertex indices
//  @param P #V An array containing the per vertex change in mean curvature
//  @param L #Vx#V Sparse cotangent laplacian matrix
template <typename DerivedV, typename DerivedF, typename DerivedP>
FLO_API void spin_xform(Eigen::MatrixBase<DerivedV>& V,
                        const Eigen::MatrixBase<DerivedF>& F,
                        const Eigen::MatrixBase<DerivedP>& P,
                        const Eigen::SparseMatrix<real>& L);

#include "spin_xform.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SPIN_XFORM
