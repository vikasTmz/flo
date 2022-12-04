#ifndef FLO_INCLUDED_WILLMORE_FLOW
#define FLO_INCLUDED_WILLMORE_FLOW

#include "flo_internal.hpp"
#include <Eigen/Dense>
#include "willmore_flow.hpp"
#include "mean_curvature.hpp"
#include "flo_matrix_operation.hpp"
#include "spin_xform.hpp"
#include "vertex_mass.hpp"
#include "orthonormalize.hpp"
#include "project_basis.hpp"
#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOFF.h>

FLO_HOST_NAMESPACE_BEGIN

/// @brief Computes a single iteration of conformal willmore flow
//  @param V #Vx3 A column major matrix of vertex positions
//  @param F #Fx3 A column major matrix of face vertex indices
//  @param integrator A function used to integrate the change in mean curvature
//  for a time step.
template <typename DerivedV, typename DerivedF, typename BinaryOp>
FLO_API void willmore_flow(Eigen::MatrixBase<DerivedV>& V,
                           const Eigen::MatrixBase<DerivedF>& F,
                           BinaryOp integrator);

#include "willmore_flow.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_INCLUDED_WILLMORE_FLOW

