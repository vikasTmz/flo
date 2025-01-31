#ifndef FLO_HOST_INCLUDED_ORTHONORMALIZE
#define FLO_HOST_INCLUDED_ORTHONORMALIZE

#include "flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

/// @breif Performs Gramm-Schmidtt orthonormalization to produce a basis
//  @param V matrix of vectors to orthonormalize
//  @param inner_product A functor defining the operation to use as an inner product
//  @param U The resulting basis matrix
template <typename DerivedV, typename BinaryOp, typename DerivedU>
FLO_API void orthonormalize(const Eigen::MatrixBase<DerivedV>& V,
                            BinaryOp inner_product,
                            Eigen::PlainObjectBase<DerivedU>& U);

#include "orthonormalize.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_ORTHONORMALIZE

